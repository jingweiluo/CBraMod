import copy
from typing import Optional, Any, Union, Callable

import torch
import torch.nn as nn
import warnings
from torch import Tensor
from torch.nn import functional as F
import math


# ============================================================
# RoPE helpers（新增，但不影响外部接口）
# ============================================================

def _build_rope_cache(seq_len: int, head_dim: int, device, dtype, base: float):
    half = head_dim // 2
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    freqs = torch.outer(pos, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    # q,k: (B, H, L, D)
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
    k = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
    return q, k


def _mha_forward_with_rope(
    mha: nn.MultiheadAttention,
    x: Tensor,
    rope_base: float,
    attn_mask: Optional[Tensor],
    key_padding_mask: Optional[Tensor],
):
    # x: (B, L, E)
    assert mha.batch_first, "This implementation assumes batch_first=True"

    B, L, E = x.shape
    H = mha.num_heads
    head_dim = E // H
    assert head_dim % 2 == 0, "RoPE requires even head_dim"

    # ---- QKV projection (reuse MHA weights)
    qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)

    q = q.view(B, L, H, head_dim).transpose(1, 2)
    k = k.view(B, L, H, head_dim).transpose(1, 2)
    v = v.view(B, L, H, head_dim).transpose(1, 2)

    # ---- RoPE
    cos, sin = _build_rope_cache(L, head_dim, x.device, x.dtype, rope_base)
    q, k = _apply_rope(q, k, cos, sin)

    # ---- Attention
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=mha.dropout if mha.training else 0.0,
        is_causal=False,
    )

    out = out.transpose(1, 2).contiguous().view(B, L, E)
    return mha.out_proj(out)


# ============================================================
# Transformer Encoder
# ============================================================

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, src: Tensor, mask=None, src_key_padding_mask=None, is_causal=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.self_attn_s = nn.MultiheadAttention(
            d_model, nhead // 2, dropout=dropout, batch_first=True
        )
        self.self_attn_t = nn.MultiheadAttention(
            d_model, nhead // 2, dropout=dropout, batch_first=True
        )

        # RoPE base（默认值，不影响外部）
        self.rope_base_s = 10000.0
        self.rope_base_t = 10000.0

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation if callable(activation) else _get_activation_fn(activation)
        self.gate_param = nn.Parameter(torch.zeros(200))

    def forward(self, src: Tensor, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x: Tensor, attn_mask, key_padding_mask):
        # x: (bz, ch_num, patch_num, patch_size)
        bz, ch_num, patch_num, patch_size = x.shape

        # -------- Spatial SA (along channel)
        xs = x.transpose(1, 2).contiguous().view(bz * patch_num, ch_num, patch_size)
        xs = _mha_forward_with_rope(
            self.self_attn_s,
            xs,
            self.rope_base_s,
            attn_mask,
            key_padding_mask,
        )
        xs = xs.view(bz, patch_num, ch_num, patch_size).transpose(1, 2)

        # -------- Temporal SA (along patch)
        xt = x.contiguous().view(bz * ch_num, patch_num, patch_size)
        xt = _mha_forward_with_rope(
            self.self_attn_t,
            xt,
            self.rope_base_t,
            attn_mask,
            key_padding_mask,
        )
        xt = xt.view(bz, ch_num, patch_num, patch_size)

        # # -------- Gated fusion
        # if not hasattr(self, "gate_param") or self.gate_param.numel() != patch_size:
        #     self.gate_param = nn.Parameter(torch.zeros(patch_size, device=x.device, dtype=x.dtype))
        g = torch.sigmoid(self.gate_param).view(1, 1, 1, patch_size)
        x = g * xs + (1.0 - g) * xt
        return self.dropout1(x)

    def _ff_block(self, x: Tensor):
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"Unsupported activation: {activation}")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ============================================================
# Debug / Sanity Check
# ============================================================
if __name__ == "__main__":
    encoder_layer = TransformerEncoderLayer(
        d_model=256,
        nhead=4,
        dim_feedforward=1024,
        batch_first=True,
        activation=F.gelu,
    )
    encoder = TransformerEncoder(encoder_layer, num_layers=2).cuda()

    x = torch.randn(4, 19, 30, 256).cuda()
    y = encoder(x)
    print(x.shape, y.shape)
