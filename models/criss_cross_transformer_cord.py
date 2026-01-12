import copy
from typing import Optional, Any, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


# ============================================================
# RoPE helpers
# ============================================================
def _build_freqs(half_dim: int, base: float, device, dtype):
    # (half_dim,)
    # classic RoPE frequency decay
    idx = torch.arange(half_dim, device=device, dtype=dtype)
    return 1.0 / (base ** (idx / half_dim))


def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """
    q,k: (B, nhead, L, head_dim)
    cos,sin: broadcastable to (B, nhead, L, half_dim)
    """
    head_dim = q.size(-1)
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    half = head_dim // 2

    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]

    # rotate in 2D pairs
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot


def _rope_index_cos_sin(
    L: int,
    head_dim: int,
    base: float,
    device,
    dtype
) -> tuple[Tensor, Tensor]:
    """
    Returns cos,sin with shape (1,1,L,half_dim) for broadcasting.
    """
    assert head_dim % 2 == 0
    half = head_dim // 2
    freqs = _build_freqs(half, base=base, device=device, dtype=dtype)  # (half,)
    pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(-1)    # (L,1)
    theta = pos * freqs.unsqueeze(0)                                    # (L,half)
    cos = torch.cos(theta).unsqueeze(0).unsqueeze(0)                    # (1,1,L,half)
    sin = torch.sin(theta).unsqueeze(0).unsqueeze(0)                    # (1,1,L,half)
    return cos, sin


def _rope_coord_cos_sin(
    coords: Tensor,         # (L,3)
    head_dim: int,
    base: float,
    theta_scale: float,     # e.g. 2*pi
    W_dir: Tensor,          # (3, half_dim) direction matrix (unit-ish columns)
) -> tuple[Tensor, Tensor]:
    """
    Coordinate-RoPE:
      theta(L,half) = theta_scale * (coords_norm @ (W_dir * freqs))
    Returns cos,sin with shape (1,1,L,half_dim).
    """
    assert coords.dim() == 2 and coords.size(-1) == 3, "coords must be (L,3)"
    assert head_dim % 2 == 0
    device, dtype = coords.device, coords.dtype
    L = coords.size(0)
    half = head_dim // 2

    # ---- normalize coords to be montage-invariant ----
    # center (optional but stable)
    c = coords - coords.mean(dim=0, keepdim=True)
    # scale by max radius
    r = torch.linalg.norm(c, dim=-1)
    max_r = torch.clamp(r.max(), min=1e-6)
    c = c / max_r  # now roughly within [-1,1] scale

    freqs = _build_freqs(half, base=base, device=device, dtype=dtype)  # (half,)
    # W: (3,half)
    W = W_dir.to(device=device, dtype=dtype) * freqs.unsqueeze(0)
    theta = theta_scale * (c @ W)  # (L,half)

    cos = torch.cos(theta).unsqueeze(0).unsqueeze(0)  # (1,1,L,half)
    sin = torch.sin(theta).unsqueeze(0).unsqueeze(0)  # (1,1,L,half)
    return cos, sin


# ============================================================
# Rotary Multihead Attention (batch_first=True)
# ============================================================
class RotaryMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 2 == 0, "For RoPE, head_dim should be even"

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = dropout
        self.rope_base = rope_base

    def forward(
        self,
        x: Tensor,                           # (B,L,E)
        attn_mask: Optional[Tensor] = None,  # (L,L) or broadcastable, float(-inf)/0
        key_padding_mask: Optional[Tensor] = None,  # (B,L) where True means pad
        rope_cos_sin: Optional[tuple[Tensor, Tensor]] = None,  # (cos,sin) each (1,1,L,half)
        need_weights: bool = False,
    ):
        B, L, E = x.shape
        qkv = self.qkv(x)  # (B,L,3E)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, nhead, L, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE on q,k
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            # broadcast to (B,nhead,L,half)
            cos = cos.to(device=x.device, dtype=x.dtype)
            sin = sin.to(device=x.device, dtype=x.dtype)
            q, k = _apply_rope(q, k, cos, sin)

        # Build attention bias/masks for scaled_dot_product_attention
        # key_padding_mask: True for pad -> mask out
        attn_bias = None
        if attn_mask is not None:
            # attn_mask could be (L,L) additive mask (0/-inf) or boolean
            if attn_mask.dtype == torch.bool:
                attn_bias = attn_mask
            else:
                attn_bias = attn_mask  # additive
        if key_padding_mask is not None:
            # convert to boolean mask of shape (B,1,1,L) for SDPA
            kpm = key_padding_mask.view(B, 1, 1, L).to(torch.bool)
            if attn_bias is None:
                attn_bias = kpm
            else:
                # combine: if additive mask, we'd need additive; easiest: convert to bool mask
                if attn_bias.dtype != torch.bool:
                    attn_bias = attn_bias == float("-inf")
                attn_bias = attn_bias | kpm

        # SDPA expects:
        # q,k,v: (B,nhead,L,head_dim)
        # attn_mask: bool mask (True=masked) OR additive mask
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )  # (B,nhead,L,head_dim)

        out = out.transpose(1, 2).contiguous().view(B, L, E)  # (B,L,E)
        out = self.out_proj(out)
        if need_weights:
            return out, None
        return out, None


# ============================================================
# Encoder
# ============================================================
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
        coords: Optional[Tensor] = None,   # <--- 可选新增：不传不影响旧用法
    ) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, coords=coords)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['norm_first']

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
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        # 两路注意力：每路用 nhead//2 个 heads（你原来就是这么做的）
        assert (nhead // 2) >= 1, "nhead must be >=2 for split heads"
        self.self_attn_s = RotaryMultiheadAttention(
            embed_dim=d_model, num_heads=nhead // 2, dropout=dropout, bias=bias
        )
        self.self_attn_t = RotaryMultiheadAttention(
            embed_dim=d_model, num_heads=nhead // 2, dropout=dropout, bias=bias
        )

        # ---- RoPE configs (no required args) ----
        self.rope_base = 10000.0
        self.coord_theta_scale = 2.0 * 3.141592653589793  # 2*pi

        # 固定的方向矩阵 W_dir: (3, half_dim) —— 作为 buffer，跨数据集稳定
        # half_dim = head_dim/2, head_dim = d_model / (nhead//2)
        head_dim_s = self.self_attn_s.head_dim
        half = head_dim_s // 2
        # 用确定性初始化：基于固定随机种子生成，再归一化列向量
        g = torch.Generator(device='cpu')
        g.manual_seed(1234)
        W = torch.randn(3, half, generator=g)
        W = W / (W.norm(dim=0, keepdim=True) + 1e-6)
        self.register_buffer("W_dir", W, persistent=True)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

        # 门控融合参数（不依赖 patch_size，可广播后按通道投影生成 gate）
        # 用一个线性层根据特征自适应产生 gate，更稳于“直接 gate_param=patch_size”
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        coords: Optional[Tensor] = None,   # <--- 可选，不传不影响旧代码
    ) -> Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal, coords=coords)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        coords: Optional[Tensor] = None
    ) -> Tensor:
        # x: (bz, ch_num, patch_num, d_model)
        bz, ch_num, patch_num, d_model = x.shape

        # ---------------------------
        # Spatial SA over channels:
        # sequence length = ch_num
        # ---------------------------
        xs_in = x.transpose(1, 2).contiguous().view(bz * patch_num, ch_num, d_model)  # (B*P, C, E)

        # spatial rope
        if coords is not None:
            # coords should be (ch_num,3), aligned with channel order of input x
            coords_t = coords.to(device=x.device, dtype=x.dtype)
            cos_s, sin_s = _rope_coord_cos_sin(
                coords=coords_t,
                head_dim=self.self_attn_s.head_dim,
                base=self.rope_base,
                theta_scale=self.coord_theta_scale,
                W_dir=self.W_dir,
            )
            rope_s = (cos_s, sin_s)
        else:
            # fallback to index-RoPE over channel indices
            cos_s, sin_s = _rope_index_cos_sin(
                L=ch_num, head_dim=self.self_attn_s.head_dim,
                base=self.rope_base, device=x.device, dtype=x.dtype
            )
            rope_s = (cos_s, sin_s)

        xs, _ = self.self_attn_s(
            xs_in,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            rope_cos_sin=rope_s,
            need_weights=False
        )
        xs = xs.contiguous().view(bz, patch_num, ch_num, d_model).transpose(1, 2)  # (bz, ch_num, patch_num, d_model)

        # ---------------------------
        # Temporal SA over patches:
        # sequence length = patch_num
        # ---------------------------
        xt_in = x.contiguous().view(bz * ch_num, patch_num, d_model)  # (B*C, P, E)

        cos_t, sin_t = _rope_index_cos_sin(
            L=patch_num, head_dim=self.self_attn_t.head_dim,
            base=self.rope_base, device=x.device, dtype=x.dtype
        )
        rope_t = (cos_t, sin_t)

        xt, _ = self.self_attn_t(
            xt_in,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            rope_cos_sin=rope_t,
            need_weights=False
        )
        xt = xt.contiguous().view(bz, ch_num, patch_num, d_model)

        # ---------------------------
        # Gated fusion (shape preserved)
        #   gate = sigmoid(W * x)  -> data-dependent
        #   out  = gate * xs + (1-gate) * xt
        # ---------------------------
        gate = torch.sigmoid(self.gate_proj(x))  # (bz, ch_num, patch_num, d_model)
        out = gate * xs + (1.0 - gate) * xt
        return self.dropout1(out)

    # def _sa_block(
    #     self,
    #     x: Tensor,
    #     attn_mask: Optional[Tensor],
    #     key_padding_mask: Optional[Tensor],
    #     is_causal: bool = False,
    #     coords: Optional[Tensor] = None
    # ) -> Tensor:
    #     # x: (bz, ch_num, patch_num, d_model)
    #     bz, ch_num, patch_num, d_model = x.shape

    #     # ---- split on last dim (embedding/features) ----
    #     e1 = d_model // 2
    #     e2 = d_model - e1

    #     x1 = x[:, :, :, :e1]       # (B, C, P, E1)
    #     x2 = x[:, :, :, e1:]       # (B, C, P, E2)

    #     # ---- random swap per forward (train only) ----
    #     # True : x1->spatial, x2->temporal
    #     # False: x1->temporal, x2->spatial
    #     if self.training:
    #         flip = (torch.rand((), device=x.device) < 0.5)
    #     else:
    #         flip = False

    #     # ---- helper: spatial SA over channels (seq len = ch_num) ----
    #     def spatial_sa(x_part: Tensor) -> Tensor:
    #         # x_part: (B, C, P, Epart)
    #         B, C, P, E = x_part.shape
    #         xs_in = x_part.transpose(1, 2).contiguous().view(B * P, C, E)  # (B*P, C, E)

    #         # spatial RoPE
    #         if coords is not None:
    #             coords_t = coords.to(device=x.device, dtype=x.dtype)  # (C,3)
    #             cos_s, sin_s = _rope_coord_cos_sin(
    #                 coords=coords_t,
    #                 head_dim=self.self_attn_s.head_dim,
    #                 base=self.rope_base,
    #                 theta_scale=self.coord_theta_scale,
    #                 W_dir=self.W_dir,
    #             )
    #             rope_s = (cos_s, sin_s)
    #         else:
    #             cos_s, sin_s = _rope_index_cos_sin(
    #                 L=C,
    #                 head_dim=self.self_attn_s.head_dim,
    #                 base=self.rope_base,
    #                 device=x.device,
    #                 dtype=x.dtype
    #             )
    #             rope_s = (cos_s, sin_s)

    #         xs, _ = self.self_attn_s(
    #             xs_in,
    #             attn_mask=attn_mask,
    #             key_padding_mask=key_padding_mask,
    #             rope_cos_sin=rope_s,
    #             need_weights=False
    #         )
    #         xs = xs.contiguous().view(B, P, C, E).transpose(1, 2)  # (B, C, P, E)
    #         return xs

    #     # ---- helper: temporal SA over patches (seq len = patch_num) ----
    #     def temporal_sa(x_part: Tensor) -> Tensor:
    #         # x_part: (B, C, P, Epart)
    #         B, C, P, E = x_part.shape
    #         xt_in = x_part.contiguous().view(B * C, P, E)  # (B*C, P, E)

    #         cos_t, sin_t = _rope_index_cos_sin(
    #             L=P,
    #             head_dim=self.self_attn_t.head_dim,
    #             base=self.rope_base,
    #             device=x.device,
    #             dtype=x.dtype
    #         )
    #         rope_t = (cos_t, sin_t)

    #         xt, _ = self.self_attn_t(
    #             xt_in,
    #             attn_mask=attn_mask,
    #             key_padding_mask=key_padding_mask,
    #             rope_cos_sin=rope_t,
    #             need_weights=False
    #         )
    #         xt = xt.contiguous().view(B, C, P, E)  # (B, C, P, E)
    #         return xt

    #     # ---- route halves ----
    #     if flip:
    #         y1 = spatial_sa(x1)
    #         y2 = temporal_sa(x2)
    #     else:
    #         y1 = temporal_sa(x1)
    #         y2 = spatial_sa(x2)

    #     # ---- concat on last dim to restore (B, C, P, E) ----
    #     out = torch.cat([y1, y2], dim=3)
    #     return self.dropout1(out)


    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


if __name__ == '__main__':
    encoder_layer = TransformerEncoderLayer(
        d_model=256, nhead=4, dim_feedforward=1024, batch_first=True, norm_first=True,
        activation=F.gelu
    )
    encoder = TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False).cuda()

    a = torch.randn((4, 19, 30, 256)).cuda()

    # 示例：coords 可选
    coords = torch.randn(19, 3).cuda() * 0.1  # 假设已按通道顺序对齐
    b = encoder(a, coords=coords)

    print(a.shape, b.shape)
