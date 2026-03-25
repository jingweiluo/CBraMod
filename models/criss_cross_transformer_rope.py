import copy
from typing import Optional, Union, Callable, List, Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


# ============================================================
# Attention Residual
# h = sum_i softmax(q^T RMSNorm(v_i)) * v_i
# ============================================================

class AttentionResidual(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        use_output_proj: bool = False,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.query = nn.Parameter(torch.zeros(d_model, **factory_kwargs))
        nn.init.zeros_(self.query)

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        self.use_output_proj = use_output_proj
        self.out_proj = nn.Linear(d_model, d_model, **factory_kwargs) if use_output_proj else nn.Identity()

    def forward(self, sources: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        sources: list[(B, C, P, D)]
        returns:
            out:     (B, C, P, D)
            weights: (B, C, P, Nsrc)
        """
        assert len(sources) > 0, "AttentionResidual requires at least one source"

        V = torch.stack(sources, dim=0)         # (Nsrc, B, C, P, D)
        K = self.norm(V)

        logits = torch.einsum("d,nbcpd->nbcp", self.query, K)   # (Nsrc, B, C, P)
        weights = torch.softmax(logits, dim=0)                  # softmax over Nsrc

        out = torch.einsum("nbcp,nbcpd->bcpd", weights, V)      # (B, C, P, D)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out, weights.permute(1, 2, 3, 0).contiguous()


# ============================================================
# RoPE helpers
# ============================================================

def _build_freqs(half_dim: int, base: float, device, dtype):
    """
    Return RoPE frequencies of shape (half_dim,)
    """
    idx = torch.arange(half_dim, device=device, dtype=dtype)
    return 1.0 / (base ** (idx / max(half_dim, 1)))


def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """
    q, k:   (B, H, L, Hd)
    cos,sin:(1, 1, L, Hd/2) or broadcastable

    split-half style RoPE:
      q = [q1, q2]
      q_rot = [q1*cos - q2*sin, q1*sin + q2*cos]
    """
    head_dim = q.size(-1)
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    half = head_dim // 2

    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]

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
    Standard 1D index-RoPE.
    Returns cos,sin with shape (1,1,L,half_dim)
    """
    assert head_dim % 2 == 0
    half = head_dim // 2

    freqs = _build_freqs(half, base=base, device=device, dtype=dtype)   # (half,)
    pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(-1)      # (L,1)
    theta = pos * freqs.unsqueeze(0)                                     # (L,half)

    cos = torch.cos(theta).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(theta).unsqueeze(0).unsqueeze(0)
    return cos, sin


def _rope_coord_cos_sin_axis(
    coords: Tensor,         # (L,3)
    head_dim: int,
    base: float,
    theta_scale: float,
    axis_ids: Tensor,       # (half_dim,), values in {0,1,2}
) -> tuple[Tensor, Tensor]:
    """
    3D coordinate RoPE (axis-wise version)

    For each rotary pair i:
        theta_i = theta_scale * coord[axis_ids[i]] * freq_i

    Returns cos,sin with shape (1,1,L,half_dim)
    """
    assert coords.dim() == 2 and coords.size(-1) == 3, "coords must be (L, 3)"
    assert head_dim % 2 == 0

    device, dtype = coords.device, coords.dtype
    half = head_dim // 2

    # normalize coords
    c = coords - coords.mean(dim=0, keepdim=True)
    r = torch.linalg.norm(c, dim=-1)
    max_r = torch.clamp(r.max(), min=1e-6)
    c = c / max_r

    freqs = _build_freqs(half, base=base, device=device, dtype=dtype)   # (half,)
    axis_ids = axis_ids.to(device=device)

    c_selected = c[:, axis_ids]                                         # (L, half)
    theta = theta_scale * (c_selected * freqs.unsqueeze(0))             # (L, half)

    cos = torch.cos(theta).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(theta).unsqueeze(0).unsqueeze(0)
    return cos, sin


# ============================================================
# Rotary Multihead Attention
# ============================================================

class RotaryMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        rope_base: float = 10000.0,
        is_causal: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        assert batch_first is True, "This implementation only supports batch_first=True"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 2 == 0, "For RoPE, head_dim must be even"

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.dropout = dropout
        self.rope_base = rope_base
        self.is_causal = is_causal

    def forward(
        self,
        x: Tensor,                           # (B,L,E)
        attn_mask: Optional[Tensor] = None,  # additive mask preferred
        key_padding_mask: Optional[Tensor] = None,  # (B,L), True means pad
        rope_cos_sin: Optional[tuple[Tensor, Tensor]] = None,
        need_weights: bool = False,
    ):
        B, L, E = x.shape

        qkv = self.qkv(x)                    # (B,L,3E)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,L,Hd)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            cos = cos.to(device=x.device, dtype=x.dtype)
            sin = sin.to(device=x.device, dtype=x.dtype)
            q, k = _apply_rope(q, k, cos, sin)

        attn_bias = None
        if attn_mask is not None:
            attn_bias = attn_mask

        if key_padding_mask is not None:
            kpm = key_padding_mask.view(B, 1, 1, L).to(torch.bool)
            add_mask = torch.zeros((B, 1, 1, L), device=x.device, dtype=x.dtype)
            add_mask = add_mask.masked_fill(kpm, float("-inf"))
            if attn_bias is None:
                attn_bias = add_mask
            else:
                attn_bias = attn_bias + add_mask

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal
        )  # (B,H,L,Hd)

        out = out.transpose(1, 2).contiguous().view(B, L, E)
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
            return_residual_weights: bool = False,
            coords: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, List[Dict[str, Tensor]]]]:

        output = src
        history: List[Tensor] = [output]
        aux_outputs: List[Dict[str, Tensor]] = []

        for mod in self.layers:
            if return_residual_weights:
                output, history, aux = mod(
                    output,
                    history=history,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    is_causal=bool(is_causal) if is_causal is not None else False,
                    return_attn_res_weights=True,
                    coords=coords
                )
                aux_outputs.append(aux)
            else:
                output, history = mod(
                    output,
                    history=history,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    is_causal=bool(is_causal) if is_causal is not None else False,
                    return_attn_res_weights=False,
                    coords=coords
                )

        if self.norm is not None:
            output = self.norm(output)

        if return_residual_weights:
            return output, aux_outputs
        return output


# ============================================================
# Encoder Layer with Attention Residual + RoPE
# ============================================================

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
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,

        # attention residual args
        res_dropout: float = 0.1,
        res_use_output_proj: bool = True,

        # rope args
        rope_base: float = 10000.0,
        coord_theta_scale: float = 2.0 * 3.141592653589793,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        assert d_model % 2 == 0, "d_model must be even because it is split into spatial/time halves"
        assert nhead % 2 == 0, "nhead must be even because it is split into spatial/time halves"
        assert batch_first is True, "This implementation assumes batch_first=True"
        assert norm_first is True, "Attention Residual version is implemented in PreNorm style (norm_first=True)"

        self.self_attn_s = RotaryMultiheadAttention(
            d_model // 2,
            nhead // 2,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            rope_base=rope_base,
            is_causal=False,
            **factory_kwargs
        )
        self.self_attn_t = RotaryMultiheadAttention(
            d_model // 2,
            nhead // 2,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            rope_base=rope_base,
            is_causal=False,
            **factory_kwargs
        )

        self.rope_base = rope_base
        self.coord_theta_scale = coord_theta_scale

        # head_dim for branch-attention head
        # rotary half-dim = head_dim // 2
        half_dim_s = self.self_attn_s.head_dim // 2
        axis_ids = torch.arange(half_dim_s) % 3  # x, y, z, x, y, z, ...
        self.register_buffer("axis_ids", axis_ids, persistent=True)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.pre_attn_res = AttentionResidual(
            d_model=d_model,
            dropout=res_dropout,
            use_output_proj=res_use_output_proj,
            device=device,
            dtype=dtype,
        )
        self.pre_ffn_res = AttentionResidual(
            d_model=d_model,
            dropout=res_dropout,
            use_output_proj=res_use_output_proj,
            device=device,
            dtype=dtype,
        )

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(
            self,
            src: Tensor,
            history: List[Tensor],
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
            return_attn_res_weights: bool = False,
            coords: Optional[Tensor] = None
    ) -> Union[
        Tuple[Tensor, List[Tensor]],
        Tuple[Tensor, List[Tensor], Dict[str, Tensor]]
    ]:
        # 1) pre-attention residual
        h_attn_in, w_attn = self.pre_attn_res(history)
        attn_out = self._sa_block(
            self.norm1(h_attn_in),
            src_mask,
            src_key_padding_mask,
            is_causal=is_causal,
            coords=coords
        )

        history = history + [attn_out]

        # 2) pre-ffn residual
        h_ffn_in, w_ffn = self.pre_ffn_res(history)
        ff_out = self._ff_block(self.norm2(h_ffn_in))

        history = history + [ff_out]
        x = ff_out

        if return_attn_res_weights:
            aux = {
                "pre_attn_weights": w_attn,
                "pre_ffn_weights": w_ffn,
            }
            return x, history, aux

        return x, history

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        coords: Optional[Tensor] = None
    ) -> Tensor:
        """
        x: (B, C, P, D)
        Split last dim into:
          - spatial branch:  first  D/2
          - temporal branch: second D/2
        """
        bz, ch_num, patch_num, patch_size = x.shape
        assert patch_size % 2 == 0, "patch_size / d_model must be even"

        half_embed = patch_size // 2
        xs = x[:, :, :, :half_embed]   # (B, C, P, D/2)
        xt = x[:, :, :, half_embed:]   # (B, C, P, D/2)

        # ------------------------------------------------
        # spatial attention: sequence over channels C
        # input: (B*P, C, D/2)
        # ------------------------------------------------
        xs = xs.transpose(1, 2).contiguous().view(bz * patch_num, ch_num, half_embed)

        if coords is not None:
            coords_t = coords.to(device=x.device, dtype=x.dtype)  # (C,3)
            cos_s, sin_s = _rope_coord_cos_sin_axis(
                coords=coords_t,
                head_dim=self.self_attn_s.head_dim,
                base=self.rope_base,
                theta_scale=self.coord_theta_scale,
                axis_ids=self.axis_ids,
            )
        else:
            cos_s, sin_s = _rope_index_cos_sin(
                L=ch_num,
                head_dim=self.self_attn_s.head_dim,
                base=self.rope_base,
                device=x.device,
                dtype=x.dtype
            )

        xs = self.self_attn_s(
            xs,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            rope_cos_sin=(cos_s, sin_s),
            need_weights=False
        )[0]
        xs = xs.contiguous().view(bz, patch_num, ch_num, half_embed).transpose(1, 2)

        # ------------------------------------------------
        # temporal attention: sequence over patches P
        # input: (B*C, P, D/2)
        # ------------------------------------------------
        xt = xt.contiguous().view(bz * ch_num, patch_num, half_embed)

        cos_t, sin_t = _rope_index_cos_sin(
            L=patch_num,
            head_dim=self.self_attn_t.head_dim,
            base=self.rope_base,
            device=x.device,
            dtype=x.dtype
        )

        xt = self.self_attn_t(
            xt,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            rope_cos_sin=(cos_t, sin_t),
            need_weights=False
        )[0]
        xt = xt.contiguous().view(bz, ch_num, patch_num, half_embed)

        x = torch.cat((xs, xt), dim=3)
        return self.dropout1(x)

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
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            return src_size[0]
        else:
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


if __name__ == '__main__':
    encoder_layer = TransformerEncoderLayer(
        d_model=256,
        nhead=4,
        dim_feedforward=1024,
        batch_first=True,
        norm_first=True,
        activation=F.gelu,
        res_dropout=0.0,
        res_use_output_proj=False,
        rope_base=10000.0,
    )

    encoder = TransformerEncoder(
        encoder_layer,
        num_layers=2,
        enable_nested_tensor=False
    ).cuda()

    a = torch.randn((4, 19, 30, 256)).cuda()
    coords = torch.randn(19, 3).cuda() * 0.1   # optional channel coords

    b, aux = encoder(a, return_residual_weights=True, coords=coords)

    print("input shape :", a.shape)
    print("output shape:", b.shape)
    print("num layers  :", len(aux))
    print("layer0 pre_attn_weights:", aux[0]["pre_attn_weights"].shape)
    print("layer0 pre_ffn_weights :", aux[0]["pre_ffn_weights"].shape)