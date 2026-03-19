import copy
import math
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


# ============================================================
# Time Positional Encoding over patch dimension P
# Applied on (B, C, P, W)
# ============================================================

class SinCosTimePositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for time/patch dimension P.
    Input/output: (B, C, P, W)
    """
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (P, W)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (P, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, P, W)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, P, W)
        """
        P = x.size(2)
        x = x + self.pe[:, :, :P, :]
        return self.dropout(x)


class LearnableTimePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for time/patch dimension P.
    Input/output: (B, C, P, W)
    """
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, 1, max_len, d_model))  # (1,1,P,W)
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, P, W)
        """
        P = x.size(2)
        x = x + self.pe[:, :, :P, :]
        return self.dropout(x)


# ============================================================
# Channel Positional Encoding from coords
# coords: (C, coord_dim), usually coord_dim=3
# Output added on channel dimension
# ============================================================

class CoordChannelPositionalEncoding(nn.Module):
    """
    Use channel coordinates to generate channel positional embeddings.
    Input:  x      -> (B, C, P, W)
            coords -> (C, coord_dim)
    Output: (B, C, P, W)
    """
    def __init__(
        self,
        d_model: int,
        coord_dim: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        normalize_coords: bool = True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalize_coords = normalize_coords

        self.coord_proj = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:
        """
        x: (B, C, P, W)
        coords: (C, coord_dim)
        """
        assert coords is not None, "coords must be provided for CoordChannelPositionalEncoding"
        B, C, P, W = x.shape
        assert coords.shape[0] == C, f"coords.shape[0]={coords.shape[0]} must match channel dim C={C}"

        coord_feat = coords

        if self.normalize_coords:
            mean = coord_feat.mean(dim=0, keepdim=True)
            std = coord_feat.std(dim=0, keepdim=True).clamp_min(1e-6)
            coord_feat = (coord_feat - mean) / std

        ch_pe = self.coord_proj(coord_feat)          # (C, W)
        ch_pe = ch_pe.unsqueeze(0).unsqueeze(2)      # (1, C, 1, W)

        x = x + ch_pe
        return self.dropout(x)


# ============================================================
# Standard (Linear) Transformer over flattened spatiotemporal tokens
# Input:  (B, C, P, W)
# Flatten: (B, L=C*P, W)
# Output: (B, C, P, W)
# ============================================================

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        enable_nested_tensor=True,
        mask_check=True,

        use_time_positional_encoding: bool = True,
        time_pos_type: str = "sincos",   # "sincos" or "learnable"
        use_channel_coord_encoding: bool = True,

        d_model: Optional[int] = 200,
        max_len: int = 10000,
        pos_dropout: float = 0.0,

        coord_dim: int = 3,
        coord_hidden_dim: int = 128,
        coord_dropout: float = 0.0,
        normalize_coords: bool = True,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        assert d_model is not None, "d_model must be provided."

        # ----- time positional encoding -----
        self.use_time_positional_encoding = use_time_positional_encoding
        if use_time_positional_encoding:
            if time_pos_type == "sincos":
                self.time_pos_encoder = SinCosTimePositionalEncoding(
                    d_model=d_model, max_len=max_len, dropout=pos_dropout
                )
            elif time_pos_type == "learnable":
                self.time_pos_encoder = LearnableTimePositionalEncoding(
                    d_model=d_model, max_len=max_len, dropout=pos_dropout
                )
            else:
                raise ValueError(f"Unsupported time_pos_type: {time_pos_type}")
        else:
            self.time_pos_encoder = None

        # ----- channel positional encoding from coords -----
        self.use_channel_coord_encoding = use_channel_coord_encoding
        if use_channel_coord_encoding:
            self.channel_pos_encoder = CoordChannelPositionalEncoding(
                d_model=d_model,
                coord_dim=coord_dim,
                hidden_dim=coord_hidden_dim,
                dropout=coord_dropout,
                normalize_coords=normalize_coords,
            )
        else:
            self.channel_pos_encoder = None

    def forward(
        self,
        src: Tensor,                         # (B, C, P, W)
        mask: Optional[Tensor] = None,       # (L, L), L=C*P
        src_key_padding_mask: Optional[Tensor] = None,  # (B, L)
        is_causal: Optional[bool] = None,
        coords: Optional[Tensor] = None,     # (C, coord_dim)
    ) -> Tensor:
        output = src  # (B, C, P, W)

        # 1) add channel positional encoding from coords
        if self.channel_pos_encoder is not None:
            output = self.channel_pos_encoder(output, coords)

        # 2) add time positional encoding over patch dimension
        if self.time_pos_encoder is not None:
            output = self.time_pos_encoder(output)

        # 3) full spatiotemporal attention blocks
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=bool(is_causal) if is_causal is not None else False,
                coords=coords,
            )

        if self.norm is not None:
            output = self.norm(output)  # expected to act on last dim W

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
        assert batch_first is True, "This implementation assumes batch_first=True"

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            **factory_kwargs
        )

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

    def forward(
        self,
        src: Tensor,                              # (B, C, P, W)
        src_mask: Optional[Tensor] = None,        # (L,L), L=C*P
        src_key_padding_mask: Optional[Tensor] = None,  # (B,L)
        is_causal: bool = False,
        coords: Optional[Tensor] = None,          # reserved, not used in this layer
    ) -> Tensor:
        B, C, P, W = src.shape
        L = C * P

        # flatten spatiotemporal tokens
        x = src.reshape(B, L, W)   # (B, C*P, W)

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        out = x.reshape(B, C, P, W)
        return out

    def _sa_block(
        self,
        x: Tensor,                                 # (B, L, W)
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False
    ) -> Tensor:
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal
        )
        return self.dropout1(attn_out)

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
    d_model = 256

    encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=4,
        dim_feedforward=1024,
        batch_first=True,
        norm_first=True,
        activation=F.gelu
    )

    encoder = TransformerEncoder(
        encoder_layer,
        num_layers=2,
        enable_nested_tensor=False,

        use_time_positional_encoding=True,
        time_pos_type="sincos",      # or "learnable"
        use_channel_coord_encoding=True,

        d_model=d_model,
        max_len=2000,
        pos_dropout=0.0,

        coord_dim=3,
        coord_hidden_dim=128,
        coord_dropout=0.0,
        normalize_coords=True,
    ).cuda()

    a = torch.randn((4, 19, 30, 256)).cuda()   # (B, C, P, W)
    coords = torch.randn(19, 3).cuda() * 0.1   # (C, 3)

    b = encoder(a, coords=coords)
    print("input :", a.shape)
    print("output:", b.shape)