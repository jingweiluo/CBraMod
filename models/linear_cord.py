import copy
import math
from typing import Optional, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


# ============================================================
# 1D Positional Encoding
# ============================================================

class SinCos1DPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequence (B, L, D)
    """
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (D/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, L, D)
        """
        L = x.size(1)
        x = x + self.pe[:, :L]
        return self.dropout(x)


class Learnable1DPositionalEncoding(nn.Module):
    """
    Learnable positional encoding for sequence (B, L, D)
    """
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, L, D)
        """
        L = x.size(1)
        x = x + self.pe[:, :L]
        return self.dropout(x)


# ============================================================
# Standard (Linear) Transformer over flattened tokens
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
        use_positional_encoding: bool = True,
        pos_type: str = "sincos",   # "sincos" or "learnable"
        d_model: Optional[int] = 200,
        max_len: int = 10000,
        pos_dropout: float = 0.0,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            assert d_model is not None, "When use_positional_encoding=True, d_model must be provided."
            if pos_type == "sincos":
                self.pos_encoder = SinCos1DPositionalEncoding(
                    d_model=d_model, max_len=max_len, dropout=pos_dropout
                )
            elif pos_type == "learnable":
                self.pos_encoder = Learnable1DPositionalEncoding(
                    d_model=d_model, max_len=max_len, dropout=pos_dropout
                )
            else:
                raise ValueError(f"Unsupported pos_type: {pos_type}")
        else:
            self.pos_encoder = None

    def forward(
        self,
        src: Tensor,                         # (B, C, P, W)
        mask: Optional[Tensor] = None,       # (L, L) 或 broadcastable；L=C*P
        src_key_padding_mask: Optional[Tensor] = None,  # (B, L) True=pad
        is_causal: Optional[bool] = None,
        coords: Optional[Tensor] = None,     # 保留接口，但线性 Transformer 不使用
    ) -> Tensor:
        output = src

        # 先加 1D 位置编码
        if self.pos_encoder is not None:
            B, C, P, W = output.shape
            L = C * P
            x = output.reshape(B, L, W)      # (B, L, W)
            x = self.pos_encoder(x)          # (B, L, W)
            output = x.reshape(B, C, P, W)   # 恢复回去

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=bool(is_causal) if is_causal is not None else False,
                coords=coords,
            )

        if self.norm is not None:
            # 这里的 norm 期望作用在最后一维 W 上
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['norm_first']

    def __init__(
        self,
        d_model: int,          # 这里对应 W
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
        assert batch_first is True, "本实现固定 batch_first=True（更贴合你的输入）"

        # 标准 MultiheadAttention：对 flatten 后的序列 L=C*P 做 self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            **factory_kwargs
        )

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

    def forward(
        self,
        src: Tensor,                              # (B, C, P, W)
        src_mask: Optional[Tensor] = None,        # (L,L) additive/boolean，L=C*P
        src_key_padding_mask: Optional[Tensor] = None,  # (B,L) True=pad
        is_causal: bool = False,
        coords: Optional[Tensor] = None,          # 保留接口，不使用
    ) -> Tensor:
        # ---- flatten tokens: (B, C, P, W) -> (B, L, W) ----
        B, C, P, W = src.shape
        L = C * P
        x = src.reshape(B, L, W)

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        # ---- restore shape: (B, L, W) -> (B, C, P, W) ----
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
        d_model=d_model, nhead=4, dim_feedforward=1024,
        batch_first=True, norm_first=True, activation=F.gelu
    )

    encoder = TransformerEncoder(
        encoder_layer,
        num_layers=2,
        enable_nested_tensor=False,
        use_positional_encoding=True,
        pos_type="sincos",   # 可改成 "learnable"
        d_model=d_model,
        max_len=2000,
        pos_dropout=0.0,
    ).cuda()

    a = torch.randn((4, 19, 30, 256)).cuda()  # (B, C, P, W)

    coords = torch.randn(19, 3).cuda() * 0.1

    b = encoder(a, coords=coords)
    print(a.shape, b.shape)