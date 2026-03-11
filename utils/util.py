import os
import random
import signal

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import math
import mne
from utils.constants import CHAN_NAME_DICT

def generate_mask(bz, ch_num, patch_num, mask_ratio, device):
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
    mask = mask.bernoulli_(mask_ratio)
    return mask

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

def to_tensor(array):
    return torch.from_numpy(array).float()

def get_1d_sincos_pos_embed(D, L, device):
    """
    return: (L, D)
    """
    assert D % 2 == 0
    pos = torch.arange(L, device=device).float().unsqueeze(1)  # (L,1)

    div_term = torch.exp(
        torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D)
    )

    pe = torch.zeros(L, D, device=device)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe

# ============================================================
# 4D fixed sin-cos positional encoding (x, y, z, t) -> dim
# ============================================================
# def _sincos_1d(x: torch.Tensor, dim: int, base: float = 10000.0) -> torch.Tensor:
#     """
#     x: (...,) continuous scalar
#     dim: even
#     return: (..., dim)
#     """
#     assert dim % 2 == 0, f"dim must be even, got {dim}"
#     half = dim // 2
#     device = x.device
#     omega = base ** (-torch.arange(half, device=device).float() / half)  # (half,)
#     xw = x.unsqueeze(-1) * omega  # (..., half)
#     return torch.cat([torch.sin(xw), torch.cos(xw)], dim=-1)  # (..., dim)


# def build_4d_sincos_pe(
#     xyz: torch.Tensor,
#     P: int,
#     dim: int,
#     base: float = 10000.0,
#     t_norm: bool = True,
# ) -> torch.Tensor:
#     """
#     xyz: (C, 3) channel coords in the SAME order as input channels
#     P: patch_num
#     dim: embedding dim
#     return: (1, C, P, dim)
#     """
#     device = xyz.device
#     C = xyz.shape[0]

#     # ---- patch index t ----
#     t = torch.arange(P, device=device).float()  # (P,)
#     if t_norm:
#         if P > 1:
#             t = 2.0 * (t / (P - 1)) - 1.0  # [-1, 1]
#         else:
#             t = t * 0.0

#     # ---- make grids (C,P) ----
#     x = xyz[:, 0].unsqueeze(1).expand(C, P)
#     y = xyz[:, 1].unsqueeze(1).expand(C, P)
#     z = xyz[:, 2].unsqueeze(1).expand(C, P)
#     tt = t.unsqueeze(0).expand(C, P)

#     # ---- allocate dims per component ----
#     per = dim // 4
#     per = per - (per % 2)  # even
#     if per < 2:
#         per = 2

#     dx = dy = dz = dt = per
#     used = dx + dy + dz + dt
#     rem = dim - used
#     if rem > 0:
#         dt += rem
#         if dt % 2 == 1:
#             dt -= 1  # leftover 1 dim will be padded later

#     # ---- build sincos ----
#     pe_x = _sincos_1d(x, dx, base=base)
#     pe_y = _sincos_1d(y, dy, base=base)
#     pe_z = _sincos_1d(z, dz, base=base)
#     pe_t = _sincos_1d(tt, dt, base=base)

#     pe = torch.cat([pe_x, pe_y, pe_z, pe_t], dim=-1)  # (C,P,>=dim-1)

#     # ---- pad / truncate ----
#     if pe.shape[-1] < dim:
#         pad = dim - pe.shape[-1]
#         pe = torch.cat([pe, torch.zeros(C, P, pad, device=device, dtype=pe.dtype)], dim=-1)
#     elif pe.shape[-1] > dim:
#         pe = pe[..., :dim]

#     return pe.unsqueeze(0)  # (1,C,P,dim)



def build_4d_fourier_pe(
    xyz: torch.Tensor,
    P: int,
    dim: int,
    n_freq: int = 3,
    t_norm: bool = True,
    scale_t_by_xyz_std: bool = True,

    # ---------------- NEW: xyz noise ----------------
    xyz_noise_std: float = 0.0025,   # e.g. 0.01 if xyz normalized to [-1,1]
) -> torch.Tensor:
    """
    4D Fourier positional encoding (deterministic + controlled stochasticity on xyz)

    xyz: (C, 3) channel coords in the SAME order as input channels
    P: patch_num
    dim: embedding dim
    n_freq: 1D freq count per axis; total freq vectors F = n_freq^4

    xyz_noise_std: std of Gaussian noise added to xyz (simulates electrode placement error)

    return: (1, C, P, dim)
    """
    device = xyz.device
    dtype = xyz.dtype
    C = xyz.shape[0]

    # =====================================================
    # 1) xyz with optional Gaussian noise (NO in-place)
    # =====================================================
    xyz_use = xyz
    if xyz_noise_std > 0.0:
        noise = torch.randn_like(xyz) * xyz_noise_std
        xyz_use = xyz + noise     # (C,3)

    # ---- patch index t ----
    t = torch.arange(P, device=device, dtype=dtype)  # (P,)
    if t_norm:
        if P > 1:
            t = 2.0 * (t / (P - 1)) - 1.0  # [-1, 1]
        else:
            t = t * 0.0

    # (optional) match scale of t with xyz magnitude
    if scale_t_by_xyz_std:
        s_t = xyz_use.std().clamp_min(torch.tensor(1e-6, device=device, dtype=dtype))
        t = t * s_t

    # ---- make grids (C,P) ----
    x = xyz_use[:, 0].unsqueeze(1).expand(C, P)  # (C,P)
    y = xyz_use[:, 1].unsqueeze(1).expand(C, P)
    z = xyz_use[:, 2].unsqueeze(1).expand(C, P)
    tt = t.unsqueeze(0).expand(C, P)

    # ---- stack to 4D coord: (C,P,4) then flatten -> (C*P,4) ----
    pos_4d = torch.stack([x, y, z, tt], dim=-1)          # (C,P,4)
    pos_4d_flat = pos_4d.reshape(C * P, 4)               # (C*P,4)

    # ---- build 4D frequency grid: (F,4), F=n_freq^4 ----
    freq_1d = torch.linspace(
        1.0, float(n_freq), steps=n_freq, device=device, dtype=dtype
    )  # (n_freq,)
    fx, fy, fz, ft = torch.meshgrid(
        freq_1d, freq_1d, freq_1d, freq_1d, indexing="ij"
    )
    freq_grid = torch.stack([fx, fy, fz, ft], dim=-1).reshape(-1, 4)  # (F,4)

    # ---- phase: (C*P,F) ----
    phase = 2.0 * math.pi * (pos_4d_flat @ freq_grid.T)  # (C*P, F)

    # ---- sin/cos features: (C*P, 2F) ----
    sin_feat = torch.sin(phase)
    cos_feat = torch.cos(phase)
    feat = torch.cat([sin_feat, cos_feat], dim=-1)       # (C*P, 2F)

    # ---- reshape back: (C,P,2F) ----
    pe = feat.view(C, P, -1)                             # (C,P,2F)

    return pe.unsqueeze(0)  # (1,C,P,dim)



def get_ch_pos():
    montage_name = "standard_1005"
    montage = mne.channels.make_standard_montage(montage_name)
    ch_pos = montage.get_positions().get("ch_pos", {})
    ch_pos_upper = {k.upper(): v for k, v in ch_pos.items()}
    # print(ch_pos_upper)
    return ch_pos_upper

def get_ch_coord(dataset_name):
    ch_pos_upper = get_ch_pos()
    ch_names = CHAN_NAME_DICT[dataset_name]
    xyz_list = []
    for name in ch_names:
        v = ch_pos_upper.get(name.upper(), [0.0, 0.0, 0.0])
        xyz_list.append([float(v[0]), float(v[1]), float(v[2])])
    return torch.as_tensor(xyz_list, dtype=torch.float32)
# ============================================================
# 2D编码：对channel维和patch维进行编码，同时要做归一化以适应不同长度的channel和patch_num
# ============================================================
def get_1d_sincos(pos: torch.Tensor, dim: int, base: float = 10000.0):
    """
    pos: (L,) float tensor
    return: (L, dim)
    """
    assert dim % 2 == 0
    half = dim // 2
    # (half,)
    omega = torch.arange(half, device=pos.device, dtype=pos.dtype)
    omega = 1.0 / (base ** (omega / half))
    # (L, half)
    out = pos[:, None] * omega[None, :]
    pe = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return pe  # (L, dim)

def get_2d_sincos_pe(C: int, P: int, dim: int, normalize=True, base=10000.0, device="cpu"):
    """
    return: (C*P, dim)  for tokens ordered as (c-major, then p) or reshape as needed
    """
    assert dim % 2 == 0
    dim_c = dim // 2
    dim_p = dim - dim_c  # also dim/2

    c = torch.arange(C, device=device).float()
    p = torch.arange(P, device=device).float()

    if normalize:
        # 映射到 [0,1]，避免不同 C/P 尺度导致频率相位过大
        c = c / max(C - 1, 1)
        p = p / max(P - 1, 1)

    pe_c = get_1d_sincos(c, dim_c, base=base)  # (C, dim/2)
    pe_p = get_1d_sincos(p, dim_p, base=base)  # (P, dim/2)

    # 生成网格：每个 token 对应 (c,p)
    # (C, P, dim/2) and (C, P, dim/2)
    pe_c = pe_c[:, None, :].expand(C, P, dim_c)
    pe_p = pe_p[None, :, :].expand(C, P, dim_p)

    pe = torch.cat([pe_c, pe_p], dim=-1)  # (C, P, dim)
    return pe


if __name__ == '__main__':
    # a = generate_mask(192, 32, 15, mask_ratio=0.5, device=None)
    # print(a)
    ch_pos = get_ch_coord()
    print(ch_pos)