import os
import random
import signal

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
import math

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


if __name__ == '__main__':
    a = generate_mask(192, 32, 15, mask_ratio=0.5, device=None)
    print(a)