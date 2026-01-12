import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from models.criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder
from utils.util import build_4d_sincos_pe, get_ch_coord


# ============================================================
# InfoNCE / NT-Xent
# ============================================================
def _l2_normalize(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Standard NT-Xent (SimCLR-style) for paired samples.
    z1, z2: (N, D) where i-th row in z1 matches i-th row in z2.
    """
    assert z1.shape == z2.shape and z1.dim() == 2, "z1/z2 must be (N,D) and same shape"
    N, _ = z1.shape

    z1 = _l2_normalize(z1, dim=-1)
    z2 = _l2_normalize(z2, dim=-1)

    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = (z @ z.t()) / temperature  # (2N, 2N)

    # mask self-similarity
    diag = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, float("-inf"))

    # positives: i <-> i+N
    pos_idx = torch.arange(N, device=z.device)
    targets = torch.cat([pos_idx + N, pos_idx], dim=0)  # (2N,)

    loss = F.cross_entropy(sim, targets)
    return loss


# ============================================================
# Augmentations
# ============================================================
def patch_time_masking(
    x: torch.Tensor,
    mask_ratio: float = 0.2,
    apply_prob: float = 1.0,
) -> torch.Tensor:
    """
    Patch-level time masking.
    x: (B, C, P, W)
    Randomly masks a contiguous segment along W in each (B,C,P) patch.
    """
    if mask_ratio <= 0:
        return x

    B, C, P, W = x.shape
    L = max(1, int(round(W * mask_ratio)))
    L = min(L, W)

    out = x.clone()

    # Decide which patches to apply masking
    if apply_prob < 1.0:
        apply_mask = (torch.rand(B, C, P, device=x.device) < apply_prob)
    else:
        apply_mask = torch.ones(B, C, P, device=x.device, dtype=torch.bool)

    # random start indices
    max_s = max(0, W - L)
    starts = torch.randint(0, max_s + 1, size=(B, C, P), device=x.device)

    # build boolean mask along W
    ar = torch.arange(W, device=x.device).view(1, 1, 1, W)  # (1,1,1,W)
    s = starts.unsqueeze(-1)                                 # (B,C,P,1)
    time_mask = (ar >= s) & (ar < (s + L))                   # (B,C,P,W)
    time_mask = time_mask & apply_mask.unsqueeze(-1)         # only applied patches

    out = out.masked_fill(time_mask, 0.0)
    return out


def mild_band_stop(
    x: torch.Tensor,
    atten: float = 0.7,
    band_bins: int = 6,
) -> torch.Tensor:
    """
    Sequence-level mild band-stop on the FULL sequence per channel.
    - Concatenate patches: T = P*W
    - rFFT along T
    - choose random center bin, attenuate a narrow band (multiplicative)
    - iFFT back
    x: (B, C, P, W)
    """
    assert 0.0 <= atten <= 1.0, "atten should be in [0,1], where 1 means full suppression, 0 means no change"
    B, C, P, W = x.shape
    T = P * W

    xt = x.reshape(B, C, T)  # (B,C,T)

    # rFFT
    X = torch.fft.rfft(xt, dim=-1, norm="forward")  # (B,C,F)
    Fbins = X.shape[-1]
    if Fbins <= 2:
        return x  # too short to band-stop meaningfully

    # pick random center (avoid DC bin 0)
    # bandwidth = band_bins (half on each side)
    half = max(1, band_bins // 2)
    center = torch.randint(1 + half, max(2 + half, Fbins - half), (B, C), device=x.device)

    # build mask over frequency bins
    f_idx = torch.arange(Fbins, device=x.device).view(1, 1, Fbins)  # (1,1,F)
    c = center.unsqueeze(-1)                                         # (B,C,1)
    band = (f_idx >= (c - half)) & (f_idx <= (c + half))             # (B,C,F)

    # attenuate (mild): multiply by (1-atten) in that band
    scale = torch.ones((B, C, Fbins), device=x.device, dtype=X.dtype)
    scale = scale.masked_fill(band, (1.0 - atten))
    X2 = X * scale

    # iFFT back
    xt2 = torch.fft.irfft(X2, n=T, dim=-1, norm="forward")  # (B,C,T)
    return xt2.reshape(B, C, P, W)


# ============================================================
# Main Model
# ============================================================
class CBraMod(nn.Module):
    def __init__(
        self,
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
        ch_names=None,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len, ch_names=ch_names)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)

        self.proj_out = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )

        # ----------------------------
        # Contrastive learning (minimal additions)
        # ----------------------------
        self.enable_contrastive = True  # 训练时启用；推理可关
        self.temperature = 0.2
        self.lambda_patch = 0.5
        self.lambda_global = 0.5

        # Patch-level projection head
        self.proj_patch = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

        # Global-level projection head
        self.proj_global = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

        # Expose last contrastive loss without changing forward signature
        self.contrastive_loss = None

        self.apply(_weights_init)

    def _to_bcpd(self, feats: torch.Tensor, C: int, P: int) -> torch.Tensor:
        """
        Make feats into (B,C,P,D) for pooling.
        Accepts:
          - (B,C,P,D): return as is
          - (B, C*P, D): reshape
        """
        if feats.dim() == 4:
            return feats
        if feats.dim() == 3:
            B, L, D = feats.shape
            assert L == C * P, f"Expected L==C*P, got L={L}, C*P={C*P}"
            return feats.reshape(B, C, P, D)
        raise RuntimeError(f"Unsupported feats shape: {feats.shape}")

    def forward(self, x, mask=None):
        """
        x: (B,C,P,W)
        return: out (same as before)
        Side-effect:
          self.contrastive_loss is set during training when enable_contrastive=True
        """
        # ---------- main path ----------
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)
        out = self.proj_out(feats)

        # ---------- contrastive (minimal interface change: store loss) ----------
        self.contrastive_loss = None
        if self.training and self.enable_contrastive:
            B, C, P, W = x.shape

            # Patch-level: time masking (two views)
            x_p1 = patch_time_masking(x, mask_ratio=0.2, apply_prob=1.0)
            x_p2 = patch_time_masking(x, mask_ratio=0.2, apply_prob=1.0)

            pe1 = self.patch_embedding(x_p1, mask=None)  # keep minimal: contrastive uses pure augment
            pe2 = self.patch_embedding(x_p2, mask=None)

            f1 = self.encoder(pe1)
            f2 = self.encoder(pe2)

            f1 = self._to_bcpd(f1, C=C, P=P)  # (B,C,P,D)
            f2 = self._to_bcpd(f2, C=C, P=P)

            # patch-level pooled over channels -> (B,P,D)
            p1 = f1.mean(dim=1)
            p2 = f2.mean(dim=1)

            # project -> (B*P, d_proj)
            z_p1 = self.proj_patch(p1.reshape(B * P, -1))
            z_p2 = self.proj_patch(p2.reshape(B * P, -1))
            loss_patch = info_nce(z_p1, z_p2, temperature=self.temperature)

            # Sequence-level: mild band-stop (two views)
            x_s1 = mild_band_stop(x, atten=0.7, band_bins=6)
            x_s2 = mild_band_stop(x, atten=0.7, band_bins=6)

            se1 = self.patch_embedding(x_s1, mask=None)
            se2 = self.patch_embedding(x_s2, mask=None)

            g1 = self.encoder(se1)
            g2 = self.encoder(se2)

            g1 = self._to_bcpd(g1, C=C, P=P)  # (B,C,P,D)
            g2 = self._to_bcpd(g2, C=C, P=P)

            # global pooled over C,P -> (B,D)
            gg1 = g1.mean(dim=(1, 2))
            gg2 = g2.mean(dim=(1, 2))

            z_g1 = self.proj_global(gg1)
            z_g2 = self.proj_global(gg2)
            loss_global = info_nce(z_g1, z_g2, temperature=self.temperature)

            self.contrastive_loss = self.lambda_patch * loss_patch + self.lambda_global * loss_global

        return out


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len, ch_names=None):
        super().__init__()
        self.d_model = d_model

        # (unused in current path; kept to preserve your code)
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=(19, 7),
                stride=(1, 1),
                padding=(9, 3),
                groups=d_model,
            ),
        )

        # Model B: fixed 4D PE
        self.ch_names = ch_names
        ch_pos = get_ch_coord()
        xyz_list = []
        for name in self.ch_names:
            if name in ch_pos:
                v = ch_pos[name]
                xyz_list.append([float(v[0]), float(v[1]), float(v[2])])
            else:
                xyz_list.append([0.0, 0.0, 0.0])
        self.register_buffer("chan_xyz", torch.tensor(xyz_list, dtype=torch.float32), persistent=True)  # (C,3)
        self.alpha_4d_enc = nn.Parameter(torch.tensor(0.5))

        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )

        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape

        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        # time-domain conv
        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        # spectral magnitude
        mask_x2 = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x2, dim=-1, norm="forward")
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)

        patch_emb = patch_emb + spectral_emb

        # 4D fixed PE (x,y,z,t)
        pe_4d = build_4d_sincos_pe(
            xyz=self.chan_xyz.to(device=x.device, dtype=x.dtype),  # (C,3)
            P=patch_num,
            dim=self.d_model,
            base=10000.0,
            t_norm=True,
        ).to(dtype=x.dtype, device=x.device)  # (1,C,P,D)

        patch_emb = patch_emb + self.alpha_4d_enc * pe_4d
        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 你实际运行时应传入 ch_names；这里给一个占位示例
    ch_names = ["Fp1"] * 16

    model = CBraMod(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
        ch_names=ch_names,
    ).to(device)

    # model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth', map_location=device))

    a = torch.randn((8, 16, 10, 200), device=device)
    b = model(a)

    print("out:", a.shape, "->", b.shape)
    if model.contrastive_loss is not None:
        print("contrastive_loss:", float(model.contrastive_loss.detach().cpu()))
