import torch
import torch.nn as nn
import torch.fft
from timm.models.vision_transformer import Block
import mne  # pip install mne

from utils.util import random_masking


# ============================================================
# 4D fixed sin-cos positional encoding (x, y, z, t) -> dim
# ============================================================
def _sincos_1d(x: torch.Tensor, dim: int, base: float = 10000.0) -> torch.Tensor:
    """
    x: (...,) continuous scalar
    dim: even
    return: (..., dim)
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half = dim // 2
    device = x.device
    omega = base ** (-torch.arange(half, device=device).float() / half)  # (half,)
    xw = x.unsqueeze(-1) * omega  # (..., half)
    return torch.cat([torch.sin(xw), torch.cos(xw)], dim=-1)  # (..., dim)


def build_4d_sincos_pe(
    xyz: torch.Tensor,
    P: int,
    dim: int,
    base: float = 10000.0,
    t_norm: bool = True,
) -> torch.Tensor:
    """
    xyz: (C, 3) channel coords in the SAME order as input channels
    P: patch_num
    dim: embedding dim
    return: (1, C, P, dim)
    """
    device = xyz.device
    C = xyz.shape[0]

    # ---- patch index t ----
    t = torch.arange(P, device=device).float()  # (P,)
    if t_norm:
        if P > 1:
            t = 2.0 * (t / (P - 1)) - 1.0  # [-1, 1]
        else:
            t = t * 0.0

    # ---- make grids (C,P) ----
    x = xyz[:, 0].unsqueeze(1).expand(C, P)
    y = xyz[:, 1].unsqueeze(1).expand(C, P)
    z = xyz[:, 2].unsqueeze(1).expand(C, P)
    tt = t.unsqueeze(0).expand(C, P)

    # ---- allocate dims per component ----
    per = dim // 4
    per = per - (per % 2)  # even
    if per < 2:
        per = 2

    dx = dy = dz = dt = per
    used = dx + dy + dz + dt
    rem = dim - used
    if rem > 0:
        dt += rem
        if dt % 2 == 1:
            dt -= 1  # leftover 1 dim will be padded later

    # ---- build sincos ----
    pe_x = _sincos_1d(x, dx, base=base)
    pe_y = _sincos_1d(y, dy, base=base)
    pe_z = _sincos_1d(z, dz, base=base)
    pe_t = _sincos_1d(tt, dt, base=base)

    pe = torch.cat([pe_x, pe_y, pe_z, pe_t], dim=-1)  # (C,P,>=dim-1)

    # ---- pad / truncate ----
    if pe.shape[-1] < dim:
        pad = dim - pe.shape[-1]
        pe = torch.cat([pe, torch.zeros(C, P, pad, device=device, dtype=pe.dtype)], dim=-1)
    elif pe.shape[-1] > dim:
        pe = pe[..., :dim]

    return pe.unsqueeze(0)  # (1,C,P,dim)


class CBraMod(nn.Module):
    def __init__(
        self,
        chan_num=19,
        seq_len=30,          # 仅保留为默认/参考；不再强制 P==seq_len
        in_dim=200,
        out_dim=200,
        d_model=800,
        mlp_ratio=4.0,
        n_layer=12,
        nhead=16,
        decoder_embed_dim=400,
        decoder_depth=4,
        decoder_num_heads=16,
        # === new args ===
        ch_names=None,               # ✅ 模型入参：list[str]，长度=chan_num
        montage_name="standard_1005",
        use_factor_pos=True,         # 是否启用 4D PE
        pe_base=10000.0,
        pe_t_norm=True,
    ):
        super().__init__()

        self.chan_num = chan_num
        self.seq_len = seq_len
        self.d_model = d_model
        self.in_dim = in_dim
        self.use_factor_pos = use_factor_pos

        self.pe_base = pe_base
        self.pe_t_norm = pe_t_norm

        # ============================================================
        # Channel names / montage coords (build once)
        # ============================================================
        assert ch_names is not None, "ch_names must be provided as a model init argument."
        assert len(ch_names) == chan_num, f"len(ch_names) must equal chan_num. Got {len(ch_names)} vs {chan_num}"
        self.ch_names = list(ch_names)

        montage = mne.channels.make_standard_montage(montage_name)
        ch_pos = montage.get_positions().get("ch_pos", {})

        xyz_list = []
        for name in self.ch_names:
            if name in ch_pos:
                v = ch_pos[name]
                xyz_list.append([float(v[0]), float(v[1]), float(v[2])])
            else:
                xyz_list.append([0.0, 0.0, 0.0])

        # ✅ 坐标注册为 buffer：随模型.to(device) 自动搬迁；不参与训练
        self.register_buffer("chan_xyz", torch.tensor(xyz_list, dtype=torch.float32), persistent=True)  # (C,3)

        # ============================================================
        # Encoder
        # ============================================================

        # ---- time-domain conv projector (like your proj_in) ----
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

        # Conv2d 后的时间特征维度 = 25 * L，其中 L 由 in_dim 和 conv stride/kernel 决定
        # 我们不手工算 L，直接用一个 dummy forward 推断一次，保证任意 in_dim 都对齐
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 1, in_dim)  # (B=1,1, tokens=1, W=in_dim)
            out = self.proj_in(dummy)             # (1,25,1,L)
            self._time_feat_dim = out.shape[-1] * out.shape[1]  # 25*L

        # 把 time-domain 特征映射到 d_model（泛化关键）
        self.time_to_d = nn.Sequential(
            nn.Linear(self._time_feat_dim, d_model),
            nn.Dropout(0.1),
        )

        # ---- spectral projector: rFFT magnitude bins -> d_model ----
        self.spec_bins = in_dim // 2 + 1  # rfft length
        self.spectral_proj = nn.Sequential(
            nn.Linear(self.spec_bins, d_model),
            nn.Dropout(0.1),
        )

        # self.encoder_embed = nn.Linear(in_dim, d_model, bias=True)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # ✅ CLS 的“位置向量”（替代原先 cls + 1d pos_embed[:1]）
        self.cls_pos_enc = nn.Parameter(torch.zeros(1, 1, d_model))

        self.blocks = nn.ModuleList([
            Block(d_model, nhead, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.alpha_4d_enc = nn.Parameter(torch.tensor(0.5))

        # ============================================================
        # Decoder (MAE)
        # ============================================================
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_embed = nn.Linear(d_model, decoder_embed_dim, bias=True)

        self.cls_pos_dec = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(decoder_depth)
        ])
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_dim, bias=True)

        self.alpha_4d_dec = nn.Parameter(torch.tensor(0.5))

        self.initialize_weights()

    # ------------------------------------------------------------
    # Init
    # ------------------------------------------------------------
    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        nn.init.constant_(self.cls_pos_enc, 0.0)
        nn.init.constant_(self.cls_pos_dec, 0.0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ============================================================
    # Encoder
    # ============================================================
    def forward_encoder(self, x, mask_ratio):
        """
        x: (B, C, P, W)
        return: latent, mask, ids_restore
        """
        B, C, P, W = x.shape
        assert C == self.chan_xyz.shape[0], f"Input C={C} must match model chan_num={self.chan_xyz.shape[0]}."

        # 1) patch embed: (B,C,P,W) -> (B,C,P,D)
        # x = self.encoder_embed(x)


        # (B,C,P,W) -> (B,1,C*P,W)
        xt = x.contiguous().view(B, 1, C * P, W)
        feat_t = self.proj_in(xt)  # (B,25,C*P,L)
        # (B,25,C*P,L) -> (B,C*P,25*L)
        feat_t = feat_t.permute(0, 2, 1, 3).contiguous().view(B, C * P, -1)
        # -> (B,C*P,D)
        patch_emb = self.time_to_d(feat_t).view(B, C, P, self.d_model)

        # -------------------------
        # (B) spectral magnitude features
        # -------------------------
        # (B,C,P,W) -> (B*C*P,W)
        xf = x.contiguous().view(B * C * P, W)
        spec = torch.fft.rfft(xf, dim=-1, norm="forward")     # (B*C*P, spec_bins)
        spec = torch.abs(spec).view(B, C, P, self.spec_bins)  # (B,C,P,spec_bins)
        spectral_emb = self.spectral_proj(spec)               # (B,C,P,D)

        # -------------------------
        # fuse
        # -------------------------
        x = patch_emb + spectral_emb  # (B,C,P,D)

        # 2) add 4D PE on (B,C,P,D)
        if self.use_factor_pos:
            pe_4d = build_4d_sincos_pe(
                xyz=self.chan_xyz.to(device=x.device, dtype=x.dtype),  # (C,3)
                P=P,
                dim=self.d_model,
                base=self.pe_base,
                t_norm=self.pe_t_norm,
            ).to(dtype=x.dtype, device=x.device)  # (1,C,P,D)
            x = x + self.alpha_4d_enc * pe_4d

        # 3) flatten: (B,C,P,D) -> (B, L, D), L=C*P
        flat_x = x.reshape(B, C * P, self.d_model)

        # 4) masking
        visible_x, mask, ids_restore = random_masking(flat_x, mask_ratio)

        # 5) cls (+ cls pos)
        cls_embed = (self.cls_token + self.cls_pos_enc).to(dtype=visible_x.dtype, device=visible_x.device)
        cls_embed = cls_embed.expand(B, -1, -1)
        visible_x = torch.cat([cls_embed, visible_x], dim=1)

        # 6) transformer
        for blk in self.blocks:
            visible_x = blk(visible_x)
        visible_x = self.norm(visible_x)

        return visible_x, mask, ids_restore

    # ============================================================
    # Decoder
    # ============================================================
    def forward_decoder(self, x, ids_restore):
        """
        x: (B, 1+L_vis, d_model)
        ids_restore: (B, L) with L=C*P
        return: pred (B, L, in_dim)
        """
        B = x.shape[0]
        C = self.chan_xyz.shape[0]
        L = ids_restore.shape[1]
        assert L % C == 0, f"L={L} must be divisible by C={C} to infer P."
        P = L // C

        x = self.decoder_embed(x)  # (B, 1+L_vis, dec_dim)

        # unshuffle with mask tokens
        mask_tokens = self.mask_token.repeat(B, L + 1 - x.shape[1], 1)  # +1 for cls
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)               # (B, L, dec_dim)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)                         # (B, 1+L, dec_dim)

        # cls pos
        x[:, :1, :] = x[:, :1, :] + self.cls_pos_dec.to(dtype=x.dtype, device=x.device)

        # token 4D pos (decoder)
        if self.use_factor_pos:
            pe_4d_dec = build_4d_sincos_pe(
                xyz=self.chan_xyz.to(device=x.device, dtype=x.dtype),  # (C,3)
                P=P,
                dim=self.decoder_pred.in_features,  # decoder_embed_dim
                base=self.pe_base,
                t_norm=self.pe_t_norm,
            ).to(dtype=x.dtype, device=x.device)  # (1,C,P,dec_dim)
            pe_flat = pe_4d_dec.reshape(1, L, pe_4d_dec.shape[-1])
            x[:, 1:, :] = x[:, 1:, :] + self.alpha_4d_dec * pe_flat

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)  # (B, 1+L, in_dim)
        x = x[:, 1:, :]           # (B, L, in_dim)
        return x

    # ============================================================
    # Loss / Forward
    # ============================================================
    def forward_loss(self, x, pred, mask):
        b, c, p, w = x.shape
        x = x.reshape(b, c * p, w)
        loss = (pred - x) ** 2
        loss = loss.mean(dim=-1)
        denom = mask.sum().clamp_min(1.0)
        loss = (loss * mask).sum() / denom
        return loss

    def forward(self, x, mask_ratio=0):
        """
        x: (B, C, P, W)
        """
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask
