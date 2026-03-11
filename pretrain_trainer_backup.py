import os
import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.util import generate_mask


# -------------------------
# DDP helpers (GLOBAL batch contrastive WITH grad)
# -------------------------
def _is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


class _GatherLayer(torch.autograd.Function):
    """
    All-gather with backward support.

    Forward:
      input  x: (B, ...)
      output tuple(xs): world tensors, each (B, ...)

    Backward:
      receives grads for each gathered tensor; keep only local slice,
      then all-reduce so every rank contributes correctly.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        if not _is_ddp():
            return (x,)
        world = dist.get_world_size()
        out = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(out, x.contiguous())
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        # grads: tuple length = world, each is grad of corresponding gathered output
        if not _is_ddp():
            return grads[0]
        rank = dist.get_rank()
        grad_local = grads[rank].contiguous()
        # make sure all ranks participate
        dist.all_reduce(grad_local, op=dist.ReduceOp.SUM)
        return grad_local


def _ddp_all_gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """
    Gather tensor from all ranks WITH autograd. Return concatenated along dim=0.
    x: (B, ...)
    """
    if not _is_ddp():
        return x
    xs = _GatherLayer.apply(x)  # tuple of (B, ...)
    return torch.cat(list(xs), dim=0)  # (world*B, ...)


class Trainer(object):
    def __init__(self, params, data_loader, model, batch_sampler=None):
        self.params = params

        local_rank = int(getattr(self.params, "local_rank", 0))
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.rank = int(getattr(self.params, "rank", 0))
        self.world_size = int(getattr(self.params, "world_size", 1))
        self.is_ddp = self.world_size > 1

        self.data_loader = data_loader
        self.batch_sampler = batch_sampler

        self.model = model.to(self.device)

        # -------- loss / training mode
        self.criterion = nn.MSELoss(reduction="mean").to(self.device)

        if self.is_ddp:
            self.model = DDP(
                self.model,
                device_ids=[local_rank] if self.device.type == "cuda" else None,
                output_device=local_rank if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        self.data_length = len(self.data_loader)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.params.lr),
            weight_decay=float(self.params.weight_decay),
        )

        # scheduler
        if self.params.lr_scheduler == "CosineAnnealingLR":
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=40 * self.data_length, eta_min=1e-5
            )
        elif self.params.lr_scheduler == "ExponentialLR":
            self.optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.999999999
            )
        elif self.params.lr_scheduler == "StepLR":
            self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5 * self.data_length, gamma=0.5
            )
        elif self.params.lr_scheduler == "MultiStepLR":
            self.optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[10 * self.data_length, 20 * self.data_length, 30 * self.data_length],
                gamma=0.1,
            )
        elif self.params.lr_scheduler == "CyclicLR":
            self.optimizer_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=1e-6,
                max_lr=0.001,
                step_size_up=self.data_length * 5,
                step_size_down=self.data_length * 2,
                mode="exp_range",
                gamma=0.9,
                cycle_momentum=False,
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.params.lr_scheduler}")

        # -------------------------
        # Augmentation config
        # -------------------------
        self.min_keep_ch = int(getattr(self.params, "min_keep_ch", 12))
        self.max_keep_ch = int(getattr(self.params, "max_keep_ch", 0))  # 0 -> use C
        self.channel_shuffle = bool(getattr(self.params, "channel_shuffle", True))

        # self.use_gaussian_noise = bool(getattr(self.params, "use_gaussian_noise", False))
        # self.noise_std = float(getattr(self.params, "noise_std", 0.01))

        # self.use_amp_scale = bool(getattr(self.params, "use_amp_scale", False))
        # self.amp_scale_min = float(getattr(self.params, "amp_scale_min", 0.9))
        # self.amp_scale_max = float(getattr(self.params, "amp_scale_max", 1.1))

        # # -------------------------
        # # Augmentation config (EEG common augs)
        # # -------------------------
        # self.num_augs_per_view = int(getattr(self.params, "num_augs_per_view", 2))  # 每个 view 采样几种增强
        # assert self.num_augs_per_view >= 0

        # # 1) jitter
        # self.aug_jitter_std = float(getattr(self.params, "aug_jitter_std", 0.02))  # 相对幅度噪声强度（建议 0.01~0.05）

        # # 2) scaling
        # self.aug_scale_min = float(getattr(self.params, "aug_scale_min", 0.8))
        # self.aug_scale_max = float(getattr(self.params, "aug_scale_max", 1.2))

        # # 3) time shift (roll along patch axis P)
        # self.aug_shift_max_ratio = float(getattr(self.params, "aug_shift_max_ratio", 0.15))  # 最大平移比例（相对 P）

        # # 4) time mask (mask contiguous patches)
        # self.aug_time_mask_ratio = float(getattr(self.params, "aug_time_mask_ratio", 0.15))  # mask 掉多少比例的 patch（相对 P）

        # # 5) frequency dropout (on flattened time axis P*W)
        # self.aug_freq_drop_ratio = float(getattr(self.params, "aug_freq_drop_ratio", 0.12))  # dropout 的频率带宽比例（相对频点数）
        # self.aug_freq_drop_num = int(getattr(self.params, "aug_freq_drop_num", 1))          # 丢弃几个频带段

        # # 候选增强池（每次从这里采样）
        # self.aug_pool = list(getattr(
        #     self.params,
        #     "aug_pool",
        #     ["jitter", "scaling", "time_shift", "time_mask", "freq_dropout"]
        # ))

        # # -------------------------
        # # Contrastive config (SimCLR-style)
        # # -------------------------
        # self.temperature = float(getattr(self.params, "contrastive_tau", 0.1))
        # # global pooling mode for representation: "mean" / "cls"(if provided)
        # self.pool_mode = str(getattr(self.params, "contrastive_pool", "mean")).lower()

        # # Spatial constraints (optional)
        # self.use_spatial_kl = bool(getattr(self.params, "use_spatial_kl", True))
        # self.lambda_spatial_kl = float(getattr(self.params, "lambda_spatial_kl", 0.1))
        # self.sigma_s = float(getattr(self.params, "sigma_s", 0.15))  # 距离核宽度（需按你的坐标尺度调）
        # self.spatial_p_subsample = int(getattr(self.params, "spatial_p_subsample", 0))  # 0->全用；>0 随机取这么多个P做空间loss

        # self.use_anti_lowpass = bool(getattr(self.params, "use_anti_lowpass", False))
        # self.lambda_anti_lowpass = float(getattr(self.params, "lambda_anti_lowpass", 0.02))
        # self.anti_lp_eta = float(getattr(self.params, "anti_lp_eta", -1.0))  # <0 -> 自动用初始值估计
        # self.knn_k = int(getattr(self.params, "spatial_knn_k", 6))

    # -------------------------
    # Checkpoint
    # -------------------------
    def _save_ckpt(self, path, epoch, best_loss):
        if self.rank != 0:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        ckpt = {
            "epoch": epoch,
            "best_loss": best_loss,
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.optimizer_scheduler.state_dict(),
            "dataset_list": self.params.dataset_list,
            "batch_size": self.params.batch_size,
            "lr": self.params.lr,
            "weight_decay": self.params.weight_decay,
            "lr_scheduler": self.params.lr_scheduler,
            "train_mode": self.params.train_mode,
            # recon
            "mask_ratio": self.params.mask_ratio,
            # contrastive
            "contrastive_tau": float(getattr(self.params, "contrastive_tau", 0.1)),
            "contrastive_pool": str(getattr(self.params, "contrastive_pool", "mean")),
            "use_spatial_kl": bool(getattr(self.params, "use_spatial_kl", True)),
            "lambda_spatial_kl": float(getattr(self.params, "lambda_spatial_kl", 0.1)),
            "use_anti_lowpass": bool(getattr(self.params, "use_anti_lowpass", False)),
            "lambda_anti_lowpass": float(getattr(self.params, "lambda_anti_lowpass", 0.02)),
            "sigma_s": float(getattr(self.params, "sigma_s", 0.15)),
            "spatial_knn_k": int(getattr(self.params, "spatial_knn_k", 6)),
        }
        torch.save(ckpt, path)

    def _load_ckpt(self, path):
        ckpt = torch.load(path, map_location=self.device)
        model_state = ckpt.get("model", ckpt)
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_load.load_state_dict(model_state, strict=True)

        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.optimizer_scheduler.load_state_dict(ckpt["scheduler"])

        start_epoch = int(ckpt.get("epoch", 0))
        best_loss = float(ckpt.get("best_loss", 1e9))
        return start_epoch, best_loss

    # -------------------------
    # Channel select helpers
    # -------------------------
    def _choose_k(self, C: int) -> int:
        min_k = max(1, min(self.min_keep_ch, C))
        max_k = C if (self.max_keep_ch is None or int(self.max_keep_ch) <= 0) else min(int(self.max_keep_ch), C)
        if max_k < min_k:
            max_k = min_k

        if min_k == max_k:
            return min_k
        return random.randint(min_k, max_k)

    def _sample_channel_index(self, C: int, device: torch.device) -> torch.Tensor:
        k = self._choose_k(C)
        if k == C:
            idx = torch.arange(C, device=device)
        else:
            idx = torch.randperm(C, device=device)[:k]
            if not self.channel_shuffle:
                idx, _ = torch.sort(idx)
        if self.channel_shuffle:
            idx = idx[torch.randperm(idx.numel(), device=device)]
        return idx

    def _apply_channel_index(self, x: torch.Tensor, ch_coords: torch.Tensor, ch_names, idx: torch.Tensor):
        x = x.index_select(dim=1, index=idx)
        ch_coords = ch_coords.index_select(dim=0, index=idx)
        if ch_names is not None and isinstance(ch_names, (list, tuple)):
            idx_list = idx.detach().to("cpu").tolist()
            ch_names = [ch_names[i] for i in idx_list]
        return x, ch_coords, ch_names

    # # -------------------------
    # # Augmentation helpers
    # # -------------------------
    # def _aug_jitter(self, x: torch.Tensor) -> torch.Tensor:
    #     if self.aug_jitter_std <= 0:
    #         return x
    #     return x + torch.randn_like(x) * self.aug_jitter_std

    # def _aug_scaling(self, x: torch.Tensor) -> torch.Tensor:
    #     if self.aug_scale_max <= 0 or self.aug_scale_max <= self.aug_scale_min:
    #         return x
    #     # per-sample scale
    #     s = self.aug_scale_min + (self.aug_scale_max - self.aug_scale_min) * torch.rand(
    #         (x.size(0), 1, 1, 1), device=x.device, dtype=x.dtype
    #     )
    #     return x * s

    # def _aug_time_shift(self, x: torch.Tensor) -> torch.Tensor:
    #     # x: (B,C,P,W)  -> roll along P (patch axis)
    #     B, C, P, W = x.shape
    #     max_shift = int(max(1, round(P * self.aug_shift_max_ratio)))
    #     if max_shift <= 0:
    #         return x
    #     # 每个样本一个 shift（也可改成每个样本每通道）
    #     shifts = torch.randint(low=-max_shift, high=max_shift + 1, size=(B,), device=x.device)
    #     # torch.roll 不支持 batch-wise shift，手动做
    #     out = x.clone()
    #     for b in range(B):
    #         s = int(shifts[b].item())
    #         if s != 0:
    #             out[b] = torch.roll(out[b], shifts=s, dims=1)  # dims=1 对应 P 维（因为 out[b] 是 (C,P,W)）
    #     return out

    # def _aug_time_mask(self, x: torch.Tensor) -> torch.Tensor:
    #     # mask contiguous patches on P axis
    #     B, C, P, W = x.shape
    #     m = int(round(P * self.aug_time_mask_ratio))
    #     m = max(1, min(m, P))
    #     out = x.clone()
    #     for b in range(B):
    #         start = int(torch.randint(0, max(1, P - m + 1), (1,), device=x.device).item())
    #         out[b, :, start:start + m, :] = 0.0
    #     return out

    # def _aug_freq_dropout(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     在每个 (B,C) 上对时间轴做 rFFT，随机丢弃频带，再 irFFT
    #     x: (B,C,P,W) -> flatten time = P*W
    #     """
    #     B, C, P, W = x.shape
    #     T = P * W
    #     if T < 8:
    #         return x

    #     drop_ratio = float(self.aug_freq_drop_ratio)
    #     if drop_ratio <= 0:
    #         return x

    #     x_flat = x.reshape(B, C, T)

    #     # rfft: (B,C,F)
    #     X = torch.fft.rfft(x_flat, dim=-1)
    #     Fbins = X.size(-1)

    #     band = int(max(1, round(Fbins * drop_ratio)))
    #     band = min(band, max(1, Fbins - 1))

    #     # 丢弃 aug_freq_drop_num 段频带
    #     for _ in range(max(1, self.aug_freq_drop_num)):
    #         start = int(torch.randint(1, max(2, Fbins - band), (1,), device=x.device).item())
    #         X[..., start:start + band] = 0

    #     x_rec = torch.fft.irfft(X, n=T, dim=-1)
    #     return x_rec.reshape(B, C, P, W)

    # def _apply_one_aug(self, x: torch.Tensor, aug_name: str) -> torch.Tensor:
    #     if aug_name == "jitter":
    #         return self._aug_jitter(x)
    #     if aug_name == "scaling":
    #         return self._aug_scaling(x)
    #     if aug_name == "time_shift":
    #         return self._aug_time_shift(x)
    #     if aug_name == "time_mask":
    #         return self._aug_time_mask(x)
    #     if aug_name == "freq_dropout":
    #         return self._aug_freq_dropout(x)
    #     return x  # unknown -> no-op

    # def _sample_augs(self) -> list:
    #     pool = [a for a in self.aug_pool if isinstance(a, str)]
    #     if len(pool) == 0 or self.num_augs_per_view <= 0:
    #         return []
    #     k = min(self.num_augs_per_view, len(pool))
    #     # python random: rank 间会不同；这是我们想要的（每个 rank / batch 都是独立增强）
    #     return random.sample(pool, k=k)

    # def _other_augs(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     对一个 view 随机采样 N 个增强并顺序应用
    #     """
    #     augs = self._sample_augs()
    #     # 顺序也随机一下，增强多样性
    #     random.shuffle(augs)
    #     for a in augs:
    #         x = self._apply_one_aug(x, a)
    #     return x

    # def _make_two_views(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     每个 view 独立随机选 2 种增强
    #     """
    #     x1 = self._other_augs(x.clone())
    #     x2 = self._other_augs(x.clone())
    #     return x1, x2

    # # -------------------------
    # # Contrastive helpers
    # # -------------------------
    # def _encode_for_contrastive(self, x: torch.Tensor, ch_coords: torch.Tensor) -> torch.Tensor:
    #     """
    #     Return patch-level representation z with shape (B, C, P, D) or (B, T, D).
    #     You MUST have one of:
    #       - model.forward_features(x, ch_coords=...)
    #       - model.encode(x, ch_coords=...)
    #       - model.get_representation(x, ch_coords=...)
    #     """
    #     m = self.model.module if isinstance(self.model, DDP) else self.model

    #     if hasattr(m, "forward_features"):
    #         z = m.forward_features(x, ch_coords=ch_coords)
    #     elif hasattr(m, "encode"):
    #         z = m.encode(x, ch_coords=ch_coords)
    #     elif hasattr(m, "get_representation"):
    #         z = m.get_representation(x, ch_coords=ch_coords)
    #     else:
    #         raise RuntimeError(
    #             "For contrastive training, your model must implement one of:\n"
    #             "  - forward_features(x, ch_coords=...)\n"
    #             "  - encode(x, ch_coords=...)\n"
    #             "  - get_representation(x, ch_coords=...)\n"
    #             "and return a representation tensor (B,C,P,D) or (B,T,D)."
    #         )

    #     if z.dim() not in (3, 4):
    #         raise RuntimeError(f"Unexpected representation shape: {tuple(z.shape)}. Expect (B,T,D) or (B,C,P,D).")
    #     return z

    # def _pool_global(self, z: torch.Tensor) -> torch.Tensor:
    #     """
    #     z: (B, T, D) or (B, C, P, D)
    #     return g: (B, D)
    #     """
    #     if z.dim() == 4:
    #         # (B,C,P,D) -> mean over C,P
    #         g = z.mean(dim=(1, 2))
    #     else:
    #         # (B,T,D) -> mean over T
    #         g = z.mean(dim=1)
    #     g = F.normalize(g, dim=-1)
    #     return g

    # def _simclr_loss(self, g1: torch.Tensor, g2: torch.Tensor, tau: float) -> torch.Tensor:
    #     """
    #     NT-Xent (SimCLR) using GLOBAL batch across all GPUs (with grad).
    #     - Uses local anchors, global negatives (all_gather with autograd).
    #     - Symmetric loss: g1->g2 and g2->g1
    #     g1,g2: (B,D) normalized

    #     NOTE:
    #       Requires equal local batch size across ranks (use drop_last=True).
    #     """
    #     assert g1.dim() == 2 and g2.dim() == 2, f"Expect (B,D), got {g1.shape} {g2.shape}"
    #     B = g1.size(0)
    #     device = g1.device

    #     if _is_ddp():
    #         world = dist.get_world_size()
    #         rank = dist.get_rank()

    #         # gather WITH grad => truly global-batch contrastive training
    #         g1_all = _ddp_all_gather_with_grad(g1)  # (world*B, D)
    #         g2_all = _ddp_all_gather_with_grad(g2)  # (world*B, D)

    #         # local sample i on rank r corresponds to global index (r*B + i)
    #         base = rank * B
    #         labels = torch.arange(B, device=device) + base
    #     else:
    #         g1_all, g2_all = g1, g2
    #         labels = torch.arange(B, device=device)

    #     logits_12 = (g1 @ g2_all.t()) / tau  # (B, world*B)
    #     logits_21 = (g2 @ g1_all.t()) / tau  # (B, world*B)

    #     loss_12 = F.cross_entropy(logits_12, labels)
    #     loss_21 = F.cross_entropy(logits_21, labels)
    #     return 0.5 * (loss_12 + loss_21)

    # @torch.no_grad()
    # def _build_knn_graph(self, ch_coords: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Build KNN indices and distance matrix.
    #     ch_coords: (C,3) float
    #     return:
    #       knn_idx: (C,k) long
    #       dist: (C,C) float (pairwise)
    #     """
    #     C = ch_coords.size(0)
    #     dist_mat = torch.cdist(ch_coords, ch_coords)  # (C,C)
    #     # exclude self
    #     dist_self = dist_mat + torch.eye(C, device=dist_mat.device, dtype=dist_mat.dtype) * 1e9
    #     knn_idx = torch.topk(dist_self, k=min(k, C - 1), dim=1, largest=False).indices  # (C,k)
    #     return knn_idx, dist_mat

    # def _spatial_kl_loss(self, z: torch.Tensor, ch_coords: torch.Tensor, tau: float) -> torch.Tensor:
    #     """
    #     Spatial soft constraint: KL(pi || q) where
    #       pi(c') ∝ exp(-dist(c,c')/sigma_s)
    #       q(c')  ∝ exp(sim(z_c, z_c')/tau)
    #     z: (B,C,P,D) normalized recommended
    #     Return scalar.
    #     Complexity O(B*P*C^2). Use subsample P if needed.
    #     """
    #     if z.dim() != 4:
    #         return z.new_zeros(())

    #     B, C, P, D = z.shape

    #     # subsample P to control cost
    #     if self.spatial_p_subsample and self.spatial_p_subsample > 0 and self.spatial_p_subsample < P:
    #         p_idx = torch.randperm(P, device=z.device)[: self.spatial_p_subsample]
    #         z = z[:, :, p_idx, :]
    #         P = z.size(2)

    #     z = F.normalize(z, dim=-1)

    #     _, dist_mat = self._build_knn_graph(ch_coords, k=self.knn_k)
    #     # distance kernel -> pi over channels (exclude self)
    #     # pi: (C,C)
    #     pi = torch.exp(-dist_mat / max(self.sigma_s, 1e-8))
    #     pi.fill_diagonal_(0.0)
    #     pi = pi / (pi.sum(dim=1, keepdim=True).clamp_min(1e-12))  # row-stochastic

    #     # compute q from similarities per (B,P):
    #     # sim_cp: (B,P,C,C) = z[b,:,p] @ z[b,:,p]^T
    #     z_bp = z.permute(0, 2, 1, 3).contiguous()  # (B,P,C,D)
    #     sim = torch.matmul(z_bp, z_bp.transpose(-1, -2)) / tau  # (B,P,C,C)
    #     sim = sim - 1e9 * torch.eye(C, device=sim.device, dtype=sim.dtype)[None, None, :, :]
    #     q = F.softmax(sim, dim=-1)  # (B,P,C,C)

    #     # KL(pi||q) averaged over B,P,C
    #     pi_bc = pi[None, None, :, :].clamp_min(1e-12)
    #     q = q.clamp_min(1e-12)
    #     kl = (pi_bc * (pi_bc.log() - q.log())).sum(dim=-1)  # (B,P,C)
    #     return kl.mean()

    # def _anti_lowpass_loss(self, z: torch.Tensor, ch_coords: torch.Tensor) -> torch.Tensor:
    #     """
    #     Encourage not-too-smooth embeddings across channel graph:
    #       E_smooth = tr(Z^T L Z)
    #       L_anti = max(0, eta - E_smooth)
    #     z: (B,C,P,D) normalized recommended
    #     """
    #     if z.dim() != 4:
    #         return z.new_zeros(())
    #     B, C, P, D = z.shape

    #     # subsample P for cost
    #     if self.spatial_p_subsample and self.spatial_p_subsample > 0 and self.spatial_p_subsample < P:
    #         p_idx = torch.randperm(P, device=z.device)[: self.spatial_p_subsample]
    #         z = z[:, :, p_idx, :]
    #         P = z.size(2)

    #     z = F.normalize(z, dim=-1)

    #     # adjacency with KNN weights
    #     knn_idx, dist_mat = self._build_knn_graph(ch_coords, k=self.knn_k)
    #     A = torch.zeros((C, C), device=z.device, dtype=z.dtype)
    #     # weight: exp(-dist/sigma)
    #     w = torch.exp(-dist_mat / max(self.sigma_s, 1e-8))
    #     for c in range(C):
    #         nbr = knn_idx[c]
    #         A[c, nbr] = w[c, nbr]
    #     # symmetrize
    #     A = 0.5 * (A + A.t())
    #     Dg = torch.diag(A.sum(dim=1))
    #     L = Dg - A  # (C,C)

    #     # compute E_smooth per (B,P): tr(Z^T L Z) = sum_d Z[:,d]^T L Z[:,d]
    #     z_bp = z.permute(0, 2, 1, 3).contiguous()  # (B,P,C,D)
    #     zp = z_bp.view(B * P, C, D)  # (B*P,C,D)
    #     LZ = torch.matmul(L, zp)  # (C,C)@(B*P,C,D) => (B*P,C,D)
    #     Es = (zp * LZ).sum(dim=(1, 2))  # (B*P,)

    #     Es_mean = Es.mean()

    #     # auto-set eta on first call if needed
    #     if self.anti_lp_eta < 0:
    #         self.anti_lp_eta = float(0.5 * Es_mean.detach().item())
    #         if self.rank == 0:
    #             print(f"[antiLP] auto eta set to {self.anti_lp_eta:.6f}")

    #     eta = z.new_tensor(self.anti_lp_eta)
    #     loss = F.relu(eta - Es_mean)
    #     return loss

    # #================================================================
    # # wav2vec loss
    # #================================================================
    def _flatten_tokens(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B,C,P,D) -> (B, T=C*P, D)
        """
        assert z.dim() == 4
        B, C, P, D = z.shape
        return z.reshape(B, C * P, D)

    def _flatten_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        mask: (B,C,P) {0,1} -> (B, T=C*P) bool
        """
        assert mask.dim() == 3
        return (mask.reshape(mask.size(0), -1) > 0)

    # def _sample_negatives_wav2vec2(
    #     self,
    #     y: torch.Tensor,
    #     num: int,
    #     padding_count: int = 0,
    # ):
    #     """
    #     wav2vec2 的 sample_negatives 逻辑（简化版）：
    #     y: (B, T, D)  targets pool
    #     num: 每个样本需要的 anchor 数（这里就是 num_mask）
    #     return:
    #     negs: (Nneg, B, num, D)
    #     """
    #     n_neg = int(getattr(self.params, "num_negatives", 100))
    #     cross_n_neg = int(getattr(self.params, "cross_sample_negatives", 0))
    #     if n_neg == 0 and cross_n_neg == 0:
    #         return y.new_zeros((0, y.size(0), num, y.size(-1)))

    #     B, T, D = y.shape
    #     high = T - int(padding_count or 0)
    #     cross_high = T * B
    #     assert high > 1, f"Invalid T after padding_count: B={B}, T={T}, high={high}"

    #     y_flat = y.reshape(B * T, D)

    #     with torch.no_grad():
    #         # in-sample negatives
    #         if n_neg > 0:
    #             # tszs: (num*n_neg,) = [0,0,...,1,1,...]  对应每个 anchor 的时间索引
    #             tszs = torch.arange(num, device=y.device).unsqueeze(-1).expand(num, n_neg).reshape(-1)  # (num*n_neg,)
    #             neg_idxs = torch.randint(low=0, high=high - 1, size=(B, n_neg * num), device=y.device)
    #             neg_idxs[neg_idxs >= tszs.unsqueeze(0)] += 1  # 避免采到正样本自身

    #         # cross-sample negatives
    #         if cross_n_neg > 0:
    #             tszs = torch.arange(num, device=y.device).unsqueeze(-1).expand(num, cross_n_neg).reshape(-1)
    #             cross_neg_idxs = torch.randint(low=0, high=cross_high - 1, size=(B, cross_n_neg * num), device=y.device)
    #             cross_neg_idxs[cross_neg_idxs >= tszs.unsqueeze(0)] += 1

    #     if n_neg > 0:
    #         neg_idxs = neg_idxs + (torch.arange(B, device=y.device).unsqueeze(1) * high)
    #         if cross_n_neg > 0:
    #             neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)
    #     else:
    #         neg_idxs = cross_neg_idxs

    #     Ntot = n_neg + cross_n_neg
    #     negs = y_flat[neg_idxs.reshape(-1)].reshape(B, num, Ntot, D).permute(2, 0, 1, 3)  # (Ntot,B,num,D)
    #     return negs

    # def _compute_logits_wav2vec2(self, x: torch.Tensor, y_pos: torch.Tensor, y_negs: torch.Tensor, temp: float):
    #     """
    #     x: (B, num, D)         context preds at masked positions
    #     y_pos: (B, num, D)     positive targets (quantized) at masked positions
    #     y_negs: (Nneg,B,num,D) negatives
    #     return:
    #     logits: (B*num, 1+Nneg)   CE expects this
    #     labels: (B*num,) all zeros
    #     """
    #     # targets: (1+Nneg, B, num, D)
    #     targets = torch.cat([y_pos.unsqueeze(0), y_negs], dim=0)

    #     # cosine sim over last dim -> (1+Nneg, B, num)
    #     logits = F.cosine_similarity(x.unsqueeze(0).float(), targets.float(), dim=-1) / temp
    #     # -> (B, num, 1+Nneg)
    #     logits = logits.permute(1, 2, 0).contiguous()
    #     logits = logits.view(-1, logits.size(-1))  # (B*num, 1+Nneg)

    #     labels = torch.zeros((logits.size(0),), device=logits.device, dtype=torch.long)
    #     return logits, labels

    # def _wav2vec2_contrastive_loss(self, x_raw: torch.Tensor, ch_coords: torch.Tensor):
    #     assert x_raw.dim() == 4, "Expect x as (B,C,P,W)"
    #     B, C, P, _ = x_raw.shape

    #     # (1) mask
    #     mask = generate_mask(
    #         B, C, P,
    #         mask_ratio=float(getattr(self.params, "mask_ratio", 0.5)),
    #         device=x_raw.device,
    #     )  # 0/1
    #     mask_bool = self._flatten_mask(mask)  # (B,T) bool, T=C*P

    #     # (2) model outputs
    #     m = self.model.module if isinstance(self.model, DDP) else self.model
    #     feats_ctx, q_out = m.get_context_and_quantized(x_raw, mask=mask, ch_coords=ch_coords)
    #     feats_q = q_out["x"]  # (B,C,P,D)

    #     # (3) flatten tokens
    #     ctx = self._flatten_tokens(feats_ctx)  # (B,T,D)
    #     q   = self._flatten_tokens(feats_q)    # (B,T,D)

    #     # (4) flatten batch first: (B*T, D), and select masked positions
    #     T = ctx.size(1)
    #     ctx_flat = ctx.reshape(B * T, -1)
    #     q_flat   = q.reshape(B * T, -1)

    #     mask_flat = mask_bool.reshape(B * T)  # (B*T,)
    #     x = F.normalize(ctx_flat[mask_flat], dim=-1)  # (Nmask,D)
    #     y = F.normalize(q_flat[mask_flat], dim=-1)    # (Nmask,D)

    #     # (5) in-batch InfoNCE on masked tokens
    #     tau = float(getattr(self.params, "logit_temp", getattr(self.params, "contrastive_tau", 0.1)))
    #     logits_xy = (x @ y.t()) / tau  # (Nmask,Nmask)
    #     labels = torch.arange(logits_xy.size(0), device=logits_xy.device)

    #     loss_xy = F.cross_entropy(logits_xy, labels)
    #     logits_yx = (y @ x.t()) / tau
    #     loss_yx = F.cross_entropy(logits_yx, labels)

    #     loss = 0.5 * (loss_xy + loss_yx)
    #     return loss, mask, q_out



    # # ------------------------------------------------------------
    # # helpers: negatives + logits (CPC-style)
    # # ------------------------------------------------------------
    # def _sample_negatives_global(self, y_pool: torch.Tensor, pos_idx: torch.Tensor, n_neg: int):
    #     """
    #     y_pool: (N, D) global pool
    #     pos_idx: (M,) long, positive indices in [0, N)
    #     return negs: (M, n_neg, D)
    #     """
    #     N, D = y_pool.shape
    #     M = pos_idx.numel()
    #     # sample in [0, N-1], then shift to avoid == pos_idx
    #     neg = torch.randint(low=0, high=N - 1, size=(M, n_neg), device=y_pool.device)
    #     pos = pos_idx.view(M, 1)
    #     neg = neg + (neg >= pos).long()
    #     return y_pool[neg]  # (M, n_neg, D)


    # def _cpc_logits(self, x: torch.Tensor, y_pos: torch.Tensor, y_negs: torch.Tensor, tau: float):
    #     """
    #     x:     (M, D) anchor preds
    #     y_pos: (M, D) positives
    #     y_negs:(M, Nneg, D) negatives
    #     return logits: (M, 1+Nneg), labels: (M,) all zeros
    #     """
    #     x = F.normalize(x, dim=-1)
    #     y_pos = F.normalize(y_pos, dim=-1)
    #     y_negs = F.normalize(y_negs, dim=-1)

    #     pos_logit = (x * y_pos).sum(dim=-1, keepdim=True)  # (M,1)
    #     neg_logit = torch.einsum("md,mnd->mn", x, y_negs)   # (M,Nneg)
    #     logits = torch.cat([pos_logit, neg_logit], dim=1) / tau
    #     labels = torch.zeros((logits.size(0),), device=logits.device, dtype=torch.long)
    #     return logits, labels


    # # ------------------------------------------------------------
    # # CPC-style multi-step contrastive loss
    # # ------------------------------------------------------------
    # def _wav2vec2_contrastive_loss(self, x_raw: torch.Tensor, ch_coords: torch.Tensor):
    #     """
    #     CPC-like multi-step prediction.
    #     Input:  x_raw  (B,C,P,W)
    #     Output: loss, mask, q_out
    #     """
    #     assert x_raw.dim() == 4, "Expect x as (B,C,P,W)"
    #     B, C, P, _ = x_raw.shape
    #     device = x_raw.device

    #     # (1) mask (still use your masking strategy)
    #     mask = generate_mask(
    #         B, C, P,
    #         mask_ratio=float(getattr(self.params, "mask_ratio", 0.5)),
    #         device=device,
    #     )  # 0/1

    #     # (2) model outputs
    #     m = self.model.module if isinstance(self.model, DDP) else self.model
    #     out, feats_ctx, q_out = m(x_raw, mask=mask, ch_coords=ch_coords)
    #     feats_q = q_out["x"]  # (B,C,P,D)

    #     D = feats_ctx.size(-1)
    #     K = int(getattr(self.params, "prediction_step", getattr(self.params, "cpc_steps", 3)))
    #     tau = float(getattr(self.params, "logit_temp", getattr(self.params, "contrastive_tau", 0.1)))
    #     n_neg = int(getattr(self.params, "num_negatives", 100))

    #     # --- build predictor lazily: (D -> D*K), like CPC
    #     if (not hasattr(self, "cpc_predictor")) or (self.cpc_predictor.in_features != D) or (self.cpc_predictor.out_features != D * K):
    #         self.cpc_predictor = nn.Linear(D, D * K, bias=False).to(device)

    #     # ============================================================
    #     # choose sequence axis:
    #     #   - prefer time axis: within-channel over P
    #     #   - if P <= 1: fallback to spatial axis over C (sort by ch_coords)
    #     # ============================================================
    #     use_time = (P >= 2)
    #     use_space = (not use_time) and (C >= 2)

    #     # --------
    #     # Case 1: time CPC within each channel (sequence length = P)
    #     # --------
    #     if use_time and P >= (K + 1):
    #         # reshape to (B*C, P, D)
    #         ctx_seq = feats_ctx.permute(0, 1, 2, 3).contiguous().view(B * C, P, D)
    #         q_seq   = feats_q.permute(0, 1, 2, 3).contiguous().view(B * C, P, D)
    #         mask_seq = mask.view(B * C, P).bool()

    #         # global pool for negatives: all (B*C*P) tokens
    #         q_pool = q_seq.reshape(B * C * P, D)

    #         total_loss = 0.0
    #         total_steps = 0

    #         for k in range(1, K + 1):
    #             # anchors at t where t+k exists
    #             # anchor positions: we use masked positions at time t (to avoid trivial leakage)
    #             valid = mask_seq[:, :-k]  # (BC, P-k)

    #             if valid.any():
    #                 # predictor output: (BC, P-k, D*K) -> take k-th block
    #                 pred_all = self.cpc_predictor(ctx_seq[:, :-k, :])  # (BC, P-k, D*K)
    #                 pred_k = pred_all[..., (k - 1) * D : k * D]        # (BC, P-k, D)

    #                 # positives: q at t+k
    #                 pos_k = q_seq[:, k:, :]                           # (BC, P-k, D)

    #                 # flatten valid anchors
    #                 pred_k_flat = pred_k[valid]                       # (M,D)
    #                 pos_k_flat  = pos_k[valid]                        # (M,D)

    #                 # pos indices in global pool
    #                 # index mapping: idx = (bc)*P + (t+k)
    #                 bc_idx, t_idx = valid.nonzero(as_tuple=True)      # each (M,)
    #                 pos_global_idx = bc_idx * P + (t_idx + k)         # (M,)

    #                 # negatives
    #                 negs = self._sample_negatives_global(q_pool, pos_global_idx, n_neg)  # (M,n_neg,D)

    #                 # logits + CE
    #                 logits, labels = self._cpc_logits(pred_k_flat, pos_k_flat, negs, tau)
    #                 total_loss = total_loss + F.cross_entropy(logits, labels)
    #                 total_steps += 1

    #         if total_steps == 0:
    #             # if mask made all steps empty, fallback
    #             use_time = False
    #         else:
    #             loss = total_loss / total_steps
    #             return loss, mask, q_out

    #     # --------
    #     # Case 2: spatial CPC over channels when P==1 (sequence length = C)
    #     #   heuristic: order channels by coords to get a 1D sequence
    #     # --------
    #     if use_space and C >= (K + 1):
    #         # order channels by coordinates (x then y then z)
    #         # ch_coords: (C,3) (assumed)
    #         with torch.no_grad():
    #             coords = ch_coords.detach()
    #             # stable-ish ordering
    #             order = torch.lexsort((coords[:, 2], coords[:, 1], coords[:, 0])) if hasattr(torch, "lexsort") else torch.argsort(coords[:, 0] * 1e6 + coords[:, 1] * 1e3 + coords[:, 2])
    #             order = order.to(device)

    #         # for each p (here P=1), sequence over C: (B, C, D)
    #         ctx_seq = feats_ctx[:, :, 0, :].index_select(1, order)  # (B,C,D)
    #         q_seq   = feats_q  [:, :, 0, :].index_select(1, order)  # (B,C,D)
    #         mask_seq = mask[:, :, 0].index_select(1, order).bool()  # (B,C)

    #         # negatives pool: all (B*C) tokens
    #         q_pool = q_seq.reshape(B * C, D)

    #         total_loss = 0.0
    #         total_steps = 0

    #         for k in range(1, K + 1):
    #             valid = mask_seq[:, :-k]  # (B, C-k)
    #             if valid.any():
    #                 pred_all = self.cpc_predictor(ctx_seq[:, :-k, :])        # (B, C-k, D*K)
    #                 pred_k = pred_all[..., (k - 1) * D : k * D]              # (B, C-k, D)
    #                 pos_k  = q_seq[:, k:, :]                                  # (B, C-k, D)

    #                 pred_k_flat = pred_k[valid]                               # (M,D)
    #                 pos_k_flat  = pos_k[valid]                                # (M,D)

    #                 b_idx, c_idx = valid.nonzero(as_tuple=True)               # (M,)
    #                 pos_global_idx = b_idx * C + (c_idx + k)                  # (M,)

    #                 negs = self._sample_negatives_global(q_pool, pos_global_idx, n_neg)  # (M,n_neg,D)
    #                 logits, labels = self._cpc_logits(pred_k_flat, pos_k_flat, negs, tau)
    #                 total_loss = total_loss + F.cross_entropy(logits, labels)
    #                 total_steps += 1

    #         if total_steps > 0:
    #             loss = total_loss / total_steps
    #             return loss, mask, q_out

    #     # --------
    #     # Case 3: fallback (your original masked in-batch same-position contrastive)
    #     # --------
    #     # (B,C,P,D) -> (B,T,D), T=C*P
    #     ctx = self._flatten_tokens(feats_ctx)  # (B,T,D)
    #     q   = self._flatten_tokens(feats_q)    # (B,T,D)
    #     mask_bool = self._flatten_mask(mask)   # (B,T) bool

    #     Tflat = ctx.size(1)
    #     ctx_flat = ctx.reshape(B * Tflat, D)
    #     q_flat   = q.reshape(B * Tflat, D)
    #     mask_flat = mask_bool.reshape(B * Tflat)

    #     x = F.normalize(ctx_flat[mask_flat], dim=-1)  # (Nmask,D)
    #     y = F.normalize(q_flat[mask_flat], dim=-1)    # (Nmask,D)

    #     logits_xy = (x @ y.t()) / tau
    #     labels = torch.arange(logits_xy.size(0), device=device)
    #     loss_xy = F.cross_entropy(logits_xy, labels)
    #     logits_yx = (y @ x.t()) / tau
    #     loss_yx = F.cross_entropy(logits_yx, labels)

    #     loss = 0.5 * (loss_xy + loss_yx)
    #     return loss, mask, q_out

    # def _compute_contra_loss(self, contra_out, q_out, mask):
    #     assert contra_out.dim() == 4, "Expect as (B,C,P,D)"
    #     B, _, _, D = contra_out.shape
    #     device = contra_out.device
    #     tau = float(getattr(self.params, "logit_temp", getattr(self.params, "contrastive_tau", 0.1)))

    #     feats_q = q_out["x"]  # (B,C,P,D)
    #     ctx = self._flatten_tokens(contra_out)   # (B,T,D)
    #     q   = self._flatten_tokens(feats_q)      # (B,T,D)
    #     mask_bool = self._flatten_mask(mask)     # (B,T) bool

    #     Tflat = ctx.size(1)
    #     ctx_flat = ctx.reshape(B * Tflat, D)
    #     q_flat   = q.reshape(B * Tflat, D)
    #     mask_flat = mask_bool.reshape(B * Tflat)

    #     Nmask = int(mask_flat.sum().item())
    #     if Nmask < 2:
    #         return contra_out.new_tensor(0.0)

    #     x = F.normalize(ctx_flat[mask_flat], dim=-1)  # (Nmask,D)
    #     # y = F.normalize(q_flat[mask_flat], dim=-1)    # (Nmask,D) # 对比量化特征和原始特征
    #     y = F.normalize(ctx_flat[mask_flat], dim=-1)    # (Nmask,D) # 对比原始特征和原始特征

    #     # x = F.normalize(q_flat[mask_flat], dim=-1)  # (Nmask,D)
    #     # y = F.normalize(q_flat[mask_flat], dim=-1)

    #     # # 检查对角相似度 vs 非对角相似度，如果值相近说明无法区分
    #     # if self.rank == 0:
    #     #     sim = (x @ y.t())
    #     #     diag = sim.diag().mean().item()
    #     #     off  = (sim.sum() - sim.diag().sum()) / (Nmask*(Nmask-1))
    #     #     print("diag", diag, "off", off.item())

    #     logits_xy = (x @ y.t()) / tau
    #     labels = torch.arange(Nmask, device=device)
    #     loss_xy = F.cross_entropy(logits_xy, labels)

    #     logits_yx = (y @ x.t()) / tau
    #     loss_yx = F.cross_entropy(logits_yx, labels)

    #     loss = 0.5 * (loss_xy + loss_yx)

    #     # 归一化：除以 log(Nmask)（随机基线尺度）
    #     denom = torch.log(x.new_tensor(float(max(Nmask, 2))))
    #     loss = loss / denom
    #     return loss

    def _compute_contra_loss(self, contra_out, q_out, mask):
        assert contra_out.dim() == 4, "Expect as (B,C,P,D)"
        B, C, P, D = contra_out.shape
        device = contra_out.device
        tau = float(getattr(self.params, "logit_temp", getattr(self.params, "contrastive_tau", 0.1)))

        feats_q = q_out["x"]  # (B,C,P,D)

        ctx_flat = contra_out.reshape(B * P, C * D)
        q_flat   = feats_q.reshape(B * P, C * D)

        # ctx_flat = contra_out.mean(dim=1).reshape(B * P, D)
        # q_flat   = feats_q.mean(dim=1).reshape(B * P, D)

        # x = F.normalize(ctx_flat, dim=-1)
        # y = F.normalize(q_flat, dim=-1)

        x = F.normalize(q_flat, dim=-1)
        y = F.normalize(q_flat, dim=-1)

        # # 检查对角相似度 vs 非对角相似度，如果值相近说明无法区分
        # if self.rank == 0:
        #     sim = (x @ y.t())
        #     diag = sim.diag().mean().item()
        #     off  = (sim.sum() - sim.diag().sum()) / (Nmask*(Nmask-1))
        #     print("diag", diag, "off", off.item())

        logits_xy = (x @ y.t()) / tau
        labels = torch.arange(B * P, device=device)
        loss_xy = F.cross_entropy(logits_xy, labels)

        logits_yx = (y @ x.t()) / tau
        loss_yx = F.cross_entropy(logits_yx, labels)

        loss = 0.5 * (loss_xy + loss_yx)

        # 归一化：除以 log(Nmask)（随机基线尺度）
        denom = torch.log(x.new_tensor(float(max(B * P, 2))))
        loss = loss / denom
        return loss



    # -------------------------
    # Train
    # -------------------------
    def train(self, resume_path: str = ""):
        best_loss = 1e9
        start_epoch = 0

        if resume_path:
            start_epoch, best_loss = self._load_ckpt(resume_path)
            if self.rank == 0:
                print(f"[Resume] Loaded: {resume_path} | start_epoch={start_epoch}, best_loss={best_loss}")

        use_cuda = (self.device.type == "cuda")

        for epoch in range(start_epoch, int(self.params.epochs)):
            if self.batch_sampler is not None and hasattr(self.batch_sampler, "set_epoch"):
                self.batch_sampler.set_epoch(epoch) # make sure every epoch has different batch order

            self.model.train()

            pbar = None
            if self.rank == 0:
                pbar = tqdm(
                    total=len(self.data_loader),
                    desc=f"Epoch {epoch+1}/{int(self.params.epochs)} [{self.params.train_mode}]",
                    dynamic_ncols=True,
                )

            loss_sum = torch.zeros((), device=self.device)
            steps_done = 0

            for step, batch in enumerate(self.data_loader):
                x, ds_id = batch
                ds_id = int(ds_id[0].item()) if torch.is_tensor(ds_id) else int(ds_id)

                # =========================================================
                # Get channel & seq info with ds_id
                # =========================================================
                base_ch_names = self.params.ds_ch_names[ds_id]
                base_seq_len = self.params.ds_seq_len[ds_id]
                base_ch_coords = self.params.ds_ch_coords[ds_id]
                self.params.seq_len = base_seq_len
                if not torch.is_tensor(base_ch_coords):
                    base_ch_coords = torch.tensor(base_ch_coords, dtype=torch.float32)

                # ---- data to device
                if use_cuda:
                    x = x.to(self.device, non_blocking=True)
                    ch_coords = base_ch_coords.to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device)
                    ch_coords = base_ch_coords.to(self.device)

                # =========================================================
                # Optional: Channel selects and shuffle
                # =========================================================
                ch_names = base_ch_names
                if self.params.use_channel_subset:
                    idx = self._sample_channel_index(C=x.size(1), device=x.device)
                    x, ch_coords, ch_names = self._apply_channel_index(x, ch_coords, ch_names, idx)
                self.params.ch_coords = ch_coords
                self.params.ch_names = ch_names
                # =========================================================

                # print params
                if epoch == 0 and step == 0 and self.rank == 0:
                    print(self.params)

                # reset grad
                self.optimizer.zero_grad(set_to_none=True)


                # =========================================================
                # Calculate Loss
                # =========================================================
                bz, ch_num, patch_num, _ = x.shape
                mask = generate_mask(
                    bz, ch_num, patch_num,
                    mask_ratio=self.params.mask_ratio,
                    device=self.device,
                )
                # y:重建输出 feats_ctx:特征输出（masked embed as input）q_out: 量化输出（unmasked embed as input）
                recon_out, contra_out, q_out, patch_embed_unmasked = self.model(x, mask=mask, ch_coords=self.params.ch_coords)
                loss_codebook = torch.zeros((), device=self.device)

                if self.params.train_mode == "recon":
                    recon_loss = self.criterion(recon_out[mask == 1], patch_embed_unmasked[mask == 1])
                    loss = recon_loss
                elif self.params.train_mode == "contrastive":
                    contra_loss = self._compute_contra_loss(contra_out, q_out, mask)
                    if q_out is not None:
                        prob_ppl = q_out["prob_perplexity"]
                        num_vars = q_out["num_vars"]
                        loss_codebook = (num_vars - prob_ppl) / num_vars
                    loss = contra_loss + self.params.lambda_codebook * loss_codebook
                elif self.params.train_mode == "both":
                    recon_loss = self.criterion(recon_out[mask == 1], x[mask == 1])
                    contra_loss = self._compute_contra_loss(contra_out, q_out, mask)
                    if q_out is not None:
                        prob_ppl = q_out["prob_perplexity"]
                        num_vars = q_out["num_vars"]
                        loss_codebook = (num_vars - prob_ppl) / num_vars
                    loss = recon_loss + contra_loss + self.params.lambda_codebook * loss_codebook
                    if self.rank == 0 and step % 1000 == 0:
                        print(f"loss: {loss}, recon_loss: {recon_loss}, contra_loss: {contra_loss}, codebook_loss: {loss_codebook}")

                # # =========================================================
                # # 2) Contrastive training (SimCLR + spatial constraints)
                # #    -> uses GLOBAL batch across all GPUs for contrastive loss
                # # =========================================================
                # else:
                #     # ---- make two views (share channel subset, but other augs independent)
                #     # x1 = self._other_augs(x.clone())
                #     # x2 = self._other_augs(x.clone())

                #     # ---- encode to patch representations
                #     z1 = self._encode_for_contrastive(x1, ch_coords=self.params.ch_coords)
                #     z2 = self._encode_for_contrastive(x2, ch_coords=self.params.ch_coords)

                #     # ensure patch-level (B,C,P,D) for spatial constraints when possible
                #     # If returned (B,T,D), spatial constraints will be skipped automatically.
                #     if z1.dim() == 4:
                #         z1 = F.normalize(z1, dim=-1)
                #     if z2.dim() == 4:
                #         z2 = F.normalize(z2, dim=-1)

                #     # ---- global SimCLR loss (true global-batch, with grad)
                #     g1 = self._pool_global(z1)
                #     g2 = self._pool_global(z2)
                #     loss_main = self._simclr_loss(g1, g2, tau=self.temperature)

                #     loss = loss_main

                #     # ---- spatial soft KL (apply on view1 only; you也可以对两 view 取均值)
                #     if self.use_spatial_kl and (z1.dim() == 4):
                #         loss_sp = self._spatial_kl_loss(z1, ch_coords=self.params.ch_coords, tau=self.temperature)
                #         loss = loss + self.lambda_spatial_kl * loss_sp

                #     # ---- anti-lowpass hinge (apply on view1)
                #     if self.use_anti_lowpass and (z1.dim() == 4):
                #         loss_alp = self._anti_lowpass_loss(z1, ch_coords=self.params.ch_coords)
                #         loss = loss + self.lambda_anti_lowpass * loss_alp

            #     # ---- backward / step
            #     loss.backward()

            #     if self.params.clip_value > 0:
            #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

            #     self.optimizer.step()
            #     self.optimizer_scheduler.step()

            #     loss_sum += loss.detach()
            #     steps_done += 1

            #     # if step % 20 == 0:
            #     #     print('!!!!', loss_sum, steps_done)

            #     if pbar is not None:
            #         pbar.set_postfix(loss=float(loss.detach().item()), lr=float(self.optimizer.param_groups[0]["lr"]))
            #         pbar.update(1)

            # if pbar is not None:
            #     pbar.close()

            # # ---- epoch mean loss（DDP reduce）
            # mean_loss = loss_sum / max(steps_done, 1)
            # if self.is_ddp:
            #     dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            #     mean_loss = mean_loss / self.world_size

            # mean_loss_v = float(mean_loss.item())
            # lr = float(self.optimizer.param_groups[0]["lr"])

            # print(f"[rank {self.rank}] epoch={epoch+1} steps={steps_done} mean_loss={mean_loss_v:.6f} lr={lr:.6g}")

            # # ---- checkpoint
            # last_path = os.path.join(self.params.foundation_dir, "last.pth")
            # self._save_ckpt(last_path, epoch + 1, best_loss)

            # if mean_loss_v < best_loss:
            #     best_loss = mean_loss_v
            #     best_path = os.path.join(
            #         self.params.foundation_dir, f"best_epoch{epoch+1}_loss{mean_loss_v:.6f}.pth"
            #     )
            #     self._save_ckpt(best_path, epoch + 1, best_loss)
            #     if self.rank == 0:
            #         print(f"best model save in {best_path}")

            # ---- before step loop (add this)
            loss_hist = []   # store per-step loss (detached tensor)
            loss.backward()

            if self.params.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

            self.optimizer.step()
            self.optimizer_scheduler.step()

            loss_sum += loss.detach()
            steps_done += 1

            # ---- [ADD] record per-step loss for tail-mean
            loss_hist.append(loss.detach())

            # if step % 20 == 0:
            #     print('!!!!', loss_sum, steps_done)

            if pbar is not None:
                pbar.set_postfix(loss=float(loss.detach().item()), lr=float(self.optimizer.param_groups[0]["lr"]))
                pbar.update(1)

            # =========================
            # after step loop
            # =========================
            if pbar is not None:
                pbar.close()

            # ---- epoch mean loss（DDP reduce）
            mean_loss = loss_sum / max(steps_done, 1)
            if self.is_ddp:
                dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                mean_loss = mean_loss / self.world_size

            mean_loss_v = float(mean_loss.item())
            lr = float(self.optimizer.param_groups[0]["lr"])

            # ---- [ADD] tail mean loss (method 1: latter half of epoch)
            if len(loss_hist) > 0:
                start = len(loss_hist) // 2
                tail_sum = torch.stack(loss_hist[start:]).sum()
                tail_steps = len(loss_hist) - start
                tail_mean = tail_sum / max(tail_steps, 1)
            else:
                tail_mean = mean_loss  # fallback

            if self.is_ddp:
                dist.all_reduce(tail_mean, op=dist.ReduceOp.SUM)
                tail_mean = tail_mean / self.world_size

            tail_mean_v = float(tail_mean.item())

            print(
                f"[rank {self.rank}] epoch={epoch+1} steps={steps_done} "
                f"mean_loss={mean_loss_v:.6f} tail_mean={tail_mean_v:.6f} lr={lr:.6g}"
            )

            # ---- checkpoint
            last_path = os.path.join(self.params.foundation_dir, "last.pth")
            self._save_ckpt(last_path, epoch + 1, best_loss)

            # ---- [CHANGED] use tail_mean_v to select best (best_loss now tracks tail-mean)
            if tail_mean_v < best_loss:
                best_loss = tail_mean_v
                best_path = os.path.join(
                    self.params.foundation_dir, f"best_epoch{epoch+1}_tail{tail_mean_v:.6f}.pth"
                )
                self._save_ckpt(best_path, epoch + 1, best_loss)
                if self.rank == 0:
                    print(f"best model save in {best_path}")
