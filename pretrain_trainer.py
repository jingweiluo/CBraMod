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

        # Augmentation config
        self.min_keep_ch = int(getattr(self.params, "min_keep_ch", 12))
        self.max_keep_ch = int(getattr(self.params, "max_keep_ch", 0))  # 0 -> use C
        self.channel_shuffle = bool(getattr(self.params, "channel_shuffle", True))

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

    def _flatten_tokens(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B,C,P,D) -> (B, T=C*P, D)
        """
        assert z.dim() == 4
        B, C, P, D = z.shape
        return z.contiguous().reshape(B, C * P, D)

    def _flatten_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        mask: (B,C,P) {0,1} -> (B, T=C*P) bool
        """
        assert mask.dim() == 3
        return (mask.reshape(mask.size(0), -1) > 0)

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

        contra_out = contra_out.permute(0, 2, 1, 3).contiguous()
        feats_q    = feats_q.permute(0, 2, 1, 3).contiguous()

        # 缺少 permute(0,2,1,3)
        ctx_flat = contra_out.reshape(B * P, C * D)
        q_flat   = feats_q.reshape(B * P, C * D)

        # ctx_flat = contra_out.mean(dim=1).reshape(B * P, D)
        # q_flat   = feats_q.mean(dim=1).reshape(B * P, D)

        # x = F.normalize(ctx_flat, dim=-1)
        # y = F.normalize(q_flat, dim=-1)

        x = F.normalize(ctx_flat, dim=-1)
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
            loss_hist = []
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

                loss.backward()

                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                self.optimizer.step()
                self.optimizer_scheduler.step()

                loss_sum += loss.detach()
                steps_done += 1
                loss_hist.append(loss.detach())

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
