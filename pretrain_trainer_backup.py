import os
import random
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.util import generate_mask


# -------------------------
# DDP helpers (kept for compatibility)
# -------------------------
def _is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


class _GatherLayer(torch.autograd.Function):
    """
    All-gather with backward support.
    Kept only for compatibility. Not used in the current positive-only q-context alignment loss.
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
        if not _is_ddp():
            return grads[0]
        rank = dist.get_rank()
        grad_local = grads[rank].contiguous()
        dist.all_reduce(grad_local, op=dist.ReduceOp.SUM)
        return grad_local


def _ddp_all_gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """
    Gather tensor from all ranks WITH autograd. Return concatenated along dim=0.
    x: (B, ...)
    """
    if not _is_ddp():
        return x
    xs = _GatherLayer.apply(x)
    return torch.cat(list(xs), dim=0)


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
        # Channel subset config
        # -------------------------
        self.min_keep_ch = int(getattr(self.params, "min_keep_ch", 12))
        self.max_keep_ch = int(getattr(self.params, "max_keep_ch", 0))  # 0 -> use C
        self.channel_shuffle = bool(getattr(self.params, "channel_shuffle", True))

        # -------------------------
        # recon + q-context alignment weighting
        # -------------------------
        self.lambda_contra = float(getattr(self.params, "lambda_contra", 0.05))
        self.contra_warmup_epochs = int(getattr(self.params, "contra_warmup_epochs", 0))

        # -------------------------
        # codebook usage regularization
        # -------------------------
        self.lambda_usage = float(getattr(self.params, "lambda_usage", 0.01))
        self.usage_alpha = float(getattr(self.params, "usage_alpha", 0.5))
        # usage_alpha:
        #   1.0 -> only prob_perplexity
        #   0.0 -> only code_perplexity
        #   0.5 -> balanced

        # -------------------------
        # Q-context positive alignment config
        # -------------------------
        self.consistency_on = str(getattr(self.params, "consistency_on", "masked_token"))
        self.consistency_stopgrad_branch = str(
            getattr(self.params, "consistency_stopgrad_branch", "quantized")
        )

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
            "mask_ratio": self.params.mask_ratio,
            "lambda_contra": self.lambda_contra,
            "contra_warmup_epochs": self.contra_warmup_epochs,
            "lambda_usage": self.lambda_usage,
            "usage_alpha": self.usage_alpha,
            "consistency_on": self.consistency_on,
            "consistency_stopgrad_branch": self.consistency_stopgrad_branch,
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

    def _get_lambda_contra(self, epoch: int) -> float:
        if self.contra_warmup_epochs <= 0:
            return self.lambda_contra
        warm_ratio = min(1.0, float(epoch + 1) / float(self.contra_warmup_epochs))
        return self.lambda_contra * warm_ratio

    # -------------------------
    # q_out helper
    # -------------------------
    def _extract_quant_feat(self, q_out) -> torch.Tensor:
        if torch.is_tensor(q_out):
            return q_out

        if not isinstance(q_out, dict):
            raise TypeError(
                f"q_out must be a Tensor or dict, but got type={type(q_out)}"
            )

        if "x" not in q_out:
            raise RuntimeError(
                f"q_out is dict but key 'x' is missing. Available keys: {list(q_out.keys())}"
            )

        quant_feat = q_out["x"]
        if not torch.is_tensor(quant_feat):
            raise TypeError(
                f"q_out['x'] must be a tensor, but got type={type(quant_feat)}"
            )

        return quant_feat

    def _compute_usage_loss(self, q_out) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Combined usage regularization using both prob_perplexity and code_perplexity.

        usage_prob = 1 - prob_perplexity / max_perplexity
        usage_code = 1 - code_perplexity / max_perplexity
        usage_loss = alpha * usage_prob + (1 - alpha) * usage_code

        Range (approx):
            usage_prob in [0, 1]
            usage_code in [0, 1]
            usage_loss in [0, 1]
        """
        stats = {
            "prob_perplexity": 0.0,
            "code_perplexity": 0.0,
            "prob_usage_ratio": 0.0,
            "code_usage_ratio": 0.0,
            "usage_ratio": 0.0,
        }

        if not isinstance(q_out, dict):
            return torch.zeros((), device=self.device), stats

        prob_perplexity = q_out.get("prob_perplexity", None)
        code_perplexity = q_out.get("code_perplexity", None)
        max_perplexity = q_out.get("num_vars", None)

        if prob_perplexity is None or (not torch.is_tensor(prob_perplexity)):
            return torch.zeros((), device=self.device), stats

        if code_perplexity is None or (not torch.is_tensor(code_perplexity)):
            code_perplexity = prob_perplexity.detach()

        if max_perplexity is None:
            max_perplexity = torch.max(
                prob_perplexity.detach(),
                code_perplexity.detach()
            ).clamp_min(1.0)

        if not torch.is_tensor(max_perplexity):
            max_perplexity = torch.tensor(
                float(max_perplexity),
                device=prob_perplexity.device,
                dtype=prob_perplexity.dtype,
            )
        else:
            max_perplexity = max_perplexity.to(
                device=prob_perplexity.device,
                dtype=prob_perplexity.dtype,
            )

        max_perplexity = max_perplexity.clamp_min(1.0)

        prob_usage_ratio = (prob_perplexity / max_perplexity).clamp(min=0.0, max=1.0)
        code_usage_ratio = (code_perplexity / max_perplexity).clamp(min=0.0, max=1.0)

        usage_prob_loss = 1.0 - prob_usage_ratio
        usage_code_loss = 1.0 - code_usage_ratio

        alpha = max(0.0, min(1.0, float(self.usage_alpha)))
        usage_loss = alpha * usage_prob_loss + (1.0 - alpha) * usage_code_loss

        combined_usage_ratio = alpha * prob_usage_ratio + (1.0 - alpha) * code_usage_ratio

        stats["prob_perplexity"] = float(prob_perplexity.detach().item())
        stats["code_perplexity"] = float(code_perplexity.detach().item())
        stats["prob_usage_ratio"] = float(prob_usage_ratio.detach().item())
        stats["code_usage_ratio"] = float(code_usage_ratio.detach().item())
        stats["usage_ratio"] = float(combined_usage_ratio.detach().item())

        return usage_loss, stats

    # -------------------------
    # Positive-only q-context alignment helpers
    # -------------------------
    def _masked_mean_pool(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert feat.dim() == 4
        assert mask.dim() == 3

        m = mask.unsqueeze(-1).float()  # (B,C,P,1)
        denom = m.sum(dim=(1, 2)).clamp_min(1.0)
        pooled = (feat * m).sum(dim=(1, 2)) / denom
        return pooled

    def _all_mean_pool(self, feat: torch.Tensor) -> torch.Tensor:
        return feat.mean(dim=(1, 2))

    def _compute_q_context_alignment_loss(
        self,
        context_feat: torch.Tensor,
        quant_feat: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert torch.is_tensor(context_feat) and torch.is_tensor(quant_feat), (
            f"context_feat and quant_feat must both be tensors, "
            f"got {type(context_feat)} and {type(quant_feat)}"
        )
        assert context_feat.dim() == 4 and quant_feat.dim() == 4, (
            f"Expect features as (B,C,P,D), got {tuple(context_feat.shape)} and {tuple(quant_feat.shape)}"
        )
        assert context_feat.shape == quant_feat.shape, (
            f"context_feat and quant_feat must have the same shape, "
            f"got {tuple(context_feat.shape)} vs {tuple(quant_feat.shape)}"
        )
        assert mask.dim() == 3, f"Expect mask as (B,C,P), got {tuple(mask.shape)}"

        stats = {
            "avg_cos": 0.0,
            "num_pairs": 0.0,
        }

        if self.consistency_stopgrad_branch == "context":
            context_feat = context_feat.detach()
        elif self.consistency_stopgrad_branch == "quantized":
            quant_feat = quant_feat.detach()
        elif self.consistency_stopgrad_branch == "none":
            pass
        else:
            raise ValueError(
                f"Unknown consistency_stopgrad_branch: {self.consistency_stopgrad_branch}"
            )

        if self.consistency_on == "masked_mean":
            z_ctx = self._masked_mean_pool(context_feat, mask)
            z_q = self._masked_mean_pool(quant_feat, mask)

            z_ctx = F.normalize(z_ctx, dim=-1)
            z_q = F.normalize(z_q, dim=-1)

            cos = (z_ctx * z_q).sum(dim=-1)
            loss = (1.0 - cos).mean()

            stats["avg_cos"] = float(cos.detach().mean().item())
            stats["num_pairs"] = float(z_ctx.size(0))
            return loss, stats

        elif self.consistency_on == "all_mean":
            z_ctx = self._all_mean_pool(context_feat)
            z_q = self._all_mean_pool(quant_feat)

            z_ctx = F.normalize(z_ctx, dim=-1)
            z_q = F.normalize(z_q, dim=-1)

            cos = (z_ctx * z_q).sum(dim=-1)
            loss = (1.0 - cos).mean()

            stats["avg_cos"] = float(cos.detach().mean().item())
            stats["num_pairs"] = float(z_ctx.size(0))
            return loss, stats

        elif self.consistency_on == "masked_token":
            valid = (mask > 0)
            if not valid.any():
                return context_feat.new_zeros(()), stats

            z_ctx = context_feat[valid]
            z_q = quant_feat[valid]

            z_ctx = F.normalize(z_ctx, dim=-1)
            z_q = F.normalize(z_q, dim=-1)

            cos = (z_ctx * z_q).sum(dim=-1)
            loss = (1.0 - cos).mean()

            stats["avg_cos"] = float(cos.detach().mean().item())
            stats["num_pairs"] = float(z_ctx.size(0))
            return loss, stats

        else:
            raise ValueError(f"Unknown consistency_on: {self.consistency_on}")

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
                self.batch_sampler.set_epoch(epoch)

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

            tail_loss_sum = torch.zeros((), device=self.device)
            tail_steps = 0
            tail_start_step = len(self.data_loader) // 2

            effective_lambda_contra = self._get_lambda_contra(epoch)

            for step, batch in enumerate(self.data_loader):
                x, ds_id = batch
                ds_id = int(ds_id[0].item()) if torch.is_tensor(ds_id) else int(ds_id)
                dataset_name = self.params.dataset_list[ds_id]

                mean = x.mean(dim=(-1, -2), keepdim=True)          # [B, C, 1, 1]
                std = x.std(dim=(-1, -2), keepdim=True, unbiased=False)
                x = (x - mean) / (std + 1e-6)

                if step % 100 == 0:
                    z_score_mean = x.abs().mean()
                    print(f"Mean value after z-score of {dataset_name}: {z_score_mean}")
                    # print(f"{x}")


                try:
                    base_ch_names = self.params.ds_ch_names[ds_id]
                    base_seq_len = self.params.ds_seq_len[ds_id]
                    base_ch_coords = self.params.ds_ch_coords[ds_id]
                    self.params.seq_len = base_seq_len
                    if not torch.is_tensor(base_ch_coords):
                        base_ch_coords = torch.tensor(base_ch_coords, dtype=torch.float32)

                    if use_cuda:
                        x = x.to(self.device, non_blocking=True)
                        ch_coords = base_ch_coords.to(self.device, non_blocking=True)
                    else:
                        x = x.to(self.device)
                        ch_coords = base_ch_coords.to(self.device)

                    ch_names = base_ch_names
                    if self.params.use_channel_subset:
                        idx = self._sample_channel_index(C=x.size(1), device=x.device)
                        x, ch_coords, ch_names = self._apply_channel_index(x, ch_coords, ch_names, idx)

                    self.params.ch_coords = ch_coords
                    self.params.ch_names = ch_names

                    if epoch == 0 and step == 0 and self.rank == 0:
                        print(self.params)

                    self.optimizer.zero_grad(set_to_none=True)

                    bz, ch_num, patch_num, _ = x.shape

                    mask_main = generate_mask(
                        bz, ch_num, patch_num,
                        mask_ratio=self.params.mask_ratio,
                        device=self.device,
                    )

                    recon_out, contra_out_main, q_out, patch_embed_unmasked = self.model(
                        x, mask=mask_main, ch_coords=self.params.ch_coords
                    )

                    if q_out is None:
                        raise RuntimeError(
                            "q_out is None. Current trainer expects model(...) to return "
                            "(recon_out, context_feat, q_out, patch_embed_unmasked)."
                        )

                    quant_feat = self._extract_quant_feat(q_out)

                    if self.rank == 0 and epoch == 0 and step == 0:
                        if isinstance(q_out, dict):
                            print("q_out keys:", list(q_out.keys()))
                            for k, v in q_out.items():
                                if torch.is_tensor(v):
                                    print(f"q_out['{k}'] shape = {tuple(v.shape)}")
                                else:
                                    print(f"q_out['{k}'] type = {type(v)}")
                        print(f"context_feat shape = {tuple(contra_out_main.shape)}")
                        print(f"quant_feat shape   = {tuple(quant_feat.shape)}")

                    recon_loss = self.criterion(
                        recon_out[mask_main == 1],
                        x[mask_main == 1]
                    )

                    contra_loss, contra_stats = self._compute_q_context_alignment_loss(
                        context_feat=contra_out_main,
                        quant_feat=quant_feat,
                        mask=mask_main,
                    )

                    usage_loss, usage_stats = self._compute_usage_loss(q_out)

                    if self.params.train_mode == "recon":
                        loss = recon_loss

                    elif self.params.train_mode == "contrastive":
                        loss = contra_loss + self.lambda_usage * usage_loss

                    elif self.params.train_mode == "both":
                        loss = (
                            recon_loss
                            + effective_lambda_contra * contra_loss
                            + self.lambda_usage * usage_loss
                        )

                        if self.rank == 0 and step % 1000 == 0:
                            print(
                                f"loss: {loss.item():.6f}, "
                                f"recon_loss: {recon_loss.item():.6f}, "
                                f"contra_loss: {contra_loss.item():.6f}, "
                                f"usage_loss: {usage_loss.item():.6f}, "
                                f"lambda_contra: {effective_lambda_contra:.6f}, "
                                f"lambda_usage: {self.lambda_usage:.6f}, "
                                f"usage_alpha: {self.usage_alpha:.3f}, "
                                f"avg_cos: {contra_stats['avg_cos']:.4f}, "
                                f"num_pairs: {contra_stats['num_pairs']:.0f}, "
                                f"prob_ppl: {usage_stats['prob_perplexity']:.4f}, "
                                f"code_ppl: {usage_stats['code_perplexity']:.4f}, "
                                f"prob_ratio: {usage_stats['prob_usage_ratio']:.4f}, "
                                f"code_ratio: {usage_stats['code_usage_ratio']:.4f}, "
                                f"usage_ratio: {usage_stats['usage_ratio']:.4f}"
                            )
                    else:
                        raise ValueError(f"Unknown train_mode: {self.params.train_mode}")

                    loss.backward()

                    if self.params.clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                    self.optimizer.step()
                    self.optimizer_scheduler.step()

                    loss_sum += loss.detach()
                    steps_done += 1

                    if step >= tail_start_step:
                        tail_loss_sum += loss.detach()
                        tail_steps += 1

                    if pbar is not None:
                        postfix = {
                            "ds": dataset_name,
                            "loss": float(loss.detach().item()),
                            "lr": float(self.optimizer.param_groups[0]["lr"]),
                        }
                        if self.params.train_mode == "both":
                            postfix["lam_c"] = float(effective_lambda_contra)
                            postfix["lam_u"] = float(self.lambda_usage)
                            postfix["recon"] = float(recon_loss.detach().item())
                            postfix["contra"] = float(contra_loss.detach().item())
                            postfix["usage"] = float(usage_loss.detach().item())
                            postfix["avg_cos"] = float(contra_stats["avg_cos"])
                            postfix["prob_ppl"] = float(usage_stats["prob_perplexity"])
                            postfix["code_ppl"] = float(usage_stats["code_perplexity"])
                            postfix["use_r"] = float(usage_stats["usage_ratio"])
                        elif self.params.train_mode == "recon":
                            postfix["recon"] = float(recon_loss.detach().item())
                        elif self.params.train_mode == "contrastive":
                            postfix["contra"] = float(contra_loss.detach().item())
                            postfix["usage"] = float(usage_loss.detach().item())
                            postfix["avg_cos"] = float(contra_stats["avg_cos"])
                            postfix["prob_ppl"] = float(usage_stats["prob_perplexity"])
                            postfix["code_ppl"] = float(usage_stats["code_perplexity"])
                            postfix["use_r"] = float(usage_stats["usage_ratio"])

                        pbar.set_postfix(postfix)
                        pbar.update(1)

                except torch.OutOfMemoryError as e:
                    if self.rank == 0:
                        print("\n" + "=" * 80)
                        print(f"[OOM] epoch={epoch+1}, step={step+1}, dataset={dataset_name}, ds_id={ds_id}")
                        print(f"[OOM] x.shape={tuple(x.shape)}")
                        print(f"[OOM] seq_len={base_seq_len}, n_channels={len(base_ch_names)}")
                        if self.params.use_channel_subset:
                            print(f"[OOM] after subset channels={x.size(1)}")
                        print(f"[OOM] error: {str(e)}")
                        print("=" * 80 + "\n")

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise

            if pbar is not None:
                pbar.close()

            global_loss_sum = loss_sum.clone()
            global_steps = torch.tensor(float(steps_done), device=self.device)

            if self.is_ddp:
                dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(global_steps, op=dist.ReduceOp.SUM)

            mean_loss = global_loss_sum / global_steps.clamp_min(1.0)
            mean_loss_v = float(mean_loss.item())
            lr = float(self.optimizer.param_groups[0]["lr"])

            global_tail_sum = tail_loss_sum.clone()
            global_tail_steps = torch.tensor(float(tail_steps), device=self.device)

            if self.is_ddp:
                dist.all_reduce(global_tail_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(global_tail_steps, op=dist.ReduceOp.SUM)

            if global_tail_steps.item() > 0:
                tail_mean = global_tail_sum / global_tail_steps.clamp_min(1.0)
            else:
                tail_mean = mean_loss

            tail_mean_v = float(tail_mean.item())

            print(
                f"[rank {self.rank}] epoch={epoch+1} steps={steps_done} "
                f"mean_loss={mean_loss_v:.6f} tail_mean={tail_mean_v:.6f} "
                f"lr={lr:.6g} lambda_contra={effective_lambda_contra:.6f} "
                f"lambda_usage={self.lambda_usage:.6f} usage_alpha={self.usage_alpha:.3f}"
            )

            last_path = os.path.join(self.params.foundation_dir, "last.pth")
            self._save_ckpt(last_path, epoch + 1, best_loss)

            # save ckpt every ten epochs
            if epoch % max(self.params.epochs // 5000, 10) == 0:
                regular_model_path = os.path.join(
                    self.params.foundation_dir,
                    f'regular_epoch{epoch + 1}_tail{tail_mean_v:.6f}.pth'
                )
                self._save_ckpt(regular_model_path, epoch + 1, best_loss)

            if tail_mean_v < best_loss:
                best_loss = tail_mean_v
                best_path = os.path.join(
                    self.params.foundation_dir, f"best_epoch{epoch+1}_tail{tail_mean_v:.6f}.pth"
                )
                self._save_ckpt(best_path, epoch + 1, best_loss)
                if self.rank == 0:
                    print(f"best model save in {best_path}")