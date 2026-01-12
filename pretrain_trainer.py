import os
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import MSELoss
from utils.util import generate_mask

class Trainer(object):
    def __init__(self, params, data_loader, model, batch_sampler=None):
        self.params = params
        self.device = torch.device(f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader
        self.batch_sampler = batch_sampler

        self.model = model.to(self.device)
        self.criterion = MSELoss(reduction='mean').to(self.device)

        if self.params.parallel:
            device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.data_length = len(self.data_loader)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay
        )

        if self.params.lr_scheduler == 'CosineAnnealingLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=40 * self.data_length, eta_min=1e-5
            )
        elif self.params.lr_scheduler == 'ExponentialLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999999999)
        elif self.params.lr_scheduler == 'StepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5 * self.data_length, gamma=0.5
            )
        elif self.params.lr_scheduler == 'MultiStepLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[10 * self.data_length, 20 * self.data_length, 30 * self.data_length],
                gamma=0.1
            )
        elif self.params.lr_scheduler == 'CyclicLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=1e-6, max_lr=0.001,
                step_size_up=self.data_length * 5, step_size_down=self.data_length * 2,
                mode='exp_range', gamma=0.9, cycle_momentum=False
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.params.lr_scheduler}")

    def _save_ckpt(self, path, epoch, best_loss):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_state = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()

        ckpt = {
            "epoch": epoch,                  # 下一个要训练的 epoch index（0-based）
            "best_loss": float(best_loss),
            "model": model_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.optimizer_scheduler.state_dict(),
            # 记录配置，便于排查
            "dataset_list": getattr(self.params, "dataset_list", None),
            "batch_size": self.params.batch_size,
            "lr": self.params.lr,
            "weight_decay": self.params.weight_decay,
            "lr_scheduler": self.params.lr_scheduler,
            "mask_ratio": self.params.mask_ratio,
        }
        torch.save(ckpt, path)

    def _load_ckpt(self, path):
        ckpt = torch.load(path, map_location=self.device)

        # model
        model_state = ckpt.get("model", ckpt)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(model_state, strict=True)
        else:
            self.model.load_state_dict(model_state, strict=True)

        # optimizer/scheduler（若存在）
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.optimizer_scheduler.load_state_dict(ckpt["scheduler"])

        start_epoch = int(ckpt.get("epoch", 0))
        best_loss = float(ckpt.get("best_loss", 1e9))
        return start_epoch, best_loss

    def train(self, resume_path: str = ""):
        best_loss = 10000.0
        start_epoch = 0

        if resume_path:
            start_epoch, best_loss = self._load_ckpt(resume_path)
            print(f"[Resume] Loaded checkpoint: {resume_path}")
            print(f"[Resume] start_epoch={start_epoch}, best_loss={best_loss}")

        for epoch in range(start_epoch, self.params.epochs):
            if self.batch_sampler is not None and hasattr(self.batch_sampler, "set_epoch"):
                self.batch_sampler.set_epoch(epoch)

            losses = []
            for batch in tqdm(self.data_loader, mininterval=10):
                self.optimizer.zero_grad()

                x, ds_id = batch
                ds_id = int(ds_id[0].item()) if torch.is_tensor(ds_id) else int(ds_id)

                # 切换元信息（若模型 forward 用得到 coords）
                self.params.ch_names = self.params.ds_ch_names[ds_id]
                self.params.seq_len = self.params.ds_seq_len[ds_id]
                self.params.ch_coords = self.params.ds_ch_coords[ds_id].to(self.device)

                # x = x.to(self.device) / 100.0
                x = x.to(self.device)

                if self.params.need_mask:
                    bz, ch_num, patch_num, patch_size = x.shape
                    mask = generate_mask(bz, ch_num, patch_num, mask_ratio=self.params.mask_ratio, device=self.device)
                    y = self.model(x, mask=mask, ch_coords=self.params.ch_coords)
                    loss = self.criterion(y[mask == 1], x[mask == 1])
                else:
                    y = self.model(x, ch_coords=self.params.ch_coords)
                    loss = self.criterion(y, x)

                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                self.optimizer.step()
                self.optimizer_scheduler.step()
                losses.append(loss.detach().cpu().item())

            mean_loss = float(np.mean(losses))
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(f'Epoch {epoch+1}: Training Loss: {mean_loss:.6f}, Learning Rate: {lr:.6f}')

            # 1) 保存 last（便于 resume）
            last_path = os.path.join(self.params.foundation_dir, "last.pth")
            self._save_ckpt(last_path, epoch + 1, best_loss)

            # 2) 保存 best
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_path = os.path.join(self.params.foundation_dir, f"best_epoch{epoch+1}_loss{mean_loss:.6f}.pth")
                self._save_ckpt(best_path, epoch + 1, best_loss)
                print("best model save in " + best_path)
