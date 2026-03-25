import copy
import os
from timeit import default_timer as timer
import datetime

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from tqdm import tqdm

from finetune_evaluator import Evaluator


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        if self.params.downstream_dataset in [
            'FACED', 'FACED26', 'SEED-V', 'PhysioNet-MI',
            'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a'
        ]:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.downstream_dataset in [
            'SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB'
        ]:
            self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().cuda()
        else:
            raise ValueError(f"Unsupported downstream dataset: {self.params.downstream_dataset}")

        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr:  # set different learning rates for different modules
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.params.lr,
                    weight_decay=self.params.weight_decay
                )
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.params.lr,
                    momentum=0.9,
                    weight_decay=self.params.weight_decay
                )

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.params.epochs * self.data_length,
            eta_min=1e-6
        )
        print(self.model)

    def _get_coords(self, x):
        coords = getattr(self.params, "ch_coords", None)
        if coords is not None:
            coords = coords.to(x.device)
        return coords

    def _prepare_log_meta(self):
        dataset_name = self.params.downstream_dataset
        pretrained_ckpt = getattr(self.params, "foundation_dir", "None")
        now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{now_time}] Dataset: {dataset_name}"
        msg_ckpt = f"Pretrained ckpt: {pretrained_ckpt}"
        return msg, msg_ckpt

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None
        best_epoch = 0

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []

            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()

                mean = x.mean(dim=(-1, -2), keepdim=True)          # [B, C, 1, 1]
                std = x.std(dim=(-1, -2), keepdim=True, unbiased=False)
                x = (x - mean) / (std + 1e-6)

                z_score_mean = x.abs().mean()
                print(f"Mean value after z-score: {z_score_mean}")

                coords = self._get_coords(x)
                pred = self.model(x, ch_coords=coords)

                if self.params.downstream_dataset == 'ISRUC':
                    loss = self.criterion(pred.transpose(1, 2), y)
                else:
                    loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.detach().cpu().numpy())

                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        kappa,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)

                if kappa > kappa_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc, kappa, f1
                    ))
                    best_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

                # 打印测试集评估结果（不保存模型权重）
                print("***************************Test results************************")
                acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
                print("Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(acc, kappa, f1))

        if self.best_model_states is None:
            self.best_model_states = copy.deepcopy(self.model.state_dict())
            best_epoch = self.params.epochs

        self.model.load_state_dict(self.best_model_states)

        log_path = os.path.join(self.params.model_dir, self.params.log_file_name)
        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)

            msg, msg_ckpt = self._prepare_log_meta()

            print("***************************Test results************************")
            msg1 = "***************************Test************************"
            msg2 = "***************************Test results************************"
            msg3 = "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(acc, kappa, f1)
            msg4 = str(cm)

            print(msg3)
            print(cm)

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
                f.write(msg_ckpt + "\n")
                # f.write(msg1 + "\n")
                # f.write(msg2 + "\n")
                f.write(msg3 + "\n")
                # f.write(msg4 + "\n\n")

            model_path = self.params.model_dir + "/{}_epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(
                self.params.downstream_dataset, best_epoch, acc, kappa, f1
            )
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        best_epoch = 0

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []

            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()

                coords = self._get_coords(x)
                pred = self.model(x, ch_coords=coords)

                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.detach().cpu().numpy())

                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)

                if roc_auc > roc_auc_best:
                    print("roc_auc increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc, pr_auc, roc_auc
                    ))
                    best_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

                # 打印测试集评估结果（不保存模型权重）
                print("***************************Test results************************")
                acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
                print("Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(acc, pr_auc, roc_auc))

        if self.best_model_states is None:
            self.best_model_states = copy.deepcopy(self.model.state_dict())
            best_epoch = self.params.epochs

        self.model.load_state_dict(self.best_model_states)

        log_path = os.path.join(self.params.model_dir, self.params.log_file_name)
        with torch.no_grad():
            print("***************************Test************************")
            acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)

            msg, msg_ckpt = self._prepare_log_meta()

            print("***************************Test results************************")
            msg1 = "***************************Test************************"
            msg2 = "***************************Test results************************"
            msg3 = "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                acc, pr_auc, roc_auc
            )
            msg4 = str(cm)

            print(msg3)
            print(cm)

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
                f.write(msg_ckpt + "\n")
                # f.write(msg1 + "\n")
                # f.write(msg2 + "\n")
                f.write(msg3 + "\n")
                # f.write(msg4 + "\n\n")

            model_path = self.params.model_dir + "/{}_epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(
                self.params.downstream_dataset, best_epoch, acc, pr_auc, roc_auc
            )
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        best_epoch = 0

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []

            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()

                coords = self._get_coords(x)
                pred = self.model(x, ch_coords=coords)

                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.detach().cpu().numpy())

                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        corrcoef,
                        r2,
                        rmse,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )

                if r2 > r2_best:
                    print("r2 increasing....saving weights !! ")
                    print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                        corrcoef, r2, rmse
                    ))
                    best_epoch = epoch + 1
                    corrcoef_best = corrcoef
                    r2_best = r2
                    rmse_best = rmse
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        if self.best_model_states is None:
            self.best_model_states = copy.deepcopy(self.model.state_dict())
            best_epoch = self.params.epochs

        self.model.load_state_dict(self.best_model_states)

        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef, r2, rmse
                )
            )

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)

            model_path = self.params.model_dir + "/{}_epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(
                self.params.downstream_dataset, best_epoch, corrcoef, r2, rmse
            )
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)