import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .cbramod import CBraMod

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=10,
            n_layer=12, nhead=8
        )

        if param.use_pretrained_weights:
            # =========================================================================================
            # Original version
            # =========================================================================================
            # map_location = torch.device(f'cuda:{param.cuda}')
            # self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
            # =========================================================================================

            # =========================================================================================
            # Revised version
            # =========================================================================================
            map_location = torch.device(f'cuda:{param.cuda}')
            ckpt = torch.load(param.foundation_dir, map_location=map_location)
            # 兼容不同保存格式：有的 ckpt 是 {"state_dict": ...} 或 {"model": ...}
            state_dict = ckpt
            for k in ["state_dict", "model", "net", "module"]:
                if isinstance(state_dict, dict) and k in state_dict and isinstance(state_dict[k], dict):
                    state_dict = state_dict[k]
                    break
            # 1) 忽略 fixed / 非训练参数：chan_xyz（通道坐标，通道数不同会 mismatch）
            state_dict.pop("patch_embedding.chan_xyz", None)
            # 如果你某些 ckpt 是 DataParallel/DDP 保存的，可能带 "module." 前缀，也一并处理
            state_dict.pop("module.patch_embedding.chan_xyz", None)
            # 2) 宽松加载（其他不匹配的 key 也不会报错，但仍会打印 missing/unexpected）
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"[load_state_dict] missing keys: {len(missing)}")
            print(f"[load_state_dict] unexpected keys: {len(unexpected)}")
            # =========================================================================================
        self.backbone.proj_out = nn.Identity()

        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(26 * 10 * 200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(26 * 10 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(26 * 10 * 200, 10 * 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(10 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )

    def forward(self, x, ch_coords=None):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x, ch_coords=ch_coords)
        out = self.classifier(feats)
        return out



