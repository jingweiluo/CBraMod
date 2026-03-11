import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            # self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
            ckpt = torch.load(param.foundation_dir, map_location=map_location)
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=True)
        self.backbone.proj_out = nn.Identity()
        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, param.num_of_classes)
            )
        elif param.classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(64*3*200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(64*3*200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(64*3*200, 3*200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(3*200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )

    def forward(self, x, ch_coords=None):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x, ch_coords=ch_coords)
        out = self.classifier(feats)
        return out

