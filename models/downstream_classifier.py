import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=param.in_dim, out_dim=param.out_dim, d_model=param.d_model,
            dim_feedforward=param.dim_feedforward, seq_len=param.seq_len,
            n_layer=param.n_layer, nhead=param.nhead
        )
        self.ch_num = len(param.ch_names)

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            # self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
            ckpt = torch.load(param.foundation_dir, map_location=map_location)
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=True)

        # self.backbone.proj_out = nn.Identity()

        if param.downstream_dataset in ['SHU-MI', 'Mumtaz2016', 'CHB-MIT', 'MentalArithmetic']:
            if param.classifier == 'avgpooling_patch_reps':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b d c s'),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(param.out_dim, 1),
                    Rearrange('b 1 -> (b 1)'),
                )
            elif param.classifier == 'all_patch_reps_onelayer':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b (c s d)'),
                    nn.Linear(self.ch_num * param.seq_len * param.out_dim, 1),
                    Rearrange('b 1 -> (b 1)'),
                )
            elif param.classifier == 'all_patch_reps_twolayer':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b (c s d)'),
                    nn.Linear(self.ch_num * param.seq_len * param.out_dim, param.out_dim),
                    nn.ELU(),
                    nn.Dropout(param.dropout),
                    nn.Linear(param.out_dim, 1),
                    Rearrange('b 1 -> (b 1)'),
                )
            elif param.classifier == 'all_patch_reps':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b (c s d)'),
                    nn.Linear(self.ch_num * param.seq_len * param.out_dim, param.seq_len * param.out_dim),
                    nn.ELU(),
                    nn.Dropout(param.dropout),
                    nn.Linear(param.seq_len * param.out_dim, param.out_dim),
                    nn.ELU(),
                    nn.Dropout(param.dropout),
                    nn.Linear(param.out_dim, 1),
                    Rearrange('b 1 -> (b 1)'),
                )
        else:
            if param.classifier == 'avgpooling_patch_reps':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b d c s'),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(param.out_dim, param.num_of_classes),
                )
            elif param.classifier == 'all_patch_reps_onelayer':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b (c s d)'),
                    nn.Linear(self.ch_num * param.seq_len * param.out_dim, param.num_of_classes),
                )
            elif param.classifier == 'all_patch_reps_twolayer':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b (c s d)'),
                    nn.Linear(self.ch_num * param.seq_len * param.out_dim, param.out_dim),
                    nn.ELU(),
                    nn.Dropout(param.dropout),
                    nn.Linear(param.out_dim, param.num_of_classes),
                )
            elif param.classifier == 'all_patch_reps':
                self.classifier = nn.Sequential(
                    Rearrange('b c s d -> b (c s d)'),
                    nn.Linear(self.ch_num * param.seq_len * param.out_dim, param.seq_len * param.out_dim),
                    nn.ELU(),
                    nn.Dropout(param.dropout),
                    nn.Linear(param.seq_len * param.out_dim, param.out_dim),
                    nn.ELU(),
                    nn.Dropout(param.dropout),
                    nn.Linear(param.out_dim, param.num_of_classes),
                )

    def forward(self, x, ch_coords=None):
        bz, ch_num, seq_len, patch_size = x.shape
        # feats = self.backbone(x, ch_coords=ch_coords)
        feats,_,_,_ = self.backbone(x, ch_coords=ch_coords)
        out = self.classifier(feats)
        return out
