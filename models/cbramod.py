import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder
# from models.criss_cross_transformer_rope import TransformerEncoderLayer, TransformerEncoder
from models.criss_cross_transformer_cord import TransformerEncoderLayer, TransformerEncoder


from utils.util import build_4d_fourier_pe, get_ch_pos, get_2d_sincos_pe


class CBraMod(nn.Module):
    def __init__(self, in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, norm_first=True,
            activation=F.gelu
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)
        self.proj_out = nn.Sequential(
            # nn.Linear(d_model, d_model*2),
            # nn.GELU(),
            # nn.Linear(d_model*2, d_model),
            # nn.GELU(),
            nn.Linear(d_model, out_dim),
        )
        self.apply(_weights_init)

    def forward(self, x, mask=None, ch_coords=None):
        patch_emb = self.patch_embedding(x, mask)
        # feats = self.encoder(patch_emb, coords=ch_coords)
        feats = self.encoder(patch_emb)
        out = self.proj_out(feats)
        return out

class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model

        # #================================================================
        # # PE A
        # #================================================================
        # self.positional_encoding = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=d_model,
        #         out_channels=d_model,
        #         kernel_size=(19, 7),
        #         stride=(1, 1),
        #         padding=(9, 3),
        #         groups=d_model),
        # )
        # #================================================================

        # #================================================================
        # # PE B
        # #================================================================
        # self.ch_names = ch_names
        # ch_pos = get_ch_pos()
        # xyz_list = []
        # for name in self.ch_names:
        #     if name in ch_pos:
        #         v = ch_pos[name.upper()]
        #         xyz_list.append([float(v[0]), float(v[1]), float(v[2])])
        #     else:
        #         xyz_list.append([0.0, 0.0, 0.0])
        # self.register_buffer("chan_xyz", torch.tensor(xyz_list, dtype=torch.float32), persistent=True)  # (C,3)
        # self.pe_proj = nn.Linear(162, d_model)
        # #================================================================

        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        # self.proj_in = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)), # 长度浓缩，增加不同波段的view
        #     nn.GroupNorm(5, 25),
        #     nn.GELU(),

        #     nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
        #     nn.GroupNorm(5, 25),
        #     nn.GELU(),

        #     nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
        #     nn.GroupNorm(5, 25),
        #     nn.GELU(),
        # )

        self.proj_in2 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(1, 99), stride=(1, 10), padding=(0, 49)),
            nn.GroupNorm(20, 100),
            nn.GELU(),

            # 保持 T 不变：kernel=9 -> padding=4
            nn.Conv2d(100, 100, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4)),
            nn.GroupNorm(20, 100),
            nn.GELU(),

            # 再来一层也保持 T 不变：建议用奇数核更稳
            nn.Conv2d(100, 100, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(20, 100),
            nn.GELU(),
        )
        self.temporal_proj = nn.Sequential(
            nn.Linear(2000, d_model),
            nn.Dropout(0.1),
        )

        self.spectral_proj = nn.Sequential(
            nn.Linear((in_dim // 2) + 1, d_model),
            nn.Dropout(0.1),
            # nn.LayerNorm(d_model, eps=1e-5),
        )
        # self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        # self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        # self.proj_in = nn.Sequential(
        #     nn.Linear(in_dim, d_model, bias=False),
        # )


    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        # print(x)
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        # # Temporal Encoding
        # mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        # patch_emb = self.proj_in(mask_x)
        # patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        # Temporal Encoding2
        patch_emb = self.proj_in2(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, 2000)
        patch_emb = self.temporal_proj(patch_emb)

        # Spectral Encoding
        mask_x = mask_x.contiguous().view(bz*ch_num*patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, (patch_size // 2) + 1)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb

        # # PE A
        # positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        # positional_embedding = positional_embedding.permute(0, 2, 3, 1)
        # patch_emb = patch_emb + positional_embedding

        # # PE B
        # pe_4d = build_4d_fourier_pe(
        #     xyz=self.chan_xyz.to(device=x.device, dtype=x.dtype),  # (C,3)
        #     P=patch_num,
        #     dim=self.d_model,
        # ).to(dtype=x.dtype, device=x.device)  # (1,C,P,D)
        # pe_4d = self.pe_proj(pe_4d)
        # patch_emb = patch_emb + pe_4d

        # # PE C
        # pe_2d = get_2d_sincos_pe(ch_num, patch_num, self.d_model, normalize=False, device=patch_emb.device)
        # patch_emb = patch_emb + pe_2d

        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8).to(device)
    model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth',
                                     map_location=device))
    a = torch.randn((8, 16, 10, 200)).cuda()
    b = model(a)
    print(a.shape, b.shape)
