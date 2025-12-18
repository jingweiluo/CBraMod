import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

# ========= 工程路径 & 数据加载 =========
import os
import sys
current_file_path = os.path.abspath(__file__)
parent_parent_dir = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_parent_dir)
from utils.constants import ROOT_DIR, DATA_DIR_DICT, LMDB_DIR_DICT
from utils.util import random_masking, get_1d_sincos_pos_embed

class CBraMod(nn.Module):
    def __init__(self, chan_num=19, seq_len=30, in_dim=200, out_dim=200, d_model=800, mlp_ratio=4., n_layer=24,
                    nhead=16, decoder_embed_dim=400, decoder_depth=8, decoder_num_heads=16):
        super().__init__()

        # ==================================================================== #
        self.encoder_embed = nn.Linear(in_dim, d_model, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, chan_num*seq_len + 1, d_model), requires_grad=False)
        self.cls_token = nn.Parameter(torch.zeros(d_model))
        self.blocks = nn.ModuleList([
            Block(d_model, nhead, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(n_layer)])
        self.norm = nn.LayerNorm(d_model)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_embed = nn.Linear(d_model, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, chan_num*seq_len + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_dim, bias=True)
        self.initialize_weights()
        # ==================================================================== #

    # ==================================================================== #
    # WeightInit
    # ==================================================================== #
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[2], self.pos_embed.shape[1], device=self.pos_embed.device)
        self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[2], self.decoder_pos_embed.shape[1], device=self.decoder_pos_embed.device)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float().unsqueeze(0))

        # # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)





    # ==================================================================== #
    # Encoder
    # ==================================================================== #
    def forward_encoder(self, x, mask_ratio):
        # 1,2位置可考虑互换

        # 1 添加各种编码
        # x = x + self.chan_pos_embed
        # x = x + self.pach_pos_embed # B,C,P,W

        # 2 embed & flatten
        flat_x = self.patch_embed(x) # ViT中作用是对原始img做Conv2d，得到切分的patch，并flatten，最后transpose，得到B,N,D

        # 3 add seq-level pos-embed
        flat_x = flat_x + self.pos_embed[:, 1:, :]

        # 4 masking: length -> length * mask_ratio
        visible_x, mask, ids_restore = random_masking(flat_x, mask_ratio)

        # 5 append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(visible_x.shape[0], -1, -1)
        visible_x = torch.cat((cls_tokens, visible_x), dim=1) # B, (c*p)*mask_ratio+1, d_model

        # 5 apply Transformer blocks
        for blk in self.blocks:
            visible_x = blk(visible_x)

        visible_x = self.norm(visible_x)
        return visible_x, mask, ids_restore

    def patch_embed(self, x):
        """
        这里也可以改变embed_dim
        """
        b, c, p, w = x.shape
        x = self.encoder_embed(x)
        return x.contiguous().view(b, c*p, -1)

    # ==================================================================== #
    # Decoder
    # ==================================================================== #
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x) # 简单线性层，可考虑去掉

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, x, pred, mask):
        b, c, p, w = x.shape
        x = x.contiguous().view(b, c*p, w)
        loss = (pred - x)**2
        loss = loss.mean(dim=-1) # [b, c*p] per patch loss
        loss = (loss * mask).sum() / mask.sum() # mean loss on removed patches
        return loss

    def forward(self, x, mask_ratio=0):
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore) # B (C P) W
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask






if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CBraMod(
        chan_num=19,
        seq_len=30,
        in_dim=200,
        out_dim=200,
        d_model=200,
        mlp_ratio=4.,
        n_layer=12,
        nhead=8,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16
    ).to(device)
    # model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth',
    #                                  map_location=device))
    a = torch.randn((8, 19, 30, 200)).cuda()
    b = model(a, 0)
    print(a.shape, b[0])
