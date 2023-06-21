import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torchsummary import summary
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from mmTMTion.utils import get_root_logger
from ...builder import BACKBONES

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange, repeat
import ipdb
from .TMT import Transformer
from .xclip import Xclip_visual
from .vit import ViT
from .TMT import Flow_TMT
from .vision_transformer import vit_tiny, vit_small, vit_base
import torchvision
import os
import matplotlib
import cv2
import time

matplotlib.use('Agg')
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from fvcore.nn import FlopCountAnalysis

from torchvision import transforms

company_colors = [
    (0, 160, 215),  # blue
    (220, 55, 60),  # red
    (245, 180, 0),  # yellow
    (10, 120, 190),  # navy
    (40, 150, 100),  # green
    (135, 75, 145),  # purple
    (255, 215, 0),  # gold
    (34, 139, 34),  # forestgreen
    (205, 170, 125),  # burlywood
    (255, 127, 36),  # chocolate
    (255, 110, 180),  # hotpink
    (72, 118, 255),  # royalblue
]
company_colors = [(float(c[0]) / 255.0, float(c[1]) / 255.0, float(c[2]) / 255.0) for c in company_colors]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 512
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5  # ?

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # embedding matrix

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, return_weights=False):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        temp_mask = torch.tril(torch.ones((q.shape[2], q.shape[2])), diagonal=0).bool().cuda()
        dots = dots.masked_fill_(~temp_mask, float('-inf'))
        if return_weights:
            return dots
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention_fusion(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    def get_last_selfattention(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = layer[0](x) + x
                x = layer[1](x) + x
            else:
                # return attention of the last block
                att_mat = layer[0].fn(x, return_weights=True)
                return att_mat


@BACKBONES.register_module()
class HSencoder(nn.Module):

    def __init__(self,
                 frame_size=96,
                 stride=64,
                 dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_dim=1024,
                 dropout=0.1,
                 emb_dropout=0.1,
                 drop_path=0.1,
                 TMT_ckpt=None,
                 vit_ckpt=None,
                 s_dim=768,
                 fusion_layer=2,
                 fusion_head=12,
                 Apool='mean',
                 Fpool='cls',
                 temp_aid=False,
                 freeze_spa=False,
                 skip_mo=4,
                 **kwargs
                 ):
        super().__init__()
        self.stride = stride
        self.freeze_spa = freeze_spa
        self.dim = dim
        self.TMT_ckpt = TMT_ckpt
        self.vit_ckpt = vit_ckpt
        self.Fpool = Fpool
        self.frame_size = frame_size
        self.temp_aid = temp_aid
        fusion_dim = dim

        self.temporal_enc = Flow_TMT(
            frame_size=frame_size,
            stride=stride,
            dim=dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            pool=Apool,
            skip_mo=skip_mo
        )

        self.spatial_enc = vit_base(
            patch_size=16,
            drop_rate=dropout,
            attn_drop_rate=emb_dropout,
            drop_path_rate=drop_path
        )

        if freeze_spa:
            for param in self.spatial_enc.parameters():
                param.requires_grad = False

        self.pred_key = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, stride),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.kf_loss = nn.CrossEntropyLoss()
        self.a_to_v = nn.Sequential(
            nn.Linear(s_dim, dim),
        )
        self.fusion_cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.fu_pre = nn.LayerNorm(dim)
        self.fusion = Attention_fusion(
            dim=dim,
            depth=fusion_layer,
            heads=fusion_head,
            dim_head=fusion_dim // fusion_head,
            mlp_dim=1024,
            dropout=dropout
        )
        self.fu_post = nn.LayerNorm(dim)
        self.spa_normalize = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        if True:
            self.batch_counter = 0
            log_dir = 'OUTPUT/K400/vis_show'
            self.tb_logger = SummaryWriter(log_dir)
        count_parameters = False
        if count_parameters:
            print("VIT")
            summary(self.spatial_enc, (3, 224, 224))
            print("TMT")
            summary(self.temporal_enc, [(stride, 3, frame_size, frame_size), (stride, 3, frame_size, frame_size)])
            print("AF")
            summary(self.fusion, (stride + 4, dim))
            image = torch.randn(1, 3, 224, 224).cuda()
            seq = torch.randn(1, stride, 3, frame_size, frame_size).cuda()
            embedding = torch.randn(1, 69, dim).cuda()
            print('Spatial', FlopCountAnalysis(self.spatial_enc, image).total())
            print('Temporal', FlopCountAnalysis(self.temporal_enc, (seq, seq)).total())
            print('Fusion', FlopCountAnalysis(self.fusion, embedding).total())
            ipdb.set_trace()

    def replace_keys(self, old_dict):
        list_oldkey = old_dict.keys()
        new_dict = {}
        for key in list_oldkey:
            newkey = key.replace('encoder.', '', 1)
            new_dict[newkey] = old_dict[key]

        return new_dict

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if self.TMT_ckpt is not None:
            TMT_checkpoint = torch.load(self.TMT_ckpt)
            xclip_proj_weight = TMT_checkpoint['model']['visual.conv1.weight']
            xclip_proj_weight = torch.cat((xclip_proj_weight, xclip_proj_weight[:, -2:, :, :]), dim=1)
            self.temporal_enc.proj.weight = nn.Parameter(xclip_proj_weight)

        if self.vit_ckpt is not None:
            print("Vit Init: ", self.spatial_enc.blocks[0].mlp.fc1.weight[0][0])
            Vit_checkpoint = torch.load(self.vit_ckpt)
            self.spatial_enc.load_state_dict(Vit_checkpoint['state_dict'], strict=True)

            print("Vit Loaded: ", self.spatial_enc.blocks[0].mlp.fc1.weight[0][0])

    def forward(self, x, bool_masked_pos=None, is_img=False):
        """Forward function."""
        if self.freeze_spa:
            self.spatial_enc.eval()
        batch, channels, _, H, W = x.shape
        stride = self.stride
        frames = x  # [B, C, S, H, W]
        frames = frames.to(torch.float32) / 255
        vis_frames = frames.clone()
        frames = rearrange(frames, 'b c s h w -> (b s) c h w')

        gray_frame = frames.clone()
        frames = self.spa_normalize(frames)
        frames = rearrange(frames, '(b s) c h w -> b s c h w', s=stride)

        seq = rearrange(frames, 'b s c h w -> (b s) c h w')
        seq = torchvision.transforms.functional.resize(seq.float(), [self.frame_size, self.frame_size])
        gray_frame = torchvision.transforms.functional.resize(gray_frame.float(),
                                                              [self.frame_size, self.frame_size])
        seq = rearrange(seq, '(b s) c h w -> b s c h w', b=batch)
        gray_frame = rearrange(gray_frame, '(b s) c h w -> b s c h w', b=batch)
        t_emb, frame_embeddings, flow_intensity, att_weight = self.temporal_enc(seq, gray_frame)
        kf_logit = F.softmax(self.pred_key(t_emb.clone().detach()), dim=1)
        _, key_idx = torch.max(kf_logit, dim=1)
        _, att_idx = torch.max(torch.mean(att_weight, 1, True), 2)

        idx_gt = torch.zeros(batch, stride).cuda()
        for i in range(batch):
            idx_gt[i][att_idx[i] * 4] = 1
        kloss = self.kf_loss(kf_logit, idx_gt)

        if False:
            vis_frames = rearrange(vis_frames, 'b c s h w -> b s c h w', b=batch)
            key_frame = vis_frames[0, att_idx[0] * 4].clone()
            mid_frame = vis_frames[0, 32].clone()
            mid_frame = torch.cat((key_frame, mid_frame), dim=2)
            self.tb_logger.add_image('keyframe', mid_frame, self.batch_counter)
            self.batch_counter += 1

        frame = []
        for i in range(batch):
            frame.append(frames[i, 0].clone())
            frame.append(frames[i, key_idx[i]].clone())
            frame.append(frames[i, -1].clone())
        image = torch.stack(frame)

        s_emb, patch_embeddings, att_mat = self.spatial_enc(image)
        s_emb = rearrange(s_emb, '(b s) n -> b s n', b=batch)

        embedding = self.fusion_comb(s_emb, frame_embeddings)
        return embedding, kloss

    def fusion_comb(self, s_emb, frame_embeddings):
        s_emb = self.a_to_v(s_emb)
        embedding = torch.cat((s_emb, frame_embeddings), dim=1)
        embedding = self.fu_pre(embedding)
        embedding = self.fusion(embedding)
        embedding = self.fu_post(embedding)
        embedding = embedding.mean(dim=1) if self.Fpool == 'mean' else embedding[:, -1]

        return embedding

    def TMT_from_xclip(self, x):
        x, frame_embeddings, att_mat = self.temporal_enc(x)
        return x, frame_embeddings, att_mat

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(HSencoder, self).train(mode)
