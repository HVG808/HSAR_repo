import torch
from torch import nn
from collections import OrderedDict
from timm.models.layers import trunc_normal_
import sys
sys.path.append("../")
from .clip.model import QuickGELU
import ipdb


class MultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., stride=32):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.time_mask = torch.tril(torch.ones((stride+1, stride+1)), diagonal=0).bool().cuda()


    def forward(self, q, k, v, need_weights=False, attn_mask=None, average_attn_weights=False):
        x = q
        x = x.permute(1, 0, 2)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.masked_fill_(~self.time_mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(1, 0, 2)
        if need_weights:
            return attn
        return x, attn


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, stride: int, attn_mask: torch.Tensor = None):
        super().__init__()

        #self.attn = nn.modules.MultiheadAttention(d_model, n_head)
        self.attn = MultiheadAttention(d_model, n_head, stride=stride)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, return_weights=False):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if return_weights:
            return self.attn(x, x, x, need_weights=return_weights, attn_mask=self.attn_mask, average_attn_weights=False)
        else:
            return self.attn(x, x, x, need_weights=return_weights, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor, return_weights=False):
        if return_weights:
            return self.attention(self.ln_1(x), return_weights=return_weights)
        y, att_mat = self.attention(self.ln_1(x), return_weights=return_weights)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att_mat


class MultiframeIntegrationTransformer(nn.Module):
    def __init__(self, T, embed_dim=512, layers=1,):
        super().__init__()
        self.T = T
        transformer_heads = embed_dim // 64
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.empty(1, T+1, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(d_model=embed_dim, n_head=transformer_heads, stride=T) for _ in range(layers)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)
        ori_x = x
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        att_mat = []
        for i, blk in enumerate(self.resblocks):
            x, mat = blk(x)
            att_mat.append(mat)

        x = x.permute(1, 0, 2)
        x = x.type(ori_x.dtype) + ori_x

        return x, att_mat  # x.mean(dim=1, keepdim=False)

    def get_last_selfattention(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)
        x = x + self.positional_embedding

        x = x.permute(1, 0, 2)
        for i, blk in enumerate(self.resblocks):
            if i < len(self.resblocks) - 1:
                x, att_mat = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_weights=True)
