from collections import OrderedDict
from timm.models.layers import trunc_normal_
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
import sys
import ipdb
sys.path.append("../")
from .clip.model import LayerNorm, QuickGELU, DropPath
import torchvision
from einops.layers.torch import Rearrange


class CrossFramelAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath = 0., T=0, ):
        super().__init__()
        self.T = T

        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head,)
           
        self.attn = nn.MultiheadAttention(d_model, n_head,)
        self.ln_1 = LayerNorm(d_model)
        
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x):
        l, bt, d = x.size()
        b = bt // self.T

        x = x.view(l, b, self.T, d) 

        msg_token = self.message_fc(x[0,:,:,:]) 
        msg_token = msg_token.view(b, self.T, 1, d)  # equals to rearrange
        
        msg_token = msg_token.permute(1,2,0,3).view(self.T, b, d)  # equals to rearrange
        msg_token = msg_token + self.drop_path(self.message_attn(self.message_ln(msg_token),self.message_ln(msg_token),self.message_ln(msg_token),need_weights=False)[0])
        msg_token = msg_token.view(self.T, 1, b, d).permute(1,2,0,3)
        x = torch.cat([x, msg_token], dim=0)
        # ipdb.set_trace()
        x = x.view(l+1, -1, d)
        x = x + self.drop_path(self.attention(self.ln_1(x)))  # <- ditch this (frame patch attn)
        x = x[:l,:,:]
        x = x + self.drop_path(self.mlp(self.ln_2(x)))  # <- ditch this (frame patch attn)
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, droppath=None, use_checkpoint=False, T=8):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if droppath is None:
            droppath = [0.0 for i in range(layers)] 
        self.width = width
        self.layers = layers
        
        self.resblocks = nn.Sequential(*[CrossFramelAttentionBlock(width, heads, attn_mask, droppath[i], T) for i in range(layers)])
       
    def forward(self, x: torch.Tensor):
        if not self.use_checkpoint:
            return self.resblocks(x)
        else:
            return checkpoint_sequential(self.resblocks, 3, x)


class CrossFrameCommunicationTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 frame_size: int, droppath = None, T = 8, use_checkpoint = False, use_clip = False):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        hidden_dim = 768
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        self.conv = nn.Sequential(

            nn.Conv2d(3, out_channels=64, kernel_size=7, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(3, stride=2, padding=0),

            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 768, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(768),
            nn.SiLU(),
            nn.MaxPool2d(3, stride=2, padding=0),
        )
        self.conv_proj = nn.Linear(768*16, width)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        ## Attention Blocks
        self.transformer = Transformer(width, layers, heads, droppath=droppath, use_checkpoint=use_checkpoint, T=T,)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.frame_size = frame_size
        self.use_clip = use_clip
        self.to_frame_patch = nn.Conv2d(3, width, kernel_size=frame_size, stride=frame_size)
        patch_dim = 3 * frame_size * frame_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('bt c h w -> bt (c h w)', h=frame_size, w=frame_size),
            nn.Linear(patch_dim, width),
        )
        # self.to_frame_patch = nn.Linear((input_resolution // patch_size)**2, 1)
        # self.to_frame_patch = nn.Sequential(
        #    Rearrange('(b t) c h w -> (b t) (h w c)', h=patch_height, w=patch_width),
        #    nn.Linear(patch_dim, width),
        # )


    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        if not self.use_clip:
            x = torchvision.transforms.functional.resize(x, [self.frame_size, self.frame_size])

            x = self.conv(x)
            x = x.view(512, -1)
            x = self.conv_proj(x)

            # x = self.conv1(x)  # shape = [*, width, grid, grid] patch_size = 32 * 32, num_patch = 7 * 7
            # x = self.to_patch_embedding(x)
            # ipdb.set_trace()
            '''
            x = self.to_frame_patch(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1).squeeze(1)  # shape = [*, grid ** 2, width]
            '''
            cls_x = self.ln_post(x)
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                          device=x.device), x],
                          dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.positional_embedding.to(x.dtype)

            x = self.ln_pre(x)

            x = x.permute(1, 0, 2)
            x = self.transformer(x)  # <-
            x = x.permute(1, 0, 2)
            cls_x = self.ln_post(x[:, 0, :])
        #cls_x = [512, 768] or [32, 16, 768]

        #if self.proj is not None:
        #    cls_x = cls_x @ self.proj

        return cls_x  # , x[:,1:,:]
