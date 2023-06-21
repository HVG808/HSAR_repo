import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import ipdb
from math import sqrt
import numpy as np
from .positional_enc import PositionalEncoding
from torchvision import transforms
import cv2
from torchsummary import summary


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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
        inner_dim = dim
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5  # ?

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)  # embedding matrix

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


class Transformer(nn.Module):
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


class Flow_TMT(nn.Module):
    def __init__(self, *, frame_size, stride, dim, depth, heads, mlp_dim, pool='mean',
                 dim_head=64, dropout=0., emb_dropout=0., skip_mo=4):
        super().__init__()
        frame_height, frame_width = pair(frame_size)
        patch_size = frame_size
        patch_height, patch_width = pair(patch_size)
        self.stride = stride
        self.skip_mo = skip_mo
        self.no_skip = False

        assert frame_height % patch_height == 0 and frame_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'


        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        pre_conv = True
        if pre_conv:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3, padding=1),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 768, kernel_size=16, stride=16, bias=False)
            )
            frame_size = frame_size // 2

        row_patch = frame_size // 16
        self.proj = nn.Conv2d(5, 768, kernel_size=16, stride=16, bias=False)
        self.ln_pre = nn.LayerNorm(dim)
        if self.no_skip:
            self.to_embedding = nn.Sequential(
                Rearrange('(b s) d ph pw -> b s (d ph pw)', s=stride),
                nn.Linear(768 * row_patch * row_patch, dim),
            )
        else:
            self.to_embedding = nn.Sequential(
                Rearrange('(b s) d ph pw -> b s (d ph pw)', s=stride // skip_mo),
                nn.Linear(768 * row_patch * row_patch, dim),
            )

        self.position_enc = PositionalEncoding(dim, n_position=200)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.ln_post = nn.LayerNorm(dim)
        self.pool = pool
        self.to_latent = nn.Identity()

        print("in_TMT")
        summary(self.transformer, (17, dim))

    def forward(self, frame, gray_frame):
        batch, stride, channel, height, width = frame.shape

        params = [0.5, 13, 8, 10, 5, 1.1, 0]

        if self.no_skip:
            batch_flow = self.get_optical_flow(gray_frame, self.skip_mo, params)
            frame = torch.cat((frame, frame[:, :self.skip_mo, :, :]), dim=1)[:, self.skip_mo:, :, :]
            frame = torch.cat((frame, batch_flow), dim=2)
        else:
            batch_flow, flow_intensity = self.get_optical_flow(gray_frame, self.skip_mo, params)
            frame = torch.cat((frame[:, ::self.skip_mo], batch_flow), dim=2)
        frame = rearrange(frame, 'b s c h w -> (b s) c h w')
        if self.pre_conv is not None:
            patches = self.pre_conv(frame)
        else:
            patches = self.proj(frame)
        x = self.to_embedding(patches)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch)
        x = torch.cat((x, cls_tokens), dim=1)

        x = self.position_enc(x)

        x = self.ln_pre(x)
        x = self.dropout(x)

        for i, layer in enumerate(self.transformer.layers):
            if i < len(self.transformer.layers) - 1:
                x = layer[0](x) + x
                x = layer[1](x) + x
            else:
                # return attention of the last block
                att_mat = layer[0].fn(x.clone(), return_weights=True)
                x = layer[0](x) + x
                x = layer[1](x) + x
        nh = self.transformer.layers[0][0].fn.heads
        att_weight = att_mat[:, :, -1, :-1].reshape(batch, nh, -1)
        embeddings = x

        x = embeddings.mean(dim=1) if self.pool == 'mean' else embeddings[:, -1]
        x = self.ln_post(x)
        x = self.to_latent(x)
        return x, embeddings, flow_intensity, att_weight

    def get_optical_flow(self, frames, skip_mo, params):
        batch, stride, channel, height, width = frames.shape
        if self.no_skip:
            gray_frames = frames
            gray_frames2 = torch.cat((frames, frames[:, :skip_mo, :, :]), dim=1)[:, skip_mo:, :, :]
        else:
            gray_frames = frames[:, ::skip_mo]
            gray_frames2 = torch.cat((gray_frames, gray_frames[:, -1:]), dim=1)[:, 1:]

        batch_flow = []
        batch_intensity = []
        for b in range(batch):
            seq_flow = []
            seq_intensity = []
            prev_gray = gray_frames[b]
            gray = gray_frames2[b]
            for f1, f2 in zip(prev_gray, gray):
                f1 = np.uint8(f1.permute(1, 2, 0).cpu().detach().numpy() * 255)
                f2 = np.uint8(f2.permute(1, 2, 0).cpu().detach().numpy() * 255)
                f1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
                f2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(f1, f2, None, *params)
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                seq_intensity.append(np.sum(magnitude))
                flow_vec = torch.from_numpy(flow).permute(2, 0, 1)
                seq_flow.append(flow_vec)
            batch_flow.append(torch.stack(seq_flow, dim=0))
            batch_intensity.append(seq_intensity)
        batch_flow = torch.stack(batch_flow, dim=0).cuda()
        return batch_flow, batch_intensity


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(input_dim)
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        x = self.norm1(x)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], \
                         requires_grad=False).cuda()
        return x

