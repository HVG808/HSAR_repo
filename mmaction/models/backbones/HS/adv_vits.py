import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import ipdb
from math import sqrt


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def deconv_nxn_bn(inp, oup, kernal_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(inp, oup, kernal_size, stride, 1, output_padding=padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def deconv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.ConvTranspose2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


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
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
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


class MTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MAttention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))  # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        ipdb.set_trace()
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)  # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x


class Conv_tiny(nn.Module):
    def __init__(self,
                 image_size,
                 dim,
                 channels,
                 depth,
                 heads,
                 ff_dim,
                 dim_head=64,
                 expansion=4,
                 kernel_size=3,
                 patch_size=(2, 2),
                 dropout=0.,
                 emb_dropout=0.,
                 pool='cls'):
        super().__init__()
        ih, iw = (image_size, image_size)
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        conv_size = image_size // 2

        self.num_patches = (conv_size // ph) * (conv_size // pw)

        L = [2, 4, 3]
        self.channels = channels
        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))

        patch_dim = ph * pw * channels[1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, 8192))

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, patch_dim))

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b d (h ph) (w pw) -> b (h w) (ph pw d)', ph=ph, pw=pw)
        )

        self.transformer = Transformer(patch_dim, depth, heads, dim_head, ff_dim, dropout)
        self.reverse_patch_embedding = nn.Sequential(
            Rearrange('b (h w) (ph pw d) -> b d (h ph) (w pw)', h=int(sqrt(self.num_patches)), ph=ph, pw=pw)
        )
        self.to_frame = nn.Sequential(
            Rearrange('b p (h w c) -> b p c h w', h=ph*2, w=pw*2)
        )
        self.pool = pool
        #self.pool = nn.AvgPool2d(64, 1)
        self.to_embedding = nn.Linear(8192, dim)

    def forward(self, x):
        img = self.conv1(x)
        img = self.mv2[0](img)
        patches = self.to_patch_embedding[0](img)
        batch, num_frame, patch_dim = patches.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch)
        patches = torch.cat((cls_tokens, patches), dim=1)
        patches = patches + self.pos_embedding[:, 0:(num_frame + 1)]
        embeddings = self.transformer(patches)
        embeddings = self.to_embedding(embeddings)
        x = embeddings.mean(dim=1) if self.pool == 'mean' else embeddings[:, 0]

        return x, embeddings


class Conv_tiny_decoder(nn.Module):
    def __init__(self,
                 image_size,
                 dim,
                 channels,
                 depth,
                 heads,
                 ff_dim,
                 dim_head=64,
                 expansion=4,
                 kernel_size=3,
                 patch_size=(2, 2),
                 dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        ih, iw = (image_size, image_size)
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        num_patches = (ih // ph) * (iw // pw)
        patch_dim = ph * pw * channels[1]

        L = [2, 4, 3]

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=ph, pw=pw)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.transformer = Transformer(patch_dim, depth, heads, dim_head, ff_dim, dropout)

        self.Deconv1 = deconv_1x1_bn(channels[1], channels[0])
        self.to_img = deconv_nxn_bn(channels[0], 3, kernel_size, stride=2, padding=1)
        self.to_frame = nn.Sequential(
            Rearrange('b p (h w c) -> b p c h w', h=ph, w=pw)
        )

        self.pool = nn.AvgPool2d(ih // 32, 1)

    def forward(self, x):
        ipdb.set_trace()
        x = self.conv1(x)
        x = self.mv2[0](x)
        x = self.mv2[1](x)

        x = self.to_patch_embedding(x)

        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x


class Conv_base(nn.Module):
    def __init__(self,
                 image_size,
                 dim,
                 channels,
                 depth,
                 heads,
                 ff_dim,
                 dim_head=64,
                 expansion=4,
                 kernel_size=3,
                 patch_size=(2, 2),
                 dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        ih, iw = (image_size, image_size)
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        conv_size = image_size // 2

        self.num_patches = (conv_size // ph) * (conv_size // pw)

        L = [2, 4, 3]
        self.channels = channels
        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 2, expansion))

        patch_dim = ph * pw * channels[1]

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, patch_dim))

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b d (h ph) (w pw) -> b (h w) (ph pw d)', ph=ph, pw=pw)
        )

        self.transformer = Transformer(patch_dim, depth, heads, dim_head, ff_dim, dropout)
        self.reverse_patch_embedding = nn.Sequential(
            Rearrange('b (h w) (ph pw d) -> b d (h ph) (w pw)', h=int(sqrt(self.num_patches)), ph=ph, pw=pw)
        )
        self.to_frame = nn.Sequential(
            Rearrange('b p (h w c) -> b p c h w', h=ph*4, w=pw*4)
        )

        self.pool = nn.AvgPool2d(64, 1)

    def forward(self, x):
        ipdb.set_trace()
        x = self.conv1(x)
        x = self.mv2[0](x)
        x = self.mv2[1](x)

        x = self.to_patch_embedding(x)

        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x


class Conv_base_decoder(nn.Module):
    def __init__(self,
                 image_size,
                 dim,
                 channels,
                 depth,
                 heads,
                 ff_dim,
                 dim_head=64,
                 expansion=4,
                 kernel_size=3,
                 patch_size=(2, 2),
                 dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        ih, iw = (image_size, image_size)
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        num_patches = (ih // (ph*4)) * (iw // (pw*4))
        patch_dim = ph * pw * channels[1]

        L = [2, 4, 3]

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=ph, pw=pw)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.transformer = Transformer(patch_dim, depth, heads, dim_head, ff_dim, dropout)

        self.Deconv1 = deconv_1x1_bn(channels[1], channels[0])
        self.to_img = deconv_nxn_bn(channels[0], 3, kernel_size, stride=2, padding=1)
        self.to_frame = nn.Sequential(
            Rearrange('b p (h w c) -> b p c h w', h=ph, w=pw)
        )

        self.pool = nn.AvgPool2d(ih // 32, 1)

    def forward(self, x):
        ipdb.set_trace()
        x = self.conv1(x)
        x = self.mv2[0](x)
        x = self.mv2[1](x)

        x = self.to_patch_embedding(x)

        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x


def build_conv_tiny_encoder(args):
    channels = [16, 32, 64, 320]
    encoder = Conv_tiny(
        image_size=args.image_size,
        patch_size=(args.patch_size, args.patch_size),
        channels=channels,
        dim=args.dim,
        depth=args.depth,
        heads=args.num_heads,
        ff_dim=args.mlp_dim,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout
    )
    return encoder


def build_deconv_tiny_decoder(args):
    channels = [16, 32, 64, 320]
    decoder = Conv_tiny_decoder(
        image_size=args.image_size,
        patch_size=(args.patch_size, args.patch_size),
        channels=channels,
        dim=args.dim,
        depth=args.decoder_depth,
        heads=args.num_heads,
        ff_dim=args.mlp_dim,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout
    )
    return decoder


def build_conv_base_encoder(args):
    channels = [16, 32, 64, 320]
    encoder = Conv_base(
        image_size=args.image_size,
        patch_size=(args.patch_size, args.patch_size),
        channels=channels,
        dim=args.dim,
        depth=args.depth,
        heads=args.num_heads,
        ff_dim=args.mlp_dim,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout
    )
    return encoder


def build_deconv_base_decoder(args):
    channels = [16, 32, 64, 320]
    decoder = Conv_base_decoder(
        image_size=args.image_size,
        patch_size=(args.patch_size, args.patch_size),
        channels=channels,
        dim=args.dim,
        depth=args.decoder_depth,
        heads=args.num_heads,
        ff_dim=args.mlp_dim,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout
    )
    return decoder
