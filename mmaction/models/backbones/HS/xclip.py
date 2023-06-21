from typing import Tuple, Union
import torch
from torch import nn
import numpy as np
from .mit import MultiframeIntegrationTransformer
from .cct import CrossFrameCommunicationTransformer
import sys
import warnings
sys.path.append("../")
from .clip.model import CLIP,LayerNorm,Transformer
import ipdb
# import clip

class Xclip_visual(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 frame_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # video
                 T=8,
                 droppath=0.,
                 mit_layers=1,
                 use_clip=False,
                 token='cls'
                 ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )

        self.mit = MultiframeIntegrationTransformer(T=T, embed_dim=embed_dim, layers=mit_layers, )
        use_checkpoint = True

        dpr = [x.item() for x in torch.linspace(0, droppath, vision_layers)] if droppath > 0. else None

        vision_heads = vision_width // 64
        self.visual = CrossFrameCommunicationTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            frame_size=frame_size,
            droppath=dpr,
            T=T,
            use_checkpoint=use_checkpoint,
            use_clip=use_clip,
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.cache_text_features = None
        self.prompts_visual_ln = LayerNorm(vision_width)
        self.prompts_visual_proj = nn.Parameter(torch.randn(vision_width, embed_dim))

        self.initialize_parameters()
        self.token = token

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}

    def encode_image(self, image):
        return self.visual(image)

    def encode_video(self, image):
        b, t, c, h, w = image.size()
        image = image.reshape(-1, c, h, w)

        cls_features = self.encode_image(image)  # cls_features = CLS_token of cross frame message

        # img_features = self.prompts_visual_ln(img_features)  # [B x T, patch, dim]
        # img_features = img_features @ self.prompts_visual_proj

        cls_features = cls_features.view(b, t, -1)  # [B, T, dim]
        # img_features = img_features.view(b, t, -1, cls_features.shape[-1])

        video_features, att_mat = self.mit(cls_features) # <-

        return video_features, att_mat # , img_features

    def cache_text(self, text):
        self.eval()
        with torch.no_grad():
            if self.cache_text_features is None:
                self.cache_text_features = self.encode_text(text)
        self.train()
        return self.cache_text_features

    def forward(self, image):
        b = image.shape[0]
        video_features, att_mat = self.encode_video(image)

        if self.token == 'mean':
            video_token = video_features.mean(dim=1, keepdim=False)
        else:
            video_token = video_features[:, -1]


        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        return video_token, video_features, att_mat

    def get_last_selfattention(self, image):
        b, t, c, h, w = image.size()
        image = image.reshape(-1, c, h, w)
        cls_features = self.encode_image(image)  # cls_features = CLS_token of cross frame message
        cls_features = cls_features.view(b, t, -1)  # [B, T, dim]
        video_features = self.mit.get_last_selfattention(cls_features)  # <-

        return video_features  # , img_features
