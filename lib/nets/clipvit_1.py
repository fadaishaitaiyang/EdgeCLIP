

from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import  drop_path, trunc_normal_
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # print(x.shape)
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIPVisionTransformer(nn.Module):
    def __init__(self, input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=1024,
                 drop_path_rate=0.0, out_indices=[3, 5, 7, 11], pretrained=None, get_embeddings=False, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.scale =scale
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        self.out_indices = out_indices

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
            self.max = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2,padding=1, bias=False)

        embed_dim = width
        
        # self.mlp1 = nn.Sequential(  
        #     nn.Linear(768, 768 * 4),
        #     QuickGELU(),
        #     nn.Linear(768 * 4, 128) 
        # )
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(768, 768 * 4),
        #     QuickGELU(),
        #     nn.Linear(768 * 4, 256)
        #     )
        # self.mlp3 = nn.Sequential(
        #     nn.Linear(768, 768 * 4),
        #     QuickGELU(),
        #     nn.Linear(768 * 4, 512)
        #     )

        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, 128, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.ConvTranspose2d(embed_dim, 512, kernel_size=1, stride=1),
            )

            self.fpn4 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        elif patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.GroupNorm(1, embed_dim)

            self.fpn3 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.MaxPool2d(kernel_size=4, stride=4),
            )

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    print(
                        f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}')
                    cls_pos = state_dict["positional_embedding"][0:1, :]
                    spatial_pos = F.interpolate(
                        state_dict["positional_embedding"][1:, ].reshape(1, 14, 14, 768).permute(0, 3, 1, 2),
                        size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    spatial_pos = spatial_pos.reshape(768, self.spatial_size * self.spatial_size).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in vision transformer')

    def forward(self, x: torch.Tensor):
        # print(self.scale)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, H, W = x.shape #H=W=14
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:, ].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2),
                                    size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        features = []
        for i, blk in enumerate(self.transformer.resblocks):
            x = blk(x)
            # if i == 3:
            #     xp =self.mlp1(x)
            #     xp = xp.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
            #     features.append(xp.contiguous())
            # if i == 5:
            #     xp =self.mlp2(x)
            #     xp = xp.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
            #     features.append(xp.contiguous())
            # if i == 7:
            #     xp =self.mlp3(x)
            #     xp = xp.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
            #     features.append(xp.contiguous())
            if i in self.out_indices:
                # print(x.shape)
                # xp = x.permute(1,0,2)
                # print(xp.shape)
                xp = x
                features.append(xp)
        features.append(x)
        # print(len(features))
        # features[0] =self.fpn1(features[0])
        # ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        # for i in range(len(features)):
        #     features[i] = ops[i](features[i])
        

        # if self.get_embeddings:
        #     x = x.permute(1, 0, 2)
        #     x = self.ln_post(x)
        #     x = x @ self.proj
        #     # print(H)14
        #     # print(W)14
        #     # print(x.shape) #(32,197,512)
        #     # x = self.max(x)
        #     # print(x.shape) #(32,98,256)
            
        #     # print(x[:,0].shape)(32,512)
        #     # print(x[:, 1:].shape)(32,196,512)
        #     global_embedding = x[:, 0]
        #     visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)  # B C H W
        #     visual_embedding = self.max(visual_embedding)
        #     # print(visual_embedding.shape)
        #     features.append(global_embedding)
        #     features.append(visual_embedding)

        #     # features.append([global_embedding, visual_embedding])

        return tuple(features)
