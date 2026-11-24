import torch
from functools import partial
import torch.nn as nn
from timm.layers import to_2tuple, trunc_normal_

from smplx.lbs import lbs

from .smpl_head import SMPLTransformerDecoderHead


def vit(cfg):
    return ViT(img_size=(256, 192), patch_size=16, embed_dim=1280, depth=32, num_heads=16, ratio=1, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.55)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, attn_head_dim=None):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim

        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        (B, N, C) = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        (q, k, v) = (qkv[0], qkv[1], qkv[2])
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0]) * ratio ** 2
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size[0] // ratio, padding=4 + 2 * (ratio // 2 - 1))


    def forward(self, x, **kwargs):
        (B, C, H, W) = x.shape
        x = self.proj(x)
        (Hp, Wp) = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)

        return (x, (Hp, Wp))


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, hybrid_backbone=None, norm_layer=None, frozen_stages=-1, ratio=1, last_norm=True, patch_padding='pad', freeze_attn=False, freeze_ffn=False):
        super(ViT, self).__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-06)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth, device='cpu')]


        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)])
        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)


    def forward_features(self, x):
        (B, C, H, W) = x.shape
        (x, (Hp, Wp)) = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for blk in self.blocks:
            x = blk(x)

        x = self.last_norm(x)
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        return xp


    def forward(self, x):
        x = self.forward_features(x)

        return x


class SMPLGeometry(nn.Module):
    def __init__(self, state):
        super().__init__()

        for name, value in state.items():
            self.register_buffer(name, value)


    def forward(self, parameters):
        rotations = torch.cat([parameters['global_orient'], parameters['body_pose']], dim=1)
        vertices, _ = lbs(parameters['betas'].float(), rotations.float(), self.v_template,
                          self.shapedirs, self.posedirs, self.J_regressor,
                          self.parents, self.lbs_weights, pose2rot=False)

        return vertices


class HMR2(nn.Module):
    def __init__(self, cfg, state):
        super().__init__()

        with torch.device('meta'):
            self.backbone = vit(cfg)
            self.smpl_head = SMPLTransformerDecoderHead(cfg)

        self.smpl = SMPLGeometry({key[5:]: value for key, value in state.items() if key.startswith('smpl.')})
        self.load_state_dict(state, strict=True, assign=True)


    def forward(self, image):
        features = self.backbone(image[:, :, :, 32:-32])
        parameters, camera = self.smpl_head(features)

        return self.smpl(parameters), camera