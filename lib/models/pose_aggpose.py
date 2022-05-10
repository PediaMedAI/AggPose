# ------------------------------------------------------------------------------
# Copyright (c) SZAR-Lab
# Licensed under the MIT License.
# Modified by Iroh Cao (irohcao@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

logger = logging.getLogger(__name__)

BN_MOMENTUM = 0.1

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=(192, 256), patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MlpUpsample(nn.Module):
    def __init__(self, dim, decoder_dim, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.mlp = Mlp(dim, hidden_features=decoder_dim, out_features=decoder_dim)
        self.upsample = nn.Upsample(scale_factor = scale_factor)
    
    def forward(self, x, H, W):
        B = x.shape[0]
        x = self.mlp(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()        
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * self.scale_factor * W * self.scale_factor, -1).contiguous()
        return x


class AggPose(nn.Module):
    def __init__(self, img_size_x=256, img_size_y=192, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.depth_general = 3
        self.inplanes = 64
        
        self.mit_layers = ['block1', 'norm1', 'block2', 'norm2', 'block3', 'norm3', 'block4', 'norm4', 'patch_embed1', 'patch_embed2', 'patch_embed3', 'patch_embed4']
        # stem net
        
        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=(img_size_x, img_size_y), patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=(img_size_x // 4, img_size_y // 4), patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=(img_size_x // 8, img_size_y // 8), patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=(img_size_x // 16, img_size_y // 16), patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        
        self.patch_embed12 = OverlapPatchEmbed(img_size=(img_size_x // 4, img_size_y // 4), patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        
        self.patch_embed12k = OverlapPatchEmbed(img_size=(img_size_x // 4, img_size_y // 4), patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed23k = OverlapPatchEmbed(img_size=(img_size_x // 8, img_size_y // 8), patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        
        self.patch_embed12kk = OverlapPatchEmbed(img_size=(img_size_x // 4, img_size_y // 4), patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed23kk = OverlapPatchEmbed(img_size=(img_size_x // 8, img_size_y // 8), patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed34kk = OverlapPatchEmbed(img_size=(img_size_x // 16, img_size_y // 16), patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        
        self.merge_12_1 = Mlp(embed_dims[0] * 2, hidden_features=embed_dims[0], out_features=embed_dims[0])
        self.merge_12_2 = Mlp(embed_dims[1] * 2, hidden_features=embed_dims[1], out_features=embed_dims[1])
        
        self.merge_123_1 = Mlp(embed_dims[0] * 2, hidden_features=embed_dims[0], out_features=embed_dims[0])
        self.merge_123_2 = Mlp(embed_dims[1] * 3, hidden_features=embed_dims[1], out_features=embed_dims[1])
        self.merge_123_3 = Mlp(embed_dims[2] * 2, hidden_features=embed_dims[2], out_features=embed_dims[2])

        self.merge_1234_1 = Mlp(embed_dims[0] * 2, hidden_features=embed_dims[0], out_features=embed_dims[0])
        self.merge_1234_2 = Mlp(embed_dims[1] * 3, hidden_features=embed_dims[1], out_features=embed_dims[1])
        self.merge_1234_3 = Mlp(embed_dims[2] * 3, hidden_features=embed_dims[2], out_features=embed_dims[2])
        self.merge_1234_4 = Mlp(embed_dims[3] * 2, hidden_features=embed_dims[3], out_features=embed_dims[3])

        self.final = Mlp(embed_dims[0] * 4, hidden_features=embed_dims[0], out_features=embed_dims[0])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        
        self.block1_1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1_1 = norm_layer(embed_dims[0])
        
        self.block1_2 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1_2 = norm_layer(embed_dims[0])
        
        self.block1_3 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1_3 = norm_layer(embed_dims[0])
        
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        
        self.block2_1 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[0])])
        self.norm2_1 = norm_layer(embed_dims[1])
        
        self.block2_2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[0])])
        self.norm2_2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        
        self.block3_1 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[0])])
        self.norm3_1 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        
        self.upsample1_21 = MlpUpsample(embed_dims[1], embed_dims[0], 2)
        
        self.upsample2_21 = MlpUpsample(embed_dims[1], embed_dims[0], 2)
        self.upsample2_32 = MlpUpsample(embed_dims[2], embed_dims[1], 2)
        
        self.upsample3_21 = MlpUpsample(embed_dims[1], embed_dims[0], 2)
        self.upsample3_32 = MlpUpsample(embed_dims[2], embed_dims[1], 2)
        self.upsample3_43 = MlpUpsample(embed_dims[3], embed_dims[2], 2)

        self.upsample4_21 = MlpUpsample(embed_dims[1], embed_dims[0], 2)
        self.upsample4_31 = MlpUpsample(embed_dims[2], embed_dims[0], 4)
        self.upsample4_41 = MlpUpsample(embed_dims[3], embed_dims[0], 8)
        
        self.relu = nn.ReLU(True)
        self.final_layer = Mlp(embed_dims[0], hidden_features=embed_dims[0], out_features=num_classes)
        
        logger.info('=> init weights from normal distribution')
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] == '*':
                    need_init_state_dict[name] = m
            
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x1, H1, W1 = self.patch_embed1(x)
        # stage 1
        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        x2, H2, W2 = self.patch_embed2(x1)        
        x1 = x1.permute(0, 2, 3, 1).reshape(B, H1 * W1, -1).contiguous()
        
        for i, blk in enumerate(self.block1_1):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1_1(x1)
    
        for i, blk in enumerate(self.block2):
            x2 = blk(x2, H2, W2)
        x2 = self.norm2(x2)
        
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x3, H3, W3 = self.patch_embed3(x2)
        
        x1_mlp = x1.permute(0, 2, 3, 1).reshape(B, H1 * W1, -1).contiguous()
        x2_mlp = x2.permute(0, 2, 3, 1).reshape(B, H2 * W2, -1).contiguous()
        
        x1_new = torch.cat([x1_mlp, self.upsample1_21(x2_mlp, H2, W2)], dim = 2)
        x2_new = torch.cat([x2_mlp, self.patch_embed12(x1)[0]], dim = 2)
        x1 = self.merge_12_1(x1_new, H1, W1)
        x2 = self.merge_12_2(x2_new, H2, W2)
        
        for i, blk in enumerate(self.block1_2):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1_2(x1)
        
        for i, blk in enumerate(self.block2_1):
            x2 = blk(x2, H2, W2)
        x2 = self.norm2_1(x2)
        
        for i, blk in enumerate(self.block3):
            x3 = blk(x3, H3, W3)
        x3 = self.norm3(x3)
        
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        x3 = x3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        x4, H4, W4 = self.patch_embed4(x3)
        
        x1_mlp = x1.permute(0, 2, 3, 1).reshape(B, H1 * W1, -1).contiguous()
        x2_mlp = x2.permute(0, 2, 3, 1).reshape(B, H2 * W2, -1).contiguous()
        x3_mlp = x3.permute(0, 2, 3, 1).reshape(B, H3 * W3, -1).contiguous()
        
        x1_new = torch.cat([x1_mlp, self.upsample2_21(x2_mlp, H2, W2)], dim = 2)
        x2_new = torch.cat([self.patch_embed12k(x1)[0], x2_mlp, self.upsample2_32(x3_mlp, H3, W3)], dim = 2)
        x3_new = torch.cat([self.patch_embed23k(x2)[0], x3_mlp], dim = 2)
        x1 = self.merge_123_1(x1_new, H1, W1)
        x2 = self.merge_123_2(x2_new, H2, W2)
        x3 = self.merge_123_3(x3_new, H3, W3)
        
        for i, blk in enumerate(self.block1_3):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1_3(x1)
        
        for i, blk in enumerate(self.block2_2):
            x2 = blk(x2, H2, W2)
        x2 = self.norm2_2(x2)
        
        for i, blk in enumerate(self.block3_1):
            x3 = blk(x3, H3, W3)
        x3 = self.norm3_1(x3)
        
        for i, blk in enumerate(self.block4):
            x4 = blk(x4, H4, W4)
        x4 = self.norm4(x4)

        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        x3 = x3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        x4 = x4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        
        # stage final

        x1_mlp = x1.permute(0, 2, 3, 1).reshape(B, H1 * W1, -1).contiguous()
        x2_mlp = x2.permute(0, 2, 3, 1).reshape(B, H2 * W2, -1).contiguous()
        x3_mlp = x3.permute(0, 2, 3, 1).reshape(B, H3 * W3, -1).contiguous()
        x4_mlp = x4.permute(0, 2, 3, 1).reshape(B, H4 * W4, -1).contiguous()

        x1_new = torch.cat([x1_mlp, self.upsample3_21(x2_mlp, H2, W2)], dim = 2)
        x2_new = torch.cat([self.patch_embed12kk(x1)[0], x2_mlp, self.upsample3_32(x3_mlp, H3, W3)], dim = 2)
        x3_new = torch.cat([self.patch_embed23kk(x2)[0], x3_mlp, self.upsample3_43(x4_mlp, H4, W4)], dim = 2)
        x4_new = torch.cat([self.patch_embed34kk(x3)[0], x4_mlp], dim = 2)

        x1 = self.merge_1234_1(x1_new, H1, W1)
        x2 = self.merge_1234_2(x2_new, H2, W2)
        x3 = self.merge_1234_3(x3_new, H3, W3)
        x4 = self.merge_1234_4(x4_new, H4, W4)

        x_out = torch.cat([x1, self.upsample4_21(x2, H2, W2), self.upsample4_31(x3, H3, W3), self.upsample4_41(x4, H4, W4)], dim=2)
        x_out = self.final(x_out, H1, W1)
        x_out = self.final_layer(x_out, H1, W1)
        x_out = x_out.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        return x_out

    def forward(self, x):
        out = self.forward_features(x)
        return out


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

def get_pose_net(cfg, is_train, **kwargs):
    model = AggPose(img_size_x=cfg.MODEL.IMAGE_SIZE[1], img_size_y=cfg.MODEL.IMAGE_SIZE[0], patch_size=4, num_classes=cfg.MODEL.NUM_JOINTS, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
