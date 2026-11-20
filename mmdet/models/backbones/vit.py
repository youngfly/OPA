import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint, load_checkpoint
from mmcv.utils import to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils.ckpt_convert import swin_converter
from ..utils.transformer import PatchEmbed, PatchMerging

from functools import partial
import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed, Block
from .pos_embed import get_2d_sincos_pos_embed
import math

@BACKBONES.register_module()
class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, pretrained=None, init_cfg=None,
                 freeze_block=-1,
                 ):
        super(MaskedAutoencoderViT, self).__init__()
        self.freeze_block = freeze_block

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # todo 原本的预训练
        # self.initialize_weights()

        # todo 加载自己训练好的预训练模型
        # todo 打印模型的结构

        # todo 加入预训练模型的
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
            self.ini_weight(pretrained)
            self._freeze_block()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

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

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # todo 分类token去掉
        # x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        x = self.norm(x)
        return x, mask, ids_restore, outs

    # todo 新的 en——only
    def forward(self, img, mask_ratio=0):

        latent, mask, ids_restore, o=self.forward_encoder(img, mask_ratio)
        batch, length, vect = latent.shape
        single_out = latent.view(batch, int(math.sqrt(length)), int(math.sqrt(length)), vect).permute(0, 3, 1, 2)
        multiout = []
        for i in range(len(o)):
            multiout.append(o[i].view(batch, int(math.sqrt(length)), int(math.sqrt(length)), vect).permute(0, 3, 1, 2))
        return multiout

    # todo 自我构建的初始化模型
    def ini_weight(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m.weight, std=.02)
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)
        else:
            # todo 打印模型的结构
            # print(self)
            print("*******************")
            print("load the vit-ssl")
            # todo 加载模型
            md = torch.load(pretrained)
            # for i in md['model'].keys():
            #     print(i)
            for m in md['model'].keys():
                if hasattr(m.split('.')[0], 'patch_embed'):
                    self.patch_embed.load_state_dict(md['model'][m])
                if hasattr(m.split('.')[0], 'blocks'):
                    self.blocks.load_state_dict(md['model'][m])
                if hasattr(m.split('.')[0], 'norms'):
                    self.blocks.load_state_dict(md['model'][m])
            print('load is  ok')

    # todo 新的冻结参数
    def _freeze_block(self):

        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        m = self.blocks
        for i in range(0, self.freeze_block + 1):
            m[i].eval()
            for param in m.parameters():
                param.requires_grad = False


@BACKBONES.register_module()
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self,  pretrained=None, img_size=224, patch_size=16,depth=24, num_heads=16,mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,embed_dim=1024, in_chans=3, freeze_block=-1, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.freeze_block = freeze_block
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # self.global_pool = global_pool
        # if self.global_pool:
        #     norm_layer = kwargs['norm_layer']
        #     embed_dim = kwargs['embed_dim']
        #     self.fc_norm = norm_layer(embed_dim)
        #
        #     del self.norm  # remove the original norm

        # todo 加入预训练模型的
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
            self.ini_weight(pretrained)
            self._freeze_block()

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        # print(x.shape)
        # cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_tokens, x), dim=1)
        # print(self.pos_embed.shape)
        # x = x + self.pos_embed
        x = self.pos_drop(x)

        outs = []
        for blk in self.blocks:
            x = blk(x)
            x = self.norm(x)
            outs.append(x)

        # if self.global_pool:
        #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        #     outcome = self.fc_norm(x)
        # else:
        #     x = self.norm(x)
        #     outcome = x[:, 0]

        return x, outs

    def forward(self, img):

        single, multi = self.forward_features(img)
        batch, length, vect = single.shape
        multiout = []
        for i in range(len(multi)):
            multiout.append(multi[i].view(batch, int(math.sqrt(length)), int(math.sqrt(length)), vect).permute(0, 3, 1, 2))
        return multiout

    # todo 自我构建的初始化模型
    def ini_weight(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m.weight, std=.02)
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)
        else:
            # todo 打印模型的结构
            # print(self)
            print("*******************")
            print("load the vit-ssl")
            # todo 加载模型
            md = torch.load(pretrained)
            # for i in md['model'].keys():
            #     print(i)
            for m in md['model'].keys():
                if hasattr(m.split('.')[0], 'patch_embed'):
                    self.patch_embed.load_state_dict(md['model'][m])
                if hasattr(m.split('.')[0], 'blocks'):
                    self.blocks.load_state_dict(md['model'][m])
                if hasattr(m.split('.')[0], 'norms'):
                    self.blocks.load_state_dict(md['model'][m])
            print('load is  ok')

    # todo 新的冻结参数
    def _freeze_block(self):

        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        m = self.blocks
        for i in range(0, self.freeze_block + 1):
            m[i].eval()
            for param in m.parameters():
                param.requires_grad = False


