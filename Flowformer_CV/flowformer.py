import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Flow_Attention(nn.Module):
    """
    Mobile Attention with head competing mechanism, adapted to match
    the Flow_Attention API:
      - __init__(dim, num_heads=12)
      - forward(x) where x: (B, N, C)
    """
    def __init__(self, dim, num_heads=12, drop_out=0.05, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.eps = eps

        # Instead of separate query/key/value_projection, 
        # we use a single qkv projection for compatibility with Flow_Attention
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        
        # Output projection, matching Flow_Attention's "proj"
        self.proj = nn.Linear(dim, dim)

        # Optional dropout on the output
        self.dropout = nn.Dropout(drop_out)
        
    def kernel_method(self, x: torch.Tensor) -> torch.Tensor:
        """Non-negative projection via sigmoid."""
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        """
        We replicate the "mobile" style dot product from the original code
        but adapted for the dimension usage in this setup.
        """
        # In the original code, it used:
        #   kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        #   qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        # We'll keep it if we want a 2-step approach. However, we can also 
        # rely on matrix multiplication. For clarity, let's keep the einsum form.
        kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        return qkv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, N, C)
        We must:
          1) Generate q, k, v from x
          2) Perform "mobile attention" logic 
             (sink_incoming, source_outgoing, etc.)
          3) Return the output in shape (B, N, C)
        """
        B, N, C = x.shape
        # 1. QKV
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # permute to separate Q, K, V: shape -> (3, B, num_heads, N, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, N, head_dim)

        # 2. Non-negative projection
        q = self.kernel_method(q)
        k = self.kernel_method(k)

        # Let’s define some helper variables for clarity:
        # B_ = B, H_ = self.num_heads, L_ = N, D_ = C // self.num_heads

        # 3. Mobile-Attention logic

        # (a) sink_incoming:
        #    From original code, shaped as (B, n_heads, L)
        #    We do a sum over last dimension. But we must replicate the “(queries + eps)*(keys.sum(dim=2) + eps)”
        #    carefully. We'll sum over the dimension representing N in k (dim=2).
        #    The original used:
        #      sink_incoming = 1.0 / torch.sum((queries + eps) * (keys.sum(dim=1)+eps), dim=-1)
        #    but note that we have shapes (B, H, N, D). We'll adapt carefully.
        #
        #    In "Flow_Attention", for normalizing, it uses the sum across the 'N' dimension. We'll do similarly.
        #    We'll match the approach used in the original Mobile_Attention code with repeats.

        eps = self.eps
        # sum of keys over the "N" dimension => shape (B, H, D)
        k_sum = k.sum(dim=2)  # shape (B, H, D)
        # we need to broadcast for multiplication with q: (B,H,N,D)
        # so let's do something like: (k_sum[:, :, None, :] + eps)
        # to get shape (B,H,1,D)
        
        sink_incoming = 1.0 / torch.sum(
            (q + eps) * (k_sum[:, :, None, :] + eps),
            dim=-1  # summation over D dimension
        )  # shape: (B, H, N)

        # (b) source_outgoing:
        #    sum of queries over the "N" dimension => shape (B, H, D)
        q_sum = q.sum(dim=2)
        source_outgoing = 1.0 / torch.sum(
            (k + eps) * (q_sum[:, :, None, :] + eps),
            dim=-1  # sum over D
        )  # (B, H, N)

        # (c) conserved_sink
        #    from the original:
        #      conserved_sink = torch.sum((queries+eps)*((keys*source_outgoing[:, :, :, None]).sum(dim=1)+eps),dim=-1)
        #    but we have dimension order: (B,H,N,D).
        #    We'll do something similar with a sum over N dimension in k after multiplying by source_outgoing.

        # keys * source_outgoing => shape (B, H, N, D)
        k_so = k * source_outgoing[:, :, :, None]  
        k_so_sum = k_so.sum(dim=2, keepdim=True)  # (B, H, 1, D)
        
        # (queries+eps)*(k_so_sum+eps) -> sum over D => shape (B,H,N)
        conserved_sink = torch.sum((q + eps) * (k_so_sum + eps), dim=-1) + eps

        # (d) conserved_source
        #    similarly, we do queries * sink_incoming => shape (B, H, N, D)
        q_si = q * sink_incoming[:, :, :, None]
        q_si_sum = q_si.sum(dim=2, keepdim=True)  # (B, H, 1, D)
        
        conserved_source = torch.sum((k + eps) * (q_si_sum + eps), dim=-1) + eps
        # clamp for stability:
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)

        # (e) Competition & Allocation
        #     sink_allocation => shape (B,H,N)
        #        = sigmoid(conserved_sink * (float(q.shape[2])/float(k.shape[2])))
        #        i.e. scale by ratio of N in q vs. N in k. Usually same N though?
        ratio = float(q.shape[2]) / float(k.shape[2])  # typically 1 if same N
        sink_allocation = torch.sigmoid(conserved_sink * ratio)

        # source_competition => shape (B,H,N)
        #   = softmax(conserved_source, dim=-1) * float(k.shape[2])
        #   i.e. multiply by the number of tokens in k
        source_competition = F.softmax(conserved_source, dim=-1) * float(k.shape[2])

        # (f) Dot product
        #   In the original, x = dot_product(queries * sink_incoming, keys, values * source_competition)*sink_allocation
        #   We'll replicate that but with shape (B,H,N,D).
        #   queries * sink_incoming => shape (B,H,N,D)
        q_norm = q * sink_incoming[:, :, :, None]
        v_comp = v * source_competition[:, :, :, None]

        # Perform the "mobile-style" dot product, i.e. qkv = ...
        x_mobil = self.dot_product(q_norm, k, v_comp)
        # shape => (B,H,N,D)

        # multiply by sink_allocation => shape => (B,H,N,D)
        x_mobil = x_mobil * sink_allocation[:, :, :, None]

        # (g) Rearrange back to (B,N,C)
        x_mobil = x_mobil.transpose(1, 2).reshape(B, N, C)

        # (h) Final projection + dropout
        x_mobil = self.proj(x_mobil)  # (B, N, C)
        x_mobil = self.dropout(x_mobil)

        return x_mobil

class DownLayer(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=56, dim_in=64, dim_out=128):
        super().__init__()
        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.num_patches = img_size * img_size // 4

    def forward(self, x):
        B, N, C = x.size()
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        x = self.proj(x).permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.dim_out)
        return x


class Flowformer(nn.Module):

    def __init__(self, img_size=224, patch_size=4, num_classes=1000, embed_dim=[64, 128, 256, 512], depth=[2, 2, 10, 4],
                 mlp_ratio=[4, 4, 4, 4],
                 drop_rate=0., drop_path_rate=0., norm_layer=None, init_values=0.001, dropcls=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = nn.ModuleList()

        patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim[0])
        num_patches = patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))

        self.patch_embed.append(patch_embed)

        sizes = [56, 28, 14, 7]
        for i in range(4):
            sizes[i] = sizes[i] * img_size // 224

        for i in range(3):
            patch_embed = DownLayer(sizes[i], embed_dim[i], embed_dim[i + 1])
            num_patches = patch_embed.num_patches
            self.patch_embed.append(patch_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        cur = 0
        for i in range(4):
            h = sizes[i]
            w = h // 2 + 1
            blk = nn.Sequential(*[
                My_BlockLayerScale(
                    dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                    drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                    init_values=init_values)
                for j in range(depth[i])
            ])
            self.blocks.append(blk)
            cur += depth[i]

        # Classifier head
        self.norm = norm_layer(embed_dim[-1])
        self.head = nn.Linear(self.num_features, num_classes)

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def forward_features(self, x):
        for i in range(4):
            x = self.patch_embed[i](x)
            if i == 0:
                x = x + self.pos_embed
            x = self.blocks[i](x)

        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x
