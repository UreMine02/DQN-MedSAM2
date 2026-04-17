import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from sam2_train.modeling.position_encoding import apply_rotary_enc, compute_axial_cis
from sam2_train.modeling.sam2_utils import MLP
from sam2_train.utils.misc import get_sdpa_settings

OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()

class BatchNorm1d(nn.BatchNorm1d):
    def forward(self, x: torch.Tensor):
        '''Custom BatchNorm1d for [B,L,D] input'''
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
            x = super().forward(x)
            x = x.permute(0, 2, 1)
            return x
        return super().forward(x)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, context=None, mask=None):
        is_self_attn = context is None
        context = context if context is not None else x

        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        b, n, d = q.shape 

        q = q.reshape(b, -1, h, d // h).permute(0, 2, 1, 3).reshape(b*h, -1, d // h)
        k = k.reshape(b, -1, h, d // h).permute(0, 2, 1, 3).reshape(b*h, -1, d // h)
        v = v.reshape(b, -1, h, d // h).permute(0, 2, 1, 3).reshape(b*h, -1, d // h)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.reshape(b, h, n, -1).permute(0, 2, 1, 3).reshape(b, n, -1)

        return self.to_out(out)

class QFormerBlock(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_heads, dropout=0.):
        super().__init__()
        self.attn1 = CrossAttention(query_dim, heads=n_heads, dim_head=d_heads, dropout=dropout)  # is a self-attention
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(query_dim, query_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(query_dim * 4, query_dim))
        ]))
        self.attn2 = CrossAttention(query_dim, context_dim=context_dim, heads=n_heads, dim_head=d_heads, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)

    def forward(self, x, context):
        assert context is not None
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.mlp(self.norm3(x)) + x
        return x
    
# class SpatialSummarizer(nn.Module):
#     def __init__(self, n_query, query_dim, spatial_dim, n_heads, d_heads, n_layers=2, down_scale=2, dropout=0.):
#         super().__init__()
#         self.down_scale = down_scale
#         self.n_query = n_query
#         self.query_dim = query_dim
        
#         self.conv_in = nn.Conv2d(spatial_dim, spatial_dim, kernel_size=down_scale, stride=down_scale)
#         self.qformer = nn.ModuleList(
#             [QFormerBlock(query_dim, spatial_dim, n_heads, d_heads, dropout=dropout) for _ in range(n_layers)]
#         )
#         self.spatial_query = nn.Parameter(torch.rand(1, n_query, query_dim))
        
#     def forward(self, x):
#         """x: [B,C,H,W]"""
#         B, C, H, W = x.shape
        
#         spatial_query = self.spatial_query.expand(B, self.n_query, self.query_dim)
        
#         x = self.conv_in(x)
#         x = x.reshape(B, C, (H // self.down_scale) * (W // self.down_scale)).permute(0, 2, 1) # [B,L,D]
        
#         for layer in self.qformer:
#             spatial_query = layer(spatial_query, x)
            
#         return spatial_query

class PerceiverResampler(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=1, dropout=0.):
        super().__init__()
        # self.attn = CrossAttention(query_dim=hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        # self.attn = RoPEAttention(
        #     embedding_dim=hidden_dim, kv_in_dim=hidden_dim, rope_k_repeat=True, rope_theta=10000, feat_sizes=[32, 32], 
        #     num_heads=num_heads, downsample_rate=1, dropout=0.1, 
        # )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("norm", nn.LayerNorm(hidden_dim)),
            ("c_fc", nn.Linear(hidden_dim, hidden_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(hidden_dim * 4, hidden_dim))
        ]))
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
    def attention(self, x: torch.Tensor, context: torch.Tensor):
        # attn = self.attn(x, context=context)
        attn = self.attn(x, context, context, need_weights=False)[0]
        # attn = self.attn(q=x, k=context, v=context, num_k_exclude_rope={})
        return attn
        
    def forward(self, x_f, x, training=True):
        """
        Forward
        
        :param x_f: [B,L,D]
        :param x: [B,L,D]
        """
        x_f = self.norm1(x_f)
        x = self.norm2(x)
        x = x + self.attention(x, context=torch.cat([x_f, x], dim=1))
        x = x + self.mlp(x)
        return x
    
class SpatialSummarizer(nn.Module):
    def __init__(self, n_query, query_dim, spatial_dim, n_heads, d_heads, n_layers=2, down_scale=2, dropout=0.):
        super().__init__()
        self.down_scale = down_scale
        self.n_query = n_query
        self.n_layers = n_layers
        
        self.qformer = nn.ModuleList(
            [PerceiverResampler(hidden_dim=spatial_dim, num_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        scale = spatial_dim ** -0.5
        self.spatial_query = nn.Parameter(scale * torch.rand(1, n_query, spatial_dim))
        self.spatial_dim = spatial_dim
        
        self.initialize_parameters()
        
    def forward(self, x, training=True):
        """x: [B,C,H,W]"""
        B, C, H, W = x.shape
        
        x = x.reshape(B, C, -1).permute(0, 2, 1) # [B,L,D]
        spatial_query = self.spatial_query.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

        for layer in self.qformer:
            spatial_query = layer(
                x_f=x,
                x=spatial_query,
                training=training
            )
            
        return spatial_query
    
    def initialize_parameters(self):
        nn.init.normal_(self.spatial_query, std=0.02)
        
        proj_std = (self.spatial_dim ** -0.5) * ((2 * self.n_layers) ** -0.5)
        attn_std = self.spatial_dim ** -0.5
        fc_std = (2 * self.spatial_dim) ** -0.5
        for block in self.qformer:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
    
class BasicTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x, x, x, need_weights=False)[0] + x
        x = self.mlp(self.norm2(x)) + x
        return x

class BidirectionalQFormer(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_heads, dropout=0.):
        super().__init__()
        self.q_former_1 = QFormerBlock(query_dim, context_dim, n_heads, d_heads, dropout=dropout)
        self.q_former_2 = QFormerBlock(query_dim, context_dim, n_heads, d_heads, dropout=dropout)
        
    def forward(self, x, y):
        x = self.q_former_1(x, y)
        y = self.q_former_2(y, x)
        return x, y

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        with torch.backends.cuda.sdp_kernel(
            enable_flash=USE_FLASH_ATTN,
            # if Flash attention kernel is off, then math kernel needs to be enabled
            enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON,
            enable_mem_efficient=OLD_GPU,
        ):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention(Attention):
    """Attention with rotary position encoding."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        feat_sizes=(32, 32),  # [w, h] for stride 16 feats at 512 resolution
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat

        # CW GATING BEFORE PROJ
        self.ctx_gating_ptr_proj = nn.Linear(self.kv_in_dim, self.kv_in_dim)
        self.ctx_gating_mem_proj = nn.Linear(self.kv_in_dim, self.kv_in_dim)

        # # CW GATING AFTER POS EMBED
        # self.ctx_gating_ptr_proj = nn.Linear(self.internal_dim, self.internal_dim)
        # self.ctx_gating_mem_proj = nn.Linear(self.internal_dim, self.internal_dim)

        # SW GATING
        # self.ctx_gating_ptr_proj = nn.Linear(4, 4096)
        # self.ctx_gating_mem_proj = nn.Linear(4096, 4096)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, return_attn: bool, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        # # NOTE: TEST GATING
        # if num_k_exclude_rope > 0:
        #     m = num_k_exclude_rope // 4
        #     b, h, l, d = k.shape
        #     mem, ptr = k.tensor_split(indices=(-num_k_exclude_rope,), dim=2)

        #     # CW GATING
        #     mem_ = mem.reshape(b, h, m, -1, d) # [1,h,m,4096,256]
        #     ptr_ = ptr.reshape(b, h, m, -1, d) # [1,h,m,4,256]

        #     mem_ = self.ctx_gating_mem_proj(mem_)
        #     ptr_ = self.ctx_gating_ptr_proj(ptr_)

        #     ptr_ = ptr_.sum(dim=-2, keepdim=True)
        #     gating_logits = mem_ + ptr_ # [1,m,4096,64]
        #     gating_score = gating_logits.sigmoid() # [1,m,4096,64]

        #     gated_mem = mem_ * gating_score
        #     gated_mem = gated_mem.reshape(b, h, -1, d)

        #     k = torch.cat([gated_mem, ptr], dim=2)

            # # SW GATING
            # mem_ = mem.reshape(b, m, -1, d).transpose(2, 3) # [1,m,64,4096]
            # ptr_ = ptr.reshape(b, m, -1, d).transpose(2, 3) # [1,m,64,4]

            # mem_ = self.ctx_gating_mem_proj(mem_) # [1,m,64,4096]
            # ptr_ = self.ctx_gating_ptr_proj(ptr_) # [1,m,64,4096]

            # gating_logits = mem_ + ptr_ # [1,m,64,4096]
            # gating_score = gating_logits.sigmoid() # [1,m,64,4096]
            # gated_mem = mem_ * gating_score
            # gated_mem = gated_mem.transpose(2, 3).reshape(b, -1, d)

            # k = torch.cat([gated_mem, ptr], dim=1)

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        # Compute attn_weight for later use
        attn_weight = None
        if return_attn:
            scale_factor = 1 / math.sqrt(q.size(-1))
            attn_weight = q @ k.transpose(-2, -1) * scale_factor

        # # Attention
        # with torch.backends.cuda.sdp_kernel(
        #     enable_flash=USE_FLASH_ATTN,
        #     # if Flash attention kernel is off, then math kernel needs to be enabled
        #     enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON,
        #     enable_mem_efficient=OLD_GPU,
        # ):
        #     out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out, attn_weight