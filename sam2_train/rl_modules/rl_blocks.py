import torch
import torch.nn.functional as F
from torch import nn

from collections import OrderedDict

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

        self.prompt_to_prompt = False
    
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
    
class SpatialSummarizer(nn.Module):
    def __init__(self, n_query, query_dim, spatial_dim, n_heads, d_heads, n_layers=2, down_scale=2, dropout=0.):
        super().__init__()
        self.down_scale = down_scale
        self.n_query = n_query
        self.query_dim = query_dim
        
        self.conv_in = nn.Conv2d(spatial_dim, spatial_dim, kernel_size=down_scale, stride=down_scale)
        self.qformer = nn.ModuleList(
            [QFormerBlock(query_dim, spatial_dim, n_heads, d_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.spatial_query = nn.Parameter(torch.rand(1, n_query, query_dim))
        
    def forward(self, x):
        """x: [B,C,H,W]"""
        B, C, H, W = x.shape
        
        spatial_query = self.spatial_query.expand(B, self.n_query, self.query_dim)
        
        x = self.conv_in(x)
        x = x.reshape(B, C, (H // self.down_scale) * (W // self.down_scale)).permute(0, 2, 1) # [B,L,D]
        
        for layer in self.qformer:
            spatial_query = layer(spatial_query, x)
            
        return spatial_query
    
class BasicTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,  batch_first=True)
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
        