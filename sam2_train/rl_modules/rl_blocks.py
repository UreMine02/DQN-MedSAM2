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

        self.to_out = nn.Linear(inner_dim, query_dim)

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
        self.attn = CrossAttention(query_dim=hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(hidden_dim, hidden_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(hidden_dim * 4, hidden_dim))
        ]))
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
    def forward(self, x_f, x, training=True):
        """
        Forward
        
        :param x_f: [B,L,D]
        :param x: [B,L,D]
        """
        x = x + F.dropout(self.attn(self.norm1(x), context=torch.cat([x_f, x], dim=1)), p=self.dropout, training=training)
        x = x + F.dropout(self.mlp(self.norm2(x)), p=self.dropout, training=training)
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
        self.spatial_query = nn.Parameter(torch.rand(1, n_query, spatial_dim))
        self.spatial_dim = spatial_dim
        
        self.initialize_parameters()
        
    def forward(self, x, training=True):
        """x: [B,C,H,W]"""
        B, C, H, W = x.shape
        
        spatial_query = self.spatial_query.expand(B, self.n_query, self.spatial_dim)

        x = x.reshape(B, C, -1).permute(0, 2, 1) # [B,L,D]
        
        for layer in self.qformer:
            spatial_query = layer(x_f=x, x=spatial_query, training=training)
            
        return spatial_query
    
    def initialize_parameters(self):
        nn.init.normal_(self.spatial_query, std=0.02)
        
        proj_std = (self.spatial_dim ** -0.5) * ((2 * self.n_layers) ** -0.5)
        attn_std = self.spatial_dim ** -0.5
        fc_std = (2 * self.spatial_dim) ** -0.5
        for block in self.qformer:
            nn.init.normal_(block.attn.to_q.weight, std=attn_std)
            nn.init.normal_(block.attn.to_k.weight, std=attn_std)
            nn.init.normal_(block.attn.to_v.weight, std=attn_std)
            nn.init.normal_(block.attn.to_out.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
    
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
        