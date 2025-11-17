import torch
from torch import nn
from q_learning_blocks import QFormerBlock, BasicTransformerBlock, SpatialSummarizer
    
    
class SimplifiedQNetwork(nn.Module):
    def __init__(self, action_dim, num_maskmem, hidden_dim=256):
        super(SimplifiedQNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_maskmem = num_maskmem
        self.image_conv_in = nn.Conv2d(256, hidden_dim, kernel_size=32, stride=32)
        self.memory_conv_in = nn.Conv2d(64, hidden_dim, kernel_size=32, stride=32)
        
        self.temporal_summary = nn.ModuleList([QFormerBlock(hidden_dim, 8) for _ in range(6)])
        self.action_decoder = nn.ModuleList([BasicTransformerBlock(hidden_dim, 8) for _ in range(6)])
        
        self.action_pos_embed = nn.Parameter(torch.rand(1, 32, hidden_dim))
        self.temporal_query = nn.Parameter(torch.rand(1, 6, hidden_dim))
        self.action_query = nn.Parameter(torch.rand(1, 1, hidden_dim))
        
        self.proj_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, action_dim),
            nn.Dropout(0.2),
            nn.LayerNorm(action_dim),
        )

    def forward(self, image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr):
        B, T, C, H, W = bank_feat.shape
        temporal_query = self.temporal_query.repeat(B, 1, 1) # [B,L,D]
        action_query = self.action_query.repeat(B, 1, 1) # [B,L,D]
        
        bank_feat = bank_feat.reshape(B * T, C, H, W)
        bank_feat = self.memory_conv_in(bank_feat) # [B*T, C, H, W]
        bank_feat = bank_feat.reshape(B, T, self.hidden_dim, -1) # [B,T,C,L]
        bank_feat = bank_feat.permute(0, 1, 3, 2) # [B,T,L,C]
        bank_feat = bank_feat.reshape(B, -1, self.hidden_dim) # [B,T*L,C]
        for layer in self.temporal_summary:
            bank_feat = layer(temporal_query, bank_feat)
        
        image_feat = self.image_conv_in(image_feat) # [B,C,H,W]
        memory_feat = self.memory_conv_in(memory_feat) # [B,C,H,W]
        image_feat = image_feat.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)
        memory_feat = memory_feat.reshape(B, self.hidden_dim, -1).permute(0, 2, 1)
        
        memory_ptr = memory_ptr.unsqueeze(1)
        
        # L = [4, 4, 1, 6, 16, 1]
        combined_feats = torch.cat([action_query, image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr], dim=1)
        combined_feats = combined_feats + self.action_pos_embed.repeat(B, 1, 1)
        for layer in self.action_decoder:   
            combined_feats = layer(combined_feats)
        
        q_values = self.proj_out(combined_feats[:, 0, :])
        return q_values
    
class QformerSpecificQNetwork(nn.Module):
    def __init__(self, action_dim, num_maskmem, hidden_dim=256):
        super(QformerSpecificQNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_maskmem = num_maskmem
        
        self.image_spatial_summary = SpatialSummarizer(8, 512, 256, 12, 64, dropout=0.2, n_layers=4)
        self.memory_spatial_summary = SpatialSummarizer(8, 64, 64, 1, 64, dropout=0.2, n_layers=4)
        self.action_decoder = nn.ModuleList([QFormerBlock(512, 512, 12, 64, dropout=0.2) for _ in range(4)])
        
        self.non_drop_embed = nn.Parameter(torch.rand(1, 1, 512))
        
        self.proj_out = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Dropout(0.5),
            nn.LayerNorm(1),
        )

    def forward(self, image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr):
        B, T, C, H, W = bank_feat.shape
        non_drop_embed = self.non_drop_embed.repeat(B, 1, 1)
        
        combined_mem_feat = torch.cat([bank_feat, memory_feat.unsqueeze(1)], dim=1)
        combined_mem_feat = combined_mem_feat.reshape(B * (T+1), C, H, W)
        memory_spatial_query = self.memory_spatial_summary(combined_mem_feat)
        image_spatial_query = self.image_spatial_summary(image_feat)
        
        memory_spatial_query = memory_spatial_query.reshape(B, (T+1), 8, 64)
        (
            non_cond_bank_feat,
            cond_bank_feat,
            curr_mem_feat
        ) = torch.tensor_split(memory_spatial_query, (self.num_maskmem, 16), dim=1)
        # non_cond_obj_ptr, cond_obj_ptr, _ = torch.tensor_split(bank_ptr, (self.num_maskmem, 16), dim=1)

        non_cond_bank_feat = non_cond_bank_feat.flatten(2)
        cond_bank_feat = cond_bank_feat.flatten(2)
        curr_mem_feat = curr_mem_feat.flatten(2)

        # non_cond_bank_feat = torch.cat([non_cond_bank_feat, non_cond_obj_ptr], dim=-1)
        # cond_bank_feat = torch.cat([cond_bank_feat, cond_obj_ptr], dim=-1)
        # curr_mem_feat = torch.cat([curr_mem_feat, memory_ptr.unsqueeze(1)], dim=-1)
        
        action_query = torch.cat([non_drop_embed, curr_mem_feat, non_cond_bank_feat], dim=1)
        action_context = torch.cat([cond_bank_feat, image_spatial_query], dim=1)
        
        # action_query = action_query + action_query_pos_embed
        # action_context = action_context + action_context_pos_embed
        
        for layer in self.action_decoder:
            action_query = layer(action_query, action_context)
        
        q_values = self.proj_out(action_query)
        return q_values.squeeze(-1)