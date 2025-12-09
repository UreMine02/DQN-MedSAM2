# Deep Q-Learning Agent 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import random
import copy
import numpy as np
from collections import deque
from typing import Optional

from sam2_train.rl_modules.rl_blocks import QFormerBlock, SpatialSummarizer, BidirectionalQFormer, QuickGELU
from sam2_train.rl_modules.rl_components import RLReplayInstance, RLStates
from sam2_train.rl_modules.rl_base_agent import BaseAgent

class BaseQNetwork(nn.Module):
    def __init__(
        self, 
        num_maskmem, 
        n_query=16, 
        image_dim=256, 
        memory_dim=64,
        obj_ptr_dim=256,
    ):
        super(BaseQNetwork, self).__init__()
        self.num_maskmem = num_maskmem
        self.n_query = n_query
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.hidden_dim = n_query * memory_dim + obj_ptr_dim
        
        num_head = self.hidden_dim // 64
        memory_num_head = memory_dim // 64
        
        self.image_spatial_summary = SpatialSummarizer(n_query, self.hidden_dim, image_dim, num_head, 64, n_layers=2, dropout=0.2)
        self.memory_spatial_summary = SpatialSummarizer(n_query, memory_dim, memory_dim, memory_num_head, 64, n_layers=2, dropout=0.2)
        self.action_decoder = nn.ModuleList(
            [QFormerBlock(self.hidden_dim, self.hidden_dim, num_head, 64, dropout=0.2) for _ in range(2)]
        )
        self.non_drop_embed = nn.Parameter(torch.rand(1, 1, self.hidden_dim))

    def forward(self, image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr):        
        B, T, C, H, W = bank_feat.shape
        non_drop_embed = self.non_drop_embed.repeat(B, 1, 1)
        
        combined_mem_feat = torch.cat([bank_feat, memory_feat.unsqueeze(1)], dim=1)
        combined_mem_feat = combined_mem_feat.reshape(B * (T+1), C, H, W)
        memory_spatial_query = self.memory_spatial_summary(combined_mem_feat)
        image_spatial_query = self.image_spatial_summary(image_feat)
        
        memory_spatial_query = memory_spatial_query.reshape(B, (T+1), self.n_query, self.memory_dim)
        (
            non_cond_bank_feat,
            cond_bank_feat,
            curr_mem_feat
        ) = torch.tensor_split(memory_spatial_query, (self.num_maskmem, 16), dim=1)
        non_cond_obj_ptr, cond_obj_ptr, _ = torch.tensor_split(bank_ptr, (self.num_maskmem, 16), dim=1)

        non_cond_bank_feat = non_cond_bank_feat.flatten(2)
        cond_bank_feat = cond_bank_feat.flatten(2)
        curr_mem_feat = curr_mem_feat.flatten(2)

        non_cond_bank_feat = torch.cat([non_cond_bank_feat, non_cond_obj_ptr], dim=-1)
        cond_bank_feat = torch.cat([cond_bank_feat, cond_obj_ptr], dim=-1)
        curr_mem_feat = torch.cat([curr_mem_feat, memory_ptr.unsqueeze(1)], dim=-1)
        
        action_query = torch.cat([non_drop_embed, curr_mem_feat, non_cond_bank_feat], dim=1)
        action_context = torch.cat([cond_bank_feat, image_spatial_query], dim=1)
        
        for layer in self.action_decoder:
            action_query = layer(action_query, action_context)
            
        return self.head(action_query)
        
    def head(self, action_query):
        raise NotImplementedError


class BaseQAgent(BaseAgent):
    def __init__(
        self,
        num_maskmem, 
        lr=1e-4, 
        gamma=0.99, 
        beta=0.9995, 
        buffer_size=500, 
        batch_size=32, 
        device="cpu",
        sam2_dim={}
    ):
        super().__init__(
            num_maskmem, 
            lr=lr,
            gamma=gamma,
            beta=beta,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
        )
        
        self.q_net = None
        self.target_net = None
        
    def state_dict(self):
        if isinstance(self.q_net, DDP):
            return self.q_net.module.state_dict()
        return self.q_net.state_dict()
    
    def load_state_dict(self, state_dict):
        self.q_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        
    def to(self, device):
        self.device = device
        self.q_net.to(device=device, non_blocking=True)
        self.target_net.to(device=device, non_blocking=True)
        
    def select_action(self, state: RLStates, valid_actions, training=False):
        if training:
            eps = self.beta ** (self.epoch)
            if random.random() < eps:
                return {"action": random.choice(valid_actions).item()}
        
        image_feat = state.next_image_feat
        memory_feat = state.curr_memory_feat["mem_feat"]
        memory_ptr = state.curr_memory_feat["obj_ptr"]
        bank_feat = state.prev_memory_bank["mem_feat"]
        bank_ptr = state.prev_memory_bank["obj_ptr"]
        q_values = self.q_net(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr)
        
        q_values = self.transform_q_net_output(q_values)
        
        argsort = torch.argsort(q_values, dim=1, descending=True)
        valid_argidx = torch.isin(argsort.cpu(), torch.Tensor(valid_actions)).nonzero()[0, 1]
            
        return {"action": argsort[0, valid_argidx].item()}
    
    def init_new_trajectory(self):
        pass

    def update(self, num_update):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        total_loss = 0
        for _ in range(num_update):
            batch = random.sample(self.replay_buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            image_feat = torch.cat([state.next_image_feat for state in states]).to(device=self.device, non_blocking=True)
            memory_feat = torch.cat([state.curr_memory_feat["mem_feat"] for state in states]).to(device=self.device, non_blocking=True)
            memory_ptr = torch.cat([state.curr_memory_feat["obj_ptr"] for state in states]).to(device=self.device, non_blocking=True)
            bank_feat = torch.cat([state.prev_memory_bank["mem_feat"] for state in states]).to(device=self.device, non_blocking=True)
            bank_ptr = torch.cat([state.prev_memory_bank["obj_ptr"] for state in states]).to(device=self.device, non_blocking=True)
            
            next_image_feat = torch.cat([state.next_image_feat for state in next_states]).to(device=self.device, non_blocking=True)
            next_memory_feat = torch.cat([state.curr_memory_feat["mem_feat"] for state in next_states]).to(device=self.device, non_blocking=True)
            next_memory_ptr = torch.cat([state.curr_memory_feat["obj_ptr"] for state in next_states]).to(device=self.device, non_blocking=True)
            next_bank_feat = torch.cat([state.prev_memory_bank["mem_feat"] for state in next_states]).to(device=self.device, non_blocking=True)
            next_bank_ptr = torch.cat([state.prev_memory_bank["obj_ptr"] for state in next_states]).to(device=self.device, non_blocking=True)
            
            actions = torch.LongTensor(actions).unsqueeze(1).to(device=self.device, non_blocking=True)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device=self.device, non_blocking=True)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device=self.device, non_blocking=True)
            
            curr_feats = (image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr)
            next_feats = (next_image_feat, next_memory_feat, next_memory_ptr, next_bank_feat, next_bank_ptr)
            
            # Double DQN
            if random.random() < 0.5:
                q_net_a = self.q_net
                q_net_b = self.target_net
            else:
                q_net_a = self.target_net
                q_net_b = self.q_net
                
            loss = self.train_step(
                q_net_a,
                q_net_b,
                curr_feats,
                next_feats,
                actions,
                rewards,
                dones,
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / num_update
    
    def transform_q_net_output(self, q_values):
        return q_values
    
    def train_step(
        self,
        q_net_a,
        q_net_b,
        curr_feats,
        next_feats,
        actions,
        rewards,
        dones,
    ):
        with torch.enable_grad():
            q_values = q_net_a(*curr_feats).gather(1, actions)
            with torch.no_grad():
                max_next_q = q_net_b(*next_feats).max(1, keepdim=True)[0]
                target_q = rewards + self.gamma * max_next_q * (1 - dones)

            loss = F.mse_loss(q_values, target_q)
            
        return loss