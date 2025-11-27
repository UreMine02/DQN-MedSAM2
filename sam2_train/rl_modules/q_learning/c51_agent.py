# Deep Q-Learning Agent 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sam2_train.rl_modules.rl_blocks import QuickGELU
from sam2_train.rl_modules.q_learning.base_q_agent import BaseQAgent, BaseQNetwork

# -------------------------
# Q-Network
# -------------------------
class C51QNetwork(BaseQNetwork):
    def __init__(
        self, 
        num_maskmem, 
        n_query=16, 
        image_dim=256, 
        memory_dim=64,
        obj_ptr_dim=256,
        atoms=51, # for C51 variant
    ):
        super().__init__(num_maskmem, n_query, image_dim, memory_dim, obj_ptr_dim)
        
        self.proj_out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            QuickGELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 4, atoms),
        )
        
    def head(self, action_query):
        q_values = self.proj_out(action_query)
        q_values = torch.softmax(q_values, dim=-1)
        return q_values


class C51QAgent(BaseQAgent):
    def __init__(
        self,
        num_maskmem, 
        lr=1e-4, 
        gamma=0.99, 
        beta=0.9995, 
        buffer_size=500, 
        batch_size=32, 
        device="cpu",
        v_min=-10,
        v_max=10,
        atoms=51,
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
        self.q_net = C51QNetwork(num_maskmem, atoms=atoms, **sam2_dim)
        self.target_net = C51QNetwork(num_maskmem, atoms=atoms, **sam2_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = atoms
        self.Z = torch.linspace(self.v_min, self.v_max, self.atoms)
        self.delta_z = (self.v_max - self.v_min) / self.atoms
    
    def to(self, device):
        super().to(device)
        self.Z = self.Z.to(device=device, non_blocking=True)
    
    def transform_q_net_output(self, q_values):
        return (q_values * self.Z).sum(-1).cpu()
    
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
            value_dist = q_net_a(*curr_feats)
            value_dist = value_dist[torch.arange(self.batch_size), actions.squeeze()] # [B,atoms]
            with torch.no_grad():
                next_dist = q_net_b(*next_feats)
                
                # choose actions with max expected values
                B = self.batch_size
                expected_next_values = (next_dist * self.Z).sum(dim=-1)
                next_actions = torch.argmax(expected_next_values, dim=1)
                next_dist = next_dist[torch.arange(B), next_actions] # [B, atoms]
                
                T_Z = rewards + self.gamma * (1 - dones) * self.Z.expand(B, self.atoms)
                # clip the value in range [v_min,v_max]
                clip_T_Z = torch.clip(T_Z, self.v_min, self.v_max)
                
                # next return value postion  bj = (T_z - v_min)/ Dz
                b = (clip_T_Z - self.v_min) / self.delta_z
                
                # l= lower [bj] , u= upper [bj] 
                l = b.floor().clamp(0, self.atoms - 1)
                u = b.ceil().clamp(0, self.atoms - 1)
                
                d_m_l = (u + (l == u).float() - b) * next_dist
                d_m_u = (b - l) * next_dist
                
                # this is the target distribuation  
                target_dist = torch.zeros((B, self.atoms), device=self.device)
                
                for i in range(B):
                    target_dist[i].index_add_(0, l[i].long(), d_m_l[i])
                    target_dist[i].index_add_(0, u[i].long(), d_m_u[i])

            loss = F.cross_entropy(value_dist, target_dist)
            
        return loss