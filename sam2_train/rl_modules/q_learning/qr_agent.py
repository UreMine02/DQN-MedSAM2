import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sam2_train.rl_modules.q_learning.base_q_agent import BaseQAgent, BaseQNetwork

class QRQNetwork(BaseQNetwork):
    def __init__(
        self, 
        num_maskmem, 
        n_query=16, 
        image_dim=256, 
        memory_dim=64,
        obj_ptr_dim=256,
        N=100 # for QR-DQN
    ):
        super().__init__(num_maskmem, n_query, image_dim, memory_dim, obj_ptr_dim)
        
        self.logit_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, N),
        )
        self.scale_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 2),
        )

        
    def head(self, action_query):
        quantile_logits = F.softmax(self.logit_proj(action_query), dim=-1) # [B,A,N]
        quantile_scale = self.scale_proj(action_query) # [B,A,2]
        
        quantile_logits = torch.cumsum(quantile_logits, dim=-1)
        alpha, beta, _ = torch.tensor_split(quantile_scale, (1, 2), dim=2)
        alpha = F.relu(alpha)
        
        q_values = alpha * quantile_logits + beta
        return q_values


class QRQAgent(BaseQAgent):
    def __init__(
        self,
        num_maskmem, 
        lr=1e-4, 
        gamma=0.99, 
        beta=0.9995, 
        buffer_size=500, 
        batch_size=32, 
        device="cpu",
        N=100,
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
        self.q_net = QRQNetwork(num_maskmem, N=N, **sam2_dim)
        self.target_net = QRQNetwork(num_maskmem, N=N, **sam2_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        self.N = N
    
    def transform_q_net_output(self, q_values):
        return q_values.mean(dim=-1)
    
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
            curr_quantiles = q_net_a(*curr_feats)
            curr_quantiles = curr_quantiles[torch.arange(self.batch_size), actions.squeeze()] # [B,N]
            with torch.no_grad():
                next_quantiles = q_net_b(*next_feats)
                
                # choose actions with max expected values
                B = self.batch_size
                next_actions = next_quantiles.mean(dim=2, keepdim=True).argmax(dim=1, keepdim=True)
                next_quantiles = next_quantiles[torch.arange(B), next_actions] # [B,N]
                
                target_quantiles = rewards + self.gamma * (1 - dones) * next_quantiles
                
            N = self.N
            cum_prob = (torch.arange(N, device=curr_quantiles.device, dtype=torch.float) + 0.5) / N
            cum_prob = cum_prob.view(1, -1, 1)

            pairwise_delta = target_quantiles.unsqueeze(-2) - curr_quantiles.unsqueeze(-1)
            abs_pairwise_delta = torch.abs(pairwise_delta)
            huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)
            loss = torch.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss
            loss = loss.sum(dim=-2).mean()
            
        return loss