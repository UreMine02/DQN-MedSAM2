import torch.nn as nn
import torch.optim as optim

from sam2_train.rl_modules.q_learning.base_q_agent import BaseQAgent, BaseQNetwork


class DuelingQNetwork(BaseQNetwork):
    def __init__(
        self, 
        num_maskmem, 
        n_query=16, 
        image_dim=256, 
        memory_dim=64,
        obj_ptr_dim=256,
    ):
        super().__init__(num_maskmem, n_query, image_dim, memory_dim, obj_ptr_dim)
        
        self.value_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 1),
        )
        self.advantage_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 1),
        )
        
    def head(self, action_query):
        values = self.value_proj(action_query.mean(dim=1, keepdim=True))
        advantages = self.advantage_proj(action_query)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values.squeeze(-1)


class DuelingQAgent(BaseQAgent):
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
        
        self.q_net = DuelingQNetwork(num_maskmem, **sam2_dim)
        self.target_net = DuelingQNetwork(num_maskmem, **sam2_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)