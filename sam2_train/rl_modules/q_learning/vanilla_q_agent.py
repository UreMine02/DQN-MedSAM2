# Deep Q-Learning Agent 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import copy
import numpy as np
from collections import deque
from typing import Optional

from sam2_train.rl_modules.q_learning.base_q_agent import BaseQAgent, BaseQNetwork
from sam2_train.rl_modules.rl_components import RLReplayInstance, RLStates


class VanillaQNetwork(BaseQNetwork):
    def __init__(
        self, 
        num_maskmem, 
        n_query=16, 
        image_dim=256, 
        memory_dim=64,
        obj_ptr_dim=256,
    ):
        super().__init__(
            num_maskmem,
            n_query=n_query,
            image_dim=image_dim,
            memory_dim=memory_dim,
            obj_ptr_dim=obj_ptr_dim,
        )
        
        self.proj_out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 1),
        )
        
    def head(self, action_query):
        q_values = self.proj_out(action_query)
        return q_values.squeeze(-1)


class VanillaQAgent(BaseQAgent):
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
        super().__init__(num_maskmem, lr, gamma, beta, buffer_size, batch_size, device)
        self.q_net = VanillaQNetwork(num_maskmem, **sam2_dim)
        self.target_net = VanillaQNetwork(num_maskmem, **sam2_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
    