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

from sam2_train.rl_modules.rl_components import RLStates, RLReplayInstance
from sam2_train.rl_modules.policy_optimization.base_po_agent import BasePOAgent


class PPOAgent(BasePOAgent):
    def __init__(
        self, 
        num_maskmem, 
        policy_lr=0.0001, 
        value_lr=0.001, 
        gamma=0.99, 
        beta=0.9995, 
        buffer_size=500, 
        batch_size=64, 
        device="cpu", 
        entropy_weight=0.1,
        epsilon=0.2,
        sam2_dim={}
    ):
        super().__init__(
            num_maskmem, 
            policy_lr, 
            value_lr, 
            gamma, 
            beta, 
            buffer_size, 
            batch_size, 
            device,
            entropy_weight,
            sam2_dim
        )
        self.epsilon = epsilon
    
    def compute_policy_loss(self, log_prob, advantage, old_log_prob):
        ratio = (log_prob - old_log_prob).exp()
        surr_loss = ratio * advantage
        clipped_surr_loss = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
        policy_loss = -torch.min(surr_loss, clipped_surr_loss)
        return policy_loss
        