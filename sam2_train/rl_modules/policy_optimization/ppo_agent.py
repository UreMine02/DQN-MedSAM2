# Deep Q-Learning Agent 
import torch

from sam2_train.rl_modules.policy_optimization.base_po_agent import BasePOAgent


class PPOAgent(BasePOAgent):
    def __init__(
        self, 
        num_maskmem, 
        policy_lr=0.0001, 
        value_lr=0.001, 
        gamma=0.99, 
        beta=0.9995,
        tau=0.9,
        buffer_size=500, 
        batch_size=64, 
        device="cpu", 
        entropy_weight=0.1,
        epsilon=0.2,
        sam2_dim={}
    ):
        super().__init__(
            num_maskmem=num_maskmem, 
            policy_lr=policy_lr, 
            value_lr=value_lr, 
            gamma=gamma, 
            beta=beta,
            tau=tau,
            buffer_size=buffer_size, 
            batch_size=batch_size, 
            device=device,
            entropy_weight=entropy_weight,
            sam2_dim=sam2_dim
        )
        self.epsilon = epsilon
    
    def compute_policy_loss(self, log_prob, advantage, old_log_prob):
        ratio = (log_prob - old_log_prob).exp()
        surr_loss = ratio * advantage
        clipped_surr_loss = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
        policy_loss = -torch.min(surr_loss, clipped_surr_loss)
        return policy_loss
        