# Deep Q-Learning Agent 
import torch
from torch import optim

from sam2_train.rl_modules.policy_optimization.base_po_agent import (BasePOAgent, BaseFeatureSummarizer, BasePolicyNetwork)


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
        
        self.feat_summarizer = BaseFeatureSummarizer(num_maskmem, **sam2_dim)
        self.policy_net = BasePolicyNetwork(self.feat_summarizer.hidden_dim)
        self.value_net = None

        self.policy_optimizer = optim.AdamW(
            list(self.policy_net.parameters()) + \
            list(self.feat_summarizer.parameters()),
            lr=policy_lr
        )
        
    def select_action(self, state, valid_actions, num_samples=1, training=False):
        image_feat = state.next_image_feat
        memory_feat = state.curr_memory_feat["mem_feat"]
        memory_ptr = state.curr_memory_feat["obj_ptr"]
        bank_feat = state.prev_memory_bank["mem_feat"]
        bank_ptr = state.prev_memory_bank["obj_ptr"]
        
        state = self.feat_summarizer(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr)
        action_probs = self.policy_net(*state, training=False).cpu()
        
        valid_actions = torch.Tensor(valid_actions).to(torch.int64)
        valid_probs = action_probs.squeeze(0).gather(0, valid_actions)
        
        if training:
            action_idx = torch.multinomial(valid_probs, min(len(valid_actions), num_samples))
        else:
            action_idx = torch.argmax(valid_probs)
            
        return {
            "action": [valid_actions[idx].item() for idx in action_idx],
            "log_probs": [torch.log(valid_probs[idx]) for idx in action_idx]
        }
    
    def compute_policy_loss(self, log_prob, advantage, old_log_prob):
        ratio = (log_prob - old_log_prob).exp()
        surr_loss = ratio * advantage
        clipped_surr_loss = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
        policy_loss = -torch.min(surr_loss, clipped_surr_loss)
        return policy_loss
        