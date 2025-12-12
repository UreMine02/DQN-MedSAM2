# Deep Q-Learning Agent 
import random
import numpy as np
from collections import deque

import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from sam2_train.rl_modules.policy_optimization.base_po_agent import (BasePOAgent, BaseFeatureSummarizer, BasePolicyNetwork)
from sam2_train.rl_modules.rl_components import RLReplayInstance, RLStates

class GRPOReplayInstance(RLReplayInstance):
    def __init__(
        self, 
        frame_idx=None, 
        state=None, 
        action=None,
        reward=None,
        log_probs=None,
    ):
        super().__init__(frame_idx, state, action, None, None, None, reward)
        self.log_probs = log_probs
        
    def get(self):
        # Call tuple to create a copy
        return tuple((self.state, self.log_probs, self.action, self.reward, self.done))
    
    # Update after
    def update(self, loss_after):
        self.loss_after = loss_after
        
        loss_diff = self.loss_before - self.loss_after
        self.reward = self.reward + loss_diff

    def set_done(self, loss_after):
        self.loss_after = loss_after
        
        loss_diff = self.loss_before - self.loss_after
        self.reward = self.reward + loss_diff
        self.done = True
    
class GRPOGroup:
    def __init__(self):
        self.group = []
        
    def add_instance(self, instance):
        self.group.append(instance)
        
    def finalize(self):
        group_rewards = torch.Tensor([ins.reward for ins in self.group])
        # print("rewards before", group_rewards)
        group_mean = group_rewards.mean(dim=0, keepdim=True)
        group_std  = group_rewards.std(dim=0, keepdim=True)
        group_rewards = 0.05 * (group_rewards - group_mean) / (group_std + 1e-8)
        # print("rewards after", group_rewards)
        
        for i, ins in enumerate(self.group):
            ins.reward = group_rewards[i]
    
    def get_instances(self):
        return [ins.get() for ins in self.group]

class GRPOAgent(BasePOAgent):
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
            lr=policy_lr,
            weight_decay=0.01
        )
        
        self.await_group = None
        self.priority = deque(maxlen=buffer_size)
    
    def to(self, device):
        self.device = device
        self.feat_summarizer.to(device=device, non_blocking=True)
        self.policy_net.to(device=device, non_blocking=True)
    
    def init_new_group(self):
        self.await_group = GRPOGroup()
    
    def add_new_instance_to_group(self, **instance_info):
        self.await_group.add_instance(GRPOReplayInstance(**instance_info))
        
    def final_group(self):
        self.await_group.finalize()
        new_normalized_instances = self.await_group.get_instances()
        self.replay_buffer.extend(new_normalized_instances)
        for ins in new_normalized_instances:
            if ins[2] == 0:
                self.priority.append(0.25)
            else:
                self.priority.append(1.0)
        
    def select_action(self, state, valid_actions, num_samples=1, training=False):
        image_feat = state.next_image_feat
        memory_feat = state.curr_memory_feat["mem_feat"]
        memory_ptr = state.curr_memory_feat["obj_ptr"]
        bank_feat = state.prev_memory_bank["mem_feat"]
        bank_ptr = state.prev_memory_bank["obj_ptr"]
        
        state = self.feat_summarizer(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr)
        action_probs = self.policy_net(*state, training=False).cpu().squeeze(0)
        
        valid_actions = torch.Tensor(valid_actions).to(torch.int64)
        valid_probs = action_probs.gather(0, valid_actions)
        
        if training:
            main_action_idx = torch.multinomial(valid_probs, 1)
            action_idx = torch.multinomial(action_probs.squeeze(), min(len(valid_actions), num_samples))
            
            return {
                "main_action": valid_actions[main_action_idx].item(),
                "action": action_idx.tolist(),
                "log_probs": action_probs.log()[action_idx].tolist()
            }
        else:
            action_idx = torch.argmax(valid_probs)
            
            return {
                "main_action": valid_actions[action_idx].item(),
            }
        
    def update(self, num_update):
        if len(self.replay_buffer) < self.batch_size or num_update <= 0:
            return None
        
        print(f"Update agent for {num_update} steps")
        self.feat_summarizer.train()
        self.policy_net.train()
        
        total_policy_loss = 0
        for i in range(num_update):
            batch = random.sample(self.replay_buffer, k=self.batch_size)
            p = np.asanyarray(self.priority)
            p = p / p.sum()
            batch_idx = np.random.choice(len(self.replay_buffer), size=self.batch_size, replace=False, p=p)
            batch = []
            for idx in batch_idx:
                batch.append(self.replay_buffer[idx])

            policy_loss = self.train_step(batch)
            
            total_policy_loss += policy_loss
        
        self.feat_summarizer.eval()
        self.policy_net.eval()
        return {"actor_loss": total_policy_loss / num_update}
    
    def train_step(self, batch, update_value=True):
        device = self.device
        
        states, old_log_probs, actions, rewards,  dones = zip(*batch)
        
        image_feat = torch.cat([state.next_image_feat for state in states])
        memory_feat = torch.cat([state.curr_memory_feat["mem_feat"] for state in states])
        memory_ptr = torch.cat([state.curr_memory_feat["obj_ptr"] for state in states])
        bank_feat = torch.cat([state.prev_memory_bank["mem_feat"] for state in states])
        bank_ptr = torch.cat([state.prev_memory_bank["obj_ptr"] for state in states])
        
        image_feat = image_feat.to(device=device, dtype=torch.float16, non_blocking=True)
        memory_feat = memory_feat.to(device=device, dtype=torch.float16, non_blocking=True)
        memory_ptr = memory_ptr.to(device=device, dtype=torch.float16, non_blocking=True)
        bank_feat = bank_feat.to(device=device, dtype=torch.float16, non_blocking=True)
        bank_ptr = bank_ptr.to(device=device, dtype=torch.float16, non_blocking=True)
        
        actions = torch.LongTensor(actions).unsqueeze(1).to(device=device, non_blocking=True)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device=device, dtype=torch.float16, non_blocking=True)
        old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(1).to(device=device, dtype=torch.float16, non_blocking=True)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device=device, dtype=torch.float16, non_blocking=True)

        with torch.enable_grad():
            curr_feats = self.feat_summarizer(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr)
            
            policy_probs = self.policy_net(*curr_feats)
            action_probs = policy_probs.gather(1, actions)
            log_probs = torch.log(policy_probs)
            log_action_probs = torch.log(action_probs)
            
            policy_loss = self.compute_policy_loss(log_action_probs, rewards, old_log_probs)
            minus_entropy = (policy_probs * log_probs).sum(dim=1, keepdim=True)
            policy_loss += minus_entropy * self.entropy_weight # entropy regularization
            policy_loss = policy_loss.mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_net.parameters()) + \
                list(self.feat_summarizer.parameters()),
                max_norm=0.1
            )
            self.policy_optimizer.step()
            
        return policy_loss.item()
    
    def compute_policy_loss(self, log_prob, advantage, old_log_prob):
        ratio = (log_prob - old_log_prob).exp()
        surr_loss = ratio * advantage
        clipped_surr_loss = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
        policy_loss = -torch.min(surr_loss, clipped_surr_loss)
        return policy_loss
    
    def state_dict(self):
        if isinstance(self.feat_summarizer, DDP):
            return {
                "feat_summarizer": self.feat_summarizer.module.state_dict(),
                "policy_net": self.policy_net.module.state_dict(),
            }
        return {
            "feat_summarizer": self.feat_summarizer.state_dict(),
            "policy_net": self.policy_net.state_dict(),
        }
        
    def load_state_dict(self, state_dict):
        self.feat_summarizer.load_state_dict(state_dict["feat_summarizer"])
        self.policy_net.load_state_dict(state_dict["policy_net"])