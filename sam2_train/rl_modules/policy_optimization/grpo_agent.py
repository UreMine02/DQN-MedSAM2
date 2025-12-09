# Deep Q-Learning Agent 
import torch
from torch import optim

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
        group_mean = group_rewards.mean(dim=0, keepdim=True)
        group_std  = group_rewards.std(dim=0, keepdim=True)
        group_rewards = (group_rewards - group_mean) / group_std
        
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
            lr=policy_lr
        )
        
        self.await_group = None
    
    def init_new_group(self):
        self.await_group = GRPOGroup()
    
    def add_new_instance_to_group(self, **instance_info):
        self.await_group.add_instance(GRPOReplayInstance(**instance_info))
        
    def final_group(self):
        self.await_group.finalize()
        self.replay_buffer.extend(self.await_group.get_instances())
        
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
    
    def train_step(self, batch, update_value=True):
        device = self.device
        
        states, old_log_probs, actions, rewards, next_states, dones, returns, advantages = zip(*batch)
        
        image_feat = torch.cat([state.next_image_feat for state in states]).to(device=device, non_blocking=True)
        memory_feat = torch.cat([state.curr_memory_feat["mem_feat"] for state in states]).to(device=device, non_blocking=True)
        memory_ptr = torch.cat([state.curr_memory_feat["obj_ptr"] for state in states]).to(device=device, non_blocking=True)
        bank_feat = torch.cat([state.prev_memory_bank["mem_feat"] for state in states]).to(device=device, non_blocking=True)
        bank_ptr = torch.cat([state.prev_memory_bank["obj_ptr"] for state in states]).to(device=device, non_blocking=True)
        
        actions = torch.LongTensor(actions).unsqueeze(1).to(device=device, non_blocking=True)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device=device, non_blocking=True)
        old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(1).to(device=device, non_blocking=True)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device=device, non_blocking=True)
        
        adv_mean = advantages.mean(dim=0, keepdim=True)
        adv_std = advantages.std(dim=0, keepdim=True)
        advantages = 0.7 * (advantages - adv_mean) / adv_std
        
        # print("advantages", advantages.mean())
        
        with torch.enable_grad():
            curr_feats = self.feat_summarizer(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr)
            
            policy_probs = self.policy_net(*curr_feats)
            action_probs = policy_probs.gather(1, actions)
            log_probs = torch.log(policy_probs)
            log_action_probs = torch.log(action_probs)
            
            policy_loss = self.compute_policy_loss(log_action_probs, advantages, old_log_probs)
            minus_entropy = (policy_probs * log_probs).sum(dim=1, keepdim=True)
            policy_loss += minus_entropy * self.entropy_weight # entropy regularization
            policy_loss = policy_loss.mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # total_loss = policy_loss + 0.1 * value_loss
            
            # self.optimizer.zero_grad()
            # total_loss.backward()
            # self.optimizer.step()
            
            # print("loss", policy_loss, value_loss, minus_entropy.mean())
            
        return policy_loss.item()
    
    def compute_policy_loss(self, log_prob, advantage, old_log_prob):
        ratio = (log_prob - old_log_prob).exp()
        surr_loss = ratio * advantage
        clipped_surr_loss = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
        policy_loss = -torch.min(surr_loss, clipped_surr_loss)
        return policy_loss
        