import random
from scipy.linalg import circulant
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from sam2_train.rl_modules.rl_components import RLStates, RLReplayInstance
from sam2_train.rl_modules.rl_blocks import (
    QFormerBlock,
    SpatialSummarizer,
    BatchNorm1d,
    BidirectionalQFormer,
    BasicTransformerBlock,
    QuickGELU
)
from sam2_train.rl_modules.rl_base_agent import BaseAgent


def compute_gae(values: torch.Tensor, next_value: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, gamma: float, tau: float):
    """Compute gae."""
    L = values.shape[0]
    device = values.device
    delta = rewards + gamma * (1 - dones) * next_value - values
    
    coef = torch.triu(torch.full((L, L), gamma * tau, device=device))
    l = torch.Tensor(circulant(torch.arange(L))).T.to(device=device, non_blocking=True)
    coef = coef ** l
    
    return coef @ delta + values

class POReplayInstance(RLReplayInstance):
    def __init__(
        self, 
        frame_idx=None, 
        state=None, 
        action=None, 
        next_state=None, 
        loss_before=None, 
        loss_after=None, 
        reward=None, 
        eps=1e-8,
        log_probs=None
    ):
        super().__init__(frame_idx, state, action, next_state, loss_before, loss_after, reward, eps)
        self.log_probs = log_probs
        
    def get(self):
        # Call tuple to create a copy
        return tuple((self.state, self.log_probs, self.action, self.reward, self.next_state, self.done))

class Trajectory:
    def __init__(self):
        self.transitions = []
        
    def add_transition(self, transition):
        self.transitions.append(transition)
        
    def get_transitions(self, device="cpu"):
        states, log_probs, actions, rewards, next_states, dones = zip(*self.transitions)
        
        image_feat = torch.cat([state.next_image_feat for state in states]).to(device=device, non_blocking=True)
        memory_feat = torch.cat([state.curr_memory_feat["mem_feat"] for state in states]).to(device=device, non_blocking=True)
        memory_ptr = torch.cat([state.curr_memory_feat["obj_ptr"] for state in states]).to(device=device, non_blocking=True)
        bank_feat = torch.cat([state.prev_memory_bank["mem_feat"] for state in states]).to(device=device, non_blocking=True)
        bank_ptr = torch.cat([state.prev_memory_bank["obj_ptr"] for state in states]).to(device=device, non_blocking=True)
        
        next_image_feat = torch.cat([state.next_image_feat for state in next_states]).to(device=device, non_blocking=True)
        next_memory_feat = torch.cat([state.curr_memory_feat["mem_feat"] for state in next_states]).to(device=device, non_blocking=True)
        next_memory_ptr = torch.cat([state.curr_memory_feat["obj_ptr"] for state in next_states]).to(device=device, non_blocking=True)
        next_bank_feat = torch.cat([state.prev_memory_bank["mem_feat"] for state in next_states]).to(device=device, non_blocking=True)
        next_bank_ptr = torch.cat([state.prev_memory_bank["obj_ptr"] for state in next_states]).to(device=device, non_blocking=True)
        
        actions = torch.LongTensor(actions).unsqueeze(1).to(device=device, non_blocking=True)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device=device, non_blocking=True)
        log_probs = torch.FloatTensor(log_probs).unsqueeze(1).to(device=device, non_blocking=True)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device=device, non_blocking=True)
        
        curr_feats = (image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr)
        next_feats = (next_image_feat, next_memory_feat, next_memory_ptr, next_bank_feat, next_bank_ptr)
        
        return log_probs, actions, rewards, dones, curr_feats, next_feats

class BaseFeatureSummarizer(nn.Module):
    def __init__(self, num_maskmem, n_query=16, image_dim=256, memory_dim=64, obj_ptr_dim=256):
        super().__init__()
        
        self.num_maskmem = num_maskmem
        self.n_query = n_query
        self.memory_dim = memory_dim
        self.hidden_dim = n_query * memory_dim + obj_ptr_dim
        memory_num_head = memory_dim // 64
        
        self.image_spatial_summary = SpatialSummarizer(n_query, self.hidden_dim, image_dim, self.hidden_dim // 64, 64, n_layers=2)
        self.memory_spatial_summary = SpatialSummarizer(n_query, memory_dim, memory_dim, memory_num_head, 64, n_layers=2)
        
    def forward(self, image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr):
        B, T, C, H, W = bank_feat.shape
        
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
        
        return image_spatial_query, non_cond_bank_feat, cond_bank_feat, curr_mem_feat

class BasePolicyNetwork(nn.Module):
    def __init__(
        self, 
        hidden_dim,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.non_drop_embed = nn.Parameter(torch.rand(1, 1, self.hidden_dim))
        
        self.action_decoder = nn.ModuleList(
            [QFormerBlock(self.hidden_dim, self.hidden_dim, hidden_dim // 64, 64) for _ in range(2)]
        )
        self.action_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            QuickGELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 4, 1)
        )

    def forward(self, image_spatial_query, non_cond_bank_feat, cond_bank_feat, curr_mem_feat):
        B = image_spatial_query.shape[0]
        non_drop_embed = self.non_drop_embed.expand(B, 1, self.hidden_dim)
        
        action_query = torch.cat([non_drop_embed, curr_mem_feat, non_cond_bank_feat], dim=1)
        action_context = torch.cat([cond_bank_feat, image_spatial_query], dim=1)
        
        for layer in self.action_decoder:
            action_query = layer(action_query, action_context)
            
        actions_probs = self.action_proj(action_query)
        actions_probs = torch.softmax(actions_probs, dim=1)
        return actions_probs.squeeze(-1)
    
class BaseValueNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.value_query = nn.Parameter(torch.rand(1, 1, self.hidden_dim))
        
        self.value_decoder = nn.ModuleList(
            [BasicTransformerBlock(self.hidden_dim, 64) for _ in range(2)]
        )
        self.value_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            QuickGELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 4, 1)
        )

    def forward(self, image_spatial_query, non_cond_bank_feat, cond_bank_feat, curr_mem_feat):
        B = image_spatial_query.shape[0]
        value_query = self.value_query.expand(B, 1, self.hidden_dim)
        
        tokens = torch.cat([value_query, curr_mem_feat, non_cond_bank_feat, cond_bank_feat, image_spatial_query], dim=1)
        
        for layer in self.value_decoder:
            tokens = layer(tokens)
        
        return self.value_proj(tokens[:, 0, :])


class BasePOAgent(BaseAgent):
    def __init__(
        self,
        num_maskmem, 
        policy_lr=1e-4,
        value_lr=1e-3,
        gamma=0.99, 
        beta=0.9995,
        tau=0.9,
        buffer_size=500, 
        batch_size=64, 
        device="cpu",
        entropy_weight=0.1,
        sam2_dim={}
    ):
        super().__init__(num_maskmem, policy_lr, gamma, beta, buffer_size, batch_size, device)
        self.feat_summarizer = BaseFeatureSummarizer(num_maskmem, **sam2_dim)
        self.policy_net = BasePolicyNetwork(self.feat_summarizer.hidden_dim)
        self.value_net = BaseValueNetwork(self.feat_summarizer.hidden_dim)

        self.optimizer = optim.AdamW(
            list(self.policy_net.parameters()) + \
            list(self.value_net.parameters()) + \
            list(self.feat_summarizer.parameters()),
            lr=policy_lr)
        
        self.tau = tau
        self.entropy_weight= entropy_weight
    
    def init_new_trajectory(self):
        self.await_trajectory = Trajectory()
        
    def final_trajectory(self):
        self.replay_buffer.append(self.await_trajectory)
        self.await_trajectory = None
    
    def init_new_replay_instance(self, **instance_info):
        self.await_replay_instance = POReplayInstance(**instance_info)    
        
    def update_await_replay_instance(self, loss_after, next_state):
        self.await_replay_instance.update(loss_after, next_state)
        self.await_trajectory.add_transition(self.await_replay_instance.get())
        self.await_replay_instance = None
        
    def clear_await(self, loss_after):
        self.await_replay_instance.set_done(loss_after)
        self.await_trajectory.add_transition(self.await_replay_instance.get())
        self.await_replay_instance = None
        self.final_trajectory()
        
    def select_action(self, state: RLStates, valid_actions, training=False):
        image_feat = state.next_image_feat
        memory_feat = state.curr_memory_feat["mem_feat"]
        memory_ptr = state.curr_memory_feat["obj_ptr"]
        bank_feat = state.prev_memory_bank["mem_feat"]
        bank_ptr = state.prev_memory_bank["obj_ptr"]
        
        state = self.feat_summarizer(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr)
        action_probs = self.policy_net(*state).cpu()
        
        valid_actions = torch.Tensor(valid_actions).to(torch.int64)
        valid_probs = action_probs.squeeze(0).gather(0, valid_actions)
        
        if training:
            action_idx = torch.multinomial(valid_probs, 1)
        else:
            action_idx = torch.argmax(valid_probs)

        return {"action": valid_actions[action_idx].item(), "log_probs": torch.log(valid_probs[action_idx])}
    
    def to(self, device):
        self.device = device
        self.feat_summarizer.to(device=device, non_blocking=True)
        self.policy_net.to(device=device, non_blocking=True)
        self.value_net.to(device=device, non_blocking=True)

    def update(self, num_update):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        self.feat_summarizer.train()
        self.policy_net.train()
        self.value_net.train()
        
        total_policy_loss, total_value_loss = 0, 0
        for _ in range(num_update):
            batch = random.sample(self.replay_buffer, self.batch_size)

            policy_loss, value_loss = self.train_step(batch)
            
            total_policy_loss += policy_loss
            total_value_loss += value_loss
        
        self.feat_summarizer.eval()
        self.policy_net.eval()
        self.value_net.eval()
        return (total_policy_loss + total_value_loss) / num_update
        
    def train_step(self, batch):
        curr_states = []
        actions = []
        rewards = []
        dones = []
        returns = []
        old_log_probs = []
        advantages = []
        for trajectory in batch:
            log_probs, action, reward, done, curr_state, next_state = trajectory.get_transitions(self.device)

            with torch.no_grad():
                curr_feat = self.feat_summarizer(*curr_state)
                curr_value = self.value_net(*curr_feat)
                next_feat = self.feat_summarizer(*next_state)
                next_value = self.value_net(*next_feat)
                
                return_ = compute_gae(curr_value, next_value, reward, done, self.gamma, self.tau)
            
            curr_states.append(curr_state)
            old_log_probs.append(log_probs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            returns.append(return_)
            advantages.append(return_ - curr_value)

        (
            image_feat,
            memory_feat,
            memory_ptr,
            bank_feat,
            bank_ptr,
        ) = zip(*curr_states)
        
        image_feat = torch.cat(image_feat).to(device=self.device, non_blocking=True)
        memory_feat = torch.cat(memory_feat).to(device=self.device, non_blocking=True)
        memory_ptr = torch.cat(memory_ptr).to(device=self.device, non_blocking=True)
        bank_feat = torch.cat(bank_feat).to(device=self.device, non_blocking=True)
        bank_ptr = torch.cat(bank_ptr).to(device=self.device, non_blocking=True)
        
        old_log_probs = torch.cat(old_log_probs).to(device=self.device, non_blocking=True)
        actions = torch.cat(actions).to(device=self.device, non_blocking=True)
        rewards = torch.cat(rewards).to(device=self.device, non_blocking=True)
        dones = torch.cat(dones).to(device=self.device, non_blocking=True)
        returns = torch.cat(returns).to(device=self.device, non_blocking=True)
        advantages = torch.cat(advantages).to(device=self.device, non_blocking=True)
        
        with torch.enable_grad():
            curr_feats = self.feat_summarizer(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr)
            
            action_probs = self.policy_net(*curr_feats).gather(1, actions)
            log_probs = torch.log(action_probs)
            
            policy_loss = self.compute_policy_loss(log_probs, advantages, old_log_probs)
            policy_loss += -log_probs * self.entropy_weight # entropy regularization
            policy_loss = policy_loss.mean()
            
            pred_value = self.value_net(*curr_feats)
            value_loss = F.smooth_l1_loss(pred_value, returns)

            total_loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # # update policy
            # self.policy_optimizer.zero_grad()
            # policy_loss.backward(retain_graph=True)
            # self.policy_optimizer.step()

            # # update value
            # self.value_optimizer.zero_grad()
            # value_loss.backward()
            # self.value_optimizer.step()
            
        return value_loss.item(), policy_loss.item()
    
    def compute_policy_loss(self, log_prob, advantage, old_log_prob):
        return - (advantage * log_prob)
            
    def state_dict(self):
        if isinstance(self.feat_summarizer, DDP):
            return {
                "feat_summarizer": self.feat_summarizer.module.state_dict(),
                "policy_net": self.policy_net.module.state_dict(),
                "value_net": self.value_net.module.state_dict(),
            }
        return {
            "feat_summarizer": self.feat_summarizer.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
        }
        
    def load_state_dict(self, state_dict):
        self.feat_summarizer.load_state_dict(state_dict["feat_summarizer"])
        self.policy_net.load_state_dict(state_dict["policy_net"])
        self.value_net.load_state_dict(state_dict["value_net"])