import random
import numpy as np
from scipy.linalg import circulant
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from sam2_train.rl_modules.rl_components import RLStates, RLReplayInstance
from sam2_train.rl_modules.rl_blocks import (
    QFormerBlock,
    SpatialSummarizer,
    BatchNorm1d,
    BidirectionalQFormer,
    BasicTransformerBlock,
    QuickGELU,
    PerceiverResampler
)
from sam2_train.rl_modules.rl_base_agent import BaseAgent

def compute_gae(values: torch.Tensor, next_value: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, gamma: float, tau: float):
    """Compute gae."""
    L = values.shape[0]
    device = values.device
    delta = rewards + gamma * (1 - dones) * next_value - values # [L,1]

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
        log_probs=None,
        advantage=None,
        return_=None
    ):
        super().__init__(frame_idx, state, action, next_state, loss_before, loss_after, reward, eps)
        self.log_probs = log_probs
        self.advantage = advantage
        self.return_ = return_

    def get(self):
        # Call tuple to create a copy
        return tuple((self.state, self.log_probs, self.action, self.reward, self.next_state, self.done))

    def get_updated(self):
        # Call tuple to create a copy
        return tuple((
            self.state,
            self.log_probs,
            self.action,
            self.reward,
            self.next_state,
            self.done,
            self.return_,
            self.advantage
        ))

    def set_return_advantage(self, return_, advantage):
        self.return_ = return_
        self.advantage = advantage

class Trajectory:
    def __init__(self):
        self.transitions = []

    def add_transition(self, transition):
        self.transitions.append(transition)

    def get_transitions(self, device="cpu"):
        transitions = [trans.get() for trans in self.transitions]
        states, log_probs, actions, rewards, next_states, dones = zip(*transitions)

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
    def __init__(self, num_maskmem, n_query=16, image_dim=256, memory_dim=64, obj_ptr_dim=256, n_layers=4):
        super().__init__()

        self.num_maskmem = num_maskmem
        self.n_query = n_query
        self.memory_dim = memory_dim
        memory_num_head = memory_dim // 64
        self.hidden_dim = image_dim

        self.image_spatial_summary = SpatialSummarizer(
            n_query=n_query,
            query_dim=self.hidden_dim,
            spatial_dim=image_dim,
            n_heads=1,
            d_heads=image_dim,
            n_layers=n_layers,
            dropout=0.1
        )
        self.memory_spatial_summary = SpatialSummarizer(
            n_query=n_query,
            query_dim=memory_dim,
            spatial_dim=memory_dim,
            n_heads=1,
            d_heads=memory_dim,
            n_layers=n_layers,
            dropout=0.1
        )
        
        self.cond_mem_proj = nn.Linear(memory_dim, image_dim)
        self.cond_obj_proj = nn.Linear(obj_ptr_dim, image_dim)
        self.non_cond_proj = nn.Linear(n_query * memory_dim + obj_ptr_dim, image_dim)

    def forward(self, image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr, training=True):
        B, T, C, H, W = bank_feat.shape

        combined_mem_feat = torch.cat([bank_feat, memory_feat.unsqueeze(1)], dim=1)
        combined_mem_feat = combined_mem_feat.reshape(B * (T+1), C, H, W)
        memory_spatial_query = self.memory_spatial_summary(combined_mem_feat, training=training)
        image_spatial_query = self.image_spatial_summary(image_feat, training=training)

        memory_spatial_query = memory_spatial_query.reshape(B, (T+1), self.n_query, self.memory_dim)
        (
            non_cond_bank_feat,
            cond_bank_feat,
            curr_mem_feat
        ) = torch.tensor_split(memory_spatial_query, indices=(self.num_maskmem, -1), dim=1)
        non_cond_obj_ptr, cond_obj_ptr = torch.tensor_split(bank_ptr, indices=(self.num_maskmem,), dim=1)

        non_cond_bank_feat = non_cond_bank_feat.flatten(2)
        curr_mem_feat = curr_mem_feat.flatten(2)

        non_cond_bank_feat = torch.cat([non_cond_bank_feat, non_cond_obj_ptr], dim=-1)
        curr_mem_feat = torch.cat([curr_mem_feat, memory_ptr.unsqueeze(1)], dim=-1)
        
        cond_bank_feat = self.cond_mem_proj(cond_bank_feat)
        cond_obj_ptr = self.cond_obj_proj(cond_obj_ptr)
        cond_bank_feat = torch.flatten(cond_bank_feat, start_dim=1, end_dim=2)
        
        cond_bank_feat = torch.cat([cond_bank_feat, cond_obj_ptr], dim=1)
        
        non_cond_bank_feat = self.non_cond_proj(non_cond_bank_feat)
        curr_mem_feat = self.non_cond_proj(curr_mem_feat)

        return image_spatial_query, non_cond_bank_feat, cond_bank_feat, curr_mem_feat

class BasePolicyNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_layers=1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.non_drop_embed = nn.Parameter(torch.rand(1, 1, self.hidden_dim))
        self.action_decoder = nn.ModuleList(
            [PerceiverResampler(self.hidden_dim, num_heads=1, dropout=0.1) for _ in range(n_layers)]
        )
        
        self.action_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, image_spatial_query, non_cond_bank_feat, cond_bank_feat, curr_mem_feat, training=True):
        B = image_spatial_query.shape[0]
        non_drop_embed = self.non_drop_embed.expand(B, 1, self.hidden_dim)

        action_query = torch.cat([non_drop_embed, curr_mem_feat, non_cond_bank_feat], dim=1)
        action_context = torch.cat([cond_bank_feat, image_spatial_query], dim=1)

        for layer in self.action_decoder:
            action_query = layer(x_f=action_context, x=action_query, training=training)

        actions_logits = self.action_proj(action_query)
        actions_probs = torch.softmax(actions_logits, dim=1)
        
        # if not training:
        #     print(actions_logits.squeeze())
        #     print(actions_probs.squeeze())

        return actions_probs.squeeze(-1)

class BaseValueNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_layers=4
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.value_query = nn.Parameter(torch.rand(1, 1, self.hidden_dim))
        self.value_decoder = nn.ModuleList(
            [PerceiverResampler(self.hidden_dim, 1, dropout=0.1) for _ in range(n_layers)]
        )
        
        self.value_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, image_spatial_query, non_cond_bank_feat, cond_bank_feat, curr_mem_feat, training=True):
        B = image_spatial_query.shape[0]
        value_query = self.value_query.expand(B, 1, self.hidden_dim)

        tokens = torch.cat([curr_mem_feat, non_cond_bank_feat, cond_bank_feat, image_spatial_query], dim=1)

        for layer in self.value_decoder:
            value_query = layer(x_f=tokens, x=value_query, training=training)

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
        self.feat_summarizer = BaseFeatureSummarizer(num_maskmem, **sam2_dim, n_layers=4)
        self.policy_net = BasePolicyNetwork(self.feat_summarizer.hidden_dim, n_layers=4)
        self.value_net = BaseValueNetwork(self.feat_summarizer.hidden_dim, n_layers=4)

        self.policy_optimizer = optim.AdamW(
            list(self.policy_net.parameters()) + \
            list(self.feat_summarizer.parameters()),
            lr=policy_lr,
            # weight_decay=0.05,
            fused=True
        )
        self.value_optimizer = optim.AdamW(
            list(self.value_net.parameters()),
            lr=value_lr,
            # weight_decay=0.05,
            fused=True
        )

        self.tau = tau
        self.entropy_weight= entropy_weight

        # For distributed training
        self.rank = 0
        self.distributed = False

    def freeze(self):
        for param in self.feat_summarizer.parameters():
            param.requires_grad_(False)
        for param in self.policy_net.parameters():
            param.requires_grad_(False)
        for param in self.value_net.parameters():
            param.requires_grad_(False)

    def init_new_trajectory(self):
        self.await_trajectory = Trajectory()

    def final_trajectory(self):
        log_probs, action, reward, done, curr_state, next_state = self.await_trajectory.get_transitions(self.device)

        with torch.no_grad():
            curr_feat = self.feat_summarizer(*curr_state)
            curr_value = self.value_net(*curr_feat)
            next_feat = self.feat_summarizer(*next_state)
            next_value = self.value_net(*next_feat)

            return_ = compute_gae(curr_value, next_value, reward, done, self.gamma, self.tau)
            return_ = return_.squeeze(-1)
            advantage = return_ - curr_value.squeeze(-1)

        for i, ins in enumerate(self.await_trajectory.transitions):
            ins.set_return_advantage(return_[i].cpu(), advantage[i].cpu())
            self.replay_buffer.append(ins.get_updated())

        self.await_trajectory = None

    def init_new_replay_instance(self, **instance_info):
        self.await_replay_instance = POReplayInstance(**instance_info)

    def update_await_replay_instance(self, loss_after, next_state):
        self.await_replay_instance.update(loss_after, next_state)
        self.await_trajectory.add_transition(self.await_replay_instance)
        self.await_replay_instance = None

    def clear_await(self, loss_after):
        if self.await_replay_instance:
            self.await_replay_instance.set_done(loss_after)
            self.await_trajectory.add_transition(self.await_replay_instance)
            self.await_replay_instance = None
            self.final_trajectory()

    @torch.no_grad()
    def select_action(self, state: RLStates, valid_actions, training=False):
        self.feat_summarizer.eval()
        self.policy_net.eval()
        self.value_net.eval()

        image_feat = state.next_image_feat.detach().to(torch.float32)
        memory_feat = state.curr_memory_feat["mem_feat"].detach().to(torch.float32)
        memory_ptr = state.curr_memory_feat["obj_ptr"].detach().to(torch.float32)
        bank_feat = state.prev_memory_bank["mem_feat"].detach().to(torch.float32)
        bank_ptr = state.prev_memory_bank["obj_ptr"].detach().to(torch.float32)

        state = self.feat_summarizer(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr)
        action_probs = self.policy_net(*state, training=training).detach().cpu()

        valid_actions = torch.Tensor(valid_actions).to(torch.int64)
        valid_probs = action_probs.squeeze(0).gather(0, valid_actions)

        if training:
            action_idx = torch.multinomial(valid_probs, num_samples=1, replacement=False)
        else:
            action_idx = torch.argmax(valid_probs)

        return {"action": valid_actions[action_idx].item(), "log_probs": torch.log(valid_probs[action_idx])}

    def to(self, device, non_blocking=False):
        self.device = device
        self.feat_summarizer.to(device=device, non_blocking=non_blocking)
        self.policy_net.to(device=device, non_blocking=non_blocking)
        self.value_net.to(device=device, non_blocking=non_blocking)

    def to_dtype(self, dtype):
        self.dtype = dtype
        self.feat_summarizer.to(dtype=dtype)
        self.policy_net.to(dtype=dtype)
        self.value_net.to(dtype=dtype)

    def update(self, num_update):
        local_count = torch.tensor([len(self.replay_buffer)], dtype=torch.long, device=self.rank)
        if self.distributed:
            dist.all_reduce(local_count, op=dist.ReduceOp.MIN)

        if local_count < self.batch_size or num_update <= 0:
            return None

        np.random.seed(self.rank + self.epoch * 100)

        # print(f"Update agent for {num_update} steps")
        self.feat_summarizer.train()
        self.policy_net.train()
        self.value_net.train()

        total_policy_loss, total_value_loss, total_actor_gradnorm, total_critic_gradnorm = 0, 0, 0, 0
        critic_num_update = 0
        for i in range(num_update):
            batch = random.sample(self.replay_buffer, k=self.batch_size)

            # n_actions = {}
            # for sample in self.replay_buffer:
            #     action = sample[2]
            #     if action not in n_actions.keys():
            #         n_actions[action] = 0
            #     n_actions[action] += 1

            # p = []
            # for sample in self.replay_buffer:
            #     p.append(len(self.replay_buffer) / n_actions[sample[2]])

            # p = np.asanyarray(p)
            # p = p / p.sum()
            # batch_idx = np.random.choice(len(self.replay_buffer), size=self.batch_size, replace=False, p=p)
            # batch = []
            # for idx in batch_idx:
            #     batch.append(self.replay_buffer[idx])

            update_value = i % 2
            value_loss, policy_loss, actor_gradnorm, critic_gradnorm = self.train_step(batch, update_value=update_value)

            total_policy_loss += policy_loss
            total_actor_gradnorm += actor_gradnorm

            if update_value:
                total_value_loss += value_loss
                total_critic_gradnorm += critic_gradnorm
                critic_num_update += 1

        return {
            "actor_loss": total_policy_loss / num_update, 
            "critic_loss": total_value_loss / critic_num_update,
            "actor_gradnorm": total_actor_gradnorm / num_update, 
            "critic_gradnorm": total_critic_gradnorm / critic_num_update,
        }

    def train_step(self, batch, update_value=True):
        device = self.device

        states, old_log_probs, actions, rewards, next_states, dones, returns, advantages = zip(*batch)

        image_feat = torch.cat([state.next_image_feat for state in states]).detach()
        memory_feat = torch.cat([state.curr_memory_feat["mem_feat"] for state in states]).detach()
        memory_ptr = torch.cat([state.curr_memory_feat["obj_ptr"] for state in states]).detach()
        bank_feat = torch.cat([state.prev_memory_bank["mem_feat"] for state in states]).detach()
        bank_ptr = torch.cat([state.prev_memory_bank["obj_ptr"] for state in states]).detach()

        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        advantages = torch.FloatTensor(advantages).unsqueeze(1)
        returns = torch.FloatTensor(returns).unsqueeze(1)

        image_feat = image_feat.to(device=device, dtype=torch.float32, non_blocking=True)
        memory_feat = memory_feat.to(device=device, dtype=torch.float32, non_blocking=True)
        memory_ptr = memory_ptr.to(device=device, dtype=torch.float32, non_blocking=True)
        bank_feat = bank_feat.to(device=device, dtype=torch.float32, non_blocking=True)
        bank_ptr = bank_ptr.to(device=device, dtype=torch.float32, non_blocking=True)

        actions = actions.to(device=device, non_blocking=True)
        rewards = rewards.to(device=device, dtype=torch.float32, non_blocking=True)
        old_log_probs = old_log_probs.to(device=device, dtype=torch.float32, non_blocking=True)
        dones = dones.to(device=device, dtype=torch.float32, non_blocking=True)
        advantages = advantages.to(device=device, dtype=torch.float32, non_blocking=True)
        returns = returns.to(device=device, dtype=torch.float32, non_blocking=True)

        adv_mean = advantages.mean(dim=0, keepdim=True)
        adv_std = advantages.std(dim=0, keepdim=True)
        advantages = 0.5 * (advantages - adv_mean) / adv_std

        with torch.enable_grad():
            (
                image_spatial_query,
                non_cond_bank_feat,
                cond_bank_feat,
                curr_mem_feat
            ) = self.feat_summarizer(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr)

            policy_probs = self.policy_net(
                image_spatial_query,
                non_cond_bank_feat,
                cond_bank_feat,
                curr_mem_feat
            )
            action_probs = policy_probs.gather(1, actions)
            log_probs = torch.log(policy_probs)
            log_action_probs = torch.log(action_probs)

            policy_loss = self.compute_policy_loss(log_action_probs, advantages, old_log_probs)
            minus_entropy = (policy_probs * log_probs).sum(dim=1, keepdim=True)
            policy_loss += minus_entropy * self.entropy_weight # entropy regularization
            policy_loss = policy_loss.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            actor_gradnorm = nn.utils.clip_grad_norm_(
                list(self.feat_summarizer.parameters()) + list(self.policy_net.parameters()),
                max_norm=0.5
            )
            self.policy_optimizer.step()

            if update_value:
                image_spatial_query = image_spatial_query.detach()
                non_cond_bank_feat = non_cond_bank_feat.detach()
                cond_bank_feat = cond_bank_feat.detach()
                curr_mem_feat = curr_mem_feat.detach()

                pred_value = self.value_net(
                    image_spatial_query,
                    non_cond_bank_feat,
                    cond_bank_feat,
                    curr_mem_feat
                )
                value_loss = F.smooth_l1_loss(pred_value, returns)

                self.value_optimizer.zero_grad()
                value_loss.backward()
                critic_gradnorm = nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
                self.value_optimizer.step()
            else:
                value_loss = torch.Tensor([0])
                critic_gradnorm = 0

            # total_loss = policy_loss + 0.1 * value_loss

            # self.optimizer.zero_grad()
            # total_loss.backward()
            # self.optimizer.step()

            # print("loss", policy_loss, value_loss, minus_entropy.mean())

        return value_loss.detach(), policy_loss.detach(), actor_gradnorm, critic_gradnorm

    def compute_policy_loss(self, log_prob, advantage, old_log_prob):
        return -(advantage * log_prob)

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

    def to_distributed(self, rank):
        self.distributed = True
        self.rank = rank
        self.feat_summarizer = DDP(self.feat_summarizer, device_ids=[rank], output_device=rank)
        self.policy_net = DDP(self.policy_net, device_ids=[rank], output_device=rank)
        self.value_net = DDP(self.value_net, device_ids=[rank], output_device=rank)

    def num_parameters(self):
        """This function expect modules didn't wrapped by DDP"""
        return sum(p.numel() for p in self.feat_summarizer.parameters()) + \
                sum(p.numel() for p in self.policy_net.parameters()) + \
                sum(p.numel() for p in self.value_net.parameters())

