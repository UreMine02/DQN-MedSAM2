# Deep Q-Learning Agent 
import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
from collections import deque

from sam2_train.q_learning_blocks import QFormerBlock, SpatialSummarizer


# -------------------------
# Define Action Space
# -------------------------
ACTION_SPACE = {
    0: "skip",
    1: "add",
    2: "add_drop_oldest",
    3: "add_drop_lowest_iou",
    4: "add_drop_random",
}
ACTION_DIM = len(ACTION_SPACE)


# -------------------------
# States
# -------------------------
class RLStates:
    def __init__(
        self,
        frame_idx=None,
        next_image_feat=None,
        curr_memory_feat=None,
        curr_iou_score=None,
        prev_memory_bank=None
    ):
        self.frame_ix = frame_idx
        self.next_image_feat = next_image_feat
        self.curr_memory_feat = curr_memory_feat
        self.curr_iou_score = curr_iou_score
        self.prev_memory_bank = prev_memory_bank
        
    def offload_to_cpu(self):
        if self.next_image_feat is not None:
            self.next_image_feat = self.next_image_feat.detach().cpu()
        if self.curr_memory_feat["mem_feat"] is not None:
            self.curr_memory_feat["mem_feat"] = self.curr_memory_feat["mem_feat"].detach().cpu()
        if self.curr_memory_feat["obj_ptr"] is not None:
            self.curr_memory_feat["obj_ptr"] = self.curr_memory_feat["obj_ptr"].detach().cpu()
        if self.prev_memory_bank["mem_feat"] is not None:
            self.prev_memory_bank["mem_feat"] = self.prev_memory_bank["mem_feat"].detach().cpu()
        if self.prev_memory_bank["obj_ptr"] is not None:
            self.prev_memory_bank["obj_ptr"] = self.prev_memory_bank["obj_ptr"].detach().cpu()
        
        
        
# -------------------------
# Replay Buffer Instance
# -------------------------
class RLReplayInstance:
    def __init__(
        self,
        frame_idx=None,
        state=None,
        action=None,
        next_state=None,
        loss_before=None,
        loss_after=None,
        reward=None,
        eps=1e-8
    ):
        self.frame_ix = frame_idx
        self.state = state
        self.action = action
        self.next_state = next_state
        self.loss_before = loss_before
        self.loss_after = loss_after
        self.reward = reward
        self.eps = eps
        self.done = False
    
    def get(self):
        return tuple((self.state, self.action, self.reward, self.next_state, self.done)) # Call tuple to create a copy
    
    # Update_after
    def update(self, loss_after, next_state):
        self.loss_after = loss_after
        self.next_state = next_state
        
        assert self.loss_before is not None, f"Loss before not available when update loss after for frame {self.frame_ix}"
        
        loss_diff = self.loss_before - self.loss_after
        self.reward = self.reward + loss_diff

    def set_done(self):
        self.done = True


# -------------------------
# Q-Network (MLP)
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, action_dim, num_maskmem, n_query=16, image_dim=256, memory_dim=64):
        super(QNetwork, self).__init__()
        self.num_maskmem = num_maskmem
        self.n_query = n_query
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        
        hidden_dim = n_query * memory_dim
        self.hidden_dim = hidden_dim
        memory_num_head = memory_dim // 64
        
        self.image_spatial_summary = SpatialSummarizer(n_query, hidden_dim, image_dim, n_query, 64, n_layers=4)
        self.memory_spatial_summary = SpatialSummarizer(n_query, memory_dim, memory_dim, memory_num_head, 64, n_layers=4)
        self.action_decoder = nn.ModuleList([QFormerBlock(hidden_dim, hidden_dim, n_query, 64) for _ in range(4)])
        
        self.non_drop_embed = nn.Parameter(torch.rand(1, 1, hidden_dim))
        
        # self.proj_out = nn.Sequential(
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, 1),
        #     nn.Dropout(0.5),
        # )
        
        self.proj_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Dropout(0.2),
        )

    def forward(self, image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr):        
        B, T, C, H, W = bank_feat.shape
        non_drop_embed = self.non_drop_embed.repeat(B, 1, 1)
        
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
        # non_cond_obj_ptr, cond_obj_ptr, _ = torch.tensor_split(bank_ptr, (self.num_maskmem, 16), dim=1)

        non_cond_bank_feat = non_cond_bank_feat.flatten(2)
        cond_bank_feat = cond_bank_feat.flatten(2)
        curr_mem_feat = curr_mem_feat.flatten(2)

        # non_cond_bank_feat = torch.cat([non_cond_bank_feat, non_cond_obj_ptr], dim=-1)
        # cond_bank_feat = torch.cat([cond_bank_feat, cond_obj_ptr], dim=-1)
        # curr_mem_feat = torch.cat([curr_mem_feat, memory_ptr.unsqueeze(1)], dim=-1)
        
        action_query = torch.cat([non_drop_embed, curr_mem_feat, non_cond_bank_feat], dim=1)
        action_context = torch.cat([cond_bank_feat, image_spatial_query], dim=1)
        
        # action_query = action_query + action_query_pos_embed
        # action_context = action_context + action_context_pos_embed
        
        for layer in self.action_decoder:
            action_query = layer(action_query, action_context)
        
        q_values = self.proj_out(action_query)
        return q_values.squeeze(-1)


# -------------------------
# Deep Q-Learning Agent
# -------------------------
class DeepQAgent:
    def __init__(
        self, 
        action_dim, 
        num_maskmem, 
        lr=1e-4, 
        gamma=0.99, 
        beta=0.9995, 
        buffer_size=500, 
        batch_size=32, 
        device="cpu",
        **kwargs
    ):
        self.num_maskmem = num_maskmem
        self.device = device
        self.q_net = QNetwork(action_dim, num_maskmem, **kwargs)
        self.target_net = QNetwork(action_dim, num_maskmem, **kwargs)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.beta = beta
            
        self.await_replay_instance = None
        self.epoch = 0
        self.trained_step = 0
        self.update_target_freq = 10
    
    def init_new_replay_instance(self, **instance_info):
        self.await_replay_instance = RLReplayInstance(**instance_info)
        
    def update_await_replay_instance(self, loss_after, next_state):
        self.await_replay_instance.update(loss_after, next_state)
        self.replay_buffer.append(self.await_replay_instance.get())
        self.await_replay_instance = None
        
    def select_action(self, state: RLStates, valid_actions, bank_size, training=False):
        if training:
            eps = self.beta ** (self.epoch)
            if random.random() < eps:
                # if bank_size < self.num_maskmem:
                #     ACTION_WEIGHTS = [10 / (self.num_maskmem - bank_size)] * (self.num_maskmem + 2)
                #     ACTION_WEIGHTS[1] = 50
                #     for i in range(0, bank_size+2):
                #         if i == 1:
                #             continue
                #         ACTION_WEIGHTS[i] = 40 / (bank_size + 1)
                # else:
                #     ACTION_WEIGHTS = [90 / (self.num_maskmem + 1)] * (self.num_maskmem + 2)
                #     ACTION_WEIGHTS[1] = 10
                # return random.choices(range(self.num_maskmem + 2), weights=ACTION_WEIGHTS)[0]
                return random.choice(valid_actions).item()
        
        image_feat = state.next_image_feat
        memory_feat = state.curr_memory_feat["mem_feat"]
        memory_ptr = state.curr_memory_feat["obj_ptr"]
        bank_feat = state.prev_memory_bank["mem_feat"]
        bank_ptr = state.prev_memory_bank["obj_ptr"]
        
        q_values = self.q_net(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr).cpu()
        argsort = torch.argsort(q_values, dim=1, descending=True)
        valid_argidx = torch.isin(argsort, torch.Tensor(valid_actions)).nonzero()[0, 1]
        return argsort[0, valid_argidx].item()
    
    def to(self, device):
        self.device = device
        self.q_net.to(device=device, non_blocking=True)
        self.target_net.to(device=device, non_blocking=True)

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self, num_update):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        total_loss = 0
        for _ in range(num_update):
            batch = random.sample(self.replay_buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            image_feat = torch.cat([state.next_image_feat for state in states]).to(device=self.device, non_blocking=True)
            memory_feat = torch.cat([state.curr_memory_feat["mem_feat"] for state in states]).to(device=self.device, non_blocking=True)
            memory_ptr = torch.cat([state.curr_memory_feat["obj_ptr"] for state in states]).to(device=self.device, non_blocking=True)
            bank_feat = torch.cat([state.prev_memory_bank["mem_feat"] for state in states]).to(device=self.device, non_blocking=True)
            bank_ptr = torch.cat([state.prev_memory_bank["obj_ptr"] for state in states]).to(device=self.device, non_blocking=True)
            
            next_image_feat = torch.cat([state.next_image_feat for state in next_states]).to(device=self.device, non_blocking=True)
            next_memory_feat = torch.cat([state.curr_memory_feat["mem_feat"] for state in next_states]).to(device=self.device, non_blocking=True)
            next_memory_ptr = torch.cat([state.curr_memory_feat["obj_ptr"] for state in next_states]).to(device=self.device, non_blocking=True)
            next_bank_feat = torch.cat([state.prev_memory_bank["mem_feat"] for state in next_states]).to(device=self.device, non_blocking=True)
            next_bank_ptr = torch.cat([state.prev_memory_bank["obj_ptr"] for state in next_states]).to(device=self.device, non_blocking=True)
            
            actions = torch.LongTensor(actions).unsqueeze(1).to(device=self.device, non_blocking=True)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device=self.device, non_blocking=True)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device=self.device, non_blocking=True)
            
            # if random.random() < 0.5:
                # q_net_a = self.q_net
                # q_net_b = self.target_net
            # else:
                # q_net_a = self.target_net
                # q_net_b = self.q_net
            
            with torch.enable_grad():
                q_values = self.q_net(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr).gather(1, actions)
                with torch.no_grad():
                    max_next_q = self.target_net(
                        next_image_feat,
                        next_memory_feat,
                        next_memory_ptr,
                        next_bank_feat,
                        next_bank_ptr
                    ).max(1, keepdim=True)[0]
                    
                    target_q = rewards + self.gamma * max_next_q * (1 - dones)

                loss_fn = nn.MSELoss()
                loss = loss_fn(q_values, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / num_update

    def update_target(self, distributed):
        if distributed:
            self.target_net.load_state_dict(self.q_net.module.state_dict())
        else:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        
    def set_epoch(self, epoch, distributed=False):
        self.epoch = epoch
        # self.update_target(distributed)
        
    def clear_await(self):
        # self.await_replay_instance.set_done()
        # self.replay_buffer.append(self.await_replay_instance.get())
        self.await_replay_instance = None


def map_action(action, output_dict, storage_key):
    """
    Map an integer action to a drop_key (frame index) for the given storage_key.
    Returns None if no drop should happen (or no candidate to drop).
    Safety: never index into empty lists and prefer per-entry iou if available.
    """
    # get current keys in memory (ordered)
    mem_dict = output_dict.get(storage_key, {})
    sorted_keys = sorted(mem_dict.keys())
    if len(sorted_keys) == 0:
        return None

    # actions:
    # 0: skip
    # 1: add (if full, fallback to drop oldest)
    # 2: add_drop_oldest
    # 3: add_drop_lowest_iou
    # 4: add_drop_random

    # drop oldest (for action 1 or 2)
    if action == 2:
        return sorted_keys[0]

    # drop lowest IoU (action == 3)
    if action == 3:
        # try to collect IoU per stored entry (prefer entry['iou'] or entry['object_score_logits'])
        iou_list = [mem_dict[k]["ious"].item() for k in sorted_keys]
        min_idx = np.argmin(iou_list)
        return sorted_keys[min_idx]

    # drop random (action == 4)
    if action == 4:
        return random.choice(sorted_keys)

    # action == 1
    return None

