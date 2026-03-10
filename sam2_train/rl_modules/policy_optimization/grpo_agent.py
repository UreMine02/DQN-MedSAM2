# Deep Q-Learning Agent
import random
import numpy as np
from collections import deque

import torch
import torch.distributed as dist
from torch import optim, nn
from torch.utils.checkpoint import checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributions.categorical import Categorical

from sam2_train.rl_modules.policy_optimization.base_po_agent import (BasePOAgent, BaseFeatureSummarizer, BasePolicyNetwork)
from sam2_train.rl_modules.rl_components import RLReplayInstance, RLStates

from func_3d.misc import MetricLogger

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
    def __init__(self, range=0.05):
        self.range = range
        self.group = []

    def add_instance(self, instance):
        self.group.append(instance)

    def finalize(self):
        group_rewards = torch.Tensor([ins.reward for ins in self.group])

        group_mean = group_rewards.mean(dim=0, keepdim=True)
        group_std  = group_rewards.std(dim=0, keepdim=True)
        group_rewards = self.range * (group_rewards - group_mean) / (group_std + 1e-8)

        for i, ins in enumerate(self.group):
            ins.reward = group_rewards[i]

    def get_instances(self):
        return [ins.get() for ins in self.group]

class GRPOActor(nn.Module):
    def __init__(self, feat_summarizer, policy_net):
        super().__init__()

        self.feat_summarizer = feat_summarizer
        self.policy_net = policy_net

    def forward(self, image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr, training=True, return_logits=False):
        curr_feats = self.feat_summarizer(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr, training=training)
        policy_probs = self.policy_net(*curr_feats, training=training, return_logits=return_logits)
        return policy_probs

class GRPOAgent(BasePOAgent):
    def __init__(
        self,
        num_maskmem,
        policy_lr=0.0001,
        value_lr=0.001,
        gamma=0.99,
        beta=0.9995,
        tau=0.9,
        range=0.05,
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
        self.range = range

        feat_summarizer = BaseFeatureSummarizer(num_maskmem, **sam2_dim, n_layers=4)
        policy_net = BasePolicyNetwork(self.feat_summarizer.hidden_dim, n_layers=4)
        self.value_net = None
        self.actor = GRPOActor(feat_summarizer, policy_net)

        self.policy_optimizer = optim.AdamW(self.actor.parameters(), lr=policy_lr, weight_decay=0.01)

        self.await_group = None

        # For distributed training
        self.rank = 0
        self.distributed = False

    def to(self, device, non_blocking=True):
        self.device = device
        self.actor.to(device=device, non_blocking=non_blocking)

    def to_dtype(self, dtype):
        self.dtype = dtype
        self.actor.to(dtype=dtype)

    def init_new_group(self):
        self.await_group = GRPOGroup(range=self.range)

    def add_new_instance_to_group(self, **instance_info):
        self.await_group.add_instance(GRPOReplayInstance(**instance_info))

    def final_group(self):
        self.await_group.finalize()
        new_normalized_instances = self.await_group.get_instances()
        self.replay_buffer.extend(new_normalized_instances)
    
    @torch.no_grad()
    def select_action(self, state, valid_actions, num_samples=1, training=False):
        self.actor.eval()

        image_feat = state.next_image_feat.detach().to(torch.float32)
        memory_feat = state.curr_memory_feat["mem_feat"].detach().to(torch.float32)
        memory_ptr = state.curr_memory_feat["obj_ptr"].detach().to(torch.float32)
        bank_feat = state.prev_memory_bank["mem_feat"].detach().to(torch.float32)
        bank_ptr = state.prev_memory_bank["obj_ptr"].detach().to(torch.float32)

        action_logits = self.actor(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr, training=training, return_logits=True).squeeze(0)
        action_logits = action_logits.detach().cpu()
        action_dist = Categorical(logits=action_logits)

        valid_actions = torch.Tensor(valid_actions).to(torch.int64)
        valid_dist = Categorical(logits=action_logits.gather(0, valid_actions))
        valid_probs = valid_dist.probs
        
        # if not training:
        #     print({a:p for a, p in zip(valid_actions.tolist(), valid_probs.tolist())})

        if training:
            # main_action_idx = torch.argmax(valid_probs)
            main_action_idx = torch.multinomial(valid_probs, num_samples=1)
            action_idx = torch.multinomial(valid_probs.squeeze(), min(len(valid_actions), num_samples))
            # print("valid actions", valid_actions, action_idx,  valid_actions[action_idx].tolist())
            return {
                "main_action": valid_actions[main_action_idx].item(),
                "action": valid_actions[action_idx].tolist(),
                "log_probs": valid_probs.log()[action_idx].tolist()
            }
        else:
            # action_idx = torch.argmax(valid_probs)
            action_idx = torch.multinomial(valid_probs, num_samples=1)

            return {
                "main_action": valid_actions[action_idx].item(),
            }

    def update(self, num_update):
        local_count = torch.tensor([len(self.replay_buffer)], dtype=torch.long, device=self.rank)
        # print(f"[Rank {self.rank}] local_count {local_count}")
        if self.distributed:
            dist.all_reduce(local_count, op=dist.ReduceOp.MIN)
            # print(local_count)
            
        # if local_count < self.batch_size:
            # return None
        
        if local_count < self.replay_buffer.maxlen:
            return None

        np.random.seed(self.rank + self.epoch * 100)
        self.actor.train()

        device = self.device
        total_policy_loss, total_policy_gradnorm = 0, 0
        for i in range(num_update):
            batch = random.sample(self.replay_buffer, k=self.batch_size)

            states, old_log_probs, actions, rewards, dones = zip(*batch)

            image_feat = torch.cat([state.next_image_feat for state in states]).detach()
            memory_feat = torch.cat([state.curr_memory_feat["mem_feat"] for state in states]).detach()
            memory_ptr = torch.cat([state.curr_memory_feat["obj_ptr"] for state in states]).detach()
            bank_feat = torch.cat([state.prev_memory_bank["mem_feat"] for state in states]).detach()
            bank_ptr = torch.cat([state.prev_memory_bank["obj_ptr"] for state in states]).detach()

            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(1)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            image_feat = image_feat.to(device=device, dtype=torch.float32, non_blocking=True)
            memory_feat = memory_feat.to(device=device, dtype=torch.float32, non_blocking=True)
            memory_ptr = memory_ptr.to(device=device, dtype=torch.float32, non_blocking=True)
            bank_feat = bank_feat.to(device=device, dtype=torch.float32, non_blocking=True)
            bank_ptr = bank_ptr.to(device=device, dtype=torch.float32, non_blocking=True)

            actions = actions.to(device=device, non_blocking=True)
            rewards = rewards.to(device=device, dtype=torch.float32, non_blocking=True)
            old_log_probs = old_log_probs.to(device=device, dtype=torch.float32, non_blocking=True)
            dones = dones.to(device=device, dtype=torch.float32, non_blocking=True)
            
            # with torch.enable_grad():
            policy_logits = self.actor(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr, training=True, return_logits=True)
            policy_dist = Categorical(logits=policy_logits)
            action_probs = policy_dist.probs.gather(1, actions)
            log_action_probs = torch.log(action_probs)
            log_action_probs = policy_dist.log_prob(actions.squeeze(1)).unsqueeze(-1)

            policy_loss = self.compute_policy_loss(log_action_probs, rewards, old_log_probs)
            minus_entropy = -policy_dist.entropy().mean()
            policy_loss = 20 * policy_loss + minus_entropy * self.entropy_weight # entropy regularization

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            gradnorm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.policy_optimizer.step()

            total_policy_loss += policy_loss.detach()
            total_policy_gradnorm += gradnorm

        return {"actor_loss": total_policy_loss / num_update, "actor_gradnorm": total_policy_gradnorm / num_update}

    def compute_policy_loss(self, log_prob, advantage, old_log_prob):
        advantage = advantage.detach()
        old_log_prob = old_log_prob.detach()
        
        ratio = (log_prob - old_log_prob).exp()
        surr_loss = ratio * advantage
        clipped_surr_loss = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
        policy_loss = -torch.mean(torch.min(surr_loss, clipped_surr_loss))
        return policy_loss

    def state_dict(self):
        if isinstance(self.actor, DDP):
            return self.actor.module.state_dict()
        return self.actor.state_dict()

    def load_state_dict(self, state_dict):
        if "feat_summarizer" in state_dict.keys():
            self.actor.feat_summarizer.load_state_dict(state_dict["feat_summarizer"])
            self.actor.policy_net.load_state_dict(state_dict["policy_net"])
        else:
            temp_state_dict = {}
            for k, v in state_dict.items():
                if 'perceiver' in k:
                    k = k.replace('perceiver', 'qformer')
                temp_state_dict[k] = v

            self.actor.load_state_dict(temp_state_dict)

    def to_distributed(self, rank):
        self.distributed = True
        self.rank = rank
        self.actor = DDP(self.actor, device_ids=[rank], output_device=rank)
        print(f"Agent at rank {rank}")

    def num_parameters(self):
        """This function expect modules didn't wrapped by DDP"""
        return sum(p.numel() for p in self.actor.parameters())
