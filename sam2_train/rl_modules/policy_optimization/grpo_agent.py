# Deep Q-Learning Agent
import random
import time
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
        action_mask=None,
    ):
        super().__init__(frame_idx, state, action, None, None, None, reward)
        self.log_probs = log_probs
        # Actions that were selectable when this sample was drawn, so the update can
        # renormalize over the same support the log prob above was computed on.
        self.action_mask = action_mask

    def get(self):
        # Call tuple to create a copy
        return tuple((self.state, self.log_probs, self.action, self.reward, self.done, self.action_mask))

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
    def __init__(self, range=0.05, min_std=1e-3):
        self.range = range
        self.min_std = min_std
        self.group = []

    def add_instance(self, instance):
        self.group.append(instance)

    def finalize(self):
        """Turn raw rewards into group-relative advantages.

        Returns False when the group carries no usable signal, i.e. its rewards are
        (nearly) identical. Dividing by the group std turns any spread, however small,
        into unit-scale advantages, so a group whose actions make no real difference
        would otherwise be trained on as confidently as one where they do. Such a group
        is left untouched and the caller drops it.
        """
        if len(self.group) < 2:
            return False

        group_rewards = torch.tensor([float(ins.reward) for ins in self.group])

        group_mean = group_rewards.mean(dim=0, keepdim=True)
        group_std  = group_rewards.std(dim=0, keepdim=True)
        # `not >=` rather than `<` so a NaN std is rejected too.
        if not (group_std.item() >= self.min_std):
            return False

        group_rewards = self.range * (group_rewards - group_mean) / group_std

        for i, ins in enumerate(self.group):
            ins.reward = group_rewards[i]
        return True

    def get_instances(self):
        return [ins.get() for ins in self.group]

class GRPOActor(nn.Module):
    def __init__(self, feat_summarizer, policy_net):
        super().__init__()

        self.feat_summarizer = feat_summarizer
        self.policy_net = policy_net

    def forward(self, image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr):
        # curr_feats ends with the padding masks, which policy_net takes positionally.
        curr_feats = self.feat_summarizer(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr)
        policy_probs = self.policy_net(*curr_feats)
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
        min_group_std=1e-3,
        buffer_size=500,
        batch_size=64,
        device="cpu",
        entropy_weight=0.1,
        epsilon=0.2,
        lr_T_max=1000,
        min_lr=1e-6,
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
            lr_T_max=lr_T_max,
            min_lr=min_lr,
            sam2_dim=sam2_dim
        )
        self.epsilon = epsilon
        self.range = range
        # Groups whose raw reward std (in dice units) is below this are dropped.
        self.min_group_std = min_group_std

        feat_summarizer = BaseFeatureSummarizer(num_maskmem, **sam2_dim, n_layers=4)
        policy_net = BasePolicyNetwork(self.feat_summarizer.hidden_dim, n_layers=4)
        self.value_net = None
        self.actor = GRPOActor(feat_summarizer, policy_net)

        self.policy_optimizer = optim.AdamW(self.actor.parameters(), lr=policy_lr, weight_decay=0.01)
        # The base scheduler tracks an optimizer this agent doesn't use: the actor has its
        # own optimizer, so it needs its own cosine schedule, stepped once per update().
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer,
            T_max=lr_T_max,
            eta_min=min_lr,
        )

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
        self.await_group = GRPOGroup(range=self.range, min_std=self.min_group_std)

    def add_new_instance_to_group(self, **instance_info):
        self.await_group.add_instance(GRPOReplayInstance(**instance_info))

    def final_group(self):
        # A group without signal (see GRPOGroup.finalize) never reaches the buffer.
        if self.await_group.finalize():
            new_normalized_instances = self.await_group.get_instances()
            self.replay_buffer.extend(new_normalized_instances)
        self.await_group = None
    
    @torch.no_grad()
    def select_action(self, state, valid_actions, bank_is_full=True, num_samples=1, training=False):
        self.actor.eval()

        image_feat = state.next_image_feat.detach().to(torch.float32)
        memory_feat = state.curr_memory_feat["mem_feat"].detach().to(torch.float32)
        memory_ptr = state.curr_memory_feat["obj_ptr"].detach().to(torch.float32)
        bank_feat = state.prev_memory_bank["mem_feat"].detach().to(torch.float32)
        bank_ptr = state.prev_memory_bank["obj_ptr"].detach().to(torch.float32)
        
        action_logits = self.actor(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr).squeeze(0)
        action_logits = action_logits.detach().float().cpu()

        valid_actions = torch.as_tensor(valid_actions, dtype=torch.int64)
        action_mask = torch.zeros_like(action_logits, dtype=torch.bool)
        action_mask[valid_actions] = True

        # The policy is the distribution over the *selectable* actions only. Everything
        # below (acting, the stored log probs, and the update) uses this one.
        valid_log_probs = torch.log_softmax(action_logits.gather(0, valid_actions), dim=0)
        valid_probs = valid_log_probs.exp()
        
        # if not training:
        #     print({a:p for a, p in zip(valid_actions.tolist(), valid_probs.tolist())})

        if training:
            main_action_idx = torch.multinomial(valid_probs, num_samples=1) if bank_is_full else (valid_actions == 0).nonzero(as_tuple=True) 
            # main_action_idx = torch.argmax(valid_probs) if bank_is_full else (valid_actions == 0).nonzero(as_tuple=True) 
            # The group: num_samples i.i.d. draws from the policy (with replacement), as
            # in GRPO. Being on-policy samples, they enter update() with equal weight.
            # Duplicates are expected; the caller measures each distinct action once.
            action_idx = torch.multinomial(valid_probs, num_samples, replacement=True)
            return {
                "main_action": valid_actions[main_action_idx].item(),
                "action": valid_actions[action_idx].tolist(),
                "log_probs": valid_log_probs[action_idx].tolist(),
                "action_mask": action_mask,
            }
        else:
            # action_idx = torch.multinomial(valid_probs, num_samples=1) if bank_is_full else (valid_actions == 0).nonzero(as_tuple=True) 
            action_idx = torch.argmax(valid_probs) if bank_is_full else (valid_actions == 0).nonzero(as_tuple=True)
            return {
                "main_action": valid_actions[action_idx].item(),
            }

    # update() is called from inside train_sam, which runs under no_grad once SAM2 is
    # frozen (-stop_sam2_ep); the actor still needs a graph. Every other agent gets this
    # from a `with torch.enable_grad()` inside its train_step.
    @torch.enable_grad()
    def update(self, num_update):
        local_count = torch.tensor([len(self.replay_buffer)], dtype=torch.long, device=self.device)
        
        if self.distributed:
            dist.all_reduce(local_count, op=dist.ReduceOp.MIN)
        
        # Wait until the buffer is full (it is a deque with maxlen=buffer_size).
        if local_count < self.buffer_size:
            return None
        
        np.random.seed(self.rank + self.epoch * 100)
        self.actor.eval()

        device = self.device
        total_policy_loss, total_policy_gradnorm = 0, 0
        total_entropy, total_ratio, total_clip_fraction = 0, 0, 0
        buffer = list(self.replay_buffer)
        batches = []
        for _ in range(num_update):
            random.shuffle(buffer)
            batches += [buffer[s:s + self.batch_size] for s in range(0, len(buffer), self.batch_size)]
        num_batches = len(batches)

        for batch in batches:

            states, old_log_probs, actions, rewards, dones, action_masks = zip(*batch)

            image_feat = torch.cat([state.next_image_feat for state in states]).detach()
            memory_feat = torch.cat([state.curr_memory_feat["mem_feat"] for state in states]).detach()
            memory_ptr = torch.cat([state.curr_memory_feat["obj_ptr"] for state in states]).detach()
            bank_feat = torch.cat([state.prev_memory_bank["mem_feat"] for state in states]).detach()
            bank_ptr = torch.cat([state.prev_memory_bank["obj_ptr"] for state in states]).detach()

            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(1)
            dones = torch.FloatTensor(dones).unsqueeze(1)
            action_masks = torch.stack(action_masks)

            image_feat = image_feat.to(device=device, dtype=torch.float32, non_blocking=True)
            memory_feat = memory_feat.to(device=device, dtype=torch.float32, non_blocking=True)
            memory_ptr = memory_ptr.to(device=device, dtype=torch.float32, non_blocking=True)
            bank_feat = bank_feat.to(device=device, dtype=torch.float32, non_blocking=True)
            bank_ptr = bank_ptr.to(device=device, dtype=torch.float32, non_blocking=True)

            actions = actions.to(device=device, non_blocking=True)
            rewards = rewards.to(device=device, dtype=torch.float32, non_blocking=True)
            old_log_probs = old_log_probs.to(device=device, dtype=torch.float32, non_blocking=True)
            dones = dones.to(device=device, dtype=torch.float32, non_blocking=True)
            action_masks = action_masks.to(device=device, non_blocking=True)
            
            # for a in actions.unique():
            #     print(a, rewards[actions == a].median())
            
            # with torch.enable_grad():
            policy_logits = self.actor(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr).float()
            # Renormalize over exactly the actions that were selectable when the sample
            # was collected: the same distribution select_action acted with and recorded
            # old_log_probs from.
            policy_logits = policy_logits.masked_fill(~action_masks, float("-inf"))
            log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_probs = log_probs.exp()  # exactly 0 on masked actions
            log_action_probs = log_probs.gather(1, actions)

            # The group was sampled from the policy (see select_action), so the plain
            # mean over samples is the on-policy estimate of the surrogate.
            policy_loss = self.compute_policy_loss(log_action_probs, rewards, old_log_probs).mean()

            # Entropy of the whole (masked) action distribution. nan_to_num avoids the
            # 0 * -inf = nan on masked actions.
            entropy = -(policy_probs * torch.nan_to_num(log_probs, neginf=0.0)).sum(dim=1).mean()
            policy_loss = policy_loss - entropy * self.entropy_weight # entropy regularization

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            gradnorm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.1)
            self.policy_optimizer.step()
            
            with torch.no_grad():
                ratio = (log_action_probs - old_log_probs).exp()
                clipped = (ratio < 1.0 - self.epsilon) | (ratio > 1.0 + self.epsilon)
                total_ratio += ratio.mean()
                total_clip_fraction += clipped.float().mean()
                total_entropy += entropy.detach()

            total_policy_loss += policy_loss.detach()
            total_policy_gradnorm += gradnorm
        
        # Clear buffer after update for on-policy training
        self.replay_buffer.clear()

        # Logged before stepping so the value matches the LR the updates above ran at.
        current_lr = self.policy_optimizer.param_groups[0]["lr"]
        self.step_lr_scheduler()

        return {
            "actor_loss": total_policy_loss / num_batches,
            "actor_gradnorm": total_policy_gradnorm / num_batches,
            "agent_lr": current_lr,
            "policy_entropy": total_entropy / num_batches,
            "ratio_mean": total_ratio / num_batches,
            "clip_fraction": total_clip_fraction / num_batches,
        }

    def compute_policy_loss(self, log_prob, advantage, old_log_prob):
        """Per-sample clipped surrogate loss, [B,1]; update() does the reduction."""
        advantage = advantage.detach()
        old_log_prob = old_log_prob.detach()
        
        ratio = (log_prob - old_log_prob).exp()
        surr_loss = ratio * advantage
        clipped_surr_loss = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
        return -torch.min(surr_loss, clipped_surr_loss)

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

    def num_parameters(self):
        """This function expect modules didn't wrapped by DDP"""
        return sum(p.numel() for p in self.actor.parameters())
