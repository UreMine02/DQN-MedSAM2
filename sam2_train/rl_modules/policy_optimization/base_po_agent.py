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
from torch.distributions import Categorical

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
    """Compute the TD(lambda) return, i.e. GAE advantage + value.

    A trajectory here spans several chunks of the same volume, and every chunk boundary
    is a real terminal (the memory bank is rebuilt from scratch by the next chunk), so
    `dones` carries 1s in the middle of the sequence, not just at the end. Two places
    have to respect that:
      - the bootstrap in `delta`, via the (1 - dones) factor, and
      - the (gamma*tau)^(j-i) accumulation, which must not carry delta from a later
        episode segment back into an earlier one.
    The second is what `coef` masking below does; without it the terminals would only
    zero one bootstrap term and credit would still leak across the reset.
    """
    L = values.shape[0]
    device = values.device
    delta = rewards + gamma * (1 - dones) * next_value - values # [L,1]

    coef = torch.triu(torch.full((L, L), gamma * tau, device=device))
    l = torch.Tensor(circulant(torch.arange(L))).T.to(device=device, non_blocking=True)
    coef = coef ** l

    # Segment the sequence at the terminals: step t belongs to episode
    # `sum_{k<t} done_k`, so two steps share an episode iff their ids match. Masking
    # coef by that keeps each row's sum inside its own segment.
    done_flags = dones.reshape(-1)  # [L]
    episode_id = torch.cat([
        torch.zeros(1, device=device, dtype=done_flags.dtype), done_flags[:-1].cumsum(0)
    ])
    same_episode = (episode_id.unsqueeze(0) == episode_id.unsqueeze(1)).to(coef.dtype)
    coef = coef * same_episode

    return coef @ delta + values

class RunningReturnScale:
    """Running std of the discounted return, for reward scaling.

    The reward is a difference of dice losses, so returns land around 1e-2 while the
    critic's head naturally emits O(1). A critic asked to fit targets two orders of
    magnitude below its own output scale converges to predicting the mean, which reads
    as explained_variance ~ 0 no matter how long it trains. Dividing rewards by this
    keeps the value targets at unit scale.

    Scale only, no centering: shifting rewards would shift V by a constant, which is
    harmless, but it buys nothing and makes checkpoints harder to compare. The policy is
    unaffected either way, since train_step re-normalizes advantages per minibatch --
    this is purely a critic conditioning fix.
    """

    def __init__(self, eps=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: torch.Tensor):
        x = x.reshape(-1).to(torch.float64)
        n = x.numel()
        if n == 0:
            return
        batch_mean = x.mean().item()
        batch_var = x.var(unbiased=False).item()

        delta = batch_mean - self.mean
        total = self.count + n
        # Chan et al. parallel variance, so the estimate survives being fed one
        # trajectory at a time.
        self.mean += delta * n / total
        self.var = (
            self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total
        ) / total
        self.count = total

    @property
    def std(self):
        return max(self.var, 1e-12) ** 0.5

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean = state.get("mean", 0.0)
        self.var = state.get("var", 1.0)
        self.count = state.get("count", 1e-8)


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
        return_=None,
        action_mask=None,
        policy_weight=1.0
    ):
        super().__init__(frame_idx, state, action, next_state, loss_before, loss_after, reward, eps)
        self.log_probs = log_probs
        self.advantage = advantage
        self.return_ = return_
        # Support the action was sampled from, so the update can renormalize over
        # the same actions instead of the full action space.
        self.action_mask = action_mask
        # 0 for transitions whose action was forced by the environment (bank not full
        # yet): they still train the critic but must not enter the policy loss.
        self.policy_weight = policy_weight

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
            self.advantage,
            self.action_mask,
            self.policy_weight
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
        missing = [i for i, t in enumerate(self.transitions) if t.next_state is None]
        if missing:
            raise RuntimeError(
                f"transitions {missing} have no next_state; every transition must be "
                "closed by close_await_replay_instance, either by the next "
                "init_new_replay_instance or by set_await_done at a chunk boundary"
            )

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

        # Recency prior over the non-conditioning slots. The policy head is permutation
        # equivariant over slots and prepare_rl_state carries no temporal encoding, so
        # without this the agent cannot tell which memory is oldest. Warm-started from
        # SAM2's maskmem_tpos_enc via load_sam2_temporal_prior(); the projection on top
        # is trainable, which is what lets the RL loss re-separate slots that SAM2's
        # segmentation loss left nearly collinear (pretrained rows 3/4/5 sit at cosine
        # 0.92-0.98). Non-persistent: re-derived from the SAM2 weights on every run, so
        # it must not enter the agent checkpoint.
        self.register_buffer(
            "slot_tpos_prior", torch.zeros(num_maskmem, memory_dim), persistent=False
        )
        nn.init.trunc_normal_(self.slot_tpos_prior, std=0.02)
        self.slot_tpos_proj = nn.Linear(memory_dim, image_dim)

    @torch.no_grad()
    def load_sam2_temporal_prior(self, maskmem_tpos_enc, sam2_num_maskmem):
        """Warm-start the recency prior from SAM2's temporal position encoding.

        SAM2 looks it up as maskmem_tpos_enc[sam2_num_maskmem - t_pos - 1], and its
        agent_act branch assigns t_pos = t + 1 to bank slot t (slot 0 is the oldest:
        dict insertion order survives eviction, because popping preserves the relative
        order of survivors and appends are always the newest frame). So slot t maps to
        row sam2_num_maskmem - t - 2.
        """
        prior = maskmem_tpos_enc.detach().reshape(sam2_num_maskmem, -1)
        if prior.shape[-1] != self.slot_tpos_prior.shape[-1]:
            raise ValueError(
                f"SAM2 mem_dim {prior.shape[-1]} != agent memory_dim "
                f"{self.slot_tpos_prior.shape[-1]}"
            )
        rows = [sam2_num_maskmem - t - 2 for t in range(self.num_maskmem)]
        if min(rows) < 0:
            raise ValueError(
                f"memory bank size {self.num_maskmem} needs at least "
                f"{self.num_maskmem + 2} SAM2 maskmem slots, got {sam2_num_maskmem}"
            )
        self.slot_tpos_prior.copy_(prior[rows].to(self.slot_tpos_prior.dtype))

    def forward(self, image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr):
        B, T, C, H, W = bank_feat.shape

        # prepare_rl_state zero-fills unoccupied slots. Read occupancy off the raw bank,
        # before the projections below give those slots a non-zero bias.
        occupied = bank_feat.flatten(2).abs().sum(-1) != 0  # [B,T]
        non_cond_valid, cond_valid = torch.tensor_split(
            occupied, (self.num_maskmem,), dim=1
        )

        combined_mem_feat = torch.cat([bank_feat, memory_feat.unsqueeze(1)], dim=1)
        combined_mem_feat = combined_mem_feat.reshape(B * (T+1), C, H, W)
        memory_spatial_query = self.memory_spatial_summary(combined_mem_feat)
        image_spatial_query = self.image_spatial_summary(image_feat)

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

        # Recency only for the bank slots. The incoming memory shares non_cond_proj and
        # gets no distinguishing embedding on purpose: "skip the new frame" is the same
        # operation as "evict slot k", just applied to the memory that would be added.
        non_cond_bank_feat = (
            self.non_cond_proj(non_cond_bank_feat)
            + self.slot_tpos_proj(self.slot_tpos_prior).unsqueeze(0)
        )
        curr_mem_feat = self.non_cond_proj(curr_mem_feat)

        # cond_bank_feat is laid out [frame-major mem queries | one obj_ptr per frame],
        # so the per-frame validity expands the same way.
        cond_valid_tokens = torch.cat(
            [cond_valid.repeat_interleave(self.n_query, dim=1), cond_valid], dim=1
        )
        pad_masks = {"non_cond": non_cond_valid, "cond": cond_valid_tokens}

        return image_spatial_query, non_cond_bank_feat, cond_bank_feat, curr_mem_feat, pad_masks

class BasePolicyNetwork(nn.Module):
    """Pointer head: one query token per candidate, one logit per query token.

    Queries are [non_drop, incoming memory, bank slots], which fixes the action
    indices: 0 = add without dropping, 1 = skip the incoming frame, 2+ = evict bank
    slot k-2. This must stay in sync with action_frame_map in rl_utils.prepare_rl_state.
    Context is the conditioning frames plus the incoming image.
    """

    def __init__(
        self,
        hidden_dim,
        n_layers=1,
        n_heads=4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.non_drop_embed = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.trunc_normal_(self.non_drop_embed, std=0.02)
        # QFormerBlock keeps the query self-attention in its own softmax, so candidates
        # can be compared against each other instead of competing with every context
        # token for attention mass. d_heads is the PER-HEAD width: CrossAttention builds
        # inner_dim = d_heads * n_heads, so this must be hidden_dim // n_heads.
        self.action_decoder = nn.ModuleList([
            QFormerBlock(hidden_dim, hidden_dim, n_heads, hidden_dim // n_heads, dropout=0.1)
            for _ in range(n_layers)
        ])

        self.action_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1)
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        proj_std = (self.hidden_dim ** -0.5) * ((2 * self.n_layers) ** -0.5)
        attn_std = self.hidden_dim ** -0.5
        fc_std = (2 * self.hidden_dim) ** -0.5
        for block in self.action_decoder:
            for attn in (block.attn1, block.attn2):
                for proj in (attn.to_q, attn.to_k, attn.to_v):
                    nn.init.normal_(proj.weight, std=attn_std)
                nn.init.normal_(attn.to_out[0].weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, image_spatial_query, non_cond_bank_feat, cond_bank_feat, curr_mem_feat,
                pad_masks=None):
        B = image_spatial_query.shape[0]
        dtype = image_spatial_query.dtype
        device = image_spatial_query.device
        non_drop_embed = self.non_drop_embed.to(dtype) + torch.zeros(B, 1, self.hidden_dim, dtype=dtype, device=device)

        action_query = torch.cat([non_drop_embed, curr_mem_feat, non_cond_bank_feat], dim=1)
        action_context = torch.cat([cond_bank_feat, image_spatial_query], dim=1)

        context_mask, self_mask = None, None
        if pad_masks is not None:
            # Image tokens are always real; the leading two query tokens always exist.
            context_mask = torch.cat([
                pad_masks["cond"],
                torch.ones(B, image_spatial_query.shape[1], dtype=torch.bool, device=device),
            ], dim=1)
            self_mask = torch.cat([
                torch.ones(B, 1 + curr_mem_feat.shape[1], dtype=torch.bool, device=device),
                pad_masks["non_cond"],
            ], dim=1)

        for layer in self.action_decoder:
            action_query = layer(
                action_query, action_context,
                context_mask=context_mask, self_mask=self_mask,
            )

        actions_logits = self.action_proj(action_query)

        return actions_logits.squeeze(-1)

class BaseValueNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_layers=4
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.value_query = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.trunc_normal_(self.value_query, std=0.02)
        self.value_decoder = nn.ModuleList(
            [PerceiverResampler(self.hidden_dim, 1, dropout=0.1) for _ in range(n_layers)]
        )

        self.value_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1)
        )

        # Start at V(s) = 0. With default init the head emits values of std ~0.3 while
        # the returns it has to fit are far smaller, so the critic spends its early
        # budget unlearning an arbitrary offset and explained_variance starts deeply
        # negative. Both weight and bias have to go: zeroing the weight alone still
        # leaves a random constant.
        nn.init.zeros_(self.value_proj[1].weight)
        nn.init.zeros_(self.value_proj[1].bias)

    def forward(self, image_spatial_query, non_cond_bank_feat, cond_bank_feat, curr_mem_feat, pad_masks=None):
        B = image_spatial_query.shape[0]
        device = image_spatial_query.device
        value_query = self.value_query.expand(B, 1, self.hidden_dim)

        tokens = torch.cat([curr_mem_feat, non_cond_bank_feat, cond_bank_feat, image_spatial_query], dim=1)

        context_mask = None
        if pad_masks is not None:
            context_mask = torch.cat([
                torch.ones(B, curr_mem_feat.shape[1], dtype=torch.bool, device=device),
                pad_masks["non_cond"],
                pad_masks["cond"],
                torch.ones(B, image_spatial_query.shape[1], dtype=torch.bool, device=device),
            ], dim=1)

        for layer in self.value_decoder:
            value_query = layer(x_f=tokens, x=value_query, context_mask=context_mask)

        return self.value_proj(value_query).squeeze(1)


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
        lr_T_max=1000,
        min_lr=0.0,
        sam2_dim={}
    ):
        super().__init__(num_maskmem, policy_lr, gamma, beta, buffer_size, batch_size, device)
        self.feat_summarizer = BaseFeatureSummarizer(num_maskmem, **sam2_dim, n_layers=4)
        self.policy_net = BasePolicyNetwork(self.feat_summarizer.hidden_dim, n_layers=4)
        self.value_net = BaseValueNetwork(self.feat_summarizer.hidden_dim, n_layers=4)

        self.optimizer = optim.AdamW([
            {"params": self.policy_net.parameters(),      "lr": policy_lr},
            {"params": self.value_net.parameters(),       "lr": value_lr },
            {"params": self.feat_summarizer.parameters(), "lr": max(policy_lr, value_lr)},
        ])
        
        # One cosine step per `update()` call, so T_max counts agent updates over the
        # whole run, not minibatches.
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=lr_T_max,
            eta_min=min_lr,
        )

        # self.policy_optimizer = optim.AdamW(
        #     list(self.policy_net.parameters()) + 
        #     list(self.feat_summarizer.parameters()),
        #     lr=policy_lr,
        # )
        # self.value_optimizer = optim.AdamW(
        #     list(self.value_net.parameters()),
        #     lr=value_lr,
        # )

        self.tau = tau
        self.entropy_weight= entropy_weight

        # For distributed training
        self.rank = 0
        self.distributed = False

        # Episodic returns accumulated between calls to `pop_episode_return_stats`,
        # used to log mean/std of episodic return to wandb once per epoch.
        self.episode_returns = []

        # Keeps the critic's targets at unit scale; see RunningReturnScale.
        self.return_scale = RunningReturnScale()

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
        """Close out a volume: compute GAE over every chunk collected for it, push the
        transitions to the replay buffer.

        The trajectory spans all chunks of one (volume, obj_id); the chunk boundaries
        are marked done by `set_await_done`, and compute_gae segments on those so no
        credit crosses a memory-bank reset.
        """
        if self.await_trajectory is None or len(self.await_trajectory.transitions) == 0:
            self.await_trajectory = None
            return

        log_probs, action, reward, done, curr_state, next_state = self.await_trajectory.get_transitions(self.device)

        # Episodic return: total (undiscounted) RAW reward per episode, i.e. per chunk,
        # not per volume — a volume holds as many episodes as it has chunks, and their
        # count varies with volume length, so summing the whole trajectory would make
        # the metric track volume size instead of policy quality. Taken before scaling
        # so the logged number stays in dice-loss units and is comparable across runs.
        segment_ends = done.reshape(-1).nonzero(as_tuple=True)[0].tolist()
        if not segment_ends or segment_ends[-1] != len(done) - 1:
            segment_ends.append(len(done) - 1)
        start = 0
        for end in segment_ends:
            self.episode_returns.append(reward[start:end + 1].sum().item())
            start = end + 1

        # Refresh the scale from this trajectory's raw discounted return, then scale.
        # compute_gae with values=0 and tau=1 is exactly the Monte-Carlo discounted
        # return, segmented at the same terminals.
        # zeros = torch.zeros_like(reward)
        # mc_return = compute_gae(zeros, zeros, reward, done, self.gamma, 1.0)
        # self.return_scale.update(mc_return)
        # reward = reward / self.return_scale.std

        # These are value *targets*: they must be computed with dropout off. update()
        # leaves the modules in train mode, so do not rely on the last select_action.
        self.feat_summarizer.eval()
        self.value_net.eval()

        with torch.no_grad():
            curr_feat = self.feat_summarizer(*curr_state)
            curr_value = self.value_net(*curr_feat)
            next_feat = self.feat_summarizer(*next_state)
            next_value = self.value_net(*next_feat)

            # V was trained on scaled returns, so it already lives in the scaled space;
            # feeding it scaled rewards keeps every term of the GAE recursion consistent.
            return_ = compute_gae(curr_value, next_value, reward, done, self.gamma, self.tau)
            return_ = return_.squeeze(-1)
            advantage = return_ - curr_value.squeeze(-1)

        for i, ins in enumerate(self.await_trajectory.transitions):
            ins.set_return_advantage(return_[i].cpu(), advantage[i].cpu())
            self.replay_buffer.append(ins.get_updated())

        self.await_trajectory = None

    def step_lr_scheduler(self):
        """Advance the cosine schedule by one agent update. Stops at T_max: past it a
        plain cosine would anneal back up, and the number of updates a run performs
        isn't known in advance."""
        if self.scheduler.last_epoch < self.scheduler.T_max:
            self.scheduler.step()

    def pop_episode_return_stats(self):
        """Return mean/std of episodic returns accumulated since the last call, then
        clear the buffer. Empty dict if no trajectory has finished yet."""
        if len(self.episode_returns) == 0:
            return {}

        returns = torch.tensor(self.episode_returns, dtype=torch.float32)
        stats = {
            "episodic_return_mean": returns.mean().item(),
            "episodic_return_std": returns.std().item() if returns.numel() > 1 else 0.0,
        }
        self.episode_returns = []
        return stats

    def init_new_replay_instance(self, **instance_info):
        self.close_await_replay_instance(next_state=instance_info.get("state"))
        self.await_replay_instance = POReplayInstance(**instance_info)

    def store_transition(self, instance):
        """Hold transitions in the trajectory; they only reach the replay buffer once
        final_trajectory has run GAE over the whole volume."""
        self.await_trajectory.add_transition(instance)

    @torch.no_grad()
    def select_action(self, state: RLStates, valid_actions, bank_is_full, training=False):
        self.feat_summarizer.eval()
        self.policy_net.eval()
        self.value_net.eval()
        
        device = next(self.feat_summarizer.parameters()).device

        image_feat = state.next_image_feat.detach().to(device=device, dtype=torch.float32)
        memory_feat = state.curr_memory_feat["mem_feat"].detach().to(device=device, dtype=torch.float32)
        memory_ptr = state.curr_memory_feat["obj_ptr"].detach().to(device=device, dtype=torch.float32)
        bank_feat = state.prev_memory_bank["mem_feat"].detach().to(device=device, dtype=torch.float32)
        bank_ptr = state.prev_memory_bank["obj_ptr"].detach().to(device=device, dtype=torch.float32)

        state = self.feat_summarizer(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr)
        action_logits = self.policy_net(*state).squeeze(0)
        action_logits = action_logits.detach().cpu()
        action_dist = Categorical(logits=action_logits)

        valid_actions = torch.Tensor(valid_actions).to(torch.int64)
        action_mask = torch.zeros_like(action_logits, dtype=torch.bool)
        action_mask[valid_actions] = True

        valid_dist = Categorical(logits=action_logits.gather(0, valid_actions))
        valid_probs = valid_dist.probs

        if not training:
            print({a:p for a, p in zip(valid_actions.tolist(), valid_probs.tolist())})

        if not bank_is_full:
            # Filling up the bank: the action is imposed by the environment, not picked
            # by the policy, so it carries no policy gradient signal.
            action_idx = (valid_actions == 0).nonzero(as_tuple=True)
            policy_weight = 0.0
        elif training:
            # Sample from the policy: the stored log prob must be the distribution the
            # action actually came from, otherwise the importance ratio is meaningless.
            # Exploration comes from this sampling plus the entropy bonus.
            action_idx = torch.multinomial(valid_probs, num_samples=1, replacement=False)
            policy_weight = 1.0
        else:
            action_idx = torch.argmax(valid_probs, keepdim=True)
            policy_weight = 1.0

        return {
            "action": valid_actions[action_idx].item(),
            # Scalar, not a 1-element list: a list makes old_log_probs [B,1,1] in
            # train_step and the PPO ratio broadcasts into a [B,B,1] outer product.
            "log_probs": valid_probs.log()[action_idx].item(),
            "action_mask": action_mask,
            "policy_weight": policy_weight,
        }

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

    def update(self, num_ep):
        local_count = torch.tensor([len(self.replay_buffer)], dtype=torch.long, device=self.device)
        if self.distributed:
            dist.all_reduce(local_count, op=dist.ReduceOp.MIN)

        if local_count < self.batch_size or num_ep <= 0:
            return None

        np.random.seed(self.rank + self.epoch * 100)

        buffer_size = len(self.replay_buffer)
        print(f"Update agent for {num_ep} epochs over {buffer_size} transitions")
        self.feat_summarizer.train()
        self.policy_net.train()
        self.value_net.train()

        total_policy_loss, total_value_loss, total_actor_gradnorm, total_critic_gradnorm = 0, 0, 0, 0
        num_update = 0
        critic_num_update = 0
        metric_sums, metric_counts = {}, {}

        for _ in range(num_ep):
            # Full pass over the replay buffer, shuffled and split into batch_size minibatches.
            shuffle_indice = np.random.permutation(buffer_size)

            for start in range(0, buffer_size, self.batch_size):
                batch_indice = shuffle_indice[start:start + self.batch_size]
                batch = [self.replay_buffer[idx] for idx in batch_indice]

                update_value = True
                value_loss, policy_loss, actor_gradnorm, critic_gradnorm, metrics = self.train_step(batch, update_value=update_value)

                total_policy_loss += policy_loss
                total_actor_gradnorm += actor_gradnorm
                num_update += 1

                if update_value:
                    total_value_loss += value_loss
                    total_critic_gradnorm += critic_gradnorm
                    critic_num_update += 1

                for k, v in metrics.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + v
                    metric_counts[k] = metric_counts.get(k, 0) + 1

        # Logged before stepping so the value matches the LR the minibatches above ran at.
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.step_lr_scheduler()

        out = {
            "actor_loss": total_policy_loss / num_update,
            "critic_loss": total_value_loss / critic_num_update,
            "actor_gradnorm": total_actor_gradnorm / num_update,
            "critic_gradnorm": total_critic_gradnorm / critic_num_update,
            "agent_lr": current_lr,
            # Raw dice-loss units per unit of scaled return; multiply episodic_return_*
            # by this to compare against an unscaled run.
            "return_scale": self.return_scale.std,
        }
        for k, total in metric_sums.items():
            out[k] = total / metric_counts[k]
        # Mean/std of episodic return since the last update, if any trajectory finished.
        out.update(self.pop_episode_return_stats())

        return out

    def train_step(self, batch, update_value=True):
        device = self.device

        (
            states,
            old_log_probs,
            actions,
            rewards,
            next_states,
            dones,
            returns,
            advantages,
            action_masks,
            policy_weights,
        ) = zip(*batch)

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
        action_masks = torch.stack(action_masks)
        policy_weights = torch.FloatTensor(policy_weights).unsqueeze(1)

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
        action_masks = action_masks.to(device=device, non_blocking=True)
        policy_weights = policy_weights.to(device=device, dtype=torch.float32, non_blocking=True)

        # Normalize over exactly the transitions that enter the policy loss. The forced
        # bank-filling steps carry weight 0 and there are memory_bank_size of them per
        # chunk (6 out of ~15 at video_length=16), so folding them in would rescale the
        # real decisions by a mean/std they never contribute a gradient to.
        weight_sum = policy_weights.sum()
        if weight_sum > 0:
            adv_mean = (advantages * policy_weights).sum(dim=0, keepdim=True) / weight_sum
            adv_var = (((advantages - adv_mean) ** 2) * policy_weights).sum(dim=0, keepdim=True) / weight_sum
            adv_std = adv_var.clamp(min=0).sqrt()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            # Every transition in this minibatch was forced: nothing to normalize, and
            # dividing by a zero std would blow the (zero-weighted) advantages up to inf.
            adv_mean = torch.zeros_like(weight_sum).reshape(1, 1)
            adv_std = torch.zeros_like(weight_sum).reshape(1, 1)
        weight_sum = weight_sum.clamp(min=1.0)

        with torch.enable_grad():
            (
                image_spatial_query,
                non_cond_bank_feat,
                cond_bank_feat,
                curr_mem_feat,
                pad_masks
            ) = self.feat_summarizer(image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr)

            policy_logits = self.policy_net(
                image_spatial_query,
                non_cond_bank_feat,
                cond_bank_feat,
                curr_mem_feat,
                pad_masks=pad_masks
            )
            # Renormalize over exactly the actions that were selectable when the
            # transition was collected, matching the distribution select_action sampled
            # from. Without this the ratio against old_log_probs is biased and the
            # entropy bonus pushes mass onto actions that can never be taken.
            policy_logits = policy_logits.masked_fill(~action_masks, float("-inf"))
            log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_probs = log_probs.exp() # exactly 0 on masked actions
            log_action_probs = log_probs.gather(1, actions)
            policy_loss = self.compute_policy_loss(log_action_probs, advantages, old_log_probs)
            # nan_to_num avoids the 0 * -inf = nan on masked actions
            minus_entropy = (policy_probs * torch.nan_to_num(log_probs, neginf=0.0)).sum(dim=1, keepdim=True)
            policy_loss = policy_loss + minus_entropy * self.entropy_weight # entropy regularization
            # Forced (bank-filling) transitions carry weight 0
            policy_loss = (policy_loss * policy_weights).sum() / weight_sum

            with torch.no_grad():
                # Forced (bank-filling) transitions carry no policy signal, so they're
                # excluded (weight 0) from these stats just like from the loss above.
                policy_entropy = (-minus_entropy * policy_weights).sum() / weight_sum

                ratio = (log_action_probs - old_log_probs).exp()
                ratio_mean = (ratio * policy_weights).sum() / weight_sum
                ratio_var = (((ratio - ratio_mean) ** 2) * policy_weights).sum() / weight_sum
                ratio_std = ratio_var.clamp(min=0).sqrt()

                metrics = {
                    "policy_entropy": policy_entropy.item(),
                    "ratio_mean": ratio_mean.item(),
                    "ratio_std": ratio_std.item(),
                    "adv_mean": adv_mean.item(),
                    "adv_std": adv_std.item(),
                }
                if hasattr(self, "epsilon"):
                    clipped = (ratio < 1.0 - self.epsilon) | (ratio > 1.0 + self.epsilon)
                    metrics["clip_fraction"] = (clipped.float() * policy_weights).sum().item() / weight_sum.item()

            # self.policy_optimizer.zero_grad()
            # policy_loss.backward()
            # actor_gradnorm = nn.utils.clip_grad_norm_(
            #     list(self.feat_summarizer.parameters()) + list(self.policy_net.parameters()),
            #     max_norm=0.5
            # )
            # self.policy_optimizer.step()

            # if update_value:
            #     image_spatial_query = image_spatial_query.detach()
            #     non_cond_bank_feat = non_cond_bank_feat.detach()
            #     cond_bank_feat = cond_bank_feat.detach()
            #     curr_mem_feat = curr_mem_feat.detach()

            #     pred_value = self.value_net(
            #         image_spatial_query,
            #         non_cond_bank_feat,
            #         cond_bank_feat,
            #         curr_mem_feat
            #     )

            #     value_loss = F.mse_loss(pred_value, returns)
            #     self.value_optimizer.zero_grad()
            #     value_loss.backward()
            #     critic_gradnorm = nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            #     self.value_optimizer.step()

            #     with torch.no_grad():
            #         return_var = returns.var(unbiased=False)
            #         explained_variance = 1.0 - (returns - pred_value).var(unbiased=False) / return_var.clamp(min=1e-8)
            #         metrics["explained_variance"] = explained_variance.item()
            # else:
            #     value_loss = torch.Tensor([0])
            #     critic_gradnorm = torch.Tensor([0])

            pred_value = self.value_net(
                image_spatial_query,
                non_cond_bank_feat,
                cond_bank_feat,
                curr_mem_feat,
                pad_masks=pad_masks
            )
            value_loss = F.mse_loss(pred_value, returns)
            
            with torch.no_grad():
                return_var = returns.var(unbiased=False)
                explained_variance = 1.0 - (returns - pred_value).var(unbiased=False) / return_var.clamp(min=1e-8)
                metrics["explained_variance"] = explained_variance.item()

            total_loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            actor_gradnorm = nn.utils.clip_grad_norm_(
                list(self.feat_summarizer.parameters()) + list(self.policy_net.parameters()),
                max_norm=0.5
            )
            critic_gradnorm = nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            self.optimizer.step()

        return value_loss.detach(), policy_loss.detach(), actor_gradnorm, critic_gradnorm, metrics

    def compute_policy_loss(self, log_prob, advantage, old_log_prob):
        return -(advantage * log_prob)

    def state_dict(self):
        if isinstance(self.feat_summarizer, DDP):
            return {
                "feat_summarizer": self.feat_summarizer.module.state_dict(),
                "policy_net": self.policy_net.module.state_dict(),
                "value_net": self.value_net.module.state_dict(),
                "return_scale": self.return_scale.state_dict(),
            }
        return {
            "feat_summarizer": self.feat_summarizer.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            # Without this a resumed run would restart the scale at 1.0 while the critic
            # is already trained in the scaled space, silently rescaling every target.
            "return_scale": self.return_scale.state_dict(),
        }

    def load_state_dict(self, state_dict, strict=False):
        """Load agent weights.

        Defaults to strict=False so checkpoints predating the temporal prior and the
        QFormer policy decoder still load; anything absent keeps its fresh init.
        Missing/unexpected keys are printed rather than swallowed, because silently
        reinitialising the whole policy would look like a training collapse.
        """
        if "return_scale" in state_dict:
            self.return_scale.load_state_dict(state_dict["return_scale"])

        for name in ("feat_summarizer", "policy_net", "value_net"):
            module = getattr(self, name)
            if module is None or name not in state_dict:
                continue
            if isinstance(module, DDP):
                module = module.module
            incompatible = module.load_state_dict(state_dict[name], strict=strict)
            missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
            if missing or unexpected:
                print(f"[agent] {name}: {len(missing)} missing, {len(unexpected)} unexpected keys")
                if missing:
                    print(f"    missing (freshly initialised): {missing}")
                if unexpected:
                    print(f"    unexpected (ignored): {unexpected}")

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

