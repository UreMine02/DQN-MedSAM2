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

from sam2_train.rl_modules.rl_components import (
    RLStates,
    RLReplayInstance,
)
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
from sam2_train.modeling.sam2_utils import get_1d_sine_pe, MLP


def stack_state_feats(states, device=None):
    """Batch a list of RLStates into the kwargs the feature summarizer takes.

    Three call sites used to inline this (trajectory GAE, minibatch training, action
    selection) and they had to agree exactly on field order; with the pool and age tensors
    added, keeping three copies in sync is how a silent misalignment gets in.
    """
    def cat(pick, dtype=torch.float32):
        return torch.cat([pick(s) for s in states]).detach().to(
            device=device, dtype=dtype, non_blocking=True
        )

    feats = {
        "image_feat": cat(lambda s: s.next_image_feat),
        "memory_feat": cat(lambda s: s.curr_memory_feat["mem_feat"]),
        "memory_ptr": cat(lambda s: s.curr_memory_feat["obj_ptr"]),
        "bank_feat": cat(lambda s: s.prev_memory_bank["mem_feat"]),
        "bank_ptr": cat(lambda s: s.prev_memory_bank["obj_ptr"]),
    }

    if states[0].global_pool is not None:
        feats.update({
            "cand_age": cat(lambda s: s.cand_age),
            "bank_age": cat(lambda s: s.bank_age),
            "pool_feat": cat(lambda s: s.global_pool["mem_feat"]),
            "pool_ptr": cat(lambda s: s.global_pool["obj_ptr"]),
            "pool_age": cat(lambda s: s.global_pool["age"]),
            "pool_valid": cat(lambda s: s.global_pool["valid"], dtype=torch.bool),
        })

    return feats


def compute_gae(values: torch.Tensor, next_value: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, gamma: float, tau: float):
    """Compute the TD(lambda) return, i.e. GAE advantage + value.

    A trajectory spans a whole volume, which is one episode: the memory bank now survives
    chunk boundaries, so `dones` is normally a single 1 on the last step. The segmentation
    below is kept anyway, so this stays correct if a trajectory ever carries more than one
    episode. It does two things a plain GAE does not:
      - the bootstrap in `delta` is killed at a terminal, via the (1 - dones) factor, and
      - the (gamma*tau)^(j-i) accumulation is prevented from carrying delta from a later
        episode segment back into an earlier one.
    The second is what `coef` masking below does; without it a terminal would only zero
    one bootstrap term and credit would still leak past it.
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
        policy_weight=1.0,
        extra_samples=None,
        extra_weight=1.0,
    ):
        super().__init__(frame_idx, state, action, next_state, loss_before, loss_after, reward, eps)
        # log-prob of the single joint (candidate, evicted-slot) decision, or of no-op.
        self.log_probs = log_probs
        self.advantage = advantage
        self.return_ = return_
        # Support the action was sampled from, so the update can renormalize over the
        # same actions instead of the full action space.
        self.action_mask = action_mask
        # 0 for transitions whose action was forced by the environment (bank not full
        # yet): they still train the critic but must not enter the policy loss.
        self.policy_weight = policy_weight
        # Actions the environment scored at this same state but did not take, as
        # {"action", "log_probs", "reward"}; see get_side_sample. Each also gets a
        # "next_value" once the following frame can build its successor state.
        self.extra_samples = extra_samples or []
        self.extra_weight = extra_weight
        # V(s') for the successor the taken action actually produced, estimated the same
        # way as the branches' so the difference between them is meaningful. None on the
        # terminal transition, which has no continuation to compare.
        self.next_value = None

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
            self.policy_weight,
            # Real, on-trajectory transition: the only kind the critic may fit.
            1.0,
        ))

    def get_side_sample(self, extra, return_, advantage):
        """One counterfactual action, in the same tuple layout as get_updated.

        It shares this transition's state and support -- only the action, its log-prob
        under the behaviour policy, and the immediate reward differ. Its return and
        advantage are handed in by final_trajectory, which offsets this transition's own
        by the full one-step Q difference (reward *and* bootstrapped continuation), so
        the two land on one scale without the branch inheriting a future it would not
        have had.

        `next_state` is this transition's, which is not the branch's successor -- the
        buffer keeps it only for shape, since train_step reads returns and advantages
        that are already computed.

        Value weight is 0: the critic estimates V(s) under the policy, and s already
        appears once with the on-trajectory return. Letting the branches in would train
        it towards the mean over sampled actions instead, which is not V.
        """
        return tuple((
            self.state,
            extra["log_probs"],
            extra["action"],
            extra["reward"],
            self.next_state,
            self.done,
            return_,
            advantage,
            self.action_mask,
            self.policy_weight * self.extra_weight,
            0.0,
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
                "init_new_replay_instance or by set_await_done at the end of the volume"
            )

        transitions = [trans.get() for trans in self.transitions]
        states, log_probs, actions, rewards, next_states, dones = zip(*transitions)

        curr_feats = stack_state_feats(states, device=device)
        next_feats = stack_state_feats(next_states, device=device)

        actions = torch.LongTensor(actions).unsqueeze(1).to(device=device, non_blocking=True)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device=device, non_blocking=True)
        log_probs = torch.FloatTensor(log_probs).unsqueeze(1).to(device=device, non_blocking=True)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device=device, non_blocking=True)

        return log_probs, actions, rewards, dones, curr_feats, next_feats

class BaseFeatureSummarizer(nn.Module):
    def __init__(
        self, num_maskmem, n_query=16, image_dim=256, memory_dim=64, obj_ptr_dim=256,
        n_layers=4, image_summary_dim=None,
    ):
        super().__init__()

        self.num_maskmem = num_maskmem
        self.n_query = n_query
        self.memory_dim = memory_dim
        memory_num_head = memory_dim // 64
        self.hidden_dim = image_dim

        # image_spatial_summary cross-attends to the raw [image_dim,H,W] SAM2 features,
        # so its qformer stack costs O(image_dim^2) per layer -- by far the biggest line
        # item in this module (see base_po_agent parameter-count discussion). Running it
        # at a narrower image_summary_dim instead cuts that quadratically; image_down_proj
        # is the one-time cost of getting there, and SpatialSummarizer's out_proj (see
        # rl_blocks) projects back up to hidden_dim so nothing downstream has to know the
        # summary ran narrower. None (the default) means no reduction, i.e. bit-identical
        # to before this option existed.
        image_summary_dim = image_summary_dim or image_dim
        self.image_down_proj = (
            nn.Identity() if image_summary_dim == image_dim
            else nn.Conv2d(image_dim, image_summary_dim, kernel_size=1)
        )
        self.image_spatial_summary = SpatialSummarizer(
            n_query=n_query,
            query_dim=self.hidden_dim,
            spatial_dim=image_summary_dim,
            n_heads=4,
            d_heads=image_summary_dim,
            n_layers=n_layers,
            dropout=0.0
        )
        self.memory_spatial_summary = SpatialSummarizer(
            n_query=n_query,
            query_dim=memory_dim,
            spatial_dim=memory_dim,
            n_heads=2,
            d_heads=memory_dim,
            n_layers=n_layers,
            dropout=0.0
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

        # Absolute age, in frames, of the candidate / each bank slot / each pool entry.
        # slot_tpos_prior above encodes recency *rank*, which was sufficient while the bank
        # only ever held the last few frames: rank and distance were the same thing. Once a
        # memory can be recalled from the global pool, rank 0 may be 2 frames back or 200,
        # and the policy has no way to tell those apart without this.
        self.age_dim = memory_dim
        self.age_proj = nn.Linear(memory_dim, image_dim)
        # Zero-init so the age term contributes exactly nothing until trained. A checkpoint
        # from before the pool therefore reproduces its old behaviour on load instead of
        # having a randomly-initialised bias injected into every memory token.
        nn.init.zeros_(self.age_proj.weight)
        nn.init.zeros_(self.age_proj.bias)

        # Marks a token as "archived in the pool" rather than "resident in the bank". The
        # two share non_cond_proj on purpose -- both are encoded past frames, and the only
        # differences that matter are this flag and the age above.
        self.pool_type_embed = nn.Parameter(torch.zeros(image_dim))
        nn.init.trunc_normal_(self.pool_type_embed, std=0.02)

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

    def _age_embed(self, age):
        """[B,N] ages in frames -> [B,N,image_dim] additive embedding.

        get_1d_sine_pe is plain elementwise arithmetic, so autocast leaves its output in
        the input's dtype; the agent is run in bfloat16 (train_3d casts it), and matching
        the projection's dtype here keeps this working with autocast off as well.
        """
        pe = get_1d_sine_pe(age.float(), dim=self.age_dim)
        return self.age_proj(pe.to(self.age_proj.weight.dtype))

    def forward(
        self, image_feat, memory_feat, memory_ptr, bank_feat, bank_ptr,
        cand_age=None, bank_age=None,
        pool_feat=None, pool_ptr=None, pool_age=None, pool_valid=None,
    ):
        B, T, C, H, W = bank_feat.shape
        n_pool = 0 if pool_feat is None else pool_feat.shape[1]

        # prepare_rl_state zero-fills unoccupied slots. Read occupancy off the raw bank,
        # before the projections below give those slots a non-zero bias.
        occupied = bank_feat.flatten(2).abs().sum(-1) != 0  # [B,T]
        non_cond_valid, cond_valid = torch.tensor_split(
            occupied, (self.num_maskmem,), dim=1
        )

        # Pool slots carry an explicit validity mask rather than reusing the all-zero test
        # above: an archived memory is arbitrary tensor content, and inferring "empty" from
        # it would be a guess where the pool already knows the answer.
        # Appended last so the existing split indices keep their meaning.
        mem_parts = [bank_feat, memory_feat.unsqueeze(1)]
        if n_pool:
            mem_parts.append(pool_feat)
        combined_mem_feat = torch.cat(mem_parts, dim=1)
        n_mem = T + 1 + n_pool
        combined_mem_feat = combined_mem_feat.reshape(B * n_mem, C, H, W)
        memory_spatial_query = self.memory_spatial_summary(combined_mem_feat)
        image_spatial_query = self.image_spatial_summary(self.image_down_proj(image_feat))

        memory_spatial_query = memory_spatial_query.reshape(
            B, n_mem, self.n_query, self.memory_dim
        )
        (
            non_cond_bank_feat,
            cond_bank_feat,
            curr_mem_feat,
            pool_mem_feat,
        ) = torch.tensor_split(
            memory_spatial_query,
            indices=(self.num_maskmem, T, T + 1),
            dim=1,
        )
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

        if bank_age is not None:
            non_cond_bank_feat = non_cond_bank_feat + self._age_embed(bank_age)
        if cand_age is not None:
            curr_mem_feat = curr_mem_feat + self._age_embed(cand_age)

        pool_query = None
        if n_pool:
            pool_mem_feat = torch.cat([pool_mem_feat.flatten(2), pool_ptr], dim=-1)
            pool_query = self.non_cond_proj(pool_mem_feat) + self.pool_type_embed
            if pool_age is not None:
                pool_query = pool_query + self._age_embed(pool_age)

        # cond_bank_feat is laid out [frame-major mem queries | one obj_ptr per frame],
        # so the per-frame validity expands the same way.
        cond_valid_tokens = torch.cat(
            [cond_valid.repeat_interleave(self.n_query, dim=1), cond_valid], dim=1
        )
        pad_masks = {
            "non_cond": non_cond_valid,
            "cond": cond_valid_tokens,
            "pool": pool_valid,
        }

        # A dict, not a tuple: the heads consume different subsets of this.
        return {
            "image_spatial_query": image_spatial_query,
            "non_cond_bank_feat": non_cond_bank_feat,
            "cond_bank_feat": cond_bank_feat,
            "curr_mem_feat": curr_mem_feat,
            "pool_query": pool_query,
            "pad_masks": pad_masks,
        }

class BasePolicyNetwork(nn.Module):
    """Factored swap head: one query token per candidate and per bank slot, scored
    pairwise instead of independently.

    Query tokens are [noop, candidate memory (incoming frame, pool entries...), bank
    slots], refined through the same cross-/self-attention stack as before (candidates
    attend to conditioning frames + image, and to each other). That fixes the action
    indices: 0 = no-op (reject everything), 1..n_cand*M = swap(c, j) -- admit candidate
    c into bank slot j -- flattened candidate-major, action = 1 + c*M + j. Candidate 0
    is always the incoming frame; candidates 1.. are pool entries, in the order
    `pool_query` was built. This must stay in sync with sam2_video_predictor's
    agent_update_first_stage.

    The swap logit is a factored score rather than one more independent projection:

        logit_swap(c, j) = f_in(e_c) + f_out(e_bj) + g(e_c, e_bj)

    f_in/f_out score a candidate/slot in isolation (how good is this memory to admit /
    how good is this slot to give up, regardless of what it's paired with); g scores the
    specific pairing. This lets the head express an n_cand x M action grid from O(n_cand
    + M) refined embeddings instead of scoring each of the n_cand*M outcomes as its own
    independent query token.
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

        self.noop_embed = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.trunc_normal_(self.noop_embed, std=0.02)
        # QFormerBlock keeps the query self-attention in its own softmax, so candidates
        # can be compared against each other instead of competing with every context
        # token for attention mass. d_heads is the PER-HEAD width: CrossAttention builds
        # inner_dim = d_heads * n_heads, so this must be hidden_dim // n_heads.
        self.action_decoder = nn.ModuleList([
            QFormerBlock(hidden_dim, hidden_dim, n_heads, hidden_dim // n_heads, dropout=0.0)
            for _ in range(n_layers)
        ])

        self.noop_head = MLP(hidden_dim, hidden_dim, 1, num_layers=2, activation=QuickGELU)
        self.f_in = MLP(hidden_dim, hidden_dim, 1, num_layers=2, activation=QuickGELU)
        self.f_out = MLP(hidden_dim, hidden_dim, 1, num_layers=2, activation=QuickGELU)
        self.g = MLP(2 * hidden_dim, hidden_dim, 1, num_layers=2, activation=QuickGELU)

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
                pool_query=None, pad_masks=None):
        B = image_spatial_query.shape[0]
        M = non_cond_bank_feat.shape[1]
        n_cand = 1 if pool_query is None else 1 + pool_query.shape[1]
        dtype = image_spatial_query.dtype
        device = image_spatial_query.device
        noop_embed = self.noop_embed.to(dtype) + torch.zeros(B, 1, self.hidden_dim, dtype=dtype, device=device)

        cand_query = curr_mem_feat if pool_query is None else torch.cat([curr_mem_feat, pool_query], dim=1)
        action_query = torch.cat([noop_embed, cand_query, non_cond_bank_feat], dim=1)
        action_context = torch.cat([cond_bank_feat, image_spatial_query], dim=1)

        context_mask, self_mask = None, None
        if pad_masks is not None:
            # Image tokens are always real; noop and the incoming-frame candidate always exist.
            context_mask = torch.cat([
                pad_masks["cond"],
                torch.ones(B, image_spatial_query.shape[1], dtype=torch.bool, device=device),
            ], dim=1)
            self_mask = [
                torch.ones(B, 1 + curr_mem_feat.shape[1], dtype=torch.bool, device=device),
            ]
            if pool_query is not None:
                self_mask.append(pad_masks["pool"])
            self_mask.append(pad_masks["non_cond"])
            self_mask = torch.cat(self_mask, dim=1)

        for layer in self.action_decoder:
            action_query = layer(
                action_query, action_context,
                context_mask=context_mask, self_mask=self_mask,
            )

        e_noop, e_cand, e_bank = torch.split(action_query, (1, n_cand, M), dim=1)

        noop_logit = self.noop_head(e_noop).squeeze(-1)  # [B,1]
        f_in = self.f_in(e_cand).squeeze(-1)  # [B,n_cand]
        f_out = self.f_out(e_bank).squeeze(-1)  # [B,M]

        pair = torch.cat([
            e_cand.unsqueeze(2).expand(B, n_cand, M, self.hidden_dim),
            e_bank.unsqueeze(1).expand(B, n_cand, M, self.hidden_dim),
        ], dim=-1)  # [B,n_cand,M,2D]
        logit_swap = f_in.unsqueeze(-1) + f_out.unsqueeze(1) + self.g(pair).squeeze(-1)  # [B,n_cand,M]

        return torch.cat([noop_logit, logit_swap.flatten(1)], dim=1)  # [B, 1+n_cand*M]

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
            [PerceiverResampler(self.hidden_dim, 1, dropout=0.0) for _ in range(n_layers)]
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

    def forward(self, image_spatial_query, non_cond_bank_feat, cond_bank_feat, curr_mem_feat,
                pool_query=None, pad_masks=None):
        """V(s) for the state at the START of the timestep, before the swap decision."""
        B = image_spatial_query.shape[0]
        device = image_spatial_query.device
        value_query = self.value_query.expand(B, 1, self.hidden_dim)

        tokens = [curr_mem_feat, non_cond_bank_feat, cond_bank_feat, image_spatial_query]
        if pool_query is not None:
            tokens.append(pool_query)
        tokens = torch.cat(tokens, dim=1)

        context_mask = None
        if pad_masks is not None:
            context_mask = [
                torch.ones(B, curr_mem_feat.shape[1], dtype=torch.bool, device=device),
                pad_masks["non_cond"],
                pad_masks["cond"],
                torch.ones(B, image_spatial_query.shape[1], dtype=torch.bool, device=device),
            ]
            if pool_query is not None:
                context_mask.append(pad_masks["pool"])
            context_mask = torch.cat(context_mask, dim=1)

        for layer in self.value_decoder:
            value_query = layer(x_f=tokens, x=value_query, context_mask=context_mask)

        return self.value_proj(value_query).squeeze(1)


class BasePOAgent(BaseAgent):
    # The environment may ask select_action for more than one action per decision and
    # score them all; final_trajectory knows how to fold the extras into the buffer.
    # BaseAgent's other subclasses (the Q hierarchy) take no such argument.
    supports_action_resampling = True

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
        sam2_dim={},
        n_layers=2,
        target_kl=None,
    ):
        super().__init__(num_maskmem, policy_lr, gamma, beta, buffer_size, batch_size, device)
        self.feat_summarizer = BaseFeatureSummarizer(num_maskmem, **sam2_dim, n_layers=n_layers)
        self.policy_net = BasePolicyNetwork(self.feat_summarizer.hidden_dim, n_layers=n_layers)
        self.value_net = BaseValueNetwork(self.feat_summarizer.hidden_dim, n_layers=n_layers)

        self.optimizer = optim.AdamW([
            {"params": self.policy_net.parameters(),      "lr": policy_lr               },
            {"params": self.value_net.parameters(),       "lr": value_lr                },
            {"params": self.feat_summarizer.parameters(), "lr": max(policy_lr, value_lr)},
        ])
        
        # One cosine step per `update()` call, so T_max counts agent updates over the
        # whole run, not minibatches.
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=lr_T_max,
            eta_min=min_lr
        )

        self.tau = tau
        self.entropy_weight= entropy_weight
        # None (default) disables early stopping entirely -- update() always runs the
        # full num_ep epochs, exactly as before this option existed. When set, a
        # minibatch whose approx_kl exceeds this cuts the rest of that update() call
        # short: the ratio has moved far enough that further epochs on the same batch
        # of old_log_probs would be optimizing against a stale importance-sampling
        # estimate. Typical values are 0.01-0.05.
        self.target_kl = target_kl

        # For distributed training
        self.rank = 0
        self.distributed = False

        # Episodic returns accumulated between calls to `pop_episode_return_stats`,
        # used to log mean/std of episodic return to wandb once per epoch.
        self.episode_returns = []

        # Validation-time diagnostics, accumulated between calls to `pop_val_stats`.
        # Separate from episode_returns/replay_buffer on purpose: validation runs the
        # policy greedily and must never feed the training replay buffer or trajectory
        # bookkeeping, but the reward its decisions produce -- and what it actually
        # decided -- is exactly what tells you whether the policy generalizes.
        self.val_rewards = []
        self.val_actions = []
        self.val_action_entropies = []

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

        The trajectory spans all chunks of one (volume, obj_id) as a single episode --
        the memory bank is carried across chunk boundaries, so the only terminal is the
        last frame of the volume, marked by `set_await_done`.
        """
        if self.await_trajectory is None or len(self.await_trajectory.transitions) == 0:
            self.await_trajectory = None
            return

        log_probs, action, reward, done, curr_state, next_state = self.await_trajectory.get_transitions(self.device)

        # Episodic return: total (undiscounted) RAW reward per episode, which is now one
        # whole volume. It therefore scales with volume length — compare it across runs on
        # the same dataset, not across datasets. Taken before scaling so the logged number
        # stays in dice-loss units. The loop over segments still handles a multi-episode
        # trajectory, should one ever be produced.
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
        zeros = torch.zeros_like(reward)
        mc_return = compute_gae(zeros, zeros, reward, done, self.gamma, 1.0)
        self.return_scale.update(mc_return)
        reward = reward / self.return_scale.std

        # These are value *targets*: they must be computed with dropout off. update()
        # leaves the modules in train mode, so do not rely on the last select_action.
        self.feat_summarizer.eval()
        self.value_net.eval()

        with torch.no_grad():
            curr_feat = self.feat_summarizer(**curr_state)
            curr_value = self.value_net(**curr_feat)
            next_feat = self.feat_summarizer(**next_state)
            next_value = self.value_net(**next_feat)

            # V was trained on scaled returns, so it already lives in the scaled space;
            # feeding it scaled rewards keeps every term of the GAE recursion consistent.
            return_ = compute_gae(curr_value, next_value, reward, done, self.gamma, self.tau)
            return_ = return_.squeeze(-1)
            advantage = return_ - curr_value.squeeze(-1)

        scale = self.return_scale.std
        for i, ins in enumerate(self.await_trajectory.transitions):
            ins.set_return_advantage(return_[i].cpu(), advantage[i].cpu())
            self.replay_buffer.append(ins.get_updated())

            # Counterfactual actions scored at this same state (see select_action's
            # num_extra). A branch is never rolled out, so its multi-step return cannot
            # be measured -- but the difference between two actions at the same state is
            # exactly Q(s,a') - Q(s,a), and both halves of that are available: the
            # measured immediate reward, plus the critic's value of the successor each
            # action produces (attached one frame later by set_pending_branch_values).
            #
            #   A(s,a') = A_GAE(s,a) + [r' + g*V(s'_a')] - [r + g*V(s'_a)]
            #
            # so a branch inherits the taken action's full multi-step credit and then
            # departs from it by the whole one-step Q difference, long-horizon term
            # included -- not by the immediate reward alone, which would rank actions
            # myopically and hide exactly what a memory bank is for. It lands on the
            # same scale as the real advantage by construction, being that advantage
            # plus a difference of two quantities in the same (scaled) space, and it
            # collapses back to the reward difference when the critic sees no difference
            # between the successors, or at a terminal where there is no continuation.
            if not ins.extra_samples:
                continue
            terminal = float(done[i].item())
            for extra in ins.extra_samples:
                q_diff = (extra["reward"] - ins.reward) / scale
                extra_value, taken_value = extra.get("next_value"), ins.next_value
                if extra_value is not None and taken_value is not None:
                    # V is already in the scaled-return space the critic was trained in,
                    # matching the scaled reward difference above.
                    q_diff += self.gamma * (1.0 - terminal) * (extra_value - taken_value)
                self.replay_buffer.append(ins.get_side_sample(
                    extra,
                    return_[i].cpu() + q_diff,
                    advantage[i].cpu() + q_diff,
                ))

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

    def record_val_reward(self, reward):
        """Log-only counterpart to init_new_replay_instance: same reward signal, but
        for a greedy validation rollout, which must not touch the replay buffer."""
        self.val_rewards.append(reward)

    def record_val_action(self, action, entropy):
        """What the policy actually did on a greedy (validation) decision, and how
        confident it was. Called from select_action's non-training branch."""
        self.val_actions.append(action)
        self.val_action_entropies.append(entropy)

    def pop_val_stats(self):
        """Validation-time diagnostics accumulated since the last call, then clear.
        Empty dict if no validation decisions were recorded since the last call."""
        stats = {}
        if self.val_rewards:
            rewards = torch.tensor(self.val_rewards, dtype=torch.float32)
            stats["val_reward_mean"] = rewards.mean().item()
            stats["val_reward_std"] = rewards.std().item() if rewards.numel() > 1 else 0.0
            self.val_rewards = []
        if self.val_actions:
            actions = torch.tensor(self.val_actions, dtype=torch.float32)
            entropies = torch.tensor(self.val_action_entropies, dtype=torch.float32)
            # Action 0 is always no-op, regardless of pool size; see BasePolicyNetwork.
            stats["val_noop_frac"] = (actions == 0).float().mean().item()
            stats["val_action_entropy_mean"] = entropies.mean().item()
            self.val_actions = []
            self.val_action_entropies = []
        return stats

    @torch.no_grad()
    def estimate_values(self, states):
        """V for each state, as plain floats, under the current critic.

        Used by the environment to value the successors of counterfactual actions while
        the volume is still being tracked. That is the same critic `final_trajectory`
        will use minutes later -- the agent only updates between volumes -- and in any
        case only differences between these numbers are read, so they are consistent
        with each other by construction.
        """
        if not states:
            return []
        self.feat_summarizer.eval()
        self.value_net.eval()
        device = next(self.feat_summarizer.parameters()).device
        feats = stack_state_feats(states, device=device)
        values = self.value_net(**self.feat_summarizer(**feats))
        return values.reshape(-1).float().cpu().tolist()

    def set_pending_branch_values(self, taken_value, extra_values):
        """Attach the continuation estimates for the transition still awaiting closure.

        Called one frame after the decision, which is the earliest its successors can be
        built; the transition is still open at that point, so this reaches the right one.
        """
        instance = self.await_replay_instance
        if instance is None or not getattr(instance, "extra_samples", None):
            return
        instance.next_value = taken_value
        for extra, value in zip(instance.extra_samples, extra_values):
            extra["next_value"] = value

    def init_new_replay_instance(self, **instance_info):
        self.close_await_replay_instance(next_state=instance_info.get("state"))
        self.await_replay_instance = POReplayInstance(**instance_info)

    def store_transition(self, instance):
        """Hold transitions in the trajectory; they only reach the replay buffer once
        final_trajectory has run GAE over the whole volume."""
        self.await_trajectory.add_transition(instance)

    @torch.no_grad()
    def select_action(self, state: RLStates, valid_actions, training=False, num_extra=0):
        """Sample the swap/no-op action for this timestep.

        Only meaningful once the bank is full -- while it's still filling there is
        nothing to evict, so the caller inserts the incoming frame directly without
        going through the policy at all (see sam2_video_predictor.agent_update_first_stage).

        `num_extra` additionally draws that many *other* actions from the same
        distribution and returns them alongside. They are not executed: the caller scores
        each with the same counterfactual it uses for the taken action, and they come back
        as extra policy-gradient samples on this state. One decision otherwise yields one
        sample no matter how expensive the state was to reach, and this environment is
        expensive -- a frame's state costs a full tracking step, whereas scoring one more
        action costs a single forward.
        """
        self.feat_summarizer.eval()
        self.policy_net.eval()
        self.value_net.eval()

        device = next(self.feat_summarizer.parameters()).device

        feats = stack_state_feats([state], device=device)

        summary = self.feat_summarizer(**feats)
        action_logits = self.policy_net(**summary).squeeze(0)
        action_logits = action_logits.detach().cpu()

        valid_actions = torch.Tensor(valid_actions).to(torch.int64)
        action_mask = torch.zeros_like(action_logits, dtype=torch.bool)
        action_mask[valid_actions] = True

        valid_dist = Categorical(logits=action_logits.gather(0, valid_actions))
        valid_probs = valid_dist.probs

        if training:
            # Sample from the policy: the stored log prob must be the distribution the
            # action actually came from, otherwise the importance ratio is meaningless.
            # Exploration comes from this sampling plus the entropy bonus.
            action_idx = torch.multinomial(valid_probs, num_samples=1, replacement=False)
        else:
            action_idx = torch.argmax(valid_probs, keepdim=True)

        action = valid_actions[action_idx].item()
        valid_log_probs = valid_probs.log()

        # Without replacement and excluding the action taken: each extra action costs the
        # caller a full SAM2 forward to score, and a duplicate would buy nothing. Their
        # log-probs are read off the same (full valid-support) distribution as the taken
        # action's, so the PPO ratio means the same thing for all of them.
        extra_actions, extra_log_probs = [], []
        if training and num_extra > 0 and valid_actions.numel() > 1:
            residual = valid_probs.clone()
            residual[action_idx] = 0.0
            k = min(int(num_extra), int((residual > 0).sum().item()))
            if k > 0:
                extra_idx = torch.multinomial(residual, num_samples=k, replacement=False)
                extra_actions = valid_actions[extra_idx].tolist()
                extra_log_probs = valid_log_probs[extra_idx].tolist()

        if not training:
            # Greedy decisions only happen outside training (validation, or any other
            # inference-only rollout) -- track what the policy actually does and how
            # confident it is, for pop_val_stats.
            self.record_val_action(action, valid_dist.entropy().item())

        return {
            "action": action,
            # Scalar, not a 1-element list: a list makes old_log_probs [B,1,1] in
            # train_step and the PPO ratio broadcasts into a [B,B,1] outer product.
            "log_probs": valid_log_probs[action_idx].item(),
            "action_mask": action_mask,
            "policy_weight": 1.0,
            # Empty unless num_extra was asked for; the caller scores these and hands
            # them back through init_new_replay_instance's `extra_samples`.
            "extra_actions": extra_actions,
            "extra_log_probs": extra_log_probs,
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

        total_policy_loss, total_value_loss, total_policy_gradnorm, total_value_gradnorm, total_summ_gradnorm = 0, 0, 0, 0, 0
        num_update = 0
        metric_sums, metric_counts = {}, {}
        stopped_early, stopped_ep = False, num_ep - 1

        for ep in range(num_ep):
            # Full pass over the replay buffer, shuffled and split into batch_size minibatches.
            shuffle_indice = np.random.permutation(buffer_size)

            for start in range(0, buffer_size, self.batch_size):
                batch_indice = shuffle_indice[start:start + self.batch_size]
                batch = [self.replay_buffer[idx] for idx in batch_indice]

                (
                    value_loss, policy_loss, 
                    policy_gradnorm, value_gradnorm, summ_gradnorm,
                    metrics
                ) = self.train_step(batch)

                total_policy_loss += policy_loss
                total_policy_gradnorm += policy_gradnorm
                total_value_loss += value_loss
                total_value_gradnorm += value_gradnorm
                total_summ_gradnorm += summ_gradnorm
                num_update += 1

                for k, v in metrics.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + v
                    metric_counts[k] = metric_counts.get(k, 0) + 1

                if self.target_kl is not None and metrics["approx_kl"] > self.target_kl:
                    print(
                        f"Early stopping at epoch {ep} after {num_update} minibatch "
                        f"updates: approx_kl {metrics['approx_kl']:.4f} > "
                        f"target_kl {self.target_kl}"
                    )
                    stopped_early = True
                    break

            if stopped_early:
                stopped_ep = ep
                break

        # Logged before stepping so the value matches the LR the minibatches above ran at.
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.step_lr_scheduler()

        out = {
            "actor_loss": total_policy_loss / num_update,
            "critic_loss": total_value_loss / num_update,
            "policy_gradnorm": total_policy_gradnorm / num_update,
            "value_gradnorm": total_value_gradnorm / num_update,
            "summ_gradnorm": total_summ_gradnorm / num_update,
            "agent_lr": current_lr,
            # Raw dice-loss units per unit of scaled return; multiply episodic_return_*
            # by this to compare against an unscaled run.
            "return_scale": self.return_scale.std,
            "stopped_ep": stopped_ep
        }
        for k, total in metric_sums.items():
            out[k] = total / metric_counts[k]
        # Mean/std of episodic return since the last update, if any trajectory finished.
        out.update(self.pop_episode_return_stats())

        return out

    def train_step(self, batch):
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
            value_weights,
        ) = zip(*batch)

        feats = stack_state_feats(states, device=device)

        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        advantages = torch.FloatTensor(advantages).unsqueeze(1)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        action_masks = torch.stack(action_masks)
        policy_weights = torch.FloatTensor(policy_weights).unsqueeze(1)
        # 0 for the counterfactual branches: they share a state with the transition they
        # were sampled next to, so fitting the critic to their returns as well would pull
        # V(s) towards the mean over sampled actions instead of V under the policy.
        value_weights = torch.FloatTensor(value_weights).unsqueeze(1)

        actions = actions.to(device=device, non_blocking=True)
        rewards = rewards.to(device=device, dtype=torch.float32, non_blocking=True)
        old_log_probs = old_log_probs.to(device=device, dtype=torch.float32, non_blocking=True)
        dones = dones.to(device=device, dtype=torch.float32, non_blocking=True)
        advantages = advantages.to(device=device, dtype=torch.float32, non_blocking=True)
        returns = returns.to(device=device, dtype=torch.float32, non_blocking=True)
        action_masks = action_masks.to(device=device, non_blocking=True)
        policy_weights = policy_weights.to(device=device, dtype=torch.float32, non_blocking=True)
        value_weights = value_weights.to(device=device, dtype=torch.float32, non_blocking=True)

        # Normalize over exactly the transitions that enter the policy loss. The forced
        # bank-filling steps carry weight 0 -- memory_bank_size of them, once per volume
        # now that the bank is not rebuilt per chunk -- so folding them in would rescale
        # the real decisions by a mean/std they never contribute a gradient to.
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

        def masked_readout(logits, mask, taken):
            """Log-prob of the action taken, and -H, renormalized over exactly the support
            it was sampled from -- otherwise the ratio against old_log_probs is biased and
            the entropy bonus pushes mass onto unreachable actions. A support of size 1
            falls out as log-prob 0 and entropy 0.
            """
            logits = logits.masked_fill(~mask, float("-inf"))
            log_probs = torch.log_softmax(logits, dim=1)
            probs = log_probs.exp()  # exactly 0 on masked actions
            # nan_to_num avoids the 0 * -inf = nan on masked actions
            minus_entropy = (probs * torch.nan_to_num(log_probs, neginf=0.0)).sum(dim=1, keepdim=True)
            return log_probs.gather(1, taken), minus_entropy

        with torch.enable_grad():
            summary = self.feat_summarizer(**feats)
            action_logits = self.policy_net(**summary)

            log_action_probs, minus_entropy = masked_readout(
                action_logits, action_masks, actions
            )

            policy_loss = self.compute_policy_loss(log_action_probs, advantages, old_log_probs)
            policy_loss = policy_loss + minus_entropy * self.entropy_weight # entropy regularization
            # Forced (bank-filling) transitions carry weight 0
            policy_loss = (policy_loss * policy_weights).sum() / weight_sum

            with torch.no_grad():
                # Forced (bank-filling) transitions carry no policy signal, so they're
                # excluded (weight 0) from these stats just like from the loss above.
                policy_entropy = (-minus_entropy * policy_weights).sum() / weight_sum

                log_ratio = log_action_probs - old_log_probs
                ratio = log_ratio.exp()
                ratio_mean = (ratio * policy_weights).sum() / weight_sum
                ratio_var = (((ratio - ratio_mean) ** 2) * policy_weights).sum() / weight_sum
                ratio_std = ratio_var.clamp(min=0).sqrt()

                # Schulman's k3 estimator (http://joschu.net/blog/kl-approx.html):
                # unbiased and always >=0 in expectation, unlike the naive -log_ratio
                # mean, which can go negative on a single minibatch from sampling noise
                # alone and makes a poor early-stopping trigger.
                approx_kl = (((ratio - 1) - log_ratio) * policy_weights).sum() / weight_sum

                metrics = {
                    "policy_entropy": policy_entropy.item(),
                    "ratio_mean": ratio_mean.item(),
                    "ratio_std": ratio_std.item(),
                    "approx_kl": approx_kl.item(),
                    "adv_mean": adv_mean.item(),
                    "adv_std": adv_std.item(),
                }
                if hasattr(self, "epsilon"):
                    clipped = (ratio < 1.0 - self.epsilon) | (ratio > 1.0 + self.epsilon)
                    metrics["clip_fraction"] = (clipped.float() * policy_weights).sum().item() / weight_sum.item()

            # Same encode as the policy.
            pred_value = self.value_net(**summary)
            value_weight_sum = value_weights.sum().clamp(min=1.0)
            value_loss = (
                ((pred_value - returns) ** 2) * value_weights
            ).sum() / value_weight_sum

            with torch.no_grad():
                # Explained variance over the same rows the loss above fits: a branch's
                # return is the real one shifted by a reward difference, so folding them
                # in would inflate the target variance the critic is scored against.
                fitted = value_weights.reshape(-1) > 0
                if fitted.any():
                    fitted_returns = returns[fitted]
                    fitted_pred = pred_value[fitted]
                    return_var = fitted_returns.var(unbiased=False)
                    explained_variance = 1.0 - (fitted_returns - fitted_pred).var(unbiased=False) / return_var.clamp(min=1e-8)
                    metrics["explained_variance"] = explained_variance.item()
                # Share of this minibatch that came from extra sampled actions rather
                # than from steps actually taken.
                metrics["side_sample_frac"] = (1.0 - value_weights).mean().item()

            total_loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            policy_gradnorm = nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            summ_gradnorm = nn.utils.clip_grad_norm_(self.feat_summarizer.parameters(), max_norm=0.5)
            value_gradnorm = nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            self.optimizer.step()

        return (
            value_loss.detach(), policy_loss.detach(), 
            policy_gradnorm, value_gradnorm, summ_gradnorm, 
            metrics
        )

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

