import copy
import torch
import torch.nn.functional as F

# -------------------------
# Global memory pool
# -------------------------
class GlobalMemoryPool:
    """A fixed-budget archive of past frames the agent may recall into the memory bank.

    Every `stride` frames the tracker pushes that frame's encoded memory here, keyed by
    its index in the whole volume. Training now keeps one tracking state alive across all
    of a volume's chunks, so the memory bank survives chunk boundaries on its own and
    frame numbering is volume-global everywhere; the pool's job is no longer to bridge a
    reset but to hold a subsample of frames that have already been evicted from the (much
    smaller) bank, so the agent can recall one.

    Two admission policies decide *which* subsample that is:

    `policy="stride"` (the default, and what every pre-existing run used): take every
    `stride`-th frame, and on overflow do not evict the oldest entry -- that would turn a
    "global" pool into a second, slower FIFO covering only the recent past. Instead the
    stride doubles and every other entry is dropped, so the survivors stay a uniform
    subsample of everything seen so far under a constant budget. Because entries always
    sit at multiples of the stride at the time they were pushed, "keep the multiples of
    2*stride" removes exactly the odd ones -- a halving, not an arbitrary cull -- and it
    is deterministic, which reservoir sampling would not be.

    `policy="diverse"`: uniform coverage of *time* is a proxy for what actually makes a
    recallable memory useful, which is that it shows the object in a configuration the
    rest of the pool does not. So a frame is admitted only if its predicted mask is novel
    -- its IoU against every entry already pooled stays below `novelty_iou` -- and once
    the pool is full, admitting one entry evicts whichever member is then the most
    redundant, greedily minimising the pool's worst (`div_obj="minmax"`) or total
    (`"minsum"`) pairwise IoU. `stride` still applies first, as a cheap pre-filter: it
    bounds how often the pool can mutate, and every mutation invalidates the snapshot
    cache below, which is what keeps a pool-carrying replay buffer affordable.

    Entries are detached on push. A chunk's autograd graph is freed by its own backward,
    so holding a live graph tensor across chunks would either leak or blow up on the next
    backward; recalled memories therefore condition SAM2 without passing gradient back
    into the memory encoder that produced them. (The bank carried across a chunk boundary
    is detached for the same reason, by `train_advance_chunk`.)
    """

    def __init__(
        self,
        capacity=0,
        stride=4,
        policy="stride",
        novelty_iou=0.5,
        min_area=1e-4,
        mask_res=64,
        div_obj="minmax",
    ):
        self.capacity = int(capacity)
        self.stride = max(int(stride), 1)
        self.policy = policy
        self.novelty_iou = float(novelty_iou)
        self.min_area = float(min_area)
        self.mask_res = int(mask_res)
        self.div_obj = div_obj
        self.entries = []  # list of {"global_idx": int, "out": {...}}, ascending
        # Cleared on every mutation, so all the frames between two pushes share one
        # stacked snapshot. That sharing is what keeps a pool-carrying replay buffer from
        # costing `capacity` MB per stored transition.
        self._snapshot_cache = {}
        # Per-volume telemetry; the caller sums it over volumes for the epoch's log.
        self.stats = {
            "offered": 0,
            "admitted": 0,
            "replaced": 0,
            "rejected_degenerate": 0,
            "rejected_redundant": 0,
            "rejected_crowded": 0,
        }

    def __len__(self):
        return len(self.entries)

    @property
    def enabled(self):
        return self.capacity > 0

    def maybe_push(self, global_idx, out):
        """Archive `out` if `global_idx` lands on the current stride and the policy takes it."""
        if not self.enabled or global_idx % self.stride != 0:
            return False
        if any(e["global_idx"] == global_idx for e in self.entries):
            return False
        if self.policy == "diverse":
            return self._push_diverse(global_idx, out)
        return self._push_strided(global_idx, out)

    def _make_entry(self, global_idx, out, mask=None):
        maskmem_pos_enc = out.get("maskmem_pos_enc")
        entry = {
            "global_idx": global_idx,
            "out": {
                "maskmem_features": out["maskmem_features"].detach(),
                "maskmem_pos_enc": (
                    None if maskmem_pos_enc is None
                    else [x.detach() for x in maskmem_pos_enc]
                ),
                "obj_ptr": out["obj_ptr"].detach(),
            },
        }
        if mask is not None:
            # Only the diverse policy needs a mask fingerprint, and only it pays for one.
            entry["mask"] = mask
        return entry

    def _push_strided(self, global_idx, out):
        self.stats["offered"] += 1
        self.entries.append(self._make_entry(global_idx, out))
        self.entries.sort(key=lambda e: e["global_idx"])

        while len(self.entries) > self.capacity:
            self.stride *= 2
            self.entries = [e for e in self.entries if e["global_idx"] % self.stride == 0]

        self.stats["admitted"] += 1
        self._snapshot_cache = {}
        return True

    def _push_diverse(self, global_idx, out):
        """Admit only a frame whose mask is novel; at capacity, give up the most redundant one."""
        self.stats["offered"] += 1
        mask = self._fingerprint(out)
        # An empty (or near-empty) prediction has IoU 0 against everything, so it reads as
        # maximally novel and, once pooled, is never the one evicted. A frame where the
        # tracker lost the object would therefore squat in the pool for the rest of the
        # volume and be offered to the agent as a recall candidate -- the one failure mode
        # a pure diversity objective cannot see.
        if mask is None or float(mask.sum()) < self.min_area * mask.numel():
            self.stats["rejected_degenerate"] += 1
            return False

        iou = None
        if self.entries:
            iou = self._iou_matrix(
                torch.stack([e["mask"] for e in self.entries] + [mask])
            )
            if float(iou[-1, :-1].max()) >= self.novelty_iou:
                self.stats["rejected_redundant"] += 1
                return False

        if len(self.entries) < self.capacity:
            self._insert(self._make_entry(global_idx, out, mask=mask))
            return True

        drop = self._select_drop(iou, global_idx)
        if drop == len(self.entries):
            # The incoming frame is itself the pool's most redundant member: it cleared
            # `novelty_iou` against each entry individually, yet swapping it in would not
            # improve the objective. Keep what is already there.
            self.stats["rejected_crowded"] += 1
            return False

        self.entries.pop(drop)
        self.stats["replaced"] += 1
        self._insert(self._make_entry(global_idx, out, mask=mask))
        return True

    def _insert(self, entry):
        """Add `entry`, keeping `entries` ascending by frame index.

        `local_view` and `snapshot` are read positionally -- as pool slot p in the action
        space and in the agent's state -- and `prepare_rl_state` reads each slot's age off
        the same ordering, so the list has to stay sorted however it was mutated.
        """
        self.entries.append(entry)
        self.entries.sort(key=lambda e: e["global_idx"])
        self.stats["admitted"] += 1
        self._snapshot_cache = {}

    def _fingerprint(self, out):
        """Per-cell foreground occupancy of this frame's prediction, flattened.

        Average-pooled from the low-res logits rather than max-pooled: occupancy keeps the
        object's *area* proportional under the downsample, where max-pooling would inflate
        a thin or tiny structure to a full cell. Paired with the min/max (fuzzy) Jaccard in
        `_iou_matrix`, a downsampled pair of identical masks still scores IoU 1, which a
        product-based IoU over fractional cells would not -- it collapses towards 0 exactly
        for the small lesions these volumes are full of.

        `None` when the frame has no usable prediction: the object-score head says the
        object is absent, or the output carries no mask at all.
        """
        logits = out.get("pred_masks")
        if logits is None:
            return None
        score = out.get("object_score_logits")
        if score is not None and float(score.max()) <= 0:
            return None
        logits = logits.detach().float()
        h, w = logits.shape[-2:]
        # [B, 1, H, W] -> [H, W]; tracking here is per-object, so B is 1 in practice.
        binary = (logits.reshape(-1, h, w).amax(dim=0) > 0).float()[None, None]
        occ = F.adaptive_avg_pool2d(binary, self.mask_res)
        # CPU-resident: a few KB per entry, and it keeps the pairwise pass off the device
        # the tracker is busy with.
        return occ.reshape(-1).cpu()

    @staticmethod
    def _iou_matrix(masks):
        """Pairwise fuzzy Jaccard over [N, cells] occupancies -- exact IoU when binary.

        Row at a time: the full broadcast is N*N*cells floats, which a large pool at a
        high mask_res would blow up for no gain, while N rows of N*cells stay negligible.
        """
        n = masks.shape[0]
        iou = masks.new_zeros((n, n))
        for i in range(n):
            inter = torch.minimum(masks[i], masks).sum(dim=1)
            union = torch.maximum(masks[i], masks).sum(dim=1)
            iou[i] = inter / union.clamp_min(1e-6)
        return iou

    def _select_drop(self, iou, cand_idx):
        """Index in [entries + candidate] of the member to give up.

        `minmax` drops one end of the closest pair in the pool, which is a greedy step on
        max-min dispersion -- it minimises the pool's *worst* redundancy. `minsum` drops
        the member closest to all the others, minimising the average instead. Ties break
        towards the other objective, then towards the newer frame, so a tie never evicts
        the older entry whose coverage is harder to replace.
        """
        idxs = [e["global_idx"] for e in self.entries] + [cand_idx]
        off_diag = iou.clone()
        off_diag.fill_diagonal_(-1.0)
        nearest = off_diag.max(dim=1).values
        total = off_diag.clamp_min(0.0).sum(dim=1)
        primary, secondary = (
            (nearest, total) if self.div_obj == "minmax" else (total, nearest)
        )
        return max(
            range(len(idxs)),
            key=lambda i: (float(primary[i]), float(secondary[i]), idxs[i]),
        )

    def summary(self):
        """Counters plus the gauges that say whether the diversity objective is working."""
        mean_iou, max_iou = 0.0, 0.0
        if self.policy == "diverse" and len(self.entries) > 1:
            iou = self._iou_matrix(torch.stack([e["mask"] for e in self.entries]))
            off_diag = ~torch.eye(iou.shape[0], dtype=torch.bool)
            mean_iou = float(iou[off_diag].mean())
            max_iou = float(iou[off_diag].max())
        return {
            **self.stats,
            "size": len(self.entries),
            "stride": self.stride,
            "mean_pair_iou": mean_iou,
            "max_pair_iou": max_iou,
        }

    def local_view(self):
        """(frame_idx, out) pairs, oldest first.

        The tracker keys frames by their volume-global index for the whole trajectory, so
        a pooled entry can be dropped straight back into `non_cond_frame_outputs` under
        its own key: it cannot collide with a live frame, and `frame_idx - key` stays the
        true temporal distance that both the memory-bank ordering and SAM2's obj_ptr
        encoding read off the key.
        """
        return [(e["global_idx"], e["out"]) for e in self.entries]

    def snapshot(self, memory_shape, obj_ptr_shape, to_cpu):
        """Zero-padded [1, capacity, ...] stacks of the pooled memories, cached per version.

        `mem_feat` matches what prepare_rl_state builds for the bank (features plus the
        first positional encoding) so the agent's summarizer sees pool and bank slots in
        the same space. Returned tensors are shared, not cloned: every RL state built from
        this pool version points at the same storage.
        """
        key = bool(to_cpu)
        cached = self._snapshot_cache.get(key)
        if cached is not None:
            return cached

        mem_feats, obj_ptrs, valid = [], [], []
        for entry in self.entries[: self.capacity]:
            out = entry["out"]
            mem_feat = out["maskmem_features"]
            if out["maskmem_pos_enc"] is not None:
                mem_feat = mem_feat + out["maskmem_pos_enc"][0]
            obj_ptr = out["obj_ptr"]
            if to_cpu:
                mem_feat = mem_feat.cpu()
                obj_ptr = obj_ptr.cpu()
            mem_feats.append(mem_feat)
            obj_ptrs.append(obj_ptr)
            valid.append(True)

        device = torch.device("cpu") if to_cpu else (
            mem_feats[0].device if mem_feats else torch.device("cpu")
        )
        while len(mem_feats) < self.capacity:
            mem_feats.append(torch.zeros(memory_shape, device=device))
            obj_ptrs.append(torch.zeros(obj_ptr_shape, device=device))
            valid.append(False)

        snapshot = {
            "mem_feat": torch.stack(mem_feats, dim=1),
            "obj_ptr": torch.stack(obj_ptrs, dim=1),
            "valid": torch.tensor(valid, device=device).reshape(1, -1),
        }
        self._snapshot_cache[key] = snapshot
        return snapshot


# -------------------------
# States
# -------------------------
class RLStates:
    def __init__(
        self,
        frame_idx=None,
        next_image_feat=None,
        curr_memory_feat=None,
        prev_memory_bank=None,
        global_pool=None,
        cand_age=None,
        bank_age=None,
    ):
        self.frame_ix = frame_idx
        self.next_image_feat = next_image_feat
        self.curr_memory_feat = curr_memory_feat
        self.prev_memory_bank = prev_memory_bank
        # None when the pool is disabled, which keeps every downstream module on its
        # pre-pool code path. Otherwise {"mem_feat", "obj_ptr", "valid", "age"}.
        self.global_pool = global_pool
        # How many frames back the candidate / each bank slot was captured. A recalled
        # memory can be hundreds of frames old, a gap the summarizer's per-rank recency
        # prior cannot express on its own.
        self.cand_age = cand_age
        self.bank_age = bank_age

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
        if self.cand_age is not None:
            self.cand_age = self.cand_age.detach().cpu()
        if self.bank_age is not None:
            self.bank_age = self.bank_age.detach().cpu()
        if self.global_pool is not None:
            # Rebind rather than mutate: the pool dict is shared with every other state
            # built from the same pool version, and its tensors are owned by the pool's
            # snapshot cache.
            self.global_pool = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in self.global_pool.items()
            }

 
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
    
    def close(self, next_state=None, done=False):
        """Attach the successor state and the terminal flag.

        The reward is already final by the time a transition is opened: both sides of
        the counterfactual (loss_before / loss_after) are measured on the same frame,
        back to back, before the environment moves on. So closing is bookkeeping only.

        `next_state` is shared by reference with the following transition's state rather
        than materialized again -- they are the same snapshot, and a state is ~16MB
        (a [1,256,64,64] image feature plus an 11-slot memory bank). A terminal
        transition self-loops instead; nothing reads it, since done=1 masks the
        bootstrap.
        """
        self.next_state = self.state if (done or next_state is None) else next_state
        self.done = done