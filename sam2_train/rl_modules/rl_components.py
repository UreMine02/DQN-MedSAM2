import copy
import torch

# -------------------------
# Global memory pool
# -------------------------
class GlobalMemoryPool:
    """A fixed-budget archive of past frames the agent may recall into the memory bank.

    Every `stride` frames the tracker pushes that frame's encoded memory here, keyed by
    its index in the whole volume. Training now keeps one tracking state alive across all
    of a volume's chunks, so the memory bank survives chunk boundaries on its own and
    frame numbering is volume-global everywhere; the pool's job is no longer to bridge a
    reset but to hold a uniform subsample of frames that have already been evicted from
    the (much smaller) bank, so the agent can recall one.

    When the pool overflows it does not evict the oldest entry: that would turn a "global"
    pool into a second, slower FIFO covering only the recent past. Instead the stride
    doubles and every other entry is dropped, so the surviving entries stay a uniform
    subsample of everything seen so far, under a constant budget. Because entries always
    sit at multiples of the stride at the time they were pushed, "keep the multiples of
    2*stride" removes exactly the odd ones -- a halving, not an arbitrary cull -- and it
    is deterministic, which reservoir sampling would not be at eval time.

    Entries are detached on push. A chunk's autograd graph is freed by its own backward,
    so holding a live graph tensor across chunks would either leak or blow up on the next
    backward; recalled memories therefore condition SAM2 without passing gradient back
    into the memory encoder that produced them. (The bank carried across a chunk boundary
    is detached for the same reason, by `train_advance_chunk`.)
    """

    def __init__(self, capacity=0, stride=4):
        self.capacity = int(capacity)
        self.stride = max(int(stride), 1)
        self.entries = []  # list of {"global_idx": int, "out": {...}}, ascending
        # Cleared on every mutation, so all the frames between two pushes share one
        # stacked snapshot. That sharing is what keeps a pool-carrying replay buffer from
        # costing `capacity` MB per stored transition.
        self._snapshot_cache = {}

    def __len__(self):
        return len(self.entries)

    @property
    def enabled(self):
        return self.capacity > 0

    def maybe_push(self, global_idx, out):
        """Archive `out` if `global_idx` lands on the current stride."""
        if not self.enabled or global_idx % self.stride != 0:
            return False
        if any(e["global_idx"] == global_idx for e in self.entries):
            return False

        maskmem_pos_enc = out.get("maskmem_pos_enc")
        self.entries.append({
            "global_idx": global_idx,
            "out": {
                "maskmem_features": out["maskmem_features"].detach(),
                "maskmem_pos_enc": (
                    None if maskmem_pos_enc is None
                    else [x.detach() for x in maskmem_pos_enc]
                ),
                "obj_ptr": out["obj_ptr"].detach(),
            },
        })
        self.entries.sort(key=lambda e: e["global_idx"])

        while len(self.entries) > self.capacity:
            self.stride *= 2
            self.entries = [e for e in self.entries if e["global_idx"] % self.stride == 0]

        self._snapshot_cache = {}
        return True

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
        curr_iou_score=None,
        prev_memory_bank=None,
        global_pool=None,
        cand_age=None,
        bank_age=None,
    ):
        self.frame_ix = frame_idx
        self.next_image_feat = next_image_feat
        self.curr_memory_feat = curr_memory_feat
        self.curr_iou_score = curr_iou_score
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