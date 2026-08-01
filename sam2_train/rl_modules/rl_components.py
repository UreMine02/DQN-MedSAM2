import copy
import torch

# -------------------------
# Define Action Space
# -------------------------
ACTION_SPACE = {
    0: "add",
    1: "skip",
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