import copy

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
    
    # Update after
    def update(self, loss_after, next_state):
        self.loss_after = loss_after
        self.next_state = next_state
        
        loss_diff = self.loss_before - self.loss_after
        self.reward = self.reward + loss_diff

    def set_done(self, loss_after):
        self.loss_after = loss_after
        self.next_state = self.state
        
        loss_diff = self.loss_before - self.loss_after
        self.reward = self.reward + loss_diff
        self.done = True