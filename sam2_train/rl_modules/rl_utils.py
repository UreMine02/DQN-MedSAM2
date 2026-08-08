"""
Q-learning utilities cho SAM2 memory management.
"""
import math
import contextlib
import torch
import torch.nn as nn
from functools import partial
from monai.losses import DiceLoss, FocalLoss
from sam2_train.rl_modules.rl_components import RLStates
from sam2_train.modeling.position_encoding import compute_axial_cis

EPS = 1e-6

@contextlib.contextmanager
def deterministic_dropout(model):
    """Silence dropout for the duration of the block, without touching `self.training`.

    The RL reward is a counterfactual: loss with the memory bank before the agent's
    action minus loss after it. Both sides have to come from the same deterministic
    function of the bank, otherwise the difference carries dropout noise -- and a "skip"
    action, which provably leaves the bank untouched, still earns a nonzero random
    reward. SAM2's memory attention runs at dropout 0.1, so under net.train() that noise
    is always present.

    Calling .eval() would fix the noise but is the wrong tool: SAM2 also gates
    conditioning obj_ptr selection (sam2_base ~L690), non-overlap mask constraints
    (~L919) and memory-encoder mask binarization (~L928) on self.training, so an
    eval-mode measurement would score a *different* system from the one being trained.
    Here only the stochasticity is removed, so SAM2 keeps training with dropout and
    keeps co-adapting to the bank distribution the agent produces.

    Covers both ways SAM2 expresses dropout: nn.Dropout modules (memory_attention) and
    the plain `dropout_p` attribute read by scaled_dot_product_attention
    (sam/transformer).
    """
    saved = []
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            saved.append((module, "p", module.p))
            module.p = 0.0
        if hasattr(module, "dropout_p"):
            saved.append((module, "dropout_p", module.dropout_p))
            module.dropout_p = 0.0
    try:
        yield
    finally:
        for module, attr, value in saved:
            setattr(module, attr, value)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_enc(
    xq: torch.Tensor,
    freqs_cis: torch.Tensor
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq).to(xq.device)


def rotary_emb(x, internal_dim=256, num_heads=1, rope_theta=10000.0):
    compute_cis = partial(
        compute_axial_cis, dim=internal_dim // num_heads, theta=rope_theta
    )
    
    # Apply rotary position encoding
    w = h = math.sqrt(x.shape[-2])
    freqs_cis = compute_cis(end_x=w, end_y=h).to(x.device)

    x = apply_rotary_enc(
        x,
        freqs_cis=freqs_cis
    )
    
    return x

def prepare_rl_state(
    current_vision_feats,
    current_vision_pos_embeds,
    output_dict,
    frame_idx,
    num_maskmem,
    num_max_prompt=10,
    offload_to_cpu=True,
    training=False,
    global_pool=None,
):
    next_image_feat = current_vision_feats[-1] + current_vision_pos_embeds[-1]
    next_image_feat = next_image_feat.permute(1, 2, 0).reshape(1, 256, 64, 64)
    candidate_key = frame_idx - 1
    curr_memory_feat = output_dict["await_outputs"][candidate_key]
    curr_memory_feat = curr_memory_feat["maskmem_features"] + curr_memory_feat["maskmem_pos_enc"][0]
    curr_obj_ptr = output_dict["await_outputs"][candidate_key]["obj_ptr"]

    # Add non_cond memory
    cond_bank_list = list(output_dict["cond_frame_outputs"].values())
    non_cond_bank_list = list(output_dict["non_cond_frame_outputs"].values())
    memory_shape = cond_bank_list[0]["maskmem_features"].shape
    obj_ptr_shape = cond_bank_list[0]["obj_ptr"].shape
    device = "cpu" if offload_to_cpu else cond_bank_list[0]["maskmem_features"].device
    prev_memory_bank = []
    prev_obj_ptr = []
    for feat in non_cond_bank_list:
        mem_feat = feat["maskmem_features"] + feat["maskmem_pos_enc"][0]
        obj_ptr = feat["obj_ptr"]
        if offload_to_cpu:
            mem_feat = mem_feat.cpu()
            obj_ptr = obj_ptr.cpu()
        prev_memory_bank.append(mem_feat)
        prev_obj_ptr.append(obj_ptr)
    
    # Append zero memory
    while len(prev_memory_bank) < num_maskmem:
        prev_memory_bank.append(torch.zeros(memory_shape, device=device))
        prev_obj_ptr.append(torch.zeros(obj_ptr_shape, device=device))

    # Add cond memory
    for feat in cond_bank_list:
        mem_feat = feat["maskmem_features"] + feat["maskmem_pos_enc"][0]
        obj_ptr = feat["obj_ptr"]
        if offload_to_cpu:
            mem_feat = mem_feat.cpu()
            obj_ptr = obj_ptr.cpu()
        prev_memory_bank.append(mem_feat)
        prev_obj_ptr.append(obj_ptr)
    
    # Append zero memory
    while len(prev_memory_bank) < num_maskmem + num_max_prompt:
        prev_memory_bank.append(torch.zeros(memory_shape, device=device))
        prev_obj_ptr.append(torch.zeros(obj_ptr_shape, device=device))
    
    prev_memory_bank = torch.stack(prev_memory_bank, dim=1)
    prev_obj_ptr = torch.stack(prev_obj_ptr, dim=1)
    
    if offload_to_cpu:
        next_image_feat = next_image_feat.detach().cpu()
        curr_memory_feat = curr_memory_feat.detach().cpu()
        curr_obj_ptr = curr_obj_ptr.detach().cpu()

    list_frame = list(output_dict["non_cond_frame_outputs"].keys())

    # Ages only exist when the pool does. With it disabled the state carries None and the
    # summarizer skips the age term entirely, so the pool-off path stays bit-identical to
    # the pre-pool one.
    cand_age, bank_age, pool_state = None, None, None
    if global_pool is not None and global_pool.enabled:
        age_device = torch.device("cpu") if offload_to_cpu else device
        cand_age = torch.tensor(
            [[float(frame_idx - candidate_key)]], device=age_device
        )
        # Zero for the unoccupied slots prepare_rl_state zero-fills above; they are masked
        # out downstream, so the value is never read.
        bank_ages = [float(frame_idx - k) for k in list_frame]
        bank_ages += [0.0] * (num_maskmem - len(bank_ages))
        bank_age = torch.tensor([bank_ages[:num_maskmem]], device=age_device)

        pool_state = dict(global_pool.snapshot(memory_shape, obj_ptr_shape, offload_to_cpu))
        pool_ages = [
            float(frame_idx - pooled_key)
            for pooled_key, _ in global_pool.local_view()
        ][: global_pool.capacity]
        pool_ages += [0.0] * (global_pool.capacity - len(pool_ages))
        pool_state["age"] = torch.tensor([pool_ages], device=age_device)

    rl_state = {
        "frame_idx": frame_idx,
        "next_image_feat": next_image_feat.clone().detach(),
        "curr_memory_feat": {
            "mem_feat": curr_memory_feat.clone().detach(),
            "obj_ptr": curr_obj_ptr.clone().detach(),
        },
        "prev_memory_bank": {
            "mem_feat": prev_memory_bank.clone().detach(),
            "obj_ptr": prev_obj_ptr.clone().detach(),
        },
        # Shared, not cloned: the pool mutates once every `stride` frames, so cloning it
        # per state would multiply the replay buffer's footprint by the pool size for no
        # gain -- states are read-only once built.
        "global_pool": pool_state,
        "cand_age": cand_age,
        "bank_age": bank_age,
    }

    state = RLStates(**rl_state)
    # Bank slot k (the k-th non_cond_bank_feat token the summarizer builds) holds this
    # frame key. Callers turn a swap action's slot index into a frame key by indexing
    # this list directly -- there is no action-index offset baked in here anymore.
    bank_frame_keys = list_frame

    return state, bank_frame_keys

def compute_loss(
    pred_masks,
    gt_masks,
    inference_state
):
    dice_loss_fn = DiceLoss(sigmoid=True)
    focal_loss_fn = FocalLoss()

    video_H = inference_state["video_height"]
    video_W = inference_state["video_width"]
    if pred_masks.shape[-2:] == (video_H, video_W):
        video_res_masks = pred_masks.squeeze()
    else:
        video_res_masks = torch.nn.functional.interpolate(
            pred_masks,
            size=(video_H, video_W),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        
    loss = dice_loss_fn(video_res_masks, gt_masks)

    return loss

