# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import time
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sam2_train.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2_train.utils.misc import concat_points, fill_holes_in_mask_scores, load_video_frames, load_video_frames_from_data
from sam2_train.rl_modules.rl_utils import prepare_rl_state, compute_loss, deterministic_dropout
from sam2_train.rl_modules.policy_optimization.grpo_agent import GRPOAgent


def _detach_in_place(obj, seen):
    """Strip the autograd graph off every tensor reachable from `obj`, in place.

    Called when a tracking state moves on to the next chunk. The memory bank, the
    conditioning frames and the pending candidate all survive that boundary, but the
    graph that produced them was freed by the chunk's own `backward()`: reusing those
    tensors as-is would either raise "backward through the graph a second time" on the
    next chunk, or -- worse, while it still works -- pin every previous chunk's
    activations for the length of the volume.

    In place rather than rebuilding the dicts, because the same output dict is aliased
    from several places (`await_outputs[k]` and `non_cond_frame_outputs[k]` are the same
    object once a frame is admitted to the bank) and rebinding would silently fork them.
    `detach()` shares storage, so this is bookkeeping, not a copy. `seen` guards against
    walking a shared sub-dict twice.
    """
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, list):
        items = enumerate(obj)
    else:
        return
    for key, value in items:
        if torch.is_tensor(value):
            if value.grad_fn is not None or value.requires_grad:
                obj[key] = value.detach()
        else:
            _detach_in_place(value, seen)


class SAM2VideoPredictor(SAM2Base):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
        self,
        fill_hole_area=0,
        # whpether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # whether to also clear non-conditioning memory of the surrounding frames (only effective when `clear_non_cond_mem_around_input` is True).
        clear_non_cond_mem_for_multi_obj=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj

    @torch.inference_mode()
    def init_state(
        self,
        video_path,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize a inference state."""
        images, video_height, video_width = load_video_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
        )
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = torch.device("cuda")
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = torch.device("cuda")
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "await_outputs": {}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    @torch.inference_mode()
    def val_init_state(
        self,
        args,
        imgs_tensor,
        masks_tensor,
        support_imgs_tensor,
        video_height=None,
        video_width=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
        global_pool=None,
        chunk_start=0,
        total_num_frames=None,
    ):
        """Initialize a inference state."""
        if video_height is None or video_width is None:
            video_height = self.image_size
            video_width = self.image_size
        images = load_video_frames_from_data(
            imgs_tensor=imgs_tensor,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
        )
        support_images = load_video_frames_from_data(
            imgs_tensor=support_imgs_tensor,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
        )
        inference_state = {}
        inference_state["images"] = images
        inference_state["support_images"] = support_images
        inference_state["num_frames"] = len(images)
        inference_state["support_num_frames"] = args.num_support
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["gt_masks"] = masks_tensor
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["device"] = images.device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "await_outputs": {},
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        inference_state["support_set_stage"] = True
        inference_state["rl_config"] = {
            "lazy_penalty": args.lazy_penalty,
            "invalid_penalty": args.invalid_penalty,
            "memory_bank_size": args.memory_bank_size,
            "recall_every": getattr(args, "recall_every", 1),
            "agent_act_every": getattr(args, "agent_act_every", 1),
        }
        # Validation tracks a whole volume in a single pass, so `chunk_start` is 0 and
        # `train_advance_chunk` never runs; the fields exist so the two paths index
        # `images` / `gt_masks` through the same expression.
        inference_state["global_pool"] = global_pool
        inference_state["chunk_start"] = chunk_start
        inference_state["start_frame_idx"] = chunk_start
        inference_state["total_num_frames"] = (
            len(images) if total_num_frames is None else int(total_num_frames)
        )
        # Only recall can produce obj_ptr distances far outside SAM2's trained range, so
        # the saturation in _prepare_memory_conditioned_features is tied to the pool being
        # on -- a pool-free run stays bit-identical to before this feature existed.
        self.clamp_obj_ptr_tpos = global_pool is not None and global_pool.enabled

        return inference_state

    # @torch.inference_mode()
    def train_init_state(
        self,
        args,
        imgs_tensor,
        masks_tensor,
        support_imgs_tensor,
        video_height=None,
        video_width=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
        global_pool=None,
        chunk_start=0,
        total_num_frames=None,
    ):
        """Initialize a tracking state for one (volume, obj_id).

        Created once per volume, not once per chunk: `train_advance_chunk` swaps the
        image/GT window in for each subsequent chunk while everything that makes up the
        RL environment -- the memory bank, the conditioning frames, the pending candidate
        -- stays put. `chunk_start` is where the state's `images`/`gt_masks` begin in the
        volume; every frame index the tracker and the agent see is volume-global.
        """
        if video_height is None or video_width is None:
            video_height = self.image_size
            video_width = self.image_size

        images = load_video_frames_from_data(
            imgs_tensor=imgs_tensor,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
        )
        
        support_images = load_video_frames_from_data(
            imgs_tensor=support_imgs_tensor,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
        )

        inference_state = {}
        inference_state["images"] = images
        inference_state["support_images"] = support_images
        inference_state["num_frames"] = len(images)
        inference_state["support_num_frames"] = args.num_support
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["gt_masks"] = masks_tensor
        if offload_state_to_cpu:
            inference_state["device"] = torch.device("cpu")
        else:
            inference_state["device"] = images.device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "await_outputs": {},
            "prev_memory_attn_scores": {},
            "image_features": {},
            "prev_frame_idx": [],
            "dropped_frames_allres_sim_rank": [],
            "dropped_frames_lowres_sim_rank": [],
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0

        inference_state["support_set_stage"] = True

        inference_state["rl_config"] = {
            "lazy_penalty": args.lazy_penalty,
            "invalid_penalty": args.invalid_penalty,
            "memory_bank_size": args.memory_bank_size,
            "recall_every": getattr(args, "recall_every", 1),
            "agent_act_every": getattr(args, "agent_act_every", 1),
        }
        # Owned by the caller and shared by every chunk of one (volume, obj_id). Keyed by
        # volume-global frame index, same as everything else now that the state itself
        # spans the volume.
        inference_state["global_pool"] = global_pool
        # Where this chunk's `images`/`gt_masks` start in the volume: frame indices are
        # volume-global, the tensors are not, so every lookup subtracts this.
        inference_state["chunk_start"] = chunk_start
        # First frame of the whole trajectory (not of this chunk). The frame before it has
        # no encoded memory, so it is the one frame with no agent decision to make.
        inference_state["start_frame_idx"] = chunk_start
        # Length of the volume, not of this chunk -- SAM2 reads it to bound the obj_ptr
        # window, which must not shrink just because the chunk did.
        inference_state["total_num_frames"] = (
            len(images) if total_num_frames is None else int(total_num_frames)
        )
        # Only recall can produce obj_ptr distances far outside SAM2's trained range, so
        # the saturation in _prepare_memory_conditioned_features is tied to the pool being
        # on -- a pool-free run stays bit-identical to before this feature existed.
        self.clamp_obj_ptr_tpos = global_pool is not None and global_pool.enabled

        return inference_state

    def train_advance_chunk(
        self,
        inference_state,
        imgs_tensor,
        masks_tensor,
        chunk_start,
        offload_video_to_cpu=None,
        async_loading_frames=False,
    ):
        """Slide the state's image/GT window on to the next chunk of the same volume.

        This is what replaces re-running `train_init_state` per chunk. Everything the RL
        environment is made of -- `output_dict` (memory bank, conditioning frames, pending
        candidate), the global pool, the object-index maps -- is left exactly as the
        previous chunk left it, so the agent's trajectory runs the length of the volume
        instead of restarting every `video_length` frames. Only two things change: which
        frames the state can decode, and the fact that the carried-over tensors must lose
        their (already-freed) autograd graph; see `_detach_in_place`.
        """
        if offload_video_to_cpu is None:
            offload_video_to_cpu = inference_state["offload_video_to_cpu"]

        images = load_video_frames_from_data(
            imgs_tensor=imgs_tensor,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
        )
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        inference_state["gt_masks"] = masks_tensor
        inference_state["chunk_start"] = chunk_start
        # The support frames were prompted once, on the first chunk; re-entering the
        # support stage here would decode query frames out of `support_images`.
        inference_state["support_set_stage"] = False

        seen = set()
        _detach_in_place(inference_state["output_dict"], seen)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            _detach_in_place(obj_output_dict, seen)
        for obj_temp_dict in inference_state["temp_output_dict_per_obj"].values():
            _detach_in_place(obj_temp_dict, seen)
        _detach_in_place(inference_state["constants"], seen)

        return inference_state

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # This is a new object id not sent to the server before. We only allow adding
        # new objects *before* the tracking starts.
        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "await_outputs": {}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far in this session."""
        return len(inference_state["obj_idx_to_id"])

    @torch.inference_mode()
    def add_new_points(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points,
        labels,
        clear_old_points=True,
        normalize_coords=True,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension
        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            prev_sam_mask_logits = prev_out["pred_masks"].cuda(non_blocking=True)
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def add_new_bbox(
        self,
        inference_state,
        frame_idx,
        obj_id,
        bbox,
        clear_old_points=True,
        normalize_coords=True,
    ):
        if not isinstance(bbox, torch.Tensor):
            bbox = torch.tensor(bbox, dtype=torch.float32)
        bbox_coords = bbox.reshape(-1, 2, 2)
        bbox_labels = torch.tensor([2, 3], dtype=torch.int)

        out_frame_idx, out_obj_ids, out_mask_logits = self.add_new_points(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=bbox_coords,
            labels=bbox_labels,
            clear_old_points=clear_old_points,
            normalize_coords=normalize_coords,
        )
        return out_frame_idx, out_obj_ids, out_mask_logits

    # @torch.inference_mode()
    def train_add_new_bbox(
        self,
        inference_state,
        frame_idx,
        obj_id,
        bbox,
        clear_old_points=True,
        normalize_coords=True,
    ):
        if not isinstance(bbox, torch.Tensor):
            bbox = torch.tensor(bbox, dtype=torch.float32)
        bbox_coords = bbox.reshape(-1, 2, 2)
        bbox_labels = torch.tensor([2, 3], dtype=torch.int)

        out_frame_idx, out_obj_ids, out_mask_logits = self.train_add_new_points(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=bbox_coords,
            labels=bbox_labels,
            clear_old_points=clear_old_points,
            normalize_coords=normalize_coords,
        )
        return out_frame_idx, out_obj_ids, out_mask_logits

    # @torch.inference_mode()
    def train_add_new_points(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points,
        labels,
        clear_old_points=True,
        normalize_coords=True,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension
        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            prev_sam_mask_logits = prev_out["pred_masks"].cuda(non_blocking=True)
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        ) # dict_keys(['maskmem_features', 'maskmem_pos_enc', 'pred_masks_video_res', 'obj_ptr'])
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
    ):
        """Add new mask to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]  # add batch and channel dimension
        mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])

        # resize the mask if it doesn't match the model's image size
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    # @torch.inference_mode()
    def train_add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
    ):
        """Add new mask to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2 or mask.dim() == 3 or mask.dim() == 4
        if mask.dim() == 2:
            mask_H, mask_W = mask.shape
            # add batch
            mask_inputs_orig = mask[None, None]
        elif mask.dim() == 3:
            _, mask_H, mask_W = mask.shape
            # add batch
            mask_inputs_orig = mask[None]
        else:
            _, _, mask_H, mask_W = mask.shape
            mask_inputs_orig = mask

        mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])

        # resize the mask if it doesn't match the model's image size
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            agent_act=False,
            generate_rl_samples=False,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
    ):
        """
        Consolidate the per-object temporary outputs in `temp_output_dict_per_obj` on
        a frame into a single output for all objects, including
        1) fill any missing objects either from `output_dict_per_obj` (if they exist in
           `output_dict_per_obj` for this frame) or leave them as placeholder values
           (if they don't exist in `output_dict_per_obj` for this frame);
        2) if specified, rerun memory encoder after apply non-overlapping constraints
           on the object scores.
        """
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            assert not run_mem_encoder, "memory encoder cannot run at video resolution"
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.image_size // 4
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            consolidated_mask_key: torch.full(
                size=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["device"],
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.hidden_dim),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["device"],
            )
        }
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
            # we fall back and look up its previous output in "output_dict_per_obj".
            # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
            # "output_dict_per_obj" to find a previous output for this object.
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object doesn't appear in "output_dict_per_obj" either, we skip it
            # and leave its mask scores to the default scores (i.e. the NO_OBJ_SCORE
            # placeholder above) and set its object pointer to be a dummy pointer.
            if out is None:
                # Fill in dummy object pointers for those objects without any inputs or
                # tracking outcomes on this frame (only do it under `run_mem_encoder=True`,
                # i.e. when we need to build the memory for tracking).
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(
                            inference_state, frame_idx
                        )
                    # fill object pointer with a dummy pointer (based on an empty mask)
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = empty_mask_ptr
                continue
            # Add the temporary object output mask to consolidated output mask
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                # Resize first if temporary object mask has a different resolution
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]
            consolidated_out["vision_feats"] = out["vision_feats"]

        # Optionally, apply non-overlapping constraints on the consolidated scores
        # and rerun the memory encoder
        if run_mem_encoder:
            device = inference_state["device"]
            high_res_masks = torch.nn.functional.interpolate(
                consolidated_out["pred_masks"].to(device, non_blocking=True),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.non_overlap_masks_for_mem_enc:
                high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                inference_state=inference_state,
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                is_mask_from_pts=True,  # these frames are what the user interacted with
            )
            consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc


        return consolidated_out

    def _get_empty_mask_ptr(self, inference_state, frame_idx):
        """Get a dummy object pointer based on an empty mask on the current frame."""
        # A dummy (empty) mask with a single object
        batch_size = 1
        mask_inputs = torch.zeros(
            (batch_size, 1, self.image_size, self.image_size),
            dtype=torch.float32,
            device=inference_state["device"],
        )

        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # Feed the empty mask and image feature above to get a dummy object pointer
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=mask_inputs,
            output_dict={},
            num_frames=inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Tracking has started and we don't allow adding new objects until session is reset.
        inference_state["tracking_has_started"] = True
        batch_size = self._get_obj_num(inference_state)

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        # "consolidated_frame_inds" contains indices of those frames where consolidated
        # temporary outputs have been added (either in this call or any previous calls
        # to `propagate_in_video_preflight`).
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in [False, True]:
            # Separately consolidate conditioning and non-conditioning temp outptus
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # Find all the frames that contain temporary outputs for any objects
            # (these should be the frames that have just received clicks for mask inputs
            # via `add_new_points` or `add_new_mask`)
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # consolidate the temprary output across all objects on this frame
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True
                )
                # merge them into "output_dict" and also create per-object slices
                output_dict[storage_key][frame_idx] = consolidated_out

                self._add_output_per_object(
                    inference_state, frame_idx, consolidated_out, storage_key
                )
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)

            # clear temporary outputs in `temp_output_dict_per_obj`
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # edge case: if an output is added to "cond_frame_outputs", we remove any prior
        # output on the same frame in "non_cond_frame_outputs"
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # Make sure that the frame indices in "consolidated_frame_inds" are exactly those frames
        # with either points or mask inputs (which should be true under a correct workflow).
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds

    # @torch.inference_mode()
    def train_propagate_in_video_preflight(self, inference_state):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Tracking has started and we don't allow adding new objects until session is reset.
        inference_state["tracking_has_started"] = True
        batch_size = self._get_obj_num(inference_state)

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        # "consolidated_frame_inds" contains indices of those frames where consolidated
        # temporary outputs have been added (either in this call or any previous calls
        # to `propagate_in_video_preflight`).
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in [False, True]:
            # Separately consolidate conditioning and non-conditioning temp outptus
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # Find all the frames that contain temporary outputs for any objects
            # (these should be the frames that have just received clicks for mask inputs
            # via `add_new_points` or `add_new_mask`)
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # consolidate the temprary output across all objects on this frame
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True
                )
                # merge them into "output_dict" and also create per-object slices
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(
                    inference_state, frame_idx, consolidated_out, storage_key
                )
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)

            # clear temporary outputs in `temp_output_dict_per_obj`
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # edge case: if an output is added to "cond_frame_outputs", we remove any prior
        # output on the same frame in "non_cond_frame_outputs"
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # Make sure that the frame indices in "consolidated_frame_inds" are exactly those frames
        # with either points or mask inputs (which should be true under a correct workflow).
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        self.propagate_in_video_preflight(inference_state)

        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        inference_state["support_set_stage"] = False
        processing_order = range(num_frames)
        print("processing_order: ", processing_order)
        for frame_idx in processing_order:
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.

            storage_key = "non_cond_frame_outputs"
            current_out, pred_masks = self._run_single_frame_inference(
                inference_state=inference_state,
                output_dict=output_dict,
                frame_idx=frame_idx,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=reverse,
                run_mem_encoder=True,
            )
            output_dict[storage_key][frame_idx] = current_out
            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )
            yield frame_idx, obj_ids, current_out["ious"], current_out["object_score_logits"], video_res_masks

    # @torch.inference_mode()
    def train_propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        train_agent=False,
        agent_act=True,
        generate_rl_samples=False,
        start_trajectory=False,
        end_trajectory=False,
        random_drop=False,
        log_reward=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        # Only the first chunk of a volume has temporary support outputs to consolidate.
        # Re-running it on a later chunk would be worse than wasteful: it pops every
        # conditioning frame's index out of `non_cond_frame_outputs`, which used to be
        # empty at that point and now holds the agent's carried-over memory bank.
        if not inference_state["tracking_has_started"]:
            self.train_propagate_in_video_preflight(inference_state)

        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        chunk_start = inference_state["chunk_start"]
        batch_size = self._get_obj_num(inference_state)
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        inference_state["support_set_stage"] = False
        # Volume-global indices: this chunk continues where the previous one stopped,
        # rather than restarting the numbering at 0.
        processing_order = range(chunk_start, chunk_start + num_frames)

        if train_agent and start_trajectory:
            self.agent.init_new_trajectory()

        for frame_idx in processing_order:
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.

            if agent_act or generate_rl_samples or random_drop:
                storage_key = "await_outputs"
            else:
                storage_key = "non_cond_frame_outputs"

            current_out, pred_masks = self._run_single_frame_inference(
                inference_state=inference_state,
                output_dict=output_dict,
                frame_idx=frame_idx,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=reverse,
                run_mem_encoder=True,
                agent_act=agent_act,
                train_agent=train_agent,
                generate_rl_samples=generate_rl_samples,
                random_drop=random_drop,
                log_reward=log_reward,
            )
            output_dict[storage_key][frame_idx] = current_out
            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Archive this frame if it lands on the pool's stride.
            global_pool = inference_state.get("global_pool")
            if global_pool is not None:
                global_pool.maybe_push(frame_idx, current_out)

            # Only the immediately preceding frame's pending output is ever read again
            # (as the next frame's candidate); anything older was either admitted to the
            # bank -- which holds the very same dict, so this does not free it -- or
            # rejected for good. Dropping it matters now that the state lives for a whole
            # volume rather than a 16-frame chunk: otherwise every frame's memory
            # features, vision features and masks stay pinned to the end of the volume.
            if storage_key == "await_outputs":
                stale_idx = frame_idx - 1
                output_dict["await_outputs"].pop(stale_idx, None)
                for obj_output_dict in inference_state["output_dict_per_obj"].values():
                    obj_output_dict["await_outputs"].pop(stale_idx, None)

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )
            gating_score_dict = None
            if "gating_score_dict" in current_out.keys():
                gating_score_dict = current_out["gating_score_dict"]
            yield frame_idx, obj_ids, current_out["ious"], current_out["object_score_logits"], video_res_masks, gating_score_dict

        # A chunk boundary is no longer a terminal: the next chunk continues on this same
        # state, with this chunk's memory bank still in it, so the transition left pending
        # here is closed by the first decision of the next chunk and credit flows across
        # the boundary like any other step. Only the end of the volume is a real terminal.
        if train_agent and end_trajectory:
            self.agent.set_await_done()
            # One trajectory per (volume, obj_id); GAE runs once, over the whole volume.
            self.agent.final_trajectory()


    def _add_output_per_object(
        self, inference_state, frame_idx, current_out, storage_key
    ):
        """
        Split a multi-object output into per-object output slices and add them into
        `output_dict_per_obj`. The resulting slices share the same tensor storage.
        """
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        output_dict_per_obj = inference_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out


    # @torch.inference_mode()
    def reset_state(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""

        if inference_state["support_set_stage"]:
            # Support prompts have their own numbering, independent of the volume's.
            image = inference_state["support_images"][frame_idx].to(device=inference_state["device"]).float().unsqueeze(0)
        else:
            # `frame_idx` is volume-global; `images` only holds the current chunk.
            local_idx = frame_idx - inference_state.get("chunk_start", 0)
            image = inference_state["images"][local_idx].to(device=inference_state["device"]).float().unsqueeze(0)


        backbone_out = self.forward_image(image) # dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
        # Cache the most recent frame's feature (for repeated interactions with
        # a frame; we can use an LRU cache for more frames in the future).

        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
        agent_act=False,
        train_agent=False,
        generate_rl_samples=False,
        random_drop=False,
        log_reward=False,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        storage_device = inference_state["device"]

        # NOTE: Agent act on memory bank before running a track step.
        # Against the trajectory's first frame, not the chunk's: every later chunk
        # continues on the same state, so its opening frame has a real predecessor in
        # `await_outputs` and a real decision to make.
        if frame_idx > inference_state.get("start_frame_idx", 0):
            if random_drop:
                if len(output_dict["non_cond_frame_outputs"]) + 1 >= self.num_maskmem:
                    drop_frame = np.random.choice(list(output_dict["non_cond_frame_outputs"].keys()), size=1)[0]
                    output_dict["non_cond_frame_outputs"].pop(drop_frame)
                    print("Drop random frame:", drop_frame)
                output_dict["non_cond_frame_outputs"][frame_idx-1] = output_dict["await_outputs"][frame_idx-1]
            else:
                track_step_kwargs = {
                    "is_init_cond_frame": is_init_cond_frame,
                    "feat_sizes": feat_sizes,
                    "point_inputs": point_inputs,
                    "mask_inputs": mask_inputs,
                    "num_frames": inference_state["total_num_frames"],
                    "track_in_reverse": reverse,
                    "run_mem_encoder": run_mem_encoder,
                    "prev_sam_mask_logits": prev_sam_mask_logits,
                }

                self.agent_act(
                    inference_state,
                    storage_device,
                    frame_idx,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    output_dict,
                    train_agent,
                    agent_act,
                    generate_rl_samples,
                    log_reward=log_reward,
                    **track_step_kwargs
                )

        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["total_num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            agent_act=agent_act,
            random_drop=random_drop,
            memory_bank_size=inference_state["rl_config"]["memory_bank_size"]
        )

        # optionally offload the output to CPU memory to save GPU space
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.float32)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        # potentially fill holes in the predicted masks
        # if self.fill_hole_area > 0:
        #     pred_masks_gpu = fill_holes_in_mask_scores(
        #         pred_masks_gpu, self.fill_hole_area
        #     )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        ious = current_out["ious"]
        object_score_logits = current_out["object_score_logits"]
        vision_feats = current_vision_feats[-1]

        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "vision_feats": vision_feats,
            "pred_masks": pred_masks,
            "ious": ious,
            "object_score_logits": object_score_logits,
            "obj_ptr": obj_ptr,
        }

        if "gating_score_dict" in current_out.keys():
            compact_current_out["gating_score_dict"] = current_out["gating_score_dict"]

        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
        self, inference_state, frame_idx, batch_size, high_res_masks, is_mask_from_pts
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=is_mask_from_pts,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["device"]
        maskmem_features = maskmem_features.to(torch.float32)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Remove the non-conditioning memory around the input frame. When users provide
        correction clicks, the surrounding frames' non-conditioning memories can still
        contain outdated object appearance information and could confuse the model.

        This method clears those non-conditioning memories surrounding the interacted
        frame to avoid giving the model both old and new information about the object.
        """
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        output_dict = inference_state["output_dict"]
        non_cond_frame_outputs = output_dict["non_cond_frame_outputs"]
        for t in range(frame_idx_begin, frame_idx_end + 1):
            non_cond_frame_outputs.pop(t, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)

    def agent_act(
        self,
        inference_state,
        storage_device,
        frame_idx,
        current_vision_feats,
        current_vision_pos_embeds,
        output_dict,
        train_agent,
        agent_act,
        generate_rl_samples,
        log_reward=False,
        **track_step_kwargs
    ):
        if generate_rl_samples or train_agent or agent_act:
            if isinstance(self.agent, GRPOAgent):
                self.generate_rl_steps(
                    inference_state=inference_state,
                    storage_device=storage_device,
                    frame_idx=frame_idx,
                    current_vision_feats=current_vision_feats,
                    current_vision_pos_embeds=current_vision_pos_embeds,
                    output_dict=output_dict,
                    train_agent=train_agent,
                    agent_act=agent_act,
                    generate_rl_samples=generate_rl_samples,
                    **track_step_kwargs
                )
            else:
                # Opens this frame's transition and closes the previous one; the reward
                # is fully measured inside, so there is no second stage.
                self.agent_update_first_stage(
                    inference_state=inference_state,
                    storage_device=storage_device,
                    frame_idx=frame_idx,
                    current_vision_feats=current_vision_feats,
                    current_vision_pos_embeds=current_vision_pos_embeds,
                    output_dict=output_dict,
                    train_agent=train_agent,
                    log_reward=log_reward,
                    **track_step_kwargs
                )
    @torch.no_grad()
    def generate_rl_steps(
        self,
        inference_state,
        storage_device,
        frame_idx,
        current_vision_feats,
        current_vision_pos_embeds,
        output_dict,
        train_agent=False,
        agent_act=True,
        generate_rl_samples=False,
        **kwargs
    ):
        # compute loss before
        loss_before = None
        if train_agent:
            with torch.no_grad():
                output_before = self.track_step(
                    frame_idx=frame_idx,
                    current_vision_feats=current_vision_feats,
                    current_vision_pos_embeds=current_vision_pos_embeds,
                    output_dict=output_dict,
                    agent_act=True,
                    **kwargs
                )

                pred_masks = output_before["pred_masks"]
                pred_masks = pred_masks.to(storage_device, non_blocking=True).to(torch.float32)
                local_idx = frame_idx - inference_state["chunk_start"]
                gt_masks = inference_state["gt_masks"][local_idx].to(device=storage_device, non_blocking=True)
                gt_masks = gt_masks.to(torch.float32)

                loss_before = compute_loss(pred_masks, gt_masks, inference_state)

        if agent_act or generate_rl_samples:
            state, bank_frame_keys = prepare_rl_state(
                current_vision_feats,
                current_vision_pos_embeds,
                output_dict,
                frame_idx,
                # num_maskmem=self.num_maskmem - 1,
                num_maskmem=inference_state['rl_config']['memory_bank_size'],
                num_max_prompt=inference_state["support_num_frames"],
                offload_to_cpu=False,
                training=train_agent
            )

            memory_bank_size = inference_state['rl_config']['memory_bank_size']
            bank_size = len(output_dict["non_cond_frame_outputs"])
            bank_full = (bank_size >= memory_bank_size)
            agent_act_every = max(inference_state['rl_config'].get("agent_act_every", 1), 1)
            is_decision_frame = (frame_idx % agent_act_every == 0)
            if bank_full and is_decision_frame:
                # Action layout: 0 = no-op (reject), 1..M = swap(incoming frame, bank
                # slot j), j = action - 1. GRPO runs without the global pool, so the
                # incoming frame is the only candidate ever on offer.
                valid_actions = [0] + [1 + j for j in range(memory_bank_size)]
                with torch.no_grad():
                    action_out = self.agent.select_action(
                        state,
                        valid_actions=torch.tensor(valid_actions),
                        num_samples=6,
                        training=train_agent,
                    ) # ask agent
            else:
                # Either filling up the bank (nothing to evict) or between agent steps
                # (bank frozen): the incoming frame's fate is decided without consulting
                # the policy -- see sam2_video_predictor.agent_update_first_stage for the
                # same design.
                action_out = {"main_action": None, "action": [], "log_probs": []}

            # state.offload_to_cpu()

        if generate_rl_samples:
            self.agent.init_new_group()

            actions = action_out["action"]
            log_probs = action_out["log_probs"]
            for i, (action, log_prob) in enumerate(zip(actions, log_probs)):
                reward = 0
                temp_output_dict = {
                    "cond_frame_outputs": output_dict["cond_frame_outputs"].copy(),
                    "non_cond_frame_outputs": output_dict["non_cond_frame_outputs"].copy()
                }

                drop_frame = None
                storage_key = "non_cond_frame_outputs"
                if action == 0:
                    # No-op: reject the incoming frame.
                    reward = inference_state['rl_config']['lazy_penalty']
                else:
                    # Evict bank slot j = action - 1 and admit the incoming frame.
                    drop_frame = bank_frame_keys[action - 1]
                    temp_output_dict[storage_key].pop(drop_frame)
                    temp_output_dict[storage_key][frame_idx-1] = output_dict["await_outputs"][frame_idx-1]

                # print(action, reward)
                if action != 0:
                    with torch.no_grad():
                        output_before = self.track_step(
                            frame_idx=frame_idx,
                            current_vision_feats=current_vision_feats,
                            current_vision_pos_embeds=current_vision_pos_embeds,
                            output_dict=temp_output_dict,
                            agent_act=True,
                            **kwargs
                        )

                        pred_masks = output_before["pred_masks"]
                        pred_masks = pred_masks.to(storage_device, non_blocking=True).to(torch.float32)

                        loss_after = compute_loss(pred_masks, gt_masks, inference_state)

                        loss_diff = loss_before.detach().cpu() - loss_after.detach().cpu()

                        if loss_diff > 0:
                            one_hot_rw = 1
                        elif loss_diff < 0:
                            one_hot_rw = -1
                        else:
                            one_hot_rw = 0

                        # reward += (loss_before.detach().cpu() - loss_after.detach().cpu())
                        
                        reward += (1 - loss_after.detach().cpu())
                else:
                    reward += (1 - loss_before.detach().cpu())

                replay_instance_info = {
                    "frame_idx": frame_idx,
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "log_probs": log_prob,
                }

                # print(replay_instance_info["action"], replay_instance_info["reward"])

                self.agent.add_new_instance_to_group(**replay_instance_info)

            self.agent.final_group()

        if agent_act:
            drop_frame = None
            reward = 0.0
            if not bank_full:
                # Force insert: no decision to make, matches agent_update_first_stage.
                action = None
                output_dict["non_cond_frame_outputs"][frame_idx-1] = output_dict["await_outputs"][frame_idx-1]
            elif not is_decision_frame:
                # Between agent steps: bank frozen, matches agent_update_first_stage.
                action = None
            else:
                action = action_out['main_action']
                if action == 0:
                    # No-op: reject the incoming frame.
                    drop_frame = frame_idx - 1
                    reward = -0.0
                else:
                    # Evict bank slot j = action - 1 and admit the incoming frame.
                    drop_frame = bank_frame_keys[action - 1]
                    output_dict["non_cond_frame_outputs"].pop(drop_frame)
                    output_dict["non_cond_frame_outputs"][frame_idx-1] = output_dict["await_outputs"][frame_idx-1]

            if not train_agent:
                print(f"[Q] frame {frame_idx-1} "
                    f"action {action} "
                    f"drop_frame {drop_frame} "
                    f"bank_size {bank_size} ")

    @staticmethod
    def _sort_memory_bank(output_dict):
        """Reorder the memory bank by frame index, oldest first.

        SAM2 reads temporal position straight off iteration order
        (`t_pos = t + 1` in _prepare_memory_conditioned_features), and the agent's
        recency prior is indexed by slot rank, so both silently assume dict order is
        chronological order. That held while the only insertion was the incoming frame,
        which is always the newest. A recalled pool entry is older than everything already
        in the bank, and appending it would label the oldest memory as the most recent one
        -- inverting the temporal encoding for that slot. Sorting in place (rather than
        rebinding) keeps the identity of the dict that callers already hold.
        """
        bank = output_dict["non_cond_frame_outputs"]
        items = sorted(bank.items())
        bank.clear()
        bank.update(items)

    def agent_update_first_stage(
        self,
        inference_state,
        storage_device,
        frame_idx,
        current_vision_feats,
        current_vision_pos_embeds,
        output_dict,
        train_agent,
        log_reward=False,
        **kwargs
    ):
        def measure_loss():
            """Dice loss on this frame under whatever the memory bank currently holds.

            Dropout-free so the before/after pair differs only by the agent's action;
            see rl_utils.deterministic_dropout. SAM2 stays in train mode, so the memory
            encoding, obj_ptr selection and mask-decoder branches are the ones being
            trained -- only the randomness is gone.
            """
            with torch.no_grad(), deterministic_dropout(self):
                output = self.track_step(
                    frame_idx=frame_idx,
                    current_vision_feats=current_vision_feats,
                    current_vision_pos_embeds=current_vision_pos_embeds,
                    output_dict=output_dict,
                    agent_act=True,
                    memory_bank_size=inference_state["rl_config"]["memory_bank_size"],
                    **kwargs
                )
            pred_masks = output["pred_masks"].to(storage_device, non_blocking=True).to(torch.float32)
            # `frame_idx` is volume-global; `gt_masks` only holds the current chunk.
            local_idx = frame_idx - inference_state["chunk_start"]
            gt_masks = inference_state["gt_masks"][local_idx].to(device=storage_device, non_blocking=True)
            return compute_loss(pred_masks, gt_masks.to(torch.float32), inference_state)

        rl_config = inference_state["rl_config"]
        memory_bank_size = rl_config["memory_bank_size"]
        global_pool = inference_state.get("global_pool")
        storage_key = "non_cond_frame_outputs"

        # log_reward computes the same before/after counterfactual as train_agent, but
        # only to log it (see below) -- for validation, where the policy runs greedily
        # and nothing should be pushed into the live replay buffer/trajectory.
        compute_reward = train_agent or log_reward
        loss_before = measure_loss() if compute_reward else None

        # Current state
        state, bank_frame_keys = prepare_rl_state(
            current_vision_feats,
            current_vision_pos_embeds,
            output_dict,
            frame_idx,
            num_maskmem=memory_bank_size,
            num_max_prompt=inference_state["support_num_frames"],
            offload_to_cpu=True,
            training=train_agent,
            global_pool=global_pool,
        )
        state.offload_to_cpu()

        bank = output_dict[storage_key]
        bank_size = len(bank)
        bank_full = (bank_size >= memory_bank_size)

        # Action layout, matching BasePolicyNetwork's query order [noop, candidates,
        # bank slots]:
        #   0                     no-op: reject the incoming frame, bank untouched
        #   1 .. M                swap(incoming frame, bank slot j), j = action - 1
        #   M+1 .. (1+P)*M        swap(pool entry p, bank slot j), where
        #                         c, j = divmod(action - 1, M) and p = c - 1
        # M = memory_bank_size, P = pool capacity (0 when the pool is disabled). Fixed
        # size regardless of how many pool entries are actually on offer this frame --
        # unavailable ones are masked out below.
        candidate_key = frame_idx - 1
        candidate_out = output_dict["await_outputs"][candidate_key]

        pool_capacity = (
            global_pool.capacity
            if global_pool is not None and global_pool.enabled
            else 0
        )
        n_actions = 1 + (1 + pool_capacity) * memory_bank_size
        agent_act_every = max(rl_config.get("agent_act_every", 1), 1)
        is_decision_frame = (frame_idx % agent_act_every == 0)

        if not bank_full:
            # Filling up the bank: there is nothing to evict and no slot to trade away,
            # so the incoming frame is inserted directly without consulting the policy.
            # Still records a well-formed, weight-0 transition so the critic trains on
            # the state. Not gated by agent_act_every: an empty bank needs filling
            # regardless of the decision cadence.
            bank[candidate_key] = candidate_out
            action = 0
            drop_frame = None
            mutated = True
            action_mask = torch.zeros(n_actions, dtype=torch.bool)
            action_mask[0] = True
            action_out = {
                "action": 0,
                "log_probs": 0.0,
                "action_mask": action_mask,
                "policy_weight": 0.0,
            }
        elif not is_decision_frame:
            # Between agent steps: the bank is frozen -- a forced no-op, not a decision
            # the policy is scored or credited for. Bounded to the fixed action-mask
            # size like the other bypasses above, so replay-buffer batching still sees
            # a uniform shape regardless of which bypass produced the transition.
            action = 0
            drop_frame = None
            mutated = False
            action_mask = torch.zeros(n_actions, dtype=torch.bool)
            action_mask[0] = True
            action_out = {
                "action": 0,
                "log_probs": 0.0,
                "action_mask": action_mask,
                "policy_weight": 0.0,
            }
        else:
            # Which pool entries are on offer as a candidate this frame: entries already
            # resident in the bank (or identical to the incoming frame) are excluded --
            # recalling them would be a no-op that duplicates a slot.
            pool_view = []
            valid_cand = [0]
            if global_pool is not None and global_pool.enabled:
                pool_view = global_pool.local_view()
                if frame_idx % max(rl_config.get("recall_every", 1), 1) == 0:
                    valid_cand += [
                        1 + p
                        for p, (pooled_key, _) in enumerate(pool_view)
                        if pooled_key not in bank and pooled_key != candidate_key
                    ]

            valid_actions = [0] + [
                1 + c * memory_bank_size + j
                for c in valid_cand
                for j in range(memory_bank_size)
            ]
            with torch.no_grad():
                action_out = self.agent.select_action(
                    state,
                    valid_actions=torch.tensor(valid_actions),
                    training=train_agent,
                ) # ask agent

            action = action_out["action"]
            drop_frame = None
            mutated = action != 0
            if mutated:
                c, j = divmod(action - 1, memory_bank_size)
                if c > 0:
                    # Recalled from the pool: older than everything already in the bank,
                    # so insertion order stops being chronological order; see
                    # _sort_memory_bank.
                    candidate_key, candidate_out = pool_view[c - 1]
                drop_frame = bank_frame_keys[j]
                bank.pop(drop_frame)
                bank[candidate_key] = candidate_out

        if mutated:
            self._sort_memory_bank(output_dict)

        if not train_agent and is_decision_frame:
            print(
                f"[Q] frame {frame_idx}"
                f" action {action}"
                f" candidate_key {candidate_key}"
                f" drop_frame {drop_frame}"
                f" bank_size {bank_size}"
            )

        if compute_reward:
            # The other half of the counterfactual: same frame, same dropout-free path,
            # only the bank has changed. Measuring it here rather than reading the next
            # frame's await_outputs is the point of the whole restructure -- await_outputs
            # comes from the stochastic training pass, so differencing against it left
            # dropout noise in every reward, including no-op, which cannot move the bank
            # at all and must therefore score exactly 0.
            if not mutated:
                # A no-op leaves the bank feeding this frame bit-for-bit the one
                # loss_before was measured on. With dropout silenced the second pass is
                # provably identical -- take the shortcut and get an exact 0 instead of
                # paying for a forward that can only return the same number.
                loss_after = loss_before
            else:
                loss_after = measure_loss()
            reward = (loss_before - loss_after).item()

        if train_agent:
            replay_instance_info = {
                "frame_idx": frame_idx,
                "state": state,
                "loss_before": loss_before.detach().cpu(),
                "loss_after": loss_after.detach().cpu(),
                "reward": reward,
                "action": action,
                "action_mask": action_out["action_mask"],
                "log_probs": action_out["log_probs"],
                # 0 for a forced transition (bank still filling, or between agent
                # steps): it still trains the critic but must not enter the policy loss.
                "policy_weight": action_out["policy_weight"],
            }

            # Opening closes the transition pending from the previous frame, handing it
            # this state as its successor.
            self.agent.init_new_replay_instance(**replay_instance_info)
        elif log_reward:
            # Validation: same reward signal, but logged as a standalone stat instead
            # of a transition -- the replay buffer/trajectory bookkeeping is reserved
            # for real training and must not see validation frames.
            self.agent.record_val_reward(reward)

    def forward(
        self,
        imgs_tensor, masks_tensor, support_masks_tensor,
        train_state, obj_id,
        train_agent=False, agent_act=True, generate_rl_samples=False,
        start_trajectory=False, end_trajectory=False,
        random_drop=False,
        log_reward=False,
        add_support=True,
        device="cpu"
    ):
        # The support frames are prompted once per volume, on the chunk that created the
        # state. Later chunks reuse the conditioning frames those prompts produced, which
        # are still sitting in `output_dict["cond_frame_outputs"]`.
        if add_support:
            for frame_idx in range(support_masks_tensor.shape[0]):
                mask = support_masks_tensor[frame_idx]
                _, _, _ = self.train_add_new_mask(
                    inference_state=train_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=mask.to(device=device),
                )

        # Frame indices coming back from the tracker are volume-global; the image/label
        # tensors passed in only cover this chunk.
        chunk_start = train_state["chunk_start"]
        video_segments = {}  # video_segments contains the per-frame segmentation results
        propagate_kwargs = {
            "agent_act": agent_act,
            "train_agent": train_agent,
            "generate_rl_samples": generate_rl_samples,
            "start_trajectory": start_trajectory,
            "end_trajectory": end_trajectory,
            "random_drop": random_drop,
            "log_reward": log_reward,
        }
        for out_frame_idx, out_obj_ids, ious, object_score_logits, out_mask_logits, gating_score_dict in self.train_propagate_in_video(train_state, **propagate_kwargs):
            local_idx = out_frame_idx - chunk_start
            video_segments[out_frame_idx] = {
                out_obj_id: {"image_tensor": imgs_tensor[local_idx], "image_label" : masks_tensor[local_idx],
                "pred_mask": out_mask_logits[i], "iou": ious[i], "object_score_logits": object_score_logits[i],
                "gating_score_dict": gating_score_dict}
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        return video_segments