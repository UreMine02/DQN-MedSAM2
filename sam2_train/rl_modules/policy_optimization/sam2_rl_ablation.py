"""
GRPO RL ablations. Runtime value is set from train CLI (`-rl_ablation_method`) before the agent is built.
"""
import os
import torch
import torch.nn as nn

_VALID = frozenset({"default", "sam2rl"})
_ablation_rl_method = os.environ.get("GRPO_RL_ABLATION", "default").strip().lower()
if _ablation_rl_method not in _VALID:
    _ablation_rl_method = "default"


def set_ablation_rl_method(method: str) -> None:
    global _ablation_rl_method
    m = (method or "default").strip().lower()
    if m not in _VALID:
        raise ValueError(f"rl_ablation_method must be one of {sorted(_VALID)}, got {method!r}")
    _ablation_rl_method = m


def get_ablation_rl_method() -> str:
    return _ablation_rl_method


def is_sam2rl_ablation() -> bool:
    return _ablation_rl_method == "sam2rl"


class Sam2RLFrameIdxMLPPolicy(nn.Module):
    """Policy logits from frame index only (embedding + MLP)."""

    def __init__(self, num_actions: int, hidden_dim: int = 256, max_frame_index: int = 4096):
        super().__init__()
        self.max_frame_index = max_frame_index
        self.embedding = nn.Embedding(max_frame_index + 1, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, frame_idx: torch.Tensor) -> torch.Tensor:
        # frame_idx: [B] int64
        idx = frame_idx.long().clamp(0, self.max_frame_index)
        h = self.embedding(idx)
        return self.mlp(h)


class Sam2RLGRPOActor(nn.Module):
    """Same forward signature as GRPOActor; uses frame index logits only."""

    def __init__(self, num_maskmem: int, hidden_dim: int = 256, max_frame_index: int = 4096):
        super().__init__()
        self.num_maskmem = num_maskmem
        num_actions = 2 + num_maskmem
        self.policy_net = Sam2RLFrameIdxMLPPolicy(
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            max_frame_index=max_frame_index,
        )

    def forward(
        self,
        image_feat,
        memory_feat,
        memory_ptr,
        bank_feat,
        bank_ptr,
        training=True,
        return_logits=False,
        sam2rl_frame_ix=None,
        **kwargs,
    ):
        del memory_feat, memory_ptr, bank_feat, bank_ptr, training
        del kwargs
        _ = return_logits  # logits always returned (same as BasePolicyNetwork path)
        B = image_feat.shape[0]
        device = image_feat.device
        if sam2rl_frame_ix is None:
            ix = torch.zeros(B, dtype=torch.long, device=device)
        else:
            ix = sam2rl_frame_ix.long().to(device=device)
            if ix.dim() == 0:
                ix = ix.expand(B)
            elif ix.shape[0] != B:
                ix = ix[:B]
        return self.policy_net(ix)
