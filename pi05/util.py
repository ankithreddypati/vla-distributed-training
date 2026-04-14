"""
Training Utilities for VLA Fine-Tuning
=======================================

Helper functions used by main.py. These are separated out to keep the main
script focused on the Ray Data + Ray Train pipeline, while housing the
model-specific plumbing (freezing layers, checkpointing, collation, etc.)
in one place.
"""

import os
import tempfile
import warnings

import numpy as np
import torch
from ray.data.iterator import NumpyBatchCollateFn


# ============================================================================
# PI0.5 Attention Mask Patch
# ============================================================================

def apply_pi05_attention_mask_patch():
    """Monkey-patch make_att_2d_masks to tolerate pad/attention mask length mismatches."""
    import lerobot.policies.pi05.modeling_pi05 as mp

    if getattr(mp, "_PI05_MASK_PATCH_APPLIED", False):
        return
    _orig = mp.make_att_2d_masks

    def _patched(pad_masks, att_masks):
        pad_len, att_len = pad_masks.shape[-1], att_masks.shape[-1]
        if pad_len != att_len:
            warnings.warn(
                f"PI0.5 mask length mismatch: pad_masks={pad_len}, att_masks={att_len}; "
                f"truncating to {min(pad_len, att_len)}"
            )
            L = min(pad_len, att_len)
            return _orig(pad_masks[..., :L], att_masks[..., :L])
        return _orig(pad_masks, att_masks)

    mp.make_att_2d_masks = _patched
    mp._PI05_MASK_PATCH_APPLIED = True


# ============================================================================
# Dataset Helpers
# ============================================================================

def extract_stats(source):
    """Extract all normalization stats from a LeRobot datasource.

    Passes through all available keys (mean, std, min, max, q01, q99, etc.)
    so make_pre_post_processors has everything it needs regardless of which
    normalization mode is configured.
    """
    raw = source.meta.stats
    stats = {}
    for key in ("action", "observation.state"):
        if key in raw:
            # Pass all stat keys — don't filter to just mean/std
            stats[key] = {k: v for k, v in raw[key].items()}
    return stats


def renamed_image_keys(source, camera_rename):
    """Get camera column names after applying the rename map."""
    return [camera_rename.get(k, k) for k in source.meta.video_keys]


# ============================================================================
# Model Loading
# ============================================================================

def load_pi05_policy(pretrained_path="lerobot/pi05_base"):
    """Load PI0.5, apply the attention mask patch, and freeze the backbone."""
    from lerobot.policies.pi05 import PI05Policy

    apply_pi05_attention_mask_patch()

    policy = PI05Policy.from_pretrained(
        pretrained_path, device="cuda", dtype=torch.float16, train_expert_only=True,
    )

    # Force ALL parameters to cuda — embed_tokens can stay on CPU after load
    policy = policy.to(torch.device("cuda"))

    # Freeze everything, then unfreeze the small trainable heads
    for p in policy.parameters():
        p.requires_grad = False
    for name, module in policy.model.named_children():
        if name in {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}:
            for p in module.parameters():
                p.requires_grad = True

    return policy


# ============================================================================
# Checkpoint Save / Load
# ============================================================================

def load_checkpoint(checkpoint, policy, optimizer, scaler) -> tuple[int, int]:
    """Restore model/optimizer/scaler state from a Ray Train checkpoint."""
    import ray.cloudpickle as pickle

    with checkpoint.as_directory() as d:
        with open(os.path.join(d, "state.pkl"), "rb") as f:
            state = pickle.load(f)

    inner = policy.module if hasattr(policy, "module") else policy
    inner.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optim"])
    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state["epoch"] + 1, state.get("step", 0)


def make_checkpoint(policy, optimizer, scaler, epoch, step):
    """Serialize model + optimizer + scaler state into a Ray Train Checkpoint."""
    import ray.cloudpickle as pickle
    import ray.train

    inner = policy.module if hasattr(policy, "module") else policy
    ckpt_dir = tempfile.mkdtemp(prefix="pi05_ckpt_")
    with open(os.path.join(ckpt_dir, "state.pkl"), "wb") as f:
        pickle.dump(
            {
                "model":  inner.state_dict(),
                "optim":  optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch":  epoch,
                "step":   step,
            },
            f,
        )
    return ray.train.Checkpoint.from_directory(ckpt_dir)


# ============================================================================
# Sequence Truncation
# ============================================================================

def truncate_batch(batch: dict, max_len: int) -> dict:
    """Clip sequence and mask tensors to max_len tokens."""
    if not max_len:
        return batch
    for k in ("tokens", "input_ids", "masks", "attention_mask",
              "pad_masks", "att_masks", "img_masks", "image_masks"):
        if k in batch and hasattr(batch[k], "ndim") and batch[k].ndim >= 2:
            batch[k] = batch[k][..., :max_len]
    return batch


# ============================================================================
# Collation: numpy dicts -> torch tensors on GPU
# ============================================================================

class NumpyToTorchCollate(NumpyBatchCollateFn):
    """Convert a numpy batch dict into tensors on the target device.

    Handles ArrowVariableShapedTensorArray (object dtype) by stacking first.
    Keeps task column as a list of strings for language conditioning.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, batch: dict) -> dict:
        task = list(batch.pop("task"))
        result = {}
        for k, v in batch.items():
            arr = np.asarray(v)
            if arr.dtype == object:
                # ArrowVariableShapedTensorArray — stack into single array
                arr = np.stack(arr.tolist())
            if np.issubdtype(arr.dtype, np.integer):
                result[k] = torch.tensor(arr, dtype=torch.long, device=self.device)
            elif np.issubdtype(arr.dtype, np.bool_):
                result[k] = torch.tensor(arr, dtype=torch.bool, device=self.device)
            else:
                result[k] = torch.tensor(arr, dtype=torch.float32, device=self.device)
        result["task"] = task
        return result


# ============================================================================
# Training Step Helpers
# ============================================================================

def train_step(policy, batch, preprocessor, max_len, grad_accum, scaler):
    batch = preprocessor(batch)
    batch = truncate_batch(batch, max_len)
    batch.pop("task", None)
    batch.pop("task_index", None)

    # Preprocessor can create new tensors on CPU — move everything to GPU
    device = next(policy.parameters()).device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    with torch.autocast("cuda", torch.float16):
        out = policy(batch)
        loss = out.loss if hasattr(out, "loss") else out[0]

    scaler.scale(loss / grad_accum).backward()
    return float(loss.detach())


def optimizer_step(policy, optimizer, scaler, scheduler):
    """Unscale gradients, clip, step the optimizer, and update the LR schedule."""
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in policy.parameters() if p.requires_grad], max_norm=1.0,
    )
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()


def build_lr_scheduler(optimizer, config, num_workers, last_step):
    """Create a linear-warmup + cosine-decay LR scheduler."""
    import math

    batch_size  = int(config.get("batch_size", 1))
    grad_accum  = int(config.get("grad_accum", 1))
    num_epochs  = int(config.get("num_epochs", 1))
    total_rows  = int(config.get("total_rows", 10000))
    warmup_frac = float(config.get("warmup_frac", 0.1))

    rows_per_worker = total_rows // num_workers
    total_steps = max(rows_per_worker // (batch_size * grad_accum), 1) * num_epochs
    warmup_steps = int(total_steps * warmup_frac)

    def lr_lambda(s):
        if s < warmup_steps:
            return s / max(warmup_steps, 1)
        progress = (s - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=last_step - 1 if last_step > 0 else -1,
    )