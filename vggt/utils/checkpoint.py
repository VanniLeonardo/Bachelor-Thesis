# Copyright (c) 2026 Leonardo Vanni.
# Part of a derivative work of VGGT (Meta Platforms, Inc.); distributed under the
# same CC BY-NC 4.0 license found in the LICENSE.txt file in the root directory.

"""Loading VGGT together with the fine-tuned pose-uncertainty head.

The uncertainty extension renamed the upstream ``camera_head.pose_branch`` to
``camera_head.mean_pose_branch`` and added ``camera_head.covariance_branch``.
Everything else is bit-identical to the released
VGGT-1B weights, so a fine-tuned model is distributed as the upstream checkpoint plus a
small *head-only* file produced by :func:`export_uncertainty_head`.
"""

import os
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch

from vggt.utils.uncertainty import TRAINING_SHIELD_EPS

VGGT_1B_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
UNCERTAINTY_HEAD_PREFIXES = ("camera_head.covariance_branch.",)
LEGACY_HEAD_PREFIXES = ("camera_head.scale_embedding.",)  # removed in v1.0; ignored when loading old checkpoints
DEFAULT_HEAD_META = {"kappa": 10.0, "shield_eps": TRAINING_SHIELD_EPS, "error_convention": "body", "temperature": 1.0}


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    """Remove the ``module.`` prefix that DistributedDataParallel adds."""
    return OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state_dict.items())


def remap_upstream_state_dict(state_dict: Dict[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    """Rename upstream ``camera_head.pose_branch.*`` keys to ``camera_head.mean_pose_branch.*``."""
    out = OrderedDict()
    for k, v in strip_module_prefix(state_dict).items():
        if k.startswith("camera_head.pose_branch."):
            k = "camera_head.mean_pose_branch." + k[len("camera_head.pose_branch."):]
        out[k] = v
    return out


def _unwrap(checkpoint) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Return ``(state_dict, meta)`` from a trainer checkpoint, a head export, or a bare state dict."""
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], dict(checkpoint.get("meta", {}))
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"], {}
    return checkpoint, {}


def load_base_weights(model: torch.nn.Module, base: Optional[str] = None) -> None:
    """Load the upstream VGGT-1B weights (local path, ``VGGT_PRETRAINED_CKPT`` env var, or download)."""
    base = base or os.environ.get("VGGT_PRETRAINED_CKPT")
    if base and os.path.isfile(base):
        state = torch.load(base, map_location="cpu", weights_only=False)
    else:
        state = torch.hub.load_state_dict_from_url(base or VGGT_1B_URL, map_location="cpu")
    state, _ = _unwrap(state)
    missing, unexpected = model.load_state_dict(remap_upstream_state_dict(state), strict=False)
    bad_missing = [k for k in missing if not k.startswith(UNCERTAINTY_HEAD_PREFIXES)]
    if bad_missing:
        raise RuntimeError(f"upstream weights are missing keys the model needs: {bad_missing[:10]}")
    # unexpected keys are heads that were disabled in this model instance (depth/point/track)


def load_uncertainty_head(model: torch.nn.Module, path: str) -> Dict:
    """Load the covariance branch from a trainer checkpoint or a head export.

    Returns the metadata stored with the head (kappa, shield_eps, convention, temperature).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state, meta = _unwrap(ckpt)
    state = strip_module_prefix(state)
    head_state = OrderedDict((k, v) for k, v in state.items() if k.startswith(UNCERTAINTY_HEAD_PREFIXES))
    legacy = [k for k in state if k.startswith(LEGACY_HEAD_PREFIXES)]
    if legacy:
        print(f"Ignoring {len(legacy)} legacy scale-embedding tensors in {path} (removed in v1.0).")
    if not head_state:
        raise RuntimeError(f"{path} contains no uncertainty-head tensors ({UNCERTAINTY_HEAD_PREFIXES})")
    expected = {k for k in model.state_dict() if k.startswith(UNCERTAINTY_HEAD_PREFIXES)}
    if set(head_state) != expected:
        raise RuntimeError(
            f"uncertainty head in {path} does not match the model: "
            f"missing {sorted(expected - set(head_state))[:5]}, unexpected {sorted(set(head_state) - expected)[:5]}"
        )
    model.load_state_dict(head_state, strict=False)
    return {**DEFAULT_HEAD_META, **meta}


def load_vggt_with_uncertainty(
    uncertainty_ckpt: Optional[str],
    base: Optional[str] = None,
    device: str = "cpu",
    **model_kwargs,
) -> Tuple[torch.nn.Module, Dict]:
    """Build VGGT, load the upstream weights and (optionally) the fine-tuned uncertainty head.

    Args:
        uncertainty_ckpt: trainer checkpoint or head export; ``None`` keeps the randomly
            initialised head (covariances are then meaningless).
        base: path or URL of the upstream VGGT-1B weights (default: env var or download).
        model_kwargs: forwarded to :class:`vggt.models.vggt.VGGT` (e.g. ``enable_track=False``).

    Returns ``(model.eval() on device, meta)``.
    """
    from vggt.models.vggt import VGGT

    model = VGGT(**model_kwargs)
    load_base_weights(model, base)
    meta = dict(DEFAULT_HEAD_META)
    if uncertainty_ckpt is not None:
        meta = load_uncertainty_head(model, uncertainty_ckpt)
    return model.eval().to(device), meta


def export_uncertainty_head(checkpoint_path: str, out_path: str, meta: Optional[Dict] = None, half: bool = True) -> str:
    """Write the head-only file distributed with the release (the covariance branch).

    ``meta`` should record the training conventions (kappa, shield_eps, error_convention)
    and the fitted temperature so that consumers reproduce the trained covariance.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state, old_meta = _unwrap(ckpt)
    state = strip_module_prefix(state)
    head = OrderedDict((k, v.half() if (half and v.is_floating_point()) else v) for k, v in state.items() if k.startswith(UNCERTAINTY_HEAD_PREFIXES))
    if not head:
        raise RuntimeError(f"{checkpoint_path} contains no uncertainty-head tensors")
    payload = {
        "state_dict": head,
        "meta": {**DEFAULT_HEAD_META, **old_meta, **(meta or {}), "source_checkpoint": os.path.basename(checkpoint_path),
                 "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
                 "steps": ckpt.get("steps") if isinstance(ckpt, dict) else None},
    }
    torch.save(payload, out_path)
    return out_path
