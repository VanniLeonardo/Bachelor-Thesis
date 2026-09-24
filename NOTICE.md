# Notice

This repository is a derivative work of **VGGT: Visual Geometry Grounded Transformer**
(Copyright (c) Meta Platforms, Inc. and affiliates), https://github.com/facebookresearch/vggt,
released under the Creative Commons Attribution-NonCommercial 4.0 International license
(see `LICENSE.txt`). The whole repository, including the modifications and additions listed
below and any released weights derived from VGGT-1B, is distributed under the same license.

Modifications and additions by Leonardo Vanni (2025–2026):

- `vggt/heads/camera_head.py`: covariance branch; upstream `pose_branch` renamed `mean_pose_branch` and frozen.
- `vggt/models/vggt.py`: exposes `cholesky_vector` in the predictions.
- `vggt/utils/uncertainty.py`, `vggt/utils/checkpoint.py`: new.
- `vggt/utils/geometry.py`: first-order propagation of pose uncertainty to 3D points.
- `training/loss.py`, `training/trainer.py`, `training/launch.py`, `training/config/*.yaml`, `training/train_utils/normalization.py`, `training/data/datasets/co3d.py`: pose-uncertainty training, curriculum, logging, on-disk filtering of CO3D sequences.
- `demo_viser.py`, `demo_gradio.py`, `visual_util.py`: uncertainty visualization.
- `plot_calibration.py`, `scripts/`, `tests/`, `docs/uncertainty.md`: new.

Files that keep the original Meta copyright header are upstream files, possibly modified as
listed above. New files carry their own header and the same license.

Third-party components used by the demos and training code (DINOv2, COLMAP/pycolmap, viser,
Gradio, trimesh, the sky-segmentation model) are subject to their own licenses.
