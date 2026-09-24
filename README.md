<div align="center">
<h1>VGGT with Camera-Pose Uncertainty</h1>

<p>A probabilistic extension of <a href="https://github.com/facebookresearch/vggt">VGGT</a> (CVPR 2025) that predicts a full 6×6 covariance for every estimated camera pose, in a single feed-forward pass.</p>

<a href="https://github.com/facebookresearch/vggt"><img src="https://img.shields.io/badge/Upstream-facebookresearch%2Fvggt-blue" alt="Upstream"></a>
<a href="./LICENSE.txt"><img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey" alt="License"></a>
</div>

This repository accompanies the bachelor's thesis *Geometrically-Grounded Uncertainty Quantification for Foundational 3D Vision Models* (Leonardo Vanni, Bocconi University, 2026). VGGT's `CameraHead` is kept frozen and extended with a parallel *covariance branch* that outputs the Cholesky factor of a Gaussian over the pose error on the Lie algebra se(3). The frozen VGGT-1B backbone and mean-pose pathway are bit-identical to the upstream release, so point estimates are unchanged; only the uncertainty head is trained.

> **Note on the thesis version.** The code released here fixes several bugs found while preparing it for publication and is a slightly modified version of the code used for the thesis experiments. The revised thesis (v2) reports numbers recomputed with this code.

## What is added

- `vggt/heads/camera_head.py`: the covariance branch (an MLP on the aggregator camera token and the trunk's final hidden state) producing 21 Cholesky parameters per camera; the upstream pose pathway is renamed `mean_pose_branch` and frozen.
- `vggt/utils/uncertainty.py`: the SE(3) machinery shared by training and inference: numerically stable log map, body-centric error twist, Cholesky parameterisation, κ-weighted metric, Gaussian NLL, and helpers to turn the network output into covariances in physical units (scene units, radians), in the camera or world frame, plus 95% ellipsoids and optical-axis cones.
- `training/loss.py`: the scale-aware NLL with the three regularizers (scale, condition number, isotropy) and the warm-up / annealing curriculum; `training/trainer.py` trains only the covariance branch.
- `vggt/utils/checkpoint.py`: loads VGGT-1B plus the small head-only checkpoint distributed with this repository.
- `demo_viser.py`, `demo_gradio.py`: upstream demos extended with uncertainty ellipsoids and cones.
- `plot_calibration.py`: temperature fitting and calibration plots from the validation output.
- `scripts/`: CO3D repacking into one zip per sequence, the calibration/evaluation split of the test set, the COLMAP bundle-adjustment baseline in VGGT's gauge, the comparison with it, and rendering of the uncertainty inside the reconstructed scene.
- `reproduce/`: commands, splits, COLMAP models and frames behind every table and figure of the revised thesis.
- `tests/`: CPU unit tests for the geometry, the loss, the loaders and the baseline.

## Installation

```bash
git clone https://github.com/VanniLeonardo/Bachelor-Thesis.git
cd Bachelor-Thesis
pip install -r requirements.txt          # torch, torchvision, numpy, Pillow, huggingface_hub, einops, safetensors
pip install -e .                         # the `vggt` package
pip install -r requirements_demo.txt     # optional: viser / gradio demos
pip install -r requirements_train.txt    # optional: training and evaluation
```

Python ≥ 3.10; tested with torch 2.3.1 / CUDA 12.1 on a single 24 GB GPU. Set `VGGT_PRETRAINED_CKPT` to a local copy of the upstream `model.pt` to avoid re-downloading it.

## Weights

| File | Contents | Size |
|---|---|---|
| `facebook/VGGT-1B` `model.pt` | upstream VGGT-1B (downloaded automatically) | 5.0 GB |
| `vggt_uncertainty_head_v1.pt` | covariance branch + metadata (κ, shield, error convention, fitted temperature) | ≈0.4 GB (fp16) |

The head file will be published on the Hugging Face Hub with the v1.0.0 release; until then train it with the instructions below. Everything else is loaded from the upstream weights.

## Quick start

```python
import torch
from vggt.utils.checkpoint import load_vggt_with_uncertainty
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.uncertainty import pose_covariances_from_predictions, camera_centers, rotation_std_deg

device = "cuda" if torch.cuda.is_available() else "cpu"
model, meta = load_vggt_with_uncertainty("vggt_uncertainty_head_v1.pt", device=device)

images = load_and_preprocess_images(["img1.png", "img2.png", "img3.png"]).to(device)
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    pred = model(images)                                   # pose_enc (1,S,9), cholesky_vector (1,S,21), depth, ...

extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
E = extrinsic[0].float().cpu().numpy()                    # (S, 3, 4) camera-from-world
Sigma_body = pose_covariances_from_predictions(pred["cholesky_vector"][0], E, kappa=meta["kappa"],
                                               temperature=meta["temperature"], frame="body")
Sigma_world = pose_covariances_from_predictions(pred["cholesky_vector"][0], E, kappa=meta["kappa"],
                                               temperature=meta["temperature"], frame="world")
print(camera_centers(E))                                   # (S, 3) camera centres in world coordinates
print(Sigma_world[:, :3, :3])                              # translational covariance of each centre (scene units^2)
print(rotation_std_deg(Sigma_body))                        # per-axis rotational std-dev in degrees
```

### Conventions (read this before using the covariances)

- Extrinsics are 3×4 **camera-from-world** matrices (OpenCV), as in VGGT. Poses are in VGGT's normalized frame: camera 0 is the identity and the mean point distance from it is 1.
- The twist is ordered **(translation, rotation)**. The Gaussian is defined on the error `ξ = log(E_pred · E_gt⁻¹)`, i.e. `P_gt = P_pred · Exp(ξ)` for the camera-to-world pose `P = E⁻¹`: a perturbation in the **predicted camera's frame**, invariant to the choice of world frame. `frame="world"` rotates it into world coordinates for drawing.
- The network learns the covariance in κ-weighted coordinates (κ = 10 on the rotational components). `pose_covariances_from_predictions` undoes the weighting; the rotational block is then in radians².
- Camera 0's pose is the reference frame and carries no uncertainty; its covariance is not meaningful and is excluded from training and calibration.

## Demos

```bash
export VGGT_UNCERTAINTY_CKPT=vggt_uncertainty_head_v1.pt
python demo_viser.py --image_folder examples/kitchen/images        # interactive viewer: ellipsoids + cones
python demo_viser.py --image_folder examples/kitchen/images --propagate_point_uncertainty   # per-point ellipsoids (slow)
python demo_gradio.py                                              # web UI with uncertainty plots
```

Ellipsoids and cones are drawn at their true 95% size; the *sigma multiplier* slider only rescales the drawing. `demo_colmap.py` is unchanged from upstream.

## Training and evaluation

Fine-tuning the covariance branch on CO3D takes about 11 hours on one 24 GB GPU (20 epochs of 500 steps, validation included). See [training/README.md](training/README.md) for data preparation, the curriculum, and how to evaluate, calibrate and export the head:

```bash
export CO3D_DIR=/path/to/CO3D CO3D_ANNOTATION_DIR=/path/to/CO3D_ann VGGT_PRETRAINED_CKPT=/path/to/model.pt
cd training && torchrun --nproc_per_node=1 launch.py --config default
```

## Reproducing the thesis figures and tables

See [reproduce/README.md](reproduce/README.md): the commands, data splits, frame lists and COLMAP models behind every table and figure of the revised thesis. `reproduce/scene_figures.sh` regenerates the scene figures and the comparison with COLMAP bundle adjustment.

## Limitations

- Trained on 18 CO3D categories (object-centric videos, 2–24 frames per sample) with COLMAP pseudo-ground-truth; the learned uncertainty is relative to COLMAP's solution and may not transfer to very different scenes.
- Unimodal Gaussian on se(3): symmetric or multi-modal ambiguities are not represented, and the pose errors are heavier-tailed than a Gaussian (about 9% of held-out frames fall outside the 99% region).
- The covariance is in VGGT's normalized units; absolute metric scale is not recovered.

## Citation

```bibtex
@thesis{vanni2026vggt_uncertainty,
  title  = {Geometrically-Grounded Uncertainty Quantification for Foundational 3D Vision Models},
  author = {Vanni, Leonardo},
  school = {Bocconi University},
  type   = {Bachelor's thesis},
  year   = {2026}
}

@inproceedings{wang2025vggt,
  title     = {VGGT: Visual Geometry Grounded Transformer},
  author    = {Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}
```

## License and acknowledgements

This is a derivative work of [facebookresearch/vggt](https://github.com/facebookresearch/vggt) and is distributed under the same [CC BY-NC 4.0](./LICENSE.txt) license; see [NOTICE.md](NOTICE.md). The upstream README, demos and training framework are by the VGGT authors; the uncertainty extension is by Leonardo Vanni (supervisor: Prof. Alessandro Pigati). Thanks also to the CO3D, DINOv2 and COLMAP projects.
