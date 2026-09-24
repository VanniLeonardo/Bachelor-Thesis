# Pose uncertainty: representation and conventions

This note explains exactly what the covariance predicted by the uncertainty head means and
how to consume it. The code that implements it is `vggt/utils/uncertainty.py`.

## Frames and parameterisation

- VGGT predicts, for each input image, a 3×4 **camera-from-world** extrinsic `E = [R | t]`
  (OpenCV convention) in a normalized frame: camera 0 is the identity and the mean distance
  of the scene points from camera 0 is 1. Camera centres are `c = -Rᵀ t`.
- The camera-to-world **pose** is `P = E⁻¹`. The model describes the ground-truth pose as
  `P_gt = P_pred · Exp(ξ)` with `ξ = (ν, ω) ∈ se(3)`, translation first, rotation second.
  Equivalently `ξ = log(E_pred · E_gt⁻¹)`. This is a right (body-centric) perturbation: `ξ`
  lives in the **predicted camera's frame** and does not change if the world frame is
  re-referenced.
- `ξ ~ N(0, Σ)`. The network outputs the 21 entries of a lower-triangular `L` (row-major
  `torch.tril_indices(6, 6)` order; the diagonal entries at positions 0, 2, 5, 9, 14, 20
  are log-standard-deviations) such that `Σ_w = L Lᵀ + ε I` in **κ-weighted** coordinates
  `ξ_w = diag(1, 1, 1, κ, κ, κ) ξ`, κ = 10.
- Physical units are recovered by `Σ = W⁻¹ Σ_w W⁻¹`: translation in normalized scene units,
  rotation in radians. `pose_covariances_from_predictions(..., frame="body")` does this;
  `frame="world"` additionally applies `blockdiag(R_c2w, R_c2w)` to express the covariance
  in world coordinates, which is what you want for drawing ellipsoids around camera centres.
- A post-hoc temperature `T` multiplies the shielded covariance, `T (L Lᵀ + ε I)`; the value
  fitted on held-out data (1.015 for the released head) is stored in the head file's metadata
  and applied by the loaders. `pose_covariances_from_predictions` includes the shield `ε I` by
  default, since it is part of the distribution the head was trained and calibrated with.

## What the numbers mean

- `Σ[:3, :3]` (world frame): covariance of the camera-centre error. The 95% ellipsoid has
  semi-axes `sqrt(7.815 · eigenvalues)`.
- `Σ[3:, 3:]` (body frame): covariance of the rotation vector. Rotations about the camera
  x and y axes tilt the optical axis; `rotation_cone_half_angles` gives the 95% cone of the
  optical axis, roll is reported separately.
- Camera 0 defines the frame; its covariance is not meaningful and is excluded from
  training and calibration.
- Under a calibrated model the squared Mahalanobis distance `ξᵀ Σ⁻¹ ξ` follows χ²₆
  (mean 6). `plot_calibration.py` checks this on validation data.
- For the released head, on held-out CO3D sequences, the median of `ξᵀ Σ⁻¹ ξ` matches χ²₆
  at every level of predicted uncertainty, but the errors are heavier-tailed than a Gaussian:
  about 9% of frames fall outside the 99% region instead of 1%, and gross failures of VGGT
  (errors of tens of degrees) lie far outside it. Read the ellipsoids and cones as calibrated
  for typical frames, not as guaranteed bounds.

## Training objective (summary)

`L = w_NLL · NLL + f · (λ_scale L_scale + λ_cond L_cond + λ_bal L_bal)` with

- `NLL = ½ (ξ_wᵀ Σ_w⁻¹ ξ_w + log det Σ_w + 6 log 2π)`, the same shielded `Σ_w` in both terms;
- `L_scale = mean_i (exp(max(0, σ_i − r)) − 1)` over the three translational std-devs, r = 0.05;
- `L_cond = max(0, log(λ_max / λ_min) − log 20)`;
- `L_bal = mean(ℓ²) + Σ (exp(max(|ℓ| − log 5, 0)) − 1)` over the pairwise log-ratios of the translational std-devs;
- curriculum: `w_NLL = 0` for the warm-up epoch, then 1; `f` decays linearly from 1 towards 0, which it would reach at `max_epochs`.

## Propagating to 3D points

`vggt.utils.geometry.unproject_depth_map_with_uncertainty` propagates the body-frame pose
covariance (physical units) and a per-pixel depth variance to every unprojected point with
first-order Jacobians. Pass `pose_covariances_from_predictions(..., frame="body")`.
