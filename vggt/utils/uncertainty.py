# Copyright (c) 2026 Leonardo Vanni.
# Part of a derivative work of VGGT (Meta Platforms, Inc.); distributed under the
# same CC BY-NC 4.0 license found in the LICENSE.txt file in the root directory.

"""Shared SE(3) pose-uncertainty utilities used by the training loss and the demos.

Conventions
-----------
* VGGT ``extrinsics`` are 3x4 camera-from-world matrices (OpenCV convention).
* A twist ``xi = (nu, omega)`` in R^6 lists translation first, rotation second.
* The network predicts the 21 entries of a lower-triangular Cholesky factor ``L`` in
  row-major :func:`torch.tril_indices` order; the six diagonal entries are
  log-standard-deviations (positions 0, 2, 5, 9, 14, 20 of the vector).
* The covariance is learned in *weighted* coordinates ``xi_w = W xi`` with
  ``W = diag(1, 1, 1, kappa, kappa, kappa)``.  Use :func:`unweight_covariance` to get
  back to physical units (normalized scene units for translation, radians for rotation).
* The error convention is body-centric: ``T_err = E_pred @ inv(E_gt) = P_pred^{-1} P_gt``
  where ``P = E^{-1}`` is the camera-to-world pose, i.e. ``P_gt = P_pred Exp(xi)``.  The
  covariance therefore lives in the *predicted camera's* frame and is invariant to the
  choice of world frame.  :func:`covariance_to_world_frame` rotates it into world
  coordinates for visualization.
"""

import math
from typing import Tuple

import numpy as np
import torch

# Sigma = L L^T + eps I in the kappa-weighted twist coordinates (thesis Algorithm 2)
TRAINING_SHIELD_EPS = 1e-4

TRIL_ROWS, TRIL_COLS = torch.tril_indices(6, 6)
DIAGONAL_POSITIONS = [int(i) for i in torch.nonzero(TRIL_ROWS == TRIL_COLS).flatten()]  # [0, 2, 5, 9, 14, 20]
LOG_2PI_TIMES_6 = 6.0 * math.log(2.0 * math.pi)


# --------------------------------------------------------------------------------------
# Lie-group helpers
# --------------------------------------------------------------------------------------
def skew(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) skew-symmetric matrix such that skew(a) @ b = a x b."""
    zero = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], zero], dim=-1),
        ],
        dim=-2,
    )


def vee(M: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) skew-symmetric matrix -> (..., 3) vector (inverse of :func:`skew`)."""
    return torch.stack([M[..., 2, 1], M[..., 0, 2], M[..., 1, 0]], dim=-1)


def to_homogeneous(E: torch.Tensor) -> torch.Tensor:
    """(..., 3, 4) or (..., 4, 4) -> (..., 4, 4)."""
    if E.shape[-2] == 4:
        return E
    bottom = torch.zeros(E.shape[:-2] + (1, 4), dtype=E.dtype, device=E.device)
    bottom[..., 0, 3] = 1.0
    return torch.cat([E, bottom], dim=-2)


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of (..., 4, 4) rigid transforms."""
    R = T[..., :3, :3]
    t = T[..., :3, 3:]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3:] = -Rt @ t
    out[..., 3, 3] = 1.0
    return out


def se3_log_map(T: torch.Tensor, eps: float = 1e-7, small_angle: float = 1e-4) -> torch.Tensor:
    """Logarithm map SE(3) -> se(3), returning the twist ``(nu, omega)`` of shape (..., 6).

    Implements Algorithm 1 of the thesis: the rotation angle from the trace, the axis
    from the skew-symmetric part of ``R`` and the translational component through the
    inverse left Jacobian ``V^{-1} = I - Omega/2 + K Omega^2`` with
    ``K = (1 - theta sin(theta) / (2 (1 - cos(theta)))) / theta^2`` (-> 1/12 as theta -> 0).
    Second-order series are used below ``small_angle``.
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    cos_theta = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)
    small = theta < small_angle
    theta_safe = torch.where(small, torch.ones_like(theta), theta)
    sin_safe = torch.where(small, torch.ones_like(theta), torch.sin(theta_safe))
    one_minus_cos_safe = torch.where(small, torch.ones_like(theta), 1.0 - torch.cos(theta_safe))

    alpha = torch.where(small, 0.5 + theta**2 / 12.0, theta_safe / (2.0 * sin_safe))
    omega = alpha[..., None] * vee(R - R.transpose(-1, -2))

    Omega = skew(omega)
    K = torch.where(
        small,
        torch.full_like(theta, 1.0 / 12.0),
        (1.0 - theta_safe * sin_safe / (2.0 * one_minus_cos_safe)) / theta_safe**2,
    )
    eye = torch.eye(3, dtype=T.dtype, device=T.device).expand_as(R)
    V_inv = eye - 0.5 * Omega + K[..., None, None] * (Omega @ Omega)
    nu = (V_inv @ t[..., None]).squeeze(-1)
    return torch.cat([nu, omega], dim=-1)


def pose_error_twist(E_pred: torch.Tensor, E_gt: torch.Tensor, convention: str = "body") -> torch.Tensor:
    """Twist of the relative transform between predicted and ground-truth extrinsics.

    Args:
        E_pred, E_gt: (..., 3, 4) or (..., 4, 4) camera-from-world matrices.
        convention: ``"body"`` uses ``T_err = E_pred @ inv(E_gt)`` (perturbation in the
            predicted camera's frame, world-frame invariant); ``"world"`` uses
            ``T_err = inv(E_pred) @ E_gt`` (perturbation in the world frame).
    """
    T_pred = to_homogeneous(E_pred)
    T_gt = to_homogeneous(E_gt)
    if convention == "body":
        T_err = T_pred @ se3_inverse(T_gt)
    elif convention == "world":
        T_err = se3_inverse(T_pred) @ T_gt
    else:
        raise ValueError(f"Unknown error convention: {convention!r} (expected 'body' or 'world')")
    return se3_log_map(T_err)


# --------------------------------------------------------------------------------------
# Covariance parameterisation
# --------------------------------------------------------------------------------------
def twist_weights(kappa: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """Diagonal of ``W = diag(1, 1, 1, kappa, kappa, kappa)``."""
    return torch.tensor([1.0, 1.0, 1.0, kappa, kappa, kappa], device=device, dtype=dtype)


def cholesky_vector_to_L(
    cholesky_vector: torch.Tensor,
    log_diag_min: float = None,
    log_diag_max: float = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(..., 21) -> lower-triangular ``L`` (..., 6, 6) with ``exp`` applied to the diagonal.

    Returns ``(L, log_diag)`` where ``log_diag`` (..., 6) are the (optionally clamped)
    log-standard-deviations, i.e. ``diag(L) = exp(log_diag)``.
    """
    if cholesky_vector.shape[-1] != 21:
        raise ValueError(f"expected a 21-dim Cholesky vector, got {tuple(cholesky_vector.shape)}")
    batch_shape = cholesky_vector.shape[:-1]
    L = cholesky_vector.new_zeros(batch_shape + (6, 6))
    rows = TRIL_ROWS.to(cholesky_vector.device)
    cols = TRIL_COLS.to(cholesky_vector.device)
    L[..., rows, cols] = cholesky_vector
    log_diag = L.diagonal(dim1=-2, dim2=-1)
    if log_diag_min is not None or log_diag_max is not None:
        log_diag = torch.clamp(log_diag, min=log_diag_min, max=log_diag_max)
    L = torch.tril(L, diagonal=-1) + torch.diag_embed(torch.exp(log_diag))
    return L, log_diag


def cholesky_bias_for_targets(std_devs, off_diagonal: float = 0.0) -> torch.Tensor:
    """Bias vector (21,) that makes the covariance branch output ``diag(std_devs)``.

    Places ``log(std_dev_i)`` at the diagonal positions of the tril layout and
    ``off_diagonal`` elsewhere.  Use this instead of assuming the first six entries are
    the diagonal.
    """
    L = torch.full((6, 6), float(off_diagonal))
    L[range(6), range(6)] = torch.log(torch.as_tensor(std_devs, dtype=torch.float32))
    return L[TRIL_ROWS, TRIL_COLS]


def covariance_from_L(L: torch.Tensor, shield_eps: float = 0.0) -> torch.Tensor:
    """``Sigma = L L^T + shield_eps I``."""
    Sigma = L @ L.transpose(-1, -2)
    if shield_eps > 0:
        Sigma = Sigma + shield_eps * torch.eye(6, dtype=L.dtype, device=L.device)
    return Sigma


def unweight_covariance(Sigma_weighted: torch.Tensor, kappa: float) -> torch.Tensor:
    """Map a covariance in weighted coordinates back to physical units: ``W^-1 Sigma W^-1``."""
    w_inv = 1.0 / twist_weights(kappa, device=Sigma_weighted.device, dtype=Sigma_weighted.dtype)
    return Sigma_weighted * w_inv[:, None] * w_inv[None, :]


def covariance_from_cholesky_vector(
    cholesky_vector: torch.Tensor,
    kappa: float = 10.0,
    temperature: float = 1.0,
    shield_eps: float = 0.0,
) -> torch.Tensor:
    """Full pipeline used at inference: 21-vector -> 6x6 covariance in physical units.

    ``temperature`` multiplies the covariance (post-hoc calibration); ``shield_eps`` must
    match the value used in training if the shielded covariance is to be reproduced.
    """
    L, _ = cholesky_vector_to_L(cholesky_vector)
    Sigma_w = temperature * covariance_from_L(L, shield_eps)
    return unweight_covariance(Sigma_w, kappa)


def covariance_to_world_frame(Sigma_body: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """Rotate a body-frame (predicted-camera-frame) covariance into world coordinates.

    Args:
        Sigma_body: (..., 6, 6) covariance of the twist ``(nu, omega)`` in the camera frame.
        extrinsics: (..., 3, 4) or (..., 4, 4) camera-from-world matrices of the same cameras.
    """
    R_c2w = extrinsics[..., :3, :3].transpose(-1, -2)
    A = torch.zeros(Sigma_body.shape[:-2] + (6, 6), dtype=Sigma_body.dtype, device=Sigma_body.device)
    A[..., :3, :3] = R_c2w
    A[..., 3:, 3:] = R_c2w
    return A @ Sigma_body @ A.transpose(-1, -2)


# --------------------------------------------------------------------------------------
# Gaussian negative log-likelihood
# --------------------------------------------------------------------------------------
def gaussian_nll(
    xi: torch.Tensor, L: torch.Tensor, shield_eps: float = 0.0, include_constant: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NLL of ``xi ~ N(0, L L^T + shield_eps I)`` per sample (Algorithm 2 of the thesis).

    The *same* shielded covariance is used for the Mahalanobis term and the
    log-determinant, so the objective is bounded below.

    Returns ``(nll, mahalanobis_sq, log_det)`` each of shape ``xi.shape[:-1]``.
    """
    Sigma = covariance_from_L(L, shield_eps)
    chol = torch.linalg.cholesky(Sigma)
    y = torch.linalg.solve_triangular(chol, xi[..., None], upper=False).squeeze(-1)
    mahalanobis_sq = (y * y).sum(dim=-1)
    log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    nll = 0.5 * (mahalanobis_sq + log_det + (LOG_2PI_TIMES_6 if include_constant else 0.0))
    return nll, mahalanobis_sq, log_det


# --------------------------------------------------------------------------------------
# Inference-side helpers (numpy in / numpy out) shared by the demos
# --------------------------------------------------------------------------------------
_CHI2_TABLE = {  # chi-square quantiles for dof 1..3 at common confidence levels
    (1, 0.68): 0.989, (1, 0.90): 2.706, (1, 0.95): 3.841, (1, 0.99): 6.635,
    (2, 0.68): 2.279, (2, 0.90): 4.605, (2, 0.95): 5.991, (2, 0.99): 9.210,
    (3, 0.68): 3.506, (3, 0.90): 6.251, (3, 0.95): 7.815, (3, 0.99): 11.345,
    (6, 0.68): 6.980, (6, 0.90): 10.645, (6, 0.95): 12.592, (6, 0.99): 16.812,
}


def chi2_quantile(dof: int, confidence: float) -> float:
    """Quantile of the chi-square distribution (scipy if available, table otherwise)."""
    try:
        from scipy.stats import chi2

        return float(chi2.ppf(confidence, df=dof))
    except Exception:  # pragma: no cover - scipy is a demo dependency
        key = (dof, round(confidence, 2))
        if key not in _CHI2_TABLE:
            raise ValueError(f"no chi-square table entry for dof={dof}, confidence={confidence}; install scipy")
        return _CHI2_TABLE[key]


def _as_tensor(x, dtype=torch.float32) -> torch.Tensor:
    return x.detach().to(dtype) if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), dtype=dtype)


def pose_covariances_from_predictions(
    cholesky_vectors,
    extrinsics,
    kappa: float = 10.0,
    temperature: float = 1.0,
    shield_eps: float = TRAINING_SHIELD_EPS,
    frame: str = "world",
):
    """Turn the network output into 6x6 pose covariances in physical units.

    Args:
        cholesky_vectors: (S, 21) network output (torch or numpy).
        extrinsics: (S, 3, 4) predicted camera-from-world matrices of the same cameras.
        kappa: rotation weight used in training (rotation block is divided by kappa^2).
        temperature: post-hoc calibration factor multiplying the covariance.
        shield_eps: the shield ``eps I`` added to ``L L^T``.  The default is the training
            value, so the result is the covariance the NLL was trained and calibrated
            with; pass 0 for the raw network output.
        frame: ``"body"`` (predicted camera frame, as trained) or ``"world"``.

    Returns:
        numpy array (S, 6, 6); twist ordering (translation, rotation), translation in
        normalized scene units, rotation in radians.
    """
    chol = _as_tensor(cholesky_vectors)
    Sigma = covariance_from_cholesky_vector(chol, kappa=kappa, temperature=temperature, shield_eps=shield_eps)
    if frame == "world":
        Sigma = covariance_to_world_frame(Sigma, _as_tensor(extrinsics))
    elif frame != "body":
        raise ValueError(f"frame must be 'world' or 'body', got {frame!r}")
    return Sigma.cpu().numpy()


def camera_centers(extrinsics) -> np.ndarray:
    """(S, 3, 4) camera-from-world -> (S, 3) camera centres in world coordinates."""
    E = np.asarray(extrinsics)
    R, t = E[..., :3, :3], E[..., :3, 3]
    return -np.einsum("...ji,...j->...i", R, t)


def translation_ellipsoid_points(center, cov3, confidence: float = 0.95, num_points: int = 400) -> np.ndarray:
    """Points on the surface of the ``confidence`` ellipsoid of a 3D Gaussian (world frame)."""
    eigval, eigvec = np.linalg.eigh(np.asarray(cov3, dtype=np.float64))
    radii = np.sqrt(np.maximum(eigval, 0.0) * chi2_quantile(3, confidence))
    n = max(int(math.sqrt(num_points)), 4)
    phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi, 2 * n), np.linspace(0, np.pi, n))
    sphere = np.stack([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)], -1).reshape(-1, 3)
    return (sphere * radii) @ eigvec.T + np.asarray(center)


def rotation_cone_half_angles(Sigma_body, confidence: float = 0.95):
    """Half-angles (rad) and in-plane directions of the optical-axis uncertainty cone.

    A small body-frame rotation ``delta`` tilts the optical axis ``z`` by
    ``delta x z = (delta_y, -delta_x, 0)``; the cone's cross-section is therefore the
    ``confidence`` ellipse of that 2D tilt vector, whose covariance is
    ``M Sigma_rr[:2,:2] M^T`` with ``M = [[0, 1], [-1, 0]]``.  Rotation about the optical
    axis (roll) does not move the axis and is reported separately as a std-dev.

    Returns ``(half_angles (2,), directions (2, 2) in the camera x/y plane, roll_std)``.
    """
    Sigma = np.asarray(Sigma_body, dtype=np.float64)
    M = np.array([[0.0, 1.0], [-1.0, 0.0]])
    tilt_cov = M @ Sigma[3:5, 3:5] @ M.T
    eigval, eigvec = np.linalg.eigh(tilt_cov)
    half_angles = np.sqrt(np.maximum(eigval, 0.0) * chi2_quantile(2, confidence))
    roll_std = math.sqrt(max(Sigma[5, 5], 0.0))
    return half_angles, eigvec, roll_std


def rotation_cone_points(center, extrinsic, Sigma_body, confidence: float = 0.95, length: float = 0.1, num_points: int = 48):
    """Apex and rim points of the cone swept by the optical axis under rotational uncertainty.

    Args:
        center: (3,) camera centre in world coordinates.
        extrinsic: (3, 4) camera-from-world matrix of the camera.
        Sigma_body: (6, 6) covariance in the predicted camera's frame (physical units).
        length: cone length in scene units.
    Returns ``(apex (3,), rim (num_points, 3))``.
    """
    half_angles, directions, _ = rotation_cone_half_angles(Sigma_body, confidence)
    # A cone wider than a hemisphere has no meaning; clamp so the drawing stays finite.
    half_angles = np.minimum(half_angles, math.radians(85.0))
    R_c2w = np.asarray(extrinsic)[:3, :3].T
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    tilt_xy = (np.stack([np.cos(t), np.sin(t)], -1) * half_angles) @ directions.T  # (N, 2) tilt of the axis
    dirs_cam = np.concatenate([np.tan(tilt_xy), np.ones((num_points, 1))], axis=-1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    rim = np.asarray(center) + length * (dirs_cam @ R_c2w.T)
    return np.asarray(center), rim


def rotation_std_deg(Sigma_body) -> np.ndarray:
    """Per-axis rotational standard deviations in degrees from a body-frame covariance (..., 6, 6)."""
    Sigma = np.asarray(Sigma_body)
    return np.degrees(np.sqrt(np.maximum(np.diagonal(Sigma[..., 3:, 3:], axis1=-2, axis2=-1), 0.0)))
