# Copyright (c) 2026 Leonardo Vanni. CC BY-NC 4.0 (see LICENSE.txt).
"""Tests for the inference-side helpers (covariance conversion, ellipsoids, cones, loader)."""
import math

import numpy as np
import pytest
import torch

from vggt.utils.checkpoint import (
    UNCERTAINTY_HEAD_PREFIXES,
    export_uncertainty_head,
    load_uncertainty_head,
    remap_upstream_state_dict,
    strip_module_prefix,
)
from vggt.utils.uncertainty import (
    camera_centers,
    chi2_quantile,
    covariance_from_cholesky_vector,
    pose_covariances_from_predictions,
    rotation_cone_half_angles,
    rotation_cone_points,
    rotation_std_deg,
    translation_ellipsoid_points,
)

torch.manual_seed(1)
np.random.seed(1)


def _random_extrinsics(n):
    from scipy.spatial.transform import Rotation

    E = np.zeros((n, 3, 4))
    E[:, :, :3] = Rotation.random(n, random_state=1).as_matrix()
    E[:, :, 3] = np.random.randn(n, 3)
    return E


def test_chi2_quantiles_match_table():
    assert abs(chi2_quantile(3, 0.95) - 7.815) < 1e-3
    assert abs(chi2_quantile(2, 0.95) - 5.991) < 1e-3
    assert abs(chi2_quantile(6, 0.95) - 12.592) < 1e-3


def test_camera_centers():
    E = _random_extrinsics(4)
    c = camera_centers(E)
    for i in range(4):
        assert np.allclose(E[i, :, :3] @ c[i] + E[i, :, 3], 0, atol=1e-10)


def test_pose_covariances_from_predictions_units_and_frames():
    S = 3
    chol = np.random.randn(S, 21) * 0.1
    E = _random_extrinsics(S)
    kappa = 10.0
    body = pose_covariances_from_predictions(chol, E, kappa=kappa, shield_eps=0.0, frame="body")
    world = pose_covariances_from_predictions(chol, E, kappa=kappa, shield_eps=0.0, frame="world")
    raw = covariance_from_cholesky_vector(torch.tensor(chol, dtype=torch.float32), kappa=1.0).numpy()
    assert body.shape == (S, 6, 6)
    # the default adds the training shield, i.e. the covariance the head was calibrated with
    shielded = pose_covariances_from_predictions(chol, E, kappa=kappa, frame="body")
    assert np.allclose(shielded[:, :3, :3], body[:, :3, :3] + 1e-4 * np.eye(3), atol=1e-6)
    assert np.allclose(body[:, :3, :3], raw[:, :3, :3], atol=1e-6)
    assert np.allclose(body[:, 3:, 3:], raw[:, 3:, 3:] / kappa**2, atol=1e-7)
    # frame change is a rotation: eigenvalues of each block preserved, matrix stays symmetric PSD
    for i in range(S):
        assert np.allclose(np.linalg.eigvalsh(world[i, :3, :3]), np.linalg.eigvalsh(body[i, :3, :3]), atol=1e-6)
        assert np.allclose(world[i], world[i].T, atol=1e-7)
        assert np.linalg.eigvalsh(world[i]).min() >= -1e-8
    scaled = pose_covariances_from_predictions(chol, E, kappa=kappa, temperature=3.0, frame="body")
    assert np.allclose(scaled, 3.0 * shielded, atol=1e-6)  # T multiplies the shielded covariance
    with pytest.raises(ValueError):
        pose_covariances_from_predictions(chol, E, frame="camera")


def test_translation_ellipsoid_points_lie_on_confidence_surface():
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.02, 0.0], [0.0, 0.0, 0.09]])
    c = np.array([1.0, -2.0, 0.5])
    pts = translation_ellipsoid_points(c, cov, confidence=0.95)
    d = pts - c
    m = np.einsum("ni,ij,nj->n", d, np.linalg.inv(cov), d)
    assert np.allclose(m, chi2_quantile(3, 0.95), atol=1e-6)


def test_rotation_cone_geometry():
    sx, sy, sz = 0.02, 0.05, 0.3  # std about the camera x, y (tilt) and z (roll) axes, radians
    Sigma = np.zeros((6, 6))
    Sigma[3, 3], Sigma[4, 4], Sigma[5, 5] = sx**2, sy**2, sz**2
    half, dirs, roll = rotation_cone_half_angles(Sigma, confidence=0.95)
    q2 = math.sqrt(chi2_quantile(2, 0.95))
    # rotation about x tilts the axis along y and vice versa
    assert np.allclose(np.sort(half), np.sort([sx * q2, sy * q2]), atol=1e-9)
    assert abs(roll - sz) < 1e-12
    E = _random_extrinsics(1)[0]
    c = camera_centers(E[None])[0]
    apex, rim = rotation_cone_points(c, E, Sigma, confidence=0.95, length=0.5, num_points=64)
    assert np.allclose(apex, c)
    optical_axis_world = E[:, :3].T @ np.array([0, 0, 1.0])
    d = rim - c
    assert np.allclose(np.linalg.norm(d, axis=1), 0.5, atol=1e-9)
    angles = np.arccos(np.clip(d @ optical_axis_world / 0.5, -1, 1))
    assert abs(angles.max() - half.max()) < 1e-6 and abs(angles.min() - half.min()) < 1e-6
    assert np.allclose(rotation_std_deg(Sigma), np.degrees([sx, sy, sz]))


def test_remap_and_strip_prefix():
    sd = {"module.camera_head.pose_branch.fc1.weight": torch.zeros(1), "module.aggregator.x": torch.ones(1), "camera_head.trunk.0.w": torch.ones(1)}
    out = remap_upstream_state_dict(sd)
    assert set(out) == {"camera_head.mean_pose_branch.fc1.weight", "aggregator.x", "camera_head.trunk.0.w"}
    assert list(strip_module_prefix(sd))[0] == "camera_head.pose_branch.fc1.weight"


def test_export_and_load_uncertainty_head_roundtrip(tmp_path):
    from vggt.heads.camera_head import CameraHead

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.camera_head = CameraHead(dim_in=64, trunk_depth=1, num_heads=4)

    src, dst = Tiny(), Tiny()
    fake_ckpt = tmp_path / "checkpoint.pt"
    torch.save({"model": {"module." + k: v for k, v in src.state_dict().items()}, "epoch": 3}, fake_ckpt)
    out = export_uncertainty_head(str(fake_ckpt), str(tmp_path / "head.pt"), meta={"kappa": 7.0, "temperature": 1.5}, half=False)
    payload = torch.load(out, weights_only=False)
    assert all(k.startswith(UNCERTAINTY_HEAD_PREFIXES) for k in payload["state_dict"])
    assert payload["meta"]["kappa"] == 7.0 and payload["meta"]["epoch"] == 3
    meta = load_uncertainty_head(dst, out)
    assert meta["kappa"] == 7.0 and meta["temperature"] == 1.5 and meta["error_convention"] == "body"
    for k, v in src.state_dict().items():
        if k.startswith(UNCERTAINTY_HEAD_PREFIXES):
            assert torch.equal(dst.state_dict()[k], v)
    # the frozen mean pathway is untouched by a head load
    assert not torch.equal(dst.camera_head.mean_pose_branch.fc1.weight, src.camera_head.mean_pose_branch.fc1.weight)
    with pytest.raises(RuntimeError):
        load_uncertainty_head(Tiny(), str(fake_ckpt).replace("checkpoint.pt", "missing.pt")) if False else load_uncertainty_head(
            torch.nn.Module(), out
        )


def test_plot_calibration_fits_the_median_temperature_and_matches_the_training_nll():
    import plot_calibration as pc
    from vggt.utils.uncertainty import gaussian_nll

    rng = np.random.default_rng(0)
    d2 = 2.0 * rng.chisquare(6, size=200_000)  # a head that is overconfident by T = 2
    data = {"mahalanobis_sq": d2, "log_det": np.full_like(d2, -20.0)}
    T = pc.fit_temperature_median(d2)
    assert abs(T - 2.0) < 0.02
    s = pc.summarize(data, T)
    for p in pc.LEVELS:
        assert abs(s[f"coverage_{int(p * 100)}"] - p) < 0.005

    L = torch.linalg.cholesky(torch.diag(torch.tensor([0.3, 0.5, 0.2, 1.1, 0.7, 0.9], dtype=torch.float64)))
    xi = torch.tensor([[0.1, -0.2, 0.3, 0.05, 0.4, -0.1]], dtype=torch.float64)
    nll, m, ld = gaussian_nll(xi, L)
    frame = {"mahalanobis_sq": m.numpy(), "log_det": ld.numpy()}
    assert np.allclose(pc.nll_per_frame(frame, 1.0), nll.numpy())
    nll_T, _, _ = gaussian_nll(xi, L * math.sqrt(3.0))  # temperature 3 on the covariance
    assert np.allclose(pc.nll_per_frame(frame, 3.0), nll_T.numpy())
