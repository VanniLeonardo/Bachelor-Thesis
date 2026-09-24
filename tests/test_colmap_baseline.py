# Copyright (c) 2026 Leonardo Vanni. CC BY-NC 4.0 (see LICENSE.txt).
"""The geometry behind the COLMAP baseline (scripts/colmap_pose_covariance.py)."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("pycolmap")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from colmap_pose_covariance import centre_jacobian, fix_scale_gauge  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


def _cameras(n, rng):
    Rs = [Rotation.random(random_state=int(rng.integers(1e9))).as_matrix() for _ in range(n)]
    ts = [rng.normal(size=3) * 2 for _ in range(n)]
    return Rs, ts, [-R.T @ t for R, t in zip(Rs, ts)]


def test_centre_jacobian_matches_ceres_left_quaternion_update():
    # Ceres EigenQuaternionManifold: q' = [cos|d|, sin|d| d/|d|] * q, a rotation by 2d on the left.
    rng = np.random.default_rng(0)
    for R, t, c in zip(*_cameras(20, rng)):
        x = rng.normal(size=6) * 1e-6
        Rp = Rotation.from_rotvec(2 * x[:3]).as_matrix() @ R
        fd = -Rp.T @ (t + x[3:]) - c
        assert np.allclose(fd, centre_jacobian(R, t) @ x, rtol=1e-4, atol=1e-12)


def test_scale_gauge_projection_removes_the_scale_direction_exactly():
    rng = np.random.default_rng(1)
    Rs, ts, cs = _cameras(5, rng)
    c0 = rng.normal(size=3)
    J = [centre_jacobian(R, t) for R, t in zip(Rs, ts)]
    A = rng.normal(size=(30, 30))
    null = np.concatenate([np.concatenate([np.zeros(3), -R @ (c - c0)]) for R, c in zip(Rs, cs)])
    C = A @ A.T + 1e6 * np.outer(null, null)  # a free scale shows up as a huge variance along null
    C_fixed, G = fix_scale_gauge(C, Rs, cs, J, c0)
    assert abs(G @ C_fixed @ G) < 1e-8 * abs(G @ C @ G)  # the gauge quantity is now fixed
    assert np.abs(C_fixed).max() < 1e-3 * np.abs(C).max()  # the huge scale component is gone
    assert np.allclose(C_fixed, C_fixed.T)
