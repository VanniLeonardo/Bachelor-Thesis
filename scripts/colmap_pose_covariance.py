#!/usr/bin/env python3
# Copyright (c) 2026 Leonardo Vanni. CC BY-NC 4.0 (see LICENSE.txt).
"""Bundle-adjustment pose covariance of a COLMAP model, in the same gauge and units as VGGT.

VGGT expresses every pose relative to the first input camera, in units where the mean
distance of the scene points from that camera is one.  This script brings the COLMAP
bundle-adjustment covariance into the same frame:

1. Bundle adjustment with the pose of the first input camera held constant (6 of the 7
   gauge degrees of freedom of a monocular reconstruction), then the joint covariance
   of all other poses (``pycolmap.estimate_ba_covariance``).
2. The 7th degree of freedom, the global scale about the first camera, is still free.
   It is removed with an S-transform (a gauge projection): the covariance is projected
   onto the constraint that the mean distance of the camera centres from the first camera
   is fixed.  Holding a 3D point or one coordinate of a second camera constant instead
   would add information or make that camera degenerate.
3. The covariance is propagated to the camera centre, rotated into the first camera's
   frame and rescaled so that the mean distance of the 3D points from the first camera
   is one.

COLMAP parameterizes a rotation with Ceres' EigenQuaternionManifold:
``q' = [cos|d|, sin|d| d/|d|] * q``, i.e. a left perturbation by the rotation vector
``2 d``.  The covariance is ordered [rotation, translation].  So the rotation
covariance in radians is 4 times COLMAP's, and the centre ``c = -R^T t`` moves by
``dc = -R^T [t]x (2 d) - R^T dt``.

Needs pycolmap >= 3.12 (bundle-adjustment covariance), e.g. ``pip install pycolmap==3.14.0``.

    python scripts/colmap_pose_covariance.py --model scene/sparse/0 --images scene/images --out scene_colmap.npz
"""
import argparse
import os

import numpy as np
import pycolmap


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def centre_jacobian(R, t):
    """d(camera centre) / d(rotation tangent, translation) for COLMAP's left quaternion update."""
    return np.hstack([-2.0 * R.T @ skew(t), -R.T])


def fix_scale_gauge(C, Rs, centres, J, c0):
    """S-transform of the joint pose covariance C (6n x 6n, [rotation, translation] per camera)
    onto the gauge in which the mean distance of the camera centres from c0 is fixed."""
    n = len(Rs)
    null = np.concatenate([np.concatenate([np.zeros(3), -R @ (c - c0)]) for R, c in zip(Rs, centres)])
    G = np.concatenate([(c - c0) / np.linalg.norm(c - c0) @ Ji for c, Ji in zip(centres, J)]) / n
    P = np.eye(6 * n) - np.outer(null, G) / (G @ null)
    return P @ C @ P.T, G


def pose_of(image):
    p = image.cam_from_world
    p = p() if callable(p) else p
    T = np.asarray(p.matrix() if callable(p.matrix) else p.matrix, dtype=np.float64)
    return T[:, :3], T[:, 3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="COLMAP sparse model directory")
    ap.add_argument("--images", required=True, help="image folder given to VGGT; sorted names define the order, the first is camera 0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = sorted(os.listdir(args.images))
    recon = pycolmap.Reconstruction(args.model)
    by_name = {recon.images[i].name: i for i in recon.reg_image_ids()}
    if names[0] not in by_name:
        raise SystemExit(f"camera 0 ({names[0]}) is not registered by COLMAP; the gauges cannot be matched")
    anchor = by_name[names[0]]
    others = [by_name[n] for n in names[1:] if n in by_name]

    config = pycolmap.BundleAdjustmentConfig()
    for iid in [anchor] + others:
        config.add_image(iid)
    config.set_constant_rig_from_world_pose(recon.image(anchor).frame_id)
    adjuster = pycolmap.create_default_bundle_adjuster(pycolmap.BundleAdjustmentOptions(), config, recon)
    adjuster.solve()
    cov = pycolmap.estimate_ba_covariance(
        pycolmap.BACovarianceOptions(params=pycolmap.BACovarianceOptionsParams.POSES), recon, adjuster)
    if cov is None:
        raise SystemExit("estimate_ba_covariance failed")

    n = len(others)
    C = np.zeros((6 * n, 6 * n))
    for a, ia in enumerate(others):
        for b, ib in enumerate(others):
            block = cov.get_cam_cov_from_world(ia) if a == b else cov.get_cam_cross_cov_from_world(ia, ib)
            C[6 * a:6 * a + 6, 6 * b:6 * b + 6] = np.asarray(block)

    R0, t0 = pose_of(recon.image(anchor))
    c0 = -R0.T @ t0
    Rs, ts, cs, J = [], [], [], []
    for iid in others:
        R, t = pose_of(recon.image(iid))
        Rs.append(R), ts.append(t), cs.append(-R.T @ t)
        J.append(centre_jacobian(R, t))
    C_fixed, G = fix_scale_gauge(C, Rs, cs, J, c0)

    pts = np.array([p.xyz for p in recon.points3D.values()])
    rgb = np.array([p.color for p in recon.points3D.values()], dtype=np.uint8)
    pts_c0 = pts @ R0.T + t0
    scale = 1.0 / np.mean(np.linalg.norm(pts_c0, axis=1))

    def centre_axis_alignment(Cm):  # |cos| between each principal axis and the direction to camera 0
        out = []
        for a, (c, Ji) in enumerate(zip(cs, J)):
            w, V = np.linalg.eigh(Ji @ Cm[6 * a:6 * a + 6, 6 * a:6 * a + 6] @ Ji.T)
            out.append(abs(V[:, -1] @ (c - c0)) / np.linalg.norm(c - c0))
        return float(np.median(out))

    S = len(names)
    registered = np.zeros(S, bool)
    centres = np.full((S, 3), np.nan)
    cov_centre = np.full((S, 3, 3), np.nan)
    cov_rot = np.full((S, 3, 3), np.nan)
    registered[0], centres[0], cov_centre[0], cov_rot[0] = True, 0.0, 0.0, 0.0
    for a, (iid, R, c, Ji) in enumerate(zip(others, Rs, cs, J)):
        k = names.index(recon.images[iid].name)
        Ci = C_fixed[6 * a:6 * a + 6, 6 * a:6 * a + 6]
        registered[k] = True
        centres[k] = scale * R0 @ (c - c0)
        cov_centre[k] = scale**2 * R0 @ (Ji @ Ci @ Ji.T) @ R0.T
        cov_rot[k] = 4.0 * Ci[:3, :3]  # rad^2, left perturbation in the camera frame (as VGGT's body twist)

    gauge_residual = float(G @ C_fixed @ G / (G @ C @ G))
    print(f"{args.model}: {int(registered.sum())}/{S} images registered, scale {scale:.4g}")
    print(f"  median |cos(principal axis, direction to camera 0)|: scale free {centre_axis_alignment(C):.3f}, "
          f"scale fixed {centre_axis_alignment(C_fixed):.3f}")
    print(f"  variance of the gauge constraint after the projection (relative): {gauge_residual:.1e}")
    np.savez(args.out, names=np.array(names), registered=registered, centres=centres,
             cov_centre=cov_centre, cov_rot=cov_rot, scale=scale, gauge_residual=gauge_residual,
             points=scale * pts_c0, colors=rgb)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
