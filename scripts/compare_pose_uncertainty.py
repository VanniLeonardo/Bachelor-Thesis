#!/usr/bin/env python3
# Copyright (c) 2026 Leonardo Vanni. CC BY-NC 4.0 (see LICENSE.txt).
"""Learned versus bundle-adjustment pose uncertainty on one scene: figures and agreement.

Runs VGGT with the uncertainty head on an image folder and, when a COLMAP covariance file
from ``colmap_pose_covariance.py`` is given, compares the two.  Both are expressed in the
first camera's frame.  VGGT's units are those of its normalized output (mean distance of
the confident predicted points from camera 0 rescaled to one); COLMAP is rescaled so that
the mean distance of its camera centres from camera 0 equals VGGT's, the same quantity its
gauge holds fixed.  Ellipsoids and
cones are drawn at 95% confidence and magnified by the same factor for both methods.

Writes, for scene NAME, into --out_dir:
  NAME_translation.png   VGGT camera-centre ellipsoids on the predicted point cloud
  NAME_rotation.png      VGGT optical-axis cones
  NAME_colmap.png        COLMAP camera-centre ellipsoids, same view and magnification
  NAME_per_camera.png    translational and rotational sigma per camera, both methods
  NAME_uncertainty.npz   per-camera values; NAME_agreement.json  summary metrics

    python scripts/compare_pose_uncertainty.py --images scene/images --name kitchen \\
        --head vggt_uncertainty_head_v1.pt --colmap kitchen_colmap.npz --out_dir figures
"""
import argparse
import json
import math
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import chi2, spearmanr  # noqa: E402

from vggt.utils.checkpoint import load_vggt_with_uncertainty  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
from vggt.utils.uncertainty import camera_centers, pose_covariances_from_predictions, rotation_cone_points  # noqa: E402

VGGT_COLOR, COLMAP_COLOR = "#2a78d6", "#eb6834"  # validated categorical pair (light surface)
CONFIDENCE = 0.95


def display(p):
    """OpenCV camera-0 coordinates (x right, y down, z forward) -> plot axes (x, z, up)."""
    p = np.asarray(p)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def ellipsoid_surface(center, cov3, magnify, n=18):
    w, V = np.linalg.eigh(cov3)
    radii = magnify * np.sqrt(np.maximum(w, 0) * chi2.ppf(CONFIDENCE, 3))
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n // 2 + 1))
    sphere = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], -1)
    pts = display((sphere * radii) @ V.T + center)
    return pts[..., 0], pts[..., 1], pts[..., 2]


def times(m):
    return "true size" if m == 1 else (f"x{m:.0f}" if m >= 10 else f"x{m:.2g}")


def major_radius(cov3):
    return math.sqrt(max(np.linalg.eigvalsh(cov3)[-1], 0) * chi2.ppf(CONFIDENCE, 3))


def sigma(cov3):  # scalar size: root mean variance
    return math.sqrt(max(np.trace(cov3), 0) / 3)


def run_vggt(images_dir, head, base, device):
    names = sorted(os.listdir(images_dir))
    model, meta = load_vggt_with_uncertainty(head, base=base, device=device)
    images = load_and_preprocess_images([os.path.join(images_dir, n) for n in names]).to(device)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        pred = model(images)
    E, _ = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    E = E[0].float().cpu().numpy()
    chol = pred["cholesky_vector"][0].float().cpu()
    pts = pred["world_points"][0].float().cpu().numpy().reshape(-1, 3)
    conf = pred["world_points_conf"][0].float().cpu().numpy().reshape(-1)
    rgb = images.permute(0, 2, 3, 1).float().cpu().numpy().reshape(-1, 3)
    keep = conf >= np.median(conf)
    pts, rgb = pts[keep], rgb[keep]
    scale = 1.0 / np.mean(np.linalg.norm(pts, axis=1))  # VGGT's world frame is camera 0
    world = pose_covariances_from_predictions(chol, E, kappa=meta["kappa"], temperature=meta["temperature"], frame="world")
    body = pose_covariances_from_predictions(chol, E, kappa=meta["kappa"], temperature=meta["temperature"], frame="body")
    return {
        "names": names, "E": E, "scale": scale, "centres": scale * camera_centers(E),
        "cov_centre": scale**2 * world[:, :3, :3], "cov_rot": body[:, 3:, 3:], "body": body,
        "points": scale * pts, "colors": np.clip(rgb, 0, 1), "meta": meta,
    }


def save_trimmed(fig, path, pad=12):
    """Save and crop the white margin that 3D axes leave around the scene."""
    from PIL import Image, ImageChops
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(path).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    if box:
        im.crop((max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, im.width), min(box[3] + pad, im.height))).save(path)


def scene_axes(ax, lim):
    ax.set_xlim(*lim[0]), ax.set_ylim(*lim[1]), ax.set_zlim(*lim[2])
    ax.set_box_aspect([l[1] - l[0] for l in lim])
    ax.view_init(elev=25, azim=-65)
    ax.set_axis_off()


def draw_cloud(ax, points, colors, lim, n=15000, seed=0):
    p = display(points)
    inside = np.all([(p[:, k] >= lim[k][0]) & (p[:, k] <= lim[k][1]) for k in range(3)], axis=0)
    p, colors = p[inside], colors[inside] if np.ndim(colors) > 1 else colors
    idx = np.random.default_rng(seed).choice(len(p), size=min(n, len(p)), replace=False)
    c = colors[idx] if np.ndim(colors) > 1 else colors
    ax.scatter(p[idx, 0], p[idx, 1], p[idx, 2], c=c, s=0.4, alpha=0.5, linewidths=0, depthshade=False)


def draw_ellipsoids(ax, centres, covs, mask, magnify, color, title):
    for i in np.flatnonzero(mask):
        ax.plot_surface(*ellipsoid_surface(centres[i], covs[i], magnify), color=color, alpha=0.35, linewidth=0, shade=False)
    c = display(centres)
    ax.scatter(*c[mask].T, color="#222222", s=6, depthshade=False)
    ax.scatter(*c[:1].T, color="#222222", marker="^", s=40, depthshade=False)  # camera 0 (reference)
    ax.set_title(title, fontsize=13)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--head", required=True, help="exported uncertainty head (.pt)")
    ap.add_argument("--base", default=None, help="VGGT-1B weights (default: download)")
    ap.add_argument("--colmap", default=None, help="output of colmap_pose_covariance.py for the same images")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--magnify", type=float, default=None, help="ellipsoid magnification (default: median VGGT ellipsoid = 4%% of the scene)")
    ap.add_argument("--cone_magnify", type=float, default=None, help="cone half-angle magnification (default: median cone = 8 deg)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda suffix: os.path.join(args.out_dir, f"{args.name}_{suffix}")  # noqa: E731

    device = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 12e9 else "cpu"
    v = run_vggt(args.images, args.head, args.base, device)
    S = len(v["names"])
    cams = np.arange(S) > 0  # camera 0 is the reference and carries no uncertainty
    cm = None
    if args.colmap:
        cm = dict(np.load(args.colmap))
        assert list(cm["names"]) == v["names"], "COLMAP and VGGT image lists differ"
        cams_cm = cm["registered"] & cams
        # Same scale as VGGT: the mean distance of the camera centres from camera 0 is the
        # quantity the COLMAP gauge holds fixed, so matching it puts both in the same units.
        both = np.flatnonzero(cams_cm)
        # (median of the per-camera ratios, so that cameras on which the two trajectories disagree do not set the scale)
        k = np.median(np.linalg.norm(v["centres"][both], axis=1) / np.linalg.norm(cm["centres"][both], axis=1))
        cm["centres"], cm["cov_centre"], cm["points"] = k * cm["centres"], k**2 * cm["cov_centre"], k * cm["points"]

    # The view frames the cameras (the figure is about them); points outside it are clipped.
    # Ellipsoids are drawn at true size when the median one spans 3-25% of the camera
    # spread and rescaled to 8% otherwise; the factor is printed in every panel title.
    cam_d = display(np.concatenate([v["centres"]] + ([cm["centres"][cm["registered"]]] if cm is not None else [])))
    cam_extent = max(np.ptp(cam_d, axis=0).max(), 1e-6)
    med_v = np.median([major_radius(v["cov_centre"][i]) for i in np.flatnonzero(cams)])
    magnify = args.magnify or (1.0 if 0.03 <= med_v / cam_extent <= 0.25 else 0.08 * cam_extent / med_v)
    max_r = magnify * max(major_radius(v["cov_centre"][i]) for i in np.flatnonzero(cams))
    margin = max(0.25 * cam_extent, max_r)
    lim = [(cam_d[:, k].min() - margin, cam_d[:, k].max() + margin) for k in range(3)]
    extent = max(l[1] - l[0] for l in lim)
    magnify_cm = None
    if cm is not None:  # COLMAP drawn so that its median ellipsoid matches VGGT's drawn one
        med_c = np.median([major_radius(cm["cov_centre"][i]) for i in np.flatnonzero(cams_cm)])
        magnify_cm = args.magnify or magnify * med_v / med_c

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")
    draw_cloud(ax, v["points"], v["colors"], lim)
    draw_ellipsoids(ax, v["centres"], v["cov_centre"], cams, magnify, VGGT_COLOR, f"VGGT (learned), 95% ellipsoids, {times(magnify)}")
    scene_axes(ax, lim)
    save_trimmed(fig, out("translation.png"))

    half = [math.degrees(math.sqrt(max(np.linalg.eigvalsh(v["cov_rot"][i])[-1], 0) * chi2.ppf(CONFIDENCE, 2))) for i in np.flatnonzero(cams)]
    cone_magnify = args.cone_magnify or max(1.0, 8.0 / np.median(half))
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")
    draw_cloud(ax, v["points"], v["colors"], lim)
    for i in np.flatnonzero(cams):
        body = v["body"][i].copy()
        body[3:, 3:] *= cone_magnify**2
        E_i = v["E"][i].copy()
        E_i[:, 3] *= v["scale"]  # same units as the drawn scene
        apex, rim = rotation_cone_points(v["centres"][i], E_i, body, confidence=CONFIDENCE, length=0.08 * extent, num_points=32)
        r = display(np.vstack([rim, rim[:1]]))
        ax.plot(*r.T, color=VGGT_COLOR, linewidth=0.8)
        for k in range(0, len(rim), 8):
            ax.plot(*display(np.vstack([apex, rim[k]])).T, color=VGGT_COLOR, linewidth=0.6)
    c = display(v["centres"])
    ax.scatter(*c[:1].T, color="#222222", marker="^", s=40, depthshade=False)
    ax.set_title(f"VGGT (learned), 95% optical-axis cones, {times(cone_magnify)}", fontsize=13)
    scene_axes(ax, lim)
    save_trimmed(fig, out("rotation.png"))

    sig_t_v = np.array([sigma(c) for c in v["cov_centre"]])
    sig_r_v = np.degrees([sigma(c) for c in v["cov_rot"]])
    record = {"names": np.array(v["names"]), "vggt_centres": v["centres"], "vggt_cov_centre": v["cov_centre"],
              "vggt_cov_rot": v["cov_rot"], "vggt_scale": v["scale"], "magnify": magnify, "cone_magnify": cone_magnify}
    summary = {"scene": args.name, "images": S, "device": device, "vggt_scale": float(v["scale"]),
               "temperature": v["meta"]["temperature"], "magnify": float(magnify), "cone_magnify": float(cone_magnify),
               "magnify_colmap": None if magnify_cm is None else float(magnify_cm),
               "vggt_sigma_t_median": float(np.median(sig_t_v[cams])), "vggt_sigma_rot_deg_median": float(np.median(sig_r_v[cams]))}

    if cm is not None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection="3d")
        draw_cloud(ax, cm["points"], "#9a9a9a", lim)  # COLMAP's sparse model stores no colours
        draw_ellipsoids(ax, cm["centres"], cm["cov_centre"], cams_cm, magnify_cm, COLMAP_COLOR, f"COLMAP (bundle adjustment), 95% ellipsoids, {times(magnify_cm)}")
        scene_axes(ax, lim)
        save_trimmed(fig, out("colmap.png"))

        idx = np.flatnonzero(cams_cm)
        sig_t_c = np.array([sigma(cm["cov_centre"][i]) if cams_cm[i] else np.nan for i in range(S)])
        sig_r_c = np.degrees([sigma(cm["cov_rot"][i]) if cams_cm[i] else np.nan for i in range(S)])
        axis_cos = [abs(np.linalg.eigh(v["cov_centre"][i])[1][:, -1] @ np.linalg.eigh(cm["cov_centre"][i])[1][:, -1]) for i in idx]
        summary.update({
            "colmap_registered": int(cm["registered"].sum()), "colmap_scale": float(cm["scale"]),
            "colmap_sigma_t_median": float(np.nanmedian(sig_t_c[idx])), "colmap_sigma_rot_deg_median": float(np.nanmedian(sig_r_c[idx])),
            "ratio_sigma_t_median": float(np.median(sig_t_v[idx] / sig_t_c[idx])),
            "ratio_sigma_rot_median": float(np.median(sig_r_v[idx] / sig_r_c[idx])),
            "spearman_sigma_t": float(spearmanr(sig_t_v[idx], sig_t_c[idx])[0]),
            "spearman_sigma_rot": float(spearmanr(sig_r_v[idx], sig_r_c[idx])[0]),
            "principal_axis_abs_cos_median": float(np.median(axis_cos)),
            "centre_distance_median": float(np.median(np.linalg.norm(v["centres"][idx] - cm["centres"][idx], axis=1))),
        })
        record.update({"colmap_centres": cm["centres"], "colmap_cov_centre": cm["cov_centre"], "colmap_cov_rot": cm["cov_rot"]})

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (vals_v, key, label) in zip(axs, [(sig_t_v, "t", "Translational $\\sigma$ (scene units)"), (sig_r_v, "rot", "Rotational $\\sigma$ (deg)")]):
        k = np.flatnonzero(cams)
        ax.plot(k, vals_v[k], "-o", color=VGGT_COLOR, linewidth=2, markersize=4, label="VGGT (learned)")
        if cm is not None:
            vals_c = sig_t_c if key == "t" else sig_r_c
            ax.plot(k, vals_c[k], "-o", color=COLMAP_COLOR, linewidth=2, markersize=4, label="COLMAP (BA)")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Camera index")
        ax.set_ylabel(label)
        ax.grid(True, which="major", color="#dddddd", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    if cm is not None:
        axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out("per_camera.png"), dpi=200, bbox_inches="tight"), plt.close(fig)

    np.savez(out("uncertainty.npz"), sigma_t_vggt=sig_t_v, sigma_rot_deg_vggt=sig_r_v, **record)
    with open(out("agreement.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
