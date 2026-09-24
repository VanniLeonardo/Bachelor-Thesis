#!/usr/bin/env python3
# Copyright (c) 2026 Leonardo Vanni. CC BY-NC 4.0 (see LICENSE.txt).
"""Render the predicted pose uncertainty inside the reconstructed scene (thesis figures).

Builds the same scene as demo_viser.py (depth-based coloured point cloud, camera frustums
with their images, camera 0 in black) and renders it headlessly through viser and a
headless Chromium, from one viewpoint for all panels of a scene:

  NAME_scene_translation.png  95% ellipsoids of the camera centres (VGGT)
  NAME_scene_rotation.png     95% cones swept by the optical axes (VGGT)
  NAME_scene_colmap.png       95% ellipsoids of COLMAP bundle adjustment (with --colmap)

Ellipsoids are drawn at true size unless they would be invisible (then magnified); if they
would engulf the scene they are not drawn and the panel says so.
The magnification is printed on every panel.  CHROME must point at a headless Chromium
(e.g. the one installed by ``playwright install chromium-headless-shell``).

    CHROME=/path/to/chrome-headless-shell python scripts/render_scene_uncertainty.py \\
        --images examples/kitchen/images --name kitchen --head vggt_uncertainty_head_v1.pt
"""
import argparse
import json
import math
import os
import socket
import subprocess
import time

import numpy as np
import torch
import viser
import viser.transforms as viser_tf
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import chi2

from vggt.utils.checkpoint import load_vggt_with_uncertainty
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.uncertainty import camera_centers, pose_covariances_from_predictions, rotation_cone_points

VGGT_RGB, COLMAP_RGB = (42, 120, 214), (235, 104, 52)  # validated categorical pair
CONFIDENCE = 0.95

# Per-scene view. "rig" frames the cameras and their uncertainty with the scene behind them
# (forward-facing captures, static cameras); "scene" frames cameras and scene together.
# azimuth / elevation (degrees) rotate the automatic view, distance scales it.
VIEWS = {
    "flower": {"mode": "rig"}, "fern": {"mode": "rig"}, "P04_11": {"mode": "rig"}, "room": {"mode": "rig"},
    "kitchen": {"mode": "scene"}, "pyramid": {"mode": "scene"}, "P01_09": {"mode": "scene"},
}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def unit_sphere(n=28):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n // 2 + 1))
    V = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], -1).reshape(-1, 3)
    F = [[i * n + j, (i + 1) * n + j, i * n + j + 1] for i in range(n // 2) for j in range(n - 1)]
    F += [[i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1] for i in range(n // 2) for j in range(n - 1)]
    return V, np.array(F, dtype=np.uint32)


SPHERE = unit_sphere()


def ellipsoid_mesh(center, cov3, magnify):
    w, V = np.linalg.eigh(cov3)
    radii = magnify * np.sqrt(np.maximum(w, 0) * chi2.ppf(CONFIDENCE, 3))
    return ((SPHERE[0] * radii) @ V.T + center).astype(np.float32), SPHERE[1]


def cone_mesh(apex, rim):
    verts = np.vstack([apex[None], rim]).astype(np.float32)
    n = len(rim)
    faces = [[0, 1 + k, 1 + (k + 1) % n] for k in range(n)]
    return verts, np.array(faces, dtype=np.uint32)


def spread_subset(centres, candidates, min_dist):
    """Greedy subset of cameras at least min_dist apart, so that drawn ellipsoids do not merge."""
    chosen = []
    for i in sorted(candidates, key=lambda i: -np.linalg.norm(centres[i] - centres[0])):  # far cameras first
        if all(np.linalg.norm(centres[i] - centres[j]) >= min_dist for j in chosen):
            chosen.append(i)
    return sorted(chosen)


def major_radius(cov3):
    return math.sqrt(max(np.linalg.eigvalsh(cov3)[-1], 0) * chi2.ppf(CONFIDENCE, 3))


def infer(images_dir, head, base, device, conf_percentile):
    names = sorted(os.listdir(images_dir))
    model, meta = load_vggt_with_uncertainty(head, base=base, device=device)
    images = load_and_preprocess_images([os.path.join(images_dir, n) for n in names]).to(device)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        pred = model(images)
    E, K = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    E, K = E[0].float().cpu().numpy(), K[0].float().cpu().numpy()
    depth = pred["depth"][0].float().cpu().numpy()
    conf = pred["depth_conf"][0].float().cpu().numpy().reshape(-1)
    pts = unproject_depth_map_to_point_map(depth, E, K).reshape(-1, 3)
    imgs = images.float().cpu().numpy()
    rgb = (imgs.transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
    keep = (conf >= np.percentile(conf, conf_percentile)) & np.isfinite(pts).all(1)
    chol = pred["cholesky_vector"][0].float().cpu()
    kw = dict(kappa=meta["kappa"], temperature=meta["temperature"])
    return {"names": names, "E": E, "points": pts[keep], "colors": rgb[keep], "images": imgs,
            "world": pose_covariances_from_predictions(chol, E, frame="world", **kw),
            "body": pose_covariances_from_predictions(chol, E, frame="body", **kw)}


def auto_view(centres, E, points, tweak, extra=None):
    """Look at the cameras and the scene from behind and above the rig."""
    up = -np.mean(E[:, 1, :3], axis=0)  # camera -y axes, in world coordinates
    up /= np.linalg.norm(up)
    look = 0.5 * (np.median(points, axis=0) + centres.mean(0))
    fwd = np.mean(E[:, 2, :3], axis=0)  # optical axes
    fwd = fwd - (fwd @ up) * up
    if np.linalg.norm(fwd) < 0.3:  # cameras around the scene: look from above, slightly tilted
        fwd = E[0, 2, :3] - (E[0, 2, :3] @ up) * up
        elev = 60.0
    else:
        elev = 30.0
    fwd /= np.linalg.norm(fwd)
    az, el, dist_f = tweak.get("azimuth", 0.0), elev + tweak.get("elevation", 0.0), tweak.get("distance", 1.0)
    side = np.cross(fwd, up)
    a = math.radians(az)
    fwd = math.cos(a) * fwd + math.sin(a) * side
    e = math.radians(el)
    direction = math.cos(e) * fwd - math.sin(e) * up  # from the eye towards the target
    cam_span = np.vstack([centres] + ([extra] if extra is not None else []))
    scene_span = np.percentile(points, [5, 95], axis=0)
    rig = tweak.get("mode") == "rig" if "mode" in tweak else np.ptp(centres, axis=0).max() < 0.3 * np.ptp(scene_span, axis=0).max()
    if rig:
        # rig view: the cameras cover a small part of the scene (forward-facing capture or a
        # static camera), so frame the cameras and their uncertainty, with the scene behind
        look = cam_span.mean(0) + 0.15 * (np.median(points, axis=0) - cam_span.mean(0))
        extent = np.ptp(cam_span, axis=0).max()
        distance = dist_f * 2.2 * extent / (2 * math.tan(math.radians(22.5)))
    else:
        extent = np.ptp(np.vstack([cam_span, scene_span]), axis=0).max()
        distance = dist_f * 1.4 * extent / (2 * math.tan(math.radians(22.5)))
    return look - direction * distance, look, up


def finish(img, text, box=None, pad=24):
    """Crop to the rendered content (or to box) and put the label in a header band above it."""
    if box is None:
        ys, xs = np.nonzero((img < 245).any(-1))
        box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, img.shape[1]), min(ys.max() + pad, img.shape[0]))
    im = Image.fromarray(img).crop(box)
    size = max(26, im.width // 32)  # the label keeps the same size relative to the panel
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    header = int(1.8 * size)
    text_w = ImageDraw.Draw(im).textbbox((0, 0), text, font=font)[2] + 20
    out = Image.new("RGB", (max(im.width, text_w), im.height + header), "white")
    out.paste(im, ((out.width - im.width) // 2, header))
    ImageDraw.Draw(out).text((10, int(0.35 * size)), text, fill=(30, 30, 30), font=font)
    return out, box


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--colmap", default=None, help="output of colmap_pose_covariance.py for the same images")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--conf_percentile", type=float, default=50.0, help="drop this percentage of low-confidence points")
    ap.add_argument("--width", type=int, default=1800)
    ap.add_argument("--height", type=int, default=1300)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    chrome = os.environ.get("CHROME")
    if not chrome:
        raise SystemExit("set CHROME to a headless Chromium binary")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    v = infer(args.images, args.head, args.base, device, args.conf_percentile)
    E, S = v["E"], len(v["names"])
    centres = camera_centers(E)
    cams = np.arange(S) > 0
    cam_spread = max(np.ptp(centres, axis=0).max(), 1e-6)
    scene_extent = np.ptp(np.percentile(v["points"], [5, 95], axis=0), axis=0).max()
    nn = np.sort(np.linalg.norm(centres[:, None] - centres[None], axis=-1), axis=1)[:, 1]
    frustum_scale = float(min(0.04 * max(cam_spread, scene_extent), 0.4 * np.median(nn) if S > 1 else 1))

    med = np.median([major_radius(v["world"][i, :3, :3]) for i in np.flatnonzero(cams)])
    view_extent = np.ptp(np.vstack([centres, np.percentile(v["points"], [5, 95], axis=0)]), axis=0).max()  # cameras and scene
    ratio = med / view_extent
    if ratio > 0.25:
        magnify = None  # the ellipsoids would engulf the scene: not drawn
    elif ratio < 0.01:
        magnify = 0.03 * view_extent / med  # invisible at true size: magnified, and the factor is printed
    else:
        magnify = 1.0
    half = [math.degrees(math.sqrt(max(np.linalg.eigvalsh(v["body"][i, 3:, 3:])[-1], 0) * chi2.ppf(CONFIDENCE, 2))) for i in np.flatnonzero(cams)]
    cone_magnify = max(1.0, 8.0 / np.median(half))
    cone_length = max(4.0 * frustum_scale, 0.15 * cam_spread, 0.03 * view_extent)
    shown = list(np.flatnonzero(cams))
    if magnify is not None and S > 2 and np.median(nn) < 0.5 * magnify * med:  # dense rig: ellipsoids would merge
        shown = spread_subset(centres, shown, 1.2 * magnify * med)

    cm = None
    if args.colmap:
        cm = dict(np.load(args.colmap))
        assert list(cm["names"]) == v["names"], "COLMAP and VGGT image lists differ"
        both = np.flatnonzero(cm["registered"] & cams)
        k = np.median(np.linalg.norm(centres[both], axis=1) / np.linalg.norm(cm["centres"][both], axis=1))
        cm["centres"], cm["cov_centre"] = k * cm["centres"], k**2 * cm["cov_centre"]
        med_c = np.median([major_radius(cm["cov_centre"][i]) for i in both])
        base_mag = magnify if magnify is not None else 0.03 * view_extent / med
        magnify_cm = base_mag * med / med_c

    port = free_port()
    server = viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
    server.scene.set_up_direction("-y")
    server.scene.add_point_cloud("points", points=v["points"].astype(np.float32), colors=v["colors"],
                                 point_size=0.0035 * scene_extent, point_shape="circle")
    cam_to_world = closed_form_inverse_se3(E)
    frustums = []
    for i in range(S):
        T = viser_tf.SE3.from_matrix(cam_to_world[i, :3, :])
        img = (v["images"][i].transpose(1, 2, 0) * 255).astype(np.uint8)
        h, w = img.shape[:2]
        frustums.append(server.scene.add_camera_frustum(
            f"/cam_{i}", fov=2 * np.arctan2(h / 2, 1.1 * h), aspect=w / h, scale=frustum_scale,
            line_width=4.0 if i == 0 else 2.0, color=(0, 0, 0) if i == 0 else (90, 90, 90), image=img,
            wxyz=T.rotation().wxyz, position=T.translation()))

    layers = {"translation": [], "rotation": [], "colmap": []}
    if magnify is not None:
        for i in shown:
            V_, F_ = ellipsoid_mesh(centres[i], v["world"][i, :3, :3], magnify)
            layers["translation"].append(server.scene.add_mesh_simple(f"/ell_{i}", V_, F_, color=VGGT_RGB, opacity=0.5, visible=False))
    for i in np.flatnonzero(cams):
        body = v["body"][i].copy()
        body[3:, 3:] *= cone_magnify**2
        apex, rim = rotation_cone_points(centres[i], E[i], body, confidence=CONFIDENCE, length=cone_length, num_points=40)
        V_, F_ = cone_mesh(apex, rim)
        layers["rotation"].append(server.scene.add_mesh_simple(f"/cone_{i}", V_, F_, color=VGGT_RGB, opacity=0.65, visible=False))
    if cm is not None:
        for i in [j for j in shown if cm["registered"][j]]:
            V_, F_ = ellipsoid_mesh(cm["centres"][i], cm["cov_centre"][i], magnify_cm)
            layers["colmap"].append(server.scene.add_mesh_simple(f"/cm_ell_{i}", V_, F_, color=COLMAP_RGB, opacity=0.45, visible=False))
        layers["colmap"].append(server.scene.add_point_cloud("/cm_centres", points=cm["centres"][cm["registered"]].astype(np.float32),
                                                             colors=(0, 0, 0), point_size=0.012 * cam_spread, point_shape="circle", visible=False))

    proc = subprocess.Popen([chrome, "--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                             f"--window-size={args.width},{args.height}", f"http://127.0.0.1:{port}"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while not server.get_clients():
            if time.time() - t0 > 90:
                raise SystemExit("headless browser did not connect")
            time.sleep(0.5)
        client = next(iter(server.get_clients().values()))
        # frame the drawn ellipsoids of both methods too, so that nothing is cut at the border
        extra = []
        if magnify is not None:
            extra += [centres[i] + s * magnify * major_radius(v["world"][i, :3, :3]) for i in np.flatnonzero(cams) for s in (-1, 1)]
        if cm is not None and VIEWS.get(args.name, {}).get("mode") != "rig":  # rig views frame VGGT's rig only
            extra += [cm["centres"][i] + s * magnify_cm * major_radius(cm["cov_centre"][i]) for i in np.flatnonzero(cm["registered"] & cams) for s in (-1, 1)]
        pos, look, up = auto_view(centres, E, v["points"], VIEWS.get(args.name, {}), np.array(extra) if extra else None)
        client.camera.up_direction = tuple(up)
        client.camera.fov = math.radians(45)
        client.camera.position = tuple(pos)
        client.camera.look_at = tuple(look)
        time.sleep(3)

        def times(m):
            return "true size" if m == 1 else (f"x{m:.0f}" if m >= 10 else f"x{m:.2g}")

        subset = "" if len(shown) == int(cams.sum()) else f" ({len(shown)} of {int(cams.sum())} cameras)"
        texts = {
            "translation": ("VGGT: 95% camera-centre ellipsoids, " + times(magnify) + subset) if magnify is not None
            else "VGGT: ellipsoids larger than the scene, not drawn",
            "rotation": "VGGT: 95% optical-axis cones, " + times(cone_magnify),
            "colmap": "COLMAP bundle adjustment: 95% ellipsoids, " + (times(magnify_cm) + subset if cm is not None else ""),
        }
        written, renders = {}, {}
        for panel in ("translation", "rotation", "colmap"):
            if panel == "colmap" and cm is None:
                continue
            for name, handles in layers.items():
                for h in handles:
                    h.visible = name == panel
            for f in frustums:  # COLMAP panel: VGGT cameras would mislead, keep only the scene
                f.visible = panel != "colmap"
            time.sleep(2.5)
            renders[panel] = client.get_render(height=args.height, width=args.width)
        # one crop box for all panels of the scene, so they stay directly comparable
        ys, xs = np.nonzero(np.stack([(r < 245).any(-1) for r in renders.values()]).any(0))
        pad = 24
        box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, args.width), min(ys.max() + pad, args.height))
        for panel, img in renders.items():
            path = os.path.join(args.out_dir, f"{args.name}_scene_{panel}.png")
            finish(img, texts[panel], box)[0].save(path)
            written[panel] = path
        meta = {"ellipsoid_cameras": [int(i) for i in shown], "magnify": magnify, "cone_magnify": cone_magnify, "magnify_colmap": magnify_cm if cm is not None else None,
                "median_major_radius_over_view_extent": ratio, "view": {"position": list(map(float, pos)), "look_at": list(map(float, look))}}
        with open(os.path.join(args.out_dir, f"{args.name}_scene.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(json.dumps({"written": written, **meta}, indent=2))
    finally:
        proc.terminate()
        server.stop()


if __name__ == "__main__":
    main()
    # viser's event loop and the browser occasionally abort the interpreter at teardown;
    # everything is written at this point, so leave without running it
    os._exit(0)
