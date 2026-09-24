# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Pose-uncertainty visualization added by Leonardo Vanni (2025-2026), same license.

"""Interactive viser demo: VGGT reconstruction with camera-pose uncertainty.

    python demo_viser.py --image_folder examples/kitchen/images --uncertainty_ckpt PATH

Translational uncertainty is drawn as the 95% ellipsoid of the camera centre (world
frame), rotational uncertainty as the cone swept by the optical axis.  Both use the
covariance in physical units (normalized scene units, radians) after undoing the
training-time kappa weighting; the "sigma multiplier" slider only rescales the drawing.
"""

import argparse
import glob
import os
import threading
import time
from typing import List, Optional

import cv2
import numpy as np
import torch
import viser
import viser.transforms as viser_tf
from tqdm.auto import tqdm

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")
    onnxruntime = None

from vggt.utils.checkpoint import load_vggt_with_uncertainty
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map, unproject_depth_map_with_uncertainty
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.uncertainty import (
    camera_centers,
    pose_covariances_from_predictions,
    rotation_cone_half_angles,
    rotation_cone_points,
    rotation_std_deg,
    translation_ellipsoid_points,
)
from visual_util import download_file_from_url, segment_sky


def add_camera_uncertainty(server, pred_dict, scene_center, sigma_multiplier, show_trans, show_rot, confidence=0.95, cone_length=0.1):
    """Draw translational ellipsoids and rotational cones; returns the created handles."""
    handles = []
    if "cholesky_vector" not in pred_dict:
        return handles
    E = pred_dict["extrinsic"]
    Sigma_world = pred_dict["pose_cov_world"] * sigma_multiplier**2
    Sigma_body = pred_dict["pose_cov_body"] * sigma_multiplier**2
    centers = camera_centers(E) - scene_center
    for i in range(len(E)):
        if show_trans:
            pts = translation_ellipsoid_points(centers[i], Sigma_world[i, :3, :3], confidence)
            handles.append(server.scene.add_point_cloud(
                name=f"/uncertainty/ellipsoid_{i}", points=pts.astype(np.float32),
                colors=np.full((len(pts), 3), [255, 100, 100], dtype=np.uint8), point_size=0.002, point_shape="circle",
            ))
        if show_rot:
            apex, rim = rotation_cone_points(centers[i], E[i], Sigma_body[i], confidence, length=cone_length)
            segments = np.stack([np.repeat(apex[None], len(rim), 0), rim], axis=1)  # apex -> rim
            ring = np.stack([rim, np.roll(rim, -1, axis=0)], axis=1)
            handles.append(server.scene.add_line_segments(
                name=f"/uncertainty/cone_{i}", points=np.concatenate([segments, ring]).astype(np.float32),
                colors=(255, 165, 0), line_width=1.5,
            ))
    return handles


def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: Optional[str] = None,
):
    """Visualize predicted 3D points, camera poses and their uncertainty with viser."""
    print(f"Starting viser server on port {port}")
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    images = pred_dict["images"]
    extrinsics_cam = pred_dict["extrinsic"]
    intrinsics_cam = pred_dict["intrinsic"]

    points_cov_flat = None
    if use_point_map:
        world_points = pred_dict["world_points"]
        conf = pred_dict["world_points_conf"]
    elif "world_points_from_depth_cov" in pred_dict:
        world_points = pred_dict["world_points_from_depth"]
        points_cov_flat = pred_dict["world_points_from_depth_cov"].reshape(-1, 3, 3)
        conf = pred_dict["depth_conf"]
    else:
        world_points = unproject_depth_map_to_point_map(pred_dict["depth"], extrinsics_cam, intrinsics_cam)
        conf = pred_dict["depth_conf"]

    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    colors = images.transpose(0, 2, 3, 1)
    S, H, W, _ = world_points.shape
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world = closed_form_inverse_se3(extrinsics_cam)[:, :3, :]
    scene_center = np.nanmean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center
    frame_indices = np.repeat(np.arange(S), H * W)
    point_cloud_mask = np.zeros_like(conf_flat, dtype=bool)

    has_uncertainty = "cholesky_vector" in pred_dict
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)
    with server.gui.add_folder("Camera Pose Uncertainty (95%)"):
        gui_show_trans = server.gui.add_checkbox("Translation ellipsoids", initial_value=has_uncertainty)
        gui_show_rot = server.gui.add_checkbox("Rotation cones", initial_value=False)
        gui_sigma_mult = server.gui.add_slider("Sigma multiplier (display only)", min=0.1, max=20.0, step=0.1, initial_value=1.0)
        gui_cone_length = server.gui.add_slider("Cone length", min=0.02, max=1.0, step=0.01, initial_value=0.1)
    with server.gui.add_folder("3D Point Uncertainty"):
        gui_show_point_unc = server.gui.add_checkbox("Show point ellipsoids", initial_value=False)
        gui_point_stride = server.gui.add_slider("Display stride (1-in-N)", min=1, max=100000, step=1, initial_value=500)
    gui_points_conf = server.gui.add_slider("Confidence Percentile", min=0, max=100, step=0.1, initial_value=init_conf_threshold)
    gui_frame_selector = server.gui.add_dropdown("Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All")

    point_cloud = server.scene.add_point_cloud(name="viser_pcd", points=np.zeros((1, 3)), colors=np.zeros((1, 3)), point_size=0.001)
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []
    cam_unc_handles: List = []
    point_unc_handles: List = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in range(S):
            T_world_camera = viser_tf.SE3.from_matrix(extrinsics[img_id])
            frame_axis = server.scene.add_frame(
                f"/frame_{img_id}", wxyz=T_world_camera.rotation().wxyz, position=T_world_camera.translation(),
                axes_length=0.05, axes_radius=0.002,
            )
            frames.append(frame_axis)
            img = (images_[img_id].transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]
            fov = 2 * np.arctan2(h / 2, 1.1 * h)
            frustum_cam = server.scene.add_camera_frustum(f"/frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img)
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_uncertainty() -> None:
        for h in point_unc_handles:
            h.remove()
        point_unc_handles.clear()
        if not gui_show_point_unc.value or points_cov_flat is None:
            return
        stride = int(gui_point_stride.value)
        mult = gui_sigma_mult.value
        pts = points_centered[point_cloud_mask][::stride]
        covs = points_cov_flat[point_cloud_mask][::stride] * mult**2
        print(f"Drawing uncertainty ellipsoids for {len(pts)} points (1 in {stride}).")
        for i, (p, c) in enumerate(zip(pts, covs)):
            ell = translation_ellipsoid_points(p, c, num_points=64)
            point_unc_handles.append(server.scene.add_point_cloud(
                name=f"/point_uncertainty/ellipsoid_{i}", points=ell.astype(np.float32),
                colors=np.full((len(ell), 3), [255, 100, 100], dtype=np.uint8), point_size=0.001,
            ))

    def update_point_cloud() -> None:
        nonlocal point_cloud_mask
        threshold_val = np.percentile(conf_flat, gui_points_conf.value)
        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)
        frame_mask = np.ones_like(conf_mask) if gui_frame_selector.value == "All" else frame_indices == int(gui_frame_selector.value)
        point_cloud_mask = conf_mask & frame_mask
        if np.any(point_cloud_mask):
            point_cloud.points = points_centered[point_cloud_mask]
            point_cloud.colors = colors_flat[point_cloud_mask]
        else:
            point_cloud.points = np.zeros((1, 3))
            point_cloud.colors = np.zeros((1, 3))
        update_point_uncertainty()

    def update_camera_uncertainty() -> None:
        for h in cam_unc_handles:
            h.remove()
        cam_unc_handles.clear()
        cam_unc_handles.extend(add_camera_uncertainty(
            server, pred_dict, scene_center, gui_sigma_mult.value, gui_show_trans.value, gui_show_rot.value,
            cone_length=gui_cone_length.value,
        ))

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    for g in (gui_show_point_unc, gui_point_stride):
        @g.on_update
        def _(_) -> None:
            update_point_uncertainty()

    for g in (gui_show_trans, gui_show_rot, gui_cone_length):
        @g.on_update
        def _(_) -> None:
            update_camera_uncertainty()

    @gui_sigma_mult.on_update
    def _(_) -> None:
        update_camera_uncertainty()
        update_point_uncertainty()

    visualize_frames(cam_to_world, images)
    update_point_cloud()
    update_camera_uncertainty()

    print("Viser server running. Press Ctrl+C to stop.")
    if background_mode:
        threading.Thread(target=lambda: time.sleep(0.001), daemon=True).start()
    else:
        while True:
            time.sleep(0.01)
    return server


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """Zero the confidence of sky pixels using the upstream sky-segmentation ONNX model."""
    if onnxruntime is None:
        print("onnxruntime is not installed. Skipping sky segmentation.")
        return conf
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")
    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []
    for image_path in tqdm(image_files[:S], desc="sky masks"):
        mask_filepath = os.path.join(sky_masks_dir, os.path.basename(image_path))
        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))
        sky_mask_list.append(sky_mask)
    sky_mask_binary = (np.array(sky_mask_list) > 0.1).astype(np.float32)
    return conf * sky_mask_binary


def run_inference(args) -> dict:
    """Load the model, run VGGT on the image folder and attach pose covariances."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, meta = load_vggt_with_uncertainty(args.uncertainty_ckpt, base=args.base_ckpt, device=device)
    if args.uncertainty_ckpt is None:
        print("WARNING: no --uncertainty_ckpt given; the covariance head is randomly initialised.")
    kappa = args.kappa if args.kappa is not None else meta["kappa"]
    temperature = args.temperature if args.temperature is not None else meta["temperature"]
    print(f"Uncertainty head: kappa={kappa}, temperature={temperature}, convention={meta['error_convention']}")

    image_names = sorted(glob.glob(os.path.join(args.image_folder, "*")))
    if not image_names:
        raise ValueError(f"No images found in {args.image_folder}")
    print(f"Found {len(image_names)} images")
    images = load_and_preprocess_images(image_names).to(device)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda", dtype=dtype):
        predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    for key in list(predictions):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].float().cpu().numpy().squeeze(0)

    # Pose covariances in physical units: body frame (as trained) and world frame (for drawing).
    predictions["pose_cov_body"] = pose_covariances_from_predictions(
        predictions["cholesky_vector"], predictions["extrinsic"], kappa=kappa, temperature=temperature, frame="body")
    predictions["pose_cov_world"] = pose_covariances_from_predictions(
        predictions["cholesky_vector"], predictions["extrinsic"], kappa=kappa, temperature=temperature, frame="world")
    sig_t = np.sqrt(np.diagonal(predictions["pose_cov_body"][:, :3, :3], axis1=-2, axis2=-1))
    print("Per-camera translational sigma (scene units):", np.round(sig_t, 4).tolist())
    print("Per-camera rotational sigma (deg):", np.round(rotation_std_deg(predictions["pose_cov_body"]), 3).tolist())
    for i, S6 in enumerate(predictions["pose_cov_body"]):
        ha, _, roll = rotation_cone_half_angles(S6)
        print(f"  camera {i}: 95% optical-axis cone half-angles {np.degrees(ha).round(2).tolist()} deg, roll sigma {np.degrees(roll):.2f} deg")

    if args.propagate_point_uncertainty and not args.use_point_map:
        # First-order propagation of the body-frame pose covariance and a depth variance to
        # every 3D point (geometry.py uses body-frame Jacobians, matching the trained convention).
        print("Propagating pose uncertainty to the depth-based point cloud...")
        pts, cov = unproject_depth_map_with_uncertainty(
            depth_map=predictions["depth"], depth_conf=predictions["depth_conf"],
            extrinsics_cam=predictions["extrinsic"], intrinsics_cam=predictions["intrinsic"],
            pose_covariances=predictions["pose_cov_body"],
            depth_conf_to_variance_fn=lambda c: 1.0 / (np.asarray(c) ** 2 + 1e-6),
        )
        predictions["world_points_from_depth"] = pts
        predictions["world_points_from_depth_cov"] = cov
    return predictions


parser = argparse.ArgumentParser(description="VGGT demo with viser: 3D reconstruction and camera-pose uncertainty")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Folder containing only images")
parser.add_argument("--uncertainty_ckpt", type=str, default=os.environ.get("VGGT_UNCERTAINTY_CKPT"),
                    help="Fine-tuned uncertainty head (head export or trainer checkpoint); env VGGT_UNCERTAINTY_CKPT")
parser.add_argument("--base_ckpt", type=str, default=None, help="Upstream VGGT-1B weights (default: env VGGT_PRETRAINED_CKPT or download)")
parser.add_argument("--kappa", type=float, default=None, help="Override the rotation weight stored with the head")
parser.add_argument("--temperature", type=float, default=None, help="Override the calibration temperature stored with the head")
parser.add_argument("--use_point_map", action="store_true", help="Use the point-map branch instead of depth-based points")
parser.add_argument("--propagate_point_uncertainty", action="store_true", help="Also propagate pose uncertainty to every 3D point (slow)")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")


def main():
    args = parser.parse_args()
    predictions = run_inference(args)
    viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
    )


if __name__ == "__main__":
    main()
