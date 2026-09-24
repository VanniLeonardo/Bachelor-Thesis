# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.checkpoint import load_vggt_with_uncertainty
from vggt.utils.uncertainty import camera_centers, pose_covariances_from_predictions, rotation_std_deg, chi2_quantile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

device = "cuda" if torch.cuda.is_available() else "cpu"

# Upstream VGGT-1B weights plus the fine-tuned pose-uncertainty head.  Point
# VGGT_UNCERTAINTY_CKPT at a head export (see vggt/utils/checkpoint.py) or a trainer checkpoint.
UNCERTAINTY_CKPT = os.environ.get("VGGT_UNCERTAINTY_CKPT")
print("Initializing and loading VGGT model...")
model, HEAD_META = load_vggt_with_uncertainty(UNCERTAINTY_CKPT, device=device)
if UNCERTAINTY_CKPT is None:
    print("WARNING: VGGT_UNCERTAINTY_CKPT is not set; the covariance head is randomly initialised.")
KAPPA = HEAD_META["kappa"]
TEMPERATURE = HEAD_META["temperature"]
print(f"Uncertainty head: kappa={KAPPA}, temperature={TEMPERATURE}, convention={HEAD_META['error_convention']}")


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].float().cpu().numpy().squeeze(0)  # remove batch dimension

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# Uncertainty visualization (matplotlib summaries shown in the Gradio gallery)
# -------------------------------------------------------------------------
def _ellipse_2d(center, cov2, chi2_val, n=100):
    eigval, eigvec = np.linalg.eigh(cov2)
    radii = np.sqrt(np.maximum(eigval, 0.0) * chi2_val)
    t = np.linspace(0, 2 * np.pi, n)
    return (np.stack([np.cos(t), np.sin(t)], -1) * radii) @ eigvec.T + center


def plot_uncertainty_ellipses(predictions, target_dir, confidence=0.95):
    """Camera centres with their 95% translational uncertainty ellipsoids (world frame, true scale)."""
    if predictions.get("cholesky_vector") is None:
        return None
    extrinsics = predictions["extrinsic"]  # (S, 3, 4) camera-from-world
    Sigma = pose_covariances_from_predictions(predictions["cholesky_vector"], extrinsics, kappa=KAPPA, temperature=TEMPERATURE, frame="world")
    centers = camera_centers(extrinsics)
    q3, q2 = chi2_quantile(3, confidence), chi2_quantile(2, confidence)

    fig = plt.figure(figsize=(14, 10))
    ax3d = fig.add_subplot(221, projection="3d")
    ax3d.set_title(f"Camera centres with {int(confidence*100)}% translational uncertainty")
    axes2d = {(0, 1): fig.add_subplot(222), (0, 2): fig.add_subplot(223), (1, 2): fig.add_subplot(224)}
    names = "XYZ"
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 24), np.linspace(0, np.pi, 12))
    sphere = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], -1)
    for i, (c, S6) in enumerate(zip(centers, Sigma)):
        cov = S6[:3, :3]
        color = f"C{i % 10}"
        ax3d.scatter(*c, c=color, s=30)
        eigval, eigvec = np.linalg.eigh(cov)
        pts = (sphere * np.sqrt(np.maximum(eigval, 0) * q3)) @ eigvec.T + c
        ax3d.plot_surface(pts[..., 0], pts[..., 1], pts[..., 2], alpha=0.25, color=color)
        for (a, b), ax in axes2d.items():
            ax.scatter(c[a], c[b], c=color, s=30)
            e = _ellipse_2d(c[[a, b]], cov[np.ix_([a, b], [a, b])], q2)
            ax.plot(e[:, 0], e[:, 1], c=color, alpha=0.7)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    for (a, b), ax in axes2d.items():
        ax.set_xlabel(names[a]); ax.set_ylabel(names[b]); ax.set_aspect("equal", adjustable="datalim"); ax.grid(True, alpha=0.3)
        ax.set_title(f"{names[a]}{names[b]} projection ({int(confidence*100)}% ellipses)")
    fig.suptitle("Translational pose uncertainty (normalized scene units)")
    plt.tight_layout()
    plot_path = os.path.join(target_dir, "uncertainty_visualization.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_rotation_uncertainty(predictions, target_dir):
    """Per-camera rotational standard deviations in degrees (body frame, kappa undone)."""
    if predictions.get("cholesky_vector") is None:
        return None
    extrinsics = predictions["extrinsic"]
    Sigma_body = pose_covariances_from_predictions(predictions["cholesky_vector"], extrinsics, kappa=KAPPA, temperature=TEMPERATURE, frame="body")
    rot_std = rotation_std_deg(Sigma_body)  # (S, 3) degrees about the camera's x, y, z axes
    trans_std = np.sqrt(np.maximum(np.diagonal(Sigma_body[:, :3, :3], axis1=-2, axis2=-1), 0))
    centers = camera_centers(extrinsics)
    dist = np.linalg.norm(centers - centers[0], axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Camera pose uncertainty per frame")
    idx = np.arange(len(rot_std))
    for k, (lab, col) in enumerate(zip(["about x (pitch)", "about y (yaw)", "about z (roll)"], "rgb")):
        axes[0, 0].plot(idx, rot_std[:, k], f"{col}-o", label=lab)
    axes[0, 0].set_xlabel("Camera index"); axes[0, 0].set_ylabel("Rotation std (deg)"); axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 0].set_title("Rotational uncertainty")
    for k, (lab, col) in enumerate(zip(["x (right)", "y (down)", "z (forward)"], "rgb")):
        axes[0, 1].plot(idx, trans_std[:, k], f"{col}-o", label=lab)
    axes[0, 1].set_xlabel("Camera index"); axes[0, 1].set_ylabel("Translation std (scene units)"); axes[0, 1].legend(); axes[0, 1].grid(True)
    axes[0, 1].set_title("Translational uncertainty (camera frame)")
    axes[1, 0].plot(idx, np.linalg.norm(rot_std, axis=1), "purple", marker="o")
    axes[1, 0].set_xlabel("Camera index"); axes[1, 0].set_ylabel("Total rotation std (deg)"); axes[1, 0].grid(True)
    axes[1, 0].set_title("Total rotational uncertainty")
    axes[1, 1].scatter(dist, np.linalg.norm(trans_std, axis=1), alpha=0.8)
    axes[1, 1].set_xlabel("Distance from first camera"); axes[1, 1].set_ylabel("Total translation std"); axes[1, 1].grid(True)
    axes[1, 1].set_title("Translational uncertainty vs distance from camera 0")
    plt.tight_layout()
    plot_path = os.path.join(target_dir, "rotation_uncertainty_visualization.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, []

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
        show_uncertainty_ellipses=predictions.get("cholesky_vector") is not None,
        kappa=KAPPA,
        temperature=TEMPERATURE,
    )
    glbscene.export(file_obj=glbfile)

    # Generate uncertainty visualizations if available
    uncertainty_plots = []
    if 'cholesky_vector' in predictions and predictions['cholesky_vector'] is not None:
        try:
            print("Generating uncertainty visualizations...")
            print(f"Cholesky vector shape: {predictions['cholesky_vector'].shape}")
            
            # Create translation uncertainty plot
            trans_plot = plot_uncertainty_ellipses(predictions, target_dir)
            if trans_plot:
                uncertainty_plots.append(trans_plot)
                print(f"Generated translation uncertainty plot: {trans_plot}")
            
            # Create rotation uncertainty plot
            rot_plot = plot_rotation_uncertainty(predictions, target_dir)
            if rot_plot:
                uncertainty_plots.append(rot_plot)
                print(f"Generated rotation uncertainty plot: {rot_plot}")
                
            print(f"Generated {len(uncertainty_plots)} uncertainty plots")
        except Exception as e:
            print(f"Warning: Could not generate uncertainty plots: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No uncertainty information available for visualization")

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    
    if uncertainty_plots:
        log_msg = f"Reconstruction Success ({len(all_files)} frames). Generated uncertainty visualizations. Waiting for visualization."
    else:
        log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True), uncertainty_plots


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer. If is_example == "True", skip.
    """

    # If it's an example click, skip as requested
    if is_example == "True":
        return None, "No reconstruction available. Please click the Reconstruct button first."

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "intrinsic",
        "world_points_from_depth",
        "cholesky_vector",
    ]

    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) if key in loaded else None for key in key_list}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    # Generate uncertainty visualizations if available
    uncertainty_plots = []
    chol_vector = predictions.get('cholesky_vector')
    if chol_vector is not None:
        try:
            print(f"update_visualization: Found cholesky vectors with shape {chol_vector.shape}")
            # Check for existing uncertainty plots
            trans_plot_path = os.path.join(target_dir, "uncertainty_visualization.png")
            rot_plot_path = os.path.join(target_dir, "rotation_uncertainty_visualization.png")
            
            if os.path.exists(trans_plot_path):
                uncertainty_plots.append(trans_plot_path)
                print(f"Found existing translation plot: {trans_plot_path}")
            if os.path.exists(rot_plot_path):
                uncertainty_plots.append(rot_plot_path)
                print(f"Found existing rotation plot: {rot_plot_path}")
                
            # If plots don't exist, generate them
            if not uncertainty_plots:
                print("Generating new uncertainty plots...")
                trans_plot = plot_uncertainty_ellipses(predictions, target_dir)
                if trans_plot:
                    uncertainty_plots.append(trans_plot)
                    print(f"Generated new translation plot: {trans_plot}")
                
                rot_plot = plot_rotation_uncertainty(predictions, target_dir)
                if rot_plot:
                    uncertainty_plots.append(rot_plot)
                    print(f"Generated new rotation plot: {rot_plot}")
            
            print(f"update_visualization: Total uncertainty plots: {len(uncertainty_plots)}")
        except Exception as e:
            print(f"Warning: Could not generate uncertainty plots in update_visualization: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("update_visualization: No cholesky_vector available")

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
            show_uncertainty_ellipses=predictions.get("cholesky_vector") is not None,
            kappa=KAPPA,
            temperature=TEMPERATURE,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization", uncertainty_plots


# -------------------------------------------------------------------------
# Example images
# -------------------------------------------------------------------------

great_wall_video = "examples/videos/great_wall.mp4"
colosseum_video = "examples/videos/Colosseum.mp4"
room_video = "examples/videos/room.mp4"
kitchen_video = "examples/videos/kitchen.mp4"
fern_video = "examples/videos/fern.mp4"
single_cartoon_video = "examples/videos/single_cartoon.mp4"
single_oil_painting_video = "examples/videos/single_oil_painting.mp4"
pyramid_video = "examples/videos/pyramid.mp4"


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    
    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }
    
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """,
) as demo:
    # Instead of gr.State, we use a hidden Textbox:
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")

    gr.HTML(
        """
    <h1>🏛️ VGGT: Visual Geometry Grounded Transformer</h1>
    <p>
    <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a> |
    <a href="#">Project Page</a>
    </p>

    <div style="font-size: 16px; line-height: 1.5;">
    <p>Upload a video or a set of images to create a 3D reconstruction of a scene or object. VGGT takes these images and generates a 3D point cloud, along with estimated camera poses.</p>

    <h3>Getting Started:</h3>
    <ol>
        <li><strong>Upload Your Data:</strong> Use the "Upload Video" or "Upload Images" buttons on the left to provide your input. Videos will be automatically split into individual frames (one frame per second).</li>
        <li><strong>Preview:</strong> Your uploaded images will appear in the gallery on the left.</li>
        <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction process.</li>
        <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file. Note the visualization of 3D points may be slow for a large number of input images.</li>
        <li>
        <strong>Adjust Visualization (Optional):</strong>
        After reconstruction, you can fine-tune the visualization using the options below
        <details style="display:inline;">
            <summary style="display:inline;">(<strong>click to expand</strong>):</summary>
            <ul>
            <li><em>Confidence Threshold:</em> Adjust the filtering of points based on confidence.</li>
            <li><em>Show Points from Frame:</em> Select specific frames to display in the point cloud.</li>
            <li><em>Show Camera:</em> Toggle the display of estimated camera positions.</li>
            <li><em>Filter Sky / Filter Black Background:</em> Remove sky or black-background points.</li>
            <li><em>Select a Prediction Mode:</em> Choose between "Depthmap and Camera Branch" or "Pointmap Branch."</li>
            </ul>
        </details>
        </li>
    </ol>
    <p><strong style="color: #0ea5e9;">Please note:</strong> <span style="color: #0ea5e9; font-weight: bold;">VGGT typically reconstructs a scene in less than 1 second. However, visualizing 3D points may take tens of seconds due to third-party rendering, which are independent of VGGT's processing time. </span></p>
    </div>
    """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)

            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                log_output = gr.Markdown(
                    "Please upload a video or images, then click Reconstruct.", elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)
                
                # Add uncertainty visualization section
                with gr.Accordion("Camera Pose Uncertainty Visualization", open=False):
                    gr.Markdown("**Visualizations of predicted camera pose uncertainty using covariance matrices**")
                    uncertainty_gallery = gr.Gallery(
                        label="Uncertainty Plots",
                        columns=2,
                        height="400px",
                        show_download_button=True,
                        object_fit="contain",
                        preview=True,
                    )

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                    scale=1,
                )

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch"],
                    label="Select a Prediction Mode",
                    value="Depthmap and Camera Branch",
                    scale=1,
                    elem_id="my_radio",
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    # show_uncertainty_ellipses = gr.Checkbox(label="Show Uncertainty Ellipses", value=False)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

    # ---------------------- Examples section ----------------------
    examples = [
        [colosseum_video, "22", None, 20.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [pyramid_video, "30", None, 35.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [single_cartoon_video, "1", None, 15.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [single_oil_painting_video, "1", None, 20.0, False, False, True, True, "Depthmap and Camera Branch", "True"],
        [room_video, "8", None, 5.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [kitchen_video, "25", None, 50.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [fern_video, "20", None, 45.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
    ]

    def example_pipeline(
        input_video,
        num_images_str,
        input_images,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        prediction_mode,
        is_example_str,
    ):
        """
        1) Copy example images to new target_dir
        2) Reconstruct
        3) Return model3D + logs + new_dir + updated dropdown + gallery + uncertainty plots
        We do NOT return is_example. It's just an input.
        """
        target_dir, image_paths = handle_uploads(input_video, input_images)
        # Always use "All" for frame_filter in examples
        frame_filter = "All"
        glbfile, log_msg, dropdown, uncertainty_plots = gradio_demo(
            target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
        )
        return glbfile, log_msg, target_dir, dropdown, image_paths, uncertainty_plots

    gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])
    gr.Examples(
        examples=examples,
        inputs=[
            input_video,
            num_images,
            input_images,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        outputs=[reconstruction_output, log_output, target_dir_output, frame_filter, image_gallery, uncertainty_gallery],
        fn=example_pipeline,
        cache_examples=False,
        examples_per_page=50,
    )

    # -------------------------------------------------------------------------
    # "Reconstruct" button logic:
    #  - Clear fields
    #  - Update log
    #  - gradio_demo(...) with the existing target_dir
    #  - Then set is_example = "False"
    # -------------------------------------------------------------------------
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        outputs=[reconstruction_output, log_output, frame_filter, uncertainty_gallery],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]  # set is_example to "False"
    )

    # -------------------------------------------------------------------------
    # Real-time Visualization Updates
    # -------------------------------------------------------------------------
    conf_thres.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output, uncertainty_gallery],
    )
    frame_filter.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output, uncertainty_gallery],
    )
    mask_black_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output, uncertainty_gallery],
    )
    mask_white_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output, uncertainty_gallery],
    )
    show_cam.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output, uncertainty_gallery],
    )
    mask_sky.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output, uncertainty_gallery],
    )
    prediction_mode.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output, uncertainty_gallery],
    )
    # show_uncertainty_ellipses.change(
    #     update_visualization,
    #     [
    #         target_dir_output,
    #         conf_thres,
    #         frame_filter,
    #         mask_black_bg,
    #         mask_white_bg,
    #         show_cam,
    #         mask_sky,
    #         prediction_mode,
    #         is_example,
    #         show_uncertainty_ellipses,
    #     ],
    #     [reconstruction_output, log_output, uncertainty_gallery],
    # )

    # -------------------------------------------------------------------------
    # Auto-update gallery whenever user uploads or changes their files
    # -------------------------------------------------------------------------
    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )

    demo.queue(max_size=20).launch(show_error=True, share=True)
