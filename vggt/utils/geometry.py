# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np


from vggt.dependency.distortion import apply_distortion, iterative_undistortion, single_undistortion


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
) -> np.ndarray:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


# TODO: this code can be further cleaned up


def project_world_points_to_camera_points_batch(world_points, cam_extrinsics):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    Args:
        world_points (torch.Tensor): 3D points of shape BxSxHxWx3.
        cam_extrinsics (torch.Tensor): Extrinsic parameters of shape BxSx3x4.
    Returns:
    """
    # TODO: merge this into project_world_points_to_cam
    
    # device = world_points.device
    # with torch.autocast(device_type=device.type, enabled=False):
    ones = torch.ones_like(world_points[..., :1])  # shape: (B, S, H, W, 1)
    world_points_h = torch.cat([world_points, ones], dim=-1)  # shape: (B, S, H, W, 4)

    # extrinsics: (B, S, 3, 4) -> (B, S, 1, 1, 3, 4)
    extrinsics_exp = cam_extrinsics.unsqueeze(2).unsqueeze(3)

    # world_points_h: (B, S, H, W, 4) -> (B, S, H, W, 4, 1)
    world_points_h_exp = world_points_h.unsqueeze(-1)

    # Now perform the matrix multiplication
    # (B, S, 1, 1, 3, 4) @ (B, S, H, W, 4, 1) broadcasts to (B, S, H, W, 3, 1)
    camera_points = torch.matmul(extrinsics_exp, world_points_h_exp).squeeze(-1)

    return camera_points



def project_world_points_to_cam(
    world_points,
    cam_extrinsics,
    cam_intrinsics=None,
    distortion_params=None,
    default=0,
    only_points_cam=False,
):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    Args:
        world_points (torch.Tensor): 3D points of shape Px3.
        cam_extrinsics (torch.Tensor): Extrinsic parameters of shape Bx3x4.
        cam_intrinsics (torch.Tensor): Intrinsic parameters of shape Bx3x3.
        distortion_params (torch.Tensor): Extra parameters of shape BxN, which is used for radial distortion.
    Returns:
        torch.Tensor: Transformed 2D points of shape BxNx2.
    """
    device = world_points.device
    # with torch.autocast(device_type=device.type, dtype=torch.double):
    with torch.autocast(device_type=device.type, enabled=False):
        N = world_points.shape[0]  # Number of points
        B = cam_extrinsics.shape[0]  # Batch size, i.e., number of cameras
        world_points_homogeneous = torch.cat(
            [world_points, torch.ones_like(world_points[..., 0:1])], dim=1
        )  # Nx4
        # Reshape for batch processing
        world_points_homogeneous = world_points_homogeneous.unsqueeze(0).expand(
            B, -1, -1
        )  # BxNx4

        # Step 1: Apply extrinsic parameters
        # Transform 3D points to camera coordinate system for all cameras
        cam_points = torch.bmm(
            cam_extrinsics, world_points_homogeneous.transpose(-1, -2)
        )

        if only_points_cam:
            return None, cam_points

        # Step 2: Apply intrinsic parameters and (optional) distortion
        image_points = img_from_cam(cam_intrinsics, cam_points, distortion_params, default=default)

        return image_points, cam_points



def img_from_cam(cam_intrinsics, cam_points, distortion_params=None, default=0.0):
    """
    Applies intrinsic parameters and optional distortion to the given 3D points.

    Args:
        cam_intrinsics (torch.Tensor): Intrinsic camera parameters of shape Bx3x3.
        cam_points (torch.Tensor): 3D points in camera coordinates of shape Bx3xN.
        distortion_params (torch.Tensor, optional): Distortion parameters of shape BxN, where N can be 1, 2, or 4.
        default (float, optional): Default value to replace NaNs in the output.

    Returns:
        pixel_coords (torch.Tensor): 2D points in pixel coordinates of shape BxNx2.
    """

    # Normalized device coordinates (NDC)
    cam_points = cam_points / cam_points[:, 2:3, :]
    ndc_xy = cam_points[:, :2, :]

    # Apply distortion if distortion_params are provided
    if distortion_params is not None:
        x_distorted, y_distorted = apply_distortion(distortion_params, ndc_xy[:, 0], ndc_xy[:, 1])
        distorted_xy = torch.stack([x_distorted, y_distorted], dim=1)
    else:
        distorted_xy = ndc_xy

    # Prepare cam_points for batch matrix multiplication
    cam_coords_homo = torch.cat(
        (distorted_xy, torch.ones_like(distorted_xy[:, :1, :])), dim=1
    )  # Bx3xN
    # Apply intrinsic parameters using batch matrix multiplication
    pixel_coords = torch.bmm(cam_intrinsics, cam_coords_homo)  # Bx3xN

    # Extract x and y coordinates
    pixel_coords = pixel_coords[:, :2, :]  # Bx2xN

    # Replace NaNs with default value
    pixel_coords = torch.nan_to_num(pixel_coords, nan=default)

    return pixel_coords.transpose(1, 2)  # BxNx2




def cam_from_img(pred_tracks, intrinsics, extra_params=None):
    """
    Normalize predicted tracks based on camera intrinsics.
    Args:
    intrinsics (torch.Tensor): The camera intrinsics tensor of shape [batch_size, 3, 3].
    pred_tracks (torch.Tensor): The predicted tracks tensor of shape [batch_size, num_tracks, 2].
    extra_params (torch.Tensor, optional): Distortion parameters of shape BxN, where N can be 1, 2, or 4.
    Returns:
    torch.Tensor: Normalized tracks tensor.
    """

    # We don't want to do intrinsics_inv = torch.inverse(intrinsics) here
    # otherwise we can use something like
    #     tracks_normalized_homo = torch.bmm(pred_tracks_homo, intrinsics_inv.transpose(1, 2))

    principal_point = intrinsics[:, [0, 1], [2, 2]].unsqueeze(-2)
    focal_length = intrinsics[:, [0, 1], [0, 1]].unsqueeze(-2)
    tracks_normalized = (pred_tracks - principal_point) / focal_length

    if extra_params is not None:
        # Apply iterative undistortion
        try:
            tracks_normalized = iterative_undistortion(
                extra_params, tracks_normalized
            )
        except:
            tracks_normalized = single_undistortion(
                extra_params, tracks_normalized
            )

    return tracks_normalized

# --- First-order propagation of pose + depth uncertainty to 3D points ---

def _skew_symmetric_matrix_vectorized(vec: np.ndarray) -> np.ndarray:
    """
    Creates a vectorized skew-symmetric matrix from a vector.
    [v_x, v_y, v_z] -> [[0, -v_z, v_y], [v_z, 0, -v_x], [-v_y, v_x, 0]]

    Args:
        vec (np.ndarray): A vector of shape (..., 3).

    Returns:
        np.ndarray: The skew-symmetric matrix of shape (..., 3, 3).
    """
    zeros = np.zeros_like(vec[..., :1])
    skew = np.stack([
        zeros, -vec[..., 2:3], vec[..., 1:2],
        vec[..., 2:3], zeros, -vec[..., 0:1],
        -vec[..., 1:2], vec[..., 0:1], zeros
    ], axis=-1)
    return skew.reshape(vec.shape[:-1] + (3, 3))


def _propagate_pixel_uncertainty_vectorized(
    depth_map: np.ndarray,
    depth_conf: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    pose_covariance: np.ndarray,
    depth_conf_to_variance_fn: callable
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Core vectorized function to propagate uncertainty for a single frame.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        depth_conf (np.ndarray): Depth confidence map of shape (H, W).
        extrinsic (np.ndarray): Camera extrinsic matrix (cam from world), shape (3, 4).
        intrinsic (np.ndarray): Camera intrinsic matrix, shape (3, 3).
        pose_covariance (np.ndarray): 6x6 covariance of the pose twist (translation, rotation)
            in the *predicted camera's frame* (body-centric convention, physical units - the
            kappa weighting must already be undone; see vggt.utils.uncertainty).
        depth_conf_to_variance_fn (callable): Function to convert confidence to variance.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - world_points: (H, W, 3) 3D world coordinates.
            - world_points_cov: (H, W, 3, 3) translational covariance for each 3D point.
            - point_mask: (H, W) boolean mask of valid points.
    """
    H, W = depth_map.shape

    # 1. Unproject to get points in camera and world coordinates (reusing existing logic)
    world_points, cam_points, point_mask = depth_to_world_coords_points(depth_map, extrinsic, intrinsic)

    # Flatten points for batch processing. We will only compute for valid points.
    valid_cam_points = cam_points[point_mask] # Shape: (N_valid, 3)
    N_valid = valid_cam_points.shape[0]

    if N_valid == 0:
        # If no valid points, return empty/zero arrays of the correct shape
        world_points_cov = np.zeros((H, W, 3, 3), dtype=np.float32)
        return world_points, world_points_cov, point_mask

    # 2. Compute the Jacobian of the unprojection function w.r.t. pose and depth
    # The unprojection function is P_world = f(pose, depth)
    # The Jacobian J is a 3x7 matrix for each point: J = [∂P_world/∂t, ∂P_world/∂ω, ∂P_world/∂d]
    # where t is translation (3), ω is rotation (3), and d is depth (1).

    # Get inverse rotation (camera to world)
    R_cam_to_world = closed_form_inverse_se3(extrinsic[None])[0][:3, :3] # Shape (3, 3)

    # Jacobian w.r.t translation (J_t): ∂P_world / ∂t = -R_c^w
    J_t = -R_cam_to_world # Shape: (3, 3)

    # Jacobian w.r.t rotation (J_ω): ∂P_world / ∂ω = R_c^w * [P_cam]^x
    # where [P_cam]^x is the skew-symmetric matrix of the point in camera coordinates.
    skew_P_cam = _skew_symmetric_matrix_vectorized(valid_cam_points) # Shape: (N_valid, 3, 3)
    J_omega = R_cam_to_world @ skew_P_cam # Shape: (N_valid, 3, 3) using broadcasting

    # Jacobian w.r.t depth (J_d): ∂P_world / ∂d = R_c^w * (K^-1 * p_pixel)
    # The term (K^-1 * p_pixel) is the normalized ray from the camera center.
    # We can get this by dividing cam_points by depth.
    valid_depths = depth_map[point_mask].reshape(-1, 1) # Shape: (N_valid, 1)
    normalized_rays_cam = valid_cam_points / (valid_depths + 1e-8)
    J_d = (R_cam_to_world @ normalized_rays_cam[..., np.newaxis]).squeeze(-1) # Shape: (N_valid, 3)

    # Assemble the full Jacobian for all valid points
    # J_t is constant for all points in a frame, so we expand it.
    J_t_expanded = np.expand_dims(J_t, axis=0).repeat(N_valid, axis=0) # Shape: (N_valid, 3, 3)
    J = np.concatenate([J_t_expanded, J_omega, J_d[:, :, np.newaxis]], axis=2) # Shape: (N_valid, 3, 7)

    # 3. Assemble the 7x7 input covariance matrix for each point
    # The top-left 6x6 is the pose covariance (same for all points)
    # The bottom-right 1x1 is the depth variance (per-point)
    depth_variances = depth_conf_to_variance_fn(depth_conf[point_mask]) # Shape: (N_valid,)
    
    # Create the block-diagonal input covariance matrix for all valid points
    Sigma_in = np.zeros((N_valid, 7, 7), dtype=np.float32)
    Sigma_in[:, :6, :6] = pose_covariance # Broadcast pose covariance to all points
    Sigma_in[:, 6, 6] = depth_variances

    # 4. Propagate the uncertainty using the formula: Cov_out = J * Cov_in * J^T
    # We use np.einsum for efficient batch matrix multiplication.
    # 'nij,njk,nkl->nil' -> for each point n, do J[i,j] @ Sigma_in[j,k] @ J.T[k,l]
    J_T = J.transpose(0, 2, 1) # Shape: (N_valid, 7, 3)
    world_points_cov_flat = np.einsum('nij,njk->nik', J, Sigma_in)
    world_points_cov_flat = np.einsum('nij,njk->nik', world_points_cov_flat, J_T) # Shape: (N_valid, 3, 3)

    # 5. Un-flatten the covariance matrix to match the image shape
    world_points_cov = np.zeros((H, W, 3, 3), dtype=np.float32)
    world_points_cov[point_mask] = world_points_cov_flat

    return world_points, world_points_cov, point_mask


def unproject_depth_map_with_uncertainty(
    depth_map: np.ndarray,
    depth_conf: np.ndarray,
    extrinsics_cam: np.ndarray,
    intrinsics_cam: np.ndarray,
    pose_covariances: np.ndarray,
    depth_conf_to_variance_fn: callable = lambda c: 1.0 / (c + 1e-6)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unproject a batch of depth maps to 3D world coordinates with uncertainty.

    Args:
        depth_map (np.ndarray): Batch of depth maps, shape (S, H, W, 1) or (S, H, W).
        depth_conf (np.ndarray): Batch of depth confidences, shape (S, H, W).
        extrinsics_cam (np.ndarray): Batch of extrinsic matrices, shape (S, 3, 4).
        intrinsics_cam (np.ndarray): Batch of intrinsic matrices, shape (S, 3, 3).
        pose_covariances (np.ndarray): Batch of 6x6 pose covariance matrices, shape (S, 6, 6), in
            the predicted camera's frame with (translation, rotation) ordering and physical units
            (use pose_covariances_from_predictions(..., frame="body")).
        depth_conf_to_variance_fn (callable, optional): A function to convert a confidence
            score to a variance value. Defaults to a simple inverse relationship.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - world_points_array: Batch of 3D world coordinates, shape (S, H, W, 3).
            - world_points_cov_array: Batch of 3x3 translational covariances, shape (S, H, W, 3, 3).
    """
    # Ensure inputs are numpy arrays
    # (Your existing code already handles this, but it's good practice)
    if isinstance(depth_map, torch.Tensor): depth_map = depth_map.cpu().numpy()
    if isinstance(depth_conf, torch.Tensor): depth_conf = depth_conf.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor): extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor): intrinsics_cam = intrinsics_cam.cpu().numpy()
    if isinstance(pose_covariances, torch.Tensor): pose_covariances = pose_covariances.cpu().numpy()

    world_points_list = []
    world_points_cov_list = []

    for frame_idx in range(depth_map.shape[0]):
        # The core logic is now in a separate, vectorized function for clarity
        cur_world_points, cur_world_points_cov, _ = _propagate_pixel_uncertainty_vectorized(
            depth_map=depth_map[frame_idx].squeeze(),
            depth_conf=depth_conf[frame_idx].squeeze(),
            extrinsic=extrinsics_cam[frame_idx],
            intrinsic=intrinsics_cam[frame_idx],
            pose_covariance=pose_covariances[frame_idx],
            depth_conf_to_variance_fn=depth_conf_to_variance_fn
        )
        world_points_list.append(cur_world_points)
        world_points_cov_list.append(cur_world_points_cov)

    world_points_array = np.stack(world_points_list, axis=0)
    world_points_cov_array = np.stack(world_points_cov_list, axis=0)

    return world_points_array, world_points_cov_array
