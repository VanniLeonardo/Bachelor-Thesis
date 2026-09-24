# Copyright (c) 2026 Leonardo Vanni. CC BY-NC 4.0 (see LICENSE.txt).
"""Unit tests for the pose-uncertainty math, loss plumbing and camera head.

All tests run on CPU in a few seconds:  `.venv/bin/python -m pytest tests -q`.
"""
import math
import os

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from vggt.utils.uncertainty import (
    DIAGONAL_POSITIONS,
    cholesky_bias_for_targets,
    cholesky_vector_to_L,
    covariance_from_cholesky_vector,
    covariance_to_world_frame,
    gaussian_nll,
    pose_error_twist,
    se3_inverse,
    se3_log_map,
    to_homogeneous,
    unweight_covariance,
)

torch.manual_seed(0)
np.random.seed(0)


def _hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def _random_se3(theta, dtype=torch.float64):
    w = np.random.randn(3)
    w = w / np.linalg.norm(w) * theta
    v = np.random.randn(3)
    xi = np.zeros((4, 4))
    xi[:3, :3] = _hat(w)
    xi[:3, 3] = v
    return torch.tensor(expm(xi), dtype=dtype), torch.tensor(np.concatenate([v, w]), dtype=dtype)


def _random_extrinsics(n, dtype=torch.float64):
    Es = []
    for _ in range(n):
        T, _ = _random_se3(np.random.uniform(0.1, 2.5), dtype)
        Es.append(T[:3])
    return torch.stack(Es)


# --------------------------------------------------------------------------- log map
@pytest.mark.parametrize("theta", [1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 3.0])
def test_se3_log_map_inverts_exp(theta):
    """log(Exp(xi)) == xi including the translational part (inverse left Jacobian)."""
    for _ in range(10):
        T, xi = _random_se3(theta)
        out = se3_log_map(T[None])[0]
        assert torch.allclose(out, xi, atol=1e-6, rtol=1e-5), (theta, out, xi)


def test_se3_log_map_pure_translation_and_identity():
    T = torch.eye(4, dtype=torch.float64)
    T[:3, 3] = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    xi = se3_log_map(T[None])[0]
    assert torch.allclose(xi, torch.tensor([0.1, -0.2, 0.3, 0, 0, 0], dtype=torch.float64), atol=1e-9)
    assert torch.allclose(se3_log_map(torch.eye(4, dtype=torch.float64)[None])[0], torch.zeros(6, dtype=torch.float64))


def test_se3_log_map_float32_batched_shapes_and_grad():
    T = torch.stack([_random_se3(0.3, torch.float32)[0] for _ in range(6)]).view(2, 3, 4, 4).requires_grad_(True)
    xi = se3_log_map(T)
    assert xi.shape == (2, 3, 6)
    xi.sum().backward()
    assert torch.isfinite(T.grad).all()


def test_se3_inverse():
    T, _ = _random_se3(1.2)
    assert torch.allclose(se3_inverse(T[None])[0] @ T, torch.eye(4, dtype=torch.float64), atol=1e-12)


# --------------------------------------------------------------------------- error convention
def test_identical_prediction_gives_zero_twist_for_any_scene_scale():
    """Regression for the double-normalization bug: pred == gt must give xi == 0 regardless of scene scale."""
    E = _random_extrinsics(4)
    for convention in ("body", "world"):
        xi = pose_error_twist(E, E, convention=convention)
        assert torch.allclose(xi, torch.zeros_like(xi), atol=1e-10)


def test_body_convention_is_invariant_to_world_reference_frame():
    E_pred = _random_extrinsics(5)
    E_gt = _random_extrinsics(5)
    W, _ = _random_se3(1.7)  # arbitrary re-referencing of the world frame: P -> W P  <=>  E -> E W^{-1}
    W_inv = se3_inverse(W[None])[0]
    E_pred_w = (to_homogeneous(E_pred) @ W_inv)[..., :3, :]
    E_gt_w = (to_homogeneous(E_gt) @ W_inv)[..., :3, :]
    xi_body = pose_error_twist(E_pred, E_gt, "body")
    xi_body_w = pose_error_twist(E_pred_w, E_gt_w, "body")
    assert torch.allclose(xi_body, xi_body_w, atol=1e-9)
    xi_world = pose_error_twist(E_pred, E_gt, "world")
    xi_world_w = pose_error_twist(E_pred_w, E_gt_w, "world")
    assert not torch.allclose(xi_world, xi_world_w, atol=1e-6)


def test_body_convention_translation_is_camera_centre_error_in_camera_frame():
    """For a pure translation error the twist is R_pred (c_gt - c_pred): the centre error in the predicted camera's frame."""
    E = _random_extrinsics(1)[0]
    R, t = E[:, :3], E[:, 3]
    c_pred = -R.T @ t
    delta_world = torch.tensor([0.05, -0.02, 0.01], dtype=torch.float64)
    c_gt = c_pred + delta_world
    E_gt = torch.cat([R, (-R @ c_gt)[:, None]], dim=1)
    xi = pose_error_twist(E[None], E_gt[None], "body")[0]
    assert torch.allclose(xi[3:], torch.zeros(3, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(xi[:3], R @ delta_world, atol=1e-10)


# --------------------------------------------------------------------------- Cholesky parameterisation
def test_diagonal_positions_and_bias_layout():
    assert DIAGONAL_POSITIONS == [0, 2, 5, 9, 14, 20]
    stds = [0.5, 0.5, 0.5, 0.1, 0.1, 0.1]
    bias = cholesky_bias_for_targets(stds, off_diagonal=0.0)
    L, log_diag = cholesky_vector_to_L(bias[None])
    assert torch.allclose(torch.exp(log_diag[0]), torch.tensor(stds))
    Sigma = L[0] @ L[0].T
    assert torch.allclose(Sigma, torch.diag(torch.tensor(stds) ** 2), atol=1e-7)


def test_cholesky_vector_roundtrip_and_clamp():
    vec = torch.randn(3, 4, 21)
    L, log_diag = cholesky_vector_to_L(vec, log_diag_min=-1.0, log_diag_max=1.0)
    assert L.shape == (3, 4, 6, 6)
    assert torch.equal(torch.triu(L, 1), torch.zeros_like(L))
    assert (torch.diagonal(L, dim1=-2, dim2=-1) > 0).all()
    assert (log_diag >= -1.0).all() and (log_diag <= 1.0).all()
    # strictly-lower part is passed through unchanged
    assert torch.allclose(L[..., 1, 0], vec[..., 1])


def test_unweight_and_world_frame_conversion():
    vec = torch.randn(2, 21)
    kappa = 10.0
    Sigma_phys = covariance_from_cholesky_vector(vec, kappa=kappa, temperature=2.0)
    L, _ = cholesky_vector_to_L(vec)
    Sigma_w = 2.0 * (L @ L.transpose(-1, -2))
    # rotational block divided by kappa^2, cross terms by kappa, translational block untouched
    assert torch.allclose(Sigma_phys[..., :3, :3], Sigma_w[..., :3, :3], atol=1e-6)
    assert torch.allclose(Sigma_phys[..., 3:, 3:], Sigma_w[..., 3:, 3:] / kappa**2, atol=1e-6)
    assert torch.allclose(Sigma_phys[..., :3, 3:], Sigma_w[..., :3, 3:] / kappa, atol=1e-6)
    assert torch.allclose(unweight_covariance(Sigma_w, kappa), Sigma_phys, atol=1e-6)
    E = _random_extrinsics(2, torch.float32)
    Sigma_world = covariance_to_world_frame(Sigma_phys, E)
    assert torch.allclose(Sigma_world, Sigma_world.transpose(-1, -2), atol=1e-5)
    # a rotation preserves the eigenvalues of each 3x3 block
    assert torch.allclose(torch.linalg.eigvalsh(Sigma_world[..., :3, :3]), torch.linalg.eigvalsh(Sigma_phys[..., :3, :3]), atol=1e-4)


def test_gaussian_nll_matches_direct_formula_and_is_bounded_below():
    xi = torch.randn(5, 6, dtype=torch.float64)
    vec = torch.randn(5, 21, dtype=torch.float64)
    L, _ = cholesky_vector_to_L(vec)
    eps = 1e-3
    nll, d2, logdet = gaussian_nll(xi, L, eps)
    Sigma = L @ L.transpose(-1, -2) + eps * torch.eye(6, dtype=torch.float64)
    d2_ref = torch.einsum("bi,bij,bj->b", xi, torch.linalg.inv(Sigma), xi)
    nll_ref = 0.5 * (d2_ref + torch.logdet(Sigma) + 6 * math.log(2 * math.pi))
    assert torch.allclose(d2, d2_ref, atol=1e-8) and torch.allclose(nll, nll_ref, atol=1e-8)
    # shrinking the factor to zero cannot drive the NLL to -inf: both terms use the shielded Sigma
    tiny_L = torch.zeros(1, 6, 6, dtype=torch.float64)
    nll_tiny, _, _ = gaussian_nll(xi[:1], tiny_L, eps)
    assert torch.isfinite(nll_tiny).all()
    assert nll_tiny.item() >= 0.5 * (6 * math.log(eps) + 6 * math.log(2 * math.pi))


# --------------------------------------------------------------------------- training loss
def _synthetic_batch(B=2, S=4, scene_scale=(1.0, 7.0), noise=0.0, dtype=torch.float32):
    from vggt.utils.pose_enc import extri_intri_to_pose_encoding

    E = torch.stack([_random_extrinsics(S, dtype) for _ in range(B)])  # (B, S, 3, 4)
    E[:, 0] = torch.eye(4, dtype=dtype)[:3]  # camera 0 is the reference
    K = torch.eye(3, dtype=dtype).repeat(B, S, 1, 1)
    H, W = 224, 224
    K[..., 0, 0] = 300.0
    K[..., 1, 1] = 300.0
    K[..., 0, 2] = W / 2
    K[..., 1, 2] = H / 2
    images = torch.zeros(B, S, 3, H, W, dtype=dtype)
    E_pred = E.clone()
    if noise:
        E_pred[..., :3, 3] += noise * torch.randn_like(E_pred[..., :3, 3])
    pose_enc = extri_intri_to_pose_encoding(E_pred, K, (H, W), pose_encoding_type="absT_quaR_FoV")
    batch = {"extrinsics": E, "intrinsics": K, "images": images, "scene_scale": torch.tensor(scene_scale, dtype=dtype)}  # scene_scale kept only for logging
    return pose_enc, batch


def test_pose_nll_zero_error_for_any_scene_scale_and_shapes():
    from loss import compute_pose_nll_loss

    pose_enc, batch = _synthetic_batch(B=2, S=4, scene_scale=(1.0, 7.0))
    chol = torch.randn(2, 4, 21) * 0.1
    out = compute_pose_nll_loss({"pose_enc": pose_enc, "cholesky_vector": chol}, batch)
    assert out["mahalanobis_mean"].item() < 1e-6  # identical prediction => zero Mahalanobis distance
    assert out["twist_trans_err_mean"].item() < 1e-6
    assert out["raw_mahalanobis_sq"].shape == (2 * 3,)  # reference frame excluded
    for k in ("loss_nll", "loss_nll_base", "loss_uncertainty_scale", "loss_condition_number", "loss_uncertainty_balance"):
        assert out[k].ndim == 0 and torch.isfinite(out[k])


def test_pose_nll_single_frame_sequences_keep_shapes():
    from loss import compute_pose_nll_loss

    pose_enc, batch = _synthetic_batch(B=3, S=1, scene_scale=(1.0, 2.0, 3.0))
    chol = torch.randn(3, 1, 21) * 0.1
    out = compute_pose_nll_loss({"pose_enc": pose_enc, "cholesky_vector": chol}, batch, exclude_reference_frame=False)
    assert out["raw_mahalanobis_sq"].shape == (3,)
    assert torch.isfinite(out["loss_nll"])


def test_pose_nll_gradient_reaches_cholesky_vector_and_temperature_scales_mahalanobis():
    from loss import compute_pose_nll_loss

    pose_enc, batch = _synthetic_batch(B=2, S=3, noise=0.05)
    chol = (torch.randn(2, 3, 21) * 0.1).requires_grad_(True)
    out = compute_pose_nll_loss({"pose_enc": pose_enc, "cholesky_vector": chol}, batch, shield_eps=0.0)
    out["loss_nll"].backward()
    assert chol.grad is not None and torch.isfinite(chol.grad).all() and chol.grad.abs().sum() > 0
    out_T = compute_pose_nll_loss({"pose_enc": pose_enc, "cholesky_vector": chol.detach()}, batch, shield_eps=0.0, temperature=4.0)
    assert torch.allclose(out_T["mahalanobis_mean"], out["mahalanobis_mean"] / 4.0, rtol=1e-4)


def test_pose_nll_skips_samples_without_a_defined_scene_scale():
    # A CO3D sample whose depth masks leave no valid point is normalized with scale 1e-6,
    # so its GT translations reach ~1e7.  It must not touch the loss, metrics or gradient.
    from loss import compute_pose_nll_loss

    pose_enc, batch = _synthetic_batch(B=3, S=4, noise=0.05)
    batch["extrinsics"][2, 1:, :3, 3] *= 1e7
    batch["scene_scale_valid"] = torch.tensor([True, True, False])
    chol = (torch.randn(3, 4, 21) * 0.1).requires_grad_(True)
    out = compute_pose_nll_loss({"pose_enc": pose_enc, "cholesky_vector": chol}, batch)
    out["loss_nll"].backward()
    assert torch.isfinite(chol.grad).all() and chol.grad[2].abs().sum() == 0 and chol.grad[:2].abs().sum() > 0

    clean = {k: (v[:2] if torch.is_tensor(v) and v.shape[:1] == (3,) else v) for k, v in batch.items()}
    ref = compute_pose_nll_loss({"pose_enc": pose_enc[:2], "cholesky_vector": chol.detach()[:2]}, clean)
    for k in ("loss_nll", "mahalanobis_mean", "twist_trans_err_mean"):
        assert torch.allclose(out[k], ref[k]), k
    assert out["raw_mahalanobis_sq"].shape == (2 * 3,)

    batch["scene_scale_valid"][:] = False  # a batch with nothing usable gives zero loss, not NaN
    out = compute_pose_nll_loss({"pose_enc": pose_enc, "cholesky_vector": chol.detach()}, batch)
    assert out["loss_nll"].item() == 0 and out["raw_mahalanobis_sq"].numel() == 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast needs CUDA")
def test_pose_nll_is_exact_under_bf16_autocast_for_a_180_degree_error():
    # A real training frame: VGGT flipped the camera (179.4 deg relative rotation).  Under
    # bf16 autocast the log map returned |nu| = 2.6e10 instead of 2.955; the loss must be
    # computed in fp32 whatever the surrounding AMP context is.
    from loss import compute_pose_nll_loss
    from vggt.utils.pose_enc import extri_intri_to_pose_encoding

    eye = torch.eye(4)[:3]
    P = torch.tensor([[6.80282414e-01, -5.09025902e-02, 7.31180429e-01, -2.13810563e-01],
                      [2.85228908e-01, 9.37334895e-01, -2.00119466e-01, -8.22507683e-03],
                      [-6.75174296e-01, 3.44691515e-01, 6.52171314e-01, -6.71544531e-03]])
    G = torch.tensor([[-6.73865259e-01, 3.89874578e-01, -6.27617300e-01, -2.80846596e-01],
                      [-2.92769641e-01, 6.39015436e-01, 7.11298287e-01, -1.01225030e+00],
                      [6.78374290e-01, 6.63066447e-01, -3.16466838e-01, 1.51743698e+00]])
    H = W = 224
    K = torch.tensor([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]]).expand(1, 2, 3, 3)
    pose_enc = extri_intri_to_pose_encoding(torch.stack([eye, P])[None], K, (H, W), pose_encoding_type="absT_quaR_FoV")
    batch = {"extrinsics": torch.stack([eye, G])[None].cuda(), "images": torch.zeros(1, 2, 3, H, W).cuda()}
    preds = {"pose_enc": pose_enc.cuda(), "cholesky_vector": torch.zeros(1, 2, 21).cuda()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = compute_pose_nll_loss(preds, batch)
    assert abs(out["twist_trans_err_mean"].item() - 2.955) < 0.01
    assert abs(out["twist_rot_err_mean_deg"].item() - 179.45) < 0.1

def test_temperature_scales_the_shielded_covariance_like_inference():
    # Sigma_T = T (L L^T + eps I), as in pose_covariances_from_predictions, so d^2 scales by
    # exactly 1/T and log det shifts by exactly 6 log T even with the shield active.
    from loss import compute_pose_nll_loss

    pose_enc, batch = _synthetic_batch(B=2, S=3, noise=0.05)
    chol = torch.randn(2, 3, 21) * 0.1 - 3.0  # small variances, so the 1e-4 shield matters
    preds = {"pose_enc": pose_enc, "cholesky_vector": chol}
    out1 = compute_pose_nll_loss(preds, batch, shield_eps=1e-4)
    outT = compute_pose_nll_loss(preds, batch, shield_eps=1e-4, temperature=2.5)
    assert torch.allclose(outT["raw_mahalanobis_sq"], out1["raw_mahalanobis_sq"] / 2.5, rtol=1e-4)
    assert torch.allclose(outT["raw_log_det"], out1["raw_log_det"] + 6 * math.log(2.5), atol=1e-3)

def test_multitask_loss_curriculum_and_config_plumbing():
    from loss import MultitaskLoss

    loss = MultitaskLoss(pose_uncertainty={"kappa": 3.0, "shield_eps": 1e-3}, warmup_epochs=2, anneal_regularizers=True, eval_temperature=2.5)
    assert loss.curriculum(0, 10, "train") == (0.0, 1.0, 1.0)  # warm-up: NLL off
    assert loss.curriculum(1, 10, "train") == (0.0, 1.0, 1.0)
    w, f, T = loss.curriculum(2, 10, "train")
    assert (w, f, T) == (1.0, 1.0, 1.0)
    w, f, T = loss.curriculum(6, 10, "train")
    assert w == 1.0 and abs(f - 0.5) < 1e-9 and T == 1.0
    assert loss.curriculum(10, 10, "train")[1] == 0.0
    assert loss.curriculum(6, 10, "val") == (1.0, 1.0, 2.5)  # temperature only at evaluation

    pose_enc, batch = _synthetic_batch(B=2, S=3, noise=0.05)
    preds = {"pose_enc": pose_enc, "cholesky_vector": torch.randn(2, 3, 21) * 0.1}
    warm = loss(preds, batch, epoch=0, total_epochs=10, phase="train")
    assert warm["curriculum_nll_weight"].item() == 0.0
    assert torch.allclose(warm["objective"], warm["loss_uncertainty_scale"] + warm["loss_condition_number"] + warm["loss_uncertainty_balance"])
    full = loss(preds, batch, epoch=2, total_epochs=10, phase="train")
    assert full["curriculum_nll_weight"].item() == 1.0
    assert torch.allclose(full["objective"], full["loss_nll_base"] + full["loss_uncertainty_scale"] + full["loss_condition_number"] + full["loss_uncertainty_balance"], atol=1e-5)
    with pytest.raises(TypeError):
        MultitaskLoss(uncertainty_training={"scale_regularization": 1.0})  # stale config key must not be silently dropped


def test_default_config_instantiates_loss_and_resolves_paths(monkeypatch):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    monkeypatch.setenv("CO3D_DIR", "/data/co3d")
    monkeypatch.setenv("VGGT_PRETRAINED_CKPT", "/data/model.pt")
    cfg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training", "config")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="default")
    assert cfg.data.train.dataset.dataset_configs[0].CO3D_DIR == "/data/co3d"
    assert cfg.checkpoint.resume_checkpoint_path == "/data/model.pt"
    loss = instantiate(cfg.loss)
    assert loss.pose_uncertainty["kappa"] == 10.0 and loss.pose_uncertainty["error_convention"] == "body"
    assert loss.warmup_epochs == 1 and loss.eval_temperature == 1.0


# --------------------------------------------------------------------------- camera head
def test_camera_head_forward_shapes_and_initial_covariance():
    from vggt.heads.camera_head import CameraHead

    head = CameraHead(dim_in=64, trunk_depth=1, num_heads=4)
    head.eval()
    tokens = torch.randn(2, 5, 7, 64)  # (B, S, tokens, C); token 0 is the camera token
    out = head([tokens])
    assert set(out) == {"pose_enc", "cholesky_vector"}
    assert out["pose_enc"].shape == (2, 5, 9) and out["cholesky_vector"].shape == (2, 5, 21)
    assert head.covariance_branch[1].in_features == 2 * 64  # fused [camera token | trunk state], no scale embedding
    # the output bias alone yields the intended (nearly) diagonal covariance
    L, log_diag = cholesky_vector_to_L(head.covariance_branch[-1].bias[None])
    assert torch.allclose(torch.exp(log_diag[0]), torch.tensor([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]))
    Sigma = L[0] @ L[0].T
    assert (Sigma.diagonal() < 0.3).all()
    # the frozen mean pathway must be identical to the upstream behaviour: no grad flows into it
    trainable = {n for n, p in head.named_parameters() if p.requires_grad}
    assert any(n.startswith("covariance_branch") for n in trainable) and not any("scale_embedding" in n for n in trainable)
