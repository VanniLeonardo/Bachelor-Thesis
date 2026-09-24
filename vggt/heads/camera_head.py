# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from vggt.heads.head_act import activate_pose
from vggt.layers import Mlp
from vggt.layers.block import Block
from vggt.utils.uncertainty import cholesky_bias_for_targets


class CameraHead(nn.Module):
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",
    ):
        super().__init__()

        if pose_encoding_type == "absT_quaR_FoV":
            self.mean_dim = 9  # upstream VGGT `target_dim`
            self.cholesky_dim = 21
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        # Original (frozen) VGGT mean-pose pathway; `pose_branch` renamed `mean_pose_branch`.
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.mean_dim))
        self.embed_pose = nn.Linear(self.mean_dim, dim_in)
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)

        self.mean_pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.mean_dim, drop=0)

        # Covariance branch: fused [aggregator camera token | trunk hidden state]
        # (D = 2*dim_in = 4096) -> LN -> six Linear/LN/GELU blocks (2D, 2D, 2D, D, D/2, D/4;
        # Dropout 0.05 on the first five) -> 21 Cholesky entries.
        # Poses are predicted in VGGT's normalized frame (mean point distance from camera 0
        # equal to 1), so the scene scale is 1 by construction and needs no embedding.
        fused_dim = dim_in * 2
        self.covariance_branch = nn.Sequential(
            nn.LayerNorm(fused_dim),

            nn.Linear(fused_dim, fused_dim * 2),
            nn.LayerNorm(fused_dim * 2),
            nn.GELU(),
            nn.Dropout(0.05),

            nn.Linear(fused_dim * 2, fused_dim * 2),
            nn.LayerNorm(fused_dim * 2),
            nn.GELU(),
            nn.Dropout(0.05),

            nn.Linear(fused_dim * 2, fused_dim * 2),
            nn.LayerNorm(fused_dim * 2),
            nn.GELU(),
            nn.Dropout(0.05),

            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(0.05),

            nn.Linear(fused_dim, fused_dim // 2),
            nn.LayerNorm(fused_dim // 2),
            nn.GELU(),
            nn.Dropout(0.05),

            nn.Linear(fused_dim // 2, fused_dim // 4),
            nn.LayerNorm(fused_dim // 4),
            nn.GELU(),

            nn.Linear(fused_dim // 4, self.cholesky_dim)
        )

        self._init_uncertainty_head()

    def _init_uncertainty_head(self, trans_std: float = 0.5, rot_std: float = 0.1, off_diagonal: float = 0.01):
        """Initialise the covariance branch to output an (almost) diagonal covariance.

        The final layer's bias is set through the tril layout of the Cholesky vector so
        that the initial factor has ``diag(L) = [trans_std]*3 + [rot_std]*3`` (weighted
        rotation units) and ``off_diagonal`` elsewhere; its weights are scaled by 0.1.
        Hidden linear layers use Xavier-normal (gain 0.5) weights and zero biases.
        """
        linear_layers = [m for m in self.covariance_branch.modules() if isinstance(m, nn.Linear)]
        output_layer = linear_layers[-1]
        with torch.no_grad():
            target_bias = cholesky_bias_for_targets([trans_std] * 3 + [rot_std] * 3, off_diagonal=off_diagonal)
            output_layer.bias.copy_(target_bias.to(output_layer.bias.dtype))
            output_layer.weight.mul_(0.1)
            for module in linear_layers[:-1]:
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4) -> dict:
        """Predict the mean pose (frozen VGGT pathway) and the Cholesky factor of its covariance.

        Args:
            aggregated_tokens_list: aggregator outputs; the last tensor is used.
            num_iterations: refinement iterations of the frozen mean pathway.

        Returns:
            dict with ``pose_enc`` (B, S, 9) and ``cholesky_vector`` (B, S, 21).
        """
        tokens = aggregated_tokens_list[-1]
        aggregator_camera_tokens = tokens[:, :, 0]  # (B, S, C): what the aggregator "saw"

        # Frozen mean pathway: run in eval mode without gradients so that the
        # deterministic VGGT prediction is preserved exactly.
        mean_path_modules = [
            self.trunk, self.token_norm, self.trunk_norm,
            self.embed_pose, self.poseLN_modulation, self.adaln_norm, self.mean_pose_branch,
        ]
        original_modes = {m: m.training for m in mean_path_modules}
        for m in mean_path_modules:
            m.eval()
        with torch.no_grad():
            final_mean_pose, final_hidden_state = self.iterative_refinement(aggregator_camera_tokens, num_iterations)
        for m, mode in original_modes.items():
            m.train(mode)

        # Visual evidence (aggregator camera token) fused with the trunk's final hidden state.
        fused_features = torch.cat([aggregator_camera_tokens, final_hidden_state], dim=-1)
        cholesky_vector = self.covariance_branch(fused_features)

        activated_pose = activate_pose(
            final_mean_pose, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
        )
        return {"pose_enc": activated_pose, "cholesky_vector": cholesky_vector}

    def iterative_refinement(self, pose_tokens: torch.Tensor, num_iterations: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function now ONLY computes the mean pose iteratively. It should only be
        called within a torch.no_grad() block and with its modules in eval() mode.
        It returns the final mean pose and the final hidden state.
        """
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        final_hidden_state = None

        pose_tokens_normed = self.token_norm(pose_tokens)

        for _ in range(num_iterations):
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                module_input = self.embed_pose(pred_pose_enc) # Already detached in the main forward loop

            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens_normed), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens_normed

            trunk_output = self.trunk(pose_tokens_modulated)
            final_hidden_state = self.trunk_norm(trunk_output)

            pred_pose_enc_delta = self.mean_pose_branch(final_hidden_state)

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

        return pred_pose_enc, final_hidden_state

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift
