# Reproducing the revised thesis

Everything below uses the released head `vggt_uncertainty_head_v1.pt` (epoch 19 of run
`v1.0.0`, temperature 1.015) on top of the upstream VGGT-1B weights.

## Calibration: Table 1, Figure 2, Table 2

The CO3D test sequences are split into a calibration half (`training/splits/co3d_test_calib.txt`,
294 sequences) and an evaluation half (`co3d_test_eval.txt`, 301), see `scripts/split_co3d_test.py`.

```bash
cd training
torchrun --nproc_per_node=1 launch.py --config eval_calib checkpoint.resume_checkpoint_path=logs/v1.0.0/ckpts/checkpoint.pt
torchrun --nproc_per_node=1 launch.py --config eval_eval  checkpoint.resume_checkpoint_path=logs/v1.0.0/ckpts/checkpoint.pt
python ../plot_calibration.py --calib logs/eval_calib/calibration_epoch_0.npz \
    --test logs/eval_eval/calibration_epoch_0.npz --out calibration.png
```

`calibration.json` holds Table 1 (`raw`, `calibrated`), Table 2 (`groups_by_predicted_uncertainty`),
the rank correlations between predicted uncertainty and error, the share of the gross failures
and the uncertainty-dependent temperature check (`logdet_temperature`). `calibration.png` is Figure 2.

## Scenes and the COLMAP baseline: Figures 3-9, Table 3

```bash
pip install pycolmap==3.14.0   # in a separate environment: the covariance API needs pycolmap >= 3.12
HEAD=vggt_uncertainty_head_v1.pt BASE=model.pt COLMAP_PYTHON=/path/to/that/python bash reproduce/scene_figures.sh
```

For each scene this writes `<scene>_agreement.json` (Table 3), the per-camera plots
`<scene>_per_camera.png`, and, when `CHROME` points at a headless Chromium
(`playwright install chromium-headless-shell`), the scene renders
`<scene>_scene_translation.png`, `_scene_rotation.png` and `_scene_colmap.png` used as the
panels of Figures 3-9. They are drawn with viser in the reconstructed scene, from the view
set per scene in `scripts/render_scene_uncertainty.py`. The COLMAP models of the example
scenes are in `colmap/` (COLMAP default settings, `SIMPLE_RADIAL` camera per image), and the
eight *Pyramid* frames used in the thesis, taken from `examples/videos/pyramid.mp4`, in
`scenes/pyramid/images`. `scripts/colmap_pose_covariance.py` explains how the bundle
adjustment covariance is brought into the gauge and units of VGGT.

The EPIC-KITCHENS scenes are not redistributed. To include them, take from the videos of
EPIC-KITCHENS-100 every 5th frame from 5 to 150 of `P04_11` (30 frames) and every 6th frame
from 6 to 174 of `P01_09` (29 frames), named `frame_%06d.jpg`, into
`$EPIC_DIR/<scene>/images`, reconstruct `P04_11` with COLMAP's default settings into
`$EPIC_DIR/P04_11/sparse`, and set `EPIC_DIR` when running the script.
