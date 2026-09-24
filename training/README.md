# Training the pose-uncertainty head

This folder is the upstream VGGT training framework (Hydra config, DDP trainer, CO3D
loader) reduced to one job: fine-tune the `camera_head.covariance_branch` of a frozen
VGGT-1B so that it predicts a calibrated 6×6 covariance of the camera pose.

## 1. Data

1. Download CO3D v2 (or a subset of categories) with the official script:
   ```bash
   git clone https://github.com/facebookresearch/co3d && cd co3d
   python co3d/download_dataset.py --download_folder /data/CO3D \
       --download_categories tv,parkingmeter,baseballbat,microwave,baseballglove,pizza,toybus,bowl,donut,toytrain,toyplane,cake,broccoli,banana,toaster,cup,bicycle,car \
       --n_download_workers 2 --n_extract_workers 2 --clear_archives_after_unpacking
   ```
   The 18 categories above (≈830 GB, 4 650 training and 595 test sequences, all in VGGT's
   training list) are the ones used for the released head; any subset works.
2. Download the VGGT annotation files (all 51 categories, 0.6 GB):
   ```bash
   python -c "from huggingface_hub import snapshot_download; snapshot_download('JianyuanWang/co3d_anno', repo_type='dataset', local_dir='/data/CO3D_ann')"
   ```
   Sequences whose images or depth maps are missing on disk are skipped automatically and
   the counts are written to the log.
3. Point the config at your paths (environment variables override the defaults in `config/default.yaml`):
   ```bash
   export CO3D_DIR=/data/CO3D CO3D_ANNOTATION_DIR=/data/CO3D_ann VGGT_PRETRAINED_CKPT=/path/to/model.pt
   ```

## 2. Train

```bash
cd training
torchrun --nproc_per_node=1 launch.py --config default            # single GPU
torchrun --nproc_per_node=4 launch.py --config default            # DDP
torchrun --nproc_per_node=1 launch.py --config default max_epochs=10 exp_name=my_run   # Hydra overrides
```

Stopping and resuming: a run can be killed at any time and restarted with the same command.
It resumes from `logs/<exp_name>/ckpts/checkpoint.pt` (written after every epoch) with the epoch,
optimizer state and data order restored, so the result is identical to an uninterrupted run.
The progress of a partly finished epoch is lost. Use a new `exp_name` to start a fresh run.

What the default config does (`config/default.yaml`):

- Freezes everything except `camera_head.covariance_branch`; the frozen modules run in eval mode, so VGGT's poses are exactly the upstream ones.
- Normalizes each batch to VGGT's frame (camera 0 = identity, mean point distance 1). All losses are computed in these units; camera 0 is excluded (it carries no error).
- Loss = Gaussian NLL of the body-centric error twist `log(E_pred E_gt⁻¹)` in κ-weighted coordinates (κ = 10) with a shielded covariance `Σ = LLᵀ + εI`, plus the scale, condition-number and isotropy regularizers of the thesis (Algorithms 1–4).
- Curriculum: `warmup_epochs` with the NLL disabled (regularizers only), then NLL with the regularizer weights annealed linearly to zero at `max_epochs`.
- Dynamic batching: 2–24 frames per sample, `floor(24 / n_frames)` samples per step, a frame-reversed copy appended, 2 accumulation steps; AdamW 5e-5, 5% linear warm-up then cosine decay.
- Logs the resolved configuration, the checkpoint path and the trainable tensors to `logs/<exp_name>/log.txt`; TensorBoard scalars include the mean Mahalanobis distance (≈6 when calibrated), the mean log-det, and the mean translational / rotational σ.

Memory: `max_img_per_gpu`, `accum_steps` and `img_size` control the footprint; the defaults fit a 24 GB GPU.

## 3. Evaluate and calibrate

The CO3D test sequences are split once into two disjoint halves, balanced per category
(`splits/co3d_test_calib.txt`, 294 sequences, and `splits/co3d_test_eval.txt`, 301; see
`../scripts/split_co3d_test.py`). The temperature is fitted on the first and every
reported number comes from the second.

```bash
# 1000 validation batches on each half; each run writes logs/<exp>/calibration_epoch_0.npz
# with the per-frame d^2, log-det and translation / rotation error
torchrun --nproc_per_node=1 launch.py --config eval_calib checkpoint.resume_checkpoint_path=logs/v1.0.0/ckpts/checkpoint.pt
torchrun --nproc_per_node=1 launch.py --config eval_eval  checkpoint.resume_checkpoint_path=logs/v1.0.0/ckpts/checkpoint.pt

# fit T on the calibration half, report T = 1 and the fitted T on the evaluation half
python ../plot_calibration.py --calib logs/eval_calib/calibration_epoch_0.npz \
    --test logs/eval_eval/calibration_epoch_0.npz --out calibration.png
```

The temperature is fitted by matching the median of `d^2` to that of `chi^2_6`. The
Gaussian maximum-likelihood fit `T = mean(d^2) / 6` is dominated by the few frames where
VGGT fails by tens of degrees, so it is reported only for reference. Because the
temperature multiplies the shielded covariance, the NLL at any `T` follows exactly from
the saved `d^2` and log-det, and a second evaluation run is not needed.

## 4. Export the head for release

```bash
python ../scripts/export_uncertainty_head.py logs/v1.0.0/ckpts/checkpoint.pt vggt_uncertainty_head_v1.pt --temperature 1.015
```

The export contains only the covariance branch (fp16, ≈0.4 GB) plus metadata (κ, shield ε, error convention, temperature, source checkpoint); `vggt.utils.checkpoint.load_vggt_with_uncertainty` combines it with the upstream VGGT-1B weights.

## 5. Notes inherited from upstream

- Camera poses follow the OpenCV camera-from-world convention; depth maps are aligned with their cameras.
- The learning rate depends on the effective batch size; try 5e-6 … 5e-4 if you change it.
- To sanity-check the loader, dump `batch["world_points"]` to a PLY and inspect it (see the upstream VGGT repository for a snippet).
- Multi-dataset training (`data.composed_dataset.ComposedDataset` with several `dataset_configs` and `len_train` ratios) works as upstream.
