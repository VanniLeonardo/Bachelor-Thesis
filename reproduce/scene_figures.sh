#!/usr/bin/env bash
# Scene figures and Table 3 of the revised thesis (learned uncertainty vs COLMAP bundle adjustment).
#   HEAD=vggt_uncertainty_head_v1.pt BASE=model.pt COLMAP_PYTHON=/path/to/python-with-pycolmap-3.14 \
#       CHROME=/path/to/chrome-headless-shell bash reproduce/scene_figures.sh [out_dir]
# EPIC-KITCHENS scenes run only when EPIC_DIR points at the extracted frames (see reproduce/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-reproduce/out}
HEAD=${HEAD:?set HEAD to the exported uncertainty head}
BASE=${BASE:-}
COLMAP_PYTHON=${COLMAP_PYTHON:-python}
mkdir -p "$OUT"

scene() {  # name images [colmap_model]
  local name=$1 images=$2 model=${3:-}
  local extra=()
  if [ -n "$model" ]; then
    "$COLMAP_PYTHON" scripts/colmap_pose_covariance.py --model "$model" --images "$images" --out "$OUT/${name}_colmap_cov.npz"
    extra=(--colmap "$OUT/${name}_colmap_cov.npz")
  fi
  python scripts/compare_pose_uncertainty.py --images "$images" --name "$name" --head "$HEAD" ${BASE:+--base "$BASE"} \
      --out_dir "$OUT" "${extra[@]}"
  if [ -n "${CHROME:-}" ]; then  # scene renders used in the thesis (headless viser + Chromium)
    python scripts/render_scene_uncertainty.py --images "$images" --name "$name" --head "$HEAD" ${BASE:+--base "$BASE"} \
        --out_dir "$OUT" "${extra[@]}"
  fi
}

scene flower  examples/llff_flower/images     reproduce/colmap/llff_flower/sparse
scene fern    examples/llff_fern/images       reproduce/colmap/llff_fern/sparse
scene pyramid reproduce/scenes/pyramid/images reproduce/colmap/pyramid/sparse
scene kitchen examples/kitchen/images         reproduce/colmap/kitchen/sparse
scene room    examples/room/images            # COLMAP registers 3 of 8 images: no baseline
if [ -n "${EPIC_DIR:-}" ]; then
  scene P04_11 "$EPIC_DIR/P04_11/images" "$EPIC_DIR/P04_11/sparse"
  scene P01_09 "$EPIC_DIR/P01_09/images"      # COLMAP does not register the first frame: no baseline
fi
echo "figures and *_agreement.json in $OUT"
