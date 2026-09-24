#!/usr/bin/env python3
# Copyright (c) 2026 Leonardo Vanni. CC BY-NC 4.0 (see LICENSE.txt).
"""Split the CO3D test sequences into disjoint calibration and evaluation halves.

The temperature is fitted on the calibration half and every reported number comes from
the evaluation half, so the reported calibration is out of sample.  Each category is
shuffled with a fixed seed and halved, so both halves cover the same categories.  Only
sequences the loader would keep are split (at least 24 frames present on disk).

    python scripts/split_co3d_test.py --co3d_dir $CO3D_DIR --annotation_dir $CO3D_ANNOTATION_DIR
"""
import argparse
import gzip
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "training"))
from data.archive_store import read_manifest  # noqa: E402
from data.datasets.co3d import SEEN_CATEGORIES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--co3d_dir", required=True)
    ap.add_argument("--annotation_dir", required=True)
    ap.add_argument("--out_dir", default=os.path.join(REPO, "training", "splits"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_num_images", type=int, default=24)
    args = ap.parse_args()

    present = {seq: set(frames) for seq, frames in read_manifest(args.co3d_dir).items()}
    calib, evaluation = [], []
    for cat in sorted(SEEN_CATEGORIES):
        path = os.path.join(args.annotation_dir, f"{cat}_test.jgz")
        if not os.path.exists(path):
            continue
        with gzip.open(path, "r") as f:
            annotation = json.loads(f.read())
        seqs = sorted(
            s for s, frames in annotation.items()
            if sum(fr["filepath"] in present.get(s, ()) for fr in frames) >= args.min_num_images
        )
        random.Random(f"{args.seed}-{cat}").shuffle(seqs)
        half = len(seqs) // 2
        calib += seqs[:half]
        evaluation += seqs[half:]
        if seqs:
            print(f"{cat:14s} {len(seqs):4d} -> calib {half:3d}, eval {len(seqs) - half:3d}")

    os.makedirs(args.out_dir, exist_ok=True)
    for name, seqs in (("co3d_test_calib.txt", calib), ("co3d_test_eval.txt", evaluation)):
        with open(os.path.join(args.out_dir, name), "w") as f:
            f.write("\n".join(sorted(seqs)) + "\n")
    print(f"calibration {len(calib)}, evaluation {len(evaluation)}, overlap {len(set(calib) & set(evaluation))}")


if __name__ == "__main__":
    main()
