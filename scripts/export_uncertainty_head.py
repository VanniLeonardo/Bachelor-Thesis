# Copyright (c) 2026 Leonardo Vanni. CC BY-NC 4.0 (see LICENSE.txt).
"""Export the head-only release file from a trainer checkpoint.

    python scripts/export_uncertainty_head.py logs/v1.0.0/ckpts/checkpoint.pt vggt_uncertainty_head_v1.pt \
        --temperature 1.015 --kappa 10 --shield-eps 1e-4

The output holds the covariance branch (fp16 by default) and the metadata consumers need
(kappa, shield eps, error convention, temperature, source checkpoint).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vggt.utils.checkpoint import export_uncertainty_head  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="trainer checkpoint (contains 'model')")
    ap.add_argument("out", help="output .pt file")
    ap.add_argument("--temperature", type=float, default=1.0, help="post-hoc temperature fitted on held-out data")
    ap.add_argument("--kappa", type=float, default=10.0, help="rotation weight used in training")
    ap.add_argument("--shield-eps", type=float, default=1e-4, help="shield epsilon used in training")
    ap.add_argument("--convention", default="body", choices=["body", "world"], help="error convention used in training")
    ap.add_argument("--fp32", action="store_true", help="keep float32 weights (default: fp16)")
    args = ap.parse_args()
    meta = {"kappa": args.kappa, "shield_eps": args.shield_eps, "error_convention": args.convention, "temperature": args.temperature}
    out = export_uncertainty_head(args.checkpoint, args.out, meta=meta, half=not args.fp32)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
