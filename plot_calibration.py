# Copyright (c) 2026 Leonardo Vanni. CC BY-NC 4.0 (see LICENSE.txt).
"""Calibration analysis of the predicted pose covariances.

The validation loop saves, for every evaluated (non-reference) frame, the squared
Mahalanobis distance d^2, the log-determinant of the covariance and the translation and
rotation error to ``logs/<exp>/calibration_epoch_<E>.npz``.  Under a calibrated 6-D
Gaussian, d^2 ~ chi^2_6.  This script

* fits a scalar temperature T (Sigma_calib = T Sigma_raw) on a *calibration* file by
  matching the median: T = median(d^2) / median(chi^2_6).  A few gross VGGT failures
  (errors of tens of degrees) have d^2 in the thousands; the Gaussian maximum-likelihood
  fit T = mean(d^2) / 6 is dominated by them and is reported only for reference;
* evaluates T = 1 and the fitted T on a separate *test* file, so the reported numbers
  are out of sample: NLL (mean and median), coverage of the 50/68/95/99% regions and
  the pose errors of the frames outside the 99% region;
* writes the coverage plot and a JSON summary.

    python plot_calibration.py --calib logs/eval_calib/calibration_epoch_0.npz \\
        --test logs/eval_eval/calibration_epoch_0.npz --out calibration.png

A plain ``.npy`` of d^2 values (older runs) is accepted; the NLL is then not reported.
"""
import argparse
import json
import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import chi2, spearmanr  # noqa: E402

DOF = 6
LEVELS = (0.50, 0.68, 0.95, 0.99)


def load(path):
    if path.endswith(".npy"):
        data = {"mahalanobis_sq": np.load(path)}
    else:
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
    data = {k: v.astype(np.float64) for k, v in data.items()}
    keep = np.isfinite(data["mahalanobis_sq"])
    return {k: v[keep] for k, v in data.items()}


def fit_temperature_median(d2):
    return float(np.median(d2) / chi2.median(DOF))


def fit_temperature_mle(d2):
    """Gaussian NLL is minimised at T = mean(d^2) / dof."""
    return float(np.mean(d2) / DOF)


def nll_per_frame(data, T):
    """0.5 (d^2/T + log det(T Sigma) + 6 log 2 pi), the training NLL at temperature T."""
    return 0.5 * (data["mahalanobis_sq"] / T + data["log_det"] + DOF * math.log(T) + DOF * math.log(2 * math.pi))


def summarize(data, T):
    d2 = data["mahalanobis_sq"] / T
    out = {"T": T, "n": int(d2.size), "median_d2": float(np.median(d2)), "mean_d2": float(np.mean(d2))}
    for p in LEVELS:
        out[f"coverage_{int(p * 100)}"] = float(np.mean(d2 <= chi2.ppf(p, DOF)))
    out["frac_outside_99"] = 1.0 - out["coverage_99"]
    if "log_det" in data:
        nll = nll_per_frame(data, T)
        out["nll_mean"], out["nll_median"] = float(np.mean(nll)), float(np.median(nll))
    if "rot_err_deg" in data:
        tail = d2 > chi2.ppf(0.99, DOF)
        for name, sel in (("inside_99", ~tail), ("outside_99", tail)):
            if sel.any():
                out[f"{name}_median_rot_err_deg"] = float(np.median(data["rot_err_deg"][sel]))
                out[f"{name}_median_trans_err"] = float(np.median(data["trans_err"][sel]))
        out["outside_99_frac_rot_err_above_5deg"] = float(np.mean(data["rot_err_deg"][tail] > 5.0)) if tail.any() else 0.0
    return out


def by_predicted_uncertainty(data, T, groups=8):
    """Frames split into equal groups by log det of the predicted covariance (Table 2 of the thesis).

    Grouping by the prediction, not by the actual error: frames selected for a large error
    have a large d^2 even under a perfectly calibrated model.
    """
    d2, ld = data["mahalanobis_sq"] / T, data["log_det"]
    edges = np.quantile(ld, np.linspace(0, 1, groups + 1))
    rows = []
    for g, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        s = (ld >= lo) & (ld <= hi)
        rows.append({"group": g + 1, "median_rot_err_deg": float(np.median(data["rot_err_deg"][s])),
                     "median_d2": float(np.median(d2[s])), "outside_99": float(np.mean(d2[s] > chi2.ppf(0.99, DOF)))})
    return rows


def coverage_error(d2, levels=np.linspace(0.02, 0.99, 50)):
    return float(np.mean(np.abs((d2[:, None] <= chi2.ppf(levels, DOF)).mean(0) - levels)))


def fit_logdet_temperature(calib):
    """Per-frame temperature T_i = exp(a + b log det Sigma_i / 6), fitted on the coverage curve.
    b = 0 is plain temperature scaling; b > 0 would stretch the range of predicted uncertainties."""
    ref = float(np.median(calib["log_det"]))
    temps = lambda z, a, b: np.exp(a + b * (z["log_det"] - ref) / DOF)  # noqa: E731
    a, b = minimize(lambda p: coverage_error(calib["mahalanobis_sq"] / temps(calib, *p)), [0.0, 0.0], method="Nelder-Mead").x
    return float(a), float(b), lambda z: temps(z, a, b)


def coverage_curve(d2, T, levels=np.linspace(0.0, 1.0, 201)):
    return levels, np.array([np.mean(d2 / T <= chi2.ppf(p, DOF)) if p < 1 else 1.0 for p in levels])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True, help="calibration_epoch_*.npz used to fit T")
    ap.add_argument("--test", default=None, help="held-out calibration_epoch_*.npz to evaluate on (default: --calib, in-sample)")
    ap.add_argument("--out", default="calibration_plot.png")
    args = ap.parse_args()

    calib = load(args.calib)
    test = load(args.test) if args.test else calib
    T = fit_temperature_median(calib["mahalanobis_sq"])
    summary = {
        "calibration_file": args.calib, "test_file": args.test or args.calib, "in_sample": args.test is None,
        "T_median": T, "T_mle_for_reference": fit_temperature_mle(calib["mahalanobis_sq"]),
        "raw": summarize(test, 1.0), "calibrated": summarize(test, T),
    }
    if "log_det" in test and "rot_err_deg" in test:
        d2 = test["mahalanobis_sq"] / T
        summary["spearman_logdet_rot_err"] = float(spearmanr(test["log_det"], test["rot_err_deg"])[0])
        summary["spearman_logdet_trans_err"] = float(spearmanr(test["log_det"], test["trans_err"])[0])
        big = test["rot_err_deg"] > 5.0
        summary["frac_frames_rot_err_above_5deg"] = float(big.mean())
        summary["share_of_sum_d2_from_rot_err_above_5deg"] = float(d2[big].sum() / d2.sum())
        summary["groups_by_predicted_uncertainty"] = by_predicted_uncertainty(test, T)
        a, b, temps = fit_logdet_temperature(calib)
        summary["logdet_temperature"] = {"a": a, "b": b, "coverage_error_test": coverage_error(test["mahalanobis_sq"] / temps(test)),
                                         "coverage_error_test_single_T": coverage_error(d2)}
    print(json.dumps(summary, indent=2))
    with open(args.out.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
    for temp, label in [(1.0, "Raw ($T = 1$)"), (T, f"Calibrated ($T = {T:.2f}$)")]:
        x, y = coverage_curve(test["mahalanobis_sq"], temp)
        plt.plot(x, y, linewidth=2, label=label)
    plt.xlabel("Nominal coverage of the $\\chi^2_6$ region")
    plt.ylabel("Fraction of frames inside the region")
    if args.test is None:
        plt.title("in-sample")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal")
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
