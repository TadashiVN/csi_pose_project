"""
calibrate_threshold.py — Tự động tìm threshold tối ưu cho empty-detection.

Đọc dataset JSON (đã có nhãn activity bao gồm 'empty'), tính CV (Coefficient of
Variation) của từng CSI window, plot phân bố CV cho empty vs non-empty,
và đề xuất threshold tối ưu theo Youden's J (max(TPR - FPR)).

Chạy:
  python calibrate_threshold.py --data dataset/raw/session_01.json dataset/raw/session_02.json
  python calibrate_threshold.py --data ... --out calib.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def cv_of_window(csi):
    """csi: (time, subcarriers) — return mean coefficient of variation."""
    arr = np.asarray(csi, dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    # tránh chia 0
    cv = std / (np.abs(mean) + 1e-8)
    return float(np.mean(cv))


def variance_of_window(csi):
    """Để so sánh — variance trung bình theo time."""
    arr = np.asarray(csi, dtype=np.float32)
    return float(np.mean(np.var(arr, axis=0)))


def find_optimal_threshold(empty_vals, nonempty_vals):
    """
    Tìm threshold tối ưu theo Youden's J statistic = TPR - FPR.
    Class quy ước: "non-empty" = positive.
    Empty CV thấp, non-empty CV cao → predict positive khi CV > thr.
    """
    all_vals = np.concatenate([empty_vals, nonempty_vals])
    candidates = np.linspace(all_vals.min(), all_vals.max(), 500)

    best_j = -1.0
    best_thr = candidates[0]
    for thr in candidates:
        tp = np.sum(nonempty_vals > thr)
        fn = np.sum(nonempty_vals <= thr)
        fp = np.sum(empty_vals > thr)
        tn = np.sum(empty_vals <= thr)
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_thr = float(thr)
    return best_thr, best_j


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", default="calibration_plot.png")
    parser.add_argument("--metric", choices=["cv", "var"], default="cv",
                        help="cv = std/mean (khuyến khích), var = variance trung bình")
    args = parser.parse_args()

    fn = cv_of_window if args.metric == "cv" else variance_of_window
    label = "CV" if args.metric == "cv" else "Variance"

    empty_vals = []
    nonempty_vals = []

    for path in args.data:
        print(f"[DATA] Loading {path}...")
        d = json.load(open(path, encoding="utf-8"))
        for s in d["samples"]:
            try:
                csi = s["csi"][0]
                v = fn(csi)
                if s["activity"] == "empty":
                    empty_vals.append(v)
                else:
                    nonempty_vals.append(v)
            except Exception:
                continue

    empty_vals = np.array(empty_vals)
    nonempty_vals = np.array(nonempty_vals)

    print(f"\n[STATS] Empty samples:     {len(empty_vals):5d} | "
          f"{label} mean={empty_vals.mean():.5f} std={empty_vals.std():.5f}")
    print(f"[STATS] Non-empty samples: {len(nonempty_vals):5d} | "
          f"{label} mean={nonempty_vals.mean():.5f} std={nonempty_vals.std():.5f}")

    if len(empty_vals) == 0 or len(nonempty_vals) == 0:
        print("[!] Cần cả empty và non-empty samples để calibrate.")
        return

    thr, j = find_optimal_threshold(empty_vals, nonempty_vals)

    # Confusion ở threshold tối ưu
    tp = np.sum(nonempty_vals > thr)
    fp = np.sum(empty_vals > thr)
    tn = np.sum(empty_vals <= thr)
    fn = np.sum(nonempty_vals <= thr)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    print(f"\n[OPTIMAL] threshold = {thr:.5f} (Youden J = {j:.3f})")
    print(f"[RULE]    if {label} > {thr:.5f}  →  có người")
    print(f"          if {label} ≤ {thr:.5f}  →  empty")
    print(f"[CONFUSION] TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"[METRICS]   accuracy={accuracy:.3f} precision={precision:.3f} recall={recall:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5))
    bins = np.linspace(min(empty_vals.min(), nonempty_vals.min()),
                       max(empty_vals.max(), nonempty_vals.max()), 60)
    ax.hist(empty_vals, bins=bins, alpha=0.6, label=f"Empty (n={len(empty_vals)})",
            color="steelblue", edgecolor="black", linewidth=0.3)
    ax.hist(nonempty_vals, bins=bins, alpha=0.6, label=f"Non-empty (n={len(nonempty_vals)})",
            color="tomato", edgecolor="black", linewidth=0.3)
    ax.axvline(thr, color="black", linestyle="--", linewidth=2,
               label=f"Optimal threshold = {thr:.4f}")
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(f"CSI {label} distribution — Empty vs Non-empty\n"
                 f"Youden J = {j:.3f}, accuracy = {accuracy:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\n[SAVED] {args.out}")

    # Save threshold để realtime_pose.py đọc
    cfg_path = Path(args.out).with_suffix(".json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "metric": args.metric,
            "threshold": thr,
            "youden_j": j,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "n_empty": int(len(empty_vals)),
            "n_nonempty": int(len(nonempty_vals)),
        }, f, indent=2)
    print(f"[SAVED] {cfg_path}")


if __name__ == "__main__":
    main()
