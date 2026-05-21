"""
evaluate.py — Đánh giá model trên test set, output PCK + per-keypoint + plots.

Chạy:
  python evaluate.py --model models/pose_model.pth --data dataset/raw/session_02.json
  python evaluate.py --model models/pose_model.pth --data dataset/raw/session_02.json --save-plots eval_out/

Output:
  - In ra console: bảng metrics (PCK, MPJPE, per-keypoint error)
  - File JSON metrics
  - Plot per-keypoint error bar chart
  - Plot vài sample prediction vs ground truth
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from metrics import KEYPOINT_NAMES, compute_metrics, format_metrics
from model import N_KEYPOINTS, SUBCARRIERS, WINDOW_SIZE, load_checkpoint
from train_model import CSIPoseDataset, NormalizedDataset, eval_predict


def plot_per_keypoint_error(metrics, out_path):
    err_norm = metrics["per_keypoint_error_normalized"]
    pck10 = metrics.get("pck@0.1_per_keypoint", {})
    pck20 = metrics.get("pck@0.2_per_keypoint", {})

    names = list(err_norm.keys())
    errs = [err_norm[n] for n in names]
    pcks10 = [pck10.get(n, 0.0) * 100 for n in names]
    pcks20 = [pck20.get(n, 0.0) * 100 for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].barh(names, errs, color="tomato")
    axes[0].set_xlabel("Error / torso diagonal")
    axes[0].set_title("Per-keypoint normalized error (lower = better)")
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[0].invert_yaxis()

    y = np.arange(len(names))
    w = 0.4
    axes[1].barh(y - w/2, pcks10, w, label="PCK@0.1", color="steelblue")
    axes[1].barh(y + w/2, pcks20, w, label="PCK@0.2", color="seagreen")
    axes[1].set_yticks(y); axes[1].set_yticklabels(names)
    axes[1].set_xlabel("PCK (%)")
    axes[1].set_title("Per-keypoint PCK (higher = better)")
    axes[1].set_xlim(0, 100)
    axes[1].legend()
    axes[1].grid(True, axis="x", alpha=0.3)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_sample_predictions(preds, gts, out_path, n_samples=6):
    """Vẽ skeleton GT vs Pred cho vài sample bất kỳ."""
    SKELETON = [
        (0,1),(0,2),(1,2),
        (1,3),(2,4),(3,5),(4,6),
        (1,7),(2,8),(7,8),
        (7,9),(8,10),(9,11),(10,12),
        (0,13),(0,14),(13,15),(14,16),
    ]
    valid_mask = gts.reshape(len(gts), -1).sum(axis=1) > 0
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        print("[WARN] Không có sample valid để plot")
        return

    chosen = np.random.RandomState(0).choice(valid_idx,
                                              size=min(n_samples, len(valid_idx)),
                                              replace=False)
    n = len(chosen)
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax_i, idx in enumerate(chosen):
        ax = axes[ax_i]
        gt = gts[idx]; pr = preds[idx]
        # GT
        for a, b in SKELETON:
            ax.plot([gt[a, 0], gt[b, 0]], [gt[a, 1], gt[b, 1]], "g-", lw=2, alpha=0.7)
        ax.scatter(gt[:, 0], gt[:, 1], c="g", s=30, label="GT")
        # Pred
        for a, b in SKELETON:
            ax.plot([pr[a, 0], pr[b, 0]], [pr[a, 1], pr[b, 1]], "r--", lw=2, alpha=0.7)
        ax.scatter(pr[:, 0], pr[:, 1], c="r", s=30, label="Pred")
        ax.invert_yaxis()
        ax.set_xlim(0, 1); ax.set_ylim(1, 0)
        ax.set_aspect("equal"); ax.set_title(f"Sample #{idx}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[SAVED] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", nargs="+", required=True,
                        help="File JSON test set (KHÔNG nên cùng file dùng train)")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--save-plots", default=None,
                        help="Thư mục lưu plots (default: cạnh model)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model:  {args.model}")
    print(f"[INFO] Data:   {args.data}")

    model, mean, std, ckpt = load_checkpoint(args.model, device=device)
    print(f"[INFO] Loaded model (val_loss khi train: {ckpt.get('val_loss', 'N/A')})")

    base = CSIPoseDataset(args.data)
    if len(base) == 0:
        print("[ERROR] Không có sample hợp lệ")
        return

    ds = NormalizedDataset(base, mean, std, augment=False)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    print(f"[EVAL] {len(ds)} samples...")
    preds, gts, mse_loss = eval_predict(model, loader, device)
    print(f"[EVAL] Masked MSE: {mse_loss:.5f}")

    m = compute_metrics(preds, gts)
    print()
    print(format_metrics(m))

    # Save outputs
    out_dir = Path(args.save_plots) if args.save_plots else Path(args.model).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.model).stem + "_eval"
    metrics_path = out_dir / f"{stem}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {metrics_path}")

    plot_per_keypoint_error(m, out_dir / f"{stem}_per_keypoint.png")
    plot_sample_predictions(preds, gts, out_dir / f"{stem}_samples.png", n_samples=6)


if __name__ == "__main__":
    main()
