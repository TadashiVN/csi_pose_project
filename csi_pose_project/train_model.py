"""
train_model.py — Train CSI → 2D pose keypoints.

Cải tiến so với bản gốc:
- Import model từ model.py (chống duplicate)
- Split-by-session: train trên session đầu, val trên session cuối → tránh leakage
- Data augmentation: gaussian noise, magnitude scale, subcarrier dropout
- Early stopping (patience configurable)
- Log PCK trong train loop (không chỉ MSE)
- Lưu metrics history vào JSON

Ví dụ:
  python train_model.py --data dataset/raw/session_01.json dataset/raw/session_02.json \
    --out models/pose_model.pth --split-by-session --augment --epochs 80 --patience 15
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from metrics import compute_metrics, format_metrics
from model import CSIPoseModel, N_KEYPOINTS, SUBCARRIERS, WINDOW_SIZE

EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
VALID_RATIO = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Dataset ─────────────────────────────────────────────────────────────────
class CSIPoseDataset(Dataset):
    """Load từ 1 hoặc nhiều file JSON. Trả về (csi, kps_flat)."""

    def __init__(self, json_files):
        self.samples = []
        self.source_per_sample = []  # để tracking khi split-by-session
        for f in json_files:
            print(f"[DATA] Loading {f}...")
            d = json.load(open(f, encoding="utf-8"))
            n_before = len(self.samples)
            for s in d["samples"]:
                try:
                    csi = np.array(s["csi"][0], dtype=np.float32)
                    kps = np.array(s["keypoints"], dtype=np.float32)
                    if csi.shape != (WINDOW_SIZE, SUBCARRIERS):
                        continue
                    if kps.shape != (N_KEYPOINTS, 2):
                        continue
                    self.samples.append((csi, kps))
                    self.source_per_sample.append(f)
                except Exception:
                    continue
            print(f"  → +{len(self.samples) - n_before} samples")
        print(f"[DATA] Tổng samples hợp lệ: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csi, kps = self.samples[idx]
        return torch.from_numpy(csi), torch.from_numpy(kps.flatten())

    def indices_by_source(self):
        """Return dict {file_path: [indices...]}"""
        by_src = {}
        for i, src in enumerate(self.source_per_sample):
            by_src.setdefault(src, []).append(i)
        return by_src


# ─── Preprocessing & Augmentation ────────────────────────────────────────────
def compute_stats_from_indices(dataset, indices):
    """Tính mean/std CHỈ trên train indices (tránh leakage val)."""
    arr = np.stack([dataset[i][0].numpy() for i in indices])
    mean = arr.mean(axis=(0, 1)).astype(np.float32)
    std = (arr.std(axis=(0, 1)) + 1e-8).astype(np.float32)
    return mean, std


class NormalizedDataset(Dataset):
    def __init__(self, base, mean, std, augment=False):
        self.base = base
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def _augment(self, csi):
        # csi shape (time=20, subcarriers=20), đã normalize
        # 1) Gaussian noise
        if torch.rand(1).item() < 0.7:
            csi = csi + torch.randn_like(csi) * 0.05
        # 2) Magnitude scaling per-sample
        if torch.rand(1).item() < 0.5:
            scale = 0.95 + torch.rand(1).item() * 0.10  # [0.95, 1.05]
            csi = csi * scale
        # 3) Subcarrier dropout (zero 1-2 subcarriers ngẫu nhiên)
        if torch.rand(1).item() < 0.3:
            n_drop = int(torch.randint(1, 3, (1,)).item())
            drop_idx = torch.randperm(csi.shape[1])[:n_drop]
            csi[:, drop_idx] = 0.0
        return csi

    def __getitem__(self, idx):
        csi, kps = self.base[idx]
        csi = (csi - self.mean) / self.std
        if self.augment:
            csi = self._augment(csi)
        return csi, kps


# ─── Loss ─────────────────────────────────────────────────────────────────────
def masked_mse(pred, target):
    """MSE bỏ qua samples empty (kps toàn 0)."""
    mask = (target.sum(dim=1, keepdim=True) > 0).float()
    if mask.sum() < 1:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    diff = (pred - target) * mask
    return (diff ** 2).sum() / (mask.sum() * pred.shape[1])


# ─── Eval helper ──────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_predict(model, loader, device):
    model.eval()
    preds, gts, losses = [], [], []
    for csi, kps in loader:
        csi, kps = csi.to(device), kps.to(device)
        out = model(csi)
        losses.append(masked_mse(out, kps).item())
        preds.append(out.cpu().numpy().reshape(-1, N_KEYPOINTS, 2))
        gts.append(kps.cpu().numpy().reshape(-1, N_KEYPOINTS, 2))
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    return preds, gts, float(np.mean(losses))


# ─── Train loop ───────────────────────────────────────────────────────────────
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True,
                        help="1 hoặc nhiều file JSON")
    parser.add_argument("--out", default="models/pose_model.pth")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--augment", action="store_true",
                        help="Bật data augmentation (khuyến khích)")
    parser.add_argument("--split-by-session", action="store_true",
                        help="File cuối làm val (chống leakage). Cần ≥ 2 file.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"\n[INFO] Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    base = CSIPoseDataset(args.data)
    if len(base) == 0:
        print("[ERROR] Không có sample hợp lệ!")
        return

    # Split
    if args.split_by_session and len(args.data) >= 2:
        by_src = base.indices_by_source()
        val_file = args.data[-1]
        val_idx = by_src.get(val_file, [])
        train_idx = [i for i in range(len(base)) if i not in set(val_idx)]
        print(f"[SPLIT] By-session: train={len(train_idx)} (file đầu), val={len(val_idx)} ({Path(val_file).name})")
    else:
        n_val = int(len(base) * VALID_RATIO)
        n_train = len(base) - n_val
        perm = torch.randperm(len(base), generator=torch.Generator().manual_seed(args.seed)).tolist()
        train_idx, val_idx = perm[:n_train], perm[n_train:]
        print(f"[SPLIT] Random {1-VALID_RATIO:.0%}/{VALID_RATIO:.0%}: train={n_train}, val={n_val}")
        print("[WARN] Random split có nguy cơ leakage giữa frame liền kề.")
        print("       Khuyến khích --split-by-session khi có ≥ 2 file.")

    # Stats CHỈ trên train
    print("[INFO] Tính normalization stats trên TRAIN...")
    mean, std = compute_stats_from_indices(base, train_idx)

    train_ds = NormalizedDataset(torch.utils.data.Subset(base, train_idx), mean, std,
                                  augment=args.augment)
    val_ds = NormalizedDataset(torch.utils.data.Subset(base, val_idx), mean, std,
                                augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    if args.augment:
        print("[INFO] Data augmentation: ON (gaussian noise + scale + subcarrier dropout)")

    # Model
    model = CSIPoseModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n[TRAIN] {args.epochs} epochs, patience={args.patience}\n")

    history = {"train_loss": [], "val_loss": [], "val_pck_0.1": [], "val_pck_0.2": []}
    best_val = float("inf")
    best_state = None
    best_metrics = None
    epochs_since_best = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        t_loss = 0.0
        t0 = time.time()
        for csi, kps in train_loader:
            csi, kps = csi.to(DEVICE), kps.to(DEVICE)
            optimizer.zero_grad()
            pred = model(csi)
            loss = masked_mse(pred, kps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= max(1, len(train_loader))

        # Val
        preds, gts, v_loss = eval_predict(model, val_loader, DEVICE)
        m = compute_metrics(preds, gts, thresholds=(0.1, 0.2, 0.5))
        pck10 = m.get("pck@0.1", 0.0)
        pck20 = m.get("pck@0.2", 0.0)

        scheduler.step()
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_pck_0.1"].append(pck10)
        history["val_pck_0.2"].append(pck20)

        dt = time.time() - t0
        marker = ""
        if v_loss < best_val:
            best_val = v_loss
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = m
            epochs_since_best = 0
            marker = " ← best"
        else:
            epochs_since_best += 1

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train={t_loss:.5f} | val={v_loss:.5f} | "
              f"PCK@0.1={pck10*100:5.1f}% | PCK@0.2={pck20*100:5.1f}% | "
              f"{dt:.1f}s{marker}")

        if epochs_since_best >= args.patience:
            print(f"\n[EARLY STOP] Không cải thiện sau {args.patience} epochs.")
            break

    # Restore best & save
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "model_state": model.state_dict(),
        "mean": mean,
        "std": std,
        "val_loss": best_val,
        "val_metrics": best_metrics,
        "config": {
            "subcarriers": SUBCARRIERS,
            "window_size": WINDOW_SIZE,
            "n_keypoints": N_KEYPOINTS,
            "augment": args.augment,
            "split_by_session": args.split_by_session,
        },
    }, args.out)
    print(f"\n[SAVED] {args.out}")

    # Print full metrics
    if best_metrics is not None:
        print(format_metrics(best_metrics))

    # Save history + plot
    hist_path = str(Path(args.out).with_suffix(".history.json"))
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "best_val_loss": best_val,
            "best_metrics": best_metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {hist_path}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot([p * 100 for p in history["val_pck_0.1"]], label="PCK@0.1")
    axes[1].plot([p * 100 for p in history["val_pck_0.2"]], label="PCK@0.2")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("PCK (%)")
    axes[1].set_title("Validation PCK"); axes[1].legend(); axes[1].grid(True)

    plot_path = str(Path(args.out).with_suffix(".png"))
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    print(f"[SAVED] {plot_path}")


if __name__ == "__main__":
    train()
