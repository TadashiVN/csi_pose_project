"""
train_model.py — Train CSI -> 2D pose keypoints. HO TRO MULTI-RX.

MOI: --rx {1,2,3}
  Chon dung BAO NHIEU RX nodes lay tu data (concat theo subcarrier).
  Vi du cung 1 bo data 3-RX, chay 3 lan voi --rx 1 / --rx 2 / --rx 3
  -> ra bang so sanh 1/2/3 RX SACH (cung data, khong domain mismatch).
  subcarriers = rx * 52.

Vi du:
  python train_model.py \
    --data dataset/raw/session_01.json dataset/raw/session_02.json \
    --rx 3 --split-by-session --augment --epochs 80 --patience 15 \
    --out models/pose_rx3.pth
"""

import argparse, copy, json, os, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from metrics import compute_metrics, format_metrics
from model import CSIPoseModel, N_KEYPOINTS, SUBCARRIERS_PER_RX, WINDOW_SIZE
from dataset_io import load_session

EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 1e-3
VALID_RATIO = 0.2
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# --- Helper: gop nhieu RX thanh 1 window (20, rx*52) -------------------------
def stack_rx(csi_raw, n_rx):
    """csi_raw: list/array (nodes, 20, 52) -> (20, n_rx*52) hoac None neu thieu node."""
    arr = np.asarray(csi_raw, dtype=np.float32)
    if arr.ndim == 2:                      # data cu luu (20,52) -> coi nhu 1 RX
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[0] < n_rx:
        return None
    arr = arr[:n_rx]                       # (n_rx, 20, 52)
    if arr.shape[1] != WINDOW_SIZE or arr.shape[2] != SUBCARRIERS_PER_RX:
        return None
    arr = np.transpose(arr, (1, 0, 2))     # (20, n_rx, 52)
    return arr.reshape(WINDOW_SIZE, n_rx * SUBCARRIERS_PER_RX)


# --- Dataset -----------------------------------------------------------------
class CSIPoseDataset(Dataset):
    def __init__(self, json_files, n_rx=1):
        self.n_rx = n_rx
        self.subcarriers = n_rx * SUBCARRIERS_PER_RX
        self.samples = []
        self.source_per_sample = []
        for f in json_files:
            print(f"[DATA] Loading {f} (rx={n_rx})...")
            sess = load_session(f)
            csi_all, kps_all = sess["csi"], sess["keypoints"]
            n_before = len(self.samples)
            for i in range(len(csi_all)):
                try:
                    csi = stack_rx(csi_all[i], n_rx)
                    if csi is None:
                        continue
                    kps = kps_all[i].astype(np.float32)
                    if kps.shape != (N_KEYPOINTS, 2):
                        continue
                    self.samples.append((csi, kps))
                    self.source_per_sample.append(f)
                except Exception:
                    continue
            print(f"  -> +{len(self.samples) - n_before} samples")
        print(f"[DATA] Tong samples hop le: {len(self.samples)} | subcarriers={self.subcarriers}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csi, kps = self.samples[idx]
        return torch.from_numpy(csi), torch.from_numpy(kps.flatten())

    def indices_by_source(self):
        by_src = {}
        for i, src in enumerate(self.source_per_sample):
            by_src.setdefault(src, []).append(i)
        return by_src


def compute_stats_from_indices(dataset, indices):
    arr  = np.stack([dataset[i][0].numpy() for i in indices])
    mean = arr.mean(axis=(0, 1)).astype(np.float32)
    std  = (arr.std(axis=(0, 1)) + 1e-8).astype(np.float32)
    return mean, std


class NormalizedDataset(Dataset):
    def __init__(self, base, mean, std, augment=False):
        self.base    = base
        self.mean    = torch.from_numpy(mean)
        self.std     = torch.from_numpy(std)
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def _augment(self, csi):
        if torch.rand(1).item() < 0.7:
            csi = csi + torch.randn_like(csi) * 0.05
        if torch.rand(1).item() < 0.5:
            scale = 0.95 + torch.rand(1).item() * 0.10
            csi = csi * scale
        if torch.rand(1).item() < 0.3:
            n_drop   = int(torch.randint(1, 3, (1,)).item())
            drop_idx = torch.randperm(csi.shape[1])[:n_drop]
            csi[:, drop_idx] = 0.0
        return csi

    def __getitem__(self, idx):
        csi, kps = self.base[idx]
        csi = (csi - self.mean) / self.std
        if self.augment:
            csi = self._augment(csi)
        return csi, kps


# --- Loss --------------------------------------------------------------------
def masked_mse(pred, target):
    mask = (target.sum(dim=1, keepdim=True) > 0).float()
    if mask.sum() < 1:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    diff = (pred - target) * mask
    return (diff ** 2).sum() / (mask.sum() * pred.shape[1])


# --- Eval --------------------------------------------------------------------
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
    gts   = np.concatenate(gts,   axis=0)
    return preds, gts, float(np.mean(losses))


# --- Core training loop ------------------------------------------------------
def run_training(args, train_loader, val_loader, mean, std, subcarriers):
    model     = CSIPoseModel(subcarriers=subcarriers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n[TRAIN] {args.epochs} epochs, patience={args.patience}, subcarriers={subcarriers}\n")

    history = {"train_loss": [], "val_loss": [], "val_pck_0.1": [], "val_pck_0.2": []}
    best_val   = float("inf")
    best_state = None
    best_metrics = None
    epochs_since_best = 0

    for epoch in range(1, args.epochs + 1):
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

        preds, gts, v_loss = eval_predict(model, val_loader, DEVICE)
        m     = compute_metrics(preds, gts, thresholds=(0.1, 0.2, 0.5))
        pck10 = m.get("pck@0.1", 0.0)
        pck20 = m.get("pck@0.2", 0.0)

        scheduler.step()
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_pck_0.1"].append(pck10)
        history["val_pck_0.2"].append(pck20)

        dt     = time.time() - t0
        marker = ""
        if v_loss < best_val:
            best_val   = v_loss
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = m
            epochs_since_best = 0
            marker = " <- best"
        else:
            epochs_since_best += 1

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train={t_loss:.5f} | val={v_loss:.5f} | "
              f"PCK@0.1={pck10*100:5.1f}% | PCK@0.2={pck20*100:5.1f}% | "
              f"{dt:.1f}s{marker}")

        if epochs_since_best >= args.patience:
            print(f"\n[EARLY STOP] Khong cai thien sau {args.patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "model_state":  model.state_dict(),
        "mean":         mean,
        "std":          std,
        "val_loss":     best_val,
        "val_metrics":  best_metrics,
        "config": {
            "subcarriers": subcarriers,
            "n_rx":        args.rx,
            "window_size": WINDOW_SIZE,
            "n_keypoints": N_KEYPOINTS,
            "augment":     args.augment,
        },
    }, args.out)
    print(f"\n[SAVED] {args.out}")

    if best_metrics is not None:
        print(format_metrics(best_metrics))

    hist_path = str(Path(args.out).with_suffix(".history.json"))
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_val_loss": best_val,
                   "best_metrics": best_metrics}, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {hist_path}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot([p*100 for p in history["val_pck_0.1"]], label="PCK@0.1")
    axes[1].plot([p*100 for p in history["val_pck_0.2"]], label="PCK@0.2")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("PCK (%)")
    axes[1].set_title("Validation PCK"); axes[1].legend(); axes[1].grid(True)
    plot_path = str(Path(args.out).with_suffix(".png"))
    plt.tight_layout(); plt.savefig(plot_path, dpi=120)
    print(f"[SAVED] {plot_path}")


# --- Main --------------------------------------------------------------------
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--val",  nargs="+", default=None,
                        help="File JSON val rieng — KHUYEN KHICH")
    parser.add_argument("--rx", type=int, default=1, choices=[1, 2, 3],
                        help="So RX nodes dung (concat subcarrier). subcarriers = rx*52.")
    parser.add_argument("--out",      default="models/pose_model.pth")
    parser.add_argument("--epochs",   type=int,   default=EPOCHS)
    parser.add_argument("--batch",    type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",       type=float, default=LR)
    parser.add_argument("--patience", type=int,   default=15)
    parser.add_argument("--augment",  action="store_true")
    parser.add_argument("--split-by-session", action="store_true",
                        help="File cuoi trong --data lam val.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    subcarriers = args.rx * SUBCARRIERS_PER_RX
    print(f"\n[INFO] Device: {DEVICE} | RX={args.rx} | subcarriers={subcarriers}")
    if DEVICE == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # Mode 1: --val rieng
    if args.val:
        print("\n[MODE] Explicit val files")
        train_base = CSIPoseDataset(args.data, n_rx=args.rx)
        val_base   = CSIPoseDataset(args.val,  n_rx=args.rx)
        if len(train_base) == 0 or len(val_base) == 0:
            print("[ERROR] Khong co sample hop le!"); return
        print(f"\n[SPLIT] train={len(train_base)} | val={len(val_base)}")
        mean, std = compute_stats_from_indices(train_base, list(range(len(train_base))))
        train_ds = NormalizedDataset(train_base, mean, std, augment=args.augment)
        val_ds   = NormalizedDataset(val_base,   mean, std, augment=False)

    # Mode 2: split-by-session
    elif args.split_by_session and len(args.data) >= 2:
        print("\n[MODE] Split by session (file cuoi = val)")
        base = CSIPoseDataset(args.data, n_rx=args.rx)
        if len(base) == 0:
            print("[ERROR] Khong co sample hop le!"); return
        by_src    = base.indices_by_source()
        val_file  = args.data[-1]
        val_idx   = by_src.get(val_file, [])
        train_idx = [i for i in range(len(base)) if i not in set(val_idx)]
        print(f"[SPLIT] train={len(train_idx)} | val={len(val_idx)} ({Path(val_file).name})")
        mean, std = compute_stats_from_indices(base, train_idx)
        train_ds = NormalizedDataset(torch.utils.data.Subset(base, train_idx), mean, std, augment=args.augment)
        val_ds   = NormalizedDataset(torch.utils.data.Subset(base, val_idx),   mean, std, augment=False)

    # Mode 3: random split
    else:
        print("\n[MODE] Random split (fallback)")
        print("[WARN] Khuyen nghi dung --split-by-session hoac --val")
        base = CSIPoseDataset(args.data, n_rx=args.rx)
        if len(base) == 0:
            print("[ERROR] Khong co sample hop le!"); return
        n_val   = int(len(base) * VALID_RATIO)
        n_train = len(base) - n_val
        perm    = torch.randperm(len(base), generator=torch.Generator().manual_seed(args.seed)).tolist()
        train_idx, val_idx = perm[:n_train], perm[n_train:]
        print(f"[SPLIT] train={n_train} | val={n_val}")
        mean, std = compute_stats_from_indices(base, train_idx)
        train_ds = NormalizedDataset(torch.utils.data.Subset(base, train_idx), mean, std, augment=args.augment)
        val_ds   = NormalizedDataset(torch.utils.data.Subset(base, val_idx),   mean, std, augment=False)

    if args.augment:
        print("[INFO] Data augmentation: ON")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    run_training(args, train_loader, val_loader, mean, std, subcarriers)


if __name__ == "__main__":
    train()
