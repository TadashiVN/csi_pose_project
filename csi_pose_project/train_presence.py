"""
train_presence.py — Binary classifier: CSI -> co nguoi / khong co nguoi.
HO TRO MULTI-RX qua --rx. Undersampling 1:1.

Chay:
  python train_presence.py --data s01.json s02.json --val s03.json \
    --rx 3 --out models/presence_rx3.pth
"""

import argparse, json, os, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import CSIPresenceClassifier, SUBCARRIERS_PER_RX, WINDOW_SIZE
from train_model import stack_rx
from dataset_io import load_session

EPOCHS     = 30
BATCH_SIZE = 64
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


class PresenceDataset(Dataset):
    def __init__(self, json_files, n_rx=1, mean=None, std=None, balance=True):
        self.n_rx = n_rx
        raw_person, raw_empty = [], []
        for f in json_files:
            print(f"[DATA] Loading {f} (rx={n_rx})...")
            sess = load_session(f)
            csi_all, acts = sess["csi"], sess["activities"]
            for i in range(len(csi_all)):
                try:
                    csi = stack_rx(csi_all[i], n_rx)
                    if csi is None:
                        continue
                    (raw_empty if acts[i] == "empty" else raw_person).append(csi)
                except Exception:
                    continue

        print(f"[DATA] Truoc balance: person={len(raw_person)}, empty={len(raw_empty)}")
        if balance and raw_person and raw_empty:
            min_count = min(len(raw_person), len(raw_empty))
            np.random.seed(42)
            idx_p = np.random.choice(len(raw_person), min_count, replace=False)
            idx_e = np.random.choice(len(raw_empty),  min_count, replace=False)
            raw_person = [raw_person[i] for i in idx_p]
            raw_empty  = [raw_empty[i]  for i in idx_e]
            print(f"[DATA] Sau balance : person={min_count}, empty={min_count} — 1:1")

        self.samples = ([(c, 1) for c in raw_person] + [(c, 0) for c in raw_empty])
        np.random.shuffle(self.samples)
        print(f"[DATA] Tong: {len(self.samples)} samples")

        if mean is None:
            arr       = np.stack([s[0] for s in self.samples])
            self.mean = arr.mean(axis=(0, 1)).astype(np.float32)
            self.std  = (arr.std(axis=(0, 1)) + 1e-8).astype(np.float32)
        else:
            self.mean, self.std = mean, std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csi, label = self.samples[idx]
        csi = (csi - self.mean) / self.std
        return torch.from_numpy(csi), torch.tensor(label, dtype=torch.float32)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     nargs="+", required=True)
    parser.add_argument("--val",      nargs="+", default=None)
    parser.add_argument("--rx", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--out",      default="models/presence_model.pth")
    parser.add_argument("--epochs",   type=int,   default=EPOCHS)
    parser.add_argument("--batch",    type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",       type=float, default=LR)
    parser.add_argument("--patience", type=int,   default=10)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    subcarriers = args.rx * SUBCARRIERS_PER_RX
    print(f"\n[INFO] Device: {DEVICE} | RX={args.rx} | subcarriers={subcarriers}")

    train_ds = PresenceDataset(args.data, n_rx=args.rx, balance=True)
    mean, std = train_ds.mean, train_ds.std

    if args.val:
        val_ds = PresenceDataset(args.val, n_rx=args.rx, mean=mean, std=std, balance=False)
    else:
        n_val   = int(len(train_ds) * 0.2)
        n_train = len(train_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42))
        print(f"[SPLIT] Random: train={n_train}, val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)

    model     = CSIPresenceClassifier(subcarriers=subcarriers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_f1, best_state, no_improve = 0.0, None, 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train(); t_loss = 0.0
        for csi, label in train_loader:
            csi, label = csi.to(DEVICE), label.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(csi), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= max(1, len(train_loader))

        model.eval(); v_loss = 0.0; all_pred, all_label = [], []
        with torch.no_grad():
            for csi, label in val_loader:
                csi, label = csi.to(DEVICE), label.to(DEVICE).unsqueeze(1)
                pred = model(csi)
                v_loss += criterion(pred, label).item()
                all_pred.extend((pred > 0.5).cpu().numpy().flatten())
                all_label.extend(label.cpu().numpy().flatten())
        v_loss /= max(1, len(val_loader))

        all_pred, all_label = np.array(all_pred), np.array(all_label)
        acc = (all_pred == all_label).mean()
        tp = ((all_pred==1)&(all_label==1)).sum()
        fp = ((all_pred==1)&(all_label==0)).sum()
        fn = ((all_pred==0)&(all_label==1)).sum()
        precision = tp / max(tp+fp, 1)
        recall    = tp / max(tp+fn, 1)
        f1 = 2*precision*recall / max(precision+recall, 1e-8)
        empty_mask = all_label == 0
        empty_det = (all_pred[empty_mask]==0).mean() if empty_mask.sum()>0 else 0

        history["train_loss"].append(t_loss); history["val_loss"].append(v_loss)
        history["val_acc"].append(float(acc)); history["val_f1"].append(float(f1))

        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0; marker = " <- best"
        else:
            no_improve += 1

        print(f"Epoch {epoch:3d}/{args.epochs} | loss={t_loss:.4f}/{v_loss:.4f} | "
              f"acc={acc*100:.1f}% | F1={f1*100:.1f}% | empty_det={empty_det*100:.1f}% | "
              f"{time.time()-t0:.1f}s{marker}")

        if no_improve >= args.patience:
            print(f"\n[EARLY STOP] sau {args.patience} epochs."); break

    if best_state:
        model.load_state_dict(best_state)
    torch.save({
        "model_state": model.state_dict(), "mean": mean, "std": std,
        "val_f1": best_f1,
        "config": {"subcarriers": subcarriers, "n_rx": args.rx, "window_size": WINDOW_SIZE},
    }, args.out)
    print(f"\n[SAVED] {args.out}\n[BEST] F1={best_f1*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(history["train_loss"], label="Train"); axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot([a*100 for a in history["val_acc"]], label="Accuracy")
    axes[1].plot([a*100 for a in history["val_f1"]], label="F1")
    axes[1].set_ylim(0,100); axes[1].set_title("Val F1 & Acc"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout(); plt.savefig(str(Path(args.out).with_suffix(".png")), dpi=120)
    print(f"[SAVED] {Path(args.out).with_suffix('.png')}")


if __name__ == "__main__":
    train()
