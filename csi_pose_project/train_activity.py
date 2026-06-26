"""
train_activity.py — Train CSI -> activity (walk/stand/sit/fall/empty).
KET QUA NEN CHAC AN cua do an: activity classification tren CSI ESP32
thuong dat 80-95% acc, du suc tra loi cau hoi hoi dong.

HO TRO MULTI-RX qua --rx. In confusion matrix + per-class accuracy.

Chay:
  python train_activity.py --data s01.json s02.json \
    --rx 3 --split-by-session --augment --epochs 60 --out models/activity_rx3.pth
"""

import argparse, copy, json, os, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import CSIActivityClassifier, ACTIVITY_NAMES, SUBCARRIERS_PER_RX, WINDOW_SIZE
from train_model import stack_rx
from dataset_io import load_session

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACT2IDX = {a: i for i, a in enumerate(ACTIVITY_NAMES)}


class ActivityDataset(Dataset):
    def __init__(self, json_files, n_rx=1):
        self.n_rx = n_rx
        self.samples, self.source_per_sample = [], []
        for f in json_files:
            print(f"[DATA] Loading {f} (rx={n_rx})...")
            sess = load_session(f)
            csi_all, acts = sess["csi"], sess["activities"]
            n0 = len(self.samples)
            for i in range(len(csi_all)):
                try:
                    csi = stack_rx(csi_all[i], n_rx)
                    if csi is None or acts[i] not in ACT2IDX:
                        continue
                    self.samples.append((csi, ACT2IDX[acts[i]]))
                    self.source_per_sample.append(f)
                except Exception:
                    continue
            print(f"  -> +{len(self.samples)-n0}")
        from collections import Counter
        dist = Counter(ACTIVITY_NAMES[l] for _, l in self.samples)
        print(f"[DATA] Tong: {len(self.samples)} | {dict(dist)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        csi, label = self.samples[i]
        return torch.from_numpy(csi), torch.tensor(label, dtype=torch.long)

    def indices_by_source(self):
        by = {}
        for i, s in enumerate(self.source_per_sample):
            by.setdefault(s, []).append(i)
        return by


class Norm(Dataset):
    def __init__(self, base, mean, std, augment=False):
        self.base, self.mean, self.std = base, torch.from_numpy(mean), torch.from_numpy(std)
        self.augment = augment
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        csi, label = self.base[i]
        csi = (csi - self.mean) / self.std
        if self.augment and torch.rand(1).item() < 0.7:
            csi = csi + torch.randn_like(csi) * 0.05
        return csi, label


def stats(ds, idx):
    arr = np.stack([ds[i][0].numpy() for i in idx])
    return arr.mean(axis=(0,1)).astype(np.float32), (arr.std(axis=(0,1))+1e-8).astype(np.float32)


def train():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--val",  nargs="+", default=None)
    p.add_argument("--rx", type=int, default=1, choices=[1,2,3])
    p.add_argument("--out", default="models/activity_model.pth")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--split-by-session", action="store_true")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    subcarriers = args.rx * SUBCARRIERS_PER_RX
    print(f"\n[INFO] Device {DEVICE} | RX={args.rx} | subcarriers={subcarriers}")

    if args.val:
        tb = ActivityDataset(args.data, args.rx); vb = ActivityDataset(args.val, args.rx)
        mean, std = stats(tb, range(len(tb)))
        train_ds, val_ds = Norm(tb, mean, std, args.augment), Norm(vb, mean, std)
    elif args.split_by_session and len(args.data) >= 2:
        base = ActivityDataset(args.data, args.rx)
        vi = base.indices_by_source().get(args.data[-1], [])
        ti = [i for i in range(len(base)) if i not in set(vi)]
        print(f"[SPLIT] train={len(ti)} val={len(vi)}")
        mean, std = stats(base, ti)
        train_ds = Norm(torch.utils.data.Subset(base, ti), mean, std, args.augment)
        val_ds   = Norm(torch.utils.data.Subset(base, vi), mean, std)
    else:
        base = ActivityDataset(args.data, args.rx)
        n_val = int(len(base)*0.2)
        perm = torch.randperm(len(base), generator=torch.Generator().manual_seed(42)).tolist()
        ti, vi = perm[n_val:], perm[:n_val]
        mean, std = stats(base, ti)
        train_ds = Norm(torch.utils.data.Subset(base, ti), mean, std, args.augment)
        val_ds   = Norm(torch.utils.data.Subset(base, vi), mean, std)

    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = CSIActivityClassifier(subcarriers=subcarriers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_acc, best_state, no_imp = 0.0, None, 0
    for epoch in range(1, args.epochs+1):
        t0 = time.time(); model.train(); tloss = 0
        for csi, y in tl:
            csi, y = csi.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(csi), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tloss += loss.item()
        tloss /= max(1, len(tl))

        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for csi, y in vl:
                out = model(csi.to(DEVICE))
                preds.extend(out.argmax(1).cpu().numpy()); gts.extend(y.numpy())
        preds, gts = np.array(preds), np.array(gts)
        acc = (preds == gts).mean() if len(gts) else 0

        marker = ""
        if acc > best_acc:
            best_acc = acc; best_state = copy.deepcopy(model.state_dict()); no_imp = 0; marker=" <- best"
        else:
            no_imp += 1
        print(f"Epoch {epoch:3d}/{args.epochs} | loss={tloss:.4f} | val_acc={acc*100:.1f}% | {time.time()-t0:.1f}s{marker}")
        if no_imp >= args.patience:
            print("[EARLY STOP]"); break

    if best_state: model.load_state_dict(best_state)

    # Confusion matrix + per-class
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for csi, y in vl:
            preds.extend(model(csi.to(DEVICE)).argmax(1).cpu().numpy()); gts.extend(y.numpy())
    preds, gts = np.array(preds), np.array(gts)
    n = len(ACTIVITY_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for g, pr in zip(gts, preds): cm[g, pr] += 1
    print("\n[CONFUSION] hang=that, cot=du doan")
    print("          " + " ".join(f"{a[:5]:>6s}" for a in ACTIVITY_NAMES))
    for i, a in enumerate(ACTIVITY_NAMES):
        row = " ".join(f"{cm[i,j]:6d}" for j in range(n))
        per = cm[i,i]/max(cm[i].sum(),1)*100
        print(f"  {a:8s} {row}  ({per:.0f}%)")
    print(f"\n[BEST] overall val_acc={best_acc*100:.1f}%")

    torch.save({
        "model_state": model.state_dict(), "mean": mean, "std": std,
        "val_acc": float(best_acc),
        "config": {"subcarriers": subcarriers, "n_rx": args.rx,
                   "n_activities": n, "window_size": WINDOW_SIZE},
    }, args.out)
    print(f"[SAVED] {args.out}")

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(ACTIVITY_NAMES, rotation=45)
    ax.set_yticks(range(n)); ax.set_yticklabels(ACTIVITY_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion (acc={best_acc*100:.1f}%)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i,j], ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.tight_layout(); plt.savefig(str(Path(args.out).with_suffix(".png")), dpi=120)
    print(f"[SAVED] {Path(args.out).with_suffix('.png')}")


if __name__ == "__main__":
    train()
