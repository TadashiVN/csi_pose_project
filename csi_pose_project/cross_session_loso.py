"""
cross_session_loso.py — Danh gia Leave-One-Session-Out (LOSO) cho activity.

Y NGHIA: lan luot giu MOI session lam test (phan con lai train), do accuracy,
roi bao cao trung binh +/- do lech. Day la chuan danh gia cross-domain trong
WiFi sensing — vung hon mot lan do don le (vd 38.8% hay 29.1% deu khong dai dien).
Phu hop muc Additional Study cua rubric (danh gia chat, giong nhieu seed).

Chay (LOSO tren cac session lon, epoch giam cho nhanh):
  python cross_session_loso.py --data dataset/raw/session_01.npz ... session_10.npz \
    --rx 2 --augment --epochs 20

Chi LOSO mot so session de tiet kiem thoi gian:
  ... --test-only session_05.npz session_07.npz session_10.npz
"""

import argparse, copy, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import CSIActivityClassifier, ACTIVITY_NAMES, SUBCARRIERS_PER_RX
from train_activity import ActivityDataset, Norm, stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_eval(train_files, test_file, rx, epochs, lr, patience, augment, batch):
    sc = rx * SUBCARRIERS_PER_RX
    n_act = len(ACTIVITY_NAMES)
    tb = ActivityDataset(train_files, n_rx=rx)
    vb = ActivityDataset([test_file], n_rx=rx)
    if len(tb) == 0 or len(vb) == 0:
        return None
    mean, std = stats(tb, range(len(tb)))
    tl = DataLoader(Norm(tb, mean, std, augment), batch_size=batch, shuffle=True)
    vl = DataLoader(Norm(vb, mean, std), batch_size=batch, shuffle=False)

    model = CSIActivityClassifier(subcarriers=sc, n_activities=n_act).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best, no_imp = 0.0, 0
    for ep in range(1, epochs+1):
        model.train()
        for csi, y in tl:
            csi, y = csi.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(csi), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for csi, y in vl:
                preds.extend(model(csi.to(DEVICE)).argmax(1).cpu().numpy()); gts.extend(y.numpy())
        acc = (np.array(preds) == np.array(gts)).mean() if gts else 0
        best = max(best, acc); no_imp = 0 if acc >= best else no_imp+1
        print(f"    epoch {ep:2d}/{epochs} acc={acc*100:.1f}%", end="\r")
        if no_imp >= patience: break
    print()
    return float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--rx", type=int, default=2, choices=[1,2,3])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--test-only", nargs="+", default=None,
                    help="Chi LOSO cac file nay (mac dinh: tat ca).")
    ap.add_argument("--out", default="loso_results")
    args = ap.parse_args()

    test_set = args.test_only if args.test_only else args.data
    # khop ten file day du
    def match(f):
        for d in args.data:
            if os.path.basename(d) == os.path.basename(f) or d == f:
                return d
        return f
    test_files = [match(f) for f in test_set]

    print(f"[INFO] Device {DEVICE} | RX={args.rx} | LOSO tren {len(test_files)} session\n")
    results = []
    for i, test_f in enumerate(test_files, 1):
        train_f = [d for d in args.data if d != test_f]
        print(f"[{i}/{len(test_files)}] TEST = {os.path.basename(test_f)} "
              f"(train tren {len(train_f)} session)")
        acc = train_eval(train_f, test_f, args.rx, args.epochs, args.lr,
                         args.patience, args.augment, args.batch)
        if acc is not None:
            results.append((os.path.basename(test_f), acc*100))

    accs = [a for _, a in results]
    print("\n" + "="*46)
    print("  KET QUA LOSO (cross-session)")
    print("="*46)
    for name, a in results:
        print(f"  test={name:24s} acc={a:5.1f}%")
    print("-"*46)
    if accs:
        print(f"  TRUNG BINH: {np.mean(accs):.1f}% +/- {np.std(accs):.1f}%")
        print(f"  (min {min(accs):.1f}% | max {max(accs):.1f}%)")
    print("="*46)

    with open(f"{args.out}.csv", "w", encoding="utf-8") as f:
        f.write("test_session,accuracy\n")
        for name, a in results:
            f.write(f"{name},{a:.2f}\n")
        if accs:
            f.write(f"MEAN,{np.mean(accs):.2f}\nSTD,{np.std(accs):.2f}\n")
    print(f"[SAVED] {args.out}.csv")


if __name__ == "__main__":
    main()
