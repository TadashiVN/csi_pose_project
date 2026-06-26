"""
compare_models.py — So sanh nhieu KIEN TRUC activity tren CUNG mot split.
Lap "Lo hong 1" rubric: so sanh >=2 mo hinh + bang trade-off accuracy/Params/FPS.

So sanh 4 mo hinh:
  cnn_bilstm (chinh) | cnn_only | lstm_only | mlp (baseline)

Chay (same-domain, random split co dinh seed):
  python compare_models.py --data dataset/raw/session_01.npz ... session_10.npz \
    --rx 2 --augment --epochs 30

Xuat bang + compare_models.json/.csv de dua vao bao cao.
"""

import argparse, copy, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from model import ACTIVITY_MODELS, ACTIVITY_NAMES, SUBCARRIERS_PER_RX, WINDOW_SIZE
from train_activity import ActivityDataset, Norm, stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def latency_fps(m, sc, n=100):
    m.eval()
    x = torch.randn(1, WINDOW_SIZE, sc, device=DEVICE)
    with torch.no_grad():
        for _ in range(15): m(x)
        if DEVICE == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n): m(x)
        if DEVICE == "cuda": torch.cuda.synchronize()
        dt = (time.time()-t0)/n
    return dt*1000, (1/dt if dt>0 else 0)


def train_one(name, ModelCls, train_loader, val_loader, sc, n_act, epochs, lr, patience):
    model = ModelCls(subcarriers=sc, n_activities=n_act).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best_acc, best_state, no_imp = 0.0, None, 0
    for ep in range(1, epochs+1):
        model.train()
        for csi, y in train_loader:
            csi, y = csi.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(csi), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        # val
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for csi, y in val_loader:
                preds.extend(model(csi.to(DEVICE)).argmax(1).cpu().numpy()); gts.extend(y.numpy())
        acc = (np.array(preds) == np.array(gts)).mean() if gts else 0
        if acc > best_acc:
            best_acc, best_state, no_imp = acc, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
        print(f"  [{name}] epoch {ep:2d}/{epochs} val_acc={acc*100:.1f}%", end="\r")
        if no_imp >= patience: break
    print()
    if best_state: model.load_state_dict(best_state)
    return model, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--rx", type=int, default=2, choices=[1,2,3])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="compare_models")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    sc = args.rx * SUBCARRIERS_PER_RX
    n_act = len(ACTIVITY_NAMES)
    print(f"[INFO] Device {DEVICE} | RX={args.rx} | subcarriers={sc}\n")

    # Mot split duy nhat, dung chung cho moi mo hinh -> cong bang
    base = ActivityDataset(args.data, n_rx=args.rx)
    n_val = int(len(base)*0.2)
    perm = torch.randperm(len(base), generator=torch.Generator().manual_seed(args.seed)).tolist()
    ti, vi = perm[n_val:], perm[:n_val]
    mean, std = stats(base, ti)
    train_ds = Norm(Subset(base, ti), mean, std, args.augment)
    val_ds   = Norm(Subset(base, vi), mean, std)
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    print(f"[SPLIT] train={len(ti)} val={len(vi)} (chung cho moi mo hinh)\n")

    rows = []
    for name, ModelCls in ACTIVITY_MODELS.items():
        print(f"=== Train: {name} ===")
        model, acc = train_one(name, ModelCls, tl, vl, sc, n_act,
                               args.epochs, args.lr, args.patience)
        params = count_params(model)
        lat, fps = latency_fps(model, sc)
        rows.append({"model": name, "val_acc": round(float(acc)*100, 2),
                     "params_M": round(params/1e6, 3),
                     "latency_ms": round(lat, 3), "fps": round(fps, 1)})

    # Bang ket qua
    print("\n" + "="*74)
    print(f"  {'Model':14s} {'Accuracy(%)':>12s} {'Params(M)':>11s} {'Latency(ms)':>12s} {'FPS':>9s}")
    print("-"*74)
    for r in sorted(rows, key=lambda x: -x["val_acc"]):
        print(f"  {r['model']:14s} {r['val_acc']:>12.2f} {r['params_M']:>11.3f} "
              f"{r['latency_ms']:>12.3f} {r['fps']:>9.1f}")
    print("="*74)

    with open(f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(f"{args.out}.csv", "w", encoding="utf-8") as f:
        f.write("model,val_acc,params_M,latency_ms,fps\n")
        for r in rows:
            f.write(f"{r['model']},{r['val_acc']},{r['params_M']},{r['latency_ms']},{r['fps']}\n")
    print(f"[SAVED] {args.out}.json + {args.out}.csv")


if __name__ == "__main__":
    main()
