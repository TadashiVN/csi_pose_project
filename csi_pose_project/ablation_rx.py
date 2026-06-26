"""
ablation_rx.py — So sanh 1 RX (node0 / node1) vs 2 RX cho bai toan ACTIVITY.
                 CHAY THAT tren du lieu da thu, KHONG bia.

Vi sao chay duoc ma khong can thu lai:
  data_collection.py luu moi sample voi truong "csi" co dang
        (nodes=2, window=20, subcarriers=52)
  tuc node 0 va node 1 DA TACH RIENG san. Do do chi can:
     - node0 : csi[0]               -> (20, 52)   = chi may thu cao
     - node1 : csi[1]               -> (20, 52)   = chi may thu thap
     - both  : ghep theo subcarrier -> (20, 104)  = 2 RX
  roi huan luyen CUNG MOT model cho ba truong hop -> so sanh cong bang.

Vi sao van phai chay lenh nay:
  Project hien chua co file ket qua nao cho cau hinh 1 RX. Cac model da train
  (pose_rx2, presence_rx2, compare_models...) deu dung 2 RX. Muon co so 1 RX
  de SO SANH thi phai train them -> script nay train ho ban (khong bia so).

Cach chay (tu E:\\csi_pose_project\\):
     python ablation_rx.py                       # dung .npz trong dataset/raw/
     python ablation_rx.py --epochs 25 --holdout session_10
     python ablation_rx.py --from-json           # doc thang .json neu .npz la

Ket qua: in bang same-domain + cross-session cho node0 / node1 / both, va luu
ablation_rx_results.json. Copy bang do dan vao bao cao (Table tab:rxablation).
"""

import os, glob, json, argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("Can torch: pip install torch")

WINDOW = 20
N_PER_RX = 52          # subcarrier moi node (SUBCARRIERS_USE trong data_collection.py)


# ===========================================================================
# 1) DOC 1 SESSION -> dict {both:(N,20,104), rx0:(N,20,52), rx1:(N,20,52), y}
# ===========================================================================
def _split_one(csi):
    """csi cua 1 sample -> (rx0_(20,52), rx1_(20,52)) hoac (x, None) neu 1 node."""
    a = np.asarray(csi, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 2 and a.shape[1] == WINDOW and a.shape[2] == N_PER_RX:
        return a[0], a[1]                                   # (2,20,52) layout goc
    if a.ndim == 2 and a.shape == (WINDOW, 2 * N_PER_RX):
        return a[:, :N_PER_RX], a[:, N_PER_RX:]             # (20,104) da ghep
    if a.ndim == 2 and a.shape == (WINDOW, N_PER_RX):
        return a, None                                      # (20,52) chi 1 node
    if a.ndim == 3 and a.shape[0] >= 2:
        return a[0], a[1]
    return None, None


def load_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    samples = d.get("samples", d)
    r0, r1, ys = [], [], []
    for s in samples:
        a, b = _split_one(s["csi"])
        if a is None:
            continue
        r0.append(a); r1.append(b if b is not None else a); ys.append(str(s["activity"]))
    rx0 = np.asarray(r0, np.float32); rx1 = np.asarray(r1, np.float32)
    return {"rx0": rx0, "rx1": rx1, "both": np.concatenate([rx0, rx1], -1),
            "y": np.array(ys), "single": np.array_equal(rx0, rx1)}


def load_from_npz(path):
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.files)
    csi_key = next((k for k in keys if k.lower() in
                    ("csi", "x", "features", "window", "amp")), None)
    if csi_key is None:
        cands = [k for k in keys if npz[k].ndim >= 3 and np.issubdtype(npz[k].dtype, np.floating)] \
             or [k for k in keys if npz[k].ndim >= 2 and np.issubdtype(npz[k].dtype, np.floating)]
        csi_key = max(cands, key=lambda k: npz[k].size) if cands else None
    if csi_key is None:
        raise ValueError(f"Khong thay mang CSI. keys={keys}")
    X = np.asarray(npz[csi_key], np.float32)

    y_key = next((k for k in keys if k.lower() in
                  ("activity", "activities", "y", "label", "labels", "act")
                  and npz[k].ndim == 1 and len(npz[k]) == len(X)), None)
    if y_key is None:
        y_key = next((k for k in keys if npz[k].ndim == 1 and len(npz[k]) == len(X)
                      and k != csi_key), None)
    if y_key is None:
        raise ValueError(f"Khong thay nhan activity. keys={keys}")
    y = np.array([str(v) for v in npz[y_key]])

    if X.ndim == 4 and X.shape[1] == 2:                 # (N,2,20,52)
        rx0, rx1 = X[:, 0], X[:, 1]
    elif X.ndim == 3 and X.shape[-1] == 2 * N_PER_RX:   # (N,20,104) da ghep
        rx0, rx1 = X[..., :N_PER_RX], X[..., N_PER_RX:]
    elif X.ndim == 3 and X.shape[-1] == N_PER_RX:       # (N,20,52) 1 node
        return {"rx0": X, "rx1": X, "both": X, "y": y, "single": True}
    else:
        raise ValueError(f"Layout CSI = {X.shape} khong nhan dang. Thu --from-json.")
    return {"rx0": rx0.astype(np.float32), "rx1": rx1.astype(np.float32),
            "both": np.concatenate([rx0, rx1], -1).astype(np.float32),
            "y": y, "single": False}


# ===========================================================================
# 2) MODEL (CNN-only nho gon — giong cau hinh trong so sanh kien truc)
# ===========================================================================
class ActCNN(nn.Module):
    def __init__(self, in_sub, n_cls):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_sub, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_cls))

    def forward(self, x):                 # x: (B,20,F)
        return self.head(self.cnn(x.permute(0, 2, 1)))


def train_eval(Xtr, ytr, Xte, yte, n_cls, epochs, dev, seed=42):
    torch.manual_seed(seed)
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    m = ActCNN(Xtr.shape[-1], n_cls).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    for _ in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); lf(m(xb), yb).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(torch.tensor(Xte).to(dev)).argmax(1).cpu().numpy()
    return float((pred == yte).mean()) * 100.0


# ===========================================================================
# 3) MAIN
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/raw")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--holdout", default="session_10")
    ap.add_argument("--from-json", action="store_true")
    ap.add_argument("--cross-loso", action="store_true",
                    help="cross-session = trung binh qua TAT CA session (on dinh, cham hon)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[i] device = {dev}")

    ext = "json" if args.from_json else "npz"
    files = sorted(glob.glob(os.path.join(args.data, f"session_*.{ext}")))
    if not files:
        raise SystemExit(f"Khong thay file .{ext} trong {args.data}")

    sessions, warned = {}, False
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            d = load_from_json(f) if args.from_json else load_from_npz(f)
        except Exception as e:
            print(f"[!] {name}: bo qua ({e})"); continue
        sessions[name] = d
        if d.get("single") and not warned:
            print("[!] Du lieu chi co 1 node -> khong so sanh 2 RX duoc.")
            warned = True
        print(f"[i] {name}: N={len(d['y'])}  both{d['both'].shape}  rx0{d['rx0'].shape}")

    if not sessions:
        raise SystemExit("Khong load duoc session nao.")
    if args.holdout not in sessions:
        args.holdout = sorted(sessions)[-1]
        print(f"[i] holdout chuyen sang {args.holdout}")

    classes = sorted(set(np.concatenate([d["y"] for d in sessions.values()])))
    cmap = {c: i for i, c in enumerate(classes)}
    n_cls = len(classes)
    print(f"[i] {len(sessions)} session, {n_cls} lop: {classes}")
    enc = lambda y: np.array([cmap[v] for v in y], dtype=np.int64)

    results = {}
    for cfg in ("rx0", "rx1", "both"):
        # same-domain (1 lan, on dinh)
        X = np.concatenate([sessions[s][cfg] for s in sessions])
        Y = np.concatenate([enc(sessions[s]["y"]) for s in sessions])
        idx = np.random.permutation(len(X)); cut = int(0.8 * len(idx))
        acc_s = train_eval(X[idx[:cut]], Y[idx[:cut]], X[idx[cut:]], Y[idx[cut:]],
                           n_cls, args.epochs, dev, args.seed)

        if args.cross_loso:
            # cross-session = trung binh qua TAT CA session (kieu LOSO) -> on dinh
            accs = []
            for ho in sessions:
                trX = np.concatenate([sessions[s][cfg] for s in sessions if s != ho])
                trY = np.concatenate([enc(sessions[s]["y"]) for s in sessions if s != ho])
                a = train_eval(trX, trY, sessions[ho][cfg], enc(sessions[ho]["y"]),
                               n_cls, args.epochs, dev, args.seed)
                accs.append(a)
                print(f"    [{cfg}] holdout {ho}: {a:5.1f}%")
            acc_c = float(np.mean(accs)); acc_std = float(np.std(accs))
            results[cfg] = (acc_s, acc_c, acc_std)
            print(f"[=] {cfg:5s} same={acc_s:5.1f}%  cross-LOSO={acc_c:5.1f}+-{acc_std:.1f}%")
        else:
            trX = np.concatenate([sessions[s][cfg] for s in sessions if s != args.holdout])
            trY = np.concatenate([enc(sessions[s]["y"]) for s in sessions if s != args.holdout])
            acc_c = train_eval(trX, trY, sessions[args.holdout][cfg],
                               enc(sessions[args.holdout]["y"]), n_cls, args.epochs, dev, args.seed)
            results[cfg] = (acc_s, acc_c, None)
            print(f"[=] {cfg:5s} same={acc_s:5.1f}%  cross({args.holdout})={acc_c:5.1f}%")

    label = {"rx0": "RX0 only (node0, 52)", "rx1": "RX1 only (node1, 52)", "both": "Both (104)"}
    chdr = "Cross-LOSO %" if args.cross_loso else "Cross-session %"
    print("\n=================== BANG ABLATION (dan vao bao cao) ===================")
    print(f"{'Config':24s}{'Same-domain %':>16s}{chdr:>20s}")
    for cfg in ("rx0", "rx1", "both"):
        s, c, sd = results[cfg]
        cstr = f"{c:.1f}+-{sd:.1f}" if sd is not None else f"{c:.1f}"
        print(f"{label[cfg]:24s}{s:16.1f}{cstr:>20s}")
    print(f"\n-> RX thu hai them (same-domain):   "
          f"+{results['both'][0]-max(results['rx0'][0],results['rx1'][0]):.1f}%")
    print(f"-> RX thu hai them (cross):         "
          f"+{results['both'][1]-max(results['rx0'][1],results['rx1'][1]):.1f}%")
    with open("ablation_rx_results.json", "w") as fp:
        json.dump({k: {"same_domain": v[0], "cross": v[1], "cross_std": v[2]}
                   for k, v in results.items()}, fp, indent=2)
    print("[i] Da luu ablation_rx_results.json")


if __name__ == "__main__":
    main()
