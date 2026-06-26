"""
record_demo.py — Thu CSI (2 cong) + video camera, chay model OFFLINE, xuat MP4:
                 BEN TRAI  = camera that + nhan hoat dong
                 BEN PHAI  = skeleton 2D do CSI doan (de DOI CHIEU)

Vi sao offline tot hon realtime:
  * Cua so CAN GIUA quanh moi frame -> nhan khop dung luc dong, KHONG tre.
  * Lam muot nhan (median) + lam muot skeleton (EMA) -> het nhay.
  * CSI va frame gan timestamp -> dong bo chuan.
  Vat ly khong sua duoc: tu the tinh de lan empty; pose CSI chi THO (PCK~13%) ->
  panel pose ben phai co the lech that nguoi -> do CHINH LA dieu can doi chieu.

Chay:
  python record_demo.py --ports COM9 COM3 --cam http://192.168.51.3:4747/video ^
      --activity models\activity_demo.pth --pose models\pose_rx2.pth ^
      --secs 120 --out demo_out.mp4
"""

import argparse, threading, time
import cv2
import numpy as np
import torch
import serial

from model import (CSIActivityClassifier, CSIPoseModel, CSIPresenceClassifier,
                   ACTIVITY_NAMES, WINDOW_SIZE, N_KEYPOINTS)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_PER_NODE = 52
SKELETON = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,5),(4,6),(1,7),(2,8),(7,8),
            (7,9),(8,10),(9,11),(10,12),(0,13),(0,14),(13,15),(14,16)]


def parse_csi_amp(line):
    try:
        line = line.strip()
        if not line.startswith("CSI_DATA"):
            return None
        b = line.index("["); e = line.rindex("]")
        raw = list(map(int, line[b+1:e].split(",")))
        if len(raw) < 4:
            return None
        n = len(raw) // 2
        amp = np.abs([complex(raw[i*2+1], raw[i*2]) for i in range(n)])
        idx = np.linspace(0, len(amp)-1, N_PER_NODE, dtype=int)
        return amp[idx].astype(np.float32)
    except Exception:
        return None


class CSIReader(threading.Thread):
    def __init__(self, port, store, tag):
        super().__init__(daemon=True)
        self.port, self.store, self.tag, self.running = port, store, tag, True
    def run(self):
        try:
            ser = serial.Serial(self.port, 921600, timeout=1)
            print(f"[CSI node{self.tag}] MO OK {self.port}")
        except Exception as e:
            print(f"[CSI node{self.tag}] MO LOI {self.port}: {e}"); return
        while self.running:
            try:
                if ser.in_waiting > 16384:
                    ser.reset_input_buffer()
                amp = parse_csi_amp(ser.readline().decode("utf-8", errors="ignore"))
                if amp is not None:
                    self.store.append((time.time(), amp))
            except Exception:
                continue
        ser.close()
    def stop(self): self.running = False


def load_model(path, kind):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    st = ck["model_state"]
    sub = len(np.asarray(ck["mean"]).ravel())
    if kind == "activity":
        n = int(st["head.3.weight"].shape[0])
        m = CSIActivityClassifier(subcarriers=sub, n_activities=n)
    elif kind == "presence":
        m = CSIPresenceClassifier(subcarriers=sub)
    else:
        m = CSIPoseModel(subcarriers=sub)
    m.load_state_dict(st); m.to(DEVICE).eval()
    mean = torch.from_numpy(np.asarray(ck["mean"], np.float32)).to(DEVICE)
    std  = torch.from_numpy(np.asarray(ck["std"],  np.float32)).to(DEVICE)
    return m, mean, std, sub


def window_centered(ts, amp, t):
    if len(ts) < WINDOW_SIZE:
        return None
    j = np.searchsorted(ts, t)
    lo = max(0, j - WINDOW_SIZE // 2); hi = lo + WINDOW_SIZE
    if hi > len(ts):
        hi = len(ts); lo = hi - WINDOW_SIZE
    return np.stack(amp[lo:hi], axis=0)


def draw_pose_panel(H, W, kps):
    panel = np.zeros((H, W, 3), np.uint8)
    cv2.putText(panel, "CSI 2D pose (coarse)", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 160, 255), 2)
    if kps is None:
        return panel
    pts = [(int(np.clip(k[0],0,1)*W), int(np.clip(k[1],0,1)*H)) for k in kps]
    for i, j in SKELETON:
        cv2.line(panel, pts[i], pts[j], (70, 160, 70), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(panel, p, 4, (90, 200, 90), -1, cv2.LINE_AA)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs="+", default=["COM9", "COM3"])
    ap.add_argument("--cam", required=True)
    ap.add_argument("--activity", default="models/activity_demo.pth")
    ap.add_argument("--pose", default="models/pose_rx2.pth")
    ap.add_argument("--presence", default="models/presence_demo.pth")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--secs", type=int, default=120)
    ap.add_argument("--out", default="demo_out.mp4")
    ap.add_argument("--smooth", type=int, default=7)
    args = ap.parse_args()

    act, act_m, act_s, sub = load_model(args.activity, "activity")
    n_node = sub // N_PER_NODE
    if len(args.ports) != n_node:
        raise SystemExit(f"[LOI] Model can {n_node} cong, ban dua {len(args.ports)}.")
    pose = None
    try:
        import os
        if args.pose and os.path.exists(args.pose):
            pose, pose_m, pose_s, _ = load_model(args.pose, "pose")
            print("[INFO] pose OK -> co panel doi chieu ben phai")
    except Exception as e:
        print(f"[WARN] khong load pose ({e}) -> bo panel pose")
    pres = None
    try:
        import os as _os
        if args.presence and _os.path.exists(args.presence):
            pres, pres_m, pres_s, _ = load_model(args.presence, "presence")
            print("[INFO] presence OK -> chan empty bang presence (chinh xac hon)")
    except Exception as e:
        print(f"[WARN] khong load presence ({e})"); pres = None
    print(f"[INFO] device={DEVICE} | subcarriers={sub} | {n_node} node")

    # ---------- PHA 1: THU ----------
    stores = [[] for _ in args.ports]
    readers = [CSIReader(p, stores[i], i) for i, p in enumerate(args.ports)]
    for r in readers: r.start()
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        for r in readers: r.stop()
        raise SystemExit(f"[LOI] Khong mo duoc camera {args.cam}")
    frames = []
    t0 = time.time()
    print(f"[INFO] Thu {args.secs}s... dien CHAM, giu moi tu the 4-5s. [q]=dung som")
    while time.time() - t0 < args.secs:
        ok, fr = cap.read()
        if not ok:
            continue
        frames.append((time.time(), fr))
        disp = fr.copy()
        cv2.putText(disp, f"REC {int(time.time()-t0)}s/{args.secs}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("REC (q=dung)", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()
    for r in readers: r.stop()
    print(f"[INFO] Thu xong: {len(frames)} frame, CSI/node={[len(s) for s in stores]}. Xu ly...")

    if len(frames) < 5 or any(len(s) < WINDOW_SIZE for s in stores):
        raise SystemExit("[LOI] Khong du du lieu.")

    node_ts, node_amp = [], []
    for s in stores:
        s.sort(key=lambda x: x[0])
        node_ts.append(np.array([x[0] for x in s]))
        node_amp.append([x[1] for x in s])

    # ---------- PHA 2: MODEL OFFLINE ----------
    raw_pred, raw_conf, pose_kps = [], [], []
    sm_kp = None
    for ts, _ in frames:
        wins = [window_centered(node_ts[i], node_amp[i], ts) for i in range(n_node)]
        if any(w is None for w in wins):
            raw_pred.append(-1); raw_conf.append(0.0); pose_kps.append(None); continue
        win = np.concatenate(wins, axis=1).astype(np.float32)
        x = torch.from_numpy(win).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p = torch.softmax(act((x - act_m) / act_s), 1)[0].cpu().numpy()
            pred_i = int(p.argmax())
            # CONG CHAN: neu presence bao trong -> ep nhan = empty
            if pres is not None:
                pr = pres((x - pres_m) / pres_s)[0, 0].item()
                if pr < args.thr:
                    pred_i = ACTIVITY_NAMES.index("empty")
            raw_pred.append(pred_i); raw_conf.append(float(p.max()))
            is_empty = (pred_i == ACTIVITY_NAMES.index('empty'))
            if pose is not None and not is_empty:
                kk = pose((x - pose_m) / pose_s)[0].cpu().numpy().reshape(N_KEYPOINTS, 2)
                sm_kp = kk if sm_kp is None else 0.7 * sm_kp + 0.3 * kk
                pose_kps.append(sm_kp.copy())
            else:
                if is_empty: sm_kp = None   # phong trong -> khong ve nguoi
                pose_kps.append(None)

    # lam muot nhan (median)
    k = max(1, args.smooth | 1); half = k // 2
    sm_pred = list(raw_pred)
    for i in range(len(raw_pred)):
        seg = [v for v in raw_pred[max(0,i-half):i+half+1] if v >= 0]
        if seg:
            sm_pred[i] = int(np.bincount(seg).argmax())

    # ---------- XUAT MP4 (ghep doi) ----------
    h, w = frames[0][1].shape[:2]
    dur = frames[-1][0] - frames[0][0]
    fps = max(5.0, min(30.0, len(frames) / dur if dur > 0 else 15.0))
    out_w = w * 2 if pose is not None else w
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, h))
    for i, (ts, fr) in enumerate(frames):
        lab = ACTIVITY_NAMES[sm_pred[i]] if sm_pred[i] >= 0 else "..."
        cv2.rectangle(fr, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(fr, lab.upper(), (15, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 255, 120), 3, cv2.LINE_AA)
        cv2.putText(fr, f"{raw_conf[i]*100:.0f}%", (w-110, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 100), 2)
        cv2.putText(fr, "camera (that)", (15, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if pose is not None:
            panel = draw_pose_panel(h, w, pose_kps[i])
            fr = cv2.hconcat([fr, panel])
        vw.write(fr)
    vw.release()
    print(f"[DONE] Da xuat {args.out}  ({len(frames)} frame, {fps:.1f} fps)")


if __name__ == "__main__":
    main()
