"""
realtime_pose.py — DEMO realtime CSI 2 RX (khong can camera)
  CHINH : nhan dien HOAT DONG (walk/stand/sit/fall/empty/arms_horizontal/arms_up)
  PHU   : presence gate (co nguoi / phong trong)
  TUY CHON: pose skeleton 2D — MAC DINH TAT (chi tho, PCK@0.5~13%); bat bang [p].

Dung CHINH class + thu tu nhan THAT cua project (model.py: CSIActivityClassifier,
ACTIVITY_NAMES). Subcarrier suy tu len(mean) trong checkpoint nen luon dung
(52 = 1 RX, 104 = 2 RX...). KHONG doan.

Chay (model 2 RX = 104):
  python realtime_pose.py --ports COM9 COM3 ^
      --activity models\activity7_random.pth ^
      --presence models\presence_rx2.pth ^
      --pose     models\pose_rx2.pth
THU TU CONG phai giong luc thu data (data_collection.py --ports COM9 COM3):
  COM9 = node0 (cao ~160cm), COM3 = node1 (thap ~90cm).

Phim: [q]=thoat  [p]=bat/tat pose tho  [+]/[-]=nguong presence
"""

import argparse, threading, time, os
from collections import deque, Counter

import cv2
import numpy as np
import torch
import serial

# Dung dung dinh nghia cua project — khong dung lai kien truc
from model import (CSIActivityClassifier, CSIPresenceClassifier, CSIPoseModel,
                   ACTIVITY_NAMES, WINDOW_SIZE, N_KEYPOINTS)

BAUD_RATE    = 921600
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DISPLAY_W, DISPLAY_H = 720, 540
SMOOTH_ALPHA = 0.75
VOTE_K       = 9      # so cua so de bo phieu nhan hoat dong (on dinh chu)
CONF_MIN     = 0.55   # duoi nguong nay coi la 'khong chac' -> khong doi nhan
N_PER_NODE   = 52
SKELETON = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,5),(4,6),(1,7),(2,8),(7,8),
            (7,9),(8,10),(9,11),(10,12),(0,13),(0,14),(13,15),(14,16)]


# ----------------------------------------------------------- parse (giong data_collection.py)
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
    """Doc 1 cong -> 1 buffer. Co dem chuan doan (raw / parsed)."""
    def __init__(self, port, buf, tag):
        super().__init__(daemon=True)
        self.port, self.buf, self.tag, self.running = port, buf, tag, True
        self.n_raw = 0; self.n_ok = 0; self.ok_open = False

    def run(self):
        try:
            ser = serial.Serial(self.port, BAUD_RATE, timeout=1)
            self.ok_open = True
            print(f"[CSI node{self.tag}] MO OK {self.port}")
        except Exception as e:
            print(f"[CSI node{self.tag}] MO LOI {self.port}: {e}"); return
        while self.running:
            try:
                # CHONG DELAY: neu buffer he dieu hanh bi don (doc khong kip toc do CSI den),
                # xa bo phan cu de luon doc du lieu MOI NHAT (thoi gian thuc).
                try:
                    if ser.in_waiting > 8192:
                        ser.reset_input_buffer()
                except Exception:
                    pass
                ln = ser.readline().decode("utf-8", errors="ignore")
                if ln:
                    self.n_raw += 1
                amp = parse_csi_amp(ln)
                if amp is not None:
                    self.n_ok += 1
                    self.buf.append(amp)
            except Exception:
                continue
        ser.close()

    def stop(self): self.running = False


# ----------------------------------------------------------- load (dung class project)
def _to_t(x): return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(DEVICE)

def load_ckpt(path, kind):
    ck  = torch.load(path, map_location=DEVICE, weights_only=False)
    st  = ck["model_state"]
    sub = len(np.asarray(ck["mean"]).ravel())          # = so subcarrier (luon dung)
    if kind == "activity":
        n_act = int(st["head.3.weight"].shape[0])
        model = CSIActivityClassifier(subcarriers=sub, n_activities=n_act)
    elif kind == "presence":
        model = CSIPresenceClassifier(subcarriers=sub)
    else:
        model = CSIPoseModel(subcarriers=sub)
    model.load_state_dict(st); model.to(DEVICE).eval()
    return model, _to_t(ck["mean"]), _to_t(ck["std"]), sub


def build_window(buffers):
    if any(len(b) < WINDOW_SIZE for b in buffers):
        return None
    per = [np.stack(list(b)[-WINDOW_SIZE:], 0) for b in buffers]
    return np.concatenate(per, axis=1).astype(np.float32)        # (20, 52*N)

def draw_skeleton(frame, kps):
    h, w = frame.shape[:2]
    pts = [(int(k[0]*w), int(k[1]*h)) for k in kps]
    for i, j in SKELETON:
        cv2.line(frame, pts[i], pts[j], (70,140,70), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 4, (90,180,90), -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs="+", default=["COM9", "COM3"])
    ap.add_argument("--activity", default="models/activity7_random.pth")
    ap.add_argument("--presence", default="models/presence_rx2.pth")
    ap.add_argument("--pose",     default="models/pose_rx2.pth")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()
    print(f"[INFO] Device: {DEVICE}")

    act = pres = pose = None
    if args.activity and os.path.exists(args.activity):
        act, act_m, act_s, sub_a = load_ckpt(args.activity, "activity")
        print(f"[INFO] activity OK: subcarriers={sub_a}, lop={ACTIVITY_NAMES}")
    else:
        print(f"[WARN] khong thay activity model: {args.activity}")
    if args.presence and os.path.exists(args.presence):
        pres, pres_m, pres_s, sub_p = load_ckpt(args.presence, "presence")
        print(f"[INFO] presence OK: subcarriers={sub_p}")
    if args.pose and os.path.exists(args.pose):
        pose, pose_m, pose_s, sub_po = load_ckpt(args.pose, "pose")
        print(f"[INFO] pose OK: subcarriers={sub_po} (mac dinh TAT, bat bang [p])")

    need = sub_a if act else (sub_p if pres else sub_po)
    if N_PER_NODE * len(args.ports) != need:
        raise SystemExit(
            f"[LOI] Model can {need} subcarrier = {need//N_PER_NODE} node nhung ban dua "
            f"{len(args.ports)} cong.\n      Sua: --ports " +
            " ".join(["COM9","COM3","COM?"][:need//N_PER_NODE]))

    buffers = [deque(maxlen=WINDOW_SIZE*3) for _ in args.ports]
    readers = [CSIReader(p, buffers[i], i) for i, p in enumerate(args.ports)]
    for r in readers: r.start()

    threshold, show_pose, smooth = args.thr, False, None
    fps_buf = deque(maxlen=30); last_diag = time.time()
    vote_buf = deque(maxlen=VOTE_K); shown_label = '...'; printed_amp = False
    last_len = -1
    print("[INFO] [q]=thoat  [p]=pose tho  [+]/[-]=nguong\n")

    while True:
        t0 = time.time()
        frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), np.uint8)
        win = build_window(buffers)
        prob, person = 1.0, True

        if win is not None:
            csi = torch.from_numpy(win).unsqueeze(0).to(DEVICE)
            if pres is not None:
                with torch.no_grad():
                    prob = pres((csi - pres_m) / pres_s)[0, 0].item()
                person = prob >= threshold

            if person:
                cv2.rectangle(frame, (0,0), (DISPLAY_W,50), (0,90,0), -1)
                cv2.putText(frame, "PERSON", (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            else:
                cv2.rectangle(frame, (0,0), (DISPLAY_W,50), (45,45,45), -1)
                cv2.putText(frame, "EMPTY (phong trong)", (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160,160,160), 2)

            # kiem tra thu tu node (in 1 lan): bien do trung binh moi node
            if not printed_amp:
                half = win.shape[1] // 2
                a0 = float(win[:, :half].mean()); a1 = float(win[:, half:].mean())
                print(f"[kiem tra node] mean|amp| node0(52 dau)={a0:.1f}  node1(52 sau)={a1:.1f}  "
                      f"(2 so nen khac nhau; neu giong het co the cam trung 1 RX)")
                printed_amp = True

            if person and act is not None:
                with torch.no_grad():
                    probs = torch.softmax(act((csi - act_m) / act_s), 1)[0].cpu().numpy()
                k = int(probs.argmax()); conf = float(probs[k])
                # chi bo phieu khi du tin cay -> bot nhay lung tung
                if conf >= CONF_MIN:
                    vote_buf.append(k)
                if len(vote_buf) > 0:
                    maj, cnt = Counter(vote_buf).most_common(1)[0]
                    shown_label = ACTIVITY_NAMES[maj]
                    stable = cnt / len(vote_buf)
                else:
                    shown_label, stable = "...", 0.0
                col = (0,255,120) if conf >= CONF_MIN else (0,170,170)
                cv2.putText(frame, shown_label.upper(), (15, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.9, col, 4, cv2.LINE_AA)
                cv2.putText(frame, f"conf {conf*100:.0f}%  on_dinh {stable*100:.0f}%",
                            (18, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,100), 2)
                for r_, ci in enumerate(probs.argsort()[::-1][:3]):
                    cv2.putText(frame, f"{ACTIVITY_NAMES[ci]:>16s} {probs[ci]*100:4.0f}%",
                                (DISPLAY_W-275, 95+r_*26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

            if person and show_pose and pose is not None:
                with torch.no_grad():
                    kps = pose((csi - pose_m) / pose_s)[0].cpu().numpy().reshape(N_KEYPOINTS, 2)
                smooth = kps if smooth is None else SMOOTH_ALPHA*smooth + (1-SMOOTH_ALPHA)*kps
                draw_skeleton(frame, smooth)
                cv2.putText(frame, "POSE: COARSE / experimental (PCK@0.5~13%)",
                            (15, DISPLAY_H-70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,160,255), 1)
            else:
                smooth = None

            bw = int(prob*(DISPLAY_W-30))
            cv2.rectangle(frame, (15,DISPLAY_H-35), (15+bw,DISPLAY_H-20), (0,200,90) if person else (70,70,70), -1)
            cv2.rectangle(frame, (15,DISPLAY_H-35), (DISPLAY_W-15,DISPLAY_H-20), (90,90,90), 1)
            tx = int(threshold*(DISPLAY_W-30))+15
            cv2.line(frame, (tx,DISPLAY_H-38), (tx,DISPLAY_H-17), (0,200,255), 2)
            cv2.putText(frame, f"presence {prob:.2f}  thr {threshold:.2f}",
                        (15,DISPLAY_H-42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

        fps_buf.append(time.time()-t0); avg = sum(fps_buf)/len(fps_buf)
        cv2.putText(frame, f"FPS {1/avg:.1f}" if avg>0 else "FPS -",
                    (DISPLAY_W-110, DISPLAY_H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,0), 1)

        if win is None:
            bufmin = min(len(b) for b in buffers)
            cv2.putText(frame, f"Dang thu CSI... {bufmin}/{WINDOW_SIZE}",
                        (DISPLAY_W//2-150, DISPLAY_H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120,120,120), 1)
            # chuan doan serial moi 2s
            if time.time() - last_diag > 2.0:
                last_diag = time.time()
                for r in readers:
                    print(f"  [chuan doan] node{r.tag} {r.port}: mo={r.ok_open} "
                          f"dong_doc={r.n_raw} parse_CSI_OK={r.n_ok} buf={len(r.buf)}")

        cv2.imshow("CSI Realtime Demo  [q p + -]", frame)
        key = cv2.waitKey(10) & 0xFF
        if   key == ord("q"): break
        elif key == ord("p"): show_pose = not show_pose
        elif key in (ord("+"), ord("=")): threshold = min(0.99, threshold+0.05)
        elif key == ord("-"):             threshold = max(0.01, threshold-0.05)

    for r in readers: r.stop()
    cv2.destroyAllWindows()
    print("[INFO] Thoat.")


if __name__ == "__main__":
    main()
