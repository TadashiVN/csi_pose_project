"""
realtime_pose.py — CSI realtime → 2D pose skeleton (nền đen, không cần camera).

Cải tiến:
- Import CSIPoseModel từ model.py (chống duplicate)
- Empty detection dùng CV (std/mean), normalized → ổn định hơn variance thô
- Load threshold từ calibration_plot.json nếu có (do calibrate_threshold.py tạo)
- Phím +/- để tinh chỉnh runtime

Chạy:
  python realtime_pose.py --port COM3 --model models/pose_model.pth
  python realtime_pose.py --port COM3 --model models/pose_model.pth --calib calibration_plot.json
"""

import argparse
import json
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import serial
import torch

from model import N_KEYPOINTS, SUBCARRIERS, WINDOW_SIZE, load_checkpoint

BAUD_RATE = 921600
DISPLAY_W = 640
DISPLAY_H = 480
SMOOTH_ALPHA = 0.8
DEFAULT_CV_THRESHOLD = 0.025  # giá trị mặc định nếu không có file calib

SKELETON = [
    (0, 1), (0, 2), (1, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
    (1, 7), (2, 8), (7, 8),
    (7, 9), (8, 10), (9, 11), (10, 12),
    (0, 13), (0, 14), (13, 15), (14, 16),
]
KP_COLORS = [
    (255, 255, 255), (0, 255, 0), (0, 200, 255), (0, 255, 0), (0, 200, 255),
    (0, 255, 0), (0, 200, 255), (0, 255, 100), (0, 100, 255), (0, 255, 100),
    (0, 100, 255), (0, 255, 100), (0, 100, 255), (200, 200, 0), (200, 200, 0),
    (200, 100, 0), (200, 100, 0),
]


def parse_csi_amp(line):
    try:
        line = line.strip()
        if not line.startswith("CSI_DATA"):
            return None
        b_start = line.index("[")
        b_end = line.rindex("]")
        raw = list(map(int, line[b_start + 1:b_end].split(",")))
        if len(raw) < 4:
            return None
        n = len(raw) // 2
        amp = np.abs([complex(raw[i * 2 + 1], raw[i * 2]) for i in range(n)])
        idx = np.linspace(0, len(amp) - 1, SUBCARRIERS, dtype=int)
        return amp[idx].astype(np.float32)
    except Exception:
        return None


class CSIReader(threading.Thread):
    def __init__(self, port, buf):
        super().__init__(daemon=True)
        self.port = port
        self.buf = buf
        self.running = True

    def run(self):
        try:
            ser = serial.Serial(self.port, BAUD_RATE, timeout=1)
            print(f"[CSI] OK {self.port}")
        except serial.SerialException as e:
            print(f"[CSI] FAIL {e}")
            return
        while self.running:
            try:
                amp = parse_csi_amp(ser.readline().decode("utf-8", errors="ignore"))
                if amp is not None:
                    self.buf.append(amp)
            except Exception:
                continue
        ser.close()

    def stop(self):
        self.running = False


def coefficient_of_variation(csi_window):
    """
    CV = std/mean — normalized, không phụ thuộc biên độ tuyệt đối.
    Returns scalar (mean of per-subcarrier CV).
    """
    arr = np.stack(csi_window, axis=0)              # (20, 20)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    cv = std / (np.abs(mean) + 1e-8)
    return float(np.mean(cv))


def draw_skeleton(frame, keypoints):
    h, w = frame.shape[:2]
    pts = [(int(kp[0] * w), int(kp[1] * h)) for kp in keypoints]
    for i, j in SKELETON:
        if i < len(pts) and j < len(pts):
            cv2.line(frame, pts[i], pts[j], (80, 200, 80), 2, cv2.LINE_AA)
    for idx, pt in enumerate(pts):
        col = KP_COLORS[idx] if idx < len(KP_COLORS) else (255, 255, 255)
        cv2.circle(frame, pt, 5, col, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 6, (0, 0, 0), 1, cv2.LINE_AA)


def load_calibration(path):
    """Load CV threshold từ JSON do calibrate_threshold.py tạo."""
    if not path or not Path(path).exists():
        return None
    try:
        with open(path) as f:
            cfg = json.load(f)
        if cfg.get("metric") != "cv":
            print(f"[CALIB] [!] file {path} dùng metric '{cfg.get('metric')}' khác CV, bỏ qua.")
            return None
        return float(cfg["threshold"])
    except Exception as e:
        print(f"[CALIB] [!] không đọc được {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--model", default="models/pose_model.pth")
    parser.add_argument("--calib", default="calibration_plot.json",
                        help="JSON từ calibrate_threshold.py. Bỏ nếu không có.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="CV threshold tay (override calib)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model:  {args.model}")

    model, mean_np, std_np, ckpt = load_checkpoint(args.model, device=device)
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    print(f"[INFO] val_loss khi train: {ckpt.get('val_loss', 'N/A')}")

    # Threshold ưu tiên: --threshold > calib file > default
    if args.threshold is not None:
        threshold = float(args.threshold)
        print(f"[THR] Dùng threshold tay: {threshold:.4f}")
    else:
        loaded = load_calibration(args.calib)
        if loaded is not None:
            threshold = loaded
            print(f"[THR] Dùng threshold từ {args.calib}: {threshold:.4f}")
        else:
            threshold = DEFAULT_CV_THRESHOLD
            print(f"[THR] Dùng default: {threshold:.4f} (chạy calibrate_threshold.py để tốt hơn)")

    csi_buf = deque(maxlen=WINDOW_SIZE)
    reader = CSIReader(args.port, csi_buf)
    reader.start()

    smooth_kps = None
    fps_buf = deque(maxlen=30)

    print("\n[INFO] [q]=thoát  [+]=tăng threshold  [-]=giảm threshold\n")

    while True:
        t0 = time.time()
        frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)

        cv_val = 0.0
        if len(csi_buf) == WINDOW_SIZE:
            csi_window = list(csi_buf)
            cv_val = coefficient_of_variation(csi_window)
            has_person = cv_val > threshold

            if has_person:
                csi_np = np.stack(csi_window, axis=0)
                csi_t = torch.from_numpy(csi_np).unsqueeze(0).to(device)
                csi_t = (csi_t - mean) / std

                with torch.no_grad():
                    pred = model(csi_t)
                kps = pred[0].cpu().numpy().reshape(N_KEYPOINTS, 2)

                if smooth_kps is None:
                    smooth_kps = kps.copy()
                else:
                    smooth_kps = SMOOTH_ALPHA * smooth_kps + (1 - SMOOTH_ALPHA) * kps

                draw_skeleton(frame, smooth_kps)
                cv2.putText(frame, "PERSON DETECTED", (10, DISPLAY_H - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            else:
                smooth_kps = None
                cv2.putText(frame, "NO PERSON", (10, DISPLAY_H - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

            cv_col = (0, 255, 100) if has_person else (100, 100, 100)
            cv2.putText(frame, f"CV={cv_val:.4f}  thr={threshold:.4f}",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cv_col, 1)

        fps_buf.append(time.time() - t0)
        avg = sum(fps_buf) / len(fps_buf)
        fps = 1.0 / avg if avg > 0 else 0

        cv2.putText(frame, "CSI 2D Pose", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS:{fps:.1f}", (DISPLAY_W - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 1)
        cv2.putText(frame, f"buf:{len(csi_buf)}/{WINDOW_SIZE}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        if len(csi_buf) < WINDOW_SIZE:
            cv2.putText(frame, "Đang thu CSI...",
                        (DISPLAY_W // 2 - 90, DISPLAY_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)

        cv2.imshow("CSI 2D Pose  [q=quit  +=up  -=down]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("+") or key == ord("="):
            threshold += 0.002
            print(f"[threshold] → {threshold:.4f}")
        elif key == ord("-"):
            threshold = max(0.001, threshold - 0.002)
            print(f"[threshold] → {threshold:.4f}")

    reader.stop()
    cv2.destroyAllWindows()
    print("[INFO] Thoát.")


if __name__ == "__main__":
    main()
