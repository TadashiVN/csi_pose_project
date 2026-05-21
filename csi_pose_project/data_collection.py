"""
data_collection.py — Thu CSI + MediaPipe Pose đồng thời
Phím: 1=walk 2=stand 3=sit 4=fall 5=empty | s=ghi x=dừng q=thoát
Chạy:
  python data_collection.py --ports COM3 --cam http://192.168.51.3:4747/video --out dataset/raw/session_01.json
"""

import argparse, json, os, queue, threading, time
from collections import Counter, deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import serial

BAUD_RATE       = 921600
WINDOW_SIZE     = 20
WINDOW_STRIDE   = 10
MIN_CONF        = 0.6
SUBCARRIERS_USE = 20
ALIGN_TOLERANCE = 0.3
SAMPLE_INTERVAL = 0.1
EMPTY_KEYPOINTS = [[0.0, 0.0]] * 17  # keypoints rỗng cho activity empty

COCO_IDX = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
COCO_NAMES = [
    "nose","left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_eye","right_eye","left_ear","right_ear",
]
ACTIVITY_MAP = {"1":"walk","2":"stand","3":"sit","4":"fall","5":"empty"}


def parse_csi_amp(line):
    try:
        line = line.strip()
        if not line.startswith("CSI_DATA"):
            return None
        b_start  = line.index("[")
        b_end    = line.rindex("]")
        raw = list(map(int, line[b_start+1:b_end].split(",")))
        if len(raw) < 4:
            return None
        n    = len(raw) // 2
        amp  = np.abs([complex(raw[i*2+1], raw[i*2]) for i in range(n)])
        idx  = np.linspace(0, len(amp)-1, SUBCARRIERS_USE, dtype=int)
        return amp[idx]
    except Exception:
        return None


class CSIReader(threading.Thread):
    def __init__(self, port, node_id, out_queue):
        super().__init__(daemon=True)
        self.port = port; self.node_id = node_id
        self.q = out_queue; self.running = True

    def run(self):
        try:
            ser = serial.Serial(self.port, BAUD_RATE, timeout=1)
            print(f"[CSI node {self.node_id}] OK {self.port}")
        except serial.SerialException as e:
            print(f"[CSI node {self.node_id}] FAIL {e}"); return
        while self.running:
            try:
                amp = parse_csi_amp(ser.readline().decode("utf-8", errors="ignore"))
                if amp is not None:
                    self.q.put({"node_id": self.node_id, "timestamp": time.time(), "amp": amp.tolist()})
            except Exception:
                continue
        ser.close()

    def stop(self): self.running = False


class PoseReader(threading.Thread):
    def __init__(self, cam_source, pose_buffer, lock):
        super().__init__(daemon=True)
        self.source = cam_source; self.buffer = pose_buffer
        self.lock = lock; self.running = True
        self._frame = None; self._conf = 0.0
        self._frame_lock = threading.Lock()

    def get_display(self):
        """Atomic snapshot (frame_copy, conf) cho main thread."""
        with self._frame_lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._conf

    def _open_cap(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG if isinstance(self.source, str) else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                            min_detection_confidence=MIN_CONF, min_tracking_confidence=MIN_CONF)
        mp_draw = mp.solutions.drawing_utils
        cap = self._open_cap()
        for _ in range(10): cap.read()
        print(f"[Camera] OK {self.source}")

        while self.running:
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                print("[Camera] Reconnect...")
                cap.release(); time.sleep(1); cap = self._open_cap(); continue

            ts  = time.time()
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            keypoints = None; conf = 0.0

            if res.pose_landmarks:
                lm   = res.pose_landmarks.landmark
                conf = float(np.mean([lm[i].visibility for i in COCO_IDX]))
                if conf >= MIN_CONF:
                    keypoints = [[lm[i].x, lm[i].y] for i in COCO_IDX]
                    mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(0,200,255), thickness=2))

            with self._frame_lock:
                self._frame = frame
                self._conf = conf

            # Lưu vào buffer (kể cả khi không có người — dùng cho empty)
            with self.lock:
                self.buffer.append({"timestamp": ts, "keypoints": keypoints, "conf": conf})
                cutoff = ts - 5.0
                while self.buffer and self.buffer[0]["timestamp"] < cutoff:
                    self.buffer.popleft()

        cap.release(); pose.close()

    def stop(self): self.running = False


def find_pose(pose_buffer, ts, lock, is_empty=False):
    with lock:
        if not pose_buffer:
            return None
        best = min(pose_buffer, key=lambda p: abs(p["timestamp"] - ts))
        if abs(best["timestamp"] - ts) > ALIGN_TOLERANCE:
            return None
        # Empty: không cần keypoints — trả về keypoints rỗng
        if is_empty:
            return {"keypoints": EMPTY_KEYPOINTS}
        # Activity khác: cần keypoints hợp lệ
        if best["keypoints"] is None:
            return None
        return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", nargs="+", default=["COM3"])
    parser.add_argument("--cam",   default="0")
    parser.add_argument("--out",   default="dataset/raw/session_01.json")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir: os.makedirs(out_dir, exist_ok=True)

    try: cam_src = int(args.cam)
    except ValueError: cam_src = args.cam

    print("="*55)
    print("  WiFi CSI + Pose Data Collection")
    print(f"  Ports : {args.ports}")
    print(f"  Camera: {cam_src}")
    print(f"  Output: {args.out}")
    print("="*55)
    print("  [1]walk [2]stand [3]sit [4]fall [5]empty")
    print("  [s]Ghi  [x]Dừng  [q]Thoát\n")

    csi_queue   = queue.Queue()
    pose_buffer = deque()
    pose_lock   = threading.Lock()
    node_bufs   = {i: deque(maxlen=300) for i in range(len(args.ports))}

    samples = []; activity = None; recording = False; last_win_t = time.time()

    readers = [CSIReader(p, i, csi_queue) for i, p in enumerate(args.ports)]
    for r in readers: r.start()
    pose_reader = PoseReader(cam_src, pose_buffer, pose_lock)
    pose_reader.start()
    time.sleep(2)
    print("[INFO] Sẵn sàng.\n")

    try:
        while True:
            while not csi_queue.empty():
                item = csi_queue.get_nowait()
                node_bufs[item["node_id"]].append((item["timestamp"], item["amp"]))

            disp, conf = pose_reader.get_display()
            if disp is not None:
                h, w = disp.shape[:2]
                b_col = (0,0,180) if recording else (50,50,50)
                cv2.rectangle(disp, (0,0), (w,44), b_col, -1)
                status = f"REC [{activity}]" if recording else f"STANDBY [{activity or '---'}]"
                cv2.putText(disp, status,              (10,30),   cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)
                cv2.putText(disp, f"n={len(samples)}", (w-130,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (180,255,180), 1)
                c_col = (0,220,0) if conf >= MIN_CONF else (0,0,220)
                cv2.putText(disp, f"pose:{conf:.2f}", (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_col, 1)
                for nid, buf in node_bufs.items():
                    cv2.putText(disp, f"node{nid}:{len(buf)}fr", (10,88+nid*22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,0), 1)
                cv2.imshow("CSI Collector  [s=rec x=stop q=quit]", disp)

            now = time.time()
            if recording and activity and now - last_win_t >= SAMPLE_INTERVAL:
                last_win_t = now
                buf0 = list(node_bufs[0])
                if len(buf0) >= WINDOW_SIZE:
                    mid_ts     = buf0[-WINDOW_SIZE:][WINDOW_SIZE//2][0]
                    is_empty   = (activity == "empty")
                    match      = find_pose(pose_buffer, mid_ts, pose_lock, is_empty)
                    if match:
                        tensor = []; ok = True
                        for nid in range(len(args.ports)):
                            b = list(node_bufs[nid])
                            if len(b) < WINDOW_SIZE: ok = False; break
                            tensor.append([f[1] for f in b[-WINDOW_SIZE:]])
                        if ok:
                            samples.append({
                                "timestamp": mid_ts,
                                "activity":  activity,
                                "nodes":     len(args.ports),
                                "csi":       tensor,
                                "keypoints": match["keypoints"],
                            })
                            if len(samples) % 100 == 0:
                                print(f"  [+] {len(samples)} — {dict(Counter(s['activity'] for s in samples))}")

            key = cv2.waitKey(30) & 0xFF
            if   key == ord("q"): break
            elif key == ord("s"):
                if activity: recording = True; last_win_t = time.time(); print(f"\n  ▶ REC [{activity}]")
                else: print("  [!] Chưa chọn activity")
            elif key == ord("x"):
                recording = False
                print(f"  ■ STOP — {len(samples)} samples")
            elif chr(key) in ACTIVITY_MAP:
                activity = ACTIVITY_MAP[chr(key)]; recording = False
                print(f"\n  → [{activity}]  nhấn [s] để ghi")

    except KeyboardInterrupt:
        pass
    finally:
        for r in readers: r.stop()
        pose_reader.stop()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Tổng: {len(samples)} samples")
        if samples:
            dist = dict(Counter(s["activity"] for s in samples))
            out  = {
                "meta": {
                    "created":        datetime.now().isoformat(),
                    "ports":          args.ports,
                    "nodes":          len(args.ports),
                    "subcarriers":    SUBCARRIERS_USE,
                    "window_size":    WINDOW_SIZE,
                    "window_stride":  WINDOW_STRIDE,
                    "align_tol_s":    ALIGN_TOLERANCE,
                    "keypoints_idx":  COCO_IDX,
                    "keypoints_name": COCO_NAMES,
                    "activities":     list(ACTIVITY_MAP.values()),
                    "total_samples":  len(samples),
                    "distribution":   dist,
                },
                "samples": samples,
            }
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"[SAVED] {args.out}")
            print("  Distribution:")
            for act, cnt in sorted(dist.items()):
                print(f"    {act:8s}: {cnt:4d}  {'█'*(cnt//5)}")
        else:
            print("[!] Không có data.")

if __name__ == "__main__":
    main()
