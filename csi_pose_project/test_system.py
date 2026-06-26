"""
test_system.py — Kiểm tra toàn bộ hệ thống TX+RX1+RX2 trước khi thu data

Chạy (CMD thường):
  python test_system.py --ports COM9 COM3

Kết quả mong đợi:
  [TX] Sóng đang phát
  [RX0/COM9] OK — XX fps
  [RX1/COM3] OK — XX fps
  [SYNC] Timestamp diff < 0.1s — OK
"""

import argparse, threading, time
from collections import deque

import serial
import numpy as np

BAUD_RATE = 921600
TEST_SECS = 10


def parse_csi_amp(line):
    try:
        line = line.strip()
        if not line.startswith("CSI_DATA"):
            return None
        b = line.index("["); e = line.rindex("]")
        raw = list(map(int, line[b+1:e].split(",")))
        if len(raw) < 4: return None
        n   = len(raw) // 2
        amp = np.abs([complex(raw[i*2+1], raw[i*2]) for i in range(n)])
        return float(np.mean(amp))
    except Exception:
        return None


class RXTester(threading.Thread):
    def __init__(self, port, node_id):
        super().__init__(daemon=True)
        self.port    = port
        self.node_id = node_id
        self.frames  = 0
        self.amps    = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)
        self.error   = None
        self.running = True

    def run(self):
        try:
            ser = serial.Serial(self.port, BAUD_RATE, timeout=1,
                                dsrdtr=False, rtscts=False)
        except Exception as e:
            self.error = str(e)
            return

        # Chờ ESP32 boot xong
        time.sleep(2)

        while self.running:
            try:
                line = ser.readline().decode("utf-8", errors="ignore")
                amp  = parse_csi_amp(line)
                if amp is not None:
                    self.frames += 1
                    self.amps.append(amp)
                    self.timestamps.append(time.time())
            except Exception:
                continue
        ser.close()

    def stop(self): self.running = False

    @property
    def fps(self):
        if len(self.timestamps) < 2:
            return 0
        dt = self.timestamps[-1] - self.timestamps[0]
        return len(self.timestamps) / dt if dt > 0 else 0

    @property
    def mean_amp(self):
        return float(np.mean(self.amps)) if self.amps else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", nargs="+", default=["COM9", "COM3"])
    args = parser.parse_args()

    print("=" * 55)
    print("  Kiểm tra hệ thống TX + RX")
    print(f"  Ports: {args.ports}")
    print("=" * 55)
    print(f"\n[INFO] Đảm bảo TX đang chạy trên laptop phụ")
    print(f"[INFO] Test trong {TEST_SECS} giây...\n")

    testers = [RXTester(p, i) for i, p in enumerate(args.ports)]
    for t in testers: t.start()

    # Realtime monitor
    for sec in range(TEST_SECS):
        time.sleep(1)
        print(f"[{sec+1:2d}s] ", end="")
        for t in testers:
            if t.error:
                print(f"  RX{t.node_id}/{t.port}: FAIL({t.error})", end="")
            else:
                print(f"  RX{t.node_id}/{t.port}: {t.frames:4d}fr "
                      f"{t.fps:.1f}fps amp={t.mean_amp:.1f}", end="")
        print()

    for t in testers: t.stop()

    # Kết quả
    print("\n" + "=" * 55)
    print("  KẾT QUẢ")
    print("=" * 55)

    all_ok = True
    for t in testers:
        if t.error:
            print(f"  RX{t.node_id}/{t.port}: ❌ FAIL — {t.error}")
            all_ok = False
        elif t.frames < 50:
            print(f"  RX{t.node_id}/{t.port}: ⚠️  Ít data ({t.frames} frames) — TX đang chạy không?")
            all_ok = False
        else:
            print(f"  RX{t.node_id}/{t.port}: ✅ OK — {t.frames} frames, {t.fps:.1f} fps, amp={t.mean_amp:.1f}")

    # Sync check
    if len(testers) >= 2 and all(len(t.timestamps) > 10 for t in testers):
        diff = abs(testers[0].timestamps[-1] - testers[1].timestamps[-1])
        if diff < 0.5:
            print(f"\n  SYNC: ✅ Timestamp diff = {diff*1000:.0f}ms — OK")
        else:
            print(f"\n  SYNC: ⚠️  Timestamp diff = {diff*1000:.0f}ms — Hơi lệch")

    print()
    if all_ok:
        print("  ✅ Hệ thống sẵn sàng thu data!")
        print(f"\n  Chạy thu data:")
        ports_str = " ".join(args.ports)
        print(f"  python data_collection.py --ports {ports_str} "
              f"--cam http://192.168.51.3:4747/video "
              f"--out dataset/raw/session_new_01.json")
    else:
        print("  ❌ Có lỗi — kiểm tra lại trước khi thu data")


if __name__ == "__main__":
    main()
