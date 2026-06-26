"""
benchmark_models.py — Do chi so HIEU NANG TINH TOAN cho cac model da train.
Lap "Lo hong 2" cua rubric: Params, Model size, FLOPs, inference time, FPS.

Chay:
  python benchmark_models.py \
    --activity models/activity7_random.pth \
    --pose     models/pose_rx2.pth \
    --presence models/presence_rx2.pth

In bang dep + luu benchmark.json + benchmark.csv de dua vao bao cao.
FLOPs can thu vien 'thop' (tuy chon): pip install thop
"""

import argparse, json, os, time
import numpy as np
import torch

from model import (load_pose_checkpoint, load_activity_checkpoint,
                   load_presence_checkpoint, WINDOW_SIZE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def measure_latency(model, subcarriers, n_warmup=20, n_runs=200):
    """Do thoi gian suy luan 1 mau (batch=1). Tra ve ms va FPS."""
    model.eval()
    x = torch.randn(1, WINDOW_SIZE, subcarriers, device=DEVICE)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_runs):
            model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / n_runs
    return dt * 1000.0, (1.0 / dt if dt > 0 else 0.0)


def try_flops(model, subcarriers):
    """FLOPs neu co thu vien thop, nguoc lai tra None."""
    try:
        from thop import profile
        x = torch.randn(1, WINDOW_SIZE, subcarriers, device=DEVICE)
        flops, _ = profile(model, inputs=(x,), verbose=False)
        return flops
    except Exception:
        return None


def bench_one(name, path, loader):
    if not path or not os.path.exists(path):
        return None
    model, mean, std, ckpt = loader(path, DEVICE)
    sc = int(ckpt.get("config", {}).get("subcarriers", len(mean)))
    total, train = count_params(model)
    size_mb = os.path.getsize(path) / 1e6
    lat_ms, fps = measure_latency(model, sc)
    flops = try_flops(model, sc)
    return {
        "name": name, "checkpoint": os.path.basename(path),
        "subcarriers": sc,
        "params_total": int(total), "params_M": round(total / 1e6, 3),
        "model_size_MB": round(size_mb, 2),
        "flops": int(flops) if flops else None,
        "flops_M": round(flops / 1e6, 1) if flops else None,
        "latency_ms": round(lat_ms, 3), "fps": round(fps, 1),
        "device": DEVICE,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activity", default=None)
    ap.add_argument("--pose", default=None)
    ap.add_argument("--presence", default=None)
    ap.add_argument("--out", default="benchmark")
    args = ap.parse_args()

    print(f"[INFO] Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    rows = []
    for name, path, loader in [
        ("Activity (CNN+BiLSTM)", args.activity, load_activity_checkpoint),
        ("Pose (CNN+BiLSTM)",     args.pose,     load_pose_checkpoint),
        ("Presence (CNN+BiLSTM)", args.presence, load_presence_checkpoint),
    ]:
        r = bench_one(name, path, loader)
        if r:
            rows.append(r)
            print(f"[OK] {name}")

    if not rows:
        print("[ERROR] Khong load duoc model nao. Kiem tra duong dan."); return

    # In bang
    print("\n" + "=" * 92)
    print(f"  {'Model':24s} {'Params(M)':>10s} {'Size(MB)':>9s} {'FLOPs(M)':>10s} {'Latency(ms)':>12s} {'FPS':>8s}")
    print("-" * 92)
    for r in rows:
        fl = f"{r['flops_M']}" if r['flops_M'] else "N/A*"
        print(f"  {r['name']:24s} {r['params_M']:>10.3f} {r['model_size_MB']:>9.2f} "
              f"{fl:>10s} {r['latency_ms']:>12.3f} {r['fps']:>8.1f}")
    print("=" * 92)
    if any(r['flops_M'] is None for r in rows):
        print("  * FLOPs: cai 'pip install thop' de do. (Khong bat buoc)")
    print(f"  Inference: batch=1, input=(1,{WINDOW_SIZE},subcarriers), do tren {DEVICE}, trung binh 200 lan.")

    with open(f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    # CSV
    keys = ["name", "subcarriers", "params_M", "model_size_MB", "flops_M", "latency_ms", "fps", "device"]
    with open(f"{args.out}.csv", "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"\n[SAVED] {args.out}.json + {args.out}.csv")


if __name__ == "__main__":
    main()
