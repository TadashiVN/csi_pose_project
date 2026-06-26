"""
baseline_meanpose.py — Baseline "du doan pose trung binh".

Y NGHIA: lay trung binh tat ca keypoints tren TRAIN, dung lam du doan
co dinh cho moi sample TEST. Neu model that su KHONG hon baseline nay
bao nhieu -> model chua hoc pose, chi doan "nguoi dung giua phong".
Day la con so PHAI biet truoc khi tin vao PCK cua model.

Chay:
  python baseline_meanpose.py \
    --train dataset/raw/session_01.json \
    --test  dataset/raw/session_02.json
"""

import argparse, json
import numpy as np
from metrics import compute_metrics, format_metrics, KEYPOINT_NAMES
from dataset_io import load_session

N_KP = len(KEYPOINT_NAMES)


def load_keypoints(files):
    kps = []
    for f in files:
        sess = load_session(f)
        for k in sess["keypoints"]:
            k = np.asarray(k, dtype=np.float32)
            if k.shape == (N_KP, 2) and k.sum() > 0:   # bo empty
                kps.append(k)
    return np.array(kps) if kps else np.zeros((0, N_KP, 2), dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--test",  nargs="+", required=True)
    args = p.parse_args()

    train_kps = load_keypoints(args.train)
    test_kps  = load_keypoints(args.test)
    print(f"[DATA] train poses={len(train_kps)} | test poses={len(test_kps)}")
    if len(train_kps) == 0 or len(test_kps) == 0:
        print("[ERROR] Khong du data."); return

    mean_pose = train_kps.mean(axis=0)                      # (17,2)
    preds = np.repeat(mean_pose[None], len(test_kps), axis=0)

    print("\n###### BASELINE: MEAN POSE ######")
    print(format_metrics(compute_metrics(preds, test_kps)))
    print("\n>> Model cua ban PHAI cao hon ro ret cac so PCK nay thi moi"
          "\n>> chung minh duoc no thuc su hoc pose (khong chi doan trung binh).")


if __name__ == "__main__":
    main()
