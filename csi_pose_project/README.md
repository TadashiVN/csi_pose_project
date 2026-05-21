# Real-Time 2D Human Pose Estimation Using Wi-Fi CSI on ESP32

> Đồ án tốt nghiệp — Ước lượng tư thế người 2D theo thời gian thực sử dụng CSI WiFi trên hệ thống nhúng ESP32

## Tóm tắt

Hệ thống ước lượng tư thế người 17 keypoints (COCO subset) chỉ dùng tín hiệu WiFi
Channel State Information (CSI) thu từ 2 board ESP32 DevKit V1, không cần camera lúc
inference. Demo "cardboard barrier" minh hoạ khả năng "nhìn xuyên vật cản" của sóng
WiFi 2.4 GHz.

| Hạng mục | Giá trị |
|---|---|
| Phần cứng | 2× ESP32 DevKit V1 (TX–RX qua ESP-NOW) |
| Input | 192 subcarriers HT40 → giữ 20 subcarriers, window 20 frames |
| Output | 17 keypoints (x,y) ∈ [0,1] |
| Model | CNN (Conv1D ×2) + BiLSTM (2 layers, 256 hidden) + MLP head |
| Dataset | ~9800 samples / 2 session / 1 phòng 3.6 × 2.7 m |
| Ground truth | MediaPipe Pose (auto-label song song khi thu) |

## Cấu trúc dự án

```
csi_pose_project/
├── model.py                  Model CSIPoseModel (shared)
├── metrics.py                PCK, MPJPE, per-keypoint
├── data_collection.py        Thu CSI + MediaPipe pose
├── train_model.py            Train + augmentation + early stopping + log PCK
├── evaluate.py               Eval trên test set, plot per-keypoint + samples
├── calibrate_threshold.py    Tự động tìm CV threshold cho empty detection
├── realtime_pose.py          Inference realtime → skeleton trên nền đen
├── requirements.txt
├── README.md
├── HUONG_DAN_TRIEN_KHAI.md   Hướng dẫn end-to-end (đọc file này!)
├── CLAUDE_PROJECT_CONTEXT.md Context kỹ thuật chi tiết
├── dataset/raw/              session_01.json, session_02.json
├── models/                   pose_model.pth (best checkpoint)
└── paper/                    7 papers tham khảo
```

## Chạy nhanh

Xem [HUONG_DAN_TRIEN_KHAI.md](HUONG_DAN_TRIEN_KHAI.md) để biết quy trình đầy đủ. Tóm tắt:

```bash
# 1. Cài đặt
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Train với split-by-session (chống leakage) + augmentation
python train_model.py \
    --data dataset/raw/session_01.json dataset/raw/session_02.json \
    --out models/pose_model.pth \
    --split-by-session --augment --epochs 80 --patience 15

# 3. Evaluate đầy đủ PCK + per-keypoint + plot
python evaluate.py --model models/pose_model.pth --data dataset/raw/session_02.json

# 4. Calibrate CV threshold cho empty detection
python calibrate_threshold.py --data dataset/raw/session_01.json dataset/raw/session_02.json

# 5. Realtime
python realtime_pose.py --port COM3 --model models/pose_model.pth
```

## Giới hạn đã biết

- 1 cặp TX–RX → không phân biệt tay trái/phải, không multi-person
- Environment-specific: phải train lại khi đổi phòng
- Chỉ dùng amplitude, chưa dùng phase
- ESP32 CSI có amplitude drift theo nhiệt độ

## Tham khảo chính

7 papers trong [paper/](paper/) — xem [CLAUDE_PROJECT_CONTEXT.md](CLAUDE_PROJECT_CONTEXT.md) Section 11
để biết lý do chọn từng paper.
