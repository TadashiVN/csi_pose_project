# csi_pose_project

Real-time 2D human pose estimation using WiFi Channel State Information (CSI) on ESP32-based embedded systems.  
This project predicts 17 human body keypoints from wireless signal patterns without requiring cameras during inference.

The system combines ESP32 CSI collection, MediaPipe-generated pose labels, and deep learning (CNN + BiLSTM) to reconstruct human skeletons in real time using only WiFi signals.

---

## Overview

Traditional pose estimation relies on RGB cameras, which introduce privacy concerns and line-of-sight limitations.  
This project explores a low-cost alternative using WiFi CSI captured from ESP32 devices.

Pipeline:

```text
ESP32 CSI → Preprocessing → CNN + BiLSTM → 17 Keypoints → 2D Skeleton
```
<img width="1068" height="342" alt="image" src="https://github.com/user-attachments/assets/6ef0caae-6f67-43bd-b890-56b98138a9fd" />

## Results

<img width="1225" height="529" alt="image" src="https://github.com/user-attachments/assets/951a95d0-c372-456e-9867-594167a56dfd" />
