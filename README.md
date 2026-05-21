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

<img width="1013" height="632" alt="image" src="https://github.com/user-attachments/assets/67f87237-9c83-4c02-aad3-392c6bb7e531" />
