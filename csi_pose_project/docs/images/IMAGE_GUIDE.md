# Image Guide for README

Use simple images only. The goal is to help HR or technical reviewers understand the project quickly, not to make the README look like a scientific paper.

| File name | Priority | What to show | Suggested caption |
|---|---:|---|---|
| `system_pipeline.svg` | Already added | Simple pipeline from ESP32 CSI data to model output | Simple system pipeline for ESP32 WiFi CSI human sensing. |
| `tool_csi_wave.jpg` | Already added | CSI amplitude difference between empty room and person present | Example CSI amplitude difference between empty-room and person-present cases. |
| `esp32_csi_setup.jpg` | High | Real setup with ESP32 transmitter, two receivers, laptop, and phone camera if available | ESP32 transmitter and receivers used for WiFi CSI data collection. |
| `mediapipe_label.png` | High | A frame with MediaPipe skeleton/keypoints | MediaPipe keypoint labels used as training targets. |
| `prediction_result.png` | Optional | Predicted keypoints vs ground truth | Example predicted keypoints compared with ground-truth keypoints. |
| `training_curve.png` | Optional | Loss/accuracy curve from model training | Training curve from one CSI model experiment. |

## How to take the setup photo

For `esp32_csi_setup.jpg`, take one clear photo and mark these items if possible:

- TX: ESP32 transmitter
- RX high: receiver near upper-body/head height
- RX low: receiver near hip/body height
- Laptop/PC: receiving CSI data through USB/Serial
- Phone camera: used only for MediaPipe labels during training

## What not to add

- Do not add too many screenshots from the paper.
- Do not add raw dataset screenshots that look confusing.
- Do not claim that the pose result is highly accurate.
- Do not upload large `.bak`, `.pth`, or raw dataset files directly to GitHub.
