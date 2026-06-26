# ESP32 WiFi CSI Human Sensing

A university research prototype that uses **ESP32 WiFi Channel State Information (CSI)** to sense human presence, activity, and rough body pose without using a camera during inference.



## 1. What This Project Does

This project explores whether low-cost ESP32 boards can be used for WiFi-based human sensing.

The system collects CSI data from ESP32 receivers, processes the signal data, and trains deep learning models for three tasks:

| Task | Meaning | Status |
|---|---|---|
| Presence detection | Detect whether a person is in the room or the room is empty | Works best |
| Activity recognition | Classify activities such as walking, sitting, standing, falling, arms up, etc. | Main experiment |
| 2D pose estimation | Predict rough human body keypoints from CSI data | Experimental / coarse result |

## 2. Simple System Pipeline

![System Pipeline](docs/images/system_pipeline.svg)

**Simple explanation:**

```text
ESP32 transmitter sends WiFi packets
        ↓
Two ESP32 receivers collect CSI data
        ↓
CSI data is sent to a PC through USB/Serial
        ↓
Python scripts preprocess the CSI windows
        ↓
Deep learning models predict presence, activity, or rough body keypoints
```

During training, a camera and MediaPipe are used only to create ground-truth pose labels. During CSI inference, the model uses CSI data, not camera images.

## 3. Hardware Setup

The experiment uses:

- 1 ESP32 as transmitter
- 2 ESP32 boards as receivers
- ESP-NOW packets for CSI collection
- USB/Serial connection from receivers to PC
- A phone camera only for collecting MediaPipe labels during training

The two receivers are placed at different heights:

- Higher receiver: around head/upper-body height
- Lower receiver: around hip/body height

This gives the system two different signal views of the same human movement.

**Image to add:** `docs/images/esp32_csi_setup.jpg`  
**Caption:** ESP32 transmitter and two ESP32 receivers used for WiFi CSI data collection.  
**What to capture:** Take one clear photo of the room/setup and mark TX, RX high, RX low, laptop, and phone camera if possible.

## 4. Example CSI Signal

![CSI Signal Example](docs/images/tool_csi_wave.jpg)

**Caption:** Example CSI amplitude difference between empty-room and person-present cases.

This image helps reviewers quickly understand the idea: human presence and movement change the WiFi signal pattern.

## 5. Dataset and Labels

The dataset was collected by recording CSI data from ESP32 receivers. The project includes several classes such as:

- empty
- walking
- standing
- sitting
- falling
- arms held horizontally
- arms raised overhead

MediaPipe was used to generate 2D body keypoints as training labels for the pose estimation experiment.

Only a small demo sample is included in this GitHub version:

```text
dataset/raw/session_demo.json
dataset/raw/session_demo.npz
```

Large raw data, backup files, and model checkpoints are not included to keep the repository lightweight.

## 6. Model Experiments

The project tests several model types, including:

- 1D CNN
- BiLSTM
- CNN + BiLSTM
- MLP baseline

The main idea is to compare which model works well with ESP32 CSI data while still being lightweight enough for real-time use.

## 7. Key Results From the Report

| Experiment | Result | Note |
|---|---:|---|
| Presence detection | about 98.4% F1 | Most robust task |
| Activity recognition, same-session split | about 97.5% accuracy | Optimistic result because train/test data are from similar sessions |
| Activity recognition, cross-session test | about 31.0% average accuracy | Shows the system is sensitive to environment/session changes |
| 2D pose estimation | Better than mean-pose baseline | Can estimate coarse body location, but not accurate joint-level pose |

The honest conclusion is: **ESP32 CSI works well for presence detection and activity recognition in controlled settings, but accurate 2D pose estimation is still difficult with this low-cost hardware setup.**

## 8. Example Pose Evaluation

![Pose Evaluation Samples](eval_out/pose_rx2_eval_samples.png)

**Caption:** Example predicted keypoints compared with ground-truth keypoints.

**Important note:** This result is experimental. If asked in an interview, explain that the model can roughly locate the body, but it cannot predict precise human joints like camera-based pose estimation.

**Image to add:** `docs/images/mediapipe_label.png`  
**Caption:** Example MediaPipe keypoint labels used as training targets.  
**What to capture:** A screenshot of a person frame with MediaPipe skeleton/keypoints.

## 9. Repository Structure

```text
.
├── data_collection.py          # Collect CSI and pose-label data
├── convert_to_npz.py           # Convert raw session data to NPZ format
├── dataset_io.py               # Dataset loading utilities
├── model.py                    # PyTorch model definitions
├── train_model.py              # Train pose estimation model
├── train_presence.py           # Train presence detection model
├── train_activity.py           # Train activity recognition model
├── evaluate.py                 # Evaluate pose model
├── realtime_pose.py            # Realtime CSI demo script
├── metrics.py                  # Evaluation metrics
├── benchmark_models.py         # Compare model performance
├── compare_models.py           # Model comparison script
├── dataset/raw/                # Small demo data only
├── models/                     # Training plots and result images
├── eval_out/                   # Evaluation outputs
└── docs/images/                # README images and image guide
```

## 10. Installation

```bash
pip install -r requirements.txt
```

If you use CUDA on Windows, install PyTorch separately based on your CUDA version.

## 11. Example Commands

Train the pose model:

```bash
python train_model.py
```

Train the presence model:

```bash
python train_presence.py
```

Train the activity model:

```bash
python train_activity.py
```

Evaluate the pose model:

```bash
python evaluate.py
```

Run realtime demo:

```bash
python realtime_pose.py
```

## 12. Project Status

This is a university research prototype, not a production system.

The repository is kept simple for GitHub review. Large datasets, backup files, and heavy model checkpoints are excluded.
