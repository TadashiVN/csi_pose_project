# Simplified Paper Summary for GitHub Reviewers

This repository is based on a graduation research project about ESP32 WiFi CSI human sensing.

## Core idea

WiFi Channel State Information (CSI) changes when a person is present or moving in a room. This project uses low-cost ESP32 boards to collect CSI data and train models for human sensing tasks.

## Hardware

- 1 ESP32 transmitter
- 2 ESP32 receivers at different heights
- CSI data streamed to PC through USB/Serial
- Camera/MediaPipe used only for training labels

## Tasks

1. Presence detection: person vs empty room
2. Activity recognition: seven activity classes
3. 2D pose estimation: rough body keypoint prediction

## Main conclusion

ESP32 CSI is useful for presence detection and controlled activity recognition. However, accurate pose estimation is difficult with this low-cost amplitude-only setup, so the pose result should be treated as experimental.
