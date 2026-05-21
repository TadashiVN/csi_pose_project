"""
model.py — Định nghĩa CSIPoseModel dùng chung cho train + inference.
Tách ra để tránh sync sai giữa train_model.py và realtime_pose.py.
"""

import torch
import torch.nn as nn

SUBCARRIERS = 20
WINDOW_SIZE = 20
N_KEYPOINTS = 17


class CSIPoseModel(nn.Module):
    """
    Input:  (batch, time=20, subcarriers=20)
    Output: (batch, 34)  ← 17 keypoints × 2 (normalized [0,1])
    """

    def __init__(self, subcarriers=SUBCARRIERS, n_keypoints=N_KEYPOINTS):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(subcarriers, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=128, hidden_size=256, num_layers=2,
            batch_first=True, dropout=0.3, bidirectional=True,
        )

        self.head = nn.Sequential(
            nn.Linear(256 * 2, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_keypoints * 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, time, subcarriers)
        x = x.permute(0, 2, 1)           # (batch, subcarriers, time)
        x = self.cnn(x)                  # (batch, 128, time)
        x = x.permute(0, 2, 1)           # (batch, time, 128)
        x, _ = self.lstm(x)              # (batch, time, 512)
        return self.head(x[:, -1, :])    # (batch, 34)


def load_checkpoint(path, device="cpu"):
    """Helper: load checkpoint + dựng model + return (model, mean, std, meta)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CSIPoseModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"], ckpt
