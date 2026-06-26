"""
model.py — Models dùng chung cho train + inference.
- CSIPoseModel:           CSI window → 17 keypoints (x,y)
- CSIPresenceClassifier:  CSI window → có người / không có người
- CSIActivityClassifier:  CSI window → 1 trong N activity (walk/stand/sit/fall/empty)

GHI CHÚ MULTI-RX:
  subcarriers = số_RX × 52. Ví dụ 1 RX → 52, 2 RX → 104, 3 RX → 156.
  Số này được lưu trong checkpoint["config"]["subcarriers"] và tự đọc lại khi load.
"""

import torch
import torch.nn as nn

SUBCARRIERS_PER_RX = 52
SUBCARRIERS = 52          # mặc định 1 RX; train script sẽ override theo --rx
WINDOW_SIZE = 20
N_KEYPOINTS = 17
N_ACTIVITIES = 7          # walk, stand, sit, fall, empty, arms_horizontal, arms_up
ACTIVITY_NAMES = ["walk", "stand", "sit", "fall", "empty",
                  "arms_horizontal", "arms_up"]


class CSIPoseModel(nn.Module):
    """
    Input:  (batch, time=20, subcarriers)
    Output: (batch, 34) <- 17 keypoints x 2 (normalized [0,1])
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
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_keypoints * 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)        # (batch, subcarriers, time)
        x = self.cnn(x)               # (batch, 128, time)
        x = x.permute(0, 2, 1)        # (batch, time, 128)
        x, _ = self.lstm(x)           # (batch, time, 512)
        return self.head(x[:, -1, :])


class CSIPresenceClassifier(nn.Module):
    """
    Binary: CSI window -> co nguoi (1) / khong co nguoi (0).
    Input:  (batch, time=20, subcarriers)
    Output: (batch, 1) — sigmoid probability.
    """
    def __init__(self, subcarriers=SUBCARRIERS):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(subcarriers, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=64, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)        # (batch, subcarriers, time)
        x = self.cnn(x)               # (batch, 64, time)
        x = x.permute(0, 2, 1)        # (batch, time, 64)
        x, _ = self.lstm(x)           # (batch, time, 128)
        return self.head(x[:, -1, :]) # (batch, 1)


class CSIActivityClassifier(nn.Module):
    """
    CSI window -> activity logits (N lop). Dung CrossEntropyLoss.
    Input:  (batch, time=20, subcarriers)
    Output: (batch, n_activities) — logits (KHONG softmax san).
    """
    def __init__(self, subcarriers=SUBCARRIERS, n_activities=N_ACTIVITIES):
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
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_activities),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.head(x[:, -1, :])


# --- Cac kien truc SO SANH (baseline) cho activity ---------------------------
# Dung de so sanh >=2 mo hinh theo yeu cau rubric. Tat ca cung input (batch,20,sc).

class ActCNNOnly(nn.Module):
    """Chi CNN 1D + global pooling, KHONG co LSTM. Kiem tra dong gop cua LSTM."""
    def __init__(self, subcarriers=SUBCARRIERS, n_activities=N_ACTIVITIES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(subcarriers, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_activities),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        return self.head(x)


class ActLSTMOnly(nn.Module):
    """Chi BiLSTM tren chuoi thoi gian, KHONG co CNN. Kiem tra dong gop cua CNN."""
    def __init__(self, subcarriers=SUBCARRIERS, n_activities=N_ACTIVITIES):
        super().__init__()
        self.lstm = nn.LSTM(subcarriers, 256, num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_activities),
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.head(x[:, -1, :])


class ActMLP(nn.Module):
    """Baseline don gian: flatten toan bo cua so -> MLP. Moc 'khong CNN, khong LSTM'."""
    def __init__(self, subcarriers=SUBCARRIERS, n_activities=N_ACTIVITIES,
                 window=WINDOW_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * subcarriers, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_activities),
        )

    def forward(self, x):
        return self.net(x)


# Dang ky de chon bang ten
ACTIVITY_MODELS = {
    "cnn_bilstm": CSIActivityClassifier,   # mo hinh chinh
    "cnn_only":   ActCNNOnly,
    "lstm_only":  ActLSTMOnly,
    "mlp":        ActMLP,
}


# --- Loaders -----------------------------------------------------------------
def _subcarriers_from_ckpt(ckpt, fallback=SUBCARRIERS):
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    return int(cfg.get("subcarriers", fallback))


def load_pose_checkpoint(path, device="cpu"):
    """Load pose model. Tu dung model dung so subcarriers da train."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sc   = _subcarriers_from_ckpt(ckpt)
    model = CSIPoseModel(subcarriers=sc).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"], ckpt


# Alias de tuong thich evaluate.py (von import load_checkpoint)
load_checkpoint = load_pose_checkpoint


def load_presence_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sc   = _subcarriers_from_ckpt(ckpt)
    model = CSIPresenceClassifier(subcarriers=sc).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"], ckpt


def load_activity_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sc   = _subcarriers_from_ckpt(ckpt)
    n_act = int(ckpt.get("config", {}).get("n_activities", N_ACTIVITIES))
    model = CSIActivityClassifier(subcarriers=sc, n_activities=n_act).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"], ckpt
