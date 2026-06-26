"""
metrics.py — Metric chuẩn cho 2D Pose Estimation
- PCK (Percentage of Correct Keypoints) ở nhiều threshold
- MPJPE (Mean Per-Joint Position Error) normalized
- Per-keypoint breakdown

Quy ước: keypoints đã normalize về [0,1] (cùng frame size).
"""

import numpy as np

KEYPOINT_NAMES = [
    "nose", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip", "L_knee", "R_knee",
    "L_ankle", "R_ankle", "L_eye", "R_eye", "L_ear", "R_ear",
]

# Index theo COCO subset (xem data_collection.py)
L_SHOULDER, R_SHOULDER = 1, 2
L_HIP, R_HIP = 7, 8


def _reference_distance(gt):
    """
    Khoảng cách quy chiếu để normalize sai số.
    Dùng đường chéo torso (L_shoulder ↔ R_hip).
    Nếu torso degenerate (< 0.05), fallback dùng bounding-box diagonal của GT.
    """
    diag_torso = np.linalg.norm(gt[L_SHOULDER] - gt[R_HIP])
    if diag_torso >= 0.05:
        return diag_torso
    mins = gt.min(axis=0)
    maxs = gt.max(axis=0)
    bbox = np.linalg.norm(maxs - mins)
    return max(bbox, 0.1)  # tránh chia 0


def per_sample_errors(pred, gt):
    """
    pred, gt: (N_KEYPOINTS, 2) numpy
    Returns: (errors, ref_dist)
      errors: (N_KEYPOINTS,) — euclidean distance từng keypoint
      ref_dist: scalar — distance dùng để normalize
    """
    errors = np.linalg.norm(pred - gt, axis=1)
    ref_dist = _reference_distance(gt)
    return errors, ref_dist


def compute_metrics(preds, gts, thresholds=(0.05, 0.1, 0.2, 0.5)):
    """
    preds, gts: list/array shape (N_samples, N_keypoints, 2)
    Returns dict:
      - mpjpe: mean error pixel-equivalent (normalized [0,1] units)
      - mpjpe_normalized: mean error / ref_dist
      - pck_<thr>: PCK overall + per-keypoint dict
      - per_keypoint_error: mean error per keypoint
    """
    preds = np.asarray(preds, dtype=np.float32)
    gts = np.asarray(gts, dtype=np.float32)
    assert preds.shape == gts.shape, f"shape mismatch {preds.shape} vs {gts.shape}"

    n_samples, n_kp, _ = preds.shape

    # Lọc samples non-empty (gt.sum > 0)
    valid_mask = gts.reshape(n_samples, -1).sum(axis=1) > 0
    preds_v = preds[valid_mask]
    gts_v = gts[valid_mask]

    if len(preds_v) == 0:
        return {"error": "no valid (non-empty) samples"}

    all_errors = np.zeros((len(preds_v), n_kp), dtype=np.float32)
    all_normalized = np.zeros((len(preds_v), n_kp), dtype=np.float32)

    for i in range(len(preds_v)):
        errs, ref = per_sample_errors(preds_v[i], gts_v[i])
        all_errors[i] = errs
        all_normalized[i] = errs / ref

    metrics = {
        "n_samples_total": int(n_samples),
        "n_samples_valid": int(len(preds_v)),
        "mpjpe": float(all_errors.mean()),
        "mpjpe_normalized": float(all_normalized.mean()),
        "per_keypoint_error": {
            KEYPOINT_NAMES[k]: float(all_errors[:, k].mean()) for k in range(n_kp)
        },
        "per_keypoint_error_normalized": {
            KEYPOINT_NAMES[k]: float(all_normalized[:, k].mean()) for k in range(n_kp)
        },
    }

    for thr in thresholds:
        correct = all_normalized < thr  # (samples, kp) bool
        metrics[f"pck@{thr}"] = float(correct.mean())
        metrics[f"pck@{thr}_per_keypoint"] = {
            KEYPOINT_NAMES[k]: float(correct[:, k].mean()) for k in range(n_kp)
        }

    return metrics


def format_metrics(metrics):
    """Format đẹp để in ra console / lưu báo cáo."""
    if "error" in metrics:
        return f"[!] {metrics['error']}"

    lines = []
    lines.append("=" * 60)
    lines.append("POSE ESTIMATION METRICS")
    lines.append("=" * 60)
    lines.append(f"Samples (valid/total): {metrics['n_samples_valid']}/{metrics['n_samples_total']}")
    lines.append(f"MPJPE (normalized [0,1] units): {metrics['mpjpe']:.4f}")
    lines.append(f"MPJPE / torso ratio:            {metrics['mpjpe_normalized']:.4f}")
    lines.append("")
    lines.append("PCK overall:")
    for k, v in metrics.items():
        if k.startswith("pck@") and not k.endswith("per_keypoint"):
            lines.append(f"  {k:12s}: {v*100:6.2f}%")
    lines.append("")
    lines.append("Per-keypoint error (normalized by torso):")
    for name, err in metrics["per_keypoint_error_normalized"].items():
        bar = "█" * int(err * 50)
        lines.append(f"  {name:13s}: {err:.3f}  {bar}")
    lines.append("=" * 60)
    return "\n".join(lines)
