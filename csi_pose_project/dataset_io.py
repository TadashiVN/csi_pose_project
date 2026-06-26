"""
dataset_io.py — Load/save session data, dung chung cho moi script.

Dinh dang MOI: .npz (numpy nen) — nho hon JSON ~10-20 lan, load nhanh.
  Keys: csi (N, nodes, 20, 52) float32 | keypoints (N, 17, 2) float32
        activities (N,) str | timestamps (N,) float64 | nodes int

Van doc duoc .json CU (data thu truoc day). Neu duong dan .json chi con
metadata (sau khi data_collection moi tach ra), tu dong tim file .npz cung ten.

Dung:
  from dataset_io import load_session
  sess = load_session("dataset/raw/session_01.npz")
  sess["csi"]        # (N, nodes, 20, 52)
  sess["keypoints"]  # (N, 17, 2)
  sess["activities"] # list[str], len N
  sess["nodes"]      # int
"""

import json
import os
import numpy as np


def _sibling_npz(path):
    base = os.path.splitext(path)[0]
    cand = base + ".npz"
    return cand if os.path.exists(cand) else None


def load_session(path):
    """Tra ve dict: csi (N,nodes,20,52) f32, keypoints (N,17,2) f32,
    activities list[str], nodes int."""
    # .json nhung co the la metadata-only -> chuyen sang .npz cung ten
    if path.endswith(".json"):
        npz = _sibling_npz(path)
        if npz is not None:
            with open(path, encoding="utf-8") as f:
                head = json.load(f)
            if "samples" not in head or not head["samples"]:
                path = npz   # file json chi con meta -> dung npz

    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        return {
            "csi":        np.asarray(d["csi"], dtype=np.float32),
            "keypoints":  np.asarray(d["keypoints"], dtype=np.float32),
            "activities": [str(a) for a in d["activities"]],
            "nodes":      int(d["nodes"]) if "nodes" in d else int(d["csi"].shape[1]),
        }

    # Fallback: doc .json cu (ton RAM hon — chi dung 1 lan de convert)
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    samples = j["samples"]
    csi = np.array([s["csi"] for s in samples], dtype=np.float32)
    kps = np.array([s["keypoints"] for s in samples], dtype=np.float32)
    acts = [s["activity"] for s in samples]
    nodes = int(j.get("meta", {}).get("nodes", csi.shape[1] if csi.ndim == 4 else 1))
    del j, samples
    return {"csi": csi, "keypoints": kps, "activities": acts, "nodes": nodes}


def save_session_npz(out_path, csi, keypoints, activities, timestamps, nodes, meta=None):
    """Luu data dang .npz nen + 1 file .meta.json nho de doc phan bo.
    out_path co the la .json hoac .npz — du kieu gi cung luu .npz + .meta.json."""
    base = os.path.splitext(out_path)[0]
    npz_path  = base + ".npz"
    meta_path = base + ".json"   # giu duoi .json cho quen, nhung chi chua meta

    csi = np.asarray(csi, dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32)
    activities = np.asarray([str(a) for a in activities])
    timestamps = np.asarray(timestamps, dtype=np.float64)

    np.savez_compressed(
        npz_path,
        csi=csi, keypoints=keypoints,
        activities=activities, timestamps=timestamps,
        nodes=np.int64(nodes),
    )

    from collections import Counter
    dist = dict(Counter(str(a) for a in activities))
    meta_out = {"meta": (meta or {}), "distribution": dist,
                "total_samples": int(len(csi)), "nodes": int(nodes),
                "npz_file": os.path.basename(npz_path),
                "note": "Data thuc nam trong file .npz. File .json nay chi chua metadata."}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    return npz_path, meta_path, dist
