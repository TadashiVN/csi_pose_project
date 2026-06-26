"""
convert_to_npz.py — Chuyen file JSON cu (nang GB) sang .npz nen (nhe).
Chay 1 lan cho moi session da thu bang code cu. KHONG phai thu lai.

Chay:
  python convert_to_npz.py dataset/raw/session_01.json
  # -> tao dataset/raw/session_01.npz (nho hon ~10-20 lan)
  #    va ghi de dataset/raw/session_01.json thanh metadata nho

LUU Y: file JSON 1.6GB doc vao RAM ton ~4-6GB. Dong bot app khac truoc khi chay.
Sau khi convert xong va kiem tra .npz OK, ban co the xoa ban JSON goc da backup.
"""

import argparse, json, os, shutil
import numpy as np
from dataset_io import save_session_npz


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="Cac file .json cu can convert")
    p.add_argument("--keep-backup", action="store_true",
                   help="Giu ban JSON goc thanh *.json.bak truoc khi ghi de")
    args = p.parse_args()

    for path in args.files:
        if not path.endswith(".json"):
            print(f"[SKIP] {path} (khong phai .json)"); continue
        if not os.path.exists(path):
            print(f"[SKIP] {path} (khong ton tai)"); continue

        size_mb = os.path.getsize(path) / 1e6
        print(f"\n[CONVERT] {path}  ({size_mb:.0f} MB)")
        print("  Doc JSON vao RAM (cho chut)...")
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        samples = j["samples"]
        meta = j.get("meta", {})
        nodes = int(meta.get("nodes", 1))
        print(f"  {len(samples)} samples, nodes={nodes}")

        if args.keep_backup:
            shutil.copy(path, path + ".bak")
            print(f"  [BACKUP] {path}.bak")

        print("  Dang nen ra .npz...")
        csi = np.array([s["csi"]       for s in samples], dtype=np.float32)
        kps = np.array([s["keypoints"] for s in samples], dtype=np.float32)
        acts = [s["activity"]  for s in samples]
        tss  = [s.get("timestamp", 0.0) for s in samples]
        del j, samples   # giai phong RAM truoc khi luu

        npz_path, meta_path, dist = save_session_npz(path, csi, kps, acts, tss, nodes, meta)
        new_mb = os.path.getsize(npz_path) / 1e6
        print(f"  [SAVED] {npz_path}  ({new_mb:.0f} MB, giam {size_mb/max(new_mb,1):.0f}x)")
        print(f"  [SAVED] {meta_path}  (metadata)")
        print(f"  Distribution: {dist}")

    print("\n[DONE] Train tro thang vao file .npz (hoac .json — tu chuyen huong).")


if __name__ == "__main__":
    main()
