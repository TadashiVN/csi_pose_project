"""
fix_label.py — Sua nhan trong file .npz da thu (khong phai thu lai).

Doi nhan:
  python fix_label.py dataset/raw/session_03.npz --from arms_up --to arms_horizontal

Xoa han mot nhan:
  python fix_label.py dataset/raw/session_03.npz --drop arms_up

Mac dinh GHI DE tai cho + tao ban .bak.npz. Them --out de xuat file moi.
"""

import argparse, os, shutil
import numpy as np
from collections import Counter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="File .npz can sua")
    p.add_argument("--from", dest="src", help="Nhan cu (dung voi --to)")
    p.add_argument("--to",   dest="dst", help="Nhan moi (dung voi --from)")
    p.add_argument("--drop", help="Xoa han cac sample mang nhan nay")
    p.add_argument("--out",  help="Xuat ra file moi (mac dinh: ghi de + .bak)")
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    if not args.npz.endswith(".npz") or not os.path.exists(args.npz):
        print(f"[ERROR] Khong tim thay {args.npz} (.npz)"); return
    if not args.drop and not (args.src and args.dst):
        print("[ERROR] Dung --from X --to Y de doi, hoac --drop X de xoa."); return

    d = np.load(args.npz, allow_pickle=True)
    data = {k: d[k] for k in d.files}
    acts = np.array([str(a) for a in data["activities"]])
    print(f"[BEFORE] {dict(Counter(acts))}")

    if args.drop:
        keep = acts != args.drop
        n_drop = int((~keep).sum())
        if n_drop == 0:
            print(f"[WARN] Khong co sample nao mang nhan '{args.drop}'."); return
        for k in data:
            if hasattr(data[k], "shape") and data[k].ndim >= 1 and data[k].shape[0] == len(acts):
                data[k] = data[k][keep]
        print(f"[DROP] Xoa {n_drop} sample nhan '{args.drop}'")
    else:
        mask = acts == args.src
        n_chg = int(mask.sum())
        if n_chg == 0:
            print(f"[WARN] Khong co sample nao mang nhan '{args.src}'."); return
        acts[mask] = args.dst
        data["activities"] = acts
        print(f"[RENAME] {n_chg} sample: '{args.src}' -> '{args.dst}'")

    print(f"[AFTER]  {dict(Counter(str(a) for a in data['activities']))}")

    out = args.out or args.npz
    if out == args.npz and not args.no_backup:
        bak = args.npz.replace(".npz", ".bak.npz")
        shutil.copy(args.npz, bak)
        print(f"[BACKUP] {bak}")

    np.savez_compressed(out, **data)
    print(f"[SAVED]  {out}")


if __name__ == "__main__":
    main()
