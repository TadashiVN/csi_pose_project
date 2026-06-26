"""
inspect_ckpt.py — In dac ta that cua moi checkpoint .pth de biet file nao la 2-RX (104).
Chay:  python inspect_ckpt.py
       python inspect_ckpt.py models\pose_rx2.pth   (chi 1 file)
"""
import sys, glob, os
import numpy as np
import torch


def inspect(path):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  {path}: LOI doc ({e})"); return
    if not isinstance(ck, dict) or "model_state" not in ck:
        print(f"  {path}: khong phai checkpoint chuan (khong co 'model_state')"); return
    st = ck["model_state"]

    # feature dim theo mean (dang tin nhat: normalize tinh tren truc subcarrier)
    feat_mean = len(np.asarray(ck["mean"]).ravel()) if "mean" in ck else None
    # in-channels cua conv dau (model can dung input co bay nhieu feature)
    conv_in = None
    for k in st:
        if k.endswith("cnn.0.weight") or k == "cnn.0.weight":
            conv_in = int(st[k].shape[1]); break
    meta = ck.get("meta", {})
    vloss = ck.get("val_loss", None)
    nodes = (conv_in // 52) if (conv_in and conv_in % 52 == 0) else "?"

    print(f"\n  FILE: {path}")
    print(f"    conv in-channels (input feature can co) = {conv_in}")
    print(f"    len(mean) (feature dim khi train)        = {feat_mean}")
    print(f"    meta                                     = {meta}")
    print(f"    val_loss                                 = {vloss}")
    print(f"    => can {nodes} node (52 subcarrier/node). "
          f"{'2 RX' if conv_in==104 else '1 RX' if conv_in==52 else 'KHONG ro / sai'}")


def main():
    paths = sys.argv[1:] or sorted(glob.glob("models/*.pth")) or sorted(glob.glob("*.pth"))
    if not paths:
        print("Khong thay file .pth nao. Chay trong E:\\csi_pose_project hoac chi ro duong dan.")
        return
    print(f"Kiem tra {len(paths)} file:")
    for p in paths:
        inspect(p)
    print("\n=> File nao co 'conv in-channels = 104' la model 2-RX dung. "
          "Dung cap pose+presence cung 104 cho realtime.")


if __name__ == "__main__":
    main()
