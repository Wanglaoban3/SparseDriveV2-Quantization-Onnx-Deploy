# -*- coding: utf-8 -*-
"""把 engine_ref/val_XX.npz 转成 run_engines 可吃的 bins + manifest.tsv。

manifest 契约（v1 run_engines.cpp load_manifest）：四列 name\\tdtype\\tdims\\tfile，
dtype 串必须与 dtype_str() 一致（kFLOAT -> "f32"），dims 逗号分隔无空格。
输入名无前导 '/'（防御性过 sanitize）。
"""
import argparse
import glob
import os
import sys

import numpy as np

# run_engines.cpp dtype_str(kFLOAT) == "f32"（逐字对齐，勿改）
DT_F32 = "f32"

CONTRACT = {
    "imgs": (1, 3, 3, 256, 512),
    "projection_mat": (1, 3, 4, 4),
    "image_wh": (1, 3, 2),
    "status_feature": (1, 8),
}

# npz 键名 -> 引擎输入名（make_engine_reference.py 的落盘键）
NPZ_KEYS = {
    "imgs": "imgs",
    "proj": "projection_mat",
    "iwh": "image_wh",
    "status": "status_feature",
}


def sanitize(name: str) -> str:
    return name.lstrip("/")


def convert_one(npz_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(npz_path)
    rows = []
    for npz_key, eng_name in NPZ_KEYS.items():
        arr = data[npz_key]
        shape = CONTRACT[eng_name]
        assert arr.dtype == np.float32, f"{npz_path}:{npz_key} dtype={arr.dtype} 须 fp32"
        assert tuple(arr.shape) == shape, f"{npz_path}:{npz_key} shape={arr.shape} != {shape}"
        stem = sanitize(eng_name)
        with open(os.path.join(out_dir, stem + ".bin"), "wb") as f:
            f.write(np.ascontiguousarray(arr).tobytes())
        rows.append(f"{stem}\t{DT_F32}\t{','.join(map(str, shape))}\t{stem}.bin")
    with open(os.path.join(out_dir, "manifest.tsv"), "w", newline="\n") as f:
        f.write("\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default=os.path.join("deploy", "artifacts", "engine_ref"))
    ap.add_argument("--out-dir", default=os.path.join("deploy", "artifacts", "engine_inputs", "ref_val"))
    args = ap.parse_args()

    npzs = sorted(glob.glob(os.path.join(args.ref_dir, "val_*.npz")))
    assert npzs, f"no val_*.npz under {args.ref_dir}"
    for p in npzs:
        stem = os.path.splitext(os.path.basename(p))[0]
        convert_one(p, os.path.join(args.out_dir, stem))
    print(f"{len(npzs)} dirs written under {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
