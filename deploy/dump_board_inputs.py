# -*- coding: utf-8 -*-
"""Dump the 138 mini-split scene inputs as run_engines bins (Task 6 Step 1).

Token口径与 pdms_eval_quant.py main() 一致：ds.tokens ∩ metric_cache.tokens。
每 token 落 4 bin + manifest.tsv（契约同 prep_engine_inputs.CONTRACT / run_engines.cpp
dtype_str：name\\tf32\\tdims\\tfile，输入名无前导 '/'）。

Usage (navsim env):
  python deploy/dump_board_inputs.py --out-dir deploy/artifacts/engine_inputs/mini138
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SD_ROOT", os.path.dirname(DEPLOY_DIR))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import ptq_pipeline as pp  # noqa: E402
from navsim.navsim_v1.common.dataloader import MetricCacheLoader  # noqa: E402

METRIC_CACHE = os.environ.get("SD_METRIC_CACHE", os.path.join(ROOT, "exp", "metric_cache_mini_v1"))

# 与 prep_engine_inputs.CONTRACT 逐字一致
CONTRACT = {
    "imgs": (1, 3, 3, 256, 512),
    "projection_mat": (1, 3, 4, 4),
    "image_wh": (1, 3, 2),
    "status_feature": (1, 8),
}


def write_token_dir(arrs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, arr in arrs.items():
        shape = CONTRACT[name]
        assert arr.dtype == np.float32, f"{name} dtype={arr.dtype} 须 fp32"
        assert tuple(arr.shape) == shape, f"{name} shape={arr.shape} != {shape}"
        with open(os.path.join(out_dir, name + ".bin"), "wb") as f:
            f.write(np.ascontiguousarray(arr).tobytes())
        rows.append(f"{name}\tf32\t{','.join(map(str, shape))}\t{name}.bin")
    with open(os.path.join(out_dir, "manifest.tsv"), "w", newline="\n") as f:
        f.write("\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join("deploy", "artifacts", "engine_inputs", "mini138"))
    ap.add_argument("--tokens-only", action="store_true",
                    help="只落 tokens.txt（ds 迭代序，供 eval 与 dev csv 行序对齐），不重写 bin")
    args = ap.parse_args()

    cfg, wrapper, ds = pp.build()
    mcache = MetricCacheLoader(Path(METRIC_CACHE))
    tokens = [t for t in ds.tokens if t in mcache.tokens]
    print(f"scenes with feature cache + metric cache: {len(tokens)}")
    with open(os.path.join(args.out_dir, "tokens.txt"), "w", newline="\n") as f:
        f.write("\n".join(tokens) + "\n")
    print(f"tokens.txt written ({len(tokens)} rows, ds iteration order)")
    if args.tokens_only:
        print("DUMP138_DONE (tokens only)")
        return

    total = 0
    for n, tok in enumerate(tokens):
        inp, gt, _ = pp.to_inputs(ds, ds.tokens.index(tok))
        arrs = {k: t.float().cpu().numpy() for k, t in zip(CONTRACT, inp)}
        write_token_dir(arrs, os.path.join(args.out_dir, tok))
        total += sum(os.path.getsize(os.path.join(args.out_dir, tok, f))
                     for f in os.listdir(os.path.join(args.out_dir, tok)))
        if (n + 1) % 20 == 0 or n + 1 == len(tokens):
            print(f"  [{n + 1}/{len(tokens)}] {tok}  total={total >> 20}MB", flush=True)
    print(f"first={tokens[0]} last={tokens[-1]} dirs={len(tokens)} total={total >> 20}MB")
    print(f"DUMP138_DONE -> {args.out_dir}")


if __name__ == "__main__":
    main()
