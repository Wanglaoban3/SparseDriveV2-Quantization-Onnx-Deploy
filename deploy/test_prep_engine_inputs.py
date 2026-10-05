# -*- coding: utf-8 -*-
"""prep_engine_inputs 单元测试（standalone assert 脚本）。

计划 Task 3 Step 1：先于实现编写运行，必须先失败。
关键契约（Review Focus 2/5）：
- manifest.tsv 四列 name\\tdtype\\tdims\\tfile，dtype 串必须与 run_engines.cpp
  dtype_str(kFLOAT) == "f32" 一致；
- dims 逗号分隔无空格；
- bin 字节数 = 元素数 * 4（fp32）；
- 输入 dtype/形状断言（喂错字节板端无报错，只能在这里拦）。
"""
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONTRACT = {
    "imgs": (1, 3, 3, 256, 512),
    "projection_mat": (1, 3, 4, 4),
    "image_wh": (1, 3, 2),
    "status_feature": (1, 8),
}

# 真实 engine_ref npz 的键名（val_00.npz 实测）：imgs/proj/iwh/status
NPZ_KEY_OF = {
    "imgs": "imgs",
    "projection_mat": "proj",
    "image_wh": "iwh",
    "status_feature": "status",
}


def make_fake_npz(path):
    rng = np.random.RandomState(0)
    data = {NPZ_KEY_OF[k]: rng.rand(*v).astype(np.float32) for k, v in CONTRACT.items()}
    np.savez(path, **data)


def test_manifest_and_bins(tmp_out):
    from deploy import prep_engine_inputs as prep

    npz = os.path.join(tmp_out, "val_99.npz")
    make_fake_npz(npz)
    prep.convert_one(npz, os.path.join(tmp_out, "val_99"))

    d = os.path.join(tmp_out, "val_99")
    for name, shape in CONTRACT.items():
        p = os.path.join(d, name + ".bin")
        n = 1
        for s in shape:
            n *= s
        assert os.path.getsize(p) == n * 4, (name, os.path.getsize(p), n * 4)

    with open(os.path.join(d, "manifest.tsv"), "rb") as f:
        raw = f.read()
    assert b"\r" not in raw, "manifest 必须是 LF"
    lines = raw.decode().strip().split("\n")
    assert len(lines) == 4, lines
    for ln in lines:
        name, dtype, dims, file = ln.split("\t")
        assert dtype == "f32", dtype  # run_engines.cpp dtype_str(kFLOAT)
        assert dims == ",".join(str(s) for s in CONTRACT[name]), dims
        assert file == name + ".bin", file
        assert name == prep.sanitize(name)


def test_dtype_and_shape_asserts(tmp_out):
    from deploy import prep_engine_inputs as prep

    rng = np.random.RandomState(1)
    bad = {NPZ_KEY_OF[k]: rng.rand(*v).astype(np.float32) for k, v in CONTRACT.items()}
    bad["imgs"] = bad["imgs"].astype(np.float16)  # 错 dtype
    npz = os.path.join(tmp_out, "val_98.npz")
    np.savez(npz, **bad)
    try:
        prep.convert_one(npz, os.path.join(tmp_out, "val_98"))
        raise SystemExit("FAIL: fp16 imgs 未被拦截")
    except AssertionError:
        pass

    bad2 = {NPZ_KEY_OF[k]: rng.rand(*v).astype(np.float32) for k, v in CONTRACT.items()}
    bad2["status"] = rng.rand(1, 7).astype(np.float32)  # 错形状
    npz2 = os.path.join(tmp_out, "val_97.npz")
    np.savez(npz2, **bad2)
    try:
        prep.convert_one(npz2, os.path.join(tmp_out, "val_97"))
        raise SystemExit("FAIL: 错误形状未被拦截")
    except AssertionError:
        pass


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_out:
        fns = [test_manifest_and_bins, test_dtype_and_shape_asserts]
        failed = 0
        for fn in fns:
            try:
                fn(tmp_out)
                print("PASS", fn.__name__)
            except SystemExit as e:
                failed += 1
                print("FAIL", fn.__name__, "->", e)
            except Exception as e:
                failed += 1
                print("ERROR", fn.__name__, "->", type(e).__name__, e)
    print(f"{len(fns) - failed}/{len(fns)} pass")
    sys.exit(1 if failed else 0)
