"""Board-side: run a built TensorRT engine over the FP32 reference samples and
report accuracy drift + latency. Dependencies: tensorrt, pycuda, numpy
(no torch / no dataset needed).

Usage (on the target device, after build_engine.py):
  python3 engine_infer_check.py --engine engine_int8.plan \
      --plugin libdfa_plugin.so --ref-dir engine_ref
"""
import argparse
import ctypes
import glob
import os
import time

import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit  # noqa


def compare(ref, q):
    """ref/q: [traj, scores, 6x metric] numpy arrays."""
    return dict(
        argmax_match=int(np.argmax(ref[0][1]) == np.argmax(q[1])),
        traj_l1=float(np.abs(ref[0][0] - q[0]).mean()),
        score_mae=float(np.abs(ref[0][1] - q[1]).mean()),
        metric_mae=float(np.mean([np.abs(a - b).mean()
                                  for a, b in zip(ref[0][2:], q[2:])])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--plugin", default="libdfa_plugin.so")
    ap.add_argument("--ref-dir", default="engine_ref")
    args = ap.parse_args()

    ctypes.CDLL(os.path.abspath(args.plugin))
    log = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(log).deserialize_cuda_engine(open(args.engine, "rb").read())
    ctx = engine.create_execution_context()

    n_bind = engine.num_bindings
    names = [engine.get_binding_name(i) for i in range(n_bind)]
    host_buf, dev_buf = {}, {}
    for i, n in enumerate(names):
        shape = tuple(engine.get_binding_shape(i))
        host_buf[n] = np.empty(shape, dtype=trt.nptype(engine.get_binding_dtype(i)))
        dev_buf[n] = cuda.mem_alloc(host_buf[n].nbytes)
    print("bindings:", [(n, tuple(engine.get_binding_shape(i))) for i, n in enumerate(names)])

    files = sorted(glob.glob(os.path.join(args.ref_dir, "val_*.npz")))
    if not files:
        raise SystemExit("no val_*.npz under %s" % args.ref_dir)

    rows, lat = [], []
    in_names = ["imgs", "proj", "iwh", "status"]
    for f in files:
        d = np.load(f)
        for n in in_names:
            np.copyto(host_buf[n], d[n])
        for n in in_names:
            cuda.memcpy_htod(dev_buf[n], host_buf[n])
        ctx.execute_v2([int(dev_buf[names[i]]) for i in range(n_bind)])
        outs = []
        for n in names:
            if n not in in_names:
                cuda.memcpy_dtoh(host_buf[n], dev_buf[n])
                outs.append(host_buf[n].astype(np.float32))

        t0 = time.time()
        for _ in range(5):
            ctx.execute_v2([int(dev_buf[names[i]]) for i in range(n_bind)])
        lat.append((time.time() - t0) / 5 * 1000)

        ref = [d["ref_traj"], d["ref_scores"]] + [d["ref_metric_%d" % j] for j in range(6)]
        m = compare(ref, outs)
        gt_l1 = float(np.abs(outs[0] - d["gt_traj"]).mean())
        rows.append((m, gt_l1))
        print("%s  traj_l1=%.4f  score_mae=%.4f  metric_mae=%.4f  argmax=%d  GT_l1=%.4f"
              % (os.path.basename(f), m["traj_l1"], m["score_mae"],
                 m["metric_mae"], m["argmax_match"], gt_l1))

    agg = {k: float(np.mean([r[0][k] for r in rows])) for k in rows[0][0]}
    print("\n== summary over %d samples ==" % len(rows))
    print("argmax match: %.3f" % (agg["argmax_match"] / 1.0 if agg["argmax_match"] <= 1 else agg["argmax_match"]))
    print("traj L1   : %.4f m" % agg["traj_l1"])
    print("score MAE : %.4f" % agg["score_mae"])
    print("metric MAE: %.4f" % agg["metric_mae"])
    print("latency   : %.1f ms/frame (avg)" % (sum(lat) / len(lat)))


if __name__ == "__main__":
    main()
