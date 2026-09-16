"""Board-side: build TRT engines from the QDQ / fp32 ONNX models.

Loads dfa_plugin (sparsedrivev2::DeformableAggregation) then parses ONNX.
FP16: plain flag. INT8: quantized ONNX already carries QDQ (explicit quantization);
also provides an implicit-quantization alternative with an entropy calibrator
fed by the calib/*.npz dumped on the dev machine.

Usage:
  python build_engine.py --onnx sparsedrive_int8_qdq.onnx --out engine_int8.plan --fp16
  python build_engine.py --onnx sparsedrive_fp32.onnx --out engine_fp16.plan --fp16
  python build_engine.py --onnx sparsedrive_fp32.onnx --out engine_int8_implicit.plan --int8
"""
import argparse
import ctypes
import glob
import os

import numpy as np

SP = os.environ.get("TRT_PYTHON_SITE", "")  # set if tensorrt pip layout needs path help
if SP:
    import sys
    sys.path.insert(0, SP)

import tensorrt as trt


def load_plugin(path_glob):
    for p in glob.glob(path_glob):
        if p.endswith((".so", ".dll", ".dylib")):
            ctypes.CDLL(p)
            print("plugin loaded:", p)
            return True
    raise FileNotFoundError(f"dfa_plugin not found: {path_glob}")


class CalibDataset(trt.IInt8EntropyCalibrator2):
    """Feeds calib_XX.npz batches (imgs/proj/iwh/status) to the builder."""

    def __init__(self, calib_dir, cache_file="calib.cache"):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(calib_dir, "calib_*.npz")))
        self.idx = 0
        self.cache = cache_file
        self.device_buffers = {}

    def get_batch_size(self):
        return 1

    def _to_dev(self, name, arr):
        import pycuda.driver  # noqa  # TRT calibrator assumes pycuda-managed pointers
        import pycuda.gpuarray as ga
        self.device_buffers[name] = ga.to_gpu(arr.astype(np.float32).ravel())
        return int(self.device_buffers[name].ptr)

    def get_batch(self, names):
        if self.idx >= len(self.files):
            return None
        d = np.load(self.files[self.idx])
        self.idx += 1
        blobs = [d["imgs"], d["proj"], d["iwh"], d["status"]]
        return [self._to_dev(n, b) for n, b in zip(names, blobs)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache):
            return open(self.cache, "rb").read()
        return None

    def write_calibration_cache(self, cache):
        open(self.cache, "wb").write(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plugin", default="build/libdfa_plugin.so")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true", help="implicit int8 (entropy calibrator); "
                  "use --onnx *_int8_qdq.onnx instead for explicit quantization")
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--workspace", type=int, default=4096, help="MB")
    args = ap.parse_args()

    load_plugin(args.plugin)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if args.int8:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("parse error:", parser.get_error(i))
            raise SystemExit(1)
    print("ONNX parsed OK; layers:", network.num_layers)

    config = builder.create_builder_config()
    config.max_workspace_size = args.workspace << 20
    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if args.int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = CalibDataset(args.calib_dir)
    # sparse weights are Hopper+; harmless to leave off.

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise SystemExit("engine build FAILED")
    with open(args.out, "wb") as f:
        f.write(plan)
    print("engine written:", args.out, len(plan), "bytes")


if __name__ == "__main__":
    main()
