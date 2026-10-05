"""GPU-free validation of the deploy ONNX against the compiled DFA plugin.

Proves, before going to the target device:
  1. the plugin .so registers its creator (name / namespace / version)
  2. the ONNX parser resolves ALL THREE sparsedrivev2::DeformableAggregation
     custom ops to the plugin (a missing/mismatched plugin makes parse fail)
  3. network I/O matches the deployment contract
     (4 inputs, 8 outputs: trajectory + traj_scores + 6 metric logits)

Engine BUILD additionally requires a CUDA device; run build_engine.py on the
target machine (or any Linux box with a CUDA GPU) for the engine itself.

Usage:
  python3 trt_parse_check.py libdfa_plugin.so sparsedrive_int8_qdq_feat8_folded.onnx
"""
import ctypes
import os
import sys

import tensorrt as trt


def main():
    so_path, onnx_path = sys.argv[1], sys.argv[2]
    ctypes.CDLL(os.path.abspath(so_path))
    print("[1/3] plugin .so loaded")

    registry = trt.get_plugin_registry()
    creators = [c for c in registry.plugin_creator_list
                if "DeformableAggregation" in c.name]
    if not creators:
        sys.exit("FAIL: plugin creator not found in registry")
    for c in creators:
        print("      creator: name=%s namespace=%r version=%s"
              % (c.name, c.plugin_namespace, c.plugin_version))
    # TRT 8.6 board-proven: registerCreator dedups by (name, version) keeping
    # the FIRST, and the parser queries the EMPTY namespace — the surviving
    # creator must therefore sit in "".
    assert any(c.plugin_namespace == "" for c in creators), \
        "creator must live in the EMPTY namespace (parser query ns); got %r" \
        % [c.plugin_namespace for c in creators]
    print("[2/3] creator registered under namespace '' (parser query ns)")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(open(onnx_path, "rb").read()):
        for i in range(parser.num_errors):
            print("parse error:", parser.get_error(i))
        sys.exit("FAIL: ONNX parse failed")
    print("[3/3] ONNX parsed")

    # TRT 8.6: IPluginV2DynamicExt layers show up as PLUGIN_V2 (enum 34),
    # not the legacy PLUGIN (enum 9) — accept either.
    _plugin_types = {trt.LayerType.PLUGIN, trt.LayerType.PLUGIN_V2}
    plugin_layers = [l for l in network if l.type in _plugin_types]
    print("      layers total=%d  plugin(DFA)=%d  inputs=%d  outputs=%d"
          % (network.num_layers, len(plugin_layers),
             network.num_inputs, network.num_outputs))
    for l in plugin_layers:
        p = getattr(l, "plugin", None)
        if p is not None:
            print("      DFA site: %s (plugin=%s ns=%s ver=%s)"
                  % (l.name, p.plugin_type, p.plugin_namespace, p.plugin_version))
    assert len(plugin_layers) == 3, "expected 3 DFA plugin layers"
    assert network.num_inputs == 4 and network.num_outputs == 8, \
        "network I/O does not match the deployment contract"
    print("PASS: graph, plugin resolution and I/O contract all verified")


if __name__ == "__main__":
    main()
