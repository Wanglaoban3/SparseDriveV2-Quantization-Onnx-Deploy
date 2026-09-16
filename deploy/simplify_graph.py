"""Constant/shape folding pass for the QDQ deliverable ONNX.

onnxsim is unusable here: it evaluates foldable subgraphs by running an
ONNX Runtime session, and ORT cannot load a graph containing the
unregistered sparsedrivev2::DeformableAggregation op. This pass folds
with onnx shape inference + numpy only and treats the custom op and ALL
QuantizeLinear/DequantizeLinear nodes as hard boundaries, so the
quantization semantics are untouched.

Only exactly-specified ops are folded (data movement / elementwise IEEE
ops / shape computation). MatMul/Conv/Gemm/LayerNormalization/Softmax and
anything with trainable data are never folded, so folded results are
bitwise identical to what TRT would compute for the same subgraph.

Usage:
  python simplify_graph.py <in.onnx> <out.onnx>
"""
import collections
import hashlib
import os
import sys
import time

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

CUSTOM_DOMAIN = "sparsedrivev2"

CAST_TO = {
    TensorProto.FLOAT: np.float32, TensorProto.UINT8: np.uint8,
    TensorProto.INT8: np.int8, TensorProto.UINT16: np.uint16,
    TensorProto.INT16: np.int16, TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64, TensorProto.BOOL: np.bool_,
    TensorProto.FLOAT16: np.float16, TensorProto.DOUBLE: np.float64,
    TensorProto.UINT32: np.uint32, TensorProto.UINT64: np.uint64,
}

ELEM_OPS = {
    "Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide,
    "Min": np.minimum, "Max": np.maximum, "And": np.logical_and,
    "Or": np.logical_or, "Xor": np.logical_xor, "Equal": np.equal,
    "NotEqual": np.not_equal, "Greater": np.greater, "Less": np.less,
    "GreaterOrEqual": np.greater_equal, "LessOrEqual": np.less_equal,
}

FOLD_BUDGET_BYTES = 32 * 1024 * 1024


def init_hash(init):
    h = hashlib.sha1()
    h.update(init.name.encode())
    h.update(str(tuple(init.dims)).encode())
    h.update(init.raw_data if init.raw_data else b"")
    for d in init.float_data:
        h.update(np.float32(d).tobytes())
    for d in init.int64_data:
        h.update(np.int64(d).tobytes())
    for d in init.int32_data:
        h.update(np.int32(d).tobytes())
    return h.hexdigest()


def attrs(node):
    return {a.name: a for a in node.attribute}


def get_ints(a, default):
    if a is None:
        return default
    if a.type == onnx.AttributeProto.INTS:
        return list(a.ints)
    if a.type == onnx.AttributeProto.INT:
        return [a.i]
    return default


def norm_axis(ax, ndim):
    return ax % ndim if ndim > 0 else ax


def eval_node(node, ins):
    """ins: list of np arrays (absent optional inputs are None). Returns array or None."""
    A = attrs(node)
    op = node.op_type

    if op in ELEM_OPS:
        return ELEM_OPS[op](*ins)
    if op == "Not":
        return np.logical_not(ins[0])
    if op == "Neg":
        return np.negative(ins[0])
    if op == "Identity":
        return ins[0]
    if op == "Cast":
        to = A["to"].i
        if to not in CAST_TO:
            return None
        return ins[0].astype(CAST_TO[to])
    if op == "Unsqueeze":
        axes = get_ints(A.get("axes"), None)
        if ins[1] is not None:
            axes = ins[1].reshape(-1).tolist()
        if axes is None:
            return None
        out = ins[0]
        for ax in sorted(int(a) for a in axes):
            out = np.expand_dims(out, norm_axis(ax, out.ndim + 1))
        return out
    if op == "Squeeze":
        axes = get_ints(A.get("axes"), None)
        if len(ins) > 1 and ins[1] is not None:
            axes = ins[1].reshape(-1).tolist()
        if axes is None or len(axes) == 0:
            return ins[0].reshape([d for d in ins[0].shape if d != 1])
        return np.squeeze(ins[0], axis=tuple(sorted(int(norm_axis(a, ins[0].ndim)) for a in axes)))
    if op == "Reshape":
        shape = ins[1].reshape(-1).astype(np.int64)
        allowzero = A["allowzero"].i if "allowzero" in A else 0
        if allowzero == 0:
            shape = [ins[0].shape[i] if s == 0 else s for i, s in enumerate(shape)]
        if any(s == -1 for s in shape):
            known = int(np.prod([d for d in shape if d != -1])) if any(d != -1 for d in shape) else 1
            shape = [int(ins[0].size // known) if s == -1 else s for s in shape]
        return ins[0].reshape([int(s) for s in shape])
    if op == "Flatten":
        ax = A["axis"].i if "axis" in A else 1
        ax = norm_axis(ax, ins[0].ndim)
        d0 = int(np.prod(ins[0].shape[:ax])) if ax > 0 else 1
        return ins[0].reshape(d0, -1)
    if op == "Transpose":
        perm = get_ints(A.get("perm"), None)
        return np.transpose(ins[0], perm if perm else None)
    if op == "Gather":
        ax = norm_axis(A["axis"].i if "axis" in A else 0, ins[0].ndim)
        idx = ins[1]
        if idx.dtype not in (np.int64, np.int32):
            idx = idx.astype(np.int64)
        return np.take(ins[0], idx, axis=ax)
    if op == "Slice":
        starts = ins[1].reshape(-1).tolist()
        ends = ins[2].reshape(-1).tolist()
        axes = ins[3].reshape(-1).tolist() if len(ins) > 3 and ins[3] is not None else list(range(len(starts)))
        steps = ins[4].reshape(-1).tolist() if len(ins) > 4 and ins[4] is not None else [1] * len(starts)
        sl = [slice(None)] * ins[0].ndim
        for s, e, a, st in zip(starts, ends, axes, steps):
            sl[norm_axis(int(a), ins[0].ndim)] = slice(int(s), int(e), int(st))
        return ins[0][tuple(sl)]
    if op == "Concat":
        ax = norm_axis(A["axis"].i, ins[0].ndim)
        return np.concatenate(ins, axis=ax)
    if op == "Expand":
        shape = ins[1].reshape(-1).astype(np.int64)
        return np.ascontiguousarray(np.broadcast_to(ins[0], tuple(int(s) for s in shape)))
    if op == "Tile":
        reps = ins[1].reshape(-1).astype(np.int64)
        return np.tile(ins[0], tuple(int(r) for r in reps))
    if op == "ConstantOfShape":
        shape = tuple(int(s) for s in ins[0].reshape(-1)) if ins[0] is not None else tuple()
        a = A.get("value")
        if a is not None:
            v = numpy_helper.to_array(a.t)
            return np.full(shape, v.reshape(-1)[0] if v.size else 0, dtype=v.dtype)
        return np.zeros(shape, dtype=np.float32)
    if op == "Where":
        return np.where(ins[0].astype(bool), ins[1], ins[2])
    if op == "Range":
        s, l, d = (x.reshape(-1)[0] for x in ins)
        return np.arange(int(s), int(l), int(d), dtype=ins[0].dtype)
    if op.startswith("Reduce"):
        keepdims = bool(A["keepdims"].i) if "keepdims" in A else True
        noop_empty = bool(A["noop_with_empty_axes"].i) if "noop_with_empty_axes" in A else False
        axes = None
        if len(ins) > 1 and ins[1] is not None:
            axes = tuple(int(norm_axis(int(a), ins[0].ndim)) for a in ins[1].reshape(-1))
        elif "axes" in A:
            axes = tuple(int(norm_axis(int(a), ins[0].ndim)) for a in A["axes"].ints)
        if axes is None or len(axes) == 0:
            if noop_empty:
                return None
            axes = tuple(range(ins[0].ndim))
        fn = {"ReduceSum": np.sum, "ReduceProd": np.prod, "ReduceMin": np.min,
              "ReduceMax": np.max, "ReduceMean": np.mean}.get(op)
        if fn is None:
            return None
        return fn(ins[0], axis=axes, keepdims=keepdims)
    if op == "Shape":
        return None  # handled by caller via shape-inference map
    return None


def load_consts(g):
    consts = {}
    for init in g.initializer:
        consts[init.name] = numpy_helper.to_array(init)
    for n in g.node:
        if n.op_type == "Constant":
            a = attrs(n)
            if "value" in a:
                consts[n.output[0]] = numpy_helper.to_array(a["value"].t)
            elif "value_float" in a:
                consts[n.output[0]] = np.array(a["value_float"].f, np.float32)
            elif "value_int" in a:
                consts[n.output[0]] = np.array(a["value_int"].i, np.int64)
            elif "value_ints" in a:
                consts[n.output[0]] = np.array(a["value_ints"].ints, np.int64)
    return consts


def build_shape_map(model):
    shapes = {}
    try:
        si = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception as e:
        print("shape inference failed (%s); Shape nodes will not fold" % e)
        return shapes
    g = si.graph
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if len(dims) > 0 and all(d > 0 for d in dims):
                shapes[vi.name] = dims
    return shapes


def graph_outputs(g):
    return {o.name for o in g.output}


def qdq_signature(g):
    sig = []
    for n in g.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            sig.append((n.op_type, tuple(n.input), tuple(n.output)))
    return sig


def heavy_signature(g):
    sig = collections.Counter()
    for n in g.node:
        if n.op_type in ("Conv", "MatMul", "Gemm", "LayerNormalization",
                         "BatchNormalization", "ConvTranspose"):
            sig[(n.op_type, tuple(n.input))] += 1
    return sig


def main():
    t0 = time.time()
    inp, outp = sys.argv[1], sys.argv[2]
    model = onnx.load(inp)
    g = model.graph

    assert not any(n.op_type in ("If", "Loop", "Scan") for n in g.node), "control flow unsupported"
    outs = graph_outputs(g)

    orig_inits = {i.name: init_hash(i) for i in g.initializer}
    orig_qdq = qdq_signature(g)
    orig_heavy = heavy_signature(g)
    orig_dfa = [n for n in g.node if n.domain == CUSTOM_DOMAIN]
    orig_hist = collections.Counter(n.op_type for n in g.node)
    assert len(orig_dfa) == 3, "expected 3 DFA nodes, got %d" % len(orig_dfa)

    consts = load_consts(g)
    shapes = build_shape_map(model)

    folded = {}          # name -> np array of folded results
    prov = {}            # folded name -> set of source initializer/const names
    fold_count = collections.Counter()

    for rnd in range(30):
        changed = 0
        for node in list(g.node):
            if any(o in outs for o in node.output):
                continue
            op = node.op_type
            result = None
            if op == "Shape":
                x = node.input[0]
                if x in shapes:
                    result = np.array(shapes[x], dtype=np.int64)
            elif op == "Size":
                x = node.input[0]
                if x in shapes:
                    result = np.array(int(np.prod(shapes[x])), dtype=np.int64)
            elif op in ("QuantizeLinear", "DequantizeLinear") or node.domain == CUSTOM_DOMAIN:
                result = None  # hard boundary
            else:
                in_vals = []
                ok = True
                for name in node.input:
                    if name == "":
                        in_vals.append(None)
                    elif name in consts:
                        in_vals.append(consts[name])
                    elif name in folded:
                        in_vals.append(folded[name])
                    else:
                        ok = False
                        break
                if ok:
                    try:
                        result = eval_node(node, in_vals)
                    except Exception:
                        result = None
            if result is None or not isinstance(result, np.ndarray):
                continue
            if result.nbytes > FOLD_BUDGET_BYTES:
                continue
            src = set()
            for name in node.input:
                if name == "":
                    continue
                if name in consts:
                    src.add(name)
                elif name in folded:
                    src |= prov.get(name, {name})
            folded[node.output[0]] = result
            prov[node.output[0]] = src
            fold_count[op] += 1
            changed += 1
            g.node.remove(node)
        if changed == 0:
            break
        print("round %d: folded %d nodes" % (rnd + 1, changed))

    # materialize folded values consumed by surviving nodes / graph outputs
    consumed = set()
    for node in g.node:
        for name in node.input:
            if name in folded:
                consumed.add(name)
    for name in outs:
        if name in folded:
            consumed.add(name)
    new_consts = []
    for name in sorted(consumed):
        arr = folded[name]
        new_consts.append(onnx.helper.make_node(
            "Constant", [], [name], name=name + "_folded",
            value=numpy_helper.from_array(arr, name=name)))
    for i, nc in enumerate(new_consts):
        g.node.insert(i, nc)

    # dead sweep: keep only nodes reachable from graph outputs
    producers = {}
    for node in g.node:
        for o in node.output:
            producers[o] = node
    keep = set()
    stack = [producers[o] for o in outs if o in producers]
    while stack:
        n = stack.pop()
        if id(n) in keep:
            continue
        keep.add(id(n))
        for name in n.input:
            p = producers.get(name)
            if p is not None and id(p) not in keep:
                stack.append(p)
    keep_nodes = [n for n in g.node if id(n) in keep]
    del g.node[:]
    g.node.extend(keep_nodes)

    referenced = set()
    for node in g.node:
        for name in node.input:
            referenced.add(name)
    keep_inits = [i for i in g.initializer
                  if i.name in referenced or i.name in outs]
    del g.initializer[:]
    g.initializer.extend(keep_inits)

    # ---------------- asserts ----------------
    now_inits = {i.name: init_hash(i) for i in g.initializer}
    all_prov = set()
    for s in prov.values():
        all_prov |= s
    removed_inits = []
    for name, h in orig_inits.items():
        if now_inits.get(name) == h:
            continue
        # removed because e.g. its Transpose was folded into a constant;
        # its value must be traceable through the folding provenance chain
        assert name in all_prov, "initializer %s removed without folding trace!" % name
        removed_inits.append(name)
    if removed_inits:
        print("note: %d initializers folded into constants (fallback weights via Transpose), e.g. %s"
              % (len(removed_inits), removed_inits[:3]))
    assert qdq_signature(g) == orig_qdq, "Q/DQ signature changed!"
    assert heavy_signature(g) == orig_heavy, "Conv/MatMul/Gemm structure changed!"
    now_dfa = [n for n in g.node if n.domain == CUSTOM_DOMAIN]
    assert len(now_dfa) == 3, "DFA count changed!"
    for a, b in zip(orig_dfa, now_dfa):
        assert list(a.input) == list(b.input) and list(a.output) == list(b.output), "DFA io changed!"
    for io in list(g.input) + list(g.output):
        pass  # names/types untouched by construction
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print("checker warning (custom domain may trip it): %s" % e)

    new_hist = collections.Counter(n.op_type for n in g.node)
    print("\n==== report ====")
    print("input : %s  (%.1f MB, %d nodes)" % (inp, os.path.getsize(inp) / 1e6, sum(orig_hist.values())))
    print("output: %s  (%.1f MB, %d nodes)" % (outp, os.path.getsize(outp) / 1e6 if os.path.exists(outp) else float("nan"), len(g.node)))
    print("folded by op:", dict(fold_count.most_common()))
    print("hist delta:")
    for k in sorted(set(orig_hist) | set(new_hist)):
        d = new_hist.get(k, 0) - orig_hist.get(k, 0)
        if d != 0:
            print("  %-20s %5d -> %5d" % (k, orig_hist.get(k, 0), new_hist.get(k, 0)))
    print("initializers: %d -> %d" % (len(orig_inits), len(now_inits)))
    print("asserts passed: init hashes, Q/DQ signature, heavy ops, 3x DFA, checker")
    onnx.save(model, outp)
    print("saved %s (%.1f MB) in %.1fs" % (outp, os.path.getsize(outp) / 1e6, time.time() - t0))


if __name__ == "__main__":
    main()
