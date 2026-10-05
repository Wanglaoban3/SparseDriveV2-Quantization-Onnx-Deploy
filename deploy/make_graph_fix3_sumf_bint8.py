# -*- coding: utf-8 -*-
"""Backbone-only int8 实验：把 ModelOpt 校准的 QDQ（donor 图中 img_backbone
全部 conv 的激活侧 + 权重侧）嫁接进现役 fix3_sumf 图。neck/decoder 保持
fp16 不动（neck 输出紧贴 myelin 边界，是 E1 雷区）。

匹配锚点 = conv 的 weight initializer 名（两图同源、backbone 未动过）。
Q/DQ 的 scale/zp 用全新 initializer（无 Cast 链）——TRT 才能把权重 Q
折叠进 conv 构建期完成。weight roundtrip 校验同 E1。

usage: python make_graph_fix3_sumf_bint8.py --out <path.onnx>
"""
import argparse
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

import numpy as np
import onnx
from onnx import numpy_helper
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
FIX3S = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_fp16_graph_fix3_sumf.onnx")
DONOR = os.path.join(_ROOT, "deploy/artifacts/sparsedrive_int8_qdq.onnx")
BB = "/model/_backbone/img_backbone"

PASS = ("Cast", "Transpose", "Reshape", "Identity", "Squeeze", "Unsqueeze")

CAST_NP = {1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16,
           6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.float64}


def resolve_const(d_prod, d_init, t, depth=0):
    """donor 常量张量的 numpy 值: initializer / Constant / Cast 链。"""
    if depth > 6:
        raise RuntimeError(f"resolve depth exceeded at {t}")
    if t in d_init:
        return numpy_helper.to_array(d_init[t])
    p = d_prod.get(t)
    if p is not None and p.op_type == "Constant":
        for a in p.attribute:
            if a.name == "value":
                return numpy_helper.to_array(a.t)
    if p is not None and p.op_type == "Cast":
        base = resolve_const(d_prod, d_init, p.input[0], depth + 1)
        return base.astype(CAST_NP[p.attribute[0].i])
    raise RuntimeError(f"cannot resolve donor const {t}")


def walk_back(prod, t, want):
    """从 t 向上穿过 PASS，直到遇到 want 中的 op。返回 (node, 根张量)。"""
    cur = t
    for _ in range(12):
        p = prod.get(cur)
        if p is None:
            return None, cur
        if p.op_type in want:
            return p, cur
        if p.op_type not in PASS:
            return None, cur
        cur = p.input[0]
    return None, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fm = onnx.load(FIX3S)
    dm = onnx.load(DONOR)
    fg, dg = fm.graph, dm.graph

    f_init = {i.name: i for i in fg.initializer}
    d_init = {i.name: i for i in dg.initializer}
    f_prod, d_prod = {}, {}
    for g, m in ((fg, f_prod), (dg, d_prod)):
        for n in g.node:
            for o in n.output:
                m[o] = n

    new_nodes = []
    new_inits = {}
    used_names = {n.name for n in fg.node}
    produced = {i.name for i in fg.initializer} | {v.name for v in fg.input}
    act_dq = {}   # 激活根张量 -> DQ 输出（多个 conv 共享同一激活时复用）

    def fresh(t, sfx, dtype=np.float32):
        """donor 常量 → 全新 initializer（scale=fp32，zp=int8——dtype 必须与
        Q 输出类型一致，fp32 的 zp 是 spec 违规且会让 TRT 拒绝折叠）。"""
        arr = resolve_const(d_prod, d_init, t).astype(dtype)
        name = t.replace("/", "_") + sfx
        assert name not in used_names, name
        new_inits[name] = numpy_helper.from_array(arr, name)
        used_names.add(name)
        return name

    def make_qdq(data, sc_t, zp_t, qname, axis=None):
        """构建 Q→DQ 对（scale=fp32 initializer，zp=int8 initializer）。"""
        sname = fresh(sc_t, "_s")
        zname = (fresh(zp_t, "_z", np.int8) if zp_t else
                 data.replace("/", "_") + "_zp8")
        if zp_t is None or zp_t == "":
            new_inits[zname] = numpy_helper.from_array(np.int8(0), zname)
            used_names.add(zname)
        attrs = {} if axis is None else {"axis": axis}
        q_out = qname + "_q"
        dq_out = qname + "_d"
        new_nodes.append(onnx.helper.make_node(
            "QuantizeLinear", [data, sname, zname], [q_out],
            name=qname + "_Q", **attrs))
        new_nodes.append(onnx.helper.make_node(
            "DequantizeLinear", [q_out, sname, zname], [dq_out],
            name=qname + "_DQ", **attrs))
        return dq_out

    d_convs = [n for n in dg.node if n.op_type == "Conv" and BB in n.name]
    print(f"donor img_backbone convs: {len(d_convs)}")

    grafted, skipped = [], []
    for dc in d_convs:
        tag = dc.name.replace(BB + "/", "")
        dq_w, _ = walk_back(d_prod, dc.input[1], {"DequantizeLinear"})
        dq_a, _ = walk_back(d_prod, dc.input[0], {"DequantizeLinear"})
        if dq_w is None or dq_a is None:
            skipped.append((tag, "donor 无 QDQ(被保护?)"))
            continue
        qw = d_prod.get(dq_w.input[0])
        qa = d_prod.get(dq_a.input[0])
        assert qw is not None and qw.op_type == "QuantizeLinear", tag
        assert qa is not None and qa.op_type == "QuantizeLinear", tag

        wname = qw.input[0]                       # donor 权重 initializer 名
        assert wname in f_init, f"fix3 缺 weight init: {wname}"
        f_conv = next((n for n in fg.node
                       if n.op_type == "Conv" and wname in n.input[1:]), None)
        assert f_conv is not None, f"fix3 找不到 conv (weight={wname})"
        assert f_conv.input[0] == qa.input[0], \
            f"{tag}: 激活根不匹配 {f_conv.input[0][-50:]} vs {qa.input[0][-50:]}"

        # 权重侧（每 conv 独立）；权重转回 fp32 并**写回 fg.initializer**
        # （TRT 权重 Q 折叠要求 fp32 权重；只改本地字典不落盘是隐藏 bug）
        if f_init[wname].data_type != 1:
            arr32 = numpy_helper.to_array(f_init[wname]).astype(np.float32)
            new_w32 = numpy_helper.from_array(arr32, wname)
            for k, init in enumerate(fg.initializer):
                if init.name == wname:
                    fg.initializer[k].CopyFrom(new_w32)
                    break
            f_init[wname] = new_w32
        axw = next((a.i for a in qw.attribute if a.name == "axis"), None)
        axa = next((a.i for a in qa.attribute if a.name == "axis"), None)
        zpw = qw.input[2] if len(qw.input) > 2 else ""
        f_conv.input[1] = make_qdq(wname, qw.input[1], zpw,
                                   wname.replace("/", "_") + "_wtq8", axw)
        # 激活侧（同 root 复用；per-tensor 无 axis 属性）
        if root_a_dq := act_dq.get(qa.input[0]):
            f_conv.input[0] = root_a_dq
        else:
            zpa = qa.input[2] if len(qa.input) > 2 else ""
            new_a = make_qdq(qa.input[0], qa.input[1], zpa,
                             qa.input[0].replace("/", "_") + "_actq8", axa)
            f_conv.input[0] = new_a
            act_dq[qa.input[0]] = new_a

        # weight roundtrip 校验
        w = numpy_helper.to_array(f_init[wname]).astype(np.float64)
        s = resolve_const(d_prod, d_init, qw.input[1]).astype(np.float64)
        z = (resolve_const(d_prod, d_init, qw.input[2]).astype(np.float64)
             if len(qw.input) > 2 and qw.input[2] else np.zeros_like(s))
        if s.ndim == 0:
            s, z = s.reshape(1), z.reshape(1)
        ax = 0 if s.size == w.shape[0] else (1 if s.size == w.shape[-1] else None)
        assert ax is not None, f"{tag}: scale {s.size} vs weight {w.shape}"
        sh = [1] * w.ndim
        sh[ax] = s.size
        s, z = s.reshape(sh), z.reshape(sh)
        q8 = np.clip(np.round(w / s) + z, -127, 127)
        err = np.abs((q8 - z) * s - w).max()
        assert err <= s.max(), f"{tag}: roundtrip err {err} > scale"
        grafted.append(tag)
        print(f"  [ok] {tag:52s} w={w.shape} roundtrip={err:.2e}")

    print(f"\ngrafted {len(grafted)} convs, skipped {len(skipped)}")
    for tag, why in skipped:
        print(f"  [skip] {tag}: {why}")
    assert grafted, "nothing grafted"

    # 拼接：新节点按"输入就绪即插"保持拓扑序
    produced |= set(new_inits.keys())          # fresh scale/zp 初始就绪
    remaining = list(new_nodes)
    out_final = []
    for n in fg.node:
        out_final.append(n)
        for o in n.output:
            produced.add(o)
        prog = True
        while prog:
            prog = False
            for pn in list(remaining):
                if all((not i) or (i in produced) for i in pn.input):
                    out_final.append(pn)
                    remaining.remove(pn)
                    for o in pn.output:
                        produced.add(o)
                    prog = True
    assert not remaining, f"{len(remaining)} 个新节点无法就位"
    del fg.node[:]
    fg.node.extend(out_final)
    existing = {i.name for i in fg.initializer}
    for name, init in new_inits.items():
        if name not in existing:
            fg.initializer.append(init)

    onnx.save(fm, args.out)
    print(f"saved -> {args.out}")

    onnx.checker.check_model(fm)
    n_q = sum(1 for n in fg.node if n.op_type == "QuantizeLinear")
    print(f"nodes={len(fg.node)} Q={n_q} SURGERY_OK")


if __name__ == "__main__":
    main()
