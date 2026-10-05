# -*- coding: utf-8 -*-
"""SparseDriveV2 板端任务驱动 CLI。

用法：python deploy/board/board_v2.py <stage>
stage 随计划任务逐步扩充（check / plugin / push_onnx / parse_check / build_engine /
baseline / profile / dump24 / push138 / run138 / fetch138 / opt_ab / rawlogits ...）。
凭据从环境变量 BOARD_HOST / BOARD_PASS 读取（见 board_common.py 纪律）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.board import board_common as bc  # noqa: E402

BOARD = bc.BOARD_WORK


def stage_check(cli):
    """Task 0 Step 4/5：连通性 + 板端环境清点 + 后台/文本规范实测。"""
    stages = [
        ("uname", "uname -a"),
        ("trtexec", "ls /usr/src/tensorrt/bin/trtexec"),
        ("trt_version", "/usr/src/tensorrt/bin/trtexec --version 2>&1 | head -n 3"),
        ("run_engines", "ls -s /usr/local/bin/run_engines"),
        ("v1_plugin", "md5sum /usr/local/lib/libdfaplug_v8.so 2>/dev/null || echo none"),
        ("py38", "which python3.8 && python3.8 -c 'import numpy; print(\"numpy ok\")' 2>&1"),
        ("py38_trt", "python3.8 -c 'import tensorrt; print(\"trt-bindings\", tensorrt.__version__)' 2>&1"),
        ("nvcc", "which nvcc && nvcc --version | tail -n 1"),
    ]
    for name, cmd in stages:
        rc, out = bc.run(cli, cmd, timeout_s=60)
        print(f"[{name}] rc={rc}\n{out.strip()}")

    rc, out = bc.run(cli, f"mkdir -p {BOARD}/logs {BOARD}/onnx {BOARD}/plugin "
                          f"{BOARD}/engine {BOARD}/inputs {BOARD}/outs {BOARD}/prof", timeout_s=30)
    assert rc == 0, f"mkdir failed: {out}"

    # 后台/文本规范实测（Step 5）
    hello = "#!/bin/bash\necho hello\ntouch {}/logs/HW_DONE\n".format(BOARD)
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_hello.sh")
    with open(tmp, "w", newline="\n") as f:
        f.write(hello)
    bc.run(cli, f"rm -f {BOARD}/logs/HW_DONE", timeout_s=15)
    bc.push(cli, tmp, f"{BOARD}/hello.sh")
    bc.launch(cli, f"bash {BOARD}/hello.sh", f"{BOARD}/logs/hello.log")
    ok = bc.poll_marker(cli, f"{BOARD}/logs/HW_DONE", deadline_s=60, poll_s=3)
    rc, out = bc.run(cli, f"grep -c $'\\r' {BOARD}/hello.sh || true", timeout_s=15)
    print(f"[bg_marker] {'PASS' if ok else 'FAIL'}; [crlf_count]={out.strip()} (expect 0)")
    os.remove(tmp)
    assert ok and out.strip() == "0", "后台 marker 或 CRLF 规范实测未过"
    print("CHECK PASS")


def stage_plugin(cli):
    """Task 1：push 插件源码 → 板上编 libdfa_sd.so + dfa_i8_test → 6 用例单测。
    复合编排全部落 bash 脚本（v1 坑：nohup 无法执行 cd 内建，重定向绑错命令列表）。"""
    src = os.path.join(ROOT, "deploy", "artifacts", "plugin")
    for f in ("dfa_plugin.cu", "dfa_i8_test.cu", "trt_parse_check.py"):
        bc.push(cli, os.path.join(src, f), f"{BOARD}/plugin/{f}")
    bc.run(cli, f"rm -f {BOARD}/logs/PLUGIN_DONE {BOARD}/logs/PLUGIN_FAIL "
                f"{BOARD}/logs/I8TEST_DONE {BOARD}/logs/I8TEST_FAIL", timeout_s=15)

    build_sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
cd {BOARD}/plugin || exit 1
# /opt/m0 noexec: .so 必须落 /usr/local/lib 才能被 dlopen，二进制必须落 /usr/local/bin
nvcc -O3 -std=c++17 -arch=sm_87 -shared -Xcompiler -fPIC -o /usr/local/lib/libdfa_sd.so dfa_plugin.cu -lnvinfer > $LOG/plugin_build.log 2>&1
rc=$?
echo "so_build rc=$rc" >> $LOG/plugin_build.log
if [ $rc -eq 0 ] && [ -s /usr/local/lib/libdfa_sd.so ]; then touch $LOG/PLUGIN_DONE; else echo $rc > $LOG/PLUGIN_FAIL; fi
exit $rc
"""
    bc.push_script(cli, build_sh, f"{BOARD}/plugin/build.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/build.sh", f"{BOARD}/logs/plugin_build_wrap.log")
    assert poll_done_or_fail(cli, "PLUGIN"), "插件编译超时/失败"

    test_sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
cd {BOARD}/plugin || exit 1
nvcc -O3 -std=c++14 -I /usr/src/tensorrt/include dfa_plugin.cu dfa_i8_test.cu -o /usr/local/bin/dfa_i8_test -lcudart -lnvinfer > $LOG/i8test_build.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo $rc > $LOG/I8TEST_FAIL; exit $rc; fi
/usr/local/bin/dfa_i8_test > $LOG/dfa_i8_test.log 2>&1
rc=$?
if [ $rc -eq 0 ]; then touch $LOG/I8TEST_DONE; else echo $rc > $LOG/I8TEST_FAIL; fi
exit $rc
"""
    bc.push_script(cli, test_sh, f"{BOARD}/plugin/test.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/test.sh", f"{BOARD}/logs/i8test_wrap.log")
    assert poll_done_or_fail(cli, "I8TEST"), "kernel 单测编译/运行失败"

    rc, out = bc.run(cli, f"tail -n 20 {BOARD}/logs/dfa_i8_test.log", timeout_s=15)
    print(out)
    rc, md = bc.run(cli, f"md5sum /usr/local/lib/libdfa_sd.so; rm -f {BOARD}/plugin/libdfa_sd.so", timeout_s=15)
    print(md.strip())
    assert "PASS" in out.upper() or "pass" in out, "单测日志未见 PASS"
    print("PLUGIN STAGE PASS")


def poll_done_or_fail(cli, tag: str, deadline_s: float = 600.0, poll_s: float = 5.0):
    """DONE/FAIL 双 marker 轮询：DONE→True，FAIL→抛错并带回日志尾部。"""
    done, fail = f"{BOARD}/logs/{tag}_DONE", f"{BOARD}/logs/{tag}_FAIL"
    deadline = __import__("time").time() + deadline_s
    while __import__("time").time() < deadline:
        rc, out = bc.run(cli, f"test -f {done} && echo DONE; test -f {fail} && echo FAIL",
                         timeout_s=15)
        if "DONE" in out:
            return True
        if "FAIL" in out:
            rc, log = bc.run(cli, f"tail -n 30 {BOARD}/logs/*{tag.lower()}*.log 2>/dev/null | tail -n 40",
                             timeout_s=30)
            raise RuntimeError(f"{tag} 板上失败，日志尾部：\n{log}")
        __import__("time").sleep(poll_s)
    return False


def stage_sh(cli):
    """通用同步命令：python board_v2.py sh "<cmd>"。"""
    rc, out = bc.run(cli, sys.argv[2], timeout_s=300)
    print(out)
    print(f"rc={rc}")


def stage_log(cli):
    """看板端日志尾部：python board_v2.py log <remote_path> [n]。"""
    n = sys.argv[3] if len(sys.argv) > 3 else "40"
    rc, out = bc.run(cli, f"tail -n {n} {sys.argv[2]}", timeout_s=30)
    print(out)


def _md5_local(path: str) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_push_onnx(cli):
    """Task 4 Step 1：push 编译输入 ONNX + 24 样本输入树，双向 md5 验证。"""
    import glob as _g
    onnx_local = os.path.join(ROOT, "deploy", "artifacts", "sparsedrive_int8_qdq_folded.onnx")
    assert os.path.isfile(onnx_local), onnx_local
    print(f"pushing {onnx_local} (~198MB) ...")
    bc.push(cli, onnx_local, f"{BOARD}/onnx/sparsedrive_int8_qdq_folded.onnx")
    rc, out = bc.run(cli, f"md5sum {BOARD}/onnx/sparsedrive_int8_qdq_folded.onnx", timeout_s=120)
    board_md5 = out.split()[0]
    local_md5 = _md5_local(onnx_local)
    print(f"onnx md5 local={local_md5} board={board_md5}")
    assert board_md5 == local_md5, "ONNX 传输 md5 不一致"

    inroot = os.path.join(ROOT, "deploy", "artifacts", "engine_inputs", "ref_val")
    dirs = sorted(_g.glob(os.path.join(inroot, "val_*")))
    assert len(dirs) == 24, len(dirs)
    for i, d in enumerate(dirs):
        name = os.path.basename(d)
        for f in sorted(os.listdir(d)):
            bc.push(cli, os.path.join(d, f), f"{BOARD}/inputs/ref_val/{name}/{f}")
        if (i + 1) % 8 == 0:
            print(f"  {i + 1}/24 dirs pushed")
    # 抽样 4 目录 md5 对拍
    import random
    random.seed(0)
    for d in random.sample(dirs, 4):
        name = os.path.basename(d)
        for f in sorted(os.listdir(d)):
            rc, out = bc.run(cli, f"md5sum {BOARD}/inputs/ref_val/{name}/{f}", timeout_s=30)
            assert out.split()[0] == _md5_local(os.path.join(d, f)), f"{name}/{f} md5 mismatch"
    print("PUSH_ONNX PASS")


def stage_parse_check(cli):
    """Task 4 Step 2 主路径：板载 python3.8（有 tensorrt bindings，无 numpy 也能跑）。
    先推脚本——本地改完直接跑会踩板上旧版（2026-10-05 实坑）。"""
    src = os.path.join(ROOT, "deploy", "artifacts", "plugin", "trt_parse_check.py")
    bc.push(cli, src, f"{BOARD}/plugin/trt_parse_check.py")
    cmd = (f"cd {BOARD} && python3.8 plugin/trt_parse_check.py "
           f"/usr/local/lib/libdfa_sd.so onnx/sparsedrive_int8_qdq_folded.onnx")
    rc, out = bc.run(cli, cmd, timeout_s=600)
    print(out)
    assert rc == 0 and "PASS" in out, "parse check 失败"
    print("PARSE_CHECK PASS")


def stage_build_engine(cli):
    """Task 4 Step 3：全图编译（后台 + DONE/FAIL marker，deadline 90min）。"""
    bc.run(cli, f"rm -f {BOARD}/logs/BUILD_DONE {BOARD}/logs/BUILD_FAIL", timeout_s=15)
    build_sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
mkdir -p {BOARD}/engine {BOARD}/prof
# QDQ 显式量化图在本 TRT 8.6.1.2 必须 --int8（否则 network validate 报
# "Int8 precision has been set ... but int8 is not configured"）；图内自带
# scale/zp，无需校准。v1 全部交付命令同为 --int8 --fp16。
/usr/src/tensorrt/bin/trtexec --onnx={BOARD}/onnx/sparsedrive_int8_qdq_folded.onnx --saveEngine={BOARD}/engine/e_sd2.engine --int8 --fp16 --plugins=/usr/local/lib/libdfa_sd.so --timingCacheFile={BOARD}/engine/sd.cache --memPoolSize=workspace:4096 > $LOG/build.log 2>&1
rc=$?
tail -n 5 $LOG/build.log >> $LOG/build.log
if [ $rc -eq 0 ] && [ -s {BOARD}/engine/e_sd2.engine ]; then touch $LOG/BUILD_DONE; else echo $rc > $LOG/BUILD_FAIL; fi
exit $rc
"""
    bc.push_script(cli, build_sh, f"{BOARD}/plugin/build_engine.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/build_engine.sh", f"{BOARD}/logs/build_wrap.log")
    print("build launched, polling (deadline 5400s) ...")
    assert poll_done_or_fail(cli, "BUILD", deadline_s=5400.0, poll_s=15.0), "编译超时"
    rc, out = bc.run(cli, f"ls -s {BOARD}/engine/e_sd2.engine; md5sum {BOARD}/engine/e_sd2.engine",
                     timeout_s=120)
    print(out)
    print("BUILD_ENGINE PASS")


def stage_baseline(cli):
    """Task 4 Step 4：run_engines 端到端延迟基线（100 iters cudaEvent MEAN）。"""
    bc.run(cli, f"rm -f {BOARD}/logs/BASE_DONE {BOARD}/logs/BASE_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
mkdir -p {BOARD}/outs/bench00
/usr/local/bin/run_engines {BOARD}/engine/e_sd2.engine /usr/local/lib/libdfa_sd.so {BOARD}/inputs/ref_val/val_00 --warmup 10 --iters 100 --dump {BOARD}/outs/bench00 > $LOG/baseline.log 2>&1
rc=$?
if [ $rc -eq 0 ]; then touch $LOG/BASE_DONE; else echo $rc > $LOG/BASE_FAIL; fi
exit $rc
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/run_baseline.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/run_baseline.sh", f"{BOARD}/logs/baseline_wrap.log")
    assert poll_done_or_fail(cli, "BASE", deadline_s=900.0), "基线运行失败"
    rc, out = bc.run(cli, f"tail -n 25 {BOARD}/logs/baseline.log", timeout_s=30)
    print(out)
    print("BASELINE PASS")


def stage_profile(cli):
    """Task 4 Step 5：trtexec 逐层 profile（300 iters）+ layerinfo 导出。"""
    bc.run(cli, f"rm -f {BOARD}/logs/PROF_DONE {BOARD}/logs/PROF_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
/usr/src/tensorrt/bin/trtexec --loadEngine={BOARD}/engine/e_sd2.engine --plugins=/usr/local/lib/libdfa_sd.so --dumpProfile --exportProfile={BOARD}/prof/prof_m1.json --exportLayerInfo={BOARD}/prof/layerinfo.json --warmUp=200 --iterations=300 --avgRuns=10 --useSpinWait > $LOG/prof.log 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -s {BOARD}/prof/prof_m1.json ]; then touch $LOG/PROF_DONE; else echo $rc > $LOG/PROF_FAIL; fi
exit $rc
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/run_prof.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/run_prof.sh", f"{BOARD}/logs/prof_wrap.log")
    assert poll_done_or_fail(cli, "PROF", deadline_s=1200.0), "profile 运行失败"
    rc, out = bc.run(cli, f"tail -n 15 {BOARD}/logs/prof.log", timeout_s=30)
    print(out)
    print("PROFILE PASS")


def stage_fetch(cli):
    """拉板端产物：python board_v2.py fetch <remote_subdir> <local_dir> [pattern]。
    相对路径按 BOARD_WORK 解析（2026-10-05 实坑：sftp 不吃 cwd 相对路径）。"""
    remote = sys.argv[2]
    if not remote.startswith("/"):
        remote = f"{BOARD}/{remote}"
    local = sys.argv[3]
    pattern = sys.argv[4] if len(sys.argv) > 4 else "*"
    n = bc.pull(cli, remote, local, pattern)
    print(f"pulled {n} files -> {local}")


def stage_py38(cli):
    """推本地 python 脚本到板用 python3.8 跑：board_v2.py py38 <local.py> [args...]"""
    import time
    local = sys.argv[2]
    remote = f"/tmp/sd2_py{int(time.time())}.py"
    bc.push_script(cli, open(local, "r", encoding="utf-8").read(), remote)
    cmd = "python3.8 " + remote + " " + " ".join(sys.argv[3:])
    rc, out = bc.run(cli, cmd, timeout_s=600)
    print(out)
    print(f"rc={rc}")


def stage_dump24(cli):
    """Task 5 Step 2：24 样本逐一推理并 dump 输出（outs/dump24/val_XX）。
    run_engines --dump 落最后一次 enqueue 的输出 + manifest.tsv（binding 序）。"""
    bc.run(cli, f"rm -f {BOARD}/logs/DUMP_DONE {BOARD}/logs/DUMP_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
mkdir -p {BOARD}/outs/dump24
for d in {BOARD}/inputs/ref_val/val_*; do
  n=$(basename $d)
  mkdir -p {BOARD}/outs/dump24/$n
  /usr/local/bin/run_engines {BOARD}/engine/e_sd2.engine /usr/local/lib/libdfa_sd.so \\
    $d --warmup 3 --iters 1 --dump {BOARD}/outs/dump24/$n >> $LOG/dump24.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "$n rc=$rc" >> $LOG/dump24.log; echo $rc > $LOG/DUMP_FAIL; exit $rc; fi
done
echo all_done >> $LOG/dump24.log
touch $LOG/DUMP_DONE
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/run_dump24.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/run_dump24.sh", f"{BOARD}/logs/dump24_wrap.log")
    assert poll_done_or_fail(cli, "DUMP", deadline_s=1800.0), "dump24 失败"
    rc, out = bc.run(cli, f"tail -n 5 {BOARD}/logs/dump24.log; "
                          f"ls {BOARD}/outs/dump24 | wc -l", timeout_s=30)
    print(out)
    print("DUMP24 PASS")


def stage_push138(cli):
    """Task 6 Step 2a：推 138 场景输入树 inputs/mini138（~630MB），抽样 md5 对拍。"""
    import glob as _g
    import random
    inroot = os.path.join(ROOT, "deploy", "artifacts", "engine_inputs", "mini138")
    dirs = sorted(d for d in _g.glob(os.path.join(inroot, "*")) if os.path.isdir(d))
    assert len(dirs) == 138, f"expect 138 token dirs, got {len(dirs)}"
    total = 0
    for i, d in enumerate(dirs):
        name = os.path.basename(d)
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            total += os.path.getsize(p)
            bc.push(cli, p, f"{BOARD}/inputs/mini138/{name}/{f}")
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/138 pushed ({total / 1e6:.0f} MB so far)", flush=True)
    print(f"pushed 138 dirs, {total / 1e6:.0f} MB total")
    random.seed(0)
    for d in random.sample(dirs, 6):
        name = os.path.basename(d)
        for f in sorted(os.listdir(d)):
            rc, out = bc.run(cli, f"md5sum {BOARD}/inputs/mini138/{name}/{f}", timeout_s=30)
            assert out.split()[0] == _md5_local(os.path.join(d, f)), f"{name}/{f} md5 mismatch"
    print("PUSH138 PASS")


def stage_run138(cli):
    """Task 6 Step 2b：板上 138 场景 e_fix2full 逐一推理并 dump（outs/mini138/<token>）。"""
    bc.run(cli, f"rm -f {BOARD}/logs/RUN138_DONE {BOARD}/logs/RUN138_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
ENG={BOARD}/engine/e_fix2full.engine
test -s $ENG || {{ echo "engine missing" > $LOG/RUN138_FAIL; exit 1; }}
n=0
for d in {BOARD}/inputs/mini138/*/; do
  tok=$(basename $d)
  mkdir -p {BOARD}/outs/mini138/$tok
  /usr/local/bin/run_engines $ENG /usr/local/lib/libdfa_sd.so \\
    $d --warmup 1 --iters 1 --dump {BOARD}/outs/mini138/$tok >> $LOG/run138.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "$tok rc=$rc" >> $LOG/run138.log; echo $rc > {BOARD}/logs/RUN138_FAIL; exit $rc; fi
  n=$((n+1))
done
echo "scenes=$n" >> $LOG/run138.log
touch {BOARD}/logs/RUN138_DONE
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/run138.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/run138.sh", f"{BOARD}/logs/run138_wrap.log")
    print("run138 launched, polling (deadline 3600s) ...", flush=True)
    assert poll_done_or_fail(cli, "RUN138", deadline_s=3600.0, poll_s=15.0), "run138 超时"
    rc, out = bc.run(cli, f"tail -n 3 {BOARD}/logs/run138.log; "
                          f"ls {BOARD}/outs/mini138 | wc -l", timeout_s=30)
    print(out)
    print("RUN138 PASS")


def stage_fetch138(cli):
    """Task 6 Step 2c：拉回 138 场景全部 dump（outputs 很小，~12KB/场景全拉）。"""
    local = os.path.join(ROOT, "deploy", "artifacts", "board_outs", "mini138")
    n = bc.pull(cli, f"{BOARD}/outs/mini138", local, "*")
    print(f"pulled {n} files -> {local}")


def stage_pgbench(cli):
    """Pg 优化 Step 1：push 新插件源码 → 备份并重编 libdfa_sd.so（含 v1+Pg 两
    个 creator，旧引擎反序列化不受影响）→ 编 dfa_bench_pg → 跑 real/worst 两
    分布 bench（对拍+确定性+计时）。"""
    src = os.path.join(ROOT, "deploy", "artifacts", "plugin")
    for f in ("dfa_plugin.cu", "dfa_bench_pg.cu"):
        bc.push(cli, os.path.join(src, f), f"{BOARD}/plugin/{f}")
    bc.run(cli, f"rm -f {BOARD}/logs/PGBENCH_DONE {BOARD}/logs/PGBENCH_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
cd {BOARD}/plugin || exit 1
cp -n /usr/local/lib/libdfa_sd.so /usr/local/lib/libdfa_sd.prePg.bak
nvcc -O3 -std=c++17 -arch=sm_87 -shared -Xcompiler -fPIC -o /usr/local/lib/libdfa_sd.so dfa_plugin.cu -lnvinfer > $LOG/pg_so_build.log 2>&1
rc=$?
echo "pg so_build rc=$rc" >> $LOG/pg_so_build.log
if [ $rc -ne 0 ]; then echo $rc > $LOG/PGBENCH_FAIL; exit $rc; fi
nvcc -O3 -std=c++17 -arch=sm_87 -I /usr/src/tensorrt/include dfa_plugin.cu dfa_bench_pg.cu -o /usr/local/bin/dfa_bench_pg -lcudart -lnvinfer > $LOG/pg_bench_build.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo $rc > $LOG/PGBENCH_FAIL; exit $rc; fi
echo "=== real ===" > $LOG/pg_bench.log
/usr/local/bin/dfa_bench_pg real 20 0.0 >> $LOG/pg_bench.log 2>&1
r1=$?
echo "=== worst ===" >> $LOG/pg_bench.log
/usr/local/bin/dfa_bench_pg worst 20 0.0 >> $LOG/pg_bench.log 2>&1
r2=$?
if [ $r1 -eq 0 ] && [ $r2 -eq 0 ]; then touch $LOG/PGBENCH_DONE; else echo "$r1 $r2" > $LOG/PGBENCH_FAIL; fi
exit 0
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/pgbench.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/pgbench.sh", f"{BOARD}/logs/pgbench_wrap.log")
    print("pgbench launched, polling ...", flush=True)
    assert poll_done_or_fail(cli, "PGBENCH", deadline_s=1800.0, poll_s=10.0), "pgbench 超时"
    rc, out = bc.run(cli, f"cat {BOARD}/logs/pg_bench.log; "
                          f"md5sum /usr/local/lib/libdfa_sd.so", timeout_s=30)
    print(out)
    print("PGBENCH STAGE DONE")


def stage_pgbuild(cli):
    """Pg 优化 Step 2：push fix3 onnx（softmax 并入插件）→ trtexec 重编
    e_fix3full_h.engine（命令行与 e_fix2full_h 完全一致）。"""
    onnx_local = os.path.join(ROOT, "deploy", "artifacts",
                              "sparsedrive_fp16_graph_fix3.onnx")
    bc.push(cli, onnx_local, f"{BOARD}/onnx/sparsedrive_fp16_graph_fix3.onnx")
    bc.run(cli, f"rm -f {BOARD}/logs/PGBUILD_DONE {BOARD}/logs/PGBUILD_FAIL",
           timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
/usr/src/tensorrt/bin/trtexec --onnx={BOARD}/onnx/sparsedrive_fp16_graph_fix3.onnx --saveEngine={BOARD}/engine/e_fix3full_h.engine --fp16 --plugins=/usr/local/lib/libdfa_sd.so --timingCacheFile={BOARD}/engine/sd.cache --memPoolSize=workspace:4096 > $LOG/pg_build.log 2>&1
rc=$?
echo "pgbuild rc=$rc" >> $LOG/pg_build.log
if [ $rc -eq 0 ] && [ -s {BOARD}/engine/e_fix3full_h.engine ]; then touch $LOG/PGBUILD_DONE; else echo $rc > $LOG/PGBUILD_FAIL; fi
exit 0
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/pgbuild.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/pgbuild.sh", f"{BOARD}/logs/pgbuild_wrap.log")
    print("pgbuild launched, polling (deadline 3600s) ...", flush=True)
    assert poll_done_or_fail(cli, "PGBUILD", deadline_s=3600.0, poll_s=20.0), "pgbuild 超时"
    rc, out = bc.run(cli, f"grep -E 'dfapg|PGBUILD|Engine built' {BOARD}/logs/pg_build.log | "
                          f"tail -n 5; md5sum {BOARD}/engine/e_fix3full_h.engine", timeout_s=30)
    print(out)
    print("PGBUILD STAGE DONE")


def stage_pgval(cli):
    """Pg 优化 Step 3：真数据（val_00）e2e 计时（eager + graph）+ trtexec
    逐层 profile（--loadInputs 真实输入，规避随机输入假象）。"""
    bc.run(cli, f"rm -f {BOARD}/logs/PGVAL_DONE {BOARD}/logs/PGVAL_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
ENG={BOARD}/engine/e_fix3full_h.engine
IN={BOARD}/inputs/ref_val/val_00
test -s $ENG || {{ echo "engine missing" > $LOG/PGVAL_FAIL; exit 1; }}
/usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so $IN --warmup 10 --iters 100 > $LOG/pgval_eager.log 2>&1
r1=$?
/usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so $IN --warmup 10 --iters 100 --graph > $LOG/pgval_graph.log 2>&1
r2=$?
/usr/src/tensorrt/bin/trtexec --loadEngine=$ENG --plugins=/usr/local/lib/libdfa_sd.so --loadInputs=imgs:$IN/imgs.bin,projection_mat:$IN/projection_mat.bin,image_wh:$IN/image_wh.bin,status_feature:$IN/status_feature.bin --dumpProfile --exportProfile={BOARD}/prof/prof_real_pg.json --warmUp=0 --iterations=100 --avgRuns=100 --useSpinWait > $LOG/pgval_prof.log 2>&1
r3=$?
if [ $r1 -eq 0 ] && [ $r2 -eq 0 ] && [ $r3 -eq 0 ]; then touch $LOG/PGVAL_DONE; else echo "$r1 $r2 $r3" > $LOG/PGVAL_FAIL; fi
exit 0
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/pgval.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/pgval.sh", f"{BOARD}/logs/pgval_wrap.log")
    print("pgval launched, polling ...", flush=True)
    assert poll_done_or_fail(cli, "PGVAL", deadline_s=1800.0, poll_s=10.0), "pgval 超时"
    rc, out = bc.run(cli, f"grep -E 'MEAN|dfapg' {BOARD}/logs/pgval_eager.log "
                          f"{BOARD}/logs/pgval_graph.log | head -n 8", timeout_s=30)
    print(out)
    print("PGVAL STAGE DONE")


def stage_pgm2(cli):
    """Pg 优化 Step 4：e_fix3full_h 逐样本 dump 24 样本 -> outs/pg24/<val_XX>。"""
    bc.run(cli, f"rm -f {BOARD}/logs/PGM2_DONE {BOARD}/logs/PGM2_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
ENG={BOARD}/engine/e_fix3full_h.engine
test -s $ENG || {{ echo "engine missing" > $LOG/PGM2_FAIL; exit 1; }}
mkdir -p {BOARD}/outs/pg24
n=0
for d in {BOARD}/inputs/ref_val/val_*; do
  tok=$(basename $d)
  mkdir -p {BOARD}/outs/pg24/$tok
  /usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so \\
    $d --warmup 3 --iters 1 --dump {BOARD}/outs/pg24/$tok >> $LOG/pgm2.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "$tok rc=$rc" >> $LOG/pgm2.log; echo $rc > $LOG/PGM2_FAIL; exit $rc; fi
  n=$((n+1))
done
echo "samples=$n" >> $LOG/pgm2.log
touch $LOG/PGM2_DONE
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/pgm2.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/pgm2.sh", f"{BOARD}/logs/pgm2_wrap.log")
    print("pgm2 launched, polling ...", flush=True)
    assert poll_done_or_fail(cli, "PGM2", deadline_s=1200.0, poll_s=10.0), "pgm2 超时"
    rc, out = bc.run(cli, f"tail -n 3 {BOARD}/logs/pgm2.log; "
                          f"ls {BOARD}/outs/pg24 | wc -l", timeout_s=30)
    print(out)
    print("PGM2 PASS")


def stage_run138pg(cli):
    """Pg 优化 Step 5：e_fix3full_h 跑 138 场景 dump -> outs/mini138pg/<token>。"""
    bc.run(cli, f"rm -f {BOARD}/logs/RUN138PG_DONE {BOARD}/logs/RUN138PG_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
ENG={BOARD}/engine/e_fix3full_h.engine
test -s $ENG || {{ echo "engine missing" > $LOG/RUN138PG_FAIL; exit 1; }}
n=0
for d in {BOARD}/inputs/mini138/*/; do
  tok=$(basename $d)
  mkdir -p {BOARD}/outs/mini138pg/$tok
  /usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so \\
    $d --warmup 1 --iters 1 --dump {BOARD}/outs/mini138pg/$tok >> $LOG/run138pg.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "$tok rc=$rc" >> $LOG/run138pg.log; echo $rc > $LOG/RUN138PG_FAIL; exit $rc; fi
  n=$((n+1))
done
echo "scenes=$n" >> $LOG/run138pg.log
touch $LOG/RUN138PG_DONE
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/run138pg.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/run138pg.sh", f"{BOARD}/logs/run138pg_wrap.log")
    print("run138pg launched, polling (deadline 3600s) ...", flush=True)
    assert poll_done_or_fail(cli, "RUN138PG", deadline_s=3600.0, poll_s=15.0), "run138pg 超时"
    rc, out = bc.run(cli, f"tail -n 3 {BOARD}/logs/run138pg.log; "
                          f"ls {BOARD}/outs/mini138pg | wc -l", timeout_s=30)
    print(out)
    print("RUN138PG PASS")


def stage_pgreal(cli):
    """DFA 瓶颈归因：真数据（kstat loc/w bin）bench_real —— plan/gather/地板/
    L2 驻留/宽加载 六对照计时 + plan 输出与宿主构建对拍。只编 bench 二进制，
    不动 libdfa_sd.so。"""
    src = os.path.join(ROOT, "deploy", "artifacts", "plugin")
    bc.push(cli, os.path.join(src, "dfa_bench_real.cu"), f"{BOARD}/plugin/dfa_bench_real.cu")
    bc.run(cli, f"rm -f {BOARD}/logs/PGREAL_DONE {BOARD}/logs/PGREAL_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
cd {BOARD}/plugin || exit 1
nvcc -O3 -std=c++17 -arch=sm_87 dfa_bench_real.cu -o /usr/local/bin/dfa_bench_real -lcudart -L/usr/local/lib -ldfa_sd -Xlinker -rpath -Xlinker /usr/local/lib > $LOG/pgreal_build.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then tail -n 30 $LOG/pgreal_build.log; echo $rc > $LOG/PGREAL_FAIL; exit $rc; fi
/usr/local/bin/dfa_bench_real > $LOG/pgreal_bench.log 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q BENCH_REAL_DONE $LOG/pgreal_bench.log; then touch $LOG/PGREAL_DONE; else echo $rc > $LOG/PGREAL_FAIL; fi
exit $rc
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/pgreal.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/pgreal.sh", f"{BOARD}/logs/pgreal_wrap.log")
    print("pgreal launched, polling ...", flush=True)
    assert poll_done_or_fail(cli, "PGREAL", deadline_s=1800.0, poll_s=10.0), "pgreal 超时"
    rc, out = bc.run(cli, f"cat {BOARD}/logs/pgreal_bench.log", timeout_s=30)
    print(out)
    print("PGREAL PASS")


def _make_qdq_stages(tag):
    """QDQ 嫁接引擎三件套：build(--int8 --fp16) / val(真数据计时+profile) / m2(24样本 dump)。
    tag=e1/e2 -> onnx sparsedrive_fp16_graph_fix3_qdq_<tag>.onnx,
    engine e_fix3full_h_qdq_<tag>.engine。
    tag 后缀变体：'f' = 不加 --int8（QDQ 仅作 cast）；'n' = 不用 timingCache。"""
    base = tag.rstrip("fn")
    onnx_name = f"sparsedrive_fp16_graph_fix3_qdq_{base}.onnx"
    eng = f"{BOARD}/engine/e_fix3full_h_qdq_{tag}.engine"
    int8 = "--int8" if not tag.endswith("f") else ""
    cache = (f"--timingCacheFile={BOARD}/engine/sd.cache"
             if not tag.endswith("n") else "")

    def build(cli):
        onnx_local = os.path.join(ROOT, "deploy", "artifacts", onnx_name)
        bc.push(cli, onnx_local, f"{BOARD}/onnx/{onnx_name}")
        bc.run(cli, f"rm -f {BOARD}/logs/QDQB_DONE {BOARD}/logs/QDQB_FAIL",
               timeout_s=15)
        sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
/usr/src/tensorrt/bin/trtexec --onnx={BOARD}/onnx/{onnx_name} --saveEngine={eng} {int8} --fp16 --plugins=/usr/local/lib/libdfa_sd.so {cache} --memPoolSize=workspace:4096 > $LOG/qdqb_{tag}.log 2>&1
rc=$?
echo "qdqbuild rc=$rc" >> $LOG/qdqb_{tag}.log
if [ $rc -eq 0 ] && [ -s {eng} ]; then touch $LOG/QDQB_DONE; else echo $rc > $LOG/QDQB_FAIL; fi
exit 0
"""
        bc.push_script(cli, sh, f"{BOARD}/plugin/qdqb_{tag}.sh")
        bc.launch(cli, f"bash {BOARD}/plugin/qdqb_{tag}.sh",
                  f"{BOARD}/logs/qdqb_{tag}_wrap.log")
        print(f"qdqbuild[{tag}] launched, polling (deadline 5400s) ...", flush=True)
        assert poll_done_or_fail(cli, "QDQB", deadline_s=5400.0, poll_s=20.0), \
            "qdqbuild 超时"
        rc, out = bc.run(
            cli,
            f"tail -n 4 {BOARD}/logs/qdqb_{tag}.log; md5sum {eng}", timeout_s=30)
        print(out)
        print(f"QDQB[{tag}] STAGE DONE")

    def val(cli):
        bc.run(cli, f"rm -f {BOARD}/logs/QDQV_DONE {BOARD}/logs/QDQV_FAIL",
               timeout_s=15)
        sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
ENG={eng}
IN={BOARD}/inputs/ref_val/val_00
test -s $ENG || {{ echo "engine missing" > $LOG/QDQV_FAIL; exit 1; }}
/usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so $IN --warmup 10 --iters 100 > $LOG/qdqv_{tag}_eager.log 2>&1
r1=$?
/usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so $IN --warmup 10 --iters 100 --graph > $LOG/qdqv_{tag}_graph.log 2>&1
r2=$?
/usr/src/tensorrt/bin/trtexec --loadEngine=$ENG --plugins=/usr/local/lib/libdfa_sd.so --loadInputs=imgs:$IN/imgs.bin,projection_mat:$IN/projection_mat.bin,image_wh:$IN/image_wh.bin,status_feature:$IN/status_feature.bin --dumpProfile --exportProfile={BOARD}/prof/prof_real_qdq_{tag}.json --warmUp=0 --iterations=100 --avgRuns=100 --useSpinWait > $LOG/qdqv_{tag}_prof.log 2>&1
r3=$?
if [ $r1 -eq 0 ] && [ $r2 -eq 0 ] && [ $r3 -eq 0 ]; then touch $LOG/QDQV_DONE; else echo "$r1 $r2 $r3" > $LOG/QDQV_FAIL; fi
exit 0
"""
        bc.push_script(cli, sh, f"{BOARD}/plugin/qdqv_{tag}.sh")
        bc.launch(cli, f"bash {BOARD}/plugin/qdqv_{tag}.sh",
                  f"{BOARD}/logs/qdqv_{tag}_wrap.log")
        print(f"qdqval[{tag}] launched, polling ...", flush=True)
        assert poll_done_or_fail(cli, "QDQV", deadline_s=1800.0, poll_s=10.0), \
            "qdqval 超时"
        rc, out = bc.run(
            cli,
            f"grep -E 'MEAN' {BOARD}/logs/qdqv_{tag}_eager.log "
            f"{BOARD}/logs/qdqv_{tag}_graph.log | head -n 4", timeout_s=30)
        print(out)
        print(f"QDQV[{tag}] STAGE DONE")

    def m2(cli):
        bc.run(cli, f"rm -f {BOARD}/logs/QDQM2_DONE {BOARD}/logs/QDQM2_FAIL",
               timeout_s=15)
        sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
ENG={eng}
test -s $ENG || {{ echo "engine missing" > $LOG/QDQM2_FAIL; exit 1; }}
mkdir -p {BOARD}/outs/qdq24_{tag}
rm -rf {BOARD}/outs/qdq24_{tag}/*
n=0
for d in {BOARD}/inputs/ref_val/val_*; do
  tok=$(basename $d)
  mkdir -p {BOARD}/outs/qdq24_{tag}/$tok
  /usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so \\
    $d --warmup 3 --iters 1 --dump {BOARD}/outs/qdq24_{tag}/$tok >> $LOG/qdqm2_{tag}.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "$tok rc=$rc" >> $LOG/qdqm2_{tag}.log; echo $rc > $LOG/QDQM2_FAIL; exit $rc; fi
  n=$((n+1))
done
echo "samples=$n" >> $LOG/qdqm2_{tag}.log
touch $LOG/QDQM2_DONE
"""
        bc.push_script(cli, sh, f"{BOARD}/plugin/qdqm2_{tag}.sh")
        bc.launch(cli, f"bash {BOARD}/plugin/qdqm2_{tag}.sh",
                  f"{BOARD}/logs/qdqm2_{tag}_wrap.log")
        print(f"qdqm2[{tag}] launched, polling ...", flush=True)
        assert poll_done_or_fail(cli, "QDQM2", deadline_s=1200.0, poll_s=10.0), \
            "qdqm2 超时"
        rc, out = bc.run(cli, f"tail -n 2 {BOARD}/logs/qdqm2_{tag}.log; "
                              f"ls {BOARD}/outs/qdq24_{tag} | wc -l", timeout_s=30)
        print(out)
        print(f"QDQM2[{tag}] PASS")

    return {"build": build, "val": val, "m2": m2}


_QDQ_E1 = _make_qdq_stages("e1")
_QDQ_E2 = _make_qdq_stages("e2")
_QDQ_E1N = _make_qdq_stages("e1n")
_QDQ_E1F = _make_qdq_stages("e1f")


def stage_pg4(cli):
    """DFA v4：Entry 64B→48B（wg half 化）+ plan phase C 除法→s_inv 乘法。
    备份 v3 .so（libdfa_sd.v3.bak）→ 重编 libdfa_sd.so（接口不变，引擎不重编）
    → bench_v4：plan counts/off/wk 对拍 + wg half 数值窗 + 确定性 + T1/T4 计时
    + 输出对宿主 fp32 参考的数值差。"""
    src = os.path.join(ROOT, "deploy", "artifacts", "plugin")
    for f in ("dfa_plugin.cu", "dfa_bench_v4.cu"):
        bc.push(cli, os.path.join(src, f), f"{BOARD}/plugin/{f}")
    bc.run(cli, f"rm -f {BOARD}/logs/PG4_DONE {BOARD}/logs/PG4_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
cd {BOARD}/plugin || exit 1
[ -f /usr/local/lib/libdfa_sd.v3.bak ] || cp /usr/local/lib/libdfa_sd.so /usr/local/lib/libdfa_sd.v3.bak
md5sum /usr/local/lib/libdfa_sd.v3.bak >> $LOG/pg4_so_build.log
nvcc -O3 -std=c++17 -arch=sm_87 -shared -Xcompiler -fPIC -o /usr/local/lib/libdfa_sd.so dfa_plugin.cu -lnvinfer > $LOG/pg4_so_build.log 2>&1
rc=$?
echo "pg4 so_build rc=$rc" >> $LOG/pg4_so_build.log
if [ $rc -ne 0 ]; then tail -n 30 $LOG/pg4_so_build.log; echo $rc > $LOG/PG4_FAIL; exit $rc; fi
nvcc -O3 -std=c++17 -arch=sm_87 dfa_bench_v4.cu -o /usr/local/bin/dfa_bench_v4 -lcudart -L/usr/local/lib -ldfa_sd -Xlinker -rpath -Xlinker /usr/local/lib > $LOG/pg4_bench_build.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then tail -n 30 $LOG/pg4_bench_build.log; echo $rc > $LOG/PG4_FAIL; exit $rc; fi
/usr/local/bin/dfa_bench_v4 > $LOG/pg4_bench.log 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q BENCH_V4_DONE $LOG/pg4_bench.log; then touch $LOG/PG4_DONE; else echo $rc > $LOG/PG4_FAIL; fi
exit $rc
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/pg4.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/pg4.sh", f"{BOARD}/logs/pg4_wrap.log")
    print("pg4 launched, polling ...", flush=True)
    assert poll_done_or_fail(cli, "PG4", deadline_s=1800.0, poll_s=10.0), "pg4 超时"
    rc, out = bc.run(cli, f"cat {BOARD}/logs/pg4_bench.log; "
                          f"md5sum /usr/local/lib/libdfa_sd.so", timeout_s=30)
    print(out)
    print("PG4 PASS")


def stage_mha(cli):
    """MHA Step 1：push dfa_plugin.cu(+FusedMHA) 与 bench → 备份 v4 .so
    （guard 防重复覆盖）→ 重编 libdfa_sd.so（DFA v4 + FusedMHA 同 TU）
    → 编 bench（同 TU 直链）→ 跑 bench：C1 图保真 fp16 参考对拍
    （abs/rel 混合门）+ C2 确定性 + T 计时 + amp=8 sharp pass。"""
    src = os.path.join(ROOT, "deploy", "artifacts", "plugin")
    for f in ("dfa_plugin.cu", "dfa_bench_mha.cu"):
        bc.push(cli, os.path.join(src, f), f"{BOARD}/plugin/{f}")
    bc.run(cli, f"rm -f {BOARD}/logs/MHA_DONE {BOARD}/logs/MHA_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
cd {BOARD}/plugin || exit 1
[ -f /usr/local/lib/libdfa_sd.v4.bak ] || cp /usr/local/lib/libdfa_sd.so /usr/local/lib/libdfa_sd.v4.bak
md5sum /usr/local/lib/libdfa_sd.v4.bak >> $LOG/mha_so_build.log
nvcc -O3 -std=c++17 -arch=sm_87 -shared -Xcompiler -fPIC -o /usr/local/lib/libdfa_sd.so dfa_plugin.cu -lnvinfer > $LOG/mha_so_build.log 2>&1
rc=$?
echo "mha so_build rc=$rc" >> $LOG/mha_so_build.log
if [ $rc -ne 0 ]; then tail -n 40 $LOG/mha_so_build.log; echo $rc > $LOG/MHA_FAIL; exit $rc; fi
md5sum /usr/local/lib/libdfa_sd.so >> $LOG/mha_so_build.log
nvcc -O3 -std=c++17 -arch=sm_87 dfa_plugin.cu dfa_bench_mha.cu -o /usr/local/bin/dfa_bench_mha -lcudart -lnvinfer > $LOG/mha_bench_build.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then tail -n 40 $LOG/mha_bench_build.log; echo $rc > $LOG/MHA_FAIL; exit $rc; fi
/usr/local/bin/dfa_bench_mha > $LOG/mha_bench.log 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q MHA_BENCH_DONE $LOG/mha_bench.log; then touch $LOG/MHA_DONE; else echo $rc > $LOG/MHA_FAIL; fi
exit $rc
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/mha.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/mha.sh", f"{BOARD}/logs/mha_wrap.log")
    print("mha launched, polling ...", flush=True)
    assert poll_done_or_fail(cli, "MHA", deadline_s=1800.0, poll_s=10.0), "mha 超时"
    rc, out = bc.run(cli, f"cat {BOARD}/logs/mha_bench.log; "
                          f"md5sum /usr/local/lib/libdfa_sd.so", timeout_s=30)
    print(out)
    print("MHA PASS")


def stage_mhabuild(cli):
    """MHA Step 2：push fix3_mha onnx → trtexec 重编 e_fix3mha.engine
    （命令行与 e_fix3full_h 一致，仅换 onnx）。"""
    onnx_local = os.path.join(ROOT, "deploy", "artifacts",
                              "sparsedrive_fp16_graph_fix3_mha.onnx")
    bc.push(cli, onnx_local, f"{BOARD}/onnx/sparsedrive_fp16_graph_fix3_mha.onnx")
    bc.run(cli, f"rm -f {BOARD}/logs/MHABUILD_DONE {BOARD}/logs/MHABUILD_FAIL",
           timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
/usr/src/tensorrt/bin/trtexec --onnx={BOARD}/onnx/sparsedrive_fp16_graph_fix3_mha.onnx --saveEngine={BOARD}/engine/e_fix3mha.engine --fp16 --plugins=/usr/local/lib/libdfa_sd.so --timingCacheFile={BOARD}/engine/sd.cache --memPoolSize=workspace:4096 > $LOG/mha_build.log 2>&1
rc=$?
echo "mhabuild rc=$rc" >> $LOG/mha_build.log
if [ $rc -eq 0 ] && [ -s {BOARD}/engine/e_fix3mha.engine ]; then touch $LOG/MHABUILD_DONE; else echo $rc > $LOG/MHABUILD_FAIL; fi
exit 0
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/mhabuild.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/mhabuild.sh", f"{BOARD}/logs/mhabuild_wrap.log")
    print("mhabuild launched, polling (deadline 3600s) ...", flush=True)
    assert poll_done_or_fail(cli, "MHABUILD", deadline_s=3600.0, poll_s=20.0), "mhabuild 超时"
    rc, out = bc.run(cli, f"grep -E 'fusedmha|MHABUILD|Engine built|FusedMHA' "
                          f"{BOARD}/logs/mha_build.log | tail -n 8; "
                          f"md5sum {BOARD}/engine/e_fix3mha.engine", timeout_s=30)
    print(out)
    print("MHABUILD STAGE DONE")


def stage_mhaval(cli):
    """MHA Step 3：真数据（val_00）e2e 计时（eager + graph）+ trtexec 逐层
    profile（真实输入）——与 pgval 同参，profile 存 prof_real_mha.json
    供 A/B 对比 prof_real_pg.json（myelin 分区风险仲裁）。"""
    bc.run(cli, f"rm -f {BOARD}/logs/MHAVAL_DONE {BOARD}/logs/MHAVAL_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
ENG={BOARD}/engine/e_fix3mha.engine
IN={BOARD}/inputs/ref_val/val_00
test -s $ENG || {{ echo "engine missing" > $LOG/MHAVAL_FAIL; exit 1; }}
/usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so $IN --warmup 10 --iters 100 > $LOG/mhaval_eager.log 2>&1
r1=$?
/usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so $IN --warmup 10 --iters 100 --graph > $LOG/mhaval_graph.log 2>&1
r2=$?
/usr/src/tensorrt/bin/trtexec --loadEngine=$ENG --plugins=/usr/local/lib/libdfa_sd.so --loadInputs=imgs:$IN/imgs.bin,projection_mat:$IN/projection_mat.bin,image_wh:$IN/image_wh.bin,status_feature:$IN/status_feature.bin --dumpProfile --exportProfile={BOARD}/prof/prof_real_mha.json --warmUp=0 --iterations=100 --avgRuns=100 --useSpinWait > $LOG/mhaval_prof.log 2>&1
r3=$?
if [ $r1 -eq 0 ] && [ $r2 -eq 0 ] && [ $r3 -eq 0 ]; then touch $LOG/MHAVAL_DONE; else echo "$r1 $r2 $r3" > $LOG/MHAVAL_FAIL; fi
exit 0
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/mhaval.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/mhaval.sh", f"{BOARD}/logs/mhaval_wrap.log")
    print("mhaval launched, polling ...", flush=True)
    assert poll_done_or_fail(cli, "MHAVAL", deadline_s=1800.0, poll_s=10.0), "mhaval 超时"
    rc, out = bc.run(cli, f"grep -E 'MEAN|fusedmha' {BOARD}/logs/mhaval_eager.log "
                          f"{BOARD}/logs/mhaval_graph.log | head -n 8", timeout_s=30)
    print(out)
    print("MHAVAL STAGE DONE")


def stage_mham2(cli):
    """MHA Step 4：e_fix3mha 逐样本 dump 24 样本 -> outs/mha24/<val_XX>。"""
    bc.run(cli, f"rm -f {BOARD}/logs/MHAM2_DONE {BOARD}/logs/MHAM2_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
ENG={BOARD}/engine/e_fix3mha.engine
test -s $ENG || {{ echo "engine missing" > $LOG/MHAM2_FAIL; exit 1; }}
mkdir -p {BOARD}/outs/mha24
n=0
for d in {BOARD}/inputs/ref_val/val_*; do
  tok=$(basename $d)
  mkdir -p {BOARD}/outs/mha24/$tok
  /usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so \\
    $d --warmup 3 --iters 1 --dump {BOARD}/outs/mha24/$tok >> $LOG/mham2.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "$tok rc=$rc" >> $LOG/mham2.log; echo $rc > $LOG/MHAM2_FAIL; exit $rc; fi
  n=$((n+1))
done
echo "samples=$n" >> $LOG/mham2.log
touch $LOG/MHAM2_DONE
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/mham2.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/mham2.sh", f"{BOARD}/logs/mham2_wrap.log")
    print("mham2 launched, polling ...", flush=True)
    assert poll_done_or_fail(cli, "MHAM2", deadline_s=1200.0, poll_s=10.0), "mham2 超时"
    rc, out = bc.run(cli, f"tail -n 3 {BOARD}/logs/mham2.log; "
                          f"ls {BOARD}/outs/mha24 | wc -l", timeout_s=30)
    print(out)
    print("MHAM2 PASS")


def stage_run138mha(cli):
    """MHA Step 5：e_fix3mha 跑 138 场景 dump -> outs/mini138mha/<token>。"""
    bc.run(cli, f"rm -f {BOARD}/logs/RUN138MHA_DONE {BOARD}/logs/RUN138MHA_FAIL",
           timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
ENG={BOARD}/engine/e_fix3mha.engine
test -s $ENG || {{ echo "engine missing" > $LOG/RUN138MHA_FAIL; exit 1; }}
n=0
for d in {BOARD}/inputs/mini138/*/; do
  tok=$(basename $d)
  mkdir -p {BOARD}/outs/mini138mha/$tok
  /usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so \\
    $d --warmup 1 --iters 1 --dump {BOARD}/outs/mini138mha/$tok >> $LOG/run138mha.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "$tok rc=$rc" >> $LOG/run138mha.log; echo $rc > $LOG/RUN138MHA_FAIL; exit $rc; fi
  n=$((n+1))
done
echo "scenes=$n" >> $LOG/run138mha.log
touch $LOG/RUN138MHA_DONE
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/run138mha.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/run138mha.sh", f"{BOARD}/logs/run138mha_wrap.log")
    print("run138mha launched, polling (deadline 3600s) ...", flush=True)
    assert poll_done_or_fail(cli, "RUN138MHA", deadline_s=3600.0, poll_s=15.0), "run138mha 超时"
    rc, out = bc.run(cli, f"tail -n 3 {BOARD}/logs/run138mha.log; "
                          f"ls {BOARD}/outs/mini138mha | wc -l", timeout_s=30)
    print(out)
    print("RUN138MHA PASS")


def stage_mha5(cli):
    """SumF Step 1：push dfa_plugin.cu(gather v5=anchor求和入插件) 与 bench_v5
    → 备份现役 .so（guard）→ 重编 libdfa_sd.so → 编 bench_v5 → 跑
    （3 几何 case：S1 counts + S5 anchor级数值门 + D 确定性 + T5 计时）。
    注意：.so 换 v5 后旧 e_fix3mha 引擎语义不兼容（静默错），勿再用它跑数。"""
    src = os.path.join(ROOT, "deploy", "artifacts", "plugin")
    for f in ("dfa_plugin.cu", "dfa_bench_v5.cu"):
        bc.push(cli, os.path.join(src, f), f"{BOARD}/plugin/{f}")
    bc.run(cli, f"rm -f {BOARD}/logs/MHA5_DONE {BOARD}/logs/MHA5_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
cd {BOARD}/plugin || exit 1
[ -f /usr/local/lib/libdfa_sd.presumf.bak ] || cp /usr/local/lib/libdfa_sd.so /usr/local/lib/libdfa_sd.presumf.bak
md5sum /usr/local/lib/libdfa_sd.presumf.bak >> $LOG/mha5_so_build.log
nvcc -O3 -std=c++17 -arch=sm_87 -shared -Xcompiler -fPIC -o /usr/local/lib/libdfa_sd.so dfa_plugin.cu -lnvinfer > $LOG/mha5_so_build.log 2>&1
rc=$?
echo "mha5 so_build rc=$rc" >> $LOG/mha5_so_build.log
if [ $rc -ne 0 ]; then tail -n 40 $LOG/mha5_so_build.log; echo $rc > $LOG/MHA5_FAIL; exit $rc; fi
md5sum /usr/local/lib/libdfa_sd.so >> $LOG/mha5_so_build.log
nvcc -O3 -std=c++17 -arch=sm_87 dfa_plugin.cu dfa_bench_v5.cu -o /usr/local/bin/dfa_bench_v5 -lcudart -lnvinfer > $LOG/mha5_bench_build.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then tail -n 40 $LOG/mha5_bench_build.log; echo $rc > $LOG/MHA5_FAIL; exit $rc; fi
/usr/local/bin/dfa_bench_v5 > $LOG/mha5_bench.log 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q BENCH_V5_DONE $LOG/mha5_bench.log; then touch $LOG/MHA5_DONE; else echo $rc > $LOG/MHA5_FAIL; fi
exit $rc
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/mha5.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/mha5.sh", f"{BOARD}/logs/mha5_wrap.log")
    print("mha5 launched, polling ...", flush=True)
    assert poll_done_or_fail(cli, "MHA5", deadline_s=1800.0, poll_s=10.0), "mha5 超时"
    rc, out = bc.run(cli, f"cat {BOARD}/logs/mha5_bench.log; "
                          f"md5sum /usr/local/lib/libdfa_sd.so", timeout_s=30)
    print(out)
    print("MHA5 PASS")


def stage_mha5build(cli):
    """SumF Step 2：push fix3_sumf onnx → trtexec 编 e_fix3sumf.engine。"""
    onnx_local = os.path.join(ROOT, "deploy", "artifacts",
                              "sparsedrive_fp16_graph_fix3_sumf.onnx")
    bc.push(cli, onnx_local, f"{BOARD}/onnx/sparsedrive_fp16_graph_fix3_sumf.onnx")
    bc.run(cli, f"rm -f {BOARD}/logs/MHA5B_DONE {BOARD}/logs/MHA5B_FAIL",
           timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
/usr/src/tensorrt/bin/trtexec --onnx={BOARD}/onnx/sparsedrive_fp16_graph_fix3_sumf.onnx --saveEngine={BOARD}/engine/e_fix3sumf.engine --fp16 --plugins=/usr/local/lib/libdfa_sd.so --timingCacheFile={BOARD}/engine/sd.cache --memPoolSize=workspace:4096 > $LOG/mha5_build.log 2>&1
rc=$?
echo "mha5build rc=$rc" >> $LOG/mha5_build.log
if [ $rc -eq 0 ] && [ -s {BOARD}/engine/e_fix3sumf.engine ]; then touch $LOG/MHA5B_DONE; else echo $rc > $LOG/MHA5B_FAIL; fi
exit 0
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/mha5build.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/mha5build.sh", f"{BOARD}/logs/mha5b_wrap.log")
    print("mha5build launched, polling (deadline 3600s) ...", flush=True)
    assert poll_done_or_fail(cli, "MHA5B", deadline_s=3600.0, poll_s=20.0), "mha5build 超时"
    rc, out = bc.run(cli, f"grep -E 'dfapg|Engine built' {BOARD}/logs/mha5_build.log | "
                          f"tail -n 5; md5sum {BOARD}/engine/e_fix3sumf.engine", timeout_s=30)
    print(out)
    print("MHA5B STAGE DONE")


def stage_mha5val(cli):
    """SumF Step 3：真数据 e2e（eager/graph）+ 逐层 profile → prof_real_sumf.json。"""
    bc.run(cli, f"rm -f {BOARD}/logs/MHA5V_DONE {BOARD}/logs/MHA5V_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
ENG={BOARD}/engine/e_fix3sumf.engine
IN={BOARD}/inputs/ref_val/val_00
test -s $ENG || {{ echo "engine missing" > $LOG/MHA5V_FAIL; exit 1; }}
/usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so $IN --warmup 10 --iters 100 > $LOG/mha5val_eager.log 2>&1
r1=$?
/usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so $IN --warmup 10 --iters 100 --graph > $LOG/mha5val_graph.log 2>&1
r2=$?
/usr/src/tensorrt/bin/trtexec --loadEngine=$ENG --plugins=/usr/local/lib/libdfa_sd.so --loadInputs=imgs:$IN/imgs.bin,projection_mat:$IN/projection_mat.bin,image_wh:$IN/image_wh.bin,status_feature:$IN/status_feature.bin --dumpProfile --exportProfile={BOARD}/prof/prof_real_sumf.json --warmUp=0 --iterations=100 --avgRuns=100 --useSpinWait > $LOG/mha5val_prof.log 2>&1
r3=$?
if [ $r1 -eq 0 ] && [ $r2 -eq 0 ] && [ $r3 -eq 0 ]; then touch $LOG/MHA5V_DONE; else echo "$r1 $r2 $r3" > $LOG/MHA5V_FAIL; fi
exit 0
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/mha5val.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/mha5val.sh", f"{BOARD}/logs/mha5v_wrap.log")
    print("mha5val launched, polling ...", flush=True)
    assert poll_done_or_fail(cli, "MHA5V", deadline_s=1800.0, poll_s=10.0), "mha5val 超时"
    rc, out = bc.run(cli, f"grep -E 'MEAN|dfapg' {BOARD}/logs/mha5val_eager.log "
                          f"{BOARD}/logs/mha5val_graph.log | head -n 8", timeout_s=30)
    print(out)
    print("MHA5V STAGE DONE")


def stage_mha5m2(cli):
    """SumF Step 4：e_fix3sumf dump 24 样本 -> outs/sumf24/<val_XX>。"""
    bc.run(cli, f"rm -f {BOARD}/logs/MHA5M_DONE {BOARD}/logs/MHA5M_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
ENG={BOARD}/engine/e_fix3sumf.engine
test -s $ENG || {{ echo "engine missing" > $LOG/MHA5M_FAIL; exit 1; }}
mkdir -p {BOARD}/outs/sumf24
n=0
for d in {BOARD}/inputs/ref_val/val_*; do
  tok=$(basename $d)
  mkdir -p {BOARD}/outs/sumf24/$tok
  /usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so \\
    $d --warmup 3 --iters 1 --dump {BOARD}/outs/sumf24/$tok >> $LOG/mha5m2.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "$tok rc=$rc" >> $LOG/mha5m2.log; echo $rc > $LOG/MHA5M_FAIL; exit $rc; fi
  n=$((n+1))
done
echo "samples=$n" >> $LOG/mha5m2.log
touch $LOG/MHA5M_DONE
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/mha5m2.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/mha5m2.sh", f"{BOARD}/logs/mha5m_wrap.log")
    print("mha5m2 launched, polling ...", flush=True)
    assert poll_done_or_fail(cli, "MHA5M", deadline_s=1200.0, poll_s=10.0), "mha5m2 超时"
    rc, out = bc.run(cli, f"tail -n 3 {BOARD}/logs/mha5m2.log; "
                          f"ls {BOARD}/outs/sumf24 | wc -l", timeout_s=30)
    print(out)
    print("MHA5M PASS")


def stage_run138sumf(cli):
    """SumF Step 5：e_fix3sumf 跑 138 场景 dump -> outs/mini138sumf/<token>。"""
    bc.run(cli, f"rm -f {BOARD}/logs/R138S_DONE {BOARD}/logs/R138S_FAIL",
           timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
ENG={BOARD}/engine/e_fix3sumf.engine
test -s $ENG || {{ echo "engine missing" > $LOG/R138S_FAIL; exit 1; }}
n=0
for d in {BOARD}/inputs/mini138/*/; do
  tok=$(basename $d)
  mkdir -p {BOARD}/outs/mini138sumf/$tok
  /usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so \\
    $d --warmup 1 --iters 1 --dump {BOARD}/outs/mini138sumf/$tok >> $LOG/run138sumf.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "$tok rc=$rc" >> $LOG/run138sumf.log; echo $rc > $LOG/R138S_FAIL; exit $rc; fi
  n=$((n+1))
done
echo "scenes=$n" >> $LOG/run138sumf.log
touch $LOG/R138S_DONE
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/run138sumf.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/run138sumf.sh", f"{BOARD}/logs/r138s_wrap.log")
    print("run138sumf launched, polling (deadline 3600s) ...", flush=True)
    assert poll_done_or_fail(cli, "R138S", deadline_s=3600.0, poll_s=15.0), "run138sumf 超时"
    rc, out = bc.run(cli, f"tail -n 3 {BOARD}/logs/run138sumf.log; "
                          f"ls {BOARD}/outs/mini138sumf | wc -l", timeout_s=30)
    print(out)
    print("RUN138SUMF PASS")


def stage_b8build(cli):
    """Backbone int8 诊断实验：push bint8 图 → trtexec --int8 --fp16 编
    e_fix3sumf_b8.engine（仅 img_backbone 33 conv 量化，neck/decoder fp16）。"""
    onnx_local = os.path.join(ROOT, "deploy", "artifacts",
                              "sparsedrive_fp16_graph_fix3_sumf_bint8.onnx")
    bc.push(cli, onnx_local, f"{BOARD}/onnx/sparsedrive_fp16_graph_fix3_sumf_bint8.onnx")
    bc.run(cli, f"rm -f {BOARD}/logs/B8B_DONE {BOARD}/logs/B8B_FAIL",
           timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
/usr/src/tensorrt/bin/trtexec --onnx={BOARD}/onnx/sparsedrive_fp16_graph_fix3_sumf_bint8.onnx --int8 --fp16 --saveEngine={BOARD}/engine/e_fix3sumf_b8.engine --plugins=/usr/local/lib/libdfa_sd.so --timingCacheFile={BOARD}/engine/sd.cache --memPoolSize=workspace:4096 > $LOG/b8_build.log 2>&1
rc=$?
echo "b8build rc=$rc" >> $LOG/b8_build.log
if [ $rc -eq 0 ] && [ -s {BOARD}/engine/e_fix3sumf_b8.engine ]; then touch $LOG/B8B_DONE; else echo $rc > $LOG/B8B_FAIL; fi
exit 0
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/b8build.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/b8build.sh", f"{BOARD}/logs/b8b_wrap.log")
    print("b8build launched, polling (deadline 3600s) ...", flush=True)
    assert poll_done_or_fail(cli, "B8B", deadline_s=3600.0, poll_s=20.0), "b8build 超时"
    rc, out = bc.run(cli, f"grep -E 'Engine built' {BOARD}/logs/b8_build.log | tail -n 2; "
                          f"md5sum {BOARD}/engine/e_fix3sumf_b8.engine", timeout_s=30)
    print(out)
    print("B8B STAGE DONE")


def stage_b8val(cli):
    """Backbone int8 诊断：真数据 e2e（eager/graph）+ 逐层 profile →
    prof_real_b8.json，与 prof_real_sumf.json 对比看 backbone conv 动没动。"""
    bc.run(cli, f"rm -f {BOARD}/logs/B8V_DONE {BOARD}/logs/B8V_FAIL", timeout_s=15)
    sh = f"""#!/bin/bash
set -u
LOG={BOARD}/logs
ENG={BOARD}/engine/e_fix3sumf_b8.engine
IN={BOARD}/inputs/ref_val/val_00
test -s $ENG || {{ echo "engine missing" > $LOG/B8V_FAIL; exit 1; }}
/usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so $IN --warmup 10 --iters 100 > $LOG/b8val_eager.log 2>&1
r1=$?
/usr/local/bin/run_engines2 $ENG /usr/local/lib/libdfa_sd.so $IN --warmup 10 --iters 100 --graph > $LOG/b8val_graph.log 2>&1
r2=$?
/usr/src/tensorrt/bin/trtexec --loadEngine=$ENG --plugins=/usr/local/lib/libdfa_sd.so --loadInputs=imgs:$IN/imgs.bin,projection_mat:$IN/projection_mat.bin,image_wh:$IN/image_wh.bin,status_feature:$IN/status_feature.bin --dumpProfile --exportProfile={BOARD}/prof/prof_real_b8.json --warmUp=0 --iterations=100 --avgRuns=100 --useSpinWait > $LOG/b8val_prof.log 2>&1
r3=$?
if [ $r1 -eq 0 ] && [ $r2 -eq 0 ] && [ $r3 -eq 0 ]; then touch $LOG/B8V_DONE; else echo "$r1 $r2 $r3" > $LOG/B8V_FAIL; fi
exit 0
"""
    bc.push_script(cli, sh, f"{BOARD}/plugin/b8val.sh")
    bc.launch(cli, f"bash {BOARD}/plugin/b8val.sh", f"{BOARD}/logs/b8v_wrap.log")
    print("b8val launched, polling ...", flush=True)
    assert poll_done_or_fail(cli, "B8V", deadline_s=1800.0, poll_s=10.0), "b8val 超时"
    rc, out = bc.run(cli, f"grep MEAN {BOARD}/logs/b8val_eager.log "
                          f"{BOARD}/logs/b8val_graph.log", timeout_s=30)
    print(out)
    print("B8V STAGE DONE")


STAGES = {
    "check": stage_check,
    "plugin": stage_plugin,
    "sh": stage_sh,
    "log": stage_log,
    "py38": stage_py38,
    "push_onnx": stage_push_onnx,
    "parse_check": stage_parse_check,
    "build_engine": stage_build_engine,
    "baseline": stage_baseline,
    "profile": stage_profile,
    "dump24": stage_dump24,
    "push138": stage_push138,
    "run138": stage_run138,
    "fetch138": stage_fetch138,
    "fetch": stage_fetch,
    "pgbench": stage_pgbench,
    "pgbuild": stage_pgbuild,
    "pgval": stage_pgval,
    "pgm2": stage_pgm2,
    "run138pg": stage_run138pg,
    "pgreal": stage_pgreal,
    "pg4": stage_pg4,
    "mha": stage_mha,
    "mhabuild": stage_mhabuild,
    "mhaval": stage_mhaval,
    "mham2": stage_mham2,
    "run138mha": stage_run138mha,
    "mha5": stage_mha5,
    "mha5build": stage_mha5build,
    "mha5val": stage_mha5val,
    "mha5m2": stage_mha5m2,
    "run138sumf": stage_run138sumf,
    "b8build": stage_b8build,
    "b8val": stage_b8val,
    "qdqbuild_e1": _QDQ_E1["build"],
    "qdqval_e1": _QDQ_E1["val"],
    "qdqm2_e1": _QDQ_E1["m2"],
    "qdqbuild_e2": _QDQ_E2["build"],
    "qdqval_e2": _QDQ_E2["val"],
    "qdqm2_e2": _QDQ_E2["m2"],
    "qdqbuild_e1n": _QDQ_E1N["build"],
    "qdqval_e1n": _QDQ_E1N["val"],
    "qdqbuild_e1f": _QDQ_E1F["build"],
    "qdqval_e1f": _QDQ_E1F["val"],
}


def main():
    assert len(sys.argv) >= 2, "usage: python deploy/board/board_v2.py <stage>"
    stage = sys.argv[1]
    fn = STAGES.get(stage)
    assert fn, f"unknown stage: {stage}; have: {sorted(STAGES)}"
    cli = bc.connect()
    try:
        fn(cli)
    finally:
        cli.close()


if __name__ == "__main__":
    main()
