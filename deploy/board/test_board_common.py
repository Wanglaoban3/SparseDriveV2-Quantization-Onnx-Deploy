# -*- coding: utf-8 -*-
"""board_common 单元测试（standalone assert 脚本，退出码 0 即全过）。

计划 Task 0 Step 1：先于实现编写并运行，必须先失败（模块不存在/函数缺失）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_sanitize_name_strips_leading_slash():
    from deploy.board.board_common import sanitize_name
    # v1 实坑：引擎图张量名带前导 '/'，sftp 拉回落盘会被当路径分隔
    assert sanitize_name("/Reshape_9_output_0") == "Reshape_9_output_0"
    assert sanitize_name("trajectory") == "trajectory"
    assert sanitize_name("/") == ""
    assert sanitize_name("") == ""


def test_to_lf_normalizes_text_bytes():
    from deploy.board.board_common import to_lf
    # Review Focus 4：Windows CRLF 落板会静默死，push 前文本必须规范化为 LF
    assert to_lf(b"a\r\nb\r\nc", is_text=True) == b"a\nb\nc"
    assert to_lf(b"a\rb", is_text=True) == b"a\nb"
    # 二进制内容（.bin/.onnx/.engine/.so）一个字节都不能动
    blob = b"\x00\x01\r\n\xff"
    assert to_lf(blob, is_text=False) == blob


def test_is_text_path_ext():
    from deploy.board.board_common import is_text_path
    assert is_text_path("/opt/m0/sd2/run.sh")
    assert is_text_path("prep_engine_inputs.py")
    assert is_text_path("manifest.tsv")
    assert not is_text_path("imgs.bin")
    assert not is_text_path("sparsedrive_int8_qdq_folded.onnx")
    assert not is_text_path("e_sd2.engine")


def test_launch_cmd_orphan_paren_form():
    from deploy.board.board_common import launch_cmd
    cmd = launch_cmd("bash /opt/m0/sd2/hello.sh",
                     "/opt/m0/sd2/logs/hello.log",
                     "/opt/m0/sd2/logs/HW_DONE")
    # v1 M1 实坑：括号孤儿 + 绝对路径，否则 wrapper bash 卡 do_wait 通道不 EOF
    assert cmd.startswith("(setsid nohup ")
    assert "&); echo GO" in cmd
    assert "> /opt/m0/sd2/logs/hello.log 2>&1 < /dev/null" in cmd


def test_poll_and_run_shapes():
    import inspect
    from deploy.board import board_common as bc
    # 签名契约：后续任务按这些名字调用
    assert callable(bc.connect)
    sig = inspect.signature(bc.run)
    assert list(sig.parameters) == ["client", "cmd", "timeout_s"]
    sig = inspect.signature(bc.poll_marker)
    assert list(sig.parameters)[:3] == ["client", "marker_path", "deadline_s"]
    assert bc.BOARD_WORK == "/opt/m0/sd2"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", fn.__name__, "->", e)
        except Exception as e:  # ImportError 等
            failed += 1
            print("ERROR", fn.__name__, "->", type(e).__name__, e)
    print(f"{len(fns) - failed}/{len(fns)} pass")
    sys.exit(1 if failed else 0)
