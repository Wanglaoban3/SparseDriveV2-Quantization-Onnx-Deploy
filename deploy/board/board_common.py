# -*- coding: utf-8 -*-
"""SparseDriveV2 板端驱动基座（paramiko）。

模板承自 v1 项目 board_m1.py 的实测坑集：
- 后台启动必须"括号孤儿 + 绝对路径"：`(setsid nohup CMD > log 2>&1 < /dev/null &); echo GO`，
  否则 wrapper bash 卡 do_wait、paramiko 通道永不 EOF；
- 完成 marker 由远端命令自身成功条件化 touch（`&& touch MARKER`），启动脚本里
  cd 之后 touch 会落错目录，marker 一律绝对路径；
- sftp 不自动建远端目录，push 前先 `mkdir -p`；
- 宿主机文本文件若带 CRLF，落板后 bash 会把 \r 当路径一部分静默死，push 前规范化 LF。

凭据纪律：BOARD_HOST / BOARD_PASS（可 BOARD_USER，默认 root）只在运行时环境变量，
禁止写入任何文件。
"""
import fnmatch
import os
import posixpath
import stat
import time

BOARD_WORK = "/opt/m0/sd2"

_TEXT_EXTS = {".sh", ".py", ".txt", ".tsv", ".json", ".md", ".cu", ".cpp",
              ".cc", ".h", ".hpp", ".yaml", ".yml", ".cfg", ".log"}


def sanitize_name(name: str) -> str:
    """引擎图张量名带前导 '/'（v1 实坑：落盘被当路径分隔），统一剥掉。"""
    return name.lstrip("/")


def is_text_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _TEXT_EXTS


def to_lf(data: bytes, is_text: bool) -> bytes:
    """文本字节 CRLF/CR → LF；二进制原样返回（一个字节不动）。"""
    if not is_text:
        return data
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def launch_cmd(cmd: str, log_path: str, marker_path: str = "") -> str:
    """括号孤儿启动命令串。marker_path 由调用方命令自行 `&& touch`（成功条件化），
    这里不代 touch——只有调用方知道什么算成功。"""
    return f"(setsid nohup {cmd} > {log_path} 2>&1 < /dev/null &); echo GO"


def connect():
    import paramiko
    host = os.environ.get("BOARD_HOST")
    passwd = os.environ.get("BOARD_PASS")
    if not host or not passwd:
        raise RuntimeError("BOARD_HOST / BOARD_PASS 环境变量未注入（凭据纪律：仅运行时注入）")
    user = os.environ.get("BOARD_USER", "root")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(hostname=host, port=22, username=user, password=passwd,
                timeout=20, banner_timeout=30, auth_timeout=30)
    return cli


def run(client, cmd: str, timeout_s: int = 120):
    """同步短命令。返回 (rc, 合并了 stderr 的输出)。"""
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout_s)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    if err:
        out = out + err if not out else out + "\n" + err
    return rc, out


def launch(client, cmd: str, log_path: str, marker_path: str = ""):
    """后台启动，立即返回（v1 坑：不能等通道 EOF）。"""
    rc, out = run(client, launch_cmd(cmd, log_path, marker_path), timeout_s=15)
    if "GO" not in out:
        raise RuntimeError(f"launch 未收到 GO（out={out!r}）")


def poll_marker(client, marker_path: str, deadline_s: float, poll_s: float = 5.0) -> bool:
    """轮询绝对路径 marker 文件直到出现或超时。"""
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        rc, out = run(client, f"test -f {marker_path} && echo Y || echo N", timeout_s=15)
        if "Y" in out.splitlines()[:1]:
            return True
        time.sleep(poll_s)
    return False


def push(client, local: str, remote: str):
    """push 单文件；先 mkdir -p 远端目录（sftp 不自动建）；文本规范化 LF。"""
    rdir = posixpath.dirname(remote.replace("\\", "/"))
    if rdir:
        run(client, f'mkdir -p "{rdir}"', timeout_s=30)
    with open(local, "rb") as f:
        data = f.read()
    data = to_lf(data, is_text_path(local))
    sftp = client.open_sftp()
    try:
        with sftp.open(remote, "wb") as f:
            f.write(data)
    finally:
        sftp.close()


def push_script(client, text: str, remote: str):
    """把脚本文本直接推到板上（LF 规范化），不经过本地临时文件。
    复合编排（cd/&&/;）一律落脚本后 `bash x.sh` 启动——nohup 无法执行 shell 内建
    （`nohup cd` 报 failed to run command 'cd'），重定向也只绑最后一个命令列表。"""
    rdir = posixpath.dirname(remote.replace("\\", "/"))
    if rdir:
        run(client, f'mkdir -p "{rdir}"', timeout_s=30)
    data = to_lf(text.encode("utf-8"), True)
    sftp = client.open_sftp()
    try:
        with sftp.open(remote, "wb") as f:
            f.write(data)
    finally:
        sftp.close()


def pull(client, remote_dir: str, local_dir: str, pattern: str = "*"):
    """批量拉取 remote_dir 下匹配 pattern 的文件到 local_dir，文件名 sanitize。
    递归：子目录整体镜像到 local_dir 下（2026-10-05 dump24 24 子目录实测需要）。"""
    os.makedirs(local_dir, exist_ok=True)
    sftp = client.open_sftp()
    n = 0
    try:
        for entry in sftp.listdir_attr(remote_dir):
            rpath = posixpath.join(remote_dir, entry.filename)
            lpath = os.path.join(local_dir, sanitize_name(entry.filename))
            if stat.S_ISDIR(entry.st_mode):
                n += pull(client, rpath, lpath, pattern)
                continue
            if not fnmatch.fnmatch(entry.filename, pattern):
                continue
            sftp.get(rpath, lpath)
            n += 1
    finally:
        sftp.close()
    return n
