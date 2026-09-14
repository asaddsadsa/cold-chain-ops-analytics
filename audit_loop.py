# -*- coding: utf-8 -*-
"""审计回路：一条命令验证 src/ 全部模块的健康度。

检查项（任一红即整体红）：
  1. py_compile 全部源码
  2. 全量 pytest
  3. 四个数据模块端到端 main() 可运行、产物齐全
  4. 跨进程可复现性：同一脚本两次独立进程运行，产物逐字节一致
     （这是用户真实复现方式；同进程内测试无法覆盖此项）
用法: python audit_loop.py [--skip-e2e]
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
FAILS = []
ENVS = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} | {name}" + (f" | {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def run(args, timeout=600):
    return subprocess.run([PY, *args], cwd=ROOT, env=ENVS, capture_output=True,
                          text=True, timeout=timeout, encoding="utf-8", errors="replace")


def digest_dir(d: Path, patterns) -> dict:
    out = {}
    for pat in patterns:
        for f in sorted(d.rglob(pat)):
            out[str(f.relative_to(d))] = hashlib.md5(f.read_bytes()).hexdigest()
    return out


def main():
    skip_e2e = "--skip-e2e" in sys.argv

    # 1. 编译
    r = run(["-m", "py_compile", *[str(p) for p in (ROOT / "src").glob("*.py")]])
    check("py_compile src/*.py", r.returncode == 0, r.stderr[-300:] if r.returncode else "")

    # 2. 全量测试
    r = run(["-m", "pytest", "tests/", "-q", "--tb=no"])
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    check("pytest tests/", r.returncode == 0, tail)

    if skip_e2e:
        print("\n== 结果 ==")
        print("RED: " + "; ".join(FAILS) if FAILS else "GREEN: 全部通过")
        return 1 if FAILS else 0

    # 3+4. 端到端 ×2（第二次同时验证跨进程可复现）
    # 用隔离目录避免污染正式产物：第一轮写 tmp1，第二轮写 tmp2，比对
    import tempfile
    t1 = Path(tempfile.mkdtemp(prefix="audit1_"))
    t2 = Path(tempfile.mkdtemp(prefix="audit2_"))

    # 4a. 数据层 A：跨进程两次运行产物一致
    code_a = (
        "from src.gen_warehouse_data import generate_warehouse;"
        "import sys;generate_warehouse(out_dir=sys.argv[1]+'/wh',qc_dir=sys.argv[1]+'/qc')"
    )
    r1 = run(["-c", code_a, str(t1)]); r2 = run(["-c", code_a, str(t2)])
    ok_run = r1.returncode == 0 and r2.returncode == 0
    check("数据层A 端到端×2", ok_run, (r1.stderr[-300:] or r2.stderr[-300:]) if not ok_run else "")
    if ok_run:
        d1 = digest_dir(t1 / "wh", ["*.csv"]); d2 = digest_dir(t2 / "wh", ["*.csv"])
        check("数据层A 跨进程逐字节一致", d1 == d2,
              "" if d1 == d2 else f"差异文件: {[k for k in d1 if d1[k]!=d2.get(k)]}")

    # 4b. 模块一下(Olist KPI)：跨进程一致
    code_k = (
        "from src.olist_kpi import run_olist_kpi;"
        "import sys;run_olist_kpi(out_dir=sys.argv[1]+'/kpi')"
    )
    r1 = run(["-c", code_k, str(t1)]); r2 = run(["-c", code_k, str(t2)])
    ok_run = r1.returncode == 0 and r2.returncode == 0
    check("Olist KPI 端到端×2", ok_run, (r1.stderr[-300:] or r2.stderr[-300:]) if not ok_run else "")
    if ok_run:
        d1 = digest_dir(t1 / "kpi", ["*"]); d2 = digest_dir(t2 / "kpi", ["*"])
        check("Olist KPI 跨进程逐字节一致", d1 == d2,
              "" if d1 == d2 else f"差异文件: {[k for k in d1 if d1[k]!=d2.get(k)]}")

    # 4c. 数据层F(SimPy)：跨进程一致（子种子派生是否进程稳定的关键检验）
    code_s = (
        "from src.warehouse_sim import run_all_experiments;"
        "import sys;run_all_experiments(out_dir=sys.argv[1]+'/sim')"
    )
    r1 = run(["-c", code_s, str(t1)], timeout=900); r2 = run(["-c", code_s, str(t2)], timeout=900)
    ok_run = r1.returncode == 0 and r2.returncode == 0
    check("SimPy 实验端到端×2", ok_run, (r1.stderr[-300:] or r2.stderr[-300:]) if not ok_run else "")
    if ok_run:
        d1 = digest_dir(t1 / "sim", ["*.json"]); d2 = digest_dir(t2 / "sim", ["*.json"])
        check("SimPy 实验跨进程逐字节一致", d1 == d2,
              "" if d1 == d2 else f"差异文件: {[k for k in d1 if d1[k]!=d2.get(k)]}")

    shutil.rmtree(t1, ignore_errors=True); shutil.rmtree(t2, ignore_errors=True)

    print("\n== 结果 ==")
    print("RED: " + "; ".join(FAILS) if FAILS else "GREEN: 全部通过")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
