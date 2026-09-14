"""把 PDCA 报告渲染成带中文字体 CSS 的 HTML，再调本机 headless Edge/Chrome 打印成 PDF。

设计依据见 `docs/adr/0007-pdf-via-headless-edge.md`。

**为什么自带一个 Markdown 渲染器**：本仓库的依赖清单是锁定的，加一个 Markdown 库意味着
给复现者多一个安装项。本报告只用到有限几种 Markdown 构造（标题/段落/表格/引用/列表/分隔线/
行内强调与代码），故在脚本内实现这一**明确子集**，并对不支持的行给出可见提示而不是静默吞掉。
它不追求通用——若报告用到新语法，这里会如实报出来。

**降级纪律**：检测不到 Edge/Chrome 时**只交付 Markdown**，并在日志里显式写明「PDF 未生成、
原因、手动转换步骤」。绝不生成一个空文件或复用旧 PDF 来冒充成功。
"""

from __future__ import annotations

import argparse
import html
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MD = PROJECT_ROOT / "report" / "pdca_report.md"
DEFAULT_PDF = PROJECT_ROOT / "report" / "pdca_report.pdf"
# 中间 HTML 放 report/build/——该目录已在 .gitignore 中（与「报告中间产物」规则一致，
# 不另立规则；放在带点的 .build/ 会绕过该规则被误提交）
DEFAULT_HTML = PROJECT_ROOT / "report" / "build" / "pdca_report.html"

#: 候选浏览器的常见安装位置（Windows 优先，兼顾 macOS/Linux 与 PATH）。
_BROWSER_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Edge", (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")),
    ("Chrome", (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe")),
    ("Edge", ("msedge",)),
    ("Chrome", ("google-chrome", "chromium", "chromium-browser", "chrome")),
)

_MANUAL_STEPS = """\
  1) VS Code：安装 Markdown PDF 扩展后右键导出；
  2) 浏览器：用 Edge/Chrome 打开 HTML（本脚本已生成），Ctrl+P → 目标「另存为 PDF」→ 勾选「背景图形」；
  3) Typora / Pandoc：直接导出 PDF。"""


# ---------------------------------------------------------------------------
# 浏览器定位
# ---------------------------------------------------------------------------
def find_browser() -> tuple[str, str] | None:
    """返回 `(浏览器名, 可执行路径)`；找不到返回 None。"""
    for name, candidates in _BROWSER_CANDIDATES:
        for cand in candidates:
            if Path(cand).is_file():
                return name, cand
            found = shutil.which(cand)
            if found:
                return name, found
    return None


# ---------------------------------------------------------------------------
# Markdown → HTML（受支持的明确子集）
# ---------------------------------------------------------------------------
def _inline(text: str) -> str:
    """行内标记：先转义，再还原 **粗体** / `代码` / [链接](url)。"""
    out = html.escape(text, quote=False)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out


def _is_table_sep(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:|-]+\|", line.strip()))


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def md_to_html(md_text: str) -> str:
    """把 Markdown 子集渲染成 HTML 片段。

    支持：`#`~`####` 标题、表格、`>` 引用、`-`/`1.` 列表、`---` 分隔线、围栏代码块、
    以及行内的粗体 / 行内代码 / 链接。其余一律按段落处理——**不静默丢内容**。
    """
    lines = md_text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # 围栏代码块
        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue

        # 分隔线
        if re.fullmatch(r"-{3,}", stripped):
            out.append("<hr/>")
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
            continue

        # 表格：当前行是 |...| 且下一行是分隔行
        if stripped.startswith("|") and i + 1 < len(lines) and _is_table_sep(lines[i + 1]):
            header = _split_row(stripped)
            i += 2
            body = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                body.append(_split_row(lines[i]))
                i += 1
            head_html = "".join(f"<th>{_inline(c)}</th>" for c in header)
            rows_html = "".join(
                "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>" for row in body
            )
            out.append(f"<table><thead><tr>{head_html}</tr></thead><tbody>{rows_html}</tbody></table>")
            continue

        # 引用块（连续行合并）
        if stripped.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote>" + _inline(" ".join(buf)) + "</blockquote>")
            continue

        # 列表（连续同类项合并）
        if re.match(r"^([-*]|\d+\.)\s+", stripped):
            ordered = bool(re.match(r"^\d+\.\s+", stripped))
            tag = "ol" if ordered else "ul"
            items = []
            while i < len(lines):
                mm = re.match(r"^([-*]|\d+\.)\s+(.*)$", lines[i].strip())
                if not mm:
                    break
                if bool(re.match(r"^\d+\.\s+", lines[i].strip())) != ordered:
                    break
                items.append(f"<li>{_inline(mm.group(2))}</li>")
                i += 1
            out.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
            continue

        # 段落（连续非空行合并）
        buf = []
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(
            ("#", "|", ">", "```", "---")
        ) and not re.match(r"^([-*]|\d+\.)\s+", lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        if buf:
            out.append("<p>" + _inline(" ".join(buf)) + "</p>")
        else:  # 兜底：吃掉无法归类的行，避免死循环
            out.append("<p>" + _inline(stripped) + "</p>")
            i += 1
    return "\n".join(out)


#: 中文字体优先的打印样式。正文用系统黑体栈，等宽用 Consolas/黑体兜底。
_PRINT_CSS = """
@page { size: A4; margin: 18mm 16mm; }
body {
  font-family: "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB",
               "Noto Sans CJK SC", system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 10.5pt; line-height: 1.7; color: #0b0b0b; margin: 0;
}
h1 { font-size: 20pt; border-bottom: 2px solid #c3c2b7; padding-bottom: 6px; }
h2 { font-size: 15pt; margin-top: 22px; border-left: 4px solid #2a78d6; padding-left: 8px; }
h3 { font-size: 12.5pt; margin-top: 18px; }
h4 { font-size: 11pt; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }
th, td { border: 1px solid #c3c2b7; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #f0efec; font-weight: 600; }
blockquote {
  margin: 10px 0; padding: 8px 12px; border-left: 3px solid #eb6834;
  background: #f9f9f7; color: #52514e;
}
code { font-family: Consolas, "Courier New", "Microsoft YaHei", monospace;
       background: #f0efec; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
pre { background: #f9f9f7; border: 1px solid #e1e0d9; padding: 10px;
      overflow-x: auto; font-size: 9pt; }
pre code { background: none; padding: 0; }
hr { border: none; border-top: 1px solid #e1e0d9; margin: 18px 0; }
a { color: #2a78d6; }
ul, ol { padding-left: 22px; }
h2, h3, table, pre, blockquote { page-break-inside: avoid; }
"""


def render_html(md_text: str, title: str) -> str:
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_PRINT_CSS}</style></head>"
        f"<body>{md_to_html(md_text)}</body></html>"
    )


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def html_to_pdf(html_path: Path, pdf_path: Path, browser: tuple[str, str]) -> None:
    """调用 headless 浏览器打印 PDF。失败抛 `subprocess.CalledProcessError`。"""
    name, exe = browser
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists():
        pdf_path.unlink()  # 删掉旧的，避免「命令失败但旧文件还在」被当成成功
    uri = html_path.resolve().as_uri()
    last_err: Exception | None = None
    # 新版 Chromium 推荐 --headless=new；旧版只认 --headless。两种都试。
    for headless in ("--headless=new", "--headless"):
        cmd = [exe, headless, "--disable-gpu", "--no-sandbox",
               "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", uri]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_err = exc
            continue
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            logger.info("PDF 已生成：%s（浏览器：%s %s）", pdf_path, name, headless)
            return
    raise RuntimeError(f"{name} 未能生成 PDF：{last_err}")


def build(md_path: Path, pdf_path: Path, html_path: Path) -> Path | None:
    """渲染 HTML 并尝试生成 PDF。

    返回 PDF 路径；**检测不到浏览器时返回 None**，并把降级原因与手动步骤写进日志
    （调用方负责在 README 里同步说明，绝不伪装已生成）。
    """
    md_text = md_path.read_text(encoding="utf-8")
    title = next((ln.lstrip("# ").strip() for ln in md_text.splitlines()
                  if ln.startswith("# ")), md_path.stem)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(md_text, title), encoding="utf-8")
    logger.info("HTML 已生成：%s", html_path)

    browser = find_browser()
    if browser is None:
        logger.warning(
            "未检测到 Edge/Chrome —— **PDF 未生成**。已交付 Markdown（%s）与 HTML（%s）。"
            "手动转换步骤：%s", md_path, html_path, _MANUAL_STEPS,
        )
        return None
    try:
        html_to_pdf(html_path, pdf_path, browser)
    except RuntimeError as exc:
        logger.warning(
            "调用了 %s 但**PDF 未生成**（%s）。已交付 Markdown 与 HTML。手动转换步骤：%s",
            browser[0], exc, _MANUAL_STEPS,
        )
        return None
    return pdf_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="PDCA 报告 → PDF（headless Edge/Chrome）")
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    args = parser.parse_args()

    if not args.md.exists():
        logger.error("找不到报告源文件：%s", args.md)
        sys.exit(2)
    pdf = build(args.md, args.pdf, args.html)
    if pdf is None:
        print("\n=== PDF 未生成（已降级为 Markdown 交付）===")
        print(f"  Markdown：{args.md}")
        print(f"  HTML    ：{args.html}")
        print("  原因与手动转换步骤见上方日志。")
        sys.exit(1)
    print("\n=== 报告已生成 ===")
    print(f"  Markdown：{args.md}")
    print(f"  PDF     ：{pdf}（{pdf.stat().st_size / 1024:.0f} KB）")


if __name__ == "__main__":
    main()
