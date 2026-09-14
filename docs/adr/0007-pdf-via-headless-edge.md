# 报告 PDF 用 headless Edge 打印生成，Markdown 为兜底

需求要求 PDCA 报告交付 Markdown 与 PDF 两版，但本机实测 weasyprint 无法运行（Windows 缺 GTK 运行库，报 `cannot load library 'libgobject-2.0-0'`），pandoc/typst/LaTeX 均未安装。本机中文字体齐全（微软雅黑/黑体/宋体/楷体）且装有 Edge 与 Chrome。

我们决定：`report/build_pdf.py` 把 Markdown 渲染为带中文字体 CSS 的 HTML，调用本机 `msedge --headless --print-to-pdf` 生成 PDF——**已实测转通**（含中文表格的 A4 样例正常输出）。脚本检测不到 Edge/Chrome 时自动降级：只交付 Markdown，并在 README 与运行日志显式写明「PDF 未生成、原因、手动转换步骤（VS Code / 浏览器打印 / Typora）」，绝不伪装已生成。

## Considered Options

- weasyprint：本机实测跑不通；给复现者强加 Windows GTK 安装步骤，违背一键复现。
- pandoc/typst：需安装非 Python 工具链，本机均无。
- 只交付 Markdown：合规但少一个交付物，在 headless Edge 已验证可行后没必要退到这步。

## Consequences

- PDF 生成依赖复现者本机有 Edge 或 Chrome（Windows 默认自带 Edge），README 注明。
- 生成命令与字体 CSS 固化在 build_pdf.py 中，保证不同机器产出的 PDF 样式一致。
