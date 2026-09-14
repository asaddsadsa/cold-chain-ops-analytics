# 去掉 Excel 校验工作簿（openpyxl）

需求 3.x 要求「openpyxl 生成校验工作簿，对库存准确率 / 平均履约时效 / 真实准时交付率三指标做『程序结果 vs Excel 公式复算』对照」。08 号票已按此实现（真实 Excel 公式 + 回读单元格求值 + 反例测试），但**用户明确表示不需要任何与 Excel 相关的产物**。

我们决定：**移除 Excel 校验工作簿及其全部配套代码与文档**——删除 `warehouse_kpi.py` 的 `build_excel_check` / `verify_excel_check` / `eval_excel_formula`、config 的 `WAREHOUSE_EXCEL_CHECK`、requirements.txt 的 `openpyxl`，并同步更新 08 / 09 号票、台账、实现计划与需求文档。KPI 的**口径回归改由单元测试承担**：每个 KPI 至少一个手工构造的小输入 + 手算期望值断言（这是 spec「Testing Decisions」里本来就有的主缝，Excel 校验只是它的一个附加对照面）。

## Considered Options

- 保留工作簿但不写入产物目录：用户要的是「不使用 Excel 相关的东西」，留一堆死代码与死依赖仍然违背意图。
- 用 CSV 版「对照表」替代：KPI 的对照本质是「程序结果 vs 独立复算」，而独立复算在单测里已经存在且能自动失败，多一份 CSV 对照表只是重复。

## Consequences

- `data/processed/warehouse_kpi/` 不再产出 `.xlsx`；报告去掉「五、Excel 校验」章节。
- `requirements.txt` 不再含 openpyxl；ADR-0006 的「全栈可装」包列表同步更新。
- 需求 3.x 的 Excel 技能展示项整体取消；08 号票该项在验收清单中标记为「按用户要求去掉」而非「已完成」，不伪装成已交付。
- 09 号票（Olist KPI）原本要「供 08 的 Excel 校验补对照项」的遗留项一并作废——模块一自此不依赖数据层 B 的产物。
