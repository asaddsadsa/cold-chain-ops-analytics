"""报告引用的数字必须与产物一致——这份对账此前不存在。

ADR-0014 记录过一次真实的分叉（0.1 km / 0.06 元，重构造成的），并写下了「同一口径两处
各算一遍正是数字对不上的经典来源」的教训。但那条教训没有被施加到报告层：`pdca_report.md`
里的数字是**手写的字面量**，改善页里的数字是从产物拼出来的，两者之间没有任何东西比对。
谁改了产物、忘了改报告，或者反过来，都不会有测试变红——而报告是要交给招聘方逐项核对的。

本文件就是那份比对：把报告里每一条承重的数字，拿产物重算一遍，断言**报告文本里确实出现
了这个字面量**。两边任一侧漂移都会红。

这不是「报告格式测试」：断言的是数值，改排版不会让它红，改数字才会。
"""

from __future__ import annotations

import re

import pytest

from report import build_pdf
from src import config as C
from src.dashboard import data as D
from src.dashboard import theme


def _report() -> str:
    return (C.PROJECT_ROOT / "report" / "pdca_report.md").read_text(encoding="utf-8")


class TestReportReconcilesWithArtifacts:
    """逐条对账：报告里承重的数字 == 产物里的值。"""

    def test_slotting_walk_cost(self):
        """P-3 / C-1：行走成本与每行行走距离（库位重排的两个卖点数字）。"""
        s = D.warehouse_kpi().slotting
        text = _report()
        assert f"{s.walk_cost_before:,.0f}" in text, "重排前行走成本"
        assert f"{s.walk_cost_after:,.0f}" in text, "重排后行走成本"
        assert f"{s.reduction_pct:.2f}" in text, "行走成本降幅"
        assert f"{s.walk_m_per_line_before:.2f}" in text, "每行行走距离（前）"
        assert f"{s.walk_m_per_line_after:.2f}" in text, "每行行走距离（后）"

    def test_abc_class_a(self):
        """P-4：A 类 SKU 个数与它承担的出库行数占比（ADR-0002 的口径卖点）。"""
        a = D.warehouse_kpi().abc_by_class["A"]
        text = _report()
        assert f"{a.n_sku} 个" in text, "A 类 SKU 数"
        assert f"{a.line_share:.1%}" in text, "A 类承担的出库行数占比"
        assert f"{a.sku_share:.1%}" in text, "A 类占全部 SKU 的比例"

    def test_pick_slowdown_embedding(self):
        """P-1：14–16 点低谷的三个数字。

        这三个数取自**数据层 KPI**（`kpi_overall.json`），报告与运输页引用的都是它。
        数据层 A 的质检摘要另有一份独立推导，两者现已**逐字段相等**
        （`test_warehouse_kpi.py::TestGeneratorQcVersusKpiReDerivation` 守着这一点）。
        """
        slow = D.warehouse_kpi().pick_slowdown
        text = _report()
        assert f"{slow.sec_per_line_14_16:.1f}" in text
        assert f"{slow.sec_per_line_other:.1f}" in text
        assert f"{slow.ratio:.2f}" in text

    def test_p03_discrepancy_embedding(self):
        """P-2：P03 盘差埋点的三个数字。"""
        p03 = D.warehouse_kpi().p03_discrepancy
        text = _report()
        assert f"{p03.rate:.2%}" in text
        assert f"{p03.other_rate:.2%}" in text
        assert f"{p03.ratio:.2f}" in text

    def test_inventory_accuracy(self):
        """摘要：库存准确率（金额加权口径）。"""
        assert f"{D.warehouse_kpi().inventory_accuracy_rate:.2%}" in _report()

    def test_representative_day_and_route_kpis(self):
        """P-4 / C-2：代表日与优化前后里程、用车数、柴油单均成本。"""
        k = D.transport_kpi()
        text = _report()
        assert k.representative_day in text
        assert f"{k.baseline.total_distance_km:,.2f}" in text
        assert f"{k.optimized.total_distance_km:,.2f}" in text
        assert f"{k.baseline.n_vehicles} 台" in text
        assert f"{k.optimized.n_vehicles} 台" in text
        assert f"{k.baseline.diesel_per_order:.2f} 元" in text
        assert f"{k.optimized.diesel_per_order:.2f} 元" in text

    def test_diesel_total_cost_before_and_after(self):
        """C-2 的成本行：日总成本（报告比 KPI 卡片多引了总额这一列）。"""
        raw = D.artifact("transport_kpi")
        text = _report()
        for which in ("baseline", "optimized"):
            total = raw[which]["cost"]["diesel"]["total"]
            assert f"{total:,.2f}" in text, f"{which} 柴油日总成本"

    def test_breakeven_km(self):
        """A-1 的季度复核触发线：单车日均里程跌破盈亏平衡点。"""
        tco = D.artifact("transport_tco")
        assert f"{tco['breakeven_km']['km']:.1f} km" in _report()


class TestReportDoesNotInheritUnbackedClaims:
    """报告里的类别与口径必须与产物同源，不能自己发明一个数。"""

    def test_every_pdca_section_cites_an_artifact(self):
        """P 与 C 的每一条都要给出证据文件——报告是「可逐项溯源」的作品集承诺。"""
        text = _report()
        for section in ("### P-1", "### P-2", "### P-3", "### P-4", "### P-5",
                        "### C-1", "### C-2", "### C-3", "### C-4"):
            assert section in text, section
        # 至少出现这些产物文件名，否则「证据文件」列是空的
        for name in ("kpi_overall.json", "transport_kpi.json"):
            assert name in text, name


class TestPrintCssComesFromTheme:
    """报告层不得自带一份色板。

    `theme.py` 自称「单一定义」（改色值前必须先重跑校验），而 `build_pdf.py` 原先手写了
    9 个十六进制色值加一个字体栈，逐个都能在 theme 里找到同一个值——改色时没有任何机制能
    拦住两边漂移。报告是交给招聘方看的，和看板必须是同一套视觉语言。
    """

    def test_print_css_uses_only_theme_colours(self):
        allowed = {
            theme.CHROME["light"]["axis"], theme.CHROME["light"]["primary_ink"],
            theme.CHROME["light"]["page"], theme.CHROME["light"]["grid"],
            theme.CHROME["light"]["secondary_ink"], theme.DIVERGING["mid"],
            theme.series(0), theme.series(1),
        }
        used = set(re.findall(r"#[0-9a-fA-F]{6}", build_pdf.print_css()))
        assert used <= allowed, f"打印样式里出现了主题色板之外的色值：{sorted(used - allowed)}"

    def test_print_font_stack_is_declared_in_theme(self):
        assert theme.FONT_STACK_PRINT in build_pdf.print_css()

    def test_no_unsubstituted_placeholder(self):
        """`Template.substitute` 缺键会抛，但**漏写 `$`** 的静态文本不会被发现——
        那样 CSS 里会留下一个没被替换的词，而页面照样能打开。"""
        assert "$" not in build_pdf.print_css()
