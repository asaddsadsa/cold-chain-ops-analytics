"""`charts.py` 的构造器接口测试。

这一层此前零测试。补的第一个缝是 `dual_line_chart` 的**槽位参数**——它原先把颜色按
「序列在字典里的次序」从槽 0 起分配，于是页面想给某个实体固定一个槽位时，只能事后逐条
重写 trace 颜色（`pages/2_运输分析.py` 的 `_multi_line` 就是这么干的，那是把构造器的漏洞
补在调用方）。同一个页面画多张图时，不传槽位就意味着「第 1 条序列」在这张图是柴油、
在那张图是别的实体，同一个蓝有了两种读法。
"""

from __future__ import annotations

import pytest

from src.dashboard import charts, theme


def _colors(fig) -> list[str]:
    return [trace.line.color for trace in fig.data]


def _by_name(fig) -> dict[str, str]:
    """实体 → 颜色。比 `_colors` 更贴近要守的性质：颜色是**身份的属性**，不是次序的属性。"""
    return {trace.name: trace.line.color for trace in fig.data}


class TestDualLineChartSlots:
    SERIES = {"柴油自购": [1, 2, 3], "纯电租赁": [3, 2, 1]}

    def test_without_slots_colours_follow_dict_order(self):
        fig = charts.dual_line_chart([1, 2, 3], self.SERIES, y_title="元")
        assert _colors(fig) == [theme.series(0), theme.series(1)]

    def test_slots_pin_each_entity_to_its_own_colour(self):
        fig = charts.dual_line_chart([1, 2, 3], self.SERIES, y_title="元",
                                     slots={"柴油自购": 2, "纯电租赁": 3})
        assert _colors(fig) == [theme.series(2), theme.series(3)]

    def test_reordering_the_dict_does_not_recolour_a_pinned_entity(self):
        """这正是槽位要解决的问题：实体的颜色不该随画图次序漂移。

        比的是「实体 → 颜色」的映射，不是颜色列表——重排字典会改变 traces 的先后，
        但那不该改变任何一个实体拿到哪个槽位。
        """
        slots = {"柴油自购": 0, "纯电租赁": 1}
        forward = charts.dual_line_chart([1, 2], self.SERIES, y_title="元", slots=slots)
        reversed_ = charts.dual_line_chart(
            [1, 2], dict(reversed(list(self.SERIES.items()))), y_title="元", slots=slots)
        assert _by_name(forward) == _by_name(reversed_) == {
            "柴油自购": theme.series(0), "纯电租赁": theme.series(1)}

    def test_marker_matches_line_colour(self):
        """线色与点色必须同源，否则图例与数据点会对不上。"""
        fig = charts.dual_line_chart([1, 2, 3], self.SERIES, y_title="元",
                                     slots={"柴油自购": 5, "纯电租赁": 6})
        assert [t.marker.color for t in fig.data] == _colors(fig)

    def test_unknown_slot_key_raises_instead_of_silently_recolouring(self):
        """槽位表漏了某个序列时必须抛——静默回落到次序取色会让那张图悄悄变色。"""
        with pytest.raises(KeyError):
            charts.dual_line_chart([1, 2, 3], self.SERIES, y_title="元",
                                   slots={"柴油自购": 0})


class TestSlotBars:
    """每根柱各自绑定一个槽位（原先在运输页里就地实现）。"""

    def test_each_bar_takes_its_own_slot(self):
        fig = charts.slot_bars(["优化前", "优化后"], [100.0, 70.0], [0, 1], unit="km")
        # 两根柱在同一条 trace 上，颜色按柱给（不是两条 trace）
        assert list(fig.data[0].marker.color) == [theme.series(0), theme.series(1)]

    def test_no_legend_because_the_x_labels_carry_identity(self):
        fig = charts.slot_bars(["优化前", "优化后"], [1.0, 2.0], [0, 1])
        assert fig.layout.showlegend is False

    def test_values_are_labelled_on_the_bars(self):
        fig = charts.slot_bars(["优化前", "优化后"], [1234.5, 999.0], [0, 1], decimals=1)
        assert list(fig.data[0].text) == ["1,234.5", "999.0"]


class TestBarWithCI:
    """两臂对照 + 95% CI 误差棒（原先在仓储页里就地实现，页面注释说「charts 不含误差棒」）。"""

    def test_error_bars_are_asymmetric_around_the_mean(self):
        fig = charts.bar_with_ci(["原始", "分区"], [100.0, 80.0], [90.0, 75.0], [110.0, 85.0],
                                 label="耗时长", unit=" 秒")
        err = fig.data[0].error_y
        assert list(err.array) == [10.0, 5.0]        # 上界 − 均值
        assert list(err.arrayminus) == [10.0, 5.0]   # 均值 − 下界
        assert err.symmetric is False

    def test_single_series_so_no_legend(self):
        fig = charts.bar_with_ci(["A", "B"], [1.0, 2.0], [0.5, 1.5], [1.5, 2.5], label="x")
        assert fig.layout.showlegend is False


class TestTcoCurve:
    """单条 TCO 曲线：标注盈亏平衡里程与参考里程（原先在改善页里就地实现）。"""

    def test_marks_the_breakeven_line_and_the_reference_point(self):
        fig = charts.tco_curve("柴油自购", [0, 50, 100], [353.0, 408.5, 464.0], slot=0,
                               breakeven_km=46.27, ref_km=87.62, ref_cost=450.3)
        assert len(fig.layout.shapes) >= 1                      # 盈亏平衡竖线
        assert any("46.3" in str(a.text) for a in fig.layout.annotations)
        assert any("87.6" in str(a.text) for a in fig.layout.annotations)

    def test_colour_is_pinned_to_the_entity_slot(self):
        fig = charts.tco_curve("纯电租赁", [0, 50], [384.0, 406.0], slot=1,
                               breakeven_km=46.27, ref_km=87.62, ref_cost=422.6)
        assert fig.data[0].line.color == theme.series(1)
