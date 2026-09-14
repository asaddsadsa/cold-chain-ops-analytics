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
