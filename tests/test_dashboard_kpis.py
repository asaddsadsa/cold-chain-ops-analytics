"""驾驶舱 KPI 窗口聚合与环比单元测试（12 号票）。

测试缝：`src/dashboard/kpis.py` 是**不依赖 Streamlit 的纯函数层**，所以「筛选窗口一变、
卡片数字怎么重算」这件事能被直接断言。看板页面本身按 spec 决策不做自动化测试
（以「4 页打开无报错 + 交互手测清单」验收）。
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.dashboard import kpis

TS = pd.Timestamp


def wh_daily() -> pd.DataFrame:
    """两天：金额与条数刻意不等，用来区分「金额加权」与「逐日率取平均」。"""
    return pd.DataFrame(
        [
            # 6-01：差异 50 / 账面 2000 → 准确率 0.975
            {"date": "2026-06-01", "inventory_accuracy": 0.975,
             "abs_diff_value": 50.0, "book_value": 2000.0},
            # 6-02：差异 5 / 账面 20000 → 准确率 0.99975
            {"date": "2026-06-02", "inventory_accuracy": 0.99975,
             "abs_diff_value": 5.0, "book_value": 20000.0},
        ]
    )


def tr_daily() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2026-06-01", "n_orders_total": 100, "n_orders_served": 90,
             "n_ontime": 60, "n_trips": 2, "load_rate_mean": 0.50,
             "diesel_cost": 1000.0, "ev_cost": 800.0},
            {"date": "2026-06-02", "n_orders_total": 300, "n_orders_served": 300,
             "n_ontime": 270, "n_trips": 18, "load_rate_mean": 0.90,
             "diesel_cost": 3000.0, "ev_cost": 2400.0},
        ]
    )


class TestWindow:
    def test_previous_is_adjacent_and_equal_length(self):
        w = kpis.Window(TS("2026-06-11"), TS("2026-06-20"))
        p = w.previous()
        assert w.days == 10 and p.days == 10
        assert p.end == TS("2026-06-10")
        assert p.start == TS("2026-06-01")

    def test_single_day_window_previous_is_the_day_before(self):
        w = kpis.Window(TS("2026-06-05"), TS("2026-06-05"))
        assert w.days == 1 and w.previous() == kpis.Window(TS("2026-06-04"), TS("2026-06-04"))

    def test_slice_is_inclusive_on_both_ends(self):
        df = pd.DataFrame({"date": pd.to_datetime(
            ["2026-06-01", "2026-06-02", "2026-06-03"])})
        got = kpis.slice_window(df, kpis.Window(TS("2026-06-01"), TS("2026-06-02")))
        assert list(got["date"].dt.day) == [1, 2]

    def test_slice_tolerates_a_time_component(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2026-06-01 23:59:59"])})
        got = kpis.slice_window(df, kpis.Window(TS("2026-06-01"), TS("2026-06-01")))
        assert len(got) == 1


class TestWindowAggregation:
    """聚合必须**先求和再相除**，不是把逐日率取平均。"""

    def test_inventory_accuracy_is_amount_weighted_across_days(self):
        got = kpis.inventory_accuracy_over(wh_daily(), kpis.Window(TS("2026-06-01"), TS("2026-06-02")))
        # Σ差异 55 / Σ账面 22000；若按逐日率平均会是 (0.975 + 0.99975) / 2 = 0.987375
        assert got == pytest.approx(1 - 55 / 22000)
        assert got != pytest.approx((0.975 + 0.99975) / 2)

    def test_time_window_rate_is_ontime_over_served(self):
        got = kpis.time_window_rate_over(tr_daily(), kpis.Window(TS("2026-06-01"), TS("2026-06-02")))
        assert got == pytest.approx(330 / 390)
        # 未服务单不进分母：若按 n_orders_total 算会得到 330/400
        assert got != pytest.approx(330 / 400)

    def test_load_rate_is_trip_weighted_not_day_averaged(self):
        got = kpis.load_rate_over(tr_daily(), kpis.Window(TS("2026-06-01"), TS("2026-06-02")))
        # 趟次加权：(0.5×2 + 0.9×18) / 20；逐日取平均会得到 0.70
        assert got == pytest.approx((0.5 * 2 + 0.9 * 18) / 20)
        assert got != pytest.approx(0.70)

    def test_cost_per_order_sums_cost_and_orders_separately(self):
        got = kpis.cost_per_order_over(tr_daily(), kpis.Window(TS("2026-06-01"), TS("2026-06-02")), "ev")
        assert got == pytest.approx(3200 / 400)

    def test_single_day_window_matches_that_day(self):
        w = kpis.Window(TS("2026-06-01"), TS("2026-06-01"))
        assert kpis.time_window_rate_over(tr_daily(), w) == pytest.approx(60 / 90)
        assert kpis.cost_per_order_over(tr_daily(), w, "diesel") == pytest.approx(10.0)

    def test_empty_window_returns_none_not_zero(self):
        w = kpis.Window(TS("2026-07-01"), TS("2026-07-02"))
        assert kpis.inventory_accuracy_over(wh_daily(), w) is None
        assert kpis.time_window_rate_over(tr_daily(), w) is None
        assert kpis.load_rate_over(tr_daily(), w) is None
        assert kpis.cost_per_order_over(tr_daily(), w) is None


class TestMetricDelta:
    """环比的方向语义：指标「变大是好是坏」由指标本身决定，与箭头方向无关。"""

    def test_cost_decrease_is_good_even_though_the_arrow_points_down(self):
        m = kpis.Metric("cost", "单均成本", value=30.0, previous=40.0, unit="元/单",
                        basis="x", higher_is_better=False)
        assert m.direction == "down"
        assert m.is_good is True
        assert m.delta == pytest.approx(-10.0)
        assert m.delta_pct == pytest.approx(-25.0)

    def test_rate_decrease_is_bad(self):
        m = kpis.Metric("tw", "时间窗达成率", value=0.80, previous=0.90, unit="%",
                        basis="x", higher_is_better=True)
        assert m.direction == "down" and m.is_good is False

    def test_missing_previous_gives_na_and_no_verdict(self):
        m = kpis.Metric("tw", "x", value=0.8, previous=None, unit="%", basis="b",
                        higher_is_better=True)
        assert m.direction == "na" and m.is_good is None
        assert "—" in m.format(None)

    def test_flat_is_neither_good_nor_bad(self):
        m = kpis.Metric("tw", "x", value=0.8, previous=0.8, unit="%", basis="b",
                        higher_is_better=True)
        assert m.direction == "flat" and m.is_good is None

    def test_formatting_follows_the_unit(self):
        pct = kpis.Metric("a", "a", 0.7955, None, "%", "b", True, decimals=2)
        money = kpis.Metric("b", "b", 34.106, None, "元/单", "b", False, decimals=2)
        assert pct.format(pct.value) == "79.55%"
        assert money.format(money.value) == "34.11 元/单"


class TestBuildMetrics:
    def test_four_cards_with_basis_and_the_selected_cost_mode(self):
        got = kpis.build_metrics(wh_daily(), tr_daily(),
                                 kpis.Window(TS("2026-06-01"), TS("2026-06-02")),
                                 cost_mode="diesel")
        assert [m.key for m in got] == ["inventory_accuracy", "time_window_rate",
                                        "load_rate", "cost_per_order"]
        assert "柴油自购" in got[-1].label
        # 每张卡片都必须带口径：四张卡片来自三个模块、四种分母，不写口径必然被误读
        assert all(m.basis for m in got)

    def test_cost_card_uses_the_requested_mode(self):
        w = kpis.Window(TS("2026-06-01"), TS("2026-06-01"))
        ev = kpis.build_metrics(wh_daily(), tr_daily(), w, cost_mode="ev")[-1]
        diesel = kpis.build_metrics(wh_daily(), tr_daily(), w, cost_mode="diesel")[-1]
        assert ev.value == pytest.approx(800 / 100)
        assert diesel.value == pytest.approx(1000 / 100)

    def test_previous_window_is_the_adjacent_equal_length_one(self):
        two_days = pd.concat([wh_daily(), pd.DataFrame(
            [{"date": "2026-06-03", "inventory_accuracy": 1.0,
              "abs_diff_value": 0.0, "book_value": 5000.0}])], ignore_index=True)
        w = kpis.Window(TS("2026-06-03"), TS("2026-06-03"))
        got = kpis.build_metrics(two_days, tr_daily(), w)[0]
        # 当日窗口的前置区间是 6-02 单日：准确率 0.99975，而非空
        assert got.previous == pytest.approx(0.99975)
        assert got.direction == "up"


class TestFiltersPure:
    """筛选器的**纯函数部分**可测（页面渲染本身按 spec 决策走手测清单）。"""

    def _filters(self, regions=("R07",)):
        from src.dashboard.filters import Filters
        return Filters(window=kpis.Window(TS("2026-06-01"), TS("2026-06-30")),
                       regions=regions, cost_mode="diesel")

    def test_region_filter_keeps_only_selected_regions(self):
        from src.dashboard.filters import apply_regions
        df = pd.DataFrame({"region": ["R01", "R07", "R07", "R08"]})
        assert list(apply_regions(df, self._filters()).index) == [1, 2]

    def test_empty_selection_means_all_not_none(self):
        from src.dashboard.filters import apply_regions
        df = pd.DataFrame({"region": ["R01", "R07"]})
        assert len(apply_regions(df, self._filters(regions=()))) == 2

    def test_missing_column_is_a_noop_rather_than_an_error(self):
        """没有片区维度的表（如仓内指标）经不起这个筛选——原样返回，由页面的
        `scope_note` 负责如实声明本页吃哪几个筛选器，而不是假装筛过了。"""
        from src.dashboard.filters import apply_regions
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert len(apply_regions(df, self._filters())) == 3

    def test_scope_note_states_what_the_page_actually_honours(self):
        note = self._filters().scope_note("日期范围、配送区域")
        assert "日期范围、配送区域" in note and "R07" in note


class TestThemePalette:
    """色板是**校验过**的：槽位顺序即色盲安全机制，不能被随手改动。"""

    def test_eight_slots_in_fixed_order_and_no_ninth(self):
        from src.dashboard import theme
        assert len(theme.CATEGORICAL_LIGHT) == 8
        assert theme.series(0) == "#2a78d6" and theme.series(7) == "#e34948"
        with pytest.raises(ValueError, match="第 9 个序列"):
            theme.series(8)

    def test_light_and_dark_are_the_same_hues_stepped_for_each_surface(self):
        from src.dashboard import theme
        assert len(theme.CATEGORICAL_LIGHT) == len(theme.CATEGORICAL_DARK)
        # 绿槽两模式同值（该色相在两个底色上都过门槛），其余为各自取阶而非明暗翻转
        assert theme.CATEGORICAL_LIGHT[5] == theme.CATEGORICAL_DARK[5]

    def test_status_colours_are_reserved_and_distinct_from_series(self):
        from src.dashboard import theme
        assert set(theme.STATUS) == {"good", "warning", "serious", "critical"}
        assert not (set(theme.STATUS.values()) & set(theme.CATEGORICAL_LIGHT))

    def test_sequential_ramp_is_one_hue_light_to_dark(self):
        from src.dashboard import theme
        ramp = theme.SEQUENTIAL_BLUE
        assert len(ramp) >= 10 and len(set(ramp)) == len(ramp)
        # 单色阶：每个步阶的蓝色通道都应大于红、绿通道（不允许彩虹色阶）
        for hexv in ramp:
            r, g, b = (int(hexv[i:i + 2], 16) for i in (1, 3, 5))
            assert b > r and b >= g, hexv
