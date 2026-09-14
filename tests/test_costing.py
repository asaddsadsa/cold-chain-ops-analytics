"""车队成本模型的回归测试（`src/costing.py`）。

这些测试原先散在三处——`test_config`（成本原语住在 config 时）、`test_delivery_data`
（货拉拉计价与 TCO 分析住在数据生成器时）、`test_transport_decisions`（外包对照住在决策
模块时）。收口后函数都搬进 `costing`，测试随之搬来：**测随模块走**，找实现与找测试是同一个
地方。三处里重复的那几份（敏感性单调性、盈亏平衡对工资不敏感）合并成一份。

测试缝仍是 spec 的「成本模型缝」——纯函数，对手工算例断言。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import costing


# ---------------------------------------------------------------------------
# 成本原语
# ---------------------------------------------------------------------------
class TestFixedPerDay:
    """日固定成本。中位锚点对齐 data_sources_ledger.md 第 4 节。"""

    def test_diesel_fixed_mid(self):
        # 柴油日固定 = 折旧 62 + 保险 22 + 工资 269 = 353
        assert costing.fixed_per_day("diesel") == pytest.approx(353.0)

    def test_ev_fixed_mid(self):
        # 纯电日固定 = 租金 115 + 工资 269 = 384
        # 注：台账「≈385」为月除未取整（3000/26=115.4、7000/26=269.2 → 384.6）；
        # config 忠实于文档显式的每日锚点 115+269=384。
        assert costing.fixed_per_day("ev") == pytest.approx(384.0)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            costing.fixed_per_day("hybrid")


class TestPerKm:
    def test_diesel_per_km_mid(self):
        # 柴油公里变动 = 燃油 0.96 + 尿素 0.05 + 维保 0.10 = 1.11
        assert costing.per_km("diesel") == pytest.approx(1.11)

    def test_ev_per_km_mid(self):
        assert costing.per_km("ev") == pytest.approx(0.44)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            costing.per_km("hybrid")


class TestDailyTotal:
    def test_is_fixed_plus_mileage_times_variable(self):
        km = 120.0
        expected = costing.fixed_per_day("diesel") + km * costing.per_km("diesel")
        assert costing.daily_total("diesel", km) == pytest.approx(expected)

    def test_one_at_a_time_sensitivity(self):
        # 单变一个参数、其余留 mid：三个敏感性参数各自独立生效
        base = costing.daily_total("diesel", 100.0)
        # 仅调高司机工资 → 固定成本上升，变动不变
        assert costing.daily_total("diesel", 100.0, driver_wage="high") > base
        # 仅调高油价 → 变动成本上升
        assert costing.daily_total("diesel", 100.0, energy_price="high") > base
        # 仅调高租金对 diesel 无影响（柴油无租金）
        assert costing.daily_total("diesel", 100.0, rent="high") == pytest.approx(base)
        # 租金对 ev 有影响
        assert costing.daily_total("ev", 100.0, rent="high") > costing.daily_total("ev", 100.0)


class TestBreakeven:
    def test_diesel_cheaper_below_ev_cheaper_above(self):
        # 纯电日固定更高、公里变动更低 → 存在唯一盈亏平衡里程
        be = costing.breakeven_km()
        assert be > 0
        assert costing.daily_total("ev", be * 2) < costing.daily_total("diesel", be * 2)
        assert costing.daily_total("ev", be / 2) > costing.daily_total("diesel", be / 2)

    def test_is_exactly_the_crossover(self):
        be = costing.breakeven_km()
        assert costing.daily_total("ev", be) == pytest.approx(
            costing.daily_total("diesel", be), rel=1e-6
        )

    def test_independent_of_driver_wage(self):
        # 司机工资对两模式同额，相减抵消 → 盈亏平衡里程不受工资档位影响
        assert costing.breakeven_km(driver_wage="low") == pytest.approx(
            costing.breakeven_km(driver_wage="high")
        )

    def test_equal_variable_costs_have_no_crossover(self, monkeypatch):
        """两模式公里变动成本相等时无盈亏平衡点，必须显式抛错而不是返回 inf。

        直接打桩 `per_km` 而不是凑一组 config 数值：浮点相加不保证精确相等
        （0.29 + 0.05 + 0.10 = 0.44000000000000006），凑出来的「相等」其实差一个 eps，
        除零分支根本不会触发——那样测的是运气，不是守卫。
        """
        monkeypatch.setattr(costing, "per_km", lambda mode, **kw: 0.5)
        with pytest.raises(ZeroDivisionError):
            costing.breakeven_km()


class TestCheaperMode:
    def test_picks_the_lower_and_prefers_ev_on_tie(self):
        assert costing.cheaper_mode(100.0, 80.0) == "ev"
        assert costing.cheaper_mode(80.0, 100.0) == "diesel"
        assert costing.cheaper_mode(90.0, 90.0) == "ev"


# ---------------------------------------------------------------------------
# 货拉拉外包计价
# ---------------------------------------------------------------------------
class TestHuolalaCost:
    """货拉拉计价（需求 21/22；参数读 config，台账登记为情景假设）。"""

    def test_short_trip_is_just_the_start_fee(self):
        # 起步 90 元含前 5km，8 点以内无超点费
        assert costing.huolala_cost(5.0, 8) == pytest.approx(C.HUOLALA_START_FEE)

    def test_medium_trip_charges_first_band(self):
        # 25km / 8 点 = 90 + (25-5)×5 = 190
        assert costing.huolala_cost(25.0, 8) == pytest.approx(190.0)

    def test_long_trip_adds_second_band_and_extra_stops(self):
        # 60km / 12 点 = 90 + 20×5 + 35×avg(3.5,4.8) + 4×38
        expected = 90.0 + 20 * 5.0 + 35 * 4.15 + 4 * 38.0
        assert costing.huolala_cost(60.0, 12) == pytest.approx(expected)


class TestHuolalaCrossover:
    def test_crossover_is_a_real_sign_change(self):
        """平价里程两侧的符号必须真的相反。

        `f = 自营 − 货拉拉`：短途时货拉拉的起步价（含前 5km）压倒自营要摊的日固定成本，
        f > 0（外包更省）；里程一长，货拉拉 4–5 元/km 远高于自营 1.11 元/km，f 转负
        （自营更省）。这个交点就是「这趟该不该外包」的分界。
        """
        km = costing.huolala_crossover_km(12, "diesel")
        assert km is not None

        def f(d: float) -> float:
            return costing.daily_total("diesel", d) - costing.huolala_cost(d, 12)

        assert f(km - 15) > 0 > f(km + 15)

    def test_returns_none_when_the_curves_never_cross(self):
        # 点位数极大时外包报价（超出免费点数按 38 元/点）远高于自营，0–400km 内无交点
        assert costing.huolala_crossover_km(60, "diesel") is None


# ---------------------------------------------------------------------------
# TCO 分析
# ---------------------------------------------------------------------------
class TestTcoAnalysis:
    """双模式 TCO 曲线、盈亏平衡里程、三档敏感性、外包对照与建议（需求 21/22）。"""

    def test_curve_is_fixed_plus_mileage_times_variable(self):
        tco = costing.tco_analysis()
        for mode in ("diesel", "ev"):
            c = tco["curves"][mode]
            for km, cost in zip(c["mileage_km"], c["daily_total_cost"]):
                assert cost == pytest.approx(c["daily_fixed"] + km * c["per_km"], abs=0.01)

    def test_curve_covers_configured_mileage_grid(self):
        tco = costing.tco_analysis()
        assert tuple(tco["curves"]["diesel"]["mileage_km"]) == C.TCO_MILEAGE_GRID_KM

    def test_breakeven_matches_the_primitive(self):
        tco = costing.tco_analysis()
        assert tco["breakeven_km"]["km"] == pytest.approx(costing.breakeven_km(), abs=0.01)

    def test_breakeven_really_is_the_cost_crossover(self):
        # 报告的盈亏平衡里程必须真的是两条成本曲线的交点（自洽性）
        tco = costing.tco_analysis()
        be = tco["breakeven_km"]["km"]
        d, e = tco["curves"]["diesel"], tco["curves"]["ev"]
        at = lambda c, km: c["daily_fixed"] + km * c["per_km"]
        assert at(e, be + 20) < at(d, be + 20)  # 高于盈亏平衡：纯电更省
        assert at(d, be - 20) < at(e, be - 20)  # 低于盈亏平衡：柴油更省

    def test_comparison_table_has_a_verdict_per_profile(self):
        tco = costing.tco_analysis()
        for r in tco["huolala_comparison"]:
            assert r["verdict"] in {"自营更省", "外包更省", "基本持平"}
            assert r["huolala_cost"] > 0 and r["self_diesel_cost"] > 0 and r["self_ev_cost"] > 0

    def test_comparison_table_discriminates(self):
        # 对照表必须能区分两种结论，否则没有决策价值
        verdicts = {r["verdict"] for r in costing.tco_analysis()["huolala_comparison"]}
        assert len(verdicts) >= 2

    def test_recommendation_cites_numbers_from_its_own_output(self):
        tco = costing.tco_analysis()
        rec = tco["recommendation"]
        assert rec["recommended_mode"] in {"ev", "diesel"}
        assert rec["breakeven_km"] == pytest.approx(tco["breakeven_km"]["km"], abs=0.01)
        assert rec["rationale"]


class TestSensitivity:
    """三档敏感性（司机工资 / 能源价格 / 租金）。合并自原先分散在三个测试文件里的重复断言。"""

    def test_covers_three_params_three_levels(self):
        s = costing.tco_analysis()["sensitivity"]
        for param in C.SENSITIVITY_PARAMS:
            assert set(s[param]) == set(C.SENSITIVITY_LEVELS)
            for rec in s[param].values():
                assert rec["diesel_daily_cost"] > 0 and rec["ev_daily_cost"] > 0
                assert np.isfinite(rec["breakeven_km"])

    def test_costs_move_in_the_right_direction(self):
        s = costing.tco_analysis()["sensitivity"]
        lo, hi = C.SENSITIVITY_LEVELS[0], C.SENSITIVITY_LEVELS[-1]
        # 工资 / 能源 / 租金上涨都会推高纯电日总成本
        for param in C.SENSITIVITY_PARAMS:
            assert s[param][lo]["ev_daily_cost"] < s[param][hi]["ev_daily_cost"]
        # 工资与能源上涨推高柴油日总成本（租金不进柴油成本）
        for param in ("driver_wage", "energy_price"):
            assert s[param][lo]["diesel_daily_cost"] < s[param][hi]["diesel_daily_cost"]
        # 租金只进纯电固定成本，上涨必然推高盈亏平衡里程
        rent_be = [s["rent"][lv]["breakeven_km"] for lv in C.SENSITIVITY_LEVELS]
        assert rent_be[0] < rent_be[1] < rent_be[2]

    def test_one_at_a_time_holds_other_parameters_at_mid(self):
        """「中位档」必须**真的**等于不带敏感性参数的默认口径，否则敏感性表描述的是一个
        不存在的基线组合。"""
        s = costing.tco_analysis(100.0)["sensitivity"]
        assert s["driver_wage"]["mid"]["diesel_daily_cost"] == pytest.approx(
            costing.daily_total("diesel", 100.0)
        )

    def test_driver_wage_leaves_breakeven_unchanged(self):
        # 司机工资对两模式同额、相减抵消（costing.breakeven_km 的性质），敏感性表须如实反映
        s = costing.tco_analysis()["sensitivity"]
        be = [s["driver_wage"][lv]["breakeven_km"] for lv in C.SENSITIVITY_LEVELS]
        assert be[0] == pytest.approx(be[1]) == pytest.approx(be[2])
        # 工资却实实在在推高两模式的日总成本
        costs = [s["driver_wage"][lv]["diesel_daily_cost"] for lv in C.SENSITIVITY_LEVELS]
        assert costs[0] < costs[1] < costs[2]


# ---------------------------------------------------------------------------
# 自营 vs 外包逐趟对照
# ---------------------------------------------------------------------------
def _trips(plan: str, rows) -> pd.DataFrame:
    """最小逐趟表：`(trip_id, n_stops, n_orders, distance_km)`。"""
    return pd.DataFrame(
        [{"trip_id": t, "plan": plan, "n_stops": s, "n_orders": o, "distance_km": d}
         for t, s, o, d in rows]
    )


class TestOutsourceComparison:
    """自营（优化后）单趟成本 vs 货拉拉外包对照。"""

    def test_uses_each_trips_own_distance_and_stop_count(self):
        """对照必须用**该趟自己的**里程与点数：短途轻载与长程多点的结论是相反的。"""
        trips = _trips("optimized", [("O1", 6, 5, 25.0), ("O2", 18, 20, 120.0)])
        by_trip = costing.outsource_comparison(trips)["per_trip"].set_index("trip_id")
        assert by_trip.loc["O1", "huolala_cost"] == pytest.approx(
            costing.huolala_cost(25.0, 6))
        assert by_trip.loc["O2", "huolala_cost"] == pytest.approx(
            costing.huolala_cost(120.0, 18))
        # 短途轻载外包更省、长程多点自营更省——结论双向可分，不是一刀切
        assert by_trip.loc["O1", "verdict"] == "外包更省"
        assert by_trip.loc["O2", "verdict"] == "自营更省"

    def test_totals_compare_fleet_self_cost_against_full_outsourcing(self):
        trips = _trips("optimized", [("O1", 6, 5, 25.0), ("O2", 18, 20, 120.0)])
        got = costing.outsource_comparison(trips)
        assert got["total_self_cost"] == pytest.approx(got["per_trip"]["self_cost"].sum())
        assert got["total_huolala_cost"] == pytest.approx(
            got["per_trip"]["huolala_cost"].sum())
        assert got["n_trips_self_cheaper"] + got["n_trips_outsource_cheaper"] + \
            got["n_trips_tie"] == len(trips)

    def test_before_and_after_are_reported_separately(self):
        trips = pd.concat(
            [_trips("baseline", [("B1", 18, 20, 150.0)]),
             _trips("optimized", [("O1", 18, 20, 100.0)])],
            ignore_index=True,
        )
        got = costing.outsource_comparison(trips)
        assert set(got["by_plan"]) == {"baseline", "optimized"}
        assert got["by_plan"]["optimized"]["total_self_cost"] < \
            got["by_plan"]["baseline"]["total_self_cost"]

    def test_primary_plan_prefers_the_optimized_one(self):
        trips = pd.concat(
            [_trips("baseline", [("B1", 6, 5, 25.0)]),
             _trips("optimized", [("O1", 6, 5, 25.0)])],
            ignore_index=True,
        )
        assert costing.outsource_comparison(trips)["primary_plan"] == "optimized"

    def test_empty_trips_raises_instead_of_returning_an_empty_verdict(self):
        with pytest.raises(ValueError, match="无法做外包对照"):
            costing.outsource_comparison(pd.DataFrame(
                columns=["trip_id", "plan", "n_stops", "n_orders", "distance_km"]))
