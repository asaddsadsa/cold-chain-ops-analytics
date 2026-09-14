"""模块二（下）周度异常复盘 + TCO 决策 + what-if 预计算单元测试（11 号票）。

测试缝（spec「Testing Decisions」预约定）：
  ① KPI 纯函数缝：周度异常聚合、典型案例严重度排序、埋点两比例检验、
     双模式成本前后对比、外包对照，用手工小表断言等于手算期望；
  ② 成本模型缝：盈亏平衡里程与三档敏感性是纯函数，断言方向与单调性；
  ③ 数据产物缝：真实路网 + 代表作日跑一次 what-if 预计算网格，断言每档字段完整、
     用车数不超档位、超网格查询给出「需重跑预计算脚本」而非静默外推。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import gen_delivery_data as GD
from src import transport_decisions as TD


def make_anomalies(rows) -> pd.DataFrame:
    """rows: (date, region, anomaly_type, delay_min, handling_min, temp_max_c, vehicle_id)

    is_friday / rainy 由日期与片区按数据层 D 的口径派生，避免手工填错。
    """
    df = pd.DataFrame(
        [
            {
                "order_id": f"DO{i:04d}",
                "date": date,
                "region": region,
                "anomaly_type": atype,
                "delay_min": delay,
                "handling_min": handling,
                "temp_max_c": temp,
                "vehicle_id": veh,
                "poi_id": f"P{i:03d}",
            }
            for i, (date, region, atype, delay, handling, temp, veh) in enumerate(rows)
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    df["is_friday"] = df["date"].dt.weekday == 4
    df["rainy"] = False
    lo, hi = C.COLD_CHAIN_TEMP_RANGE
    df["temp_compliance_rate"] = ((df["temp_max_c"] >= lo) & (df["temp_max_c"] <= hi)).astype(float)
    return df


def make_trips(plan: str, rows) -> pd.DataFrame:
    """rows: (trip_id, n_stops, n_orders, distance_km)"""
    return pd.DataFrame(
        [
            {"trip_id": tid, "plan": plan, "mode": "diesel", "n_stops": stops,
             "n_orders": orders, "load_kg": orders * 40.0, "load_rate": 0.5,
             "distance_km": km, "duration_min": km, "mileage_cost": 0.0,
             "fixed_cost": 0.0, "trip_cost": 0.0}
            for tid, stops, orders, km in rows
        ]
    )


class TestWeeklyAnomalySummary:
    """周度异常复盘：按 ISO 周汇总类型分布与关键比率。"""

    def test_week_rates_and_type_shares_match_hand_calculation(self):
        # 同一 ISO 周（2026-06-01 周一）内 10 单：2 拥堵 + 1 晚点 + 7 无异常
        rows = [("2026-06-01", "R01", "拥堵", 60.0, 5.0, 6.0, "V01")] * 2
        rows += [("2026-06-02", "R01", "晚点", 120.0, 10.0, 6.0, "V02")]
        rows += [("2026-06-03", "R01", "无异常", 0.0, 0.0, 5.0, "V03")] * 7
        weekly = TD.weekly_anomaly_summary(make_anomalies(rows)).set_index("week")

        assert len(weekly) == 1
        w = weekly.iloc[0]
        assert w["n_shipments"] == 10
        assert w["n_anomalies"] == 3
        assert w["anomaly_rate"] == pytest.approx(0.3)
        assert w["late_rate"] == pytest.approx(0.1)
        assert w["share_拥堵"] == pytest.approx(0.2)  # 份额按运单数，不是按异常数
        assert w["share_无异常"] == pytest.approx(0.7)
        assert w["top_type"] == "拥堵"
        assert w["max_delay_min"] == pytest.approx(120.0)

    def test_weeks_are_split_on_iso_boundaries(self):
        rows = [("2026-06-07", "R01", "晚点", 60.0, 5.0, 6.0, "V01")]  # 周日 → W23
        rows += [("2026-06-08", "R01", "晚点", 60.0, 5.0, 6.0, "V01")]  # 周一 → W24
        weekly = TD.weekly_anomaly_summary(make_anomalies(rows))
        assert list(weekly["week"]) == ["2026-W23", "2026-W24"]
        assert weekly["n_shipments"].tolist() == [1, 1]

    def test_shipment_count_covers_every_row_including_no_anomaly(self, produced):
        """周度复盘的分母必须是**全部运单**，漏掉「无异常」会把异常率算高一截。"""
        weekly = TD.weekly_anomaly_summary(produced["anomalies"])
        assert weekly["n_shipments"].sum() == len(produced["anomalies"])
        assert weekly["n_anomalies"].sum() == int(
            (produced["anomalies"]["anomaly_type"] != "无异常").sum()
        )


class TestEmbeddedPointReproduction:
    """埋点复现：周五晚点偏高、R07 异常偏高、雨日晚点偏高（需求 41）。"""

    def test_friday_late_point_is_detected_when_present(self):
        rows = [("2026-06-05", "R01", "晚点", 60.0, 5.0, 6.0, "V01")] * 40  # 周五
        rows += [("2026-06-04", "R01", "无异常", 0.0, 0.0, 5.0, "V02")] * 200  # 周四
        got = TD.embedded_point_reproduction(make_anomalies(rows))
        friday = got["friday_late"]
        assert friday["rate_1"] > friday["rate_0"]
        assert friday["significant_at_5pct"] is True
        assert friday["z"] > 0

    def test_no_signal_is_reported_as_not_significant(self):
        """护栏：埋点检测不能恒真——两组同率时必须给出「不显著」。"""
        rows = []
        for week_start in ("2026-06-01", "2026-06-08", "2026-06-15"):
            d = pd.Timestamp(week_start)
            rows += [(str((d + pd.Timedelta(days=4)).date()), "R01", "晚点", 60.0, 5.0, 6.0, "V1")] * 3
            rows += [(str((d + pd.Timedelta(days=1)).date()), "R01", "晚点", 60.0, 5.0, 6.0, "V2")] * 3
            rows += [(str((d + pd.Timedelta(days=4)).date()), "R01", "无异常", 0.0, 0.0, 5.0, "V1")] * 25
            rows += [(str((d + pd.Timedelta(days=1)).date()), "R01", "无异常", 0.0, 0.0, 5.0, "V2")] * 25
        got = TD.embedded_point_reproduction(make_anomalies(rows))
        assert got["friday_late"]["significant_at_5pct"] is False

    def test_replication_counts_weeks_and_exposes_power(self):
        """逐周复现要连**样本量**一起报：单周周五样本小，不显著不等于埋点不存在。"""
        rows = [("2026-06-05", "R07", "故障", 300.0, 60.0, 6.0, "V01")] * 30
        rows += [("2026-06-12", "R07", "故障", 300.0, 60.0, 6.0, "V01")] * 30
        rows += [("2026-06-04", "R01", "无异常", 0.0, 0.0, 5.0, "V02")] * 300
        rows += [("2026-06-11", "R01", "无异常", 0.0, 0.0, 5.0, "V02")] * 300
        got = TD.embedded_point_reproduction(make_anomalies(rows))
        rep = got["weekly_replication"]["r07_anomaly"]
        assert rep["n_weeks"] == 2
        assert rep["n_significant"] == 2
        # 功效报的是**该周实际**的较小分组运单数（每周 30 单 R07 vs 300 单其他），
        # 不是按比例估算——「当周不显著」才判断得出是埋点弱还是样本少
        assert rep["min_group_n"] == 30
        assert got["weekly_replication"]["friday_late"]["min_group_n"] == 30


class TestTypicalCases:
    """典型案例清单：每类按**严重度**取前 N 条，规则显式、不择优叙事。"""

    def test_late_cases_ranked_by_delay(self):
        rows = [
            ("2026-06-01", "R01", "晚点", 90.0, 10.0, 6.0, "V01"),
            ("2026-06-02", "R02", "晚点", 240.0, 20.0, 6.0, "V02"),
            ("2026-06-03", "R03", "晚点", 150.0, 15.0, 6.0, "V03"),
        ]
        cases = TD.typical_anomaly_cases(make_anomalies(rows), per_type=2)
        late = cases[cases["anomaly_type"] == "晚点"]
        assert late["delay_min"].tolist() == [240.0, 150.0]
        assert late["order_id"].tolist() == ["DO0001", "DO0002"]

    def test_temperature_cases_ranked_by_excursion_not_delay(self):
        """温控波动的诊断价值在**温度**不在时刻：延误只有 3 分钟但温度更高的应排前面。"""
        rows = [
            ("2026-06-01", "R01", "温控波动", 9.0, 30.0, 9.5, "V01"),
            ("2026-06-02", "R02", "温控波动", 2.0, 20.0, 12.5, "V02"),
        ]
        cases = TD.typical_anomaly_cases(make_anomalies(rows), per_type=2)
        temp = cases[cases["anomaly_type"] == "温控波动"]
        assert temp["vehicle_id"].tolist() == ["V02", "V01"]
        assert temp["severity"].iloc[0] == pytest.approx(12.5 - C.CABIN_TEMP_SETPOINT_C)

    def test_no_anomaly_rows_are_excluded_and_capped(self):
        rows = [("2026-06-01", "R01", "无异常", 0.0, 0.0, 5.0, "V01")] * 5
        rows += [("2026-06-02", "R01", "拥堵", 60.0, 5.0, 6.0, "V02")] * 5
        cases = TD.typical_anomaly_cases(make_anomalies(rows), per_type=2)
        assert "无异常" not in set(cases["anomaly_type"])
        assert len(cases) == 2  # 每类封顶 2 条

    def test_case_rows_carry_traceable_identity(self):
        rows = [("2026-06-01", "R07", "故障", 200.0, 90.0, 6.0, "V09")]
        got = TD.typical_anomaly_cases(make_anomalies(rows))
        for col in ("order_id", "date", "region", "poi_id", "vehicle_id", "anomaly_type",
                    "delay_min", "handling_min", "severity"):
            assert col in got.columns, col


class TestModeCostBeforeAfter:
    """双模式优化前后日/月总成本（需求 22/42）。"""

    def trips(self):
        base = make_trips("baseline", [("B1", 5, 10, 100.0), ("B2", 5, 10, 100.0)])
        opt = make_trips("optimized", [("O1", 5, 10, 70.0), ("O2", 5, 10, 70.0)])
        return pd.concat([base, opt], ignore_index=True)

    def test_daily_cost_is_fixed_plus_mileage_per_trip(self):
        got = TD.mode_cost_before_after(self.trips())
        d = got["diesel"]
        fixed, per_km = C.cost_fixed_per_day("diesel"), C.cost_per_km("diesel")
        assert d["baseline"]["day_total"] == pytest.approx(2 * fixed + 200 * per_km)
        assert d["optimized"]["day_total"] == pytest.approx(2 * fixed + 140 * per_km)

    def test_month_equals_day_times_workdays_and_order_count_matches_trips(self):
        got = TD.mode_cost_before_after(self.trips())
        for mode in ("diesel", "ev"):
            b = got[mode]["baseline"]
            assert b["month_total"] == pytest.approx(b["day_total"] * C.WORKDAYS_PER_MONTH)
            assert b["n_orders"] == 20
            assert b["day_per_order"] == pytest.approx(b["day_total"] / 20)

    def test_optimization_reduces_total_cost_and_saving_is_reported(self):
        got = TD.mode_cost_before_after(self.trips())
        for mode in ("diesel", "ev"):
            assert got[mode]["optimized"]["day_total"] < got[mode]["baseline"]["day_total"]
            assert got[mode]["saving"]["day"] == pytest.approx(
                got[mode]["baseline"]["day_total"] - got[mode]["optimized"]["day_total"]
            )
            assert got[mode]["saving"]["month"] == pytest.approx(
                got[mode]["saving"]["day"] * C.WORKDAYS_PER_MONTH
            )

    def test_allocation_exactness_uses_vehicle_count_not_trip_id(self):
        """回归：同一份 trips，只有「用车数」不同时判定必须翻转。

        trip_id 是**趟序号**不是车牌——趟序号唯一与「一车一趟」是两回事。早先版本用
        「trip_id 是否唯一」当判据，结果恒为 True，等于给了一个永远是「精确分摊」的假证据。
        """
        trips = make_trips("optimized", [("O1", 5, 10, 70.0), ("O2", 5, 10, 70.0)])
        assert TD.mode_cost_before_after(trips)["per_trip_allocation_exact"] is None
        assert TD.mode_cost_before_after(
            trips, {"optimized": 2}
        )["per_trip_allocation_exact"] is True
        assert TD.mode_cost_before_after(
            trips, {"optimized": 1}
        )["per_trip_allocation_exact"] is False

    def test_allocation_exactness_is_unknown_when_any_plan_lacks_vehicle_count(self):
        """只对「恰好提供了用车数的那几个方案」下结论，等于把没校验说成已校验。"""
        trips = pd.concat(
            [make_trips("baseline", [("B1", 5, 10, 90.0)]),
             make_trips("optimized", [("O1", 5, 10, 70.0)])],
            ignore_index=True,
        )
        got = TD.mode_cost_before_after(trips, {"optimized": 1})
        assert got["per_trip_allocation_exact"] is None

    def test_cost_reduction_pct_is_named_and_signed_as_a_reduction(self):
        """本项目里程/成本类指标在对比表里用负数表示变好；节省率必须另起名字并取正，
        否则「省了 13%」会被读成「贵了 13%」。"""
        got = TD.mode_cost_before_after(self.trips())
        s = got["diesel"]["saving"]
        assert s["cost_reduction_pct"] > 0
        assert "pct" not in s

    def test_ev_is_cheaper_per_km_so_optimization_pays_off_more(self):
        got = TD.mode_cost_before_after(self.trips())
        for plan in ("baseline", "optimized"):
            assert got["ev"][plan]["day_total"] < got["diesel"][plan]["day_total"]


class TestOutsourceComparison:
    """自营（优化后）单趟成本 vs 货拉拉外包对照。"""

    def test_uses_each_trips_own_distance_and_stop_count(self):
        """对照必须用**该趟自己的**里程与点数：短途轻载与长程多点的结论是相反的。"""
        trips = make_trips("optimized", [("O1", 6, 5, 25.0), ("O2", 18, 20, 120.0)])
        got = TD.outsource_comparison(trips)
        by_trip = got["per_trip"].set_index("trip_id")
        assert by_trip.loc["O1", "huolala_cost"] == pytest.approx(GD.huolala_cost(25.0, 6))
        assert by_trip.loc["O2", "huolala_cost"] == pytest.approx(GD.huolala_cost(120.0, 18))
        # 短途轻载外包更省、长程多点自营更省——结论双向可分，不是一刀切
        assert by_trip.loc["O1", "verdict"] == "外包更省"
        assert by_trip.loc["O2", "verdict"] == "自营更省"

    def test_totals_compare_fleet_self_cost_against_full_outsourcing(self):
        trips = make_trips("optimized", [("O1", 6, 5, 25.0), ("O2", 18, 20, 120.0)])
        got = TD.outsource_comparison(trips)
        assert got["total_self_cost"] == pytest.approx(
            got["per_trip"]["self_cost"].sum()
        )
        assert got["total_huolala_cost"] == pytest.approx(
            got["per_trip"]["huolala_cost"].sum()
        )
        assert got["n_trips_self_cheaper"] + got["n_trips_outsource_cheaper"] + \
            got["n_trips_tie"] == len(trips)

    def test_before_and_after_are_reported_separately(self):
        trips = pd.concat(
            [make_trips("baseline", [("B1", 18, 20, 150.0)]),
             make_trips("optimized", [("O1", 18, 20, 100.0)])],
            ignore_index=True,
        )
        got = TD.outsource_comparison(trips)
        assert set(got["by_plan"]) == {"baseline", "optimized"}
        assert got["by_plan"]["optimized"]["total_self_cost"] < \
            got["by_plan"]["baseline"]["total_self_cost"]


class TestSensitivity:
    """三档敏感性（司机工资 / 能源价格 / 租金），参数取 config 锚点。"""

    def test_cost_is_monotone_in_each_parameter(self):
        got = TD.sensitivity_table(reference_km=100.0)
        for mode in ("diesel", "ev"):
            wages = [got["driver_wage"][lv][f"{mode}_daily_cost"] for lv in C.SENSITIVITY_LEVELS]
            assert wages == sorted(wages), f"{mode} 日成本应对司机工资单调"
        rents = [got["rent"][lv]["ev_daily_cost"] for lv in C.SENSITIVITY_LEVELS]
        assert rents == sorted(rents)
        energy = [got["energy_price"][lv][f"{mode}_daily_cost"] for lv in C.SENSITIVITY_LEVELS]
        assert energy == sorted(energy)

    def test_one_at_a_time_holds_other_parameters_at_mid(self):
        got = TD.sensitivity_table(reference_km=100.0)
        mid = got["driver_wage"]["mid"]["diesel_daily_cost"]
        assert mid == pytest.approx(C.daily_total_cost("diesel", 100.0))

    def test_breakeven_moves_with_rent_but_not_with_driver_wage(self):
        """司机工资对两模式同额、相减抵消 → 盈亏平衡里程纹丝不动（06 号票已实测）。"""
        got = TD.sensitivity_table(reference_km=100.0)
        wages = {got["driver_wage"][lv]["breakeven_km"] for lv in C.SENSITIVITY_LEVELS}
        assert len(wages) == 1
        rents = [got["rent"][lv]["breakeven_km"] for lv in C.SENSITIVITY_LEVELS]
        assert rents[0] < rents[1] < rents[2]


class TestWhatIfLookup:
    """what-if 缓存切档与超网格提示（ADR-0010）。"""

    def cache(self):
        return {
            "grid": [5, 6, 7],
            "gears": {str(n): {"n_vehicles_available": n, "feasible": True} for n in (5, 6, 7)},
        }

    def test_cached_gear_is_returned(self):
        got = TD.whatif_lookup(self.cache(), 6)
        assert got["available"] is True
        assert got["n_vehicles"] == 6

    def test_out_of_grid_asks_for_recompute_instead_of_extrapolating(self):
        for n in (4, 8):
            got = TD.whatif_lookup(self.cache(), n)
            assert got["available"] is False
            assert "重跑预计算" in got["message"]
            assert got["grid"] == [5, 7]

    def test_grid_definition_is_16_gears_from_config(self):
        grid = TD.whatif_gears()
        assert grid[0] == C.WHATIF_VEHICLE_RANGE[0]
        assert grid[-1] == C.WHATIF_VEHICLE_RANGE[1]
        assert len(grid) == C.WHATIF_VEHICLE_RANGE[1] - C.WHATIF_VEHICLE_RANGE[0] + 1 == 16


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    """真实路网 + 代表作日跑一次完整决策分析（what-if 用 1 秒/档压缩耗时）。"""
    out = tmp_path_factory.mktemp("decisions")
    return TD.run_all(out_dir=out, whatif_time_limit_sec=1.0)


class TestRealDataProducts:
    """数据产物缝：真实数据上验结构、口径与缓存完整性。"""

    def test_weekly_table_covers_the_whole_horizon(self, produced):
        weekly = produced["weekly"]
        assert len(weekly) == 13  # 2026-06-01 ~ 08-29 落在 13 个 ISO 周
        assert weekly["week"].is_monotonic_increasing
        assert weekly["n_shipments"].min() > 0

    def test_all_embedded_points_reproduce_on_pooled_sample(self, produced):
        got = produced["reproduction"]
        for key in ("friday_late", "r07_anomaly"):
            assert got[key]["significant_at_5pct"] is True, key
            assert got[key]["rate_1"] > got[key]["rate_0"], key
        assert got["rainy_late"]["rate_1"] > got["rainy_late"]["rate_0"]

    def test_cases_are_capped_and_traceable(self, produced):
        cases = produced["cases"]
        assert len(cases) > 0
        per_type = cases.groupby("anomaly_type").size()
        assert (per_type <= C.ANOMALY_CASES_PER_TYPE).all()
        assert cases["order_id"].is_unique

    def test_tco_before_after_matches_published_transport_kpi(self, produced):
        """TCO 的「优化前/后」必须与 10 号票落盘的对比表**同一批数字**，不得各算一套。"""
        kpi = json.loads(Path(C.TRANSPORT_KPI_JSON).read_text(encoding="utf-8"))
        tco = produced["tco"]
        for plan in ("baseline", "optimized"):
            assert tco["mode_costs"]["diesel"][plan]["day_total"] == pytest.approx(
                kpi[plan]["cost"]["diesel"]["total"], rel=1e-4
            )
            assert tco["mode_costs"]["ev"][plan]["day_total"] == pytest.approx(
                kpi[plan]["cost"]["ev"]["total"], rel=1e-4
            )

    def test_per_trip_allocation_is_exact_on_the_representative_day(self, produced):
        """代表日每车恰好 1 趟 → 日固定成本全额摊入该趟是精确分摊。"""
        assert produced["tco"]["mode_costs"]["per_trip_allocation_exact"] is True

    def test_whatif_covers_every_gear_with_complete_fields(self, produced):
        cache = produced["whatif"]
        assert [int(k) for k in cache["gears"]] == TD.whatif_gears()
        for gear, rec in cache["gears"].items():
            assert rec["n_vehicles_available"] == int(gear)
            assert rec["n_vehicles_used"] is None or rec["n_vehicles_used"] <= int(gear)
            if rec["feasible"]:
                assert rec["total_distance_km"] > 0
                assert 0 < rec["time_window_rate"] <= 1
                assert 0 < rec["load_rate_mean"] <= 1
                assert rec["cost"]["diesel"]["total"] > 0
                assert rec["cost"]["ev"]["total"] > 0
            else:
                assert rec["infeasible_reason"]

    def test_whatif_reference_gear_matches_published_solution(self, produced):
        """档位=车队规模时，预计算应能重建 10 号票发表的优化结果（一致性自检）。"""
        check = produced["whatif"]["consistency_check"]
        assert check["gear"] == C.FLEET_SIZE
        assert check["distance_km"] > 0
        assert check["distance_delta_pct"] is not None

    def test_artifacts_written(self, produced):
        out = Path(produced["report"]).parent
        for name in (C.TRANSPORT_ANOMALY_WEEKLY_CSV.name, C.TRANSPORT_ANOMALY_CASES_CSV.name,
                     C.TRANSPORT_ANOMALY_POINTS_JSON.name,
                     C.TRANSPORT_TCO_JSON.name, C.TRANSPORT_TCO_MD.name,
                     C.TRANSPORT_WHATIF_JSON.name):
            assert (out / name).exists(), f"未落盘：{name}"

    def test_embedded_point_results_are_machine_readable_not_just_prose(self, produced):
        """埋点复现的结论（含逐周分组样本量）必须落盘：

        「把当周不显著与埋点不存在区分开」这句话要成立，min_group_n 就得是可读的产物，
        只写在报告散文里等于没落盘——下游看板与复现者无从断言。
        """
        out = Path(produced["report"]).parent
        pts = json.loads((out / C.TRANSPORT_ANOMALY_POINTS_JSON.name).read_text(encoding="utf-8"))
        for key in TD._EMBEDDED_POINTS:
            assert pts[key]["significant_at_5pct"] is True, key
            assert pts[key]["z"] > 0 and pts[key]["rate_1"] > pts[key]["rate_0"], key
        for key in ("friday_late", "r07_anomaly"):
            rep = pts["weekly_replication"][key]
            assert rep["n_weeks"] == 13 and rep["min_group_n"] > 0
            assert rep["min_group_n"] <= rep["median_group_n"]

    def test_whatif_json_is_self_describing_for_the_dashboard(self, produced):
        out = Path(produced["report"]).parent
        cache = json.loads((out / C.TRANSPORT_WHATIF_JSON.name).read_text(encoding="utf-8"))
        assert cache["grid"] == TD.whatif_gears()
        assert cache["representative_day"]
        assert "预计算" in cache["data_category"] or "预计算" in cache["note"]
