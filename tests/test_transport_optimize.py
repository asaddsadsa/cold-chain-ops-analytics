"""模块二（上）运输调度优化核心单元测试（10 号票）。

测试缝（spec「Testing Decisions」预约定）：
  ① KPI 纯函数缝：需求聚合、时间窗达成率逐单判定、满载率、里程利用率、成本，
     用手工小矩阵断言等于手算期望；
  ② 算法门禁缝：基线贪心在手工小矩阵上的路线必须与手推一致；
  ③ 数据产物缝：真实路网跑通，断言对比表降幅方向、GeoJSON 结构、外推换算。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import transport_optimize as TO

# --- 手工小矩阵：DC + 3 个节点 -------------------------------------------------
# 距离(km)：DC-A 10、DC-B 20、DC-C 30、A-B 5、B-C 5、A-C 25
MINI_IDS = ["DC", "PA", "PB", "PC"]
MINI_DIST = pd.DataFrame(
    [
        [0.0, 10.0, 20.0, 30.0],
        [10.0, 0.0, 5.0, 25.0],
        [20.0, 5.0, 0.0, 5.0],
        [30.0, 25.0, 5.0, 0.0],
    ],
    index=MINI_IDS,
    columns=MINI_IDS,
)
# 时间(min)：与距离同比例，1 km = 2 min
MINI_TIME = MINI_DIST * 2.0


def make_nodes(weights=(40.0, 40.0, 40.0), windows=((480, 700), (480, 700), (480, 700)),
               volumes=(0.1, 0.1, 0.1), orders=(1, 1, 1)):
    return pd.DataFrame(
        {
            "poi_id": ["PA", "PB", "PC"],
            "weight_kg": list(weights),
            "volume_m3": list(volumes),
            "window_start": [w[0] for w in windows],
            "window_end": [w[1] for w in windows],
            "n_orders": list(orders),
            "full_coverage": [True] * len(orders),
            "lng": [104.1, 104.2, 104.3],
            "lat": [30.6, 30.7, 30.8],
        }
    )


def make_day_orders(rows):
    """rows: (order_id, poi_id, weight, volume, w_start, w_end)"""
    return pd.DataFrame(
        [
            {
                "order_id": oid, "poi_id": pid, "weight_kg": w, "volume_m3": v,
                "window_start": ws, "window_end": we, "lng": 104.1, "lat": 30.6,
            }
            for oid, pid, w, v, ws, we in rows
        ]
    )


DEFAULT_GREEDY = dict(
    rated_payload_kg=100.0,
    rated_volume_l=18000.0,
    service_min=15.0,
    depot_open_min=480.0,
    max_route_min=600.0,
    max_vehicles=15,
)


class TestAggregateDemand:
    """需求聚合：重量体积求和、时间窗取最优点访窗、单点一访（ADR-0004 + 收紧规则）。"""

    def test_sums_weight_volume_and_narrows_window_to_the_common_intersection(self):
        day = make_day_orders(
            [
                ("O1", "PA", 10.0, 0.05, 500, 560),
                ("O2", "PA", 25.0, 0.15, 480, 620),
                ("O3", "PB", 5.0, 0.01, 600, 660),
            ]
        )
        nodes = TO.aggregate_demand(day)
        assert list(nodes["poi_id"]) == ["PA", "PB"]  # 单点一访
        pa = nodes.set_index("poi_id").loc["PA"]
        assert pa["weight_kg"] == pytest.approx(35.0)
        assert pa["volume_m3"] == pytest.approx(0.20)
        # 两窗有公共交集 [500,560]：到访落在其中可同时满足两张单，
        # 故节点窗取交集而非外包络 [480,620]（后者允许停在信封边缘、一单都不满足）。
        # 排线用窗止再减去延误缓冲，给在途异常留吸收空间；服务窗止原值保留在另一列
        buffer = C.TRANSPORT_DELAY_BUFFER_MIN
        assert (pa["window_start"], pa["window_end_service"]) == (500, 560)
        assert pa["window_end"] == 560 - buffer
        assert pa["window_covered_orders"] == 2 and pa["full_coverage"]
        pb = nodes.set_index("poi_id").loc["PB"]
        assert pb["n_orders"] == 1 and pb["weight_kg"] == pytest.approx(5.0)
        assert pb["full_coverage"]

    def test_conflicting_windows_fall_back_to_the_highest_coverage_window(self):
        # 三张单：两张窗重叠、一张完全错开 → 任何一次到访最多覆盖两张，必须显式标记
        day = make_day_orders(
            [
                ("O1", "PA", 1.0, 0.01, 500, 560),
                ("O2", "PA", 1.0, 0.01, 510, 570),
                ("O3", "PA", 1.0, 0.01, 900, 960),
            ]
        )
        pa = TO.aggregate_demand(day).set_index("poi_id").loc["PA"]
        assert pa["window_covered_orders"] == 2
        assert not bool(pa["full_coverage"])
        assert (pa["window_start"], pa["window_end_service"]) == (510, 560)
        assert pa["window_end"] == 560 - C.TRANSPORT_DELAY_BUFFER_MIN

    def test_single_order_keeps_its_own_window(self):
        day = make_day_orders([("O1", "PA", 1.0, 0.01, 500, 560)])
        nodes = TO.aggregate_demand(day)
        assert (nodes["window_start"].iloc[0], nodes["window_end_service"].iloc[0]) == (500, 560)
        assert nodes["window_end"].iloc[0] == 560 - C.TRANSPORT_DELAY_BUFFER_MIN
        assert bool(nodes["full_coverage"].iloc[0])


class TestBaselineGreedy:
    """基线最近邻贪心：手工小矩阵上手推路线必须一致。"""

    def test_route_sequence_and_distance_match_hand_calculation(self):
        routes = TO.nearest_neighbor_routes(make_nodes(), MINI_DIST.to_numpy(),
                                            MINI_TIME.to_numpy(), **DEFAULT_GREEDY)
        # 就近：DC→A(10)→B(5)；C 加进去会超载 120>100，故换车 DC→C(30)→DC
        assert [list(r["node_idx"]) for r in routes] == [[0, 1], [2]]
        assert routes[0]["distance_m"] == pytest.approx((10 + 5 + 20) * 1000)
        assert routes[1]["distance_m"] == pytest.approx((30 + 30) * 1000)

    def test_arrival_times_account_for_service_and_waiting(self):
        nodes = make_nodes(windows=((500, 700), (480, 700), (480, 700)))
        routes = TO.nearest_neighbor_routes(nodes, MINI_DIST.to_numpy(),
                                            MINI_TIME.to_numpy(), **DEFAULT_GREEDY)
        # DC→A 行 10km=20min，480+20=500 恰在窗起，无需等待
        assert routes[0]["arrival_min"][0] == pytest.approx(500.0)
        # A 服务 15min 后 515 出发，A→B 5km=10min → 525
        assert routes[0]["arrival_min"][1] == pytest.approx(525.0)

    def test_early_arrival_waits_until_window_start(self):
        nodes = make_nodes(windows=((560, 700), (480, 700), (480, 700)))
        routes = TO.nearest_neighbor_routes(nodes, MINI_DIST.to_numpy(),
                                            MINI_TIME.to_numpy(), **DEFAULT_GREEDY)
        assert routes[0]["arrival_min"][0] == pytest.approx(560.0)  # 等到窗起再卸

    def test_node_that_cannot_be_reached_in_window_is_not_served(self):
        # C 的窗在 08:0x 就关，DC→C 要 60 分钟（30km×2）→ 不可达
        nodes = make_nodes(windows=((480, 700), (480, 700), (480, 500)))
        routes = TO.nearest_neighbor_routes(nodes, MINI_DIST.to_numpy(),
                                            MINI_TIME.to_numpy(), **DEFAULT_GREEDY)
        served = {i for r in routes for i in r["node_idx"]}
        assert 2 not in served

    def test_volume_constraint_also_splits_the_route(self):
        # 重量都只有 1kg，但体积各 10m³、额定 18m³ → 一车只能装一个点
        nodes = make_nodes(weights=(1.0, 1.0, 1.0), volumes=(10.0, 10.0, 10.0))
        routes = TO.nearest_neighbor_routes(nodes, MINI_DIST.to_numpy(),
                                            MINI_TIME.to_numpy(), **DEFAULT_GREEDY)
        assert all(len(r["node_idx"]) == 1 for r in routes)
        assert len(routes) == 3

    def test_never_exceeds_vehicle_limit(self):
        nodes = make_nodes(weights=(60.0, 60.0, 60.0))
        routes = TO.nearest_neighbor_routes(nodes, MINI_DIST.to_numpy(), MINI_TIME.to_numpy(),
                                            **{**DEFAULT_GREEDY, "max_vehicles": 2})
        assert len(routes) <= 2


    def test_greedy_respects_max_route_min_including_waiting_time(self):
        # 早到等待必须算进「单车单日最长在途时长」：窗起很晚时，等完再回程会超限
        nodes = make_nodes(weights=(1.0, 1.0, 1.0),
                           windows=((900, 980), (900, 980), (900, 980)))
        routes = TO.nearest_neighbor_routes(nodes, MINI_DIST.to_numpy(), MINI_TIME.to_numpy(),
                                            **{**DEFAULT_GREEDY, "max_route_min": 300})
        for r in routes:
            assert r["duration_min"] <= 300 + 1e-6

    def test_served_index_helper_exposes_dropped_nodes(self):
        nodes = make_nodes(windows=((480, 700), (480, 700), (480, 500)))
        routes = TO.nearest_neighbor_routes(nodes, MINI_DIST.to_numpy(),
                                            MINI_TIME.to_numpy(), **DEFAULT_GREEDY)
        assert TO.served_node_indices(routes) == {0, 1}  # 2 号点够不到，必须显式暴露


class TestTimeWindowAchievement:
    """时间窗达成率：逐单判定，不被「外包络满足」蒙混（ADR-0004）。"""

    def test_orders_at_same_poi_are_judged_individually(self):
        nodes = make_nodes(orders=(2, 1, 1))
        # 到访时刻 510：订单 O1 的窗 [500,520] 达成，O2 的窗 [600,700] 未达成。
        # 外包络 [500,700] 被满足——若按外包络判定会误判为 100%。
        routes = [{"node_idx": [0, 1, 2], "arrival_min": [510.0, 600.0, 630.0]}]
        day = make_day_orders(
            [
                ("O1", "PA", 10.0, 0.05, 500, 520),
                ("O2", "PA", 10.0, 0.05, 600, 700),
                ("O3", "PB", 10.0, 0.05, 580, 620),
                ("O4", "PC", 10.0, 0.05, 600, 700),
            ]
        )
        got = TO.time_window_achievement(routes, nodes, day)
        assert got["n_orders"] == 4
        assert got["n_ontime"] == 3
        assert got["rate"] == pytest.approx(0.75)

    def test_boundaries_are_inclusive(self):
        nodes = make_nodes(orders=(1, 1, 1))
        routes = [{"node_idx": [0, 1, 2], "arrival_min": [500.0, 700.0, 630.0]}]
        day = make_day_orders([("O1", "PA", 1.0, 0.01, 500, 600),
                               ("O2", "PB", 1.0, 0.01, 600, 700),
                               ("O3", "PC", 1.0, 0.01, 600, 700)])
        got = TO.time_window_achievement(routes, nodes, day)
        assert got["n_ontime"] == 3

    def test_unserved_orders_count_into_the_denominator(self):
        nodes = make_nodes(orders=(1, 1, 1))
        routes = [{"node_idx": [0], "arrival_min": [520.0]}]  # PC 没被服务
        day = make_day_orders([("O1", "PA", 1.0, 0.01, 500, 600),
                               ("O2", "PC", 1.0, 0.01, 500, 600)])
        got = TO.time_window_achievement(routes, nodes, day)
        assert got["n_orders"] == 1 and got["n_unserved"] == 1
        assert got["rate"] == pytest.approx(1.0)  # 在已服务单中算达成率，未服务单独计数


class TestSingleVisitCeiling:
    """「单点一访」下的逐单达成率理论上限——用于区分「规则限制」与「目标函数不含准时」。"""

    def test_overlapping_windows_allow_multiple_orders_per_visit(self):
        nodes = make_nodes(orders=(2, 1, 1))
        day = make_day_orders([("O1", "PA", 1.0, 0.01, 500, 700),
                               ("O2", "PA", 1.0, 0.01, 550, 650)])
        got = TO.optimal_single_visit_ceiling(nodes, day)
        assert got["max_ontime_orders"] == 2
        assert got["ceiling_rate"] == pytest.approx(1.0)

    def test_disjoint_windows_cap_at_one_order_per_visit(self):
        nodes = make_nodes(orders=(2, 1, 1))
        day = make_day_orders([("O1", "PA", 1.0, 0.01, 500, 560),
                               ("O2", "PA", 1.0, 0.01, 600, 660)])
        got = TO.optimal_single_visit_ceiling(nodes, day)
        assert got["max_ontime_orders"] == 1  # 两个窗不重叠，一次到访只能命中一个
        assert got["ceiling_rate"] == pytest.approx(0.5)

    def test_ceiling_bounds_the_achieved_rate(self, produced):
        for plan in ("baseline", "optimized"):
            tw = produced["kpi"][plan]["time_window"]
            assert tw["rate"] <= tw["ceiling"]["ceiling_rate"] + 1e-9
            assert tw["gap_to_ceiling_pp"] >= 0


class TestLoadAndMileage:
    """满载率与里程利用率（需求 3.7）。"""

    @pytest.fixture(scope="class")
    def routes(self):
        return TO.nearest_neighbor_routes(make_nodes(), MINI_DIST.to_numpy(),
                                          MINI_TIME.to_numpy(), **DEFAULT_GREEDY)

    def test_load_rate_mean_and_distribution(self, routes):
        got = TO.load_rate_stats(routes, rated_payload_kg=100.0)
        # 两趟：80/100 = 0.8、40/100 = 0.4
        assert got["n_trips"] == 2
        assert got["mean"] == pytest.approx(0.6)
        assert got["min"] == pytest.approx(0.4) and got["max"] == pytest.approx(0.8)

    def test_mileage_utilization_excludes_the_empty_return_leg(self, routes):
        # 载货里程 = DC→A(10) + A→B(5) + DC→C(30) = 45；空驶 = B→DC(20) + C→DC(30) = 50
        got = TO.mileage_utilization(routes, MINI_DIST.to_numpy())
        assert got == pytest.approx(45 / 95)

    def test_total_cost_is_fixed_plus_mileage_per_trip(self, routes):
        got = TO.total_cost(routes, "diesel")
        per_km, fixed = C.cost_per_km("diesel"), C.cost_fixed_per_day("diesel")
        assert got == pytest.approx(2 * fixed + 95 * per_km)

    def test_ev_costs_less_per_km_than_diesel(self, routes):
        assert TO.total_cost(routes, "ev") < TO.total_cost(routes, "diesel")


class TestGeojson:
    def test_each_route_becomes_a_closed_linestring(self):
        routes = TO.nearest_neighbor_routes(make_nodes(), MINI_DIST.to_numpy(),
                                            MINI_TIME.to_numpy(), **DEFAULT_GREEDY)
        gj = TO.routes_to_geojson(routes, make_nodes(), (104.0, 30.5), {"plan": "baseline"})
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == 2
        f0 = gj["features"][0]
        assert f0["geometry"]["type"] == "LineString"
        coords = f0["geometry"]["coordinates"]
        assert coords[0] == [104.0, 30.5] and coords[-1] == [104.0, 30.5]  # 从 DC 出发返回
        assert f0["properties"]["plan"] == "baseline"
        assert len(f0["properties"]["poi_ids"]) == f0["properties"]["n_stops"]
        # 需求 45：地图要显示各点时间窗，故窗必须随路线带出
        p = f0["properties"]
        for key in ("window_start_min", "window_end_min", "arrival_min", "n_orders"):
            assert len(p[key]) == p["n_stops"], key
        assert all(s <= a <= e for s, a, e in
                   zip(p["window_start_min"], p["arrival_min"], p["window_end_min"]))


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    """真实路网 + 真实订单跑一次（OR-Tools 30s 时限），断言落在产物上。"""
    out = tmp_path_factory.mktemp("transport")
    return TO.run_all(out_dir=out)


class TestRealDataProducts:
    """数据产物缝：真实数据上验方向、结构与换算。"""

    def test_representative_day_is_the_busiest_weekday(self, produced):
        kpi = produced["kpi"]
        ctx = TO.load_transport_context()
        orders = ctx["orders"]
        day = pd.Timestamp(kpi["representative_day"])
        assert day.weekday() < 5
        counts = orders[orders["date"].dt.weekday < 5].groupby("date").size()
        assert counts.max() == kpi["representative_day_orders"] == counts.loc[day]

    def test_optimized_beats_baseline_on_distance_and_vehicles(self, produced):
        c = produced["comparison"].set_index("metric")
        assert c.loc["总里程 (km)", "optimized"] < c.loc["总里程 (km)", "baseline"]
        assert c.loc["总里程 (km)", "change_pct"] < 0
        assert c.loc["用车数 (台)", "optimized"] <= c.loc["用车数 (台)", "baseline"]

    def test_optimized_does_not_worsen_time_window_achievement(self, produced):
        c = produced["comparison"].set_index("metric")
        assert c.loc["时间窗达成率", "optimized"] >= c.loc["时间窗达成率", "baseline"] - 1e-9

    def test_all_days_baseline_reports_cost_and_windows_and_unserved(self, produced):
        # 票据要求「跑全 90 天记录总里程/用车数/总成本/时间窗达成率」——四项都要有
        ad = produced["kpi"]["baseline_all_days"]
        assert ad["days"] == C.SIM_DAYS
        assert ad["total_distance_km"] > 0 and ad["n_trips"] > 0
        assert ad["cost"]["diesel"] > 0 and ad["cost"]["ev"] > 0
        assert 0 < ad["time_window_rate"] <= 1
        # 未服务单必须显式计数，且不得被算成「已覆盖」
        assert ad["n_orders_served"] + ad["n_orders_unserved"] == ad["n_orders_total"]
        assert ad["n_orders_total"] == TO.load_transport_context()["orders"].shape[0]

    def test_all_baseline_routes_respect_capacity_and_windows(self, produced):
        nodes = produced["nodes"]
        rated_kg = float(TO.load_transport_context()["vehicles"]["rated_payload_kg"].iloc[0])
        rated_l = float(TO.load_transport_context()["vehicles"]["rated_volume_m3"].iloc[0]) * 1000
        for plan in ("baseline_routes", "optimized_routes"):
            for r in produced[plan]:
                assert r["load_kg"] <= rated_kg + 1e-6
                assert r["load_l"] <= rated_l + 1e-6
                for idx, arr in zip(r["node_idx"], r["arrival_min"]):
                    assert arr <= nodes.loc[idx, "window_end"] + 1e-6

    def test_every_served_node_is_visited_exactly_once(self, produced):
        seen = [i for r in produced["optimized_routes"] for i in r["node_idx"]]
        assert sorted(seen) == list(range(len(produced["nodes"])))

    def test_artifacts_written_and_geojson_is_valid(self, produced, tmp_path):
        out = Path(produced["report"]).parent
        for name in (C.TRANSPORT_KPI_JSON.name, C.TRANSPORT_KPI_MD.name,
                     C.TRANSPORT_COMPARISON_CSV.name, C.TRANSPORT_TRIPS_CSV.name,
                     C.TRANSPORT_SAVINGS_JSON.name,
                     C.TRANSPORT_ROUTES_BASELINE_GEOJSON.name,
                     C.TRANSPORT_ROUTES_OPTIMIZED_GEOJSON.name):
            assert (out / name).exists(), f"未落盘：{name}"
        for name in (C.TRANSPORT_ROUTES_BASELINE_GEOJSON.name,
                     C.TRANSPORT_ROUTES_OPTIMIZED_GEOJSON.name):
            gj = json.loads((out / name).read_text(encoding="utf-8"))
            assert gj["type"] == "FeatureCollection" and gj["features"]

    def test_savings_extrapolation_is_traceable(self, produced):
        s = produced["savings"]
        assert s["workdays_per_month"] == C.WORKDAYS_PER_MONTH
        d = s["daily_saving_by_mode"]["diesel"]
        assert d["per_month"] == pytest.approx(d["per_day"] * C.WORKDAYS_PER_MONTH, rel=1e-6)
        assert d["per_year"] == pytest.approx(d["per_day"] * C.WORKDAYS_PER_YEAR, rel=1e-6)
        assert "外推" in s["assumption"]

    def test_kpi_json_records_the_reproducibility_contract(self, produced):
        kpi = json.loads(Path(produced["report"]).parent.joinpath(C.TRANSPORT_KPI_JSON.name)
                         .read_text(encoding="utf-8"))
        assert kpi["matrix_source"] in {"amap", "osm"}
        assert kpi["solve"]["first_solution_strategy"] == "PARALLEL_CHEAPEST_INSERTION"
        assert kpi["baseline_all_days"]["days"] == C.SIM_DAYS
        assert kpi["baseline_all_days"]["n_orders_total"] == TO.load_transport_context()["orders"].shape[0]
        assert "车队规模" not in kpi["baseline_all_days"].get("unserved_note", "")


class TestDailyBaselineKpis:
    """逐日基线 KPI 表：驾驶舱趋势线的数据源，且必须与全量汇总同源。"""

    def _daily(self):
        return pd.DataFrame(
            [
                {"date": "2026-06-01", "n_orders_total": 10, "n_orders_served": 9,
                 "n_orders_unserved": 1, "n_trips": 2, "total_distance_km": 100.0,
                 "n_ontime": 6, "time_window_rate": 6 / 9, "load_rate_mean": 0.5,
                 "mileage_utilization": 0.6, "diesel_cost": 1000.0, "diesel_per_order": 100.0,
                 "ev_cost": 800.0, "ev_per_order": 80.0},
                {"date": "2026-06-02", "n_orders_total": 20, "n_orders_served": 20,
                 "n_orders_unserved": 0, "n_trips": 3, "total_distance_km": 150.0,
                 "n_ontime": 18, "time_window_rate": 0.9, "load_rate_mean": 0.7,
                 "mileage_utilization": 0.5, "diesel_cost": 2000.0, "diesel_per_order": 100.0,
                 "ev_cost": 1600.0, "ev_per_order": 80.0},
            ]
        )

    def test_summary_is_re_aggregated_from_the_daily_table(self):
        got = TO.summarize_all_days(self._daily())
        assert got["days"] == 2
        assert got["n_orders_total"] == 30
        assert got["n_orders_served"] == 29 and got["n_orders_unserved"] == 1
        assert got["days_with_unserved"] == 1
        assert got["n_trips"] == 5
        assert got["total_distance_km"] == pytest.approx(250.0)
        assert got["cost"]["diesel"] == pytest.approx(3000.0)
        # 全量达成率 = 全部按时单 / 全部**已服务**单，不是逐日率的平均（产物保留 4 位小数）
        assert got["time_window_rate"] == pytest.approx(round(24 / 29, 4))

    def test_daily_table_covers_every_day_and_reconciles_with_the_all_days_block(self, produced):
        daily = produced["daily"]
        assert len(daily) == C.SIM_DAYS
        assert daily["date"].is_monotonic_increasing
        assert daily["n_orders_total"].sum() == TO.load_transport_context()["orders"].shape[0]
        # 汇总必须由日表再聚合而来（同一份数据两个视图），不许两处各跑一遍循环
        assert TO.summarize_all_days(daily) == produced["kpi"]["baseline_all_days"]

    def test_published_all_days_reconciles_with_the_daily_artifact_on_disk(self):
        """对**磁盘上已发表的产物**断言，而不是同一次运行内部自比。

        自比是循环论证：上一次的 `summarize_all_days(daily) == kpi["baseline_all_days"]`
        两边都出自同一次 `run_all`，所以「日表先各自取整再求和 vs 整体求和后取整」这类
        偏差它**永远抓不到**——评审正是这样在已发表产物上量出 96,932.7 vs 96,932.6 km
        的 0.1 km 回归。这里改为读盘核对，日表与 KPI 一旦重新分叉就会立刻失败。
        """
        published = json.loads(C.TRANSPORT_KPI_JSON.read_text(encoding="utf-8"))["baseline_all_days"]
        daily = pd.read_csv(C.TRANSPORT_DAILY_CSV, encoding="utf-8-sig")
        got = TO.summarize_all_days(daily)
        for key in ("days", "total_distance_km", "n_trips", "n_orders_total",
                    "n_orders_served", "n_orders_unserved", "n_ontime",
                    "days_with_unserved", "time_window_rate"):
            assert got[key] == published[key], f"{key}：日表再聚合 {got[key]} vs 已发表 {published[key]}"
        assert got["cost"] == published["cost"]

    def test_daily_metrics_stay_in_range(self, produced):
        daily = produced["daily"]
        assert daily["time_window_rate"].between(0, 1).all()
        assert daily["load_rate_mean"].between(0, 1).all()
        assert daily["mileage_utilization"].between(0, 1).all()
        assert (daily["n_orders_served"] + daily["n_orders_unserved"]
                == daily["n_orders_total"]).all()

    def test_daily_only_refresh_leaves_the_representative_day_result_untouched(self, tmp_path):
        """逐日表与代表日的 30 秒启发式搜索无关；单独刷新它**不得**动到已发表的优化结果。

        若为补一张确定性日表而重跑 run_all，就会把那个「某一次搜索的解」一并换掉，
        连带 10/11 号票已发表的全部里程与成本作废。
        """
        before = (C.TRANSPORT_KPI_JSON.read_bytes(), C.TRANSPORT_KPI_JSON.stat().st_mtime_ns)
        path = TO.write_daily_only(tmp_path)
        after = (C.TRANSPORT_KPI_JSON.read_bytes(), C.TRANSPORT_KPI_JSON.stat().st_mtime_ns)
        assert path.exists() and path.name == C.TRANSPORT_DAILY_CSV.name
        assert before == after
