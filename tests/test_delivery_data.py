"""数据层 D（配送情景）单元测试。

测试缝（spec「Testing Decisions」预约定）：
  ① KPI 纯函数缝：泊松订单数、温度越界、满载率、时间窗、异常概率等纯函数，
     用手工构造小输入断言等于手算期望值；
  ② 数据产物缝（埋点回归）：读取落盘质检摘要，断言四埋点成立。
真实数据产物相关的断言集中在 TestDataProducts（依赖已生成产物）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import gen_delivery_data as DD


# 手工构造的最小 POI 表（3 个行政区 × 2 个 POI），用于片区划分断言
MINI_POI = pd.DataFrame(
    {
        "poi_id": ["A1", "A2", "B1", "B2", "C1", "C2"],
        "name": [f"店{i}" for i in range(6)],
        "lng": [104.00, 104.01, 104.20, 104.21, 104.10, 104.11],
        "lat": [30.60, 30.61, 30.70, 30.71, 30.65, 30.66],
        "district": ["甲区", "甲区", "乙区", "乙区", "丙区", "丙区"],
        "address": [f"路{i}号" for i in range(6)],
    }
)


class TestRegionAssignment:
    """片区划分：空间聚簇、划分规则落盘（需求 19）。"""

    def test_every_poi_gets_exactly_one_region_code(self):
        out, _ = DD.assign_regions(MINI_POI)
        assert len(out) == len(MINI_POI)
        assert set(out["region"]).issubset(set(C.REGION_CODES))
        assert out["region"].notna().all()

    def test_same_district_stays_in_same_region(self):
        # 同一空间单元不得被拆碎（实测无约束聚簇会产生 1 个 POI 的碎簇，故按行政区聚合）
        out, _ = DD.assign_regions(MINI_POI)
        n_regions = out.groupby("district")["region"].nunique()
        assert (n_regions == 1).all()

    def test_region_numbering_follows_centroid_longitude(self):
        # 由西向东编号：甲区(104.005) < 丙区(104.105) < 乙区(104.205)
        out, rule = DD.assign_regions(MINI_POI)
        r = dict(zip(out["district"], out["region"]))
        assert r["甲区"] == "R01"
        assert r["丙区"] == "R02"
        assert r["乙区"] == "R03"
        assert rule["method"]["label_rule"].startswith("质心经度升序")

    def test_rule_dict_is_self_describing(self):
        # 划分规则必须随代码落盘（需求 19「划分规则落盘并登记台账」）
        _, rule = DD.assign_regions(MINI_POI)
        assert rule["method"]["unit"] == "行政区（空间连续单元）"
        # 必须记录实测排除的无约束聚簇方案及其弃用理由（供复现者复核）
        rejected = rule["method"]["rejected_alternative"]
        assert "空间聚簇" in rejected and "弃用" in rejected
        assert rule["regions"]["R01"]["district"] == "甲区"
        assert rule["regions"]["R01"]["n_poi"] == 2
        assert rule["regions"]["R01"]["centroid_lng"] == pytest.approx(104.005)


class TestOrderDemandSampling:
    """配送订单生成：泊松到达、Olist 形状抽样、时间窗（需求 19）。"""

    def test_daily_counts_are_poisson(self):
        # 泊松分布的行为特征：均值 = 方差 = lambda，且恒为非负整数
        rng = np.random.default_rng(7)
        lam = np.full(4000, 2.0)
        counts = DD.sample_daily_counts(rng, lam)
        assert (counts >= 0).all() and np.issubdtype(counts.dtype, np.integer)
        assert counts.mean() == pytest.approx(2.0, abs=0.06)
        assert counts.var() == pytest.approx(2.0, abs=0.15)

    def test_store_lambdas_stay_in_configured_range(self):
        rng = np.random.default_rng(7)
        lam = DD.sample_store_lambdas(rng, 500)
        lo, hi = C.ORDER_POISSON_MEAN_RANGE
        assert lam.min() >= lo and lam.max() <= hi

    def test_demand_median_matches_scenario_anchor(self):
        # 尺度锚定：中位数 = 单批次冷链补货量锚点（config）
        rng = np.random.default_rng(7)
        w = DD.sample_demand(rng, 200_000, C.ORDER_WEIGHT_MEDIAN_KG, DD.weight_shape_sigma())
        assert (w > 0).all()
        assert np.median(w) == pytest.approx(C.ORDER_WEIGHT_MEDIAN_KG, rel=0.03)

    def test_demand_borrows_olist_lognormal_shape(self):
        # 形状借自 Olist：P95/P50 应等于对数正态 σ 的理论倍数 exp(1.645σ)
        rng = np.random.default_rng(11)
        sigma = DD.weight_shape_sigma()
        w = DD.sample_demand(rng, 400_000, C.ORDER_WEIGHT_MEDIAN_KG, sigma)
        expected_ratio = np.exp(1.6449 * sigma)
        assert np.quantile(w, 0.95) / np.median(w) == pytest.approx(expected_ratio, rel=0.05)

    def test_volume_uses_its_own_shape(self):
        rng = np.random.default_rng(13)
        v = DD.sample_demand(rng, 200_000, C.ORDER_VOLUME_MEDIAN_M3, DD.volume_shape_sigma())
        assert np.median(v) == pytest.approx(C.ORDER_VOLUME_MEDIAN_M3, rel=0.03)

    def test_receiving_waves_spread_over_the_configured_range(self):
        rng = np.random.default_rng(7)
        waves = DD.make_receiving_waves(rng, 20_000)
        lo, hi = C.RECEIVING_WAVE_START_RANGE
        assert waves.min() >= lo and waves.max() <= hi
        # 波次应铺满可用区间，而不是挤在一处
        assert waves.max() - waves.min() > (hi - lo) * 0.9

    def test_time_windows_are_anchored_on_the_store_wave(self):
        rng = np.random.default_rng(7)
        waves = np.full(20_000, 600.0)  # 所有单同属一波（同店当日）
        start, end = DD.make_time_windows(rng, waves)
        open_min = C.TIME_WINDOW_OPEN[0] * 60
        close_min = C.TIME_WINDOW_OPEN[1] * 60
        dur = C.DEFAULT_TIME_WINDOW_HOURS * 60
        assert start.min() >= waves.min()
        assert start.max() <= waves.max() + C.RECEIVING_WAVE_JITTER_MIN
        assert end.max() <= close_min and start.min() >= open_min
        assert np.allclose(end - start, dur)

    def test_same_store_orders_share_a_common_window(self):
        # ADR-0004「单点一访 + 逐单判定」自洽的前提：同店各单的窗必须高度重叠
        poi, _ = DD.assign_regions(MINI_POI)
        orders = DD.generate_orders(poi, days=20, seed=C.SEED_DELIVERY)
        grp = orders.groupby(["date", "poi_id"])
        multi = grp.filter(lambda g: len(g) > 1)
        assert len(multi) > 0
        for _, g in multi.groupby(["date", "poi_id"]):
            # 所有单的窗有公共交集（交集非空即意味着一次到访可全部命中）
            assert g["window_start"].max() <= g["window_end"].min()


@pytest.fixture(scope="module")
def orders():
    poi, _ = DD.assign_regions(MINI_POI)
    return DD.generate_orders(poi, days=10, seed=C.SEED_DELIVERY)


class TestGenerateOrders:
    """90 天订单表的整体行为（需求 19）。"""

    def test_one_row_per_delivery_with_full_fields(self, orders):
        assert list(orders.columns) == [
            "order_id", "date", "poi_id", "poi_name", "region", "lng", "lat",
            "weight_kg", "volume_m3", "window_start", "window_end", "priority",
        ]
        assert orders["order_id"].is_unique
        assert (orders["weight_kg"] > 0).all() and (orders["volume_m3"] > 0).all()

    def test_covers_every_poi_and_every_day(self, orders):
        assert orders["date"].nunique() == 10
        assert set(orders["poi_id"]) == set(MINI_POI["poi_id"])
        # 坐标取自 POI 表（真实观测），未被扰动
        merged = orders.merge(MINI_POI, on="poi_id", suffixes=("", "_src"))
        assert np.allclose(merged["lng"], merged["lng_src"])
        assert np.allclose(merged["lat"], merged["lat_src"])

    def test_per_store_daily_mean_within_configured_range(self, orders):
        # 门店 λ 均匀落在 [1,3]，样本均值应围绕区间中点（小样本 + 泊松噪声，放到 0.6）
        per_store_day = orders.groupby(["poi_id", "date"]).size()
        lo, hi = C.ORDER_POISSON_MEAN_RANGE
        assert per_store_day.mean() == pytest.approx((lo + hi) / 2, abs=0.6)

    def test_same_seed_reproduces_identical_table(self):
        poi, _ = DD.assign_regions(MINI_POI)
        a = DD.generate_orders(poi, days=5, seed=C.SEED_DELIVERY)
        b = DD.generate_orders(poi, days=5, seed=C.SEED_DELIVERY)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_changes_table(self):
        poi, _ = DD.assign_regions(MINI_POI)
        a = DD.generate_orders(poi, days=5, seed=C.SEED_DELIVERY)
        b = DD.generate_orders(poi, days=5, seed=C.SEED_DELIVERY + 1)
        assert not a["weight_kg"].equals(b["weight_kg"])


class TestFleet:
    """车辆表：15 台冷链轻卡，参数一律读 config（需求 21）。"""

    def test_fleet_size_plates_and_config_costs(self):
        fleet = DD.build_fleet()
        assert len(fleet) == C.FLEET_SIZE
        assert fleet["vehicle_id"].is_unique
        assert fleet["plate"].is_unique
        assert (fleet["rated_payload_kg"] == C.RATED_PAYLOAD_T * 1000).all()
        assert (fleet["rated_volume_m3"] == C.RATED_VOLUME_M3).all()
        for mode in ("diesel", "ev"):
            sub = fleet[fleet["mode"] == mode]
            assert (sub["fixed_cost_per_day"] == C.cost_fixed_per_day(mode)).all()
            assert (sub["cost_per_km"] == C.cost_per_km(mode)).all()

    def test_mode_mix_matches_config(self):
        fleet = DD.build_fleet()
        assert fleet["mode"].value_counts().to_dict() == C.FLEET_MODE_MIX

    def test_plates_are_sichuan_plates(self):
        fleet = DD.build_fleet()
        assert fleet["plate"].str.startswith("川A").all()
        assert fleet["plate"].str.len().max() == 7  # 川A + 5 位


class TestLoadRate:
    """满载率 / 容积利用率 / 超载判定（需求 37）。"""

    def test_load_rate_is_hand_computed_weight_ratio(self):
        assert DD.load_rate(350.0, 1400.0) == pytest.approx(0.25)
        assert DD.load_rate(0.0, 1400.0) == pytest.approx(0.0)

    def test_volume_rate_is_hand_computed_volume_ratio(self):
        assert DD.volume_rate(4.5, 18.0) == pytest.approx(0.25)

    def test_exceeds_capacity_on_either_dimension(self):
        assert DD.exceeds_capacity(1400.1, 1.0, 1400.0, 18.0) is True
        assert DD.exceeds_capacity(1.0, 18.1, 1400.0, 18.0) is True
        assert DD.exceeds_capacity(1400.0, 18.0, 1400.0, 18.0) is False


class TestHuolalaCost:
    """货拉拉外包对标计价（需求 21/22；参数读 config，台账登记为情景假设）。"""

    def test_short_trip_is_just_the_start_fee(self):
        # 起步 90 元含前 5km，8 点以内无超点费
        assert DD.huolala_cost(5.0, 8) == pytest.approx(C.HUOLALA_START_FEE)

    def test_medium_trip_charges_first_band(self):
        # 25km / 8 点 = 90 + (25-5)×5 = 190
        assert DD.huolala_cost(25.0, 8) == pytest.approx(190.0)

    def test_long_trip_adds_second_band_and_extra_stops(self):
        # 60km / 12 点 = 90 + 20×5 + 35×avg(3.5,4.8) + 4×38
        expected = 90.0 + 20 * 5.0 + 35 * 4.15 + 4 * 38.0
        assert DD.huolala_cost(60.0, 12) == pytest.approx(expected)


class TestTcoAnalysis:
    """双模式 TCO 曲线、盈亏平衡里程、三档敏感性、外包对照与建议（需求 21/22）。"""

    def test_curve_is_fixed_plus_mileage_times_variable(self):
        tco = DD.build_tco_analysis()
        for mode in ("diesel", "ev"):
            c = tco["curves"][mode]
            for km, cost in zip(c["mileage_km"], c["daily_total_cost"]):
                assert cost == pytest.approx(c["daily_fixed"] + km * c["per_km"], abs=0.01)

    def test_curve_covers_configured_mileage_grid(self):
        tco = DD.build_tco_analysis()
        assert tuple(tco["curves"]["diesel"]["mileage_km"]) == C.TCO_MILEAGE_GRID_KM

    def test_breakeven_matches_config_formula(self):
        tco = DD.build_tco_analysis()
        assert tco["breakeven_km"]["km"] == pytest.approx(C.breakeven_km(), abs=0.01)

    def test_breakeven_really_is_the_cost_crossover(self):
        # 报告的盈亏平衡里程必须真的是两条成本曲线的交点（自洽性）
        tco = DD.build_tco_analysis()
        be = tco["breakeven_km"]["km"]
        d, e = tco["curves"]["diesel"], tco["curves"]["ev"]
        at = lambda c, km: c["daily_fixed"] + km * c["per_km"]
        assert at(e, be + 20) < at(d, be + 20)  # 高于盈亏平衡：纯电更省
        assert at(d, be - 20) < at(e, be - 20)  # 低于盈亏平衡：柴油更省

    def test_sensitivity_covers_three_params_three_levels(self):
        tco = DD.build_tco_analysis()
        for param in C.SENSITIVITY_PARAMS:
            assert set(tco["sensitivity"][param]) == set(C.SENSITIVITY_LEVELS)
            for rec in tco["sensitivity"][param].values():
                assert rec["diesel_daily_cost"] > 0 and rec["ev_daily_cost"] > 0
                assert np.isfinite(rec["breakeven_km"])

    def test_sensitivity_moves_costs_in_the_right_direction(self):
        tco = DD.build_tco_analysis()
        s = tco["sensitivity"]
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

    def test_driver_wage_leaves_breakeven_unchanged(self):
        # 司机工资对两模式同额、相减抵消（config.breakeven_km 的性质），敏感性表须如实反映
        tco = DD.build_tco_analysis()
        be = [tco["sensitivity"]["driver_wage"][lv]["breakeven_km"] for lv in C.SENSITIVITY_LEVELS]
        assert be[0] == pytest.approx(be[1]) == pytest.approx(be[2])
        # 工资却实实在在推高两模式的日总成本
        costs = [tco["sensitivity"]["driver_wage"][lv]["diesel_daily_cost"] for lv in C.SENSITIVITY_LEVELS]
        assert costs[0] < costs[1] < costs[2]

    def test_comparison_table_has_a_verdict_per_profile(self):
        tco = DD.build_tco_analysis()
        for r in tco["huolala_comparison"]:
            assert r["verdict"] in {"自营更省", "外包更省", "基本持平"}
            assert r["huolala_cost"] > 0 and r["self_diesel_cost"] > 0 and r["self_ev_cost"] > 0

    def test_comparison_table_discriminates(self):
        # 对照表必须能区分两种结论，否则没有决策价值
        tco = DD.build_tco_analysis()
        verdicts = {r["verdict"] for r in tco["huolala_comparison"]}
        assert len(verdicts) >= 2
        assert {r["verdict"] for r in tco["huolala_comparison"]} <= {"自营更省", "外包更省", "基本持平"}

    def test_recommendation_cites_numbers_from_its_own_output(self):
        tco = DD.build_tco_analysis()
        rec = tco["recommendation"]
        assert rec["recommended_mode"] in {"ev", "diesel"}
        assert rec["breakeven_km"] == pytest.approx(tco["breakeven_km"]["km"], abs=0.01)
        assert rec["rationale"]


class TestAnomalyModel:
    """在途异常概率与类型构成（需求 23）。"""

    BASE = 0.0813

    def test_type_shares_sum_to_one(self):
        shares = DD.anomaly_type_shares(False, False)
        assert set(shares) == {"拥堵", "晚点", "温控波动", "故障"}
        assert sum(shares.values()) == pytest.approx(1.0)

    def test_region_weight_multiplies_only_the_high_region(self):
        w = DD.anomaly_weights(["R01", C.HIGH_ANOMALY_REGION], [False, False])
        assert w[0] == pytest.approx(1.0)
        assert w[1] == pytest.approx(C.ANOMALY_REGION_MULTIPLIER)

    def test_friday_weight_multiplies_overall(self):
        w = DD.anomaly_weights(["R01", "R01"], [False, True])
        assert w[1] == pytest.approx(w[0] * C.ANOMALY_FRIDAY_OVERALL_MULTIPLIER)

    def test_probabilities_are_normalised_to_the_olist_rate(self):
        # 全票最关键的标定承诺：一批运单的概率均值必须等于 Olist 实测延迟率
        regions = np.array(["R01", "R02", "R03", "R01", "R04", "R05"] + [C.HIGH_ANOMALY_REGION] * 3)
        friday = np.array([False] * 6 + [False, False, True])
        probs = DD.anomaly_probabilities(regions, friday, self.BASE)
        assert probs.mean() == pytest.approx(self.BASE)
        # 高发片区、周五的运单确实分到更高的概率；其余运单相应被压低
        assert probs[6] > probs[0]
        assert probs[8] > probs[6]
        assert max(probs[:6]) < min(probs[6:])

    def test_friday_and_rain_shift_the_mix_towards_late(self):
        base = DD.anomaly_type_shares(False, False)["晚点"]
        fri = DD.anomaly_type_shares(True, False)["晚点"]
        rain = DD.anomaly_type_shares(False, True)["晚点"]
        both = DD.anomaly_type_shares(True, True)["晚点"]
        assert fri > base and rain > base
        assert both > fri  # 两个倍率叠加
        # 非「晚点」类型的份额相应被摊薄，总和仍为 1
        assert sum(DD.anomaly_type_shares(True, True).values()) == pytest.approx(1.0)
        assert DD.anomaly_type_shares(True, True)["故障"] < DD.anomaly_type_shares(False, False)["故障"]


class TestColdChainTemperature:
    """车厢温度越界判定与温控达标率（需求 20/37）。"""

    def test_range_is_closed_interval_from_config(self):
        lo, hi = C.COLD_CHAIN_TEMP_RANGE
        assert DD.in_cold_chain_range(lo) is True
        assert DD.in_cold_chain_range(hi) is True
        assert DD.in_cold_chain_range(lo - 0.1) is False
        assert DD.in_cold_chain_range(hi + 0.1) is False

    def test_compliance_rate_is_hand_computed(self):
        # 3 个达标 / 4 个采样
        assert DD.temp_compliance_rate([3.0, 5.0, 7.0, 9.0]) == pytest.approx(0.75)

    def test_compliance_rate_all_in_range(self):
        assert DD.temp_compliance_rate([2.0, 5.0, 8.0]) == pytest.approx(1.0)


@pytest.fixture(scope="module")
def fleet():
    return DD.build_fleet()


class TestVehicleLoad:
    """车辆装载率与「长期低满载」识别（需求 23 埋点④ / 需求 37 满载率）。"""

    @staticmethod
    def _orders(vehicle_ids, weights, volumes):
        return pd.DataFrame(
            {
                "vehicle_id": vehicle_ids,
                "date": pd.to_datetime(["2026-06-01"] * len(vehicle_ids)),
                "weight_kg": weights,
                "volume_m3": volumes,
            }
        )

    def test_daily_load_rate_is_hand_computed(self, fleet):
        orders = self._orders(["V01", "V01", "V02"], [400.0, 200.0, 700.0], [2.0, 1.0, 3.0])
        out = DD.vehicle_daily_load(orders, fleet)
        v01 = out[out["vehicle_id"] == "V01"].iloc[0]
        assert v01["load_kg"] == pytest.approx(600.0)
        assert v01["load_rate"] == pytest.approx(600.0 / (C.RATED_PAYLOAD_T * 1000))
        assert v01["volume_rate"] == pytest.approx(3.0 / C.RATED_VOLUME_M3)

    def test_one_row_per_vehicle_day(self, fleet):
        orders = self._orders(["V01", "V02", "V01"], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
        out = DD.vehicle_daily_load(orders, fleet)
        assert len(out) == 2  # V01 两单合并为一行 + V02 一行

    def test_summary_flags_chronically_underloaded_vehicles(self, fleet):
        # A 车连日低载、B 车连日高载 → 只有 A 被认定为「长期低满载」
        load = pd.DataFrame(
            {
                "vehicle_id": ["V01"] * 3 + ["V02"] * 3,
                "date": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-03"] * 2),
                "load_kg": [100.0] * 3 + [1000.0] * 3,
                "load_rate": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
            }
        )
        summary = DD.summarize_vehicle_load(load, threshold=0.35)
        assert summary.set_index("vehicle_id").loc["V01", "is_low_load"]
        assert not summary.set_index("vehicle_id").loc["V02", "is_low_load"]
        v01 = summary.set_index("vehicle_id").loc["V01"]
        assert v01["mean_load_rate"] == pytest.approx(0.2)
        assert v01["days_below_threshold"] == 3


#: 手工构造的最小路网（DC + 两个 POI），用于轨迹/异常端到端断言
MINI_DIST = pd.DataFrame(
    {"DC": [0.0, 10.0, 20.0], "PA": [10.0, 0.0, 12.0], "PB": [20.0, 12.0, 0.0]},
    index=["DC", "PA", "PB"],
)
MINI_TIME = pd.DataFrame(
    {"DC": [0.0, 30.0, 50.0], "PA": [30.0, 0.0, 35.0], "PB": [50.0, 35.0, 0.0]},
    index=["DC", "PA", "PB"],
)
MINI_NET_POI = pd.DataFrame(
    {
        "poi_id": ["PA", "PB"],
        "name": ["甲店", "乙店"],
        "lng": [104.10, 104.20],
        "lat": [30.60, 30.70],
        "district": ["甲区", "乙区"],
        "address": ["路1", "路2"],
    }
)
MINI_DC_COORD = (104.00, 30.50)


def _produce_net_tables(days=90):
    """用同一批输入重建轨迹/异常（供复现性与一致性断言共用）。"""
    poi, _ = DD.assign_regions(MINI_NET_POI)
    orders = DD.generate_orders(poi, days=days, seed=C.SEED_DELIVERY)
    return DD.generate_tracking_and_anomalies(
        orders, DD.build_fleet(), MINI_DIST, MINI_TIME, MINI_DC_COORD, seed=C.SEED_DELIVERY
    )


@pytest.fixture(scope="module")
def produced():
    return _produce_net_tables()


class TestTrackingAndAnomalies:
    """在途轨迹与异常表端到端（需求 23）。"""

    def test_one_anomaly_row_per_order(self, produced):
        _, anomalies = produced
        assert anomalies["order_id"].is_unique
        assert set(anomalies["anomaly_type"]) <= {"拥堵", "晚点", "温控波动", "故障", "无异常"}

    def test_tracking_covers_every_order_with_plausible_ranges(self, produced):
        tracking, anomalies = produced
        assert set(tracking["order_id"]) == set(anomalies["order_id"])
        assert tracking["speed_kmh"].min() >= 0
        assert tracking["cabin_temp_c"].between(-5, 20).all()
        # 轨迹首点应贴近 DC、末点应贴近该单送达点
        assert tracking["sample_seq"].min() == 0
        assert (tracking.groupby("order_id")["sample_seq"].min() == 0).all()

    def test_sampling_respects_configured_interval(self, produced):
        tracking, _ = produced
        ts = tracking.sort_values(["order_id", "sample_seq"]).groupby("order_id")["ts_min"]
        gaps = ts.diff().dropna()
        assert (gaps > 0).all()
        # 末点严格落在到店时刻，故最后一跳可能短于采样间隔；不得超出间隔（容差=ts_min 舍入位）
        assert gaps.max() <= C.TRACKING_SAMPLE_MINUTES + 0.01

    def test_anomalous_shipments_carry_delay_and_handling_time(self, produced):
        _, anomalies = produced
        anom = anomalies[anomalies["anomaly_type"] != "无异常"]
        assert len(anom) > 0
        assert (anom["handling_min"] > 0).all()
        # 非温控波动类异常必然造成送达延误；温控波动未必
        transport_anom = anom[anom["anomaly_type"] != "温控波动"]
        assert (transport_anom["delay_min"] > 0).all()
        assert (anomalies.loc[anomalies["anomaly_type"] == "无异常", "delay_min"] == 0).all()

    def test_thermal_fluctuation_shipments_break_the_cold_chain_range(self, produced):
        tracking, anomalies = produced
        fluct = anomalies.loc[anomalies["anomaly_type"] == "温控波动", "order_id"]
        assert len(fluct) > 0, "样本量应足以出现温控波动异常"
        sub = tracking[tracking["order_id"].isin(set(fluct))]
        lo, hi = C.COLD_CHAIN_TEMP_RANGE
        assert ((sub["cabin_temp_c"] < lo) | (sub["cabin_temp_c"] > hi)).any()

    def test_same_seed_reproduces_both_tables(self, produced):
        tracking, anomalies = produced
        again = _produce_net_tables()
        pd.testing.assert_frame_equal(tracking, again[0])
        pd.testing.assert_frame_equal(anomalies, again[1])

    def test_late_flag_is_consistent_with_arrival_and_window(self, produced):
        _, anomalies = produced
        expected = anomalies["arrival_time"] <= anomalies["window_end"]
        assert (anomalies["on_time"] == expected).all()


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    """用真实 POI/路网跑一次完整数据层 D，断言落在落盘质检摘要上（数据产物缝）。"""
    out = tmp_path_factory.mktemp("delivery")
    return DD.generate_delivery_scenario(out_dir=out, qc_dir=out / "qc")


class TestDataProducts:
    """读取落盘质检摘要断言四埋点成立（spec「数据产物缝（埋点回归）」）。"""

    def test_all_artifacts_written(self, scenario):
        for key, path in scenario["paths"].items():
            assert Path(path).exists(), f"{key} 未落盘：{path}"
        for key in ("qc_json", "qc_md", "tco_json", "tco_md"):
            assert Path(scenario[key]).exists(), f"{key} 未落盘"

    def test_date_range_is_ninety_days(self, scenario):
        assert scenario["qc"]["date_range"]["days"] == C.SIM_DAYS
        assert scenario["qc"]["date_range"]["start"] == C.SIM_START_DATE

    def test_every_poi_belongs_to_exactly_one_of_eight_regions(self, scenario):
        regions = scenario["qc"]["regions"]
        assert regions["n_regions"] == len(C.REGION_CODES)
        assert sum(regions["poi_per_region"].values()) == len(DD.load_poi_table())

    def test_weight_median_matches_scenario_anchor(self, scenario):
        w = scenario["qc"]["order_stats"]["weight_kg"]
        assert w["median"] == pytest.approx(C.ORDER_WEIGHT_MEDIAN_KG, rel=0.10)
        assert w["mean"] > w["median"]  # 长尾：均值显著高于中位

    def test_embedding1_high_anomaly_region(self, scenario):
        e = scenario["qc"]["embedding_checks"]["region_anomaly_rate"]
        assert e["high_region"] == C.HIGH_ANOMALY_REGION
        assert e["ratio"] > 2.0
        assert e["significant_at_5pct"], f"片区异常率差异不显著：z={e['z']}"
        assert e["per_region"][C.HIGH_ANOMALY_REGION] == max(e["per_region"].values())

    def test_embedding2_friday_late_rate(self, scenario):
        e = scenario["qc"]["embedding_checks"]["friday_late_rate"]
        assert e["ratio"] > 1.5
        assert e["significant_at_5pct"], f"周五晚点率差异不显著：z={e['z']}"

    def test_embedding3_rainy_late_rate(self, scenario):
        e = scenario["qc"]["embedding_checks"]["rainy_late_rate"]
        assert e["ratio"] > 1.2
        assert e["significant_at_5pct"], f"雨日晚点率差异不显著：z={e['z']}"
        assert 0.0 < e["rainy_day_share"] < 1.0

    def test_embedding4_some_vehicles_chronically_underloaded(self, scenario):
        e = scenario["qc"]["embedding_checks"]["vehicle_low_load"]
        assert e["n_low_load"] >= 2
        assert e["low_load_mean_rate"] < e["threshold"] < e["other_mean_rate"]

    def test_overall_anomaly_rate_is_calibrated_to_olist(self, scenario):
        # 片区/周五倍率经归一后不得抬高全局水位——总体异常率必须锚定 Olist 实测延迟率
        a = scenario["qc"]["anomaly"]
        assert a["overall_rate"] == pytest.approx(a["olist_baseline_rate"], abs=0.01)

    def test_anomaly_types_and_handling_times_present(self, scenario):
        a = scenario["qc"]["anomaly"]
        for t in ("拥堵", "晚点", "温控波动", "故障"):
            assert t in a["by_type"]
            assert a["mean_handling_min_by_type"][t] > 0

    def test_cold_chain_compliance_is_plausible(self, scenario):
        c = scenario["qc"]["embedding_checks"]["cold_chain"]
        assert c["range_c"] == list(C.COLD_CHAIN_TEMP_RANGE)
        assert 0.80 < c["compliance_rate"] < 0.99
        assert c["n_out_of_range"] > 0  # 必须存在越界采样，否则温控达标率无诊断价值

    def test_no_integrity_violations(self, scenario):
        assert all(v == 0 for v in scenario["qc"]["integrity"].values())

    def test_region_rule_is_written_and_self_describing(self, scenario):
        rule = json.loads(Path(scenario["paths"]["region_rule"]).read_text(encoding="utf-8"))
        assert rule["n_regions"] == len(C.REGION_CODES)
        assert len(rule["poi_to_region"]) == len(DD.load_poi_table())
        assert rule["method"]["label_rule"].startswith("质心经度升序")

    def test_region_rule_records_measured_clustering_diagnosis(self, scenario):
        # 「无约束聚簇会产生碎簇」必须是可复现的实测结论，不是事后叙述
        rule = json.loads(Path(scenario["paths"]["region_rule"]).read_text(encoding="utf-8"))
        diag = rule["unconstrained_clustering_diagnosis"]
        assert diag["k"] == len(C.REGION_CODES)
        assert min(diag["n_below_min_size"].values()) >= 1
        for sizes in diag["cluster_sizes"].values():
            assert len(sizes) == len(C.REGION_CODES)
            assert sum(sizes) == len(DD.load_poi_table())

    def test_per_region_rate_tracks_the_shared_probability_model(self, scenario):
        # 抽样必须走 anomaly_probabilities 这一份模型：落盘片区异常率若系统性地偏离模型期望，
        # 说明概率模型被复制成了第二份并在抽样路径上走偏。用逐片区 z 值判定（而非固定容差），
        # 才能把「模型走偏」与「抽样涨落」区分开。
        e = scenario["qc"]["embedding_checks"]["region_anomaly_rate"]
        assert e["max_abs_deviation_z"] < 4.0
        for region, predicted in e["model_predicted_rate"].items():
            assert e["per_region"][region] == pytest.approx(predicted, abs=0.05)

    def test_tco_comparison_states_its_basis(self, scenario):
        # 对照表是「优化前」口径，必须显式声明，避免下游误当作优化后结果
        assert "优化前" in scenario["tco"]["recommendation"]["comparison_basis"]

    def test_fleet_and_tco_outputs_are_consistent(self, scenario):
        fleet = pd.read_csv(scenario["paths"]["vehicles"])
        assert len(fleet) == C.FLEET_SIZE
        tco = scenario["tco"]
        assert tco["breakeven_km"]["km"] == pytest.approx(C.breakeven_km(), abs=0.01)
        assert tco["recommendation"]["rationale"]

    def test_vehicle_share_weights_sum_to_one(self):
        assert sum(C.FLEET_ALLOC_WEIGHTS) == pytest.approx(1.0)
        assert len(C.FLEET_ALLOC_WEIGHTS) == C.FLEET_SIZE
