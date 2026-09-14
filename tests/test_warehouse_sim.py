"""数据层 F（warehouse_sim）单元测试。

测试缝：纯函数/小规模仿真——NHPP 到达分布特性、布局映射方向、ABC 抽样权重、
CI 聚合、子种子派生、权衡曲线拐点、仿真确定性与利用率合理域、校准对照结构。
全部用小参数（少订单/少重复）保证测试快。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import warehouse_sim as W


@pytest.fixture(scope="module")
def small_master() -> tuple[pd.DataFrame, pd.DataFrame]:
    """小规模 SKU/库位主数据（20 SKU × 40 库位），结构同数据层 A。"""
    rng = np.random.default_rng(7)
    n_sku = 20
    labels = ["A"] * 6 + ["B"] * 6 + ["C"] * 8
    sku = pd.DataFrame({
        "sku_id": [f"SKU{i:04d}" for i in range(1, n_sku + 1)],
        "abc_initial_label": labels,
    })
    rows = np.repeat(np.arange(1, 5), 10)
    cols = np.tile(np.arange(1, 11), 4)
    loc = pd.DataFrame({
        "loc_id": [f"L{i:04d}" for i in range(1, 41)],
        "row": rows, "col": cols, "zone": "Z1",
        "walk_dist_m": rows * 3.0 + cols * 1.5,
    })
    return sku, loc


@pytest.fixture(scope="module")
def small_assignment(small_master) -> pd.DataFrame:
    """合成一份「SKU → 库位」分配，按数据层 A 的**类级**规则排：A 类在最远端、C 类在最近端。

    真实的那份由数据层 A 落盘（`sku_location_assignment.csv`），仿真器读它来复刻原始布局——
    这里用合成小表代替，测的是「original 布局走交付的分配」这条路径本身。
    """
    sku, loc = small_master
    by_dist = loc.sort_values("walk_dist_m", ascending=False)["loc_id"].tolist()
    # A 从最远端取、C 从最近端取、B 取中间一段（互不重叠）
    pools = {"A": by_dist, "B": by_dist[6:], "C": by_dist[::-1]}
    cursor = {"A": 0, "B": 0, "C": 0}
    rows = []
    for sid, label in zip(sku["sku_id"], sku["abc_initial_label"]):
        rows.append({"sku_id": sid, "loc_id": pools[label][cursor[label]]})
        cursor[label] += 1
    return pd.DataFrame(rows)


class TestLayout:
    def test_original_reads_the_delivered_assignment(self, small_master, small_assignment):
        """original 布局必须**照交付的分配来**，而不是另算一套。

        它原先是按 ABC 标签近似复刻的（同类内按 sku_id 排序，而非按需求频率），逐 SKU 与
        真实布局并不相同；而实验一要拿这一臂去比对数据层 A 的 outbound（两条证据链互证），
        基线必须是同一份布局。
        """
        sku, loc = small_master
        lay = W.build_layout(sku, loc, "original", assignment=small_assignment)
        dist = dict(zip(loc["loc_id"], loc["walk_dist_m"]))
        expected = dict(zip(small_assignment["sku_id"],
                            small_assignment["loc_id"].map(dist)))
        assert lay == pytest.approx(expected)
        a_dists = [lay[s] for s, l in zip(sku["sku_id"], sku["abc_initial_label"]) if l == "A"]
        c_dists = [lay[s] for s, l in zip(sku["sku_id"], sku["abc_initial_label"]) if l == "C"]
        assert np.mean(a_dists) > np.mean(c_dists)  # 原始布局：A 类更远

    def test_original_without_assignment_raises(self, small_master):
        """缺交付时必须报错——静默退回近似复刻，等于把「互证」的基线悄悄换掉。"""
        sku, loc = small_master
        with pytest.raises(ValueError, match="交付的库位分配"):
            W.build_layout(sku, loc, "original")

    def test_original_rejects_loc_ids_not_in_the_master(self, small_master, small_assignment):
        sku, loc = small_master
        bad = small_assignment.copy()
        bad.loc[0, "loc_id"] = "L9999"
        with pytest.raises(ValueError, match="不在库位主数据"):
            W.build_layout(sku, loc, "original", assignment=bad)

    def test_abc_zoned_is_the_treatment_not_a_delivery(self, small_master, small_assignment):
        """ABC 分区是实验的**处理组**，由本模块构造，与交付的分配无关。"""
        sku, loc = small_master
        lay = W.build_layout(sku, loc, "abc_zoned")
        lay_with_assignment = W.build_layout(sku, loc, "abc_zoned",
                                             assignment=small_assignment)
        assert lay == lay_with_assignment
        a_dists = [lay[s] for s, l in zip(sku["sku_id"], sku["abc_initial_label"]) if l == "A"]
        c_dists = [lay[s] for s, l in zip(sku["sku_id"], sku["abc_initial_label"]) if l == "C"]
        assert np.mean(a_dists) < np.mean(c_dists)  # ABC 分区：A 类更近

    def test_unknown_mode_raises(self, small_master):
        sku, loc = small_master
        with pytest.raises(ValueError):
            W.build_layout(sku, loc, "random")


class TestSamplingWeights:
    def test_a_class_higher_weight(self, small_master):
        sku, _ = small_master
        ids, w = W.sku_sampling_weights(sku)
        wmap = dict(zip(ids, w))
        a_w = [wmap[s] for s, l in zip(sku["sku_id"], sku["abc_initial_label"]) if l == "A"]
        c_w = [wmap[s] for s, l in zip(sku["sku_id"], sku["abc_initial_label"]) if l == "C"]
        assert np.mean(a_w) > np.mean(c_w)
        assert w.sum() == pytest.approx(1.0)


class TestNHPP:
    def test_peak_hours_dominant(self):
        # 大量抽样下，高峰时段（9–11、14–16）到达数应显著多于平峰
        rng = np.random.default_rng(0)
        secs = W.sample_arrival_seconds(rng, 20000, peak_intensity=3.0)
        hours = C.SIM_PEAK_HOURS  # ((9,11),(14,16))
        # secs 为绝对秒轴（08:00=28800），//3600 即得时钟小时 8–18
        clock_hour = secs // 3600
        in_peak = sum(((clock_hour >= h0) & (clock_hour < h1)).sum() for h0, h1 in hours)
        # 4 个高峰小时承载份额应 > 4/10（平峰均摊份额）
        assert in_peak / len(secs) > 0.5  # 3× 强度下理论份额 ≈ 12/18 ≈ 0.67

    def test_arrivals_sorted_in_window(self):
        rng = np.random.default_rng(1)
        secs = W.sample_arrival_seconds(rng, 100)
        assert (np.diff(secs) >= 0).all()
        assert secs.min() >= 8 * 3600 and secs.max() < 18 * 3600


class TestSeedDerivation:
    def test_deterministic(self):
        assert W.derive_seed(45, "arm", 3) == W.derive_seed(45, "arm", 3)

    def test_different_parts_differ(self):
        assert W.derive_seed(45, "arm", 3) != W.derive_seed(45, "arm", 4)
        assert W.derive_seed(45, "arm", 3) != W.derive_seed(46, "arm", 3)

    def test_stable_across_processes(self):
        """回归（审计 2026-09-13）：builtin hash() 有 PYTHONHASHSEED 进程级随机化，
        曾使 derive_seed 跨进程漂移、产物无法复现。必须在真实跨进程缝上锁定：
        spawn 两个独立解释器（默认随机 hash seed），断言输出一致。"""
        import subprocess
        import sys
        root = Path(__file__).resolve().parent.parent
        code = "from src.warehouse_sim import derive_seed; print(derive_seed(45, 'arm', 3))"
        outs = set()
        for _ in range(2):
            r = subprocess.run(
                [sys.executable, "-c", code], cwd=root,
                capture_output=True, text=True, timeout=60,
            )
            assert r.returncode == 0, r.stderr
            outs.add(r.stdout.strip())
        assert len(outs) == 1, f"derive_seed 跨进程漂移: {outs}"


class TestRunOneSim:
    def test_metrics_sane_and_deterministic(self, small_master):
        sku, loc = small_master
        lay = W.build_layout(sku, loc, "abc_zoned")
        ids, w = W.sku_sampling_weights(sku)
        r1 = W.run_one_sim(lay, n_pickers=4, seed=123, n_orders=60, sku_ids=ids, sku_weights=w)
        r2 = W.run_one_sim(lay, n_pickers=4, seed=123, n_orders=60, sku_ids=ids, sku_weights=w)
        # 同种子完全一致（NaN 字段单独比对，因 NaN != NaN）
        for k in r1:
            if isinstance(r1[k], float) and np.isnan(r1[k]):
                assert np.isnan(r2[k])
            else:
                assert r1[k] == r2[k], k
        assert 0 < r1["picker_utilization"] <= 1.0
        assert r1["avg_fulfillment_sec"] > 0
        assert r1["line_pick_sec_mean"] > 0
        assert r1["total_walk_m"] > 0

    def test_abc_layout_reduces_walk(self, small_master, small_assignment):
        # 同一订单流下，ABC 布局总行走距离 < 原始布局
        sku, loc = small_master
        ids, w = W.sku_sampling_weights(sku)
        rng = np.random.default_rng(9)
        orders = W.pregenerate_orders(rng, 80, ids, w)
        lay_o = W.build_layout(sku, loc, "original", assignment=small_assignment)
        lay_a = W.build_layout(sku, loc, "abc_zoned")
        ro = W.run_one_sim(lay_o, 5, seed=7, orders=orders)
        ra = W.run_one_sim(lay_a, 5, seed=7, orders=orders)
        assert ra["total_walk_m"] < ro["total_walk_m"]
        assert ra["avg_order_pick_sec"] < ro["avg_order_pick_sec"]


class TestDeliveredAssignmentReconcilesWithOutbound:
    """交付的分配表必须与数据层 A 真正用过的布局一致。

    这是「互证」的地基：实验一的原始布局臂读的是这份表，而数据层 A 的 outbound 是按
    它生成出库行的。两者若不一致，所谓互证就是拿两份不同的布局在比。
    """

    def test_assignment_matches_the_layout_behind_outbound(self):
        from src import config as C
        if not C.WAREHOUSE_SKU_LOCATION_CSV.exists():
            pytest.skip("数据层 A 尚未生成（缺 sku_location_assignment.csv）")
        asg = W.load_sku_location_assignment().set_index("sku_id")["loc_id"]
        out = pd.read_csv(C.WAREHOUSE_TABLES["outbound"], encoding="utf-8-sig")
        used = out.groupby("sku_id")["loc_id"].agg(lambda s: s.mode().iloc[0])
        assert len(asg) == C.SKU_COUNT
        assert (asg.reindex(used.index) == used).all()

    def test_missing_assignment_names_the_command(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="python -m src.gen_warehouse_data"):
            W.load_sku_location_assignment(tmp_path / "absent.csv")


class TestArrivalIntensityIsSharedWithLayerA:
    """到达过程是两层**共用**的一个参数，不是各写一份。

    ADR-0001 写明「仿真的校准锚点是自家的仿真仓表——**到达过程**与 14–16 点低谷埋点相互
    印证」。互证只有在两边真的用同一个到达过程时才成立；先前两层各写一份且差一个数量级
    （层 A 1.8 / 层 F 10.0），而写着「与数据层 F 互证」的注释还留在原处。
    """

    def test_both_layers_compute_the_same_minute_distribution(self):
        from src import gen_warehouse_data as GD
        a = GD._minute_intensity()
        b = W.minute_intensity()
        assert len(a) == len(b)
        assert np.allclose(a, b, atol=0, rtol=0), "数据层 A 与 F 的到达分布必须逐元素相同"

    def test_peak_share_follows_the_configured_ratio(self):
        """高峰 4 小时承载的份额要由 `WAREHOUSE_PEAK_INTENSITY` 决定，不是各自拍脑袋。"""
        p = W.minute_intensity()
        minutes = np.arange(C.WAREHOUSE_OPEN_HOURS[0] * 60, C.WAREHOUSE_OPEN_HOURS[1] * 60)
        hours = minutes // 60  # 分钟轴上的时钟小时（8–17）
        peak = sum(p[(hours >= h0) & (hours < h1)].sum() for h0, h1 in C.SIM_PEAK_HOURS)
        # 6 平峰小时 × 1 + 4 高峰小时 × r，高峰份额 = 4r / (6 + 4r)
        r = C.WAREHOUSE_PEAK_INTENSITY
        assert peak == pytest.approx(4 * r / (6 + 4 * r))

    def test_no_stale_intensity_constant(self):
        """旧的 `SIM_PEAK_INTENSITY` 必须已经删掉——留着它就会有人再改那一份。"""
        assert not hasattr(C, "SIM_PEAK_INTENSITY")


class TestKneeIsOnlyReportedWhenItExists:
    """拐点只在**边际收益真的递减**时报出来。

    原实现取 `sec_saved_per_yuan` 的最大值当拐点，与 docstring 写的「边际收益骤降处」是
    两回事：边际一非单调就会翻到最后一档。到达强度统一后各档边际变成 [0.0005, 0.0065]，
    原实现便报出「拐点 = 6 人」——而 6 正是测试区间的上界，那不是拐点。
    """

    @staticmethod
    def _arm(n_pickers: int, mean_sec: float) -> dict:
        return {"n_pickers": n_pickers,
                "metrics": {"avg_fulfillment_sec": {"mean": mean_sec}}}

    def test_reports_a_knee_when_returns_diminish(self):
        arms = [self._arm(4, 195.6), self._arm(5, 186.9), self._arm(6, 184.5)]
        got = W.tradeoff_curve_and_knee(arms)
        assert got["knee_at_pickers"] == 5
        assert got["knee_note"]

    def test_no_knee_when_the_biggest_gain_is_at_the_last_gear(self):
        """最大边际落在最后一档 = 「再加人还在变好」= 区间没覆盖到拐点，不是拐点。"""
        arms = [self._arm(4, 170.9), self._arm(5, 170.8), self._arm(6, 169.3)]
        got = W.tradeoff_curve_and_knee(arms)
        assert got["knee_at_pickers"] is None
        assert "没覆盖到拐点" in got["knee_note"]

    def test_every_case_states_its_reason(self):
        """无论有没有拐点，都给出可读的理由——None 不带原因等于什么都没说。"""
        for arms in ([self._arm(4, 195.6), self._arm(5, 186.9), self._arm(6, 184.5)],
                     [self._arm(4, 170.9), self._arm(5, 170.8), self._arm(6, 169.3)]):
            assert W.tradeoff_curve_and_knee(arms)["knee_note"]


class TestAggregate:
    def test_ci_contains_mean_and_n(self):
        runs = [{"m": v} for v in (10.0, 12.0, 11.0, 9.0, 13.0)]
        agg = W.aggregate_repeats(runs, "m")
        assert agg["n"] == 5
        assert agg["ci95_low"] < agg["mean"] < agg["ci95_high"]
        assert agg["mean"] == pytest.approx(11.0)

    def test_constant_values_zero_width(self):
        runs = [{"m": 5.0}] * 10
        agg = W.aggregate_repeats(runs, "m")
        assert agg["ci95_low"] == agg["ci95_high"] == 5.0


class TestTradeoff:
    def test_knee_at_best_marginal(self):
        # 构造：4→5 大幅改善、5→6 微改善 → 拐点应在 5
        arms = [
            {"n_pickers": 4, "metrics": {"avg_fulfillment_sec": {"mean": 1000.0}}},
            {"n_pickers": 5, "metrics": {"avg_fulfillment_sec": {"mean": 700.0}}},
            {"n_pickers": 6, "metrics": {"avg_fulfillment_sec": {"mean": 690.0}}},
        ]
        out = W.tradeoff_curve_and_knee(arms)
        assert out["knee_at_pickers"] == 5
        assert len(out["curve"]) == 3
        assert out["curve"][0]["daily_labor_cost"] == 4 * C.PICKER_DAILY_COST


class TestExperimentArm:
    def test_arm_repeats_and_aggregates(self, small_master):
        sku, loc = small_master
        lay = W.build_layout(sku, loc, "abc_zoned")
        ids, w = W.sku_sampling_weights(sku)
        arm = W.run_experiment_arm(lay, 4, base_seed=99, arm_id="t", repeats=3,
                                   sku_ids=ids, sku_weights=w)
        assert arm["repeats"] == 3
        assert arm["metrics"]["avg_fulfillment_sec"]["n"] == 3
        assert "ci95_low" in arm["metrics"]["avg_fulfillment_sec"]


class TestCalibration:
    def test_calibration_structures(self, tmp_path, small_master):
        # 构造微型数据层 A outbound 与 Olist clean_orders
        ob = pd.DataFrame({
            "pick_start": pd.date_range("2026-06-01 09:00", periods=5, freq="10min"),
            "pick_end": pd.date_range("2026-06-01 09:01:30", periods=5, freq="10min"),
        })
        ob_csv = tmp_path / "ob.csv"
        ob.to_csv(ob_csv, index=False)
        co = pd.DataFrame({"outbound_response_days": [1.0, 2.0, 3.0, 2.5, 1.5]})
        co_csv = tmp_path / "co.csv"
        co.to_csv(co_csv, index=False)

        sim_runs = [{"line_pick_sec_mean": 90.0, "line_pick_sec_cv": 0.5,
                     "fulfillment_cv": 0.6, "fulfillment_skew": 1.2}]
        cal_a = W.calibrate_against_layer_a(sim_runs, ob_csv)
        assert cal_a["layer_a"]["pick_sec_per_line_mean"] == pytest.approx(90.0)
        assert cal_a["mean_ratio_sim_over_a"] == pytest.approx(1.0)
        cal_o = W.calibrate_against_olist(sim_runs, co_csv)
        assert "cv" in cal_o["olist_outbound_response"]
        assert "不做绝对时长 KS" in cal_o["note"]
