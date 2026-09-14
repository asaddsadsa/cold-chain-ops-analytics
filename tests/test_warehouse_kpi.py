"""模块一（上）仓内 KPI 单元测试（08 号票）。

测试缝（spec「Testing Decisions」预约定）：
  ① KPI 纯函数缝：六 KPI + ABC + 库位行走成本，用手工构造小输入断言等于手算期望；
  ② 数据产物缝：跑通全流程，断言两个埋点复现、重排降幅、产物齐备且不含 Excel 残留。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import warehouse_kpi as WK

MINI_SKU = pd.DataFrame(
    {
        "sku_id": ["S1", "S2", "S3", "S4"],
        "category": ["P03", "P01", "P01", "P02"],
        "unit_price": [10.0, 20.0, 5.0, 1.0],
        "weight_kg": [1.0, 2.0, 0.5, 0.1],
        "volume_m3": [0.1, 0.2, 0.05, 0.01],
        "abc_initial_label": ["A", "B", "C", "C"],
    }
)


def _ts(seq):
    return pd.to_datetime(pd.Series(seq))


class TestInventoryAccuracy:
    """库存准确率：金额加权（需求 3.7）。"""

    def test_hand_computed_amount_weighted_ratio(self):
        stocktake = pd.DataFrame(
            {
                "sku_id": ["S1", "S2"],
                "book_qty": [100, 50],
                "physical_qty": [95, 50],
                "diff_qty": [-5, 0],
            }
        )
        got = WK.inventory_accuracy(stocktake, MINI_SKU)
        # Σ|差异|×单价 = 5×10 = 50；Σ账面×单价 = 100×10 + 50×20 = 2000
        assert got["abs_diff_value"] == pytest.approx(50.0)
        assert got["book_value"] == pytest.approx(2000.0)
        assert got["rate"] == pytest.approx(1 - 50 / 2000)

    def test_amount_weighting_differs_from_count_weighting(self):
        # 高值 SKU 错 1 件 vs 低值 SKU 错 50 件：件数口径会判断反，金额口径不会
        stocktake = pd.DataFrame(
            {
                "sku_id": ["S2", "S4"],  # 单价 20 与 1
                "book_qty": [100, 100],
                "physical_qty": [99, 50],
                "diff_qty": [-1, -50],
            }
        )
        got = WK.inventory_accuracy(stocktake, MINI_SKU)
        # 金额：|−1|×20 + |−50|×1 = 70；账面 100×20 + 100×1 = 2100
        assert got["abs_diff_value"] == pytest.approx(70.0)
        assert got["rate"] == pytest.approx(1 - 70 / 2100)

    def test_unknown_sku_raises(self):
        stocktake = pd.DataFrame(
            {"sku_id": ["NOPE"], "book_qty": [1], "physical_qty": [1], "diff_qty": [0]}
        )
        with pytest.raises(ValueError):
            WK.inventory_accuracy(stocktake, MINI_SKU)


class TestReceiptTimeliness:
    """收货及时率：预约 + 30 分钟容差（需求 3.7）。"""

    def test_tolerance_boundary_is_inclusive(self):
        inbound = pd.DataFrame(
            {
                "expected_arrival": _ts(["2026-06-01 10:00"] * 3),
                "actual_arrival": _ts(
                    ["2026-06-01 10:30", "2026-06-01 10:31", "2026-06-01 09:00"]
                ),
            }
        )
        got = WK.receipt_timeliness(inbound)
        assert got["tolerance_min"] == C.RECEIPT_TOLERANCE_MIN
        assert got["n_late"] == 1  # 恰好 30 分钟算及时，31 分钟不算
        assert got["rate"] == pytest.approx(2 / 3)

    def test_zero_tolerance_requires_early_or_on_time(self):
        inbound = pd.DataFrame(
            {
                "expected_arrival": _ts(["2026-06-01 10:00"]),
                "actual_arrival": _ts(["2026-06-01 10:00"]),
            }
        )
        assert WK.receipt_timeliness(inbound, tolerance_min=0)["rate"] == pytest.approx(1.0)


class TestPickingEfficiency:
    """拣货效率：行/人时（需求 3.7）。"""

    def test_hand_computed_lines_per_picker_hour(self):
        outbound = pd.DataFrame(
            {
                "pick_start": _ts(["2026-06-01 09:00", "2026-06-01 09:00", "2026-06-01 10:00"]),
                "pick_end": _ts(["2026-06-01 09:01", "2026-06-01 09:02", "2026-06-01 10:03"]),
            }
        )
        got = WK.picking_efficiency(outbound)
        # 3 行 / (60+120+180 秒 = 0.1 人时) = 30 行/人时
        assert got["n_lines"] == 3
        assert got["picker_hours"] == pytest.approx(0.1)
        assert got["lines_per_hour"] == pytest.approx(30.0)
        assert got["sec_per_line"] == pytest.approx(120.0)

    def test_by_hour_drilldown_keeps_each_hour(self):
        outbound = pd.DataFrame(
            {
                "pick_start": _ts(["2026-06-01 09:00", "2026-06-01 14:00"]),
                "pick_end": _ts(["2026-06-01 09:01", "2026-06-01 14:05"]),
            }
        )
        by_hour = WK.picking_efficiency_by_hour(outbound)
        assert dict(zip(by_hour["hour"], by_hour["sec_per_line"])) == {9: 60.0, 14: 300.0}


class TestAvgFulfillment:
    """订单平均履约时效：按订单聚合，不被多行订单放大权重（需求 3.7）。"""

    def test_orders_are_weighted_equally_not_by_line_count(self):
        outbound = pd.DataFrame(
            {
                # 订单 A 有 2 行（4 小时），订单 B 有 1 行（2 小时）→ 均值应为 3 小时
                "order_id": ["A", "A", "B"],
                "order_time": _ts(["2026-06-01 08:00"] * 3),
                "ship_time": _ts(["2026-06-01 12:00", "2026-06-01 12:00", "2026-06-01 10:00"]),
            }
        )
        got = WK.avg_fulfillment_hours(outbound)
        assert got["n_orders"] == 2
        assert got["hours"] == pytest.approx(3.0)


class TestLocationUtilization:
    """库容利用率 = 已占用库位 / 总库位（需求 3.7）。"""

    def test_hand_computed(self):
        loc = pd.DataFrame({"loc_id": [f"L{i}" for i in range(8)], "walk_dist_m": range(8)})
        assignment = pd.Series({"S1": "L0", "S2": "L1", "S2b": "L1"})
        got = WK.location_utilization(assignment, loc)
        assert got["occupied"] == 2  # 两个 SKU 挤同一库位只算一个占用
        assert got["rate"] == pytest.approx(0.25)


MINI_STOCKTAKE = pd.DataFrame(
    {
        "sku_id": ["S1", "S1", "S2", "S3", "S4"],
        "diff_type": ["无差异", "盘亏", "盘盈", "无差异", "无差异"],
    }
)


class TestStocktakeDiscrepancy:
    """盘点差异率与分品类下钻（需求 3.7）。"""

    def test_rate_counts_both_directions(self):
        got = WK.stocktake_discrepancy(MINI_STOCKTAKE, MINI_SKU)
        assert got["n_diff"] == 2  # 盘亏与盘盈都算差异
        assert got["rate"] == pytest.approx(2 / 5)

    def test_category_drilldown_splits_by_sku_category(self):
        by_cat = WK.stocktake_discrepancy(MINI_STOCKTAKE, MINI_SKU, by="category")
        rates = dict(zip(by_cat["category"], by_cat["rate"]))
        assert rates["P03"] == pytest.approx(0.5)  # S1 两条，一差一无
        assert rates["P01"] == pytest.approx(0.5)  # S2 盘盈 / S3 无差异
        assert rates["P02"] == pytest.approx(0.0)  # S4 无差异

    def test_unknown_dimension_raises(self):
        with pytest.raises(ValueError):
            WK.stocktake_discrepancy(MINI_STOCKTAKE, MINI_SKU, by="operator")

    def test_date_drilldown_is_the_data_supported_time_dimension(self):
        # 盘点表只到「日」粒度：分时段下钻按日期，14–16 点低谷的载体是拣货效率而非盘差率
        stocktake = MINI_STOCKTAKE.assign(
            date=["2026-06-01", "2026-06-01", "2026-06-01", "2026-06-02", "2026-06-02"]
        )
        by_date = WK.stocktake_discrepancy(stocktake, MINI_SKU, by="date")
        rates = dict(zip(by_date["date"], by_date["rate"]))
        assert rates["2026-06-01"] == pytest.approx(2 / 3)  # 无差异/盘亏/盘盈
        assert rates["2026-06-02"] == pytest.approx(0.0)  # 两条都无差异


class TestAbcClassify:
    """ABC 分类：按出库行数累计占比，阈值读 config（ADR-0002）。"""

    def _outbound(self):
        # S1:5 行、S2:3 行、S3:1 行、S4:1 行 → 累计 0.5 / 0.8 / 0.9 / 1.0
        rows = [("S1", 5), ("S2", 3), ("S3", 1), ("S4", 1)]
        return pd.DataFrame(
            [{"order_id": f"O{i}", "sku_id": s} for s, n in rows for i in range(n)]
        )

    def test_class_boundaries_use_previous_cumulative_share(self):
        sku, pareto, summary = WK.abc_classify(self._outbound())
        assert dict(zip(sku["sku_id"], sku["abc_class"])) == {
            "S1": "A", "S2": "A", "S3": "B", "S4": "C"
        }
        assert summary["total_lines"] == 10

    def test_pareto_is_descending_and_cumulative_ends_at_one(self):
        _, pareto, _ = WK.abc_classify(self._outbound())
        assert pareto["outbound_lines"].is_monotonic_decreasing
        assert pareto["cum_share"].iloc[-1] == pytest.approx(1.0)

    def test_thresholds_are_configurable(self):
        sku, _, summary = WK.abc_classify(self._outbound(), thresholds=(0.4, 0.8))
        assert summary["thresholds"] == {"A": 0.4, "B": 0.8}
        assert (sku["abc_class"] == "A").sum() == 1  # 只有 S1 的上一累计 0 < 0.4

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            WK.abc_classify(self._outbound(), thresholds=(0.9, 0.7))


class TestSlotting:
    """库位重排：Σ(频次 × 距离) 与重排不等式最优配对（需求 34）。"""

    def test_walk_cost_hand_computed(self):
        assert WK.walk_cost([10, 1], [1, 10]) == pytest.approx(20.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            WK.walk_cost([1, 2], [1.0])

    def test_optimize_pairs_high_frequency_with_near_locations(self):
        # 频次 [10,1] 配距离 [10,1] 是最差配对：成本 101；最优配对后 20
        freq = pd.Series([10.0, 1.0], index=["HI", "LO"])
        dist = pd.Series([10.0, 1.0], index=["HI", "LO"])
        locs = pd.Series(["L_far", "L_near"], index=["HI", "LO"])
        got = WK.optimize_slotting(freq, dist, locs)
        assert got["walk_cost_before"] == pytest.approx(101.0)
        assert got["walk_cost_after"] == pytest.approx(20.0)
        assert got["reduction_pct"] == pytest.approx((101 - 20) / 101 * 100, abs=0.01)
        a = got["assignment"].set_index("sku_id")
        assert a.loc["HI", "walk_dist_after_m"] == pytest.approx(1.0)
        assert a.loc["HI", "loc_id_after"] == "L_near"

    def test_optimize_never_worsens_cost(self):
        rng = np.random.default_rng(3)
        freq = pd.Series(rng.integers(1, 100, 30).astype(float), index=[f"S{i}" for i in range(30)])
        dist = pd.Series(rng.uniform(1, 100, 30), index=freq.index)
        locs = pd.Series([f"L{i}" for i in range(30)], index=freq.index)
        got = WK.optimize_slotting(freq, dist, locs)
        assert got["walk_cost_after"] <= got["walk_cost_before"]

    def test_already_optimal_layout_reports_zero_reduction(self):
        freq = pd.Series([10.0, 1.0], index=["HI", "LO"])
        dist = pd.Series([1.0, 10.0], index=["HI", "LO"])
        got = WK.optimize_slotting(freq, dist, pd.Series(["L0", "L1"], index=freq.index))
        assert got["reduction_pct"] == pytest.approx(0.0)


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    """跑通全流程到临时目录（不污染产物目录），断言落在真实产物上。"""
    out = tmp_path_factory.mktemp("wh_kpi")
    return WK.run_all(out_dir=out)


class TestDataProducts:
    """数据产物缝：埋点复现、重排降幅、产物齐备且不含 Excel 残留。"""

    def test_all_artifacts_written(self, outputs):
        expected = {
            C.WAREHOUSE_KPI_JSON.name, C.WAREHOUSE_DRILLDOWN_CATEGORY_CSV.name,
            C.WAREHOUSE_DRILLDOWN_DATE_CSV.name, C.WAREHOUSE_DRILLDOWN_HOUR_CSV.name,
            C.WAREHOUSE_ABC_CSV.name, C.WAREHOUSE_ABC_PARETO_CSV.name,
            C.WAREHOUSE_SLOTTING_JSON.name, C.WAREHOUSE_SLOTTING_CSV.name,
        }
        assert expected <= set(outputs["paths"])
        for name, path in outputs["paths"].items():
            assert Path(path).exists(), f"未落盘：{name}"

    def test_six_kpis_are_all_in_plausible_ranges(self, outputs):
        k = outputs["result"]["kpis"]
        assert set(k) == {
            "inventory_accuracy", "receipt_timeliness", "picking_efficiency",
            "avg_fulfillment_hours", "location_utilization", "stocktake_discrepancy",
        }
        assert 0.90 < k["inventory_accuracy"]["rate"] <= 1.0
        assert 0.5 < k["receipt_timeliness"]["rate"] < 1.0
        assert k["picking_efficiency"]["lines_per_hour"] > 0
        assert k["avg_fulfillment_hours"]["hours"] > 0
        assert k["location_utilization"]["rate"] == pytest.approx(500 / 800, abs=0.01)
        assert 0.0 < k["stocktake_discrepancy"]["rate"] < 0.5

    def test_embedding_p03_high_discrepancy(self, outputs):
        e = outputs["result"]["embedding_checks"]["p03_discrepancy"]
        assert e["p03_rate"] > e["other_rate"] * 2
        assert e["ratio"] > 2.0

    def test_embedding_pick_slowdown_14_16(self, outputs):
        e = outputs["result"]["embedding_checks"]["pick_slowdown_14_16"]
        assert e["sec_per_line_14_16"] > e["sec_per_line_other"] * 1.5
        assert e["ratio"] > 1.5

    def test_abc_matches_simulation_warehouse_outbound_lines(self, outputs):
        # 分类数据源必须是仿真仓出库行数（不是 Olist），且行数守恒
        abc = outputs["result"]["abc"]
        assert abc["summary"]["total_lines"] == len(WK.load_warehouse_tables()["outbound"])
        assert abc["sku"]["outbound_lines"].sum() == abc["summary"]["total_lines"]

    def test_slotting_reduces_walk_cost_substantially(self, outputs):
        s = outputs["result"]["slotting"]
        assert s["walk_cost_after"] < s["walk_cost_before"]
        assert s["reduction_pct"] > 20  # 数据层 A 埋了「A 类远库位」，改善空间应显著

    def test_slotting_assignment_keeps_the_same_location_pool(self, outputs):
        # 重排只换占用关系、不扩仓：库位集合必须与重排前完全一致
        a = outputs["result"]["slotting"]["assignment"]
        assert a["walk_dist_after_m"].sum() == pytest.approx(a["walk_dist_before_m"].sum())
        assert set(a["walk_dist_after_m"]) == set(a["walk_dist_before_m"])

    def test_report_has_no_excel_section(self, outputs):
        # 用户不需要 Excel 产物：报告与产物目录都不得再出现 Excel 校验
        text = Path(outputs["paths"][C.WAREHOUSE_KPI_MD.name]).read_text(encoding="utf-8")
        assert "Excel" not in text and "xlsx" not in text
        assert not list(Path(outputs["paths"][C.WAREHOUSE_KPI_MD.name]).parent.glob("*.xlsx"))

    def test_report_cites_the_embedding_and_slotting_numbers(self, outputs):
        text = Path(outputs["paths"][C.WAREHOUSE_KPI_MD.name]).read_text(encoding="utf-8")
        e = outputs["result"]["embedding_checks"]
        assert f"{e['p03_discrepancy']['p03_rate']:.2%}" in text
        assert f"{e['pick_slowdown_14_16']['ratio']}%" not in text  # 比值不冒充百分比
        assert f"{outputs['result']['slotting']['reduction_pct']}%" in text

    def test_kpi_json_is_self_describing(self, outputs):
        doc = json.loads(Path(outputs["paths"][C.WAREHOUSE_KPI_JSON.name]).read_text(encoding="utf-8"))
        assert doc["data_category"] == "过程仿真"
        assert doc["abc"]["thresholds"] == {"A": C.ABC_THRESHOLDS[0], "B": C.ABC_THRESHOLDS[1]}
        assert set(doc["kpis"]) == set(outputs["result"]["kpis"])


class TestDailyKpi:
    """逐日序列（驾驶舱趋势线的数据源）：分组口径必须显式且可断言。"""

    def test_inventory_accuracy_by_date_is_amount_weighted_within_each_day(self):
        stocktake = pd.DataFrame(
            {
                "sku_id": ["S1", "S2", "S1", "S2"],
                "date": ["2026-06-01", "2026-06-01", "2026-06-02", "2026-06-02"],
                "book_qty": [100, 50, 100, 50],
                "physical_qty": [95, 50, 100, 50],
                "diff_qty": [-5, 0, 0, 0],
            }
        )
        got = WK.inventory_accuracy(stocktake, MINI_SKU, by="date").set_index("date")
        # 6-01：Σ|差异|×单价 = 5×10 = 50；Σ账面×单价 = 100×10 + 50×20 = 2000 → 1 − 50/2000
        assert got.loc["2026-06-01", "rate"] == pytest.approx(1 - 50 / 2000)
        assert got.loc["2026-06-02", "rate"] == pytest.approx(1.0)
        assert list(got.index) == ["2026-06-01", "2026-06-02"]

    def test_inventory_accuracy_rejects_unknown_drilldown(self):
        stocktake = pd.DataFrame(
            {"sku_id": ["S1"], "date": ["2026-06-01"], "book_qty": [1],
             "physical_qty": [1], "diff_qty": [0]}
        )
        with pytest.raises(ValueError, match="不支持的库存准确率下钻维度"):
            WK.inventory_accuracy(stocktake, MINI_SKU, by="category")

    def _tables(self):
        return {
            "sku": MINI_SKU,
            "stocktake": pd.DataFrame(
                {
                    "sku_id": ["S1", "S2", "S1"],
                    "date": ["2026-06-01", "2026-06-01", "2026-06-02"],
                    "book_qty": [100, 50, 100],
                    "physical_qty": [95, 50, 100],
                    "diff_qty": [-5, 0, 0],
                    "diff_type": ["盘亏", "无差异", "无差异"],
                }
            ),
            # 两张收货记录都**预约在 6-01**，其中一张实际次日才到：
            # 排班口径看「该哪天到」，故两条都算在 6-01（及时率 0.5），6-02 无收货记录。
            # 若按实际到货日分组，会得到 6-01=1.0、6-02=0.0 —— 当天的问题被挪到第二天。
            "inbound": pd.DataFrame(
                {
                    "expected_arrival": ["2026-06-01 10:00", "2026-06-01 10:00"],
                    "actual_arrival": ["2026-06-01 10:20", "2026-06-02 09:00"],
                }
            ),
            "outbound": pd.DataFrame(
                {
                    "pick_start": ["2026-06-01 09:00:00", "2026-06-01 10:00:00"],
                    "pick_end": ["2026-06-01 09:03:00", "2026-06-01 10:03:00"],
                }
            ),
        }

    def test_receipt_timeliness_is_grouped_by_expected_arrival_day(self):
        got = WK.daily_kpi(self._tables()).set_index("date")
        assert got.loc["2026-06-01", "receipt_timeliness_rate"] == pytest.approx(0.5)
        assert got.loc["2026-06-01", "n_receipts"] == 2
        assert "2026-06-02" in got.index
        assert pd.isna(got.loc["2026-06-02", "receipt_timeliness_rate"])

    def test_picking_efficiency_is_lines_per_hour_on_the_pick_day(self):
        got = WK.daily_kpi(self._tables()).set_index("date")
        # 两条各 180 秒 → 360 秒 = 0.1 人时，2 行 → 20 行/人时
        assert got.loc["2026-06-01", "picking_lines_per_hour"] == pytest.approx(20.0)
        assert got.loc["2026-06-01", "n_pick_lines"] == 2

    def test_inventory_accuracy_and_discrepancy_coexist_per_day(self):
        got = WK.daily_kpi(self._tables()).set_index("date")
        # 准确率按金额加权、差异率按记录数，两者同一天不必互补——这是有意的两个口径
        assert got.loc["2026-06-01", "inventory_accuracy"] == pytest.approx(1 - 50 / 2000)
        assert got.loc["2026-06-01", "stocktake_discrepancy_rate"] == pytest.approx(0.5)
