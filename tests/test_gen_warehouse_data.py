"""数据层 A（gen_warehouse_data）埋点回归测试。

测试缝（spec 测试缝②「数据产物缝」）：读生成产物与质检摘要，断言四埋点统计量成立、
表间外键与时间逻辑自洽、固定种子重跑产物一致。只测外部行为，不测内部实现。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import config as C
from src import gen_warehouse_data as G


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> tuple[Path, dict]:
    """整模块只生成一次（耗时操作），各测试共享产物目录与路径字典。"""
    out = tmp_path_factory.mktemp("warehouse")
    qc = out / "qc"
    paths = G.generate_warehouse(seed=C.SEED_WAREHOUSE, out_dir=out, qc_dir=qc)
    return out, paths


@pytest.fixture(scope="module")
def qc_summary(generated) -> dict:
    _, paths = generated
    return json.loads(Path(paths["qc_json"]).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tables(generated) -> dict[str, pd.DataFrame]:
    out, _ = generated
    return {
        "sku": pd.read_csv(out / "sku_master.csv"),
        "loc": pd.read_csv(out / "location_master.csv"),
        "inbound": pd.read_csv(out / "inbound_receipts.csv"),
        "outbound": pd.read_csv(out / "outbound_orders.csv", parse_dates=["order_time", "pick_start", "pick_end", "check_time", "ship_time"]),
        "stocktake": pd.read_csv(out / "inventory_stocktake.csv"),
    }


class TestScale:
    """规模符合需求（SKU 500 / 库位 800 / 日均 200–400 / 90 天）。"""

    def test_sku_count(self, tables):
        assert len(tables["sku"]) == 500

    def test_location_count(self, tables):
        assert len(tables["loc"]) == 800

    def test_date_range_90_days(self, qc_summary):
        assert qc_summary["date_range"]["days"] == 90

    def test_daily_orders_in_range(self, qc_summary):
        d = qc_summary["daily_orders"]
        assert d["in_range_200_400"] is True
        assert 200 <= d["min"] and d["max"] <= 400


class TestIntegrity:
    """外键与时间逻辑自洽。"""

    def test_no_fk_violations(self, qc_summary):
        assert qc_summary["integrity"]["fk_violations"] == 0

    def test_no_time_order_violations(self, qc_summary):
        assert qc_summary["integrity"]["time_order_violations"] == 0

    def test_time_chain_strictly_increasing(self, tables):
        o = tables["outbound"]
        assert (o["pick_end"] > o["pick_start"]).all()
        assert (o["check_time"] > o["pick_end"]).all()
        assert (o["ship_time"] > o["check_time"]).all()

    def test_no_negative_stock_in_ledger(self, qc_summary):
        # 台账最低库存非负（期初+入库−出库全程不穿仓）
        assert qc_summary["integrity"]["min_stock"] >= 0

    def test_stocktake_physical_non_negative(self, tables):
        assert (tables["stocktake"]["physical_qty"] >= 0).all()


class TestLocationDistance:
    """库位行走距离由行列坐标计算，非随机。"""

    def test_distance_from_coordinates(self, tables):
        loc = tables["loc"]
        expected = loc["row"] * 3.0 + loc["col"] * 1.5  # _ROW_SPACING_M / _COL_SPACING_M
        assert (loc["walk_dist_m"] - expected).abs().max() < 0.01

    def test_distance_positive(self, tables):
        assert (tables["loc"]["walk_dist_m"] > 0).all()


class TestEmbeddings:
    """四埋点统计验证值成立（读质检摘要）。"""

    def test_embedding1_a_class_far_locations(self, qc_summary):
        # 埋点①：A 类平均库位距离显著大于 C 类（反转布局，比值应明显 >1）
        e = qc_summary["embedding_checks"]["abc_distance"]
        assert e["ratio_A_over_C"] > 2.0
        assert e["mean_dist_A_m"] > e["mean_dist_C_m"]

    def test_embedding2_p03_high_discrepancy(self, qc_summary):
        # 埋点②：P03 差异率显著高于其他品类
        e = qc_summary["embedding_checks"]["p03_discrepancy"]
        assert e["p03_rate"] > e["other_rate"]
        assert e["ratio"] > 2.0

    def test_embedding3_pick_slowdown_14_16(self, qc_summary):
        # 埋点③：14–16 点拣货秒/行系统性高于其余时段
        e = qc_summary["embedding_checks"]["pick_slowdown_14_16"]
        assert e["sec_per_line_14_16"] > e["sec_per_line_other"]
        assert e["ratio"] > 1.4  # 放大系数下限 1.75，含行走稀释后比值仍应显著

    def test_embedding4_late_arrival(self, qc_summary):
        # 埋点④：收货晚到率落在可控比例（目标 ~18%，留抽样波动余量）
        rate = qc_summary["embedding_checks"]["late_arrival"]["late_rate"]
        assert 0.10 <= rate <= 0.26


class TestReproducibility:
    """固定种子重跑产物一致。"""

    def test_same_seed_same_outbound(self, generated):
        out1, _ = generated
        ob1 = pd.read_csv(out1 / "outbound_orders.csv")
        # 重跑一次到另一目录
        out2 = out1.parent / "warehouse_rerun"
        out2.mkdir(exist_ok=True)
        G.generate_warehouse(seed=C.SEED_WAREHOUSE, out_dir=out2, qc_dir=out2 / "qc")
        ob2 = pd.read_csv(out2 / "outbound_orders.csv")
        assert ob1.equals(ob2)

    def test_different_seed_changes_output(self, generated):
        out1, _ = generated
        ob1 = pd.read_csv(out1 / "outbound_orders.csv")
        out3 = out1.parent / "warehouse_seed99"
        out3.mkdir(exist_ok=True)
        G.generate_warehouse(seed=99, out_dir=out3, qc_dir=out3 / "qc")
        ob3 = pd.read_csv(out3 / "outbound_orders.csv")
        assert not ob1.equals(ob3)
