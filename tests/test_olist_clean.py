"""数据层 B（olist_clean）单元测试。

测试缝：清洗纯函数对手工构造小样例断言——每条剔除规则、坐标聚合、延迟率、
重量体积分布拟合。不依赖 121M 原始大文件。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import olist_clean as OC


def _mk_orders(rows: list[dict]) -> pd.DataFrame:
    """构造最小 orders 表（五时间戳 + status）。"""
    base = {
        "order_id": "o",
        "customer_id": "c",
        "order_status": "delivered",
        "order_purchase_timestamp": "2018-01-01 10:00:00",
        "order_approved_at": "2018-01-01 11:00:00",
        "order_delivered_carrier_date": "2018-01-02 10:00:00",
        "order_delivered_customer_date": "2018-01-05 10:00:00",
        "order_estimated_delivery_date": "2018-01-08 00:00:00",
    }
    recs = [{**base, **r} for r in rows]
    return pd.DataFrame(recs)


class TestCleanOrdersRules:
    def test_keeps_clean_delivered(self):
        df, reasons = OC.clean_orders(_mk_orders([{"order_id": "a"}]))
        assert len(df) == 1
        assert reasons["dropped_not_delivered"] == 0
        assert reasons["dropped_time_contradiction"] == 0

    def test_drops_non_delivered(self):
        df, reasons = OC.clean_orders(
            _mk_orders([{"order_id": "a", "order_status": "shipped"},
                        {"order_id": "b", "order_status": "canceled"},
                        {"order_id": "c", "order_status": "delivered"}])
        )
        assert reasons["dropped_not_delivered"] == 2
        assert len(df) == 1 and df["order_id"].iat[0] == "c"

    def test_drops_delivered_before_purchase(self):
        df, reasons = OC.clean_orders(
            _mk_orders([{"order_id": "a", "order_delivered_customer_date": "2017-12-31 10:00:00"}])
        )
        assert reasons["dropped_time_contradiction"] == 1
        assert len(df) == 0

    def test_drops_carrier_before_purchase(self):
        df, reasons = OC.clean_orders(
            _mk_orders([{"order_id": "a", "order_delivered_carrier_date": "2017-12-31 10:00:00"}])
        )
        assert reasons["dropped_time_contradiction"] == 1

    def test_drops_approved_before_purchase(self):
        df, reasons = OC.clean_orders(
            _mk_orders([{"order_id": "a", "order_approved_at": "2017-12-31 10:00:00"}])
        )
        assert reasons["dropped_time_contradiction"] == 1

    def test_drops_delivered_before_carrier(self):
        df, reasons = OC.clean_orders(
            _mk_orders([{"order_id": "a",
                         "order_delivered_carrier_date": "2018-01-06 10:00:00"}])
        )
        # 实际送达(01-05) 早于发货(01-06) → 矛盾
        assert reasons["dropped_time_contradiction"] == 1

    def test_drops_missing_key_timestamp(self):
        df, reasons = OC.clean_orders(
            _mk_orders([{"order_id": "a", "order_delivered_customer_date": None}])
        )
        assert reasons["dropped_missing_timestamp"] == 1
        assert len(df) == 0

    def test_timestamps_converted_to_datetime(self):
        df, _ = OC.clean_orders(_mk_orders([{"order_id": "a"}]))
        assert pd.api.types.is_datetime64_any_dtype(df["order_purchase_timestamp"])

    def test_each_row_counted_once(self):
        # 一条同时非 delivered 又时间矛盾，只计入第一个命中原因（非 delivered）
        df, reasons = OC.clean_orders(
            _mk_orders([{"order_id": "a", "order_status": "shipped",
                         "order_delivered_customer_date": "2017-12-31 10:00:00"}])
        )
        assert reasons["dropped_not_delivered"] == 1
        assert reasons["dropped_time_contradiction"] == 0


class TestGeolocationAggregation:
    def test_median_of_multiple_coords(self):
        geo = pd.DataFrame({
            "geolocation_zip_code_prefix": [100, 100, 100, 200],
            "geolocation_lat": [-23.0, -23.5, -24.0, -19.0],
            "geolocation_lng": [-46.0, -46.5, -47.0, -43.0],
        })
        agg = OC.aggregate_geolocation(geo)
        row100 = agg[agg["geolocation_zip_code_prefix"] == 100].iloc[0]
        assert row100["lat"] == pytest.approx(-23.5)  # 中位数
        assert row100["lng"] == pytest.approx(-46.5)
        assert len(agg) == 2  # 两个邮编聚合为两行

    def test_drops_nan_coords(self):
        geo = pd.DataFrame({
            "geolocation_zip_code_prefix": [100],
            "geolocation_lat": [np.nan],
            "geolocation_lng": [-46.0],
        })
        assert len(OC.aggregate_geolocation(geo)) == 0


class TestDelayRate:
    def test_delay_rate_computation(self):
        # 2 单延迟（实际送达 > 预计），1 单准时
        co = pd.DataFrame({
            "is_late": [True, True, False],
        })
        d = OC.compute_delay_rate(co)
        assert d["n_orders"] == 3
        assert d["n_late"] == 2
        assert d["delay_rate"] == pytest.approx(2 / 3, abs=1e-4)
        assert d["on_time_rate"] == pytest.approx(1 / 3, abs=1e-4)


class TestWeightVolumeDistribution:
    def test_lognorm_fit_positive(self):
        rng = np.random.default_rng(0)
        # 构造对数正态样本
        w = rng.lognormal(mean=np.log(800), sigma=0.7, size=2000)
        v = rng.lognormal(mean=np.log(2000), sigma=0.6, size=2000)
        df = pd.DataFrame({"product_weight_g": w, "product_volume_cm3": v})
        dist = OC.fit_weight_volume_distribution(df)
        assert dist["product_weight_g"]["fitted"] is True
        # 拟合 mu 应接近真值 log(800)≈6.68
        assert dist["product_weight_g"]["lognorm_mu"] == pytest.approx(np.log(800), abs=0.15)
        assert "P50" in dist["product_weight_g"]["quantiles"]

    def test_filters_nonpositive(self):
        df = pd.DataFrame({
            "product_weight_g": [0, -5, np.nan, 100, 200, 300, 400, 500, 600, 700, 800],
            "product_volume_cm3": [10] * 11,
        })
        dist = OC.fit_weight_volume_distribution(df)
        # 仅 8 个正值（<10）→ 拟合样本不足，标记 fitted=False
        assert dist["product_weight_g"]["n"] == 8
        assert dist["product_weight_g"]["fitted"] is False


class TestBuildJoins:
    def test_order_items_filtered_to_clean_orders(self):
        orders_clean = pd.DataFrame({"order_id": ["a", "b"]})
        items = pd.DataFrame({
            "order_id": ["a", "b", "z"],  # z 不在 clean_orders
            "product_id": ["p1", "p2", "p3"],
        })
        products = pd.DataFrame({
            "product_id": ["p1", "p2", "p3"],
            "product_category_name": ["c1", "c2", "c3"],
            "product_weight_g": [100, 200, 300],
            "product_length_cm": [10, 10, 10],
            "product_height_cm": [10, 10, 10],
            "product_width_cm": [10, 10, 10],
        })
        out = OC.build_order_items_clean(items, products, orders_clean)
        assert set(out["order_id"]) == {"a", "b"}
        assert (out["product_volume_cm3"] == 1000).all()  # 10×10×10
