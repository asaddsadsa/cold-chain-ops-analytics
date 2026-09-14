"""模块一（下）olist_kpi 单元测试。

测试缝：spec 测试缝①（KPI 纯函数缝）——手工构造小样例，断言每个 KPI 口径，
含准时判定边界（实际送达==预计送达算准时）、重量分段边界、延迟率、差评率。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import olist_kpi as K


def _mk_orders(rows: list[dict]) -> pd.DataFrame:
    base = {
        "order_id": "o",
        "customer_state": "SP",
        "order_purchase_timestamp": "2018-01-15 10:00:00",
        "fulfillment_days": 10.0,
        "outbound_response_days": 1.0,
        "transit_days": 9.0,
        "is_late": False,
        "order_weight_kg": np.nan,
        "weight_segment": "未知",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


class TestOntimeDelayRate:
    def test_ontime_rate_basic(self):
        df = _mk_orders([{"is_late": False}, {"is_late": False}, {"is_late": True}])
        assert K.real_ontime_rate(df) == pytest.approx(2 / 3)
        assert K.real_delay_rate(df) == pytest.approx(1 / 3)

    def test_ontime_equals_estimated_counts_as_ontime(self):
        # 准时边界：实际送达 == 预计送达 → is_late=False（03 落盘口径），算准时
        df = _mk_orders([{"is_late": False}])
        assert K.real_ontime_rate(df) == 1.0

    def test_empty_returns_nan(self):
        assert np.isnan(K.real_ontime_rate(_mk_orders([])))

    def test_rates_complement(self):
        df = _mk_orders([{"is_late": True}, {"is_late": False}])
        assert K.real_ontime_rate(df) + K.real_delay_rate(df) == pytest.approx(1.0)


class TestWeightSegmentation:
    def test_order_weight_sum_and_segment(self):
        orders = _mk_orders([{"order_id": "a"}, {"order_id": "b"}])
        items = pd.DataFrame(
            {
                "order_id": ["a", "a", "b"],
                "product_weight_g": [400, 300, 5000],  # a=700g→0.5–1kg, b=5kg→3–10kg
            }
        )
        out = K.add_order_weight(orders, items)
        a = out[out["order_id"] == "a"].iloc[0]
        b = out[out["order_id"] == "b"].iloc[0]
        assert a["order_weight_kg"] == pytest.approx(0.7)
        assert a["weight_segment"] == "0.5–1kg"
        assert b["order_weight_kg"] == pytest.approx(5.0)
        assert b["weight_segment"] == "3–10kg"

    def test_boundary_0_5kg(self):
        # 边界：恰好 0.5kg 归 ≤0.5kg（right=True，bins 上界含）
        orders = _mk_orders([{"order_id": "a"}])
        items = pd.DataFrame({"order_id": ["a"], "product_weight_g": [500]})
        out = K.add_order_weight(orders, items)
        assert out.iloc[0]["weight_segment"] == "≤0.5kg"

    def test_missing_weight_is_unknown(self):
        orders = _mk_orders([{"order_id": "a"}])
        items = pd.DataFrame({"order_id": ["zzz"], "product_weight_g": [100]})  # 不匹配
        out = K.add_order_weight(orders, items)
        assert out.iloc[0]["weight_segment"] == "未知"


class TestDrilldowns:
    def test_by_state_filters_and_sorts(self, monkeypatch):
        monkeypatch.setattr(K, "MIN_STATE_ORDERS", 2)
        df = _mk_orders(
            [
                {"customer_state": "SP", "is_late": False},
                {"customer_state": "SP", "is_late": True},
                {"customer_state": "RJ", "is_late": True},
                {"customer_state": "RJ", "is_late": True},
                {"customer_state": "MG", "is_late": True},  # 仅1单，被阈值过滤
            ]
        )
        out = K.drilldown_by_state(df)
        assert "MG" not in out["customer_state"].values  # 阈值过滤
        assert list(out["customer_state"])[0] == "RJ"  # 延迟率降序，RJ=100% 居首
        assert out[out["customer_state"] == "RJ"]["delay_rate"].iat[0] == 1.0

    def test_by_weight_ordered(self):
        df = _mk_orders(
            [
                {"weight_segment": "3–10kg", "is_late": True},
                {"weight_segment": "≤0.5kg", "is_late": False},
                {"weight_segment": "未知", "is_late": False},  # 应剔除
            ]
        )
        out = K.drilldown_by_weight(df)
        assert "未知" not in out["weight_segment"].values
        # 段顺序：≤0.5kg 在 3–10kg 前
        segs = list(out["weight_segment"])
        assert segs.index("≤0.5kg") < segs.index("3–10kg")

    def test_by_month_sorted(self):
        df = _mk_orders(
            [
                {"order_purchase_timestamp": "2018-03-01 10:00:00", "is_late": False},
                {"order_purchase_timestamp": "2018-01-01 10:00:00", "is_late": True},
                {"order_purchase_timestamp": "2018-02-01 10:00:00", "is_late": False},
            ]
        )
        out = K.drilldown_by_month(df)
        assert list(out["month"]) == ["2018-01", "2018-02", "2018-03"]


class TestWeightTimeRelationship:
    def test_monotonic_check_structure(self):
        df = _mk_orders(
            [
                {"weight_segment": "≤0.5kg", "fulfillment_days": 8.0, "transit_days": 7.0},
                {"weight_segment": ">10kg", "fulfillment_days": 14.0, "transit_days": 12.0},
            ]
        )
        out = K.weight_time_relationship(df)
        assert list(out["weight_segment"]) == ["≤0.5kg", ">10kg"]
        assert out[out["weight_segment"] == ">10kg"]["avg_fulfillment_days"].iat[0] == 14.0


class TestReviewCross:
    def test_late_vs_ontime_scores(self):
        orders = _mk_orders(
            [
                {"order_id": "a", "is_late": True},
                {"order_id": "b", "is_late": False},
                {"order_id": "c", "is_late": False},
            ]
        )
        reviews = pd.DataFrame(
            {
                "order_id": ["a", "a", "b", "c"],
                "review_score": [1, 2, 5, 4],  # a 均分1.5(差), b=5, c=4
            }
        )
        rc = K.review_cross_analysis(orders, reviews)
        assert rc["late"]["n_orders"] == 1
        assert rc["late"]["mean_review_score"] == pytest.approx(1.5)
        assert rc["late"]["bad_review_rate"] == 1.0  # 1.5 ≤ 2 → 差评
        assert rc["on_time"]["n_orders"] == 2
        assert rc["on_time"]["mean_review_score"] == pytest.approx(4.5)
        assert rc["on_time"]["bad_review_rate"] == 0.0
        assert rc["late_vs_ontime_score_gap"] == pytest.approx(3.0)

    def test_multi_review_averaged_per_order(self):
        orders = _mk_orders([{"order_id": "a", "is_late": False}])
        reviews = pd.DataFrame({"order_id": ["a", "a"], "review_score": [4, 2]})
        rc = K.review_cross_analysis(orders, reviews)
        assert rc["on_time"]["mean_review_score"] == pytest.approx(3.0)
