"""在途异常领域规则的回归测试（`src/anomaly.py`）。

这套规则原先在报告侧（`transport_decisions.typical_anomaly_cases`，有 5 条单测）与看板侧
（`dashboard.components.warning_rows`，零测试）各写一遍，两处 docstring 都写着「同一套规则」
而没有任何东西强制它们一致。收口后规则只有一份，本文件守住两件事：

  ① 规则本身：温控波动按温升、其余按延误、「无异常」不入选；
  ② **两侧消费者不得漂移**——同一份输入喂给报告侧与看板侧，严重度必须逐个相同。
     这是这次收口真正买到的东西：原先即使两处实现漂了，也不会有任何测试变红。
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import anomaly
from src import config as C
from src import transport_decisions as TD
from src.dashboard import components as UI


def _anomalies() -> pd.DataFrame:
    """一份含四类异常 + 背景「无异常」的最小异常表。"""
    return pd.DataFrame({
        "order_id": ["A1", "A2", "A3", "A4", "A5", "A6"],
        "date": pd.to_datetime(["2026-06-01", "2026-06-01", "2026-06-02",
                                "2026-06-02", "2026-06-03", "2026-06-03"]),
        "region": ["R07", "R01", "R07", "R02", "R01", "R07"],
        "poi_id": ["P001", "P002", "P003", "P004", "P005", "P006"],
        "vehicle_id": ["V01", "V02", "V03", "V04", "V05", "V06"],
        "anomaly_type": ["晚点", C.TEMP_ANOMALY_TYPE, "故障", "拥堵",
                         C.NO_ANOMALY, C.TEMP_ANOMALY_TYPE],
        # 两条温控单刻意做成「延误大但温升小」与「延误小但温升大」，
        # 好让「按延误排会把温度埋点埋掉」这件事可断言
        "delay_min": [40.0, 8.0, 90.0, 10.0, 0.0, 2.0],
        "handling_min": [5.0, 2.0, 8.0, 3.0, 0.0, 4.0],
        # 温升 = temp_max_c − 设定点；两个温控单分别温升 4 与 9
        "temp_max_c": [4.0, C.CABIN_TEMP_SETPOINT_C + 4.0, 3.0, 2.0,
                       C.CABIN_TEMP_SETPOINT_C, C.CABIN_TEMP_SETPOINT_C + 9.0],
    })


class TestSeverityRule:
    def test_excludes_the_background_row(self):
        """「无异常」不是案例，是背景——凡按异常聚合处都要排除它，
        否则异常强度会被自身的分母稀释。"""
        got = anomaly.with_severity(_anomalies())
        assert C.NO_ANOMALY not in set(got["anomaly_type"])
        assert len(got) == 5

    def test_delay_types_use_delay_minutes(self):
        got = anomaly.with_severity(_anomalies()).set_index("order_id")
        assert got.loc["A1", "severity"] == pytest.approx(40.0)
        assert got.loc["A1", "severity_basis"] == "延误 (min)"
        assert got.loc["A3", "severity"] == pytest.approx(90.0)
        assert got.loc["A4", "severity"] == pytest.approx(10.0)

    def test_temp_type_uses_the_temperature_rise(self):
        """温控波动的延误被刻意限制在 0–10 分钟，诊断价值在温度而非时刻；
        按延误排序会让温度埋点在案例清单里消失。"""
        got = anomaly.with_severity(_anomalies()).set_index("order_id")
        assert got.loc["A2", "severity"] == pytest.approx(4.0)
        assert got.loc["A2", "severity_basis"] == "温升 (℃)"
        assert got.loc["A6", "severity"] == pytest.approx(9.0)

    def test_the_rule_is_what_saves_the_temperature_embedding(self):
        """这条正是规则存在的理由：A6 延误仅 2 分钟（按延误排会垫底），温升却有 9℃；
        A2 反过来——延误 8 分钟但温升只 4℃。按延误排两者都排在末尾、温度埋点消失；
        按各自的基准排，A6 才浮上来。"""
        got = anomaly.with_severity(_anomalies()).set_index("order_id")
        assert got.loc["A6", "delay_min"] < got.loc["A2", "delay_min"]
        assert got.loc["A6", "severity"] > got.loc["A2", "severity"]

    def test_empty_input_returns_empty_not_crash(self):
        empty = _anomalies().iloc[0:0]
        assert anomaly.with_severity(empty).empty

    def test_only_background_rows_returns_empty(self):
        bg = _anomalies()
        bg = bg[bg["anomaly_type"] == C.NO_ANOMALY]
        assert anomaly.with_severity(bg).empty

    def test_column_order_is_declared_once(self):
        got = anomaly.with_severity(_anomalies())
        assert list(got.columns) == list(anomaly.SEVERITY_COLUMNS)


class TestConsumersDoNotDrift:
    """报告侧与看板侧消费同一份规则，严重度必须逐个相同。

    这是收口真正买到的东西——原先两侧各有一份实现，即使漂了也没有测试会红。
    两侧的差别只应在**选取口径**（报告按类型分组取前 N、看板按日期取最近若干条），
    而不在规则。
    """

    def test_both_sides_report_the_same_severity(self):
        raw = _anomalies()
        report = TD.typical_anomaly_cases(raw, per_type=10).set_index("order_id")
        board = UI.warning_rows(raw, limit=10).set_index("order_id")
        assert set(report.index) == set(board.index)
        for oid in report.index:
            assert report.loc[oid, "severity"] == pytest.approx(board.loc[oid, "severity"]), oid
            assert report.loc[oid, "severity_basis"] == board.loc[oid, "severity_basis"], oid

    def test_both_sides_exclude_the_background_row(self):
        raw = _anomalies()
        assert C.NO_ANOMALY not in set(TD.typical_anomaly_cases(raw, per_type=10)["anomaly_type"])
        assert C.NO_ANOMALY not in set(UI.warning_rows(raw, limit=10)["anomaly_type"])

    def test_selection_scope_differs_by_design(self):
        """两侧的差别在选取：报告每类取前 N，看板按日期取最近若干条。"""
        raw = _anomalies()
        report = TD.typical_anomaly_cases(raw, per_type=1)
        assert len(report) == 4  # 四类各一条
        board = UI.warning_rows(raw, limit=2)
        assert list(board["date"]) == sorted(board["date"], reverse=True)  # 最近的在前
        assert len(board) == 2


class TestWarningListPresentation:
    """看板预警清单的**展示口径**：默认不截断。列名读法见 `test_dashboard_labels.py`。"""

    def test_without_a_limit_every_anomaly_is_returned(self):
        """默认不截断。看板要摆出**全部**异常——截断是替用户决定「看多少」，
        而用户要的恰恰是全部（曾经写死 limit=10，265 条里只露出 10 条）。"""
        raw = _anomalies()
        n_anomalies = int((raw["anomaly_type"] != C.NO_ANOMALY).sum())
        assert len(UI.warning_rows(raw)) == n_anomalies
        assert len(UI.warning_rows(raw, limit=2)) == 2, "给了 limit 还是要照办"
