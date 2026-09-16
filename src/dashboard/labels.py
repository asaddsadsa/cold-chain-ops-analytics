"""看板表格的列名读法：**一份映射，所有表格共用**。

产物里的列名是英文的（`distance_km`、`n_orders`、`time_window_rate`），那是**数据层的
契约**——`data.py` 的登记表按这些名字声明 `usecols`，KPI 脚本按这些名字落盘，改名要动整条
链路。但**给人看的表头**不该是它们：一张写着 `distance_km` 的表，读者要先去翻台账才知道
那一列是什么。

所以读法单独住在这里，只做一件事：`产物列名 → 中文表头`。

三条规矩：

1. **同一列在任何表里都读同一个名字。** 漂移的代价是把「同一个东西」读成两个东西——
   `distance_km` 在运输页叫「里程 (km)」、在驾驶舱叫「距离」，读者会以为是两个指标。
   收在一处才守得住，散在各页面就地 `.rename()` 是守不住的（重构前正是那样：四处 ad-hoc
   改名 + 十几张表裸着英文列名）。
2. **只登记产物列名。** 页面自己聚合出来的列（`groupby(...).agg(["size","mean"])`）不在
   这里——它的名字由聚合它的那行代码就近取，写在这里会变成「`size` 永远叫运单数」这种
   别处不成立的绑定。
3. **没登记的英文列名直接报错**，不回退成英文表头。一个英文表头就是看板没做完的样子，
   静默放过它，它就一直在那儿（这条与 `data.py`「缺产物即报错，不回退」同源）。
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd

#: 产物列名 → 中文表头。措辞与报告侧、CONTEXT.md 的术语对齐（「时间窗达成率」不是「准时率」
#: ——CONTEXT.md 里「准时率」是两个必须分开的口径，表格里混用会把它们糊成一个）。
COLUMN_LABELS: Mapping[str, str] = {
    # --- 通用 -----------------------------------------------------------------
    "date": "日期",
    "region": "片区",
    # --- 数据层 A：仓内日表 ---------------------------------------------------
    "inventory_accuracy": "库存准确率",
    "abs_diff_value": "差异金额 (元)",
    "book_value": "账面金额 (元)",
    "stocktake_discrepancy_rate": "盘点差异率",
    "receipt_timeliness_rate": "收货及时率",
    "picking_lines_per_hour": "拣货行数/小时",
    # --- 数据层 A：ABC / 库位 / 分时段 ----------------------------------------
    "rank": "排名",
    "sku_id": "SKU",
    "abc_class": "ABC 类别",
    "outbound_lines": "出库行数",
    "share": "占比",
    "cum_share": "累计占比",
    "loc_id": "库位",
    "row": "排",
    "col": "列",
    "zone": "库区",
    "walk_dist_m": "行走距离 (m)",
    "hour": "时段",
    "sec_per_line": "秒/行",
    "n_lines": "行数",
    # --- 数据层 A：盘点差异下钻 -----------------------------------------------
    "category": "品类",
    "rate": "差异率",
    "n_diff": "差异记录数",
    "n_records": "记录数",
    # --- 数据层 B：Olist 州下钻 -----------------------------------------------
    "customer_state": "州",
    "n_orders": "订单数",
    "ontime_rate": "准时率",
    "delay_rate": "延迟率",
    "avg_fulfillment_days": "平均履约天数 (天)",
    # --- 运输日表 -------------------------------------------------------------
    "n_ontime": "准时单数",
    "n_orders_served": "已服务单数",
    "n_orders_unserved": "未服务单数",
    "n_orders_total": "订单总数",
    "time_window_rate": "时间窗达成率",
    "n_trips": "趟次",
    "load_rate_mean": "满载率均值",
    "total_distance_km": "总里程 (km)",
    "diesel_per_order": "柴油单均 (元)",
    "ev_per_order": "纯电单均 (元)",
    # --- 运输趟次 / 方案对照 ---------------------------------------------------
    "trip_id": "趟次",
    "plan": "方案",
    "mode": "动力",
    "n_stops": "点位数",
    "load_kg": "载重 (kg)",
    "load_rate": "满载率",
    "distance_km": "里程 (km)",
    "duration_min": "时长 (min)",
    "trip_cost": "趟成本 (元)",
    "total_self_cost": "自营总成本 (元)",
    "total_huolala_cost": "货拉拉总报价 (元)",
    "delta_pct_vs_huolala": "自营 vs 货拉拉 (%)",
    "n_trips_self_cheaper": "自营更省趟数",
    "n_trips_outsource_cheaper": "外包更省趟数",
    "n_trips_tie": "持平趟数",
    "best_mode": "自营较省模式",
    "self_best_mode": "自营较省模式",
    "self_diesel_cost": "自营-柴油 (元)",
    "self_ev_cost": "自营-纯电 (元)",
    "self_cost": "自营-较省者 (元)",
    "huolala_cost": "货拉拉 (元)",
    "verdict": "结论",
    # --- 周度异常表 -----------------------------------------------------------
    "week": "周",
    "week_start": "周起始",
    "week_end": "周结束",
    "n_shipments": "运单数",
    "anomaly_rate": "异常率",
    "friday_late_rate": "周五晚点率",
    "other_late_rate": "其他工作日晚点率",
    "r07_anomaly_rate": "R07 片区异常率",
    "other_anomaly_rate": "其他片区异常率",
    "temp_compliance_rate": "温控达标率",
    # --- 在途异常表 -----------------------------------------------------------
    "order_id": "运单",
    "poi_id": "门店",
    "vehicle_id": "车辆",
    "anomaly_type": "异常类型",
    "delay_min": "延误（分钟）",
    "handling_min": "处理（分钟）",
    "temp_max_c": "最高温（℃）",
    "severity": "严重度",
    "severity_basis": "严重度口径",
}

#: 列名里有没有中文字符。判据是「有没有中文」而不是「有没有 ASCII」——`ABC 类别`、`R07 片区
#: 异常率` 这类混写列名本来就是给人读的，不该被要求再登记一次。
_CJK = re.compile(r"[一-鿿]")

#: 登记过的读法本身也是可读的：`SKU` 没有中文字符，但它就是这个项目里该显示的名字。
_READABLE = frozenset(COLUMN_LABELS.values())


def is_readable(name: object) -> bool:
    """这个列名能不能**直接端给读者**：带中文，或是登记过的读法（如 `SKU`）。"""
    text = str(name)
    return bool(_CJK.search(text)) or text in _READABLE


def chinese_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把表头换成中文读法后返回。**只动列名，不动数据、顺序与索引。**

    一列要么**已经可读**（中文列名、`SKU` 这种登记过的术语），要么**登记过英文名**等着被
    替换；两者都不是就抛 `KeyError`（见模块 docstring 第 3 条）。
    """
    unknown = sorted(str(c) for c in df.columns
                     if not is_readable(c) and str(c) not in COLUMN_LABELS)
    if unknown:
        raise KeyError(f"这些列没有中文读法：{unknown}；"
                       "请加进 src/dashboard/labels.py 的 COLUMN_LABELS")
    return df.rename(columns=COLUMN_LABELS)
