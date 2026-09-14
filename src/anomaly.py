"""在途异常的领域规则：严重度排序。

**为什么单独一个模块。** 这套规则原先在两处各写一遍——报告侧
（`transport_decisions.anomaly_cases`，有 5 条单测守着）与看板侧
（`dashboard.components.warning_rows`，零测试）。两处的 docstring 都写着「同一套规则」，
但没有任何东西**强制**它们一致：改一处的阈值、或把「温升」改成别的基准，另一处不会红。
这正是本仓库要消灭的那类重复——口径是约定、不是代码。

放在 `src/anomaly.py` 而不是任一消费方，是为了让两侧都能引用而不产生层级倒置：
看板基建不该为了一个排序规则去 import 一个会读产物、跑统计的计算模块。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C


def with_severity(anomalies: pd.DataFrame) -> pd.DataFrame:
    """给在途异常表补 `severity` 与 `severity_basis` 两列，并剔除「无异常」。

    严重度规则：
      - 晚点 / 故障 / 拥堵 → `delay_min`（延误越长越严重）；
      - 温控波动 → `temp_max_c − 设定点`（温升越高越严重）。

    温控波动单列一套规则是有意的：它的延误被刻意限制在 0–10 分钟，诊断价值在**温度**而不在
    时刻，用延误排序会把它排到末尾、让温度埋点在案例清单里消失。

    「无异常」不入选——它不是案例，是背景。此函数**只加列不排序**：报告要按异常类型分组取
    每类前 N，看板要按日期取最近若干条，两者是展示口径的差别，不该被规则本身钉死。
    """
    df = anomalies[anomalies["anomaly_type"] != C.NO_ANOMALY].copy()
    if df.empty:
        return df
    is_temp = df["anomaly_type"] == C.TEMP_ANOMALY_TYPE
    df["severity"] = np.where(
        is_temp, df["temp_max_c"] - C.CABIN_TEMP_SETPOINT_C, df["delay_min"]
    )
    df["severity_basis"] = np.where(is_temp, "温升 (℃)", "延误 (min)")
    return df


#: 补了严重度列之后的列序（两侧展示口径共用，避免各列一份）。
SEVERITY_COLUMNS: tuple[str, ...] = (
    "order_id", "date", "region", "poi_id", "vehicle_id", "anomaly_type",
    "delay_min", "handling_min", "temp_max_c", "severity", "severity_basis",
)
