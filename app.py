"""运营驾驶舱（模块三入口，12 号票，需求 43）。

一屏回答「现在的运营状态怎么样」：四张带环比的 KPI 卡片、选中区间的核心指标趋势、
最新异常预警清单。全部数字来自 `data/processed/` 的产物文件，页面里没有任何硬编码数值。

运行：在项目根执行 `streamlit run app.py`
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="运营驾驶舱 · 区域仓配中心", page_icon="🚚", layout="wide")

from src import config as C  # noqa: E402
from src.dashboard import components as UI  # noqa: E402
from src.dashboard import data as D  # noqa: E402
from src.dashboard import filters as F  # noqa: E402
from src.dashboard import kpis, theme  # noqa: E402

theme.register_plotly_template()
D.require_artifacts()

_first, _last = D.order_window()
UI.page_header(
    "运营驾驶舱",
    f"区域冷链城配中心 · 连续 {C.SIM_DAYS} 天运营模拟（{_first.date()} ~ {_last.date()}）· "
    "数字全部来自 data/processed/ 产物文件，可逐项溯源",
)

f = F.sidebar()
UI.context_bar(f, "日期范围作用于 KPI 卡片与趋势图；配送区域作用于异常预警清单",
               "品类与运输侧无关（配送订单表无品类字段，已在侧边栏注明）")

# ---------------------------------------------------------------------------
# 一、KPI 卡片（需求 43）
# ---------------------------------------------------------------------------
wh_daily, tr_daily = D.warehouse_daily(), D.transport_daily()
metrics = kpis.build_metrics(wh_daily, tr_daily, f.window, cost_mode=f.cost_mode)
UI.kpi_cards(metrics)

prev = f.window.previous()
st.caption(
    f"环比对照区间：{prev.start.date()} ~ {prev.end.date()}（{prev.days} 天，与当前区间等长）。"
    "「时间窗达成率」为配送侧 KPI（送达落在客户时间窗内），"
    "与 Olist 侧的「真实准时交付率」是两个口径，不可互换（见 CONTEXT.md）。"
)

st.divider()

# ---------------------------------------------------------------------------
# 二、核心指标趋势（需求 43：近 30 天趋势折线）
# ---------------------------------------------------------------------------
st.subheader("核心指标趋势")
st.caption(
    "量纲不同的指标拆成**各自一根轴**的小倍数图。双 Y 轴会让两条曲线的相对位置变得任意，"
    "凭空造出数据里没有的相关性——这张页面上不会出现双 Y 轴。"
)

wh_win = kpis.slice_window(wh_daily, f.window)
tr_win = kpis.slice_window(tr_daily, f.window)

row1 = st.columns(2)
with row1[0]:
    UI.chart_block(
        UI.metric_trend(wh_win, x="date", y="inventory_accuracy",
                        label="库存准确率", unit="%", slot=0),
        caption="金额加权口径；逐日金额来自盘点记录，非逐日率的简单平均。",
        table=wh_win[["date", "inventory_accuracy", "abs_diff_value", "book_value"]],
        table_label="库存准确率数据表",
    )
with row1[1]:
    UI.chart_block(
        UI.metric_trend(tr_win, x="date", y="time_window_rate",
                        label="时间窗达成率", unit="%", slot=1),
        caption="逐单判定：当日按时单 / 当日已服务单（基线派车口径，含在途延误）。",
        table=tr_win[["date", "n_ontime", "n_orders_served", "n_orders_unserved",
                      "time_window_rate"]],
        table_label="时间窗达成率数据表",
    )

row2 = st.columns(2)
with row2[0]:
    UI.chart_block(
        UI.metric_trend(tr_win, x="date", y="load_rate_mean",
                        label="满载率", unit="%", slot=2, decimals=1),
        caption="当日全部趟次的实际载重 / 额定载重的平均值。",
        table=tr_win[["date", "n_trips", "load_rate_mean"]],
        table_label="满载率数据表",
    )
with row2[1]:
    cost_col = f"{f.cost_mode}_per_order"
    mode_label = kpis.MODE_LABELS[f.cost_mode]
    UI.chart_block(
        UI.metric_trend(tr_win, x="date", y=cost_col,
                        label=f"单均成本（{mode_label}）", unit="元", slot=3),
        caption=f"当日总成本 / 当日订单数；口径由侧边栏「成本口径」决定（当前：{mode_label}）。",
        table=tr_win[["date", "n_orders_total", cost_col]],
        table_label="单均成本数据表",
    )

trend_table = (
    wh_win[["date", "inventory_accuracy", "stocktake_discrepancy_rate",
            "receipt_timeliness_rate", "picking_lines_per_hour"]]
    .merge(
        tr_win[["date", "n_orders_total", "n_orders_served", "n_orders_unserved",
                "n_trips", "total_distance_km", "time_window_rate", "load_rate_mean",
                "diesel_per_order", "ev_per_order"]],
        on="date", how="outer",
    )
    .sort_values("date")
)
with st.expander("趋势数据表（上图全部数值）"):
    st.dataframe(trend_table, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# 三、最新异常预警清单（需求 43）
# ---------------------------------------------------------------------------
st.subheader("最新异常预警")
anom = F.apply_regions(D.anomalies(), f)
in_win = anom[
    (anom["date"] >= f.window.start.normalize()) & (anom["date"] <= f.window.end.normalize())
]
rows = UI.warning_rows(in_win, limit=10)
st.caption(
    f"区间内运单 **{len(in_win):,}** 单，其中异常 **{int((in_win['anomaly_type'] != '无异常').sum()):,}** 单"
    f"（异常率 {(in_win['anomaly_type'] != '无异常').mean():.2%}）。"
    "严重度：晚点/故障/拥堵取延误分钟，温控波动取温升——后者延误被限制在 0–10 分钟，"
    "诊断价值在温度而非时刻。"
)
UI.warning_list(rows)
if not rows.empty:
    with st.expander("预警清单数据表"):
        st.dataframe(rows, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# 四、数据来源与口径
# ---------------------------------------------------------------------------
with st.expander("数据来源与口径（每个数字的追溯入口）"):
    st.markdown(
        f"""
| 读数 | 产物文件 | 生成命令 | 数据类别 |
|---|---|---|---|
| 库存准确率 | `{C.WAREHOUSE_DAILY_CSV.name}` | `python -m src.warehouse_kpi` | 过程仿真 |
| 时间窗达成率 / 满载率 / 单均成本 | `{C.TRANSPORT_DAILY_CSV.name}` | `python -m src.transport_optimize` | 过程仿真（派生指标） |
| 异常预警清单 | `{C.DELIVERY_ANOMALIES_CSV.name}` | `python -m src.gen_delivery_data` | 情景假设（总体水平由 Olist 实测延迟率标定） |

- 逐日序列是为看板专门落盘的产物：聚合值撑不起一条趋势线，而**重跑聚合口径**会让看板
  与已发表的报告出现两套数字。日表只依赖确定性的基线贪心，与代表日的带时限搜索无关。
- 每个参数的取值、类别、来源与口径见项目根的 `data_sources_ledger.md`。
- 本页不显示 0 或占位值来掩盖缺产物；产物不全时直接列出缺哪个文件、该跑哪条命令。
"""
    )
