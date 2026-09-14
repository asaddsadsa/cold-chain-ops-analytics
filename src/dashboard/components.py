"""看板可复用组件（12 号票）：KPI 卡片、趋势图、异常预警清单、图表块。

这一层把三条图表纪律固化下来，页面里就不必每处重述：

1. **每张图都配表格视图**（`chart_block`）。浅色配色里青/黄/品红三槽对底色的对比度
   低于 3:1，规范要求以「可见标签或表格视图」缓解——这是必需项，不是加分项。
   同时它也满足「悬停提示只能增强、不能是读到数值的唯一途径」。
2. **单序列图不放图例**（标题即序列名），端点直接标数值；**多序列图必须有图例**
   （≥2 序列时身份不能只靠颜色）。
3. **绝不用双 Y 轴**。量纲不同的指标一律拆成小倍数图，各自一根轴——两个 y 轴的相对
   位置是任意的，会凭空造出数据里没有的相关性。
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import anomaly
from src import config as C
from src.dashboard import theme
from src.dashboard.kpis import Metric

_ARROW = {"up": "▲", "down": "▼", "flat": "—", "na": "—"}


# ---------------------------------------------------------------------------
# 页头
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def context_bar(f, applied: str, extra: str | None = None) -> None:
    """页面顶部的上下文条：当前筛选条件 + 本页实际吃哪几个筛选器。"""
    st.caption(f.scope_note(applied) + (f" ｜ {extra}" if extra else ""))


# ---------------------------------------------------------------------------
# KPI 卡片
# ---------------------------------------------------------------------------
def delta_badge(m: Metric) -> str:
    """环比的箭头 + 文案 + 颜色。**颜色不单独表意**：箭头与文字同时给出。"""
    t = theme.tokens()
    d = m.delta
    if d is None:
        return f'<span style="color:{t["muted"]}">环比 —（无对照区间）</span>'
    arrow = _ARROW[m.direction]
    if m.unit == "%":
        text = f"{d * 100:+.2f} pp"
    else:
        text = f"{d:+,.2f} {m.unit}"
    good = m.is_good
    if good is None:
        color = t["muted"]
    else:
        color = t["delta_good"] if good else theme.STATUS["critical"]
    verdict = "" if good is None else ("改善" if good else "恶化")
    return f'<span style="color:{color};font-weight:600">{arrow} {text}</span>' \
           f'<span style="color:{t["secondary_ink"]}"> 环比{verdict}</span>'


def kpi_cards(metrics: list[Metric]) -> None:
    """一行 KPI 卡片：数值用比例数字（不用等宽数字，大字号下会显松散）。"""
    t = theme.tokens()
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
<div style="border:1px solid {t['border']};border-radius:10px;padding:14px 16px;
            background:{t['surface']};height:100%">
  <div style="color:{t['secondary_ink']};font-size:13px;margin-bottom:4px">{m.label}</div>
  <div style="color:{t['primary_ink']};font-size:30px;line-height:1.15;font-weight:600">
    {m.format(m.value)}
  </div>
  <div style="font-size:12px;margin-top:6px">{delta_badge(m)}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.caption(f"口径：{m.basis}")


# ---------------------------------------------------------------------------
# 图表块
# ---------------------------------------------------------------------------
def chart_block(fig: go.Figure, *, caption: str | None = None,
                table: pd.DataFrame | None = None, table_label: str = "查看数据表",
                height: int = 300) -> None:
    """图 + 口径说明 + 表格视图。表格视图是规范要求的缓解手段，缺它就等于图不可读。"""
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    if caption:
        st.caption(caption)
    if table is not None and not table.empty:
        with st.expander(table_label):
            st.dataframe(table, width="stretch", hide_index=True)


def metric_trend(
    df: pd.DataFrame, *, x: str, y: str, label: str, unit: str,
    slot: int = 0, decimals: int = 2, axis_format: str | None = None,
) -> go.Figure:
    """单指标趋势线：细线 + 端点直接标数值 + 十字准星悬停。单序列故不放图例。"""
    t = theme.tokens()
    color = theme.series(slot)
    scale = 100.0 if unit == "%" else 1.0
    fig = go.Figure(
        go.Scatter(
            x=df[x], y=df[y] * scale, mode="lines+markers", name=label,
            line={"width": 2, "color": color},
            marker={"size": 6, "color": color},
            hovertemplate="%{x|%Y-%m-%d}<br>" + label + " %{y:." + str(decimals) + "f}" + unit
            + "<extra></extra>",
        )
    )
    if len(df):
        fig.add_annotation(
            x=df[x].iloc[-1], y=df[y].iloc[-1] * scale,
            text=f"{df[y].iloc[-1] * scale:.{decimals}f}{unit}",
            showarrow=False, xanchor="right", yanchor="bottom",
            font={"color": t["secondary_ink"], "size": 12},
        )
    fig.update_layout(
        title={"text": label, "font": {"size": 14, "color": t["primary_ink"]}, "x": 0},
        showlegend=False, hovermode="x unified",
        yaxis={"title": unit, "tickformat": axis_format},
        margin={"l": 56, "r": 16, "t": 44, "b": 36},
    )
    return fig


# ---------------------------------------------------------------------------
# 异常预警清单
# ---------------------------------------------------------------------------
def warning_rows(anomalies: pd.DataFrame, *, limit: int = 8) -> pd.DataFrame:
    """最近异常预警清单：按「严重度」取最近的若干条，规则显式。

    严重度 = 延误分钟（温控波动按温升），与 11 号票典型案例清单**同一套规则**——规则本身
    住在 `src/anomaly.py::with_severity`，本函数只决定**展示口径**：按日期倒序取最近若干条
    （报告侧是按异常类型分组、每类取前 N）。同分按日期与订单号兜底，保证同样输入下清单逐条
    可复现（不随排序抖动）。
    """
    if anomalies.empty:
        return anomalies
    df = anomaly.with_severity(anomalies)
    if df.empty:
        return df
    df = df.sort_values(["date", "severity", "order_id"],
                        ascending=[False, False, True], kind="stable")
    # 看板把 `date` 提到最前（清单按时间读），其余列序与报告侧共用同一份声明
    keep = ["date", *[c for c in anomaly.SEVERITY_COLUMNS if c != "date"]]
    return df[keep].head(limit).reset_index(drop=True)


def warning_list(rows: pd.DataFrame) -> None:
    """把预警清单渲染成带状态色的条目。状态色**永远配图标与文字**，不靠颜色单独表意。"""
    t = theme.tokens()
    if rows.empty:
        st.markdown(f'<span style="color:{t["muted"]}">该区间内没有异常运单。</span>',
                    unsafe_allow_html=True)
        return
    icon = {"故障": theme.STATUS_ICON["critical"], "晚点": theme.STATUS_ICON["serious"],
            "拥堵": theme.STATUS_ICON["warning"],
            C.TEMP_ANOMALY_TYPE: theme.STATUS_ICON["warning"]}
    color = {"故障": theme.STATUS["critical"], "晚点": theme.STATUS["serious"],
             "拥堵": theme.STATUS["warning"],
             C.TEMP_ANOMALY_TYPE: theme.STATUS["warning"]}
    for r in rows.itertuples(index=False):
        # 文字一律穿墨色 token，不穿序列色；身份由旁边的状态色标记承担
        st.markdown(
            f"{icon.get(r.anomaly_type, '⚠️')} "
            f'<span style="color:{color.get(r.anomaly_type, t["muted"])};font-weight:600">'
            f"{r.anomaly_type}</span> "
            f'<span style="color:{t["secondary_ink"]}">{pd.Timestamp(r.date).date()} ｜ '
            f"{r.region} ｜ {r.vehicle_id} ｜ {r.order_id}</span> "
            f'<span style="color:{t["muted"]}">严重度 {r.severity:.1f} {r.severity_basis}</span>',
            unsafe_allow_html=True,
        )
