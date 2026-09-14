"""运输分析页（14 号票）：一页看清运输调度优化的全部成果。

内容（对应票据 7 条验收）：

1. Folium 真实地图绘配送路线，「优化前 / 优化后」一键切换，各点弹窗显示时间窗与到达时刻；
2. 里程 / 用车数 / 成本前后对比柱状图（数字全部引用 `baseline_vs_optimized.csv`）；
3. 满载率分布（按方案分组的小倍数直方图）；
4. 异常类型饼图 + 周度趋势折线（含周五 / R07 两个埋点），旁列池化 z 值与显著性；
5. 温控达标率仪表（达标值从异常表逐运单 `temp_compliance_rate` 求）；
6. 双模式 TCO 曲线（标注盈亏平衡里程）+「自营 vs 货拉拉」逐趟对照表。

纪律（违反即返工，见 `theme.py` / `components.py` / `charts.py` 的模块 docstring）：

- **所有读数走 `src/dashboard/data.py`**，页面里不出现任何硬编码业务数字（连口径文字里的数字也
  从产物取，用 f-string 拼出来）。
- **每张图都配表格视图**（`UI.chart_block(..., table=...)`）——浅色配色里青/黄/品红三槽对底色
  对比度低于 3:1，规范要求以表格视图缓解，这是必需项。
- **绝不双 Y 轴**：量纲不同的指标一律拆成各自一根轴的小倍数图。
- **单序列图不放图例，多序列图必须有图例。**
- **类别色按实体固定分配、不随筛选重排**：本页槽 0/1 固定给「优化前 / 优化后」，槽 2/3 固定给
  两种动力模式（柴油自购 / 纯电租赁），槽 4–7 固定给周度埋点四序列，因此同一页里「蓝」永远
  是优化前、不会因为换了筛选就变成别的实体。
- 页首顺序：`set_page_config` → `register_plotly_template` → `require_artifacts` → 页头 →
  侧边栏 → 上下文条；缺产物直接停下报错，**不显示 0**。

运行：在项目根执行 `streamlit run app.py`，从侧边栏进入本页。
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="运输分析 · 区域仓配中心", page_icon="🚚", layout="wide")

import folium  # noqa: E402
from streamlit.components.v1 import html as st_html  # noqa: E402

from src import config as C  # noqa: E402
from src.dashboard import charts as CH  # noqa: E402
from src.dashboard import components as UI  # noqa: E402
from src.dashboard import data as D  # noqa: E402
from src.dashboard import filters as F  # noqa: E402
from src.dashboard import kpis, theme  # noqa: E402

# ---------------------------------------------------------------------------
# 本页固定的「实体 → 类别色槽位」映射（与 theme 的校验色板配套，见 theme.py）。
# 槽位是身份编码：一旦定下就不随筛选后的次序重排。
# ---------------------------------------------------------------------------
PLAN_SLOT: dict[str, int] = {"baseline": 0, "optimized": 1}
PLAN_LABEL: dict[str, str] = {"baseline": "优化前", "optimized": "优化后"}
MODE_SLOT: dict[str, int] = {"diesel": 2, "ev": 3}
#: 对比表里两种自营模式的指标名前缀（产物 `baseline_vs_optimized.csv` 的 metric 列）
MODE_PREFIX: dict[str, str] = {"diesel": "柴油", "ev": "纯电"}


# ---------------------------------------------------------------------------
# 基建
# ---------------------------------------------------------------------------
theme.register_plotly_template()
D.require_artifacts((
    "transport_kpi",              # 代表日基线与优化 KPI、时间窗口径、代表日是哪天
    "transport_comparison",       # 前后对比表（里程/用车数/成本/比率）
    "transport_trips",            # 逐趟明细（满载率分布）
    "transport_tco",              # 双模式 TCO 曲线、盈亏平衡里程、外包对照
    "transport_weekly",           # 周度异常表（含周五 / R07 埋点列）
    "transport_routes_optimized",  # 路线 GeoJSON（优化后）
    "transport_routes_baseline",   # 路线 GeoJSON（优化前，地图 radio 可切）
    "transport_anomaly_points",   # 埋点池化检验（周五 / R07 / 雨日）
    "poi",                        # POI 名称（地图弹窗的显示名）
    "anomalies",                  # 在途异常表（异常构成 / 温控达标率 / 片区）
))


# ---------------------------------------------------------------------------
# 图表小工具（复用 CH 构造器，只补齐本页特有的槽位与标题约束）
# ---------------------------------------------------------------------------
def _titled(fig: go.Figure, text: str) -> go.Figure:
    """给 CH 构造器产出的图补一个左对齐的中文标题（构造器本身不设标题）。"""
    t = theme.tokens()
    fig.update_layout(title={"text": text, "font": {"size": 14, "color": t["primary_ink"]}, "x": 0},
                      margin={"l": 64, "r": 16, "t": 48, "b": 48})
    return fig


def _delta_note(change_pct: float, *, lower_is_better: bool) -> str:
    """把对比表的 `change_pct`（负数=下降）翻成「方向 + 好坏的判词」。

    `change_pct` 的符号只表示升/降，**不表示好坏**——里程、成本、用车数下降是好事，比率类
    上升才是好事。两者分开说，读者不必自己反推符号语义。
    """
    if abs(change_pct) < 0.005:
        return f"优化前后持平（{change_pct:.2f}%），方向无变化"
    down = change_pct < 0
    good = down == lower_is_better
    return (f"{'↓' if down else '↑'} {abs(change_pct):.2f}%（{'下降' if down else '上升'}"
            f" = {'改善' if good else '变差'}）")


def _compare_bars(baseline: float, optimized: float, *, unit: str, decimals: int,
                  change_pct: float, lower_is_better: bool) -> go.Figure:
    """同一指标「优化前 / 优化后」两根柱。

    两根柱是**两个实体**（两个方案），按 PLAN_SLOT 固定上色；x 轴标签已经写明身份，
    身份不靠颜色单独承担，故不放图例（单 trace）。
    """
    fig = go.Figure(
        go.Bar(
            x=[PLAN_LABEL["baseline"], PLAN_LABEL["optimized"]],
            y=[baseline, optimized],
            marker={"color": [theme.series(PLAN_SLOT["baseline"]),
                              theme.series(PLAN_SLOT["optimized"])], "line": {"width": 0}},
            text=[f"{baseline:,.{decimals}f}", f"{optimized:,.{decimals}f}"],
            textposition="outside", cliponaxis=False,
            hovertemplate=f"%{{x}}<br>%{{y:,.{decimals}f}}{unit}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False, bargap=0.35,
                      yaxis={"title": unit}, xaxis={"title": ""})
    return fig


def _grouped_compare(categories, baseline_vals, optimized_vals, *, unit: str,
                     decimals: int = 2, change_pcts=None) -> go.Figure:
    """多指标两方案分组柱：同量纲放在一张图（本页只在「比率 (%)」里用到，全为百分数）。"""
    fig = go.Figure()
    fig.add_bar(x=list(categories), y=list(baseline_vals), name=PLAN_LABEL["baseline"],
                marker={"color": theme.series(PLAN_SLOT["baseline"]), "line": {"width": 0}},
                text=[f"{v:,.{decimals}f}" for v in baseline_vals], textposition="outside",
                cliponaxis=False,
                hovertemplate=f"优化前 · %{{x}}<br>%{{y:,.{decimals}f}}{unit}<extra></extra>")
    fig.add_bar(x=list(categories), y=list(optimized_vals), name=PLAN_LABEL["optimized"],
                marker={"color": theme.series(PLAN_SLOT["optimized"]), "line": {"width": 0}},
                text=[f"{v:,.{decimals}f}" for v in optimized_vals], textposition="outside",
                cliponaxis=False,
                hovertemplate=f"优化后 · %{{x}}<br>%{{y:,.{decimals}f}}{unit}<extra></extra>")
    fig.update_layout(barmode="group", showlegend=True, bargap=0.3, bargroupgap=0.08,
                      yaxis={"title": unit}, xaxis={"title": ""})
    return fig


def _multi_line(x, series: dict, slots: dict, *, y_title: str, unit: str = "") -> go.Figure:
    """多序列折线（同量纲、一根轴）。

    复用 `charts.dual_line_chart` 后**按实体固定槽位重上色**：构造器从槽 0 起顺序取色，而本页
    槽 0/1 已固定给「优化前/后」、2/3 给两种动力模式；不重上色的话，同一页会出现
    「蓝 = 优化前」与「蓝 = 周度某序列」两种读法，正是配色规范要禁止的漂移。
    """
    fig = CH.dual_line_chart(x, series, y_title=y_title, unit=unit)
    for i, name in enumerate(series):
        c = theme.series(slots[name])
        fig.data[i].line.color = c
        fig.data[i].marker.color = c
    return fig


def _hhmm(minutes) -> str:
    """当日分钟数 → HH:MM（时间窗与到达时刻在产物里都是分钟数）。"""
    if minutes is None:
        return "—"
    m = int(round(float(minutes)))
    return f"{m // 60:02d}:{m % 60:02d}"


# ---------------------------------------------------------------------------
# 页头与筛选
# ---------------------------------------------------------------------------
_kpi = D.transport_kpi()
UI.page_header(
    "运输分析",
    f"区域冷链城配 · 代表日 {_kpi.representative_day}（{C.SIM_DAYS} 天中订单量最大的工作日）的"
    "基线派车 vs OR-Tools 优化 · 全部数字来自 data/processed/ 产物文件，可逐项溯源",
)

f = F.sidebar()
UI.context_bar(
    f,
    "日期范围、配送区域、成本口径",
    "日期范围与配送区域作用于**异常构成饼图、温控仪表、周度埋点图**；成本口径决定前后对比里"
    "用哪一种动力的成本行。代表日口径的产物（路线地图、前后对比、满载率分布、TCO 与外包对照）"
    "没有逐日/片区维度，日期与片区筛选对它们不生效——如实声明，不假装生效。",
)

st.caption(
    f"**关键口径**：① 路线地图、前后对比、满载率分布、TCO 与外包对照均为**代表日 "
    f"{_kpi.representative_day} 单日**口径，OR-Tools 的完整精算只在该日进行；其余 "
    f"{_kpi.all_days} 天由基线贪心覆盖全量里程/成本。"
    f"② 「时间窗达成率」口径为**{_kpi.baseline.time_window_basis}**，与 Olist 侧的"
    "「真实准时交付率」是两个口径，不可互换（见 CONTEXT.md）。"
)

st.divider()

# ---------------------------------------------------------------------------
# 一、配送路线地图（Folium，优化前 / 优化后一键切换）
# ---------------------------------------------------------------------------
st.subheader("配送路线地图")
st.caption(
    "Folium 真实地图（OpenStreetMap 底图），蓝色为「优化前」、橙色为「优化后」——"
    "与全页其余图表的方案槽位一致。每条折线是一趟车的行驶路径（DC → 各门店 → 回 DC）；"
    "圆点是配送门店，点开可看该点的**时间窗**与**计划到达时刻**。时间窗与到达时刻在产物里都是"
    "当日分钟数，这里转成 HH:MM。本页未安装 `streamlit-folium`，故用 `components.v1.html` 渲染"
    "Folium 自带 HTML。"
)

_plan = st.radio(
    "路线方案", ["baseline", "optimized"],
    format_func=lambda k: PLAN_LABEL[k], horizontal=True, key="route_plan",
)

# 门店名一律经取数层取——页面自行读盘会绕开 mtime 缓存键，产物更新后这一处不会自动失效
_poi = D.artifact("poi")
_poi_name = dict(zip(_poi["poi_id"], _poi["name"]))

_geojson = D.routes_geojson(_plan)
_features = _geojson.get("features", [])
_route_color = theme.series(PLAN_SLOT[_plan])

_all_lng = [x for ft in _features for x, _ in ft["geometry"]["coordinates"]]
_all_lat = [y for ft in _features for _, y in ft["geometry"]["coordinates"]]
_center = ([sum(_all_lat) / len(_all_lat), sum(_all_lng) / len(_all_lng)]
           if _all_lat else [C.DC_FALLBACK_LAT, C.DC_FALLBACK_LNG])

_fmap = folium.Map(location=_center, zoom_start=11, tiles="OpenStreetMap", control_scale=True)
for _ft in _features:
    _p = _ft["properties"]
    _coords = [(y, x) for x, y in _ft["geometry"]["coordinates"]]
    folium.PolyLine(
        _coords, color=_route_color, weight=3, opacity=0.85,
        tooltip=(f"{_p['trip_id']} ｜ {int(_p['n_stops'])} 点 ｜ "
                 f"{float(_p['distance_km']):.1f} km ｜ 载重 {float(_p['load_kg']):.0f} kg"),
    ).add_to(_fmap)
    # 坐标为 [DC, 各点…, DC]；门店段取第 1..n_stops 个
    for _i, (_x, _y) in enumerate(_ft["geometry"]["coordinates"][1:1 + int(_p["n_stops"])]):
        _pid = _p["poi_ids"][_i]
        _ws, _we = float(_p["window_start_min"][_i]), float(_p["window_end_min"][_i])
        _arr = _p["arrival_min"][_i]
        _arr_v = float(_arr) if _arr is not None else None
        if _arr_v is None:
            _verdict = "计划到达缺记录"
        elif _arr_v < _ws:
            _verdict = f"早到 {_ws - _arr_v:.0f} 分钟"
        elif _arr_v > _we:
            _verdict = f"超窗 {_arr_v - _we:.0f} 分钟"
        else:
            _verdict = "窗内"
        folium.CircleMarker(
            [_y, _x], radius=5, color=_route_color, weight=2,
            fill=True, fill_color=_route_color, fill_opacity=0.85,
            tooltip=f"{_poi_name.get(_pid, _pid)}（{_pid}）",
            popup=folium.Popup(
                f"<b>{_poi_name.get(_pid, _pid)}</b><br>"
                f"POI {_pid} ｜ 订单 {int(_p['n_orders'][_i])} 单<br>"
                f"时间窗 {_hhmm(_ws)}–{_hhmm(_we)}<br>"
                f"计划到达 {_hhmm(_arr_v)}（{_verdict}）",
                max_width=260,
            ),
        ).add_to(_fmap)
# DC 单独标一个中性色的方点（中性灰属铬色，不占用类别槽位）
folium.CircleMarker(
    _coords[0] if _features else _center, radius=9, color=theme.tokens()["secondary_ink"],
    weight=2, fill=True, fill_color=theme.tokens()["surface"], fill_opacity=1.0,
    tooltip="区域仓配中心（DC）", popup="冷链城配 DC：代表日各趟车由此发车并回场",
).add_to(_fmap)

st_html(_fmap._repr_html_(), height=520)

_map_trips = D.artifact("transport_trips")
_map_tbl = (_map_trips[_map_trips["plan"] == _plan]
            [["trip_id", "mode", "n_stops", "n_orders", "load_kg", "load_rate",
              "distance_km", "duration_min", "trip_cost"]]
            .rename(columns={"trip_id": "趟次", "mode": "动力", "n_stops": "点位数",
                             "n_orders": "订单数", "load_kg": "载重 (kg)",
                             "load_rate": "满载率", "distance_km": "里程 (km)",
                             "duration_min": "时长 (min)", "trip_cost": "趟成本 (元)"}))
st.caption(
    f"当前方案：**{PLAN_LABEL[_plan]}**，共 {len(_map_tbl)} 趟、"
    f"总里程 {_map_tbl['里程 (km)'].sum():,.2f} km、"
    f"总成本 {_map_tbl['趟成本 (元)'].sum():,.2f} 元（逐趟明细见下表）。"
)
with st.expander(f"{PLAN_LABEL[_plan]}逐趟明细数据表"):
    st.dataframe(_map_tbl, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# 二、里程 / 用车数 / 成本前后对比（数字引用 baseline_vs_optimized.csv）
# ---------------------------------------------------------------------------
st.subheader("里程 / 用车数 / 成本前后对比")
st.caption(
    "全部数字引用产物 `baseline_vs_optimized.csv`，不在此重算。对比表里**负数表示该指标下降**："
    "里程、用车数、成本下降是好事，比率类上升才是好事——图的副标题已把方向与好坏分开写明。"
)

_cmp = D.artifact("transport_comparison")


def _cmp_row(prefix: str) -> pd.Series | None:
    """按 metric 前缀取对比表的一行（如「总里程」「柴油总成本」）。"""
    sub = _cmp[_cmp["metric"].astype(str).str.startswith(prefix)]
    return None if sub.empty else sub.iloc[0]


def _rows_table(metric: str, row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([{
        "指标": metric,
        "优化前": float(row["baseline"]),
        "优化后": float(row["optimized"]),
        "变化 (%)": float(row["change_pct"]),
    }])


_c0, _c1 = st.columns(2)
with _c0:
    _r = _cmp_row("总里程")
    UI.chart_block(
        _compare_bars(float(_r["baseline"]), float(_r["optimized"]), unit="km", decimals=2,
                      change_pct=float(_r["change_pct"]), lower_is_better=True),
        caption=f"总里程：{_delta_note(float(_r['change_pct']), lower_is_better=True)}",
        table=_rows_table("总里程 (km)", _r), table_label="总里程数据表",
    )
with _c1:
    _r = _cmp_row("用车数")
    _trips_by_plan = D.artifact("transport_trips").groupby("plan").size().to_dict()
    UI.chart_block(
        _compare_bars(float(_r["baseline"]), float(_r["optimized"]), unit="台", decimals=0,
                      change_pct=float(_r["change_pct"]), lower_is_better=True),
        caption=f"用车数：{_delta_note(float(_r['change_pct']), lower_is_better=True)}"
                f"（用车数 = 趟次数：优化前 {_trips_by_plan.get('baseline', 0)} 趟对应 "
                f"{float(_r['baseline']):.0f} 台、优化后 {_trips_by_plan.get('optimized', 0)} 趟对应 "
                f"{float(_r['optimized']):.0f} 台——代表日每台车跑一趟，日固定成本可精确摊入单趟）",
        table=_rows_table("用车数 (台)", _r), table_label="用车数数据表",
    )

_mode_lbl = kpis.MODE_LABELS[f.cost_mode]
_c2, _c3 = st.columns(2)
with _c2:
    _r = _cmp_row(MODE_PREFIX[f.cost_mode] + "总成本")
    UI.chart_block(
        _compare_bars(float(_r["baseline"]), float(_r["optimized"]), unit="元", decimals=2,
                      change_pct=float(_r["change_pct"]), lower_is_better=True),
        caption=f"代表日总成本（{_mode_lbl}，由侧边栏成本口径决定）："
                f"{_delta_note(float(_r['change_pct']), lower_is_better=True)}",
        table=_rows_table(f"{_mode_lbl}总成本 (元)", _r), table_label="总成本数据表",
    )
with _c3:
    _r = _cmp_row(MODE_PREFIX[f.cost_mode] + "单均成本")
    UI.chart_block(
        _compare_bars(float(_r["baseline"]), float(_r["optimized"]), unit="元/单", decimals=2,
                      change_pct=float(_r["change_pct"]), lower_is_better=True),
        caption=f"代表日单均成本（{_mode_lbl}）："
                f"{_delta_note(float(_r['change_pct']), lower_is_better=True)}",
        table=_rows_table(f"{_mode_lbl}单均成本 (元/单)", _r), table_label="单均成本数据表",
    )

_ratio_prefixes = ("时间窗达成率", "满载率均值", "里程利用率")
_ratio_rows = [_cmp_row(p) for p in _ratio_prefixes]
_ratio_rows = [r for r in _ratio_rows if r is not None]
if _ratio_rows:
    _cats = [str(r["metric"]) for r in _ratio_rows]
    _base_pct = [float(r["baseline"]) * 100 for r in _ratio_rows]
    _opt_pct = [float(r["optimized"]) * 100 for r in _ratio_rows]
    _tw = _cmp_row("时间窗达成率")
    _dist_chg = float(_cmp_row("总里程")["change_pct"])
    _cost_chg = float(_cmp_row(MODE_PREFIX[f.cost_mode] + "总成本")["change_pct"])
    UI.chart_block(
        _grouped_compare(_cats, _base_pct, _opt_pct, unit="%", decimals=2),
        caption=(
            f"比率类指标（同一量纲「%」，共用一根轴）。时间窗达成率优化前后同为 "
            f"{float(_tw['baseline']):.2%}——**这是真实结果而非「没优化」**：两套方案都排在时间窗内，"
            "在途延误把同样那一批配送点推出了窗（详见 10 号票记录）。优化的收益体现在里程与成本"
            f"（总里程 {_dist_chg:+.2f}% / 总成本 {_cost_chg:+.2f}%，见上），而非准时率。"
        ),
        table=pd.DataFrame([{
            "指标": str(r["metric"]),
            "优化前 (%)": float(r["baseline"]) * 100,
            "优化后 (%)": float(r["optimized"]) * 100,
            "变化 (%)": float(r["change_pct"]),
        } for r in _ratio_rows]),
        table_label="比率类指标数据表",
    )

st.divider()

# ---------------------------------------------------------------------------
# 三、满载率分布（按方案分组的小倍数直方图）
# ---------------------------------------------------------------------------
st.subheader("满载率分布")
_trips = D.artifact("transport_trips")
_n_by_plan = _trips.groupby("plan").size().to_dict()
st.caption(
    "两组是同一量纲（满载率 %），**可以**画在一起；这里选择两个小倍数图，理由：两个方案的趟次"
    f"极少（优化前 {_n_by_plan.get('baseline', 0)} 趟、优化后 {_n_by_plan.get('optimized', 0)} 趟），"
    "重叠直方图会让两条轮廓互相遮挡、读不出各自形状，小倍数图"
    "各占一格则能把「优化后低收入趟被压掉、分布整体右移」看干净。两图的组距固定为 10 个百分点、"
    "横轴都锁在 0–100%，因此可并排比较（组距若各自自适应就不可比了）。"
)
_t0, _t1 = st.columns(2)
for _col, _plan_key in ((_t0, "baseline"), (_t1, "optimized")):
    _vals = _trips.loc[_trips["plan"] == _plan_key, "load_rate"] * 100
    _fig = CH.histogram(_vals, label="满载率", unit="%", slot=PLAN_SLOT[_plan_key],
                        x_title="满载率 (%)", bins=10)
    _fig.update_traces(xbins={"start": 0, "end": 100, "size": 10})
    _fig.update_xaxes(range=[0, 100])
    with _col:
        UI.chart_block(
            _titled(_fig, f"满载率分布 · {PLAN_LABEL[_plan_key]}（{len(_vals)} 趟）"),
            caption=f"{PLAN_LABEL[_plan_key]}：均值 {_vals.mean():.2f}%、中位 {_vals.median():.2f}%、"
                    f"最小 {_vals.min():.2f}%、最大 {_vals.max():.2f}%（共 {len(_vals)} 趟）。",
            table=(_trips.loc[_trips["plan"] == _plan_key, ["trip_id", "n_stops", "n_orders",
                                                            "load_kg", "load_rate",
                                                            "distance_km"]]
                   .rename(columns={"trip_id": "趟次", "n_stops": "点位数", "n_orders": "订单数",
                                    "load_kg": "载重 (kg)", "load_rate": "满载率",
                                    "distance_km": "里程 (km)"})),
            table_label=f"{PLAN_LABEL[_plan_key]}满载率数据表",
        )

st.divider()

# ---------------------------------------------------------------------------
# 四、异常构成 + 周度埋点
# ---------------------------------------------------------------------------
st.subheader("在途异常构成与周度埋点")
_anom_all = F.apply_regions(D.artifact("anomalies"), f)
_anom = kpis.slice_window(_anom_all, f.window)
_anom_only = _anom[_anom["anomaly_type"] != "无异常"]

_p0, _p1 = st.columns(2)
with _p0:
    st.markdown("**异常类型构成**")
    if _anom_only.empty:
        st.info("当前筛选区间/片区内没有异常运单，饼图无数据可画。")
    else:
        # 固定用 config 的异常类型顺序取标签 → 扇区颜色按类型固定，不随筛选后的计数次序重排
        _order = [t for t, _ in C.ANOMALY_TYPE_SHARES]
        _counts = _anom_only["anomaly_type"].value_counts()
        _labels = [t for t in _order if int(_counts.get(t, 0)) > 0]
        _values = [int(_counts[t]) for t in _labels]
        _tot = sum(_values)
        UI.chart_block(
            CH.pie_chart(_labels, _values),
            caption=(
                f"区间 {f.window.start.date()} ~ {f.window.end.date()}（{f.regions_label}）内"
                f"异常运单 {_tot:,} 单（不含「无异常」）。饼图扇区色由图构造器按标签顺序分配；"
                "标签顺序取自 config 的异常类型定义，故同一类型在不同筛选下颜色稳定。"
            ),
            table=pd.DataFrame([{"异常类型": t,
                                 "运单数": v,
                                 "占异常比 (%)": v / _tot * 100}
                                for t, v in zip(_labels, _values)]),
            table_label="异常类型数据表",
        )
with _p1:
    st.markdown("**温控达标率**")
    if _anom.empty:
        st.info("当前筛选区间/片区内没有运单，温控达标率无数据。")
    else:
        _temp = float(_anom["temp_compliance_rate"].mean())
        _lo, _hi = C.COLD_CHAIN_TEMP_RANGE
        UI.chart_block(
            CH.gauge(_temp, title="温控达标率", unit="%", vmin=0.0, vmax=1.0),
            caption=(
                f"取值 = 区间内逐运单 `temp_compliance_rate` 的**等权平均**（每个运单一个值，"
                f"共 {len(_anom):,} 单）。达标区间 [{_lo:.1f}, {_hi:.1f}] ℃（config."
                "COLD_CHAIN_TEMP_RANGE）。产物未登记「达标目标线」，故仪表不画参考线"
                "——不凭空造一个目标值。该指标与数据台账里按**采样点池化**的口径不同（一个按运单、"
                f"一个按 {C.TRACKING_SAMPLE_MINUTES} 分钟采样点），两者会略有差异。"
            ),
            table=(_anom.groupby("region")["temp_compliance_rate"]
                   .agg(["size", "mean"]).reset_index()
                   .rename(columns={"region": "片区", "size": "运单数",
                                    "mean": "温控达标率（均值）"})),
            table_label="温控达标率数据表",
        )

st.markdown(f"**周度埋点：周五午后拥堵 与 {C.HIGH_ANOMALY_REGION} 片区高异常**")
_wk_all = D.artifact("transport_weekly")
if _wk_all.empty:
    st.info("周度异常表为空，无周趋势可画。")
else:
    _ws = pd.to_datetime(_wk_all["week_start"])
    _we = pd.to_datetime(_wk_all["week_end"])
    _wk = _wk_all.loc[(_ws <= f.window.end.normalize())
                      & (_we >= f.window.start.normalize())].copy()
    if _wk.empty:
        st.info("当前日期区间内没有完整落入的周，周度埋点图无数据；请放宽日期范围。")
    else:
        _r07_label = f"{C.HIGH_ANOMALY_REGION} 片区异常率"
        _wk_series = {
            "整体异常率": _wk["anomaly_rate"] * 100,
            "周五「晚点」率": _wk["friday_late_rate"] * 100,
            "其他工作日「晚点」率": _wk["other_late_rate"] * 100,
            _r07_label: _wk["r07_anomaly_rate"] * 100,
        }
        _wk_slots = {"整体异常率": 4, "周五「晚点」率": 5,
                     "其他工作日「晚点」率": 6, _r07_label: 7}
        _wk_c0, _wk_c1 = st.columns([3, 2])
        with _wk_c0:
            UI.chart_block(
                _multi_line(_wk["week"], _wk_series, _wk_slots,
                            y_title="比率 (%)", unit="%"),
                caption=(
                    f"四条序列同量纲（%）、共用一根轴。区间内共 {len(_wk)} 周（按各周起止与所选"
                    "日期窗口重叠截取）。两个埋点：**周五**「晚点」率高于其他工作日（午后拥堵叠加"
                    f"周末备货），**{C.HIGH_ANOMALY_REGION}** 片区异常率显著高于其他片区。周表没有"
                    "片区列，故「配送区域」筛选对这张图不生效。"
                ),
                table=_wk[["week", "week_start", "week_end", "n_shipments", "anomaly_rate",
                           "friday_late_rate", "other_late_rate", "r07_anomaly_rate",
                           "other_anomaly_rate", "temp_compliance_rate"]],
                table_label="周度异常数据表",
            )
        with _wk_c1:
            st.markdown("**埋点池化检验**")
            _pts = D.artifact("transport_anomaly_points")
            _pt_rows = []
            for _key in ("friday_late", "r07_anomaly", "rainy_late"):
                _d = _pts.get(_key)
                if not _d:
                    continue
                _pt_rows.append({
                    "埋点": _d["label"],
                    "对比组率": _d["rate_1"],
                    "参照组率": _d["rate_0"],
                    "倍数": _d["ratio"],
                    "池化 z": _d["z"],
                    "5% 下显著": "是" if _d["significant_at_5pct"] else "否",
                })
            if _pt_rows:
                st.dataframe(pd.DataFrame(_pt_rows), use_container_width=True, hide_index=True)
                _rep = (_pts.get("weekly_replication") or {})
                _fr = _rep.get("friday_late") or {}
                _r7 = _rep.get("r07_anomaly") or {}
                st.caption(
                    f"池化检验为**全量 {C.SIM_DAYS} 天**口径（不受日期筛选影响）。逐周复现："
                    f"周五「晚点」{_fr.get('n_significant', '—')}/{_fr.get('n_weeks', '—')} 周"
                    f"显著，{C.HIGH_ANOMALY_REGION} 异常 {_r7.get('n_significant', '—')}/"
                    f"{_r7.get('n_weeks', '—')} 周"
                    "显著——单周样本量小、当周不显著不等于埋点不存在（见 anomaly_points.json 的 "
                    "note）。"
                )
            else:
                st.info("埋点检验产物为空。")

st.divider()

# ---------------------------------------------------------------------------
# 五、双模式 TCO 曲线（标注盈亏平衡里程）
# ---------------------------------------------------------------------------
st.subheader("双模式 TCO：柴油自购 vs 纯电租赁")
_tco = D.artifact("transport_tco")
_curves = _tco["curves"]
_be = _tco["breakeven_km"]
_tco_series = {
    "柴油自购": _curves["diesel"]["daily_total_cost"],
    "纯电租赁": _curves["ev"]["daily_total_cost"],
}
_tco_slots = {"柴油自购": MODE_SLOT["diesel"], "纯电租赁": MODE_SLOT["ev"]}
_tco_fig = _multi_line(_curves["diesel"]["mileage_km"], _tco_series, _tco_slots,
                       y_title="日总成本 (元/日)", unit=" 元")
_tco_fig.add_vline(
    x=float(_be["km"]), line_width=2, line_color=theme.tokens()["secondary_ink"],
    annotation_text=f"盈亏平衡里程 {float(_be['km']):,.2f} km",
    annotation_position="top", annotation_font_color=theme.tokens()["secondary_ink"],
)
_be_side = "右侧" if float(_tco["reference_daily_km"]) >= float(_be["km"]) else "左侧"
UI.chart_block(
    _tco_fig,
    caption=(
        f"两条曲线同量纲（元/日），共用一根轴。竖线为**盈亏平衡里程 "
        f"{float(_be['km']):,.2f} km**（{_be['basis']}）：低于此里程柴油更省，高于此里程纯电更省"
        f"（纯电固定成本高、公里变动成本低）。代表日单车均里程 {_tco['reference_daily_km']:,.2f} km"
        f"（{_tco['reference_basis']}）落在平衡点{_be_side}，故 TCO 建议 "
        f"**{kpis.MODE_LABELS.get(_tco['recommendation']['recommended_mode'], '')}**"
        f"（{_tco['recommendation']['basis']}）。成本参数取值见 ADR-0009 与 config 的敏感性档位。"
    ),
    table=pd.DataFrame({
        "日里程 (km)": _curves["diesel"]["mileage_km"],
        "柴油自购 日总成本 (元/日)": _curves["diesel"]["daily_total_cost"],
        "纯电租赁 日总成本 (元/日)": _curves["ev"]["daily_total_cost"],
    }),
    table_label="TCO 曲线数据表",
)

# ---------------------------------------------------------------------------
# 六、自营 vs 货拉拉外包对照表
# ---------------------------------------------------------------------------
st.markdown("**自营 vs 货拉拉外包对照（逐趟真实里程口径）**")
_outs = _tco["outsourcing"]
st.caption(
    f"口径（引自产物 `outsourcing.basis`）：{_outs['basis']}"
    f"「基本持平」的判定带宽取自 `config.HUOLALA_TIE_BAND`（相差 "
    f"±{(1 - C.HUOLALA_TIE_BAND) * 100:.0f}% 以内即判持平）。"
)

_by_plan = pd.DataFrame(_outs["by_plan"]).T.reset_index().rename(columns={"index": "方案"})
_by_plan["方案"] = _by_plan["方案"].map(lambda k: PLAN_LABEL.get(k, k))
_by_plan = _by_plan[[
    "方案", "n_trips", "total_self_cost", "total_huolala_cost", "delta_pct_vs_huolala",
    "n_trips_self_cheaper", "n_trips_outsource_cheaper", "n_trips_tie", "best_mode",
]].rename(columns={
    "n_trips": "趟次", "total_self_cost": "自营总成本 (元)",
    "total_huolala_cost": "货拉拉总报价 (元)", "delta_pct_vs_huolala": "自营 vs 货拉拉 (%)",
    "n_trips_self_cheaper": "自营更省趟数", "n_trips_outsource_cheaper": "外包更省趟数",
    "n_trips_tie": "持平趟数", "best_mode": "自营较省模式",
})
st.dataframe(_by_plan, use_container_width=True, hide_index=True)
st.caption(
    f"当前主口径为 **{PLAN_LABEL.get(_outs['primary_plan'], _outs['primary_plan'])}**："
    f"全部 {_by_plan['趟次'].sum()} 趟里自营更省 "
    f"{int(_by_plan['自营更省趟数'].sum())} 趟、外包更省 {int(_by_plan['外包更省趟数'].sum())} 趟、"
    f"持平 {int(_by_plan['持平趟数'].sum())} 趟。{_tco['recommendation']['outsourcing_verdict']}。"
)

with st.expander("逐趟对照明细数据表（自营 / 货拉拉）"):
    _per_trip = pd.DataFrame(_outs["per_trip"])
    _per_trip["plan"] = _per_trip["plan"].map(lambda k: PLAN_LABEL.get(k, k))
    st.dataframe(
        _per_trip.rename(columns={
            "trip_id": "趟次", "plan": "方案", "n_stops": "点位数", "n_orders": "订单数",
            "distance_km": "里程 (km)", "self_diesel_cost": "自营-柴油 (元)",
            "self_ev_cost": "自营-纯电 (元)", "self_cost": "自营-较省者 (元)",
            "self_best_mode": "自营较省模式", "huolala_cost": "货拉拉 (元)",
            "delta_pct_vs_huolala": "自营 vs 货拉拉 (%)", "verdict": "结论",
        }),
        use_container_width=True, hide_index=True,
    )
st.caption(
    "**口径提醒**：自营单趟成本的分摊口径见上方引自产物的 `outsourcing.basis`（含「代表日每台车"
    "恰好只跑一趟、日固定成本全额摊入」这一前提）。此外，`self_cost` 取柴油/纯电较省者，隐含"
    "「车队可按线路自由选模式」，是自营竞争力的**上界**；实际单一模式车队应看对应模式那一列。"
)

st.divider()

# ---------------------------------------------------------------------------
# 数据来源与口径
# ---------------------------------------------------------------------------
with st.expander("数据来源与口径（每个数字的追溯入口）"):
    st.markdown(
        f"""
| 读数 | 产物文件 | 生成命令 | 数据类别 |
|---|---|---|---|
| 代表日前后对比 / 里程 / 成本 / 比率 | `{C.TRANSPORT_COMPARISON_CSV.name}` | `python -m src.transport_optimize` | 过程仿真（派生指标） |
| 逐趟明细（满载率分布 / 逐趟对照） | `{C.TRANSPORT_TRIPS_CSV.name}` | `python -m src.transport_optimize` | 过程仿真（派生指标） |
| 路线 GeoJSON（地图） | `{C.TRANSPORT_ROUTES_BASELINE_GEOJSON.name}` / `{C.TRANSPORT_ROUTES_OPTIMIZED_GEOJSON.name}` | `python -m src.transport_optimize` | 过程仿真（真实路网里程） |
| 周度异常 / 埋点检验 | `{C.TRANSPORT_ANOMALY_WEEKLY_CSV.name}` / `{C.TRANSPORT_ANOMALY_POINTS_JSON.name}` | `python -m src.transport_decisions` | 过程仿真 + 真实观测（总体由 Olist 延迟率标定） |
| TCO 曲线 / 盈亏平衡 / 外包对照 | `{C.TRANSPORT_TCO_JSON.name}` | `python -m src.transport_decisions` | 情景假设（成本参数见 ADR-0009） |
| 异常构成 / 温控达标率 | `{C.DELIVERY_ANOMALIES_CSV.name}` | `python -m src.gen_delivery_data` | 情景假设（总体水平由 Olist 实测延迟率标定） |
| POI 名称（地图弹窗） | `{C.POI_CSV.name}` | `python -m src.geo_poi` | 真实观测 |

- 全部读数经 `src/dashboard/data.py` 取用并缓存；缓存键含**产物 mtime**，产物一被重写就自动重读
  （另有侧边栏「刷新产物读数」按钮兜底）。页面不自行读盘——绕开取数层就会绕过 mtime 缓存键，
  产物更新后那一处不会自动失效。
- 本页不显示 0 或占位值掩盖缺产物：产物不全时直接列出缺哪个文件、该跑哪条命令并停下。
"""
    )
