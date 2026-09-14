"""看板取数层：全部读数来自 `data/processed/` 的产物文件，页面内不硬编码任何数字（12 号票）。

两条纪律写进这一层：

1. **产物缺失即报错，不回退到 0 或空图。** 一个显示「0」的看板比一个报错的看板危险得多——
   前者看起来像「今天没有异常」，后者才告诉你脚本没跑。`require_artifacts()` 因此直接
   列出缺哪个文件、该跑哪条命令、然后 `st.stop()`。
2. **读数统一缓存，且缓存会随产物自动失效。** 缓存只放在 `_read_csv` / `_read_json` 两个
   原语上，键里带**文件的 mtime**，产物一被重写就自动重读。公开 loader 一律不加
   `@st.cache_data`——它们**无参数**，装饰上去等于缓存键恒定，mtime 根本进不了键，产物更新
   后页面会一直读旧数（评审实测确认过这个坑）。另留一个「刷新产物读数」按钮做显式兜底。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config as C


@dataclass(frozen=True)
class Artifact:
    """一个看板依赖的产物文件：路径 + 缺失时该跑什么命令。"""

    key: str
    path: Path
    hint: str


#: 驾驶舱与三张子页共用的产物清单。缺任何一个，对应页面会明确说缺什么、怎么补。
ARTIFACTS: tuple[Artifact, ...] = (
    Artifact("warehouse_kpi", C.WAREHOUSE_KPI_JSON, "python -m src.warehouse_kpi"),
    Artifact("warehouse_daily", C.WAREHOUSE_DAILY_CSV, "python -m src.warehouse_kpi"),
    Artifact("transport_kpi", C.TRANSPORT_KPI_JSON, "python -m src.transport_optimize"),
    Artifact("transport_daily", C.TRANSPORT_DAILY_CSV, "python -m src.transport_optimize"),
    Artifact("delivery_orders", C.DELIVERY_ORDERS_CSV, "python -m src.gen_delivery_data"),
    Artifact("anomalies", C.DELIVERY_ANOMALIES_CSV, "python -m src.gen_delivery_data"),
    Artifact("regions", C.DELIVERY_REGIONS_CSV, "python -m src.gen_delivery_data"),
    # 以下供 13/14/15 三张子页使用
    Artifact("sku_master", C.WAREHOUSE_TABLES["sku"], "python -m src.gen_warehouse_data"),
    Artifact("location_master", C.WAREHOUSE_TABLES["location"], "python -m src.gen_warehouse_data"),
    Artifact("outbound", C.WAREHOUSE_TABLES["outbound"], "python -m src.gen_warehouse_data"),
    Artifact("sim_exp1", C.PROCESSED_DIR / "sim" / "exp1_layout.json", "python -m src.warehouse_sim"),
    Artifact("sim_exp2", C.PROCESSED_DIR / "sim" / "exp2_staffing.json", "python -m src.warehouse_sim"),
    Artifact("sim_whatif", C.PROCESSED_DIR / "sim" / "whatif_staffing.json", "python -m src.warehouse_sim"),
    Artifact("olist_overall", C.PROCESSED_DIR / "olist_kpi" / "overall_kpi.json",
             "python -m src.olist_kpi"),
    Artifact("olist_state", C.PROCESSED_DIR / "olist_kpi" / "drilldown_by_state.csv",
             "python -m src.olist_kpi"),
    Artifact("olist_orders", C.PROCESSED_DIR / "olist" / "clean_orders.csv",
             "python -m src.olist_clean"),
    Artifact("transport_trips", C.TRANSPORT_TRIPS_CSV, "python -m src.transport_optimize"),
    Artifact("transport_tco", C.TRANSPORT_TCO_JSON, "python -m src.transport_decisions"),
    Artifact("transport_whatif", C.TRANSPORT_WHATIF_JSON, "python -m src.transport_decisions"),
    Artifact("transport_routes_optimized", C.TRANSPORT_ROUTES_OPTIMIZED_GEOJSON,
             "python -m src.transport_optimize"),
    Artifact("transport_weekly", C.TRANSPORT_ANOMALY_WEEKLY_CSV, "python -m src.transport_decisions"),
)


def missing_artifacts(keys: tuple[str, ...] | None = None) -> list[Artifact]:
    """当前缺失的产物；`keys` 给定时只检查子集（每张子页只声明自己真正要用的）。"""
    wanted = ARTIFACTS if keys is None else tuple(a for a in ARTIFACTS if a.key in set(keys))
    return [a for a in wanted if not a.path.exists()]


def require_artifacts(keys: tuple[str, ...] | None = None) -> None:
    """缺产物就停下并说清怎么补；**不静默降级**（见模块 docstring 第 1 条）。"""
    missing = missing_artifacts(keys)
    if not missing:
        return
    st.error("缺少分析产物，看板无法给出可信数字（不显示 0 或其他占位值）。请先在项目根运行：")
    for a in missing:
        st.markdown(f"- `{a.path.relative_to(C.PROJECT_ROOT)}` ← `{a.hint}`")
    st.stop()


def refresh() -> None:
    """清掉取数缓存后重跑（脚本重跑出新产物后，点一下即可读到，不必重启看板）。"""
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# 读取原语
# ---------------------------------------------------------------------------
def _stamp(path: Path) -> float:
    """文件的修改时间，作为缓存键的一部分。

    把 mtime 塞进缓存键，是为了让「脚本重跑出新产品」这件事**自动**被看见：
    只按路径缓存的话，产物更新后页面还会一直读旧数，直到有人手动清缓存或重启看板
    ——那正是「看板数字与报告对不上」最常见的来源。缺文件返回 0，由 `require_artifacts`
    负责报错，这里不改变缺失语义。
    """
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float, **kw) -> pd.DataFrame:  # noqa: ARG001（mtime 仅作缓存键）
    return pd.read_csv(path, encoding="utf-8-sig", **kw)


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:  # noqa: ARG001（mtime 仅作缓存键）
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 模块一：仓内 KPI
# ---------------------------------------------------------------------------
def warehouse_kpi() -> dict:
    return _read_json(str(C.WAREHOUSE_KPI_JSON), _stamp(C.WAREHOUSE_KPI_JSON))


def warehouse_daily() -> pd.DataFrame:
    """逐日仓内指标（库存准确率 / 收货及时率 / 拣货效率 / 盘差率）。

    `date` 在这里就解析成 datetime——两个模块的日表要在页面上按日期合并，
    一边是字符串一边是 datetime 会让 merge 直接抛错（而 `slice_window` 各自都能跑通，
    所以这个错只会在页面上暴露，不会在单测里暴露）。
    """
    return _read_csv(str(C.WAREHOUSE_DAILY_CSV), _stamp(C.WAREHOUSE_DAILY_CSV),
                     parse_dates=["date"])


def abc_pareto() -> pd.DataFrame:
    return _read_csv(str(C.WAREHOUSE_ABC_PARETO_CSV), _stamp(C.WAREHOUSE_ABC_PARETO_CSV))


def stocktake_by_category() -> pd.DataFrame:
    p = C.WAREHOUSE_DRILLDOWN_CATEGORY_CSV
    return _read_csv(str(p), _stamp(p))


def picking_by_hour() -> pd.DataFrame:
    p = C.WAREHOUSE_DRILLDOWN_HOUR_CSV
    return _read_csv(str(p), _stamp(p))


# ---------------------------------------------------------------------------
# 模块二：运输
# ---------------------------------------------------------------------------
def transport_kpi() -> dict:
    return _read_json(str(C.TRANSPORT_KPI_JSON), _stamp(C.TRANSPORT_KPI_JSON))


def transport_daily() -> pd.DataFrame:
    """逐日基线 KPI（时间窗达成率 / 满载率 / 单均成本 / 里程）。"""
    return _read_csv(str(C.TRANSPORT_DAILY_CSV), _stamp(C.TRANSPORT_DAILY_CSV),
                     parse_dates=["date"])


def transport_comparison() -> pd.DataFrame:
    p = C.TRANSPORT_COMPARISON_CSV
    return _read_csv(str(p), _stamp(p))


def transport_trips() -> pd.DataFrame:
    return _read_csv(str(C.TRANSPORT_TRIPS_CSV), _stamp(C.TRANSPORT_TRIPS_CSV))


def transport_tco() -> dict:
    if not C.TRANSPORT_TCO_JSON.exists():
        return {}
    return _read_json(str(C.TRANSPORT_TCO_JSON), _stamp(C.TRANSPORT_TCO_JSON))


def whatif_vehicles() -> dict:
    """车辆数 what-if 预计算缓存（ADR-0010；滑块只切档、不实时求解）。"""
    if not C.TRANSPORT_WHATIF_JSON.exists():
        return {}
    return _read_json(str(C.TRANSPORT_WHATIF_JSON), _stamp(C.TRANSPORT_WHATIF_JSON))


def anomaly_weekly() -> pd.DataFrame:
    p = C.TRANSPORT_ANOMALY_WEEKLY_CSV
    return _read_csv(str(p), _stamp(p)) if p.exists() else pd.DataFrame()


def embedded_points() -> dict:
    """埋点复现结论（三靶点池化检验 + 逐周复现计数与分组样本量）。"""
    if not C.TRANSPORT_ANOMALY_POINTS_JSON.exists():
        return {}
    return _read_json(str(C.TRANSPORT_ANOMALY_POINTS_JSON),
                      _stamp(C.TRANSPORT_ANOMALY_POINTS_JSON))


def routes_geojson(plan: str) -> dict:
    """某套路线的 GeoJSON（`plan` ∈ {baseline, optimized}）。"""
    path = (C.TRANSPORT_ROUTES_BASELINE_GEOJSON if plan == "baseline"
            else C.TRANSPORT_ROUTES_OPTIMIZED_GEOJSON)
    return _read_json(str(path), _stamp(path))


def poi_table() -> pd.DataFrame:
    """POI 主数据（编制/名称/坐标）——地图打点与路段弹窗的显示名来源。

    运输页要显示「送到哪家店」而不是只显示 `P007`，故需要它；读数一律走本模块，
    页面不得自行读盘（否则缓存键约定会被绕过，产物更新后那一处不会自动失效）。
    """
    return _read_csv(str(C.POI_CSV), _stamp(C.POI_CSV))


# ---------------------------------------------------------------------------
# 数据层 D：配送情景（全局筛选与异常预警的数据源）
# ---------------------------------------------------------------------------
def delivery_orders() -> pd.DataFrame:
    """配送订单（含片区、坐标、时间窗；时间窗为当日分钟数）。"""
    return _read_csv(str(C.DELIVERY_ORDERS_CSV), _stamp(C.DELIVERY_ORDERS_CSV),
                     parse_dates=["date"])


def anomalies() -> pd.DataFrame:
    """在途异常表（含片区、车辆、异常类型、延误、温控达标率、是否准时）。"""
    return _read_csv(str(C.DELIVERY_ANOMALIES_CSV), _stamp(C.DELIVERY_ANOMALIES_CSV),
                     parse_dates=["date"])


def regions() -> pd.DataFrame:
    return _read_csv(str(C.DELIVERY_REGIONS_CSV), _stamp(C.DELIVERY_REGIONS_CSV))


def vehicles() -> pd.DataFrame:
    return _read_csv(str(C.DELIVERY_VEHICLES_CSV), _stamp(C.DELIVERY_VEHICLES_CSV))


def sku_master() -> pd.DataFrame:
    """SKU 主数据（品类筛选器的选项来源）。数据层 A 未生成时返回空表，页面据此禁用该项。"""
    path = C.WAREHOUSE_TABLES["sku"]
    return _read_csv(str(path), _stamp(path)) if path.exists() else pd.DataFrame()


def order_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    """全量数据的日期跨度（筛选器的可用范围，取 90 天窗口的起止）。"""
    d = delivery_orders()["date"]
    return pd.Timestamp(d.min()), pd.Timestamp(d.max())


# ---------------------------------------------------------------------------
# 子页专用读数（13/14/15 号票）
# ---------------------------------------------------------------------------
def location_master() -> pd.DataFrame:
    """库位主数据（行/列/分区/行走距离）——库位热力图的坐标来源。"""
    p = C.WAREHOUSE_TABLES["location"]
    return _read_csv(str(p), _stamp(p))


def outbound_lines() -> pd.DataFrame:
    """出库行（拣货明细）。只取热力图与阈值重算需要的列——全表 11 列、4.4 万行，
    仪表盘没必要把用不到的列也读进内存。"""
    p = C.WAREHOUSE_TABLES["outbound"]
    return _read_csv(str(p), _stamp(p),
                     usecols=["sku_id", "loc_id", "pick_start", "pick_end"])


def location_frequency() -> pd.DataFrame:
    """库位出库频次 = 该库位承载的出库行数，与库位主数据按 `loc_id` 连接。

    频次口径与 ABC 分类、库位重排的「Σ(频次×距离)」同源（都用**出库行数**，ADR-0002）。
    """
    loc = location_master()
    freq = outbound_lines().groupby("loc_id").size().rename("outbound_lines")
    out = loc.merge(freq, left_on="loc_id", right_index=True, how="left")
    out["outbound_lines"] = out["outbound_lines"].fillna(0).astype(int)
    return out


def slotting_assignment() -> pd.DataFrame:
    """库位重排结果（每 SKU 的频次、重排前后行走距离与目标库位）。"""
    p = C.WAREHOUSE_SLOTTING_CSV
    return _read_csv(str(p), _stamp(p))


# --- 数据层 F：SimPy 仿真对照实验（07 号票预计算缓存） -----------------------
def sim_exp1() -> dict:
    """实验一：原始布局 vs ABC 分区，各 30 次重复（含 95% CI）。"""
    p = C.PROCESSED_DIR / "sim" / "exp1_layout.json"
    return _read_json(str(p), _stamp(p))


def sim_exp2() -> dict:
    """实验二：拣货员 4/5/6 档，各 30 次重复 + 时长—人力成本权衡曲线。"""
    p = C.PROCESSED_DIR / "sim" / "exp2_staffing.json"
    return _read_json(str(p), _stamp(p))


def sim_whatif_staffing() -> dict:
    """人力 what-if 缓存：拣货员人数 → 指标（改善页滑块直接切档，ADR-0010 同哲学）。"""
    p = C.PROCESSED_DIR / "sim" / "whatif_staffing.json"
    return _read_json(str(p), _stamp(p))


# --- 数据层 B：Olist 真实履约（09 号票） -------------------------------------
def olist_overall() -> dict:
    p = C.PROCESSED_DIR / "olist_kpi" / "overall_kpi.json"
    return _read_json(str(p), _stamp(p))


def olist_state_drilldown() -> pd.DataFrame:
    p = C.PROCESSED_DIR / "olist_kpi" / "drilldown_by_state.csv"
    return _read_csv(str(p), _stamp(p))


def olist_weight_drilldown() -> pd.DataFrame:
    p = C.PROCESSED_DIR / "olist_kpi" / "drilldown_by_weight.csv"
    return _read_csv(str(p), _stamp(p))


def olist_durations() -> pd.DataFrame:
    """Olist 真实履约时长（逐单）。只取分布图与州下钻需要的列。"""
    p = C.PROCESSED_DIR / "olist" / "clean_orders.csv"
    return _read_csv(str(p), _stamp(p),
                     usecols=["customer_state", "fulfillment_days",
                              "outbound_response_days", "transit_days", "is_late"])
