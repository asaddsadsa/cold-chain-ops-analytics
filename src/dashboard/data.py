"""看板取数层：`ARTIFACTS` 登记表是产物的**唯一真相**，读侧只有一个 interface（12 号票基建，重构后）。

三条纪律写进这一层：

1. **登记表即契约。** 一个产物在哪里、怎么解析、需要几列、缺了怎么办、该跑哪条命令补——
   全部登记在 `ARTIFACTS` 里。调用方只需要知道 key：`artifact(key)` 拿数据，`describe(key)`
   拿元数据。**新增产物 = 登记表加一行**，不再是在这里加一个转发的 loader、再去各页面抄一份
   缺失检查（重构前正是那样：7 个产物没登记，四个页面各写了一份检查）。

2. **缺产物即报错，不回退到 0 或空图。** 一个显示「0」的看板比一个报错的看板危险得多——
   前者看起来像「今天没有异常」，后者才告诉你脚本没跑。必需的产物缺失一律抛
   `MissingArtifact`；「可选」必须在登记表里**声明**（`required=False` + `empty=`），
   不是由某个 loader 自己决定。`sku_master` 是唯一的可选产物：侧边栏缺它就禁用「品类」控件，
   这是有意为之，故它显式登记为可选。

3. **读数统一缓存，且缓存会随产物自动失效。** 缓存只放在 `_read_csv` / `_read_json` 两个
   原语上，键里带**文件的 mtime**，产物一被重写就自动重读。公开入口一律不加
   `@st.cache_data`——它们若被装饰，等于缓存键恒定，mtime 根本进不了键，产物更新后页面会
   一直读旧数（评审实测确认过这个坑）。另留一个「刷新产物读数」按钮做显式兜底。

**为什么 `st.stop()` 之后还有一句 `raise`**：`streamlit 1.63` 的 `st.stop()` 在拿不到
`ScriptRunContext` 时**不是异常而是 no-op**（函数体直接 return，尽管签名写着 `NoReturn`）。
在真 runtime 里它照常结束脚本、那句 `raise` 不可达；在单测 / CLI 等 bare mode 下，它退化成
「打一行 warning 然后继续跑」——那会让「缺产物即报错」的招牌变成空话。补一句 `raise` 让两种
模式下契约都成立。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import streamlit as st

from src import config as C


# ---------------------------------------------------------------------------
# 错误
# ---------------------------------------------------------------------------
class UnknownArtifact(KeyError):
    """登记表里没有这个 key。拼错的 key 必须立刻失败，不能静默通过。"""


class MissingArtifact(RuntimeError):
    """必需的产物缺失。带路径与补救命令，由调用方决定怎么呈现。"""

    def __init__(self, missing: Iterable["Artifact"]):
        self.missing: tuple[Artifact, ...] = tuple(missing)
        super().__init__("缺少分析产物：" + "；".join(
            f"{a.path.name} ← {a.hint}" for a in self.missing
        ))


# ---------------------------------------------------------------------------
# 登记表：产物的唯一真相
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Artifact:
    """一条产物登记项。

    `key` 是调用方唯一需要知道的字符串。`read_kwargs` 承载「怎么解析」——它必须进登记表
    而不是留在读法里，因为其中两项是**正确性前提**而非优化：`parse_dates=["date"]` 是两个
    日表在页面上按 date 合并的前提（一边 str 一边 datetime 会让 merge 直接抛错），
    `usecols` 则是下游的列契约（列名一变，热力图的 groupby 就散架）。
    """

    key: str
    path: Path
    hint: str
    kind: Literal["csv", "json"] = "csv"
    read_kwargs: Mapping[str, Any] = field(default_factory=dict)
    required: bool = True
    empty: Callable[[], Any] | None = None

    def value(self) -> Any:
        """读盘并解析。缺失时按 `required` 决定抛错还是返回声明过的空结构。"""
        if not self.path.exists():
            if self.required:
                raise MissingArtifact([self])
            if self.empty is None:
                raise ValueError(f"可选产物 {self.key!r} 未声明 empty 工厂")
            return self.empty()
        return _read(self)


#: 驾驶舱与三张子页共用的产物登记表。缺任何一个必需的，对应页面会明确说缺什么、怎么补。
ARTIFACTS: tuple[Artifact, ...] = (
    # --- 模块一：仓内 KPI（数据层 A，08 号票） ---------------------------------
    Artifact("warehouse_kpi", C.WAREHOUSE_KPI_JSON, "python -m src.warehouse_kpi", "json"),
    Artifact("warehouse_daily", C.WAREHOUSE_DAILY_CSV, "python -m src.warehouse_kpi", "csv",
             {"parse_dates": ["date"]}),
    Artifact("warehouse_abc_pareto", C.WAREHOUSE_ABC_PARETO_CSV,
             "python -m src.warehouse_kpi", "csv"),
    Artifact("warehouse_picking_hour", C.WAREHOUSE_DRILLDOWN_HOUR_CSV,
             "python -m src.warehouse_kpi", "csv"),
    Artifact("warehouse_stocktake_category", C.WAREHOUSE_DRILLDOWN_CATEGORY_CSV,
             "python -m src.warehouse_kpi", "csv"),
    # --- 模块二（上）：运输优化核心（10 号票） ---------------------------------
    Artifact("transport_kpi", C.TRANSPORT_KPI_JSON, "python -m src.transport_optimize", "json"),
    Artifact("transport_daily", C.TRANSPORT_DAILY_CSV, "python -m src.transport_optimize", "csv",
             {"parse_dates": ["date"]}),
    Artifact("transport_comparison", C.TRANSPORT_COMPARISON_CSV,
             "python -m src.transport_optimize", "csv"),
    Artifact("transport_trips", C.TRANSPORT_TRIPS_CSV,
             "python -m src.transport_optimize", "csv"),
    Artifact("transport_routes_baseline", C.TRANSPORT_ROUTES_BASELINE_GEOJSON,
             "python -m src.transport_optimize", "json"),
    Artifact("transport_routes_optimized", C.TRANSPORT_ROUTES_OPTIMIZED_GEOJSON,
             "python -m src.transport_optimize", "json"),
    # --- 模块二（下）：周度复盘 + TCO + what-if（11 号票） ---------------------
    Artifact("transport_weekly", C.TRANSPORT_ANOMALY_WEEKLY_CSV,
             "python -m src.transport_decisions", "csv"),
    Artifact("transport_anomaly_points", C.TRANSPORT_ANOMALY_POINTS_JSON,
             "python -m src.transport_decisions", "json"),
    Artifact("transport_tco", C.TRANSPORT_TCO_JSON, "python -m src.transport_decisions", "json"),
    Artifact("transport_whatif", C.TRANSPORT_WHATIF_JSON,
             "python -m src.transport_decisions", "json"),
    # --- 数据层 D：配送情景（订单 / 在途异常 / 片区） --------------------------
    Artifact("delivery_orders", C.DELIVERY_ORDERS_CSV, "python -m src.gen_delivery_data", "csv",
             {"parse_dates": ["date"]}),
    Artifact("anomalies", C.DELIVERY_ANOMALIES_CSV, "python -m src.gen_delivery_data", "csv",
             {"parse_dates": ["date"]}),
    Artifact("regions", C.DELIVERY_REGIONS_CSV, "python -m src.gen_delivery_data", "csv",
             # 读侧只用这两列（片区码 + 行政区名），列名即契约
             {"usecols": ["region", "district"]}),
    # --- 地理参照 -------------------------------------------------------------
    Artifact("poi", C.POI_CSV, "python -m src.geo_poi", "csv"),
    # --- 数据层 A：仿真仓五表（01 号票） --------------------------------------
    # sku_master 是本层唯一的可选产物：侧边栏缺它就禁用「品类」控件，不阻断整页。
    Artifact("sku_master", C.WAREHOUSE_TABLES["sku"], "python -m src.gen_warehouse_data", "csv",
             required=False, empty=pd.DataFrame),
    Artifact("location_master", C.WAREHOUSE_TABLES["location"],
             "python -m src.gen_warehouse_data", "csv"),
    # 全表 11 列 / 4.4 万行，看板只用得到这 4 列——列名是热力图与频次口径的硬依赖，
    # 故写进登记表而不是读法里。
    Artifact("outbound", C.WAREHOUSE_TABLES["outbound"], "python -m src.gen_warehouse_data",
             "csv", {"usecols": ["sku_id", "loc_id", "pick_start", "pick_end"]}),
    # --- 数据层 F：SimPy 仿真对照实验（07 号票预计算缓存） --------------------
    Artifact("sim_exp1", C.PROCESSED_DIR / "sim" / "exp1_layout.json",
             "python -m src.warehouse_sim", "json"),
    Artifact("sim_exp2", C.PROCESSED_DIR / "sim" / "exp2_staffing.json",
             "python -m src.warehouse_sim", "json"),
    Artifact("sim_whatif", C.PROCESSED_DIR / "sim" / "whatif_staffing.json",
             "python -m src.warehouse_sim", "json"),
    # --- 数据层 B：Olist 真实履约（03/09 号票） -------------------------------
    Artifact("olist_overall", C.PROCESSED_DIR / "olist_kpi" / "overall_kpi.json",
             "python -m src.olist_kpi", "json"),
    Artifact("olist_state", C.PROCESSED_DIR / "olist_kpi" / "drilldown_by_state.csv",
             "python -m src.olist_kpi", "csv"),
    Artifact("olist_orders", C.PROCESSED_DIR / "olist" / "clean_orders.csv",
             "python -m src.olist_clean", "csv",
             {"usecols": ["customer_state", "fulfillment_days", "outbound_response_days",
                          "transit_days", "is_late"]}),
)

ARTIFACT_BY_KEY: Mapping[str, Artifact] = {a.key: a for a in ARTIFACTS}
# 用 raise 而不是 assert：`python -O` 会把 assert 整段裁掉，重复 key 会静默地让后一条
# 登记项顶掉前一条，而「登记表即契约」正建立在 key 唯一之上。
_dups = sorted({a.key for a in ARTIFACTS if sum(b.key == a.key for b in ARTIFACTS) > 1})
if _dups:
    raise ValueError(f"登记表里有重复的 key：{_dups}")


# ---------------------------------------------------------------------------
# 读取原语
# ---------------------------------------------------------------------------
def _stamp(path: Path) -> float:
    """文件的修改时间，作为缓存键的一部分。

    把 mtime 塞进缓存键，是为了让「脚本重跑出新产品」这件事**自动**被看见：
    只按路径缓存的话，产物更新后页面还会一直读旧数，直到有人手动清缓存或重启看板
    ——那正是「看板数字与报告对不上」最常见的来源。缺文件返回 0，由 `Artifact.value`
    负责报错，这里不改变缺失语义。
    """
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float, **kw) -> pd.DataFrame:  # noqa: ARG001（mtime 仅作缓存键）
    return pd.read_csv(path, encoding="utf-8-sig", **kw)


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:  # noqa: ARG001（mtime 仅作缓存键）
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read(entry: Artifact) -> Any:
    path = str(entry.path)
    if entry.kind == "json":
        return _read_json(path, _stamp(entry.path))
    return _read_csv(path, _stamp(entry.path), **entry.read_kwargs)


def _reject_unknown(keys: Iterable[str]) -> None:
    """未登记的 key 立刻报错。

    重构前这里是 `a.key in set(keys)` 的过滤——拼错的 key 被**静默丢弃**，
    于是「缺产物就停」那道防线在写错名字时形同不存在（运输页的
    `"transport_comparison"` 就这样空转了很久）。
    """
    unknown = sorted(set(keys) - set(ARTIFACT_BY_KEY))
    if unknown:
        raise UnknownArtifact(
            f"未登记的产物 key：{unknown}；登记表里有 {sorted(ARTIFACT_BY_KEY)}"
        )


# ---------------------------------------------------------------------------
# 读侧 interface
# ---------------------------------------------------------------------------
def artifact(key: str) -> Any:
    """登记表里 `key` 对应的产物，已按登记的方式解析。

    CSV → `DataFrame`，JSON → `dict`。必需的缺失抛 `MissingArtifact`；
    可选的缺失返回登记表声明的空结构（不是裸 `None` / `{}`）。
    """
    if key not in ARTIFACT_BY_KEY:
        _reject_unknown([key])
    return ARTIFACT_BY_KEY[key].value()


@dataclass(frozen=True)
class ArtifactInfo:
    """一条产物的元数据（`describe` 的返回）。"""

    key: str
    path: Path
    hint: str
    exists: bool
    mtime: float
    category: str | None


def describe(key: str) -> ArtifactInfo:
    """产物的元数据：路径、是否就位、mtime、以及**产物自报**的数据类别。

    `category` 取自产物自己的 `data_category` 字段，本层不代为归类——一份产物可以同时属于
    多类（`transport_kpi.json` 是「情景假设（需求）+ 学术基准（算法）」），压成单一类别会
    把项目最高原则抹平（见 CONTEXT.md「复合类别」）。CSV 通常不自报，返回 `None`。
    """
    if key not in ARTIFACT_BY_KEY:
        _reject_unknown([key])
    entry = ARTIFACT_BY_KEY[key]
    category = None
    if entry.kind == "json" and entry.path.exists():
        category = _read_json(str(entry.path), _stamp(entry.path)).get("data_category")
    return ArtifactInfo(key=key, path=entry.path, hint=entry.hint,
                        exists=entry.path.exists(), mtime=_stamp(entry.path),
                        category=category)


def missing_artifacts(keys: Iterable[str] | None = None) -> list[Artifact]:
    """当前缺失的**必需**产物；`keys` 给定时只检查子集（每张子页只声明自己真正要用的）。

    纯函数，不碰 Streamlit——这是本层第一次可以被单测的检查点。
    """
    if keys is None:
        return [a for a in ARTIFACTS if a.required and not a.path.exists()]
    _reject_unknown(keys)
    return [a for a in (ARTIFACT_BY_KEY[k] for k in keys) if a.required and not a.path.exists()]


def require_artifacts(keys: Iterable[str] | None = None) -> None:
    """页面引导用：缺必需产物就列清单、报错、停下。**不静默降级**（见模块 docstring 第 2 条）。

    末尾那句 `raise` 是给 bare mode 兜底的——`st.stop()` 在没有 ScriptRunContext 时是
    no-op 而不是异常，不补这一句，单测/CLI 里缺产物会被静默穿透（见模块 docstring 末段）。
    """
    missing = missing_artifacts(keys)
    if not missing:
        return
    st.error("缺少分析产物，看板无法给出可信数字（不显示 0 或其他占位值）。请先在项目根运行：")
    for a in missing:
        st.markdown(f"- `{a.path.relative_to(C.PROJECT_ROOT)}` ← `{a.hint}`")
    st.stop()
    raise MissingArtifact(missing)


def refresh() -> None:
    """清掉取数缓存后重跑（脚本重跑出新产物后，点一下即可读到，不必重启看板）。"""
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# 派生视图（计算，不是转发）
# ---------------------------------------------------------------------------
def order_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    """全量数据的日期跨度（筛选器的可用范围，取 90 天窗口的起止）。"""
    d = artifact("delivery_orders")["date"]
    return pd.Timestamp(d.min()), pd.Timestamp(d.max())


def region_labels() -> Mapping[str, str]:
    """片区码 → 给人读的标签：「R07」→「青白江区（R07）」。

    码是**数据里的键**：`region` 列、`config.HIGH_ANOMALY_REGION`、台账与报告的埋点叙述
    一律写 R07。地区名是**给人读的名字**。两者都要——只给码，读者不知道那是哪儿；只给地区
    名，又切断了与 config / 台账 / 报告的对应。故标签取「地区名（码）」，地区名在前。

    映射取自 POI 表自身的 `district` 列：一个片区就是一个行政区（`assign_regions` 的构造
    前提），故该映射恒为 1:1。
    """
    pairs = artifact("regions")[["region", "district"]].dropna().drop_duplicates()
    return {code: f"{district}（{code}）"
            for code, district in zip(pairs["region"], pairs["district"])}


def label_regions(regions: pd.Series) -> pd.Series:
    """把一整列 `region` 换成可读标签；未登记的码原样保留（不静默变成 NaN）。"""
    return regions.map(region_labels()).fillna(regions)


def region_codes() -> tuple[str, ...]:
    """配送片区码表 R01–R08（筛选器「配送区域」的取值域）。

    与 `order_window()` 之于日期筛选器同理：**筛选器的取值域是产物的一张投影，不是一张
    独立维度表**。`regions.csv` 是 POI 级表——一行一个提货点（49 行），8 个片区码按 POI
    重复出现；片区清单是它去重、排序后的投影。直接取整列会得到 49 个带重复的选项，
    用户看到的下拉就是一长串重复编号。

    去重后为空（产物在但一行没有）时退回 `config.REGION_CODES`——那是片区码的规范表，
    给用户一个空下拉不如给出片区全集。缺**文件**不走这条路，由 `MissingArtifact` 拦下。
    """
    return tuple(sorted(region_labels())) or C.REGION_CODES


def location_frequency() -> pd.DataFrame:
    """库位出库频次 = 该库位承载的出库行数，与库位主数据按 `loc_id` 连接。

    频次口径与 ABC 分类、库位重排的「Σ(频次×距离)」同源（都用**出库行数**，ADR-0002）。
    """
    loc = artifact("location_master")
    freq = artifact("outbound").groupby("loc_id").size().rename("outbound_lines")
    out = loc.merge(freq, left_on="loc_id", right_index=True, how="left")
    out["outbound_lines"] = out["outbound_lines"].fillna(0).astype(int)
    return out


def routes_geojson(plan: str) -> dict:
    """某套路线的 GeoJSON（`plan` ∈ {baseline, optimized}）。

    两条路线是两条独立登记项，此处只负责「方案名 → 登记 key」的映射与校验，
    让调用方不必知道 key 的命名约定。
    """
    try:
        return artifact(f"transport_routes_{plan}")
    except UnknownArtifact as exc:  # noqa: F841（保留原因：把非法方案名与未登记 key 区分开）
        raise ValueError(f"未知路线方案：{plan!r}（应为 'baseline' 或 'optimized'）") from None


# ---------------------------------------------------------------------------
# 已解释的读取器：把「产物里的形状」也收进本层
#
# 为什么值得单独做：这些产物的嵌套路径只在**页面渲染时**才被解引用，解错了是页面上的
# KeyError——而按项目决策，页面不做自动化测试。把解释写成**纯函数**（原始 dict → 具名
# 结构），就能拿 fixture 喂、在单测里红。缺 key 一律抛错，不做兜底：那正是它的价值。
# ---------------------------------------------------------------------------
def _pick(obj: Mapping, *path: str) -> Any:
    """按路径取嵌套键；缺任何一层即抛 `KeyError`，不返回 None 兜底。"""
    cur: Any = obj
    for i, name in enumerate(path):
        if not isinstance(cur, Mapping) or name not in cur:
            raise KeyError("产物缺少字段 " + ".".join(path[: i + 1]))
        cur = cur[name]
    return cur


def _opt(obj: Mapping, *path: str) -> Any:
    """同 `_pick`，但整条路径缺失时返回 None（用于产物里**合法可缺**的字段）。"""
    cur: Any = obj
    for name in path:
        if not isinstance(cur, Mapping) or name not in cur:
            return None
        cur = cur[name]
    return cur


# --- 模块二（上）：运输 KPI -------------------------------------------------
@dataclass(frozen=True)
class PlanKpi:
    """一套派车方案（基线 / 优化）在代表日的 KPI。"""

    n_vehicles: int
    total_distance_km: float
    time_window_rate: float
    time_window_basis: str
    time_window_ceiling_rate: float | None
    gap_to_ceiling_pp: float | None
    diesel_per_order: float
    ev_per_order: float


@dataclass(frozen=True)
class TransportKpi:
    """代表日的运输 KPI：基线与优化并列，外加全量基线的天数。"""

    representative_day: str
    data_category: str
    baseline: PlanKpi
    optimized: PlanKpi
    all_days: int


def interpret_transport_kpi(raw: Mapping) -> TransportKpi:
    """把 `transport_kpi.json` 解成具名结构（纯函数，缺 key 即抛）。"""

    def plan(which: str) -> PlanKpi:
        tw = _pick(raw, which, "time_window")
        return PlanKpi(
            n_vehicles=int(_pick(raw, which, "n_vehicles")),
            total_distance_km=float(_pick(raw, which, "total_distance_km")),
            time_window_rate=float(_pick(tw, "rate")),
            time_window_basis=str(_pick(tw, "basis")),
            time_window_ceiling_rate=_opt(tw, "ceiling", "ceiling_rate"),
            gap_to_ceiling_pp=_opt(tw, "gap_to_ceiling_pp"),
            diesel_per_order=float(_pick(raw, which, "cost", "diesel", "per_order")),
            ev_per_order=float(_pick(raw, which, "cost", "ev", "per_order")),
        )

    return TransportKpi(
        representative_day=str(_pick(raw, "representative_day")),
        data_category=str(_pick(raw, "data_category")),
        baseline=plan("baseline"),
        optimized=plan("optimized"),
        all_days=int(_pick(raw, "baseline_all_days", "days")),
    )


def transport_kpi() -> TransportKpi:
    return interpret_transport_kpi(artifact("transport_kpi"))


# --- 模块一：仓内 KPI -------------------------------------------------------
@dataclass(frozen=True)
class AbcClass:
    n_sku: int
    sku_share: float
    line_share: float


@dataclass(frozen=True)
class Slotting:
    walk_cost_before: float
    walk_cost_after: float
    reduction_pct: float
    walk_m_per_line_before: float
    walk_m_per_line_after: float
    walk_sec_per_line_saved: float


@dataclass(frozen=True)
class PickSlowdown:
    sec_per_line_14_16: float
    sec_per_line_other: float
    ratio: float


@dataclass(frozen=True)
class DiscrepancyPoint:
    rate: float
    other_rate: float
    ratio: float


@dataclass(frozen=True)
class WarehouseKpi:
    """仓内 KPI 里看板真正读的那几块：ABC 三类、库位重排、两个埋点复现、库存准确率。"""

    data_category: str
    abc_by_class: Mapping[str, AbcClass]
    slotting: Slotting
    pick_slowdown: PickSlowdown
    p03_discrepancy: DiscrepancyPoint
    inventory_accuracy_rate: float


def interpret_warehouse_kpi(raw: Mapping) -> WarehouseKpi:
    """把 `kpi_overall.json` 解成具名结构（纯函数，缺 key 即抛）。

    埋点 P03 的字段名带着品类前缀（`p03_rate`、`other_rate`），此处统一成与位置无关的
    `rate` / `other_rate`——品类由 `config.HIGH_DIFF_CATEGORY` 决定，改品类名不该波及读侧。
    """
    by_class = _pick(raw, "abc", "by_class")
    slotting = _pick(raw, "slotting")
    slow = _pick(raw, "embedding_checks", "pick_slowdown_14_16")
    p03 = _pick(raw, "embedding_checks", "p03_discrepancy")
    return WarehouseKpi(
        data_category=str(_pick(raw, "data_category")),
        abc_by_class={
            cls: AbcClass(n_sku=int(_pick(by_class, cls, "n_sku")),
                          sku_share=float(_pick(by_class, cls, "sku_share")),
                          line_share=float(_pick(by_class, cls, "line_share")))
            for cls in ("A", "B", "C")
        },
        slotting=Slotting(
            walk_cost_before=float(_pick(slotting, "walk_cost_before")),
            walk_cost_after=float(_pick(slotting, "walk_cost_after")),
            reduction_pct=float(_pick(slotting, "reduction_pct")),
            walk_m_per_line_before=float(_pick(slotting, "walk_m_per_line_before")),
            walk_m_per_line_after=float(_pick(slotting, "walk_m_per_line_after")),
            walk_sec_per_line_saved=float(_pick(slotting, "walk_sec_per_line_saved")),
        ),
        pick_slowdown=PickSlowdown(
            sec_per_line_14_16=float(_pick(slow, "sec_per_line_14_16")),
            sec_per_line_other=float(_pick(slow, "sec_per_line_other")),
            ratio=float(_pick(slow, "ratio")),
        ),
        p03_discrepancy=DiscrepancyPoint(
            rate=float(_pick(p03, "p03_rate")),
            other_rate=float(_pick(p03, "other_rate")),
            ratio=float(_pick(p03, "ratio")),
        ),
        inventory_accuracy_rate=float(_pick(raw, "kpis", "inventory_accuracy", "rate")),
    )


def warehouse_kpi() -> WarehouseKpi:
    return interpret_warehouse_kpi(artifact("warehouse_kpi"))


# --- 模块二（下）：车辆数 what-if 档位缓存 ------------------------------------
@dataclass(frozen=True)
class GearPlan:
    """某档位下的一套方案结果（不可行档位的 `optimized` 为 None）。"""

    n_vehicles_used: int
    total_distance_km: float
    time_window_rate: float
    n_orders_unserved: int
    load_rate_mean: float
    cost_per_order: Mapping[str, float]


@dataclass(frozen=True)
class WhatIfGear:
    """一个车辆数档位。**不可行是结论本身**（「5 台车跑不了这一天」），故如实保留
    `feasible=False` 与原因，而不是把值填成 0——0 会被读成一个真实的零。"""

    n_vehicles_available: int
    feasible: bool
    infeasible_reason: str | None
    optimized: GearPlan | None


@dataclass(frozen=True)
class ConsistencyCheck:
    """档位 = 车队规模时，预计算解与 10 号票发表解的差值自检。

    求解的停止条件自 ADR-0015 起是**解数**而不是墙钟（routing 搜索本身没有随机源），
    同一个输入每次求解逐字段一致，故这个差值**恒为 0.0%**。它是「两处建模没走偏」的
    探测器，不是随机性的容忍带——一旦不为 0，说明有一处被改过而另一处没跟上。
    """

    gear: int
    distance_km: float
    published_distance_km: float
    distance_delta_pct: float
    note: str


@dataclass(frozen=True)
class WhatIfVehicles:
    """车辆数 what-if 预计算缓存（ADR-0010：滑块只切档，绝不实时求解）。"""

    grid: tuple[int, ...]
    representative_day: str
    n_orders: int
    n_nodes: int
    gears: Mapping[int, WhatIfGear]
    consistency_check: ConsistencyCheck | None

    def gear(self, n_vehicles: int) -> WhatIfGear | None:
        """按档位取缓存；超出网格返回 None（调用方据此提示「需重跑预计算脚本」）。

        **不插值、不外推**——车辆数对里程/成本的影响是非线性的（车少了挤不进时间窗、
        车多了要多摊固定成本），在网格之间插值会造出一个不存在的数。
        """
        return self.gears.get(int(n_vehicles))


def _gear_plan(raw: Mapping | None) -> GearPlan | None:
    if not raw:
        return None
    return GearPlan(
        n_vehicles_used=int(_pick(raw, "n_vehicles_used")),
        total_distance_km=float(_pick(raw, "total_distance_km")),
        time_window_rate=float(_pick(raw, "time_window_rate")),
        n_orders_unserved=int(_pick(raw, "n_orders_unserved")),
        load_rate_mean=float(_pick(raw, "load_rate_mean")),
        cost_per_order={m: float(_pick(raw, "cost", m, "per_order")) for m in ("diesel", "ev")},
    )


def interpret_whatif_vehicles(raw: Mapping) -> WhatIfVehicles:
    """把 `whatif_vehicles.json` 解成具名结构（纯函数，缺 key 即抛）。

    产物里档位记录是「扁平字段 + 一个嵌套的 `baseline` 兄弟键」，优化后方案的字段直接摊在
    档位顶层——这里把它拆成 `optimized` / `baseline` 两个具名位置，读侧不必再记这个形状。
    """
    gears = {
        int(n): WhatIfGear(
            n_vehicles_available=int(_pick(rec, "n_vehicles_available")),
            feasible=bool(_pick(rec, "feasible")),
            infeasible_reason=_opt(rec, "infeasible_reason"),
            optimized=_gear_plan(rec) if _pick(rec, "feasible") else None,
        )
        for n, rec in _pick(raw, "gears").items()
    }
    chk = _opt(raw, "consistency_check")
    return WhatIfVehicles(
        grid=tuple(int(g) for g in _pick(raw, "grid")),
        representative_day=str(_pick(raw, "representative_day")),
        n_orders=int(_pick(raw, "n_orders")),
        n_nodes=int(_pick(raw, "n_nodes")),
        gears=gears,
        consistency_check=ConsistencyCheck(
            gear=int(_pick(chk, "gear")),
            distance_km=float(_pick(chk, "distance_km")),
            published_distance_km=float(_pick(chk, "published_distance_km")),
            distance_delta_pct=float(_pick(chk, "distance_delta_pct")),
            note=str(_pick(chk, "note")),
        ) if chk else None,
    )


def whatif_vehicles() -> WhatIfVehicles:
    return interpret_whatif_vehicles(artifact("transport_whatif"))
