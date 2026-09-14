"""数据层 F：SimPy 仓内作业离散事件仿真（07 号票）。

仿真流程：订单非齐次泊松到达（9–11、14–16 双高峰）→ 拣货员（有限资源）按订单行
逐行拣货（行走 = SKU 库位距离 / 行走速度 + 操作耗时对数正态 ~12s/行，14–16 点整段
放大与数据层 A 互证）→ 复核台（有限资源）三角分布(30/60/120)s → 发货。

两组对照实验（各 30 次重复，报均值 + 95% CI）：
  实验一（布局）：原始（A 类远库位）vs ABC 分区（A 类近库位），比单均拣货时长 + 总行走距离；
  实验二（人力）：拣货员 4/5/6 人（布局固定 ABC 分区目标态），比履约总时长 + 利用率，
                 输出「时长—人力成本」权衡曲线与拐点。

校准（ADR-0001）：仿真拣货秒/行锚定数据层 A 的 outbound_orders（同尺度，比 mean/CV）；
与 Olist 仅做出库响应子段的无量纲形状对照（CV/偏度），跨尺度、不做绝对时长 KS。

子种子由 SEED_SIM 经 SeedSequence 派生，固定可复现。各档位预计算缓存到 processed/sim/。

运行方式（项目根）：python -m src.warehouse_sim
数据类别：过程仿真（见 data_sources_ledger.md 第 3 节）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import simpy
from scipy import stats

from src import config as C

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 仿真内部参数（情景假设，登记台账）
#
# 与数据层 A **共用**的那几项一律读 config（`WAREHOUSE_*` 那组），本模块不再自存一份。
# 两层的关系是「互相印证」，而互证只有在两边真的用同一组参数时才成立：先前各写一份、
# 靠注释声明「与数据层 A 一致」，只改一边就静默失效，而写着一致的那行注释还留在原处。
# ---------------------------------------------------------------------------
_DAY_START_HOUR, _DAY_END_HOUR = C.WAREHOUSE_OPEN_HOURS  # 营业时段（秒轴 0 点 = 起点）
# 代表日规模与到达集中度：读 config（情景假设，口径见 config 注释）
_N_ORDERS_PER_DAY = C.SIM_N_ORDERS_PER_DAY
_PEAK_INTENSITY = C.WAREHOUSE_PEAK_INTENSITY  # 到达强度（与数据层 A 同源，见 config 说明）


def _abc_class_share() -> dict[str, float]:
    """ABC 三类各自的 SKU 份额，由 `config.ABC_THRESHOLDS` 的累计切点推出。

    不另存一份 `{"A": 0.70, "B": 0.20, "C": 0.10}`——那三个数就是 70/90/100 的差分，
    存两份等于多造一处会漂移的地方。
    """
    a_cut, b_cut = C.ABC_THRESHOLDS
    return {"A": a_cut, "B": b_cut - a_cut, "C": 1.0 - b_cut}


def derive_seed(base: int, *parts) -> int:
    """由基础种子 + 标识派生确定性子种子（SeedSequence，跨进程/跨版本稳定）。

    注意：必须用 hashlib 而非 builtin hash()——后者对 str 有 PYTHONHASHSEED
    进程级随机化，会导致跨进程派生出不同子种子、产物无法复现（审计 2026-09-13 实测捕获）。
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    stable = int.from_bytes(digest[:8], "little") % (2**31)
    return int(np.random.SeedSequence([base, stable]).generate_state(1)[0])


# ---------------------------------------------------------------------------
# 布局：SKU → 行走距离（米）
# ---------------------------------------------------------------------------
def load_sku_location_assignment(path: Path | None = None) -> pd.DataFrame:
    """读数据层 A 交付的 SKU→库位分配（`config.WAREHOUSE_SKU_LOCATION_CSV`）。

    缺产物即报错并给出运行指引，不降级、不近似——近似出来的基线会让实验一的互证失效。
    """
    p = C.WAREHOUSE_SKU_LOCATION_CSV if path is None else Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"缺少数据层 A 的库位分配产物：{p}\n请先运行：python -m src.gen_warehouse_data"
        )
    return pd.read_csv(p)


def build_layout(
    sku_df: pd.DataFrame,
    loc_df: pd.DataFrame,
    mode: str,
    *,
    assignment: pd.DataFrame | None = None,
) -> dict[str, float]:
    """构建 SKU→行走距离映射。

    mode='original'：**读数据层 A 交付的库位分配**，即原始布局本身。必须传 `assignment`——
    本模块原先用 `abc_initial_label` 近似复刻它（同类内按 sku_id 排序，而不是按需求频率），
    逐 SKU 与真实布局并不相同（实测 500 个里只有 28 个距离相同）。实验一要拿这一臂去比对
    数据层 A 的 outbound（两条证据链互证），基线必须是同一份布局，不能是它的近似。

    mode='abc_zoned'：ABC 分区目标态（A 类近库位、C 类远库位）。这是实验的**处理组**，
    由本模块构造，不读交付——它本来就该是本模块自己的干预，而不是从数据层 A 搬来的。
    """
    if mode not in ("original", "abc_zoned"):
        raise ValueError(f"未知布局模式: {mode}")
    if mode == "original":
        if assignment is None:
            raise ValueError(
                "mode='original' 需要数据层 A 交付的库位分配（见 load_sku_location_assignment）；"
                "按 ABC 标签近似复刻的布局与原始布局逐 SKU 并不相同，不能当互证基线"
            )
        dist = dict(zip(loc_df["loc_id"], loc_df["walk_dist_m"]))
        missing = set(assignment["loc_id"]) - set(dist)
        if missing:
            raise ValueError(f"库位分配里有 {len(missing)} 个 loc_id 不在库位主数据中")
        return {str(s): float(dist[l])
                for s, l in zip(assignment["sku_id"], assignment["loc_id"])}

    sku = sku_df.copy()
    rank = {"A": 0, "B": 1, "C": 2}
    sku["_rank"] = sku["abc_initial_label"].map(rank)
    sku = sku.sort_values(["_rank", "sku_id"]).reset_index(drop=True)
    dists = np.sort(loc_df["walk_dist_m"].to_numpy())  # 升序
    n = len(sku)
    sampled_pos = np.linspace(0, len(dists) - 1, n).astype(int)
    pool_near_first = dists[sampled_pos]  # 升序（近→远）
    return dict(zip(sku["sku_id"].tolist(), pool_near_first.tolist()))


def sku_sampling_weights(sku_df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """按 ABC 类别份额构造 SKU 抽样权重（A 类高频被抽中概率大），与布局无关。"""
    share = _abc_class_share()
    sku_ids = sku_df["sku_id"].tolist()
    labels = sku_df["abc_initial_label"].tolist()
    counts = pd.Series(labels).value_counts().to_dict()
    w = np.array([share[lbl] / counts[lbl] for lbl in labels], dtype=float)
    return sku_ids, w / w.sum()


# ---------------------------------------------------------------------------
# 到达过程：非齐次泊松（NHPP）
# ---------------------------------------------------------------------------
def minute_intensity(peak_intensity: float = _PEAK_INTENSITY) -> np.ndarray:
    """分钟粒度到达强度（08:00–18:00），9–11、14–16 双高峰。归一化为概率。"""
    minutes = np.arange(_DAY_START_HOUR * 60, _DAY_END_HOUR * 60)
    hours = minutes // 60
    intensity = np.ones(len(minutes))
    for h0, h1 in C.SIM_PEAK_HOURS:
        intensity[(hours >= h0) & (hours < h1)] = peak_intensity
    return intensity / intensity.sum()


def sample_arrival_seconds(
    rng: np.random.Generator, n_orders: int, peak_intensity: float = _PEAK_INTENSITY
) -> np.ndarray:
    """按 NHPP 抽样 n_orders 个到达时刻（秒，自 08:00 起），升序。"""
    p = minute_intensity(peak_intensity)
    grid = np.arange(_DAY_START_HOUR * 60, _DAY_END_HOUR * 60)
    minutes = rng.choice(grid, size=n_orders, p=p)
    secs = minutes * 60 + rng.integers(0, 60, n_orders)
    return np.sort(secs)


def pregenerate_orders(
    rng: np.random.Generator,
    n_orders: int,
    sku_ids: list[str],
    sku_weights: np.ndarray,
    peak_intensity: float = _PEAK_INTENSITY,
) -> list[tuple[float, list[str]]]:
    """预生成订单批次：[(到达秒, [SKU,...]), ...]，SKU 按 ABC 频率加权抽样。"""
    arrivals = sample_arrival_seconds(rng, n_orders, peak_intensity)
    lines_each = rng.choice([1, 2, 3], size=n_orders, p=C.WAREHOUSE_LINES_PER_ORDER_P)
    orders = []
    for i in range(n_orders):
        k = int(lines_each[i])
        skus = list(rng.choice(len(sku_ids), size=k, p=sku_weights))
        orders.append((float(arrivals[i]), [sku_ids[j] for j in skus]))
    return orders


# ---------------------------------------------------------------------------
# SimPy 仿真核心
# ---------------------------------------------------------------------------
def _order_process(
    env: simpy.Environment,
    arrival_sec: float,
    line_skus: list[str],
    pickers: simpy.Resource,
    review: simpy.Resource,
    layout: dict[str, float],
    rng: np.random.Generator,
    rec: dict,
) -> None:
    """单订单作业流：等拣货员 → 逐行拣货 → 等复核台 → 复核打包 → 发货。"""
    handle_mu = math.log(C.PICK_SECONDS_PER_LINE_MEAN) - C.WAREHOUSE_PICK_HANDLE_SIGMA**2 / 2
    yield env.timeout(arrival_sec - env.now)
    arrive = env.now

    with pickers.request() as req:
        yield req
        pick_start = env.now
        walk_total = 0.0
        for sku in line_skus:
            walk = layout[sku] / C.WALK_SPEED_M_PER_SEC
            handle = rng.lognormal(handle_mu, C.WAREHOUSE_PICK_HANDLE_SIGMA)
            seg = walk + handle
            # env.now 为绝对秒轴（08:00 = 28800s），整除即得时钟小时
            hour = int(env.now // 3600)
            h0, h1 = C.PICK_SLOW_HOURS
            is_slow = h0 <= hour < h1
            if is_slow:  # 14–16 点整段放大（数据层 A 埋点③）
                seg *= rng.uniform(*C.WAREHOUSE_SLOWDOWN_RANGE)
            walk_total += walk
            rec["line_pick_secs"].append(seg)
            rec["line_slow"].append(is_slow)
            yield env.timeout(seg)
        pick_end = env.now

    rec["order_pick_secs"].append(pick_end - pick_start)
    rec["order_walk_m"].append(walk_total)
    rec["picker_busy"] += pick_end - pick_start

    with review.request() as req:
        yield req
        yield env.timeout(rng.triangular(*C.PACK_TRIANGULAR))
    ship = env.now

    rec["fulfillment_secs"].append(ship - arrive)
    rec["arrivals"].append(arrive)
    rec["ships"].append(ship)


def run_one_sim(
    layout: dict[str, float],
    n_pickers: int,
    seed: int,
    orders: list[tuple[float, list[str]]] | None = None,
    n_orders: int = _N_ORDERS_PER_DAY,
    peak_intensity: float = _PEAK_INTENSITY,
    n_review: int | None = None,
    sku_ids: list[str] | None = None,
    sku_weights: np.ndarray | None = None,
) -> dict:
    """单次仿真（一个代表日）。返回指标 dict（含校准用的逐行拣货秒统计）。

    orders 为 None 时按 sku_ids/sku_weights 现抽（用于布局实验：固定到达流、只变距离）。
    """
    n_review = C.SIM_REVIEW_STATIONS if n_review is None else n_review
    rng = np.random.default_rng(seed)
    if orders is None:
        if sku_ids is None or sku_weights is None:
            sku_ids = list(layout.keys())
            sku_weights = np.ones(len(sku_ids)) / len(sku_ids)
        orders = pregenerate_orders(rng, n_orders, sku_ids, sku_weights, peak_intensity)

    env = simpy.Environment()
    pickers = simpy.Resource(env, capacity=n_pickers)
    review = simpy.Resource(env, capacity=n_review)
    rec: dict = {
        "line_pick_secs": [], "line_slow": [], "order_pick_secs": [],
        "order_walk_m": [], "fulfillment_secs": [], "arrivals": [], "ships": [],
        "picker_busy": 0.0,
    }
    for arrival_sec, line_skus in orders:
        env.process(
            _order_process(env, arrival_sec, line_skus, pickers, review, layout, rng, rec)
        )
    env.run()

    ful = np.array(rec["fulfillment_secs"])
    makespan = (max(rec["ships"]) - min(rec["arrivals"])) if rec["arrivals"] else 1.0
    util = rec["picker_busy"] / (n_pickers * makespan) if makespan > 0 else 0.0
    line_secs = np.array(rec["line_pick_secs"])
    slow_mask = np.array(rec["line_slow"], dtype=bool)

    return {
        "n_orders": len(orders),
        "n_pickers": n_pickers,
        "avg_order_pick_sec": float(np.mean(rec["order_pick_secs"])),
        "total_walk_m": float(np.sum(rec["order_walk_m"])),
        "avg_walk_m_per_order": float(np.mean(rec["order_walk_m"])),
        "avg_fulfillment_sec": float(np.mean(ful)),
        "fulfillment_cv": float(np.std(ful) / np.mean(ful)) if np.mean(ful) > 0 else 0.0,
        "fulfillment_skew": float(stats.skew(ful)),
        "picker_utilization": float(util),
        "line_pick_sec_mean": float(np.mean(line_secs)),
        "line_pick_sec_cv": float(np.std(line_secs) / np.mean(line_secs)),
        "line_pick_sec_mean_other": float(np.mean(line_secs[~slow_mask])) if (~slow_mask).any() else float("nan"),
        "line_pick_sec_mean_slow": float(np.mean(line_secs[slow_mask])) if slow_mask.any() else float("nan"),
        "makespan_sec": float(makespan),
    }


# ---------------------------------------------------------------------------
# 统计聚合：30 次重复 → 均值 + 95% CI
# ---------------------------------------------------------------------------
def aggregate_repeats(runs: list[dict], metric: str) -> dict:
    """对 n 次重复的某指标计算均值 + 95% CI（t 分布）。"""
    vals = np.array([r[metric] for r in runs], dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    mean = float(vals.mean())
    if n >= 2:
        sem = float(stats.sem(vals))
        h = float(stats.t.ppf(0.975, n - 1) * sem)
    else:
        h = 0.0
    return {"mean": round(mean, 3), "ci95_low": round(mean - h, 3),
            "ci95_high": round(mean + h, 3), "n": n, "std": round(float(vals.std(ddof=1)) if n >= 2 else 0.0, 3)}


def run_experiment_arm(
    layout: dict[str, float], n_pickers: int, base_seed: int,
    arm_id: str, repeats: int = C.SIM_REPEATS,
    sku_ids: list[str] | None = None, sku_weights: np.ndarray | None = None,
    orders: list[tuple[float, list[str]]] | None = None,
) -> dict:
    """一个实验档位的 repeats 次重复 + 聚合。

    orders 给定（布局实验）→ 各重复共用同一订单流，只变距离；
    orders=None（人力实验）→ 各重复用派生子种子独立抽订单。
    """
    runs = []
    for r in range(repeats):
        seed = derive_seed(base_seed, arm_id, r)
        run_rng = np.random.default_rng(seed)
        arm_orders = orders
        if arm_orders is None:
            arm_orders = pregenerate_orders(
                run_rng, _N_ORDERS_PER_DAY, sku_ids, sku_weights, _PEAK_INTENSITY
            )
        runs.append(run_one_sim(layout, n_pickers, seed, orders=arm_orders))
    metrics = ["avg_order_pick_sec", "total_walk_m", "avg_walk_m_per_order",
               "avg_fulfillment_sec", "picker_utilization", "fulfillment_cv", "fulfillment_skew",
               "line_pick_sec_mean", "line_pick_sec_cv"]
    agg = {m: aggregate_repeats(runs, m) for m in metrics}
    return {"arm_id": arm_id, "n_pickers": n_pickers, "repeats": repeats, "metrics": agg}


# ---------------------------------------------------------------------------
# 权衡曲线与拐点
# ---------------------------------------------------------------------------
def tradeoff_curve_and_knee(arms: list[dict]) -> dict:
    """实验二「时长—人力成本」权衡曲线与拐点。

    人力成本 = 拣货员数 × PICKER_DAILY_COST。

    **拐点只有在「边际收益递减」真的成立时才报出来**：各档边际收益逐档下降、且首档为正，
    取首档之后的那一档（即「过了这里，再加人买的就少了」）。否则 `knee_at_pickers` 为 `None`
    并给出 `knee_note`，两种情形都如实区分：

      - 最大边际落在**最后一档** → 说明「再加人还在变好」，是测试区间没覆盖到拐点，不是拐点；
      - 各档边际都在噪声量级 → 分不出拐点。

    这条规则是 2026-09-15 补的。原实现取 `sec_saved_per_yuan` 的**最大值**当拐点，与 docstring
    写的「边际收益骤降处」是两回事：只要边际非单调，它就会翻到最后一档。数据层 A/F 的到达
    强度统一到 1.8 之后，各档边际变成 [0.0005, 0.0065] 秒/元（都在噪声内），原实现便报出
    「拐点 = 6 人」——而 6 人正是测试区间的上界，那不是拐点。
    """
    pts = sorted(
        [(a["n_pickers"], a["metrics"]["avg_fulfillment_sec"]["mean"],
          a["n_pickers"] * C.PICKER_DAILY_COST) for a in arms],
        key=lambda x: x[0],
    )
    curve = [{"n_pickers": p, "avg_fulfillment_sec": t, "daily_labor_cost": c} for p, t, c in pts]
    # 边际：每增 1 人，履约时长降多少 / 成本增多少
    marginals = []
    for i in range(1, len(pts)):
        dp = pts[i][0] - pts[i - 1][0]
        dt = pts[i][1] - pts[i - 1][1]  # 负=改善
        dc = pts[i][2] - pts[i - 1][2]
        marginals.append({
            "from_pickers": pts[i - 1][0], "to_pickers": pts[i][0],
            "delta_fulfillment_sec": round(dt, 2), "delta_cost": round(dc, 2),
            "sec_saved_per_yuan": round(-dt / dc, 4) if dc else None,
        })

    vals = [m["sec_saved_per_yuan"] or 0.0 for m in marginals]
    if len(vals) < 2:
        knee, note = None, "档位不足两档，无法判断边际收益是否递减"
    elif not all(b < a for a, b in zip(vals, vals[1:])):
        top = max(range(len(vals)), key=lambda i: vals[i])
        knee = None
        note = (
            f"各档边际收益不是逐档下降（{vals}），最大边际落在第 {top + 1} 段——"
            "「再加人还在变好」说明测试区间没覆盖到拐点，而不是存在拐点"
            if top == len(vals) - 1 else
            f"各档边际收益非单调（{vals}），拐点不可判定"
        )
    elif vals[0] <= 0:
        knee, note = None, f"首档边际收益已非正（{vals[0]} 秒/元），不存在性价比拐点"
    else:
        knee = marginals[0]["to_pickers"]
        note = f"边际收益逐档下降（{vals}），{knee} 人之后每元买到的改善显著变小"

    return {"curve": curve, "marginals": marginals,
            "knee_at_pickers": knee, "knee_note": note}


# ---------------------------------------------------------------------------
# 校准（ADR-0001）
# ---------------------------------------------------------------------------
def calibrate_against_layer_a(sim_runs: list[dict], outbound_csv: Path) -> dict:
    """仿真拣货秒/行 mean/CV 与数据层 A outbound_orders 实测对比（同尺度）。

    数据层 A 的拣货秒/行 = (pick_end − pick_start) / 订单行数，按行近似取每单时长/行数。
    """
    ob = pd.read_csv(outbound_csv, parse_dates=["pick_start", "pick_end"])
    ob["pick_sec"] = (ob["pick_end"] - ob["pick_start"]).dt.total_seconds()
    obs_mean = float(ob["pick_sec"].mean())
    obs_cv = float(ob["pick_sec"].std() / obs_mean)
    # 仿真侧：取各重复 line_pick_sec_mean 的均值（与数据层 A 整段口径一致）
    sim_mean = float(np.mean([r["line_pick_sec_mean"] for r in sim_runs]))
    sim_cv = float(np.mean([r["line_pick_sec_cv"] for r in sim_runs]))
    return {
        "scale": "同尺度（秒/行）",
        "layer_a": {"pick_sec_per_line_mean": round(obs_mean, 2), "cv": round(obs_cv, 3)},
        "sim": {"pick_sec_per_line_mean": round(sim_mean, 2), "cv": round(sim_cv, 3)},
        "mean_ratio_sim_over_a": round(sim_mean / obs_mean, 3),
        "note": "锚定数据层 A；CV 接近即形状一致，无需调参；偏差大时记录调参过程。",
    }


def calibrate_against_olist(sim_runs: list[dict], clean_orders_csv: Path) -> dict:
    """与 Olist 出库响应子段做无量纲形状对照（CV/偏度），跨尺度、不做绝对时长 KS。"""
    co = pd.read_csv(clean_orders_csv)
    resp = pd.to_numeric(co["outbound_response_days"], errors="coerce").dropna()
    resp = resp[resp >= 0]
    olist_cv = float(resp.std() / resp.mean())
    olist_skew = float(stats.skew(resp))
    sim_cv = float(np.mean([r["fulfillment_cv"] for r in sim_runs]))
    sim_skew = float(np.mean([r["fulfillment_skew"] for r in sim_runs]))
    return {
        "scale": "跨尺度（仿真=秒级仓内履约，Olist=天级出库响应）——仅比无量纲形状",
        "olist_outbound_response": {"cv": round(olist_cv, 3), "skew": round(olist_skew, 3)},
        "sim_fulfillment": {"cv": round(sim_cv, 3), "skew": round(sim_skew, 3)},
        "note": "ADR-0001：不做绝对时长 KS（尺度不可比）；CV/偏度量级接近即形状对照通过。",
    }


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------
def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_all_experiments(
    warehouse_dir: Path | None = None,
    out_dir: Path | None = None,
    olist_dir: Path | None = None,
) -> dict[str, Path]:
    """跑两组对照实验 + 校准，预计算缓存到 processed/sim/。返回产物路径。"""
    warehouse_dir = Path(warehouse_dir) if warehouse_dir else C.WAREHOUSE_DIR
    out_dir = Path(out_dir) if out_dir else C.PROCESSED_DIR / "sim"
    olist_dir = Path(olist_dir) if olist_dir else C.PROCESSED_DIR / "olist"
    out_dir.mkdir(parents=True, exist_ok=True)

    sku_df = pd.read_csv(warehouse_dir / "sku_master.csv")
    loc_df = pd.read_csv(warehouse_dir / "location_master.csv")
    assignment = load_sku_location_assignment(
        warehouse_dir / C.WAREHOUSE_SKU_LOCATION_CSV.name)
    sku_ids, sku_w = sku_sampling_weights(sku_df)

    # 原始布局读交付（同一份布局本身）；ABC 分区是本模块构造的处理组
    layout_orig = build_layout(sku_df, loc_df, "original", assignment=assignment)
    layout_abc = build_layout(sku_df, loc_df, "abc_zoned")

    base_seed = C.SEED_SIM
    # 布局实验固定订单流（到达 + SKU 序列），只变库位距离
    order_rng = np.random.default_rng(derive_seed(base_seed, "layout_orders"))
    fixed_orders = pregenerate_orders(order_rng, _N_ORDERS_PER_DAY, sku_ids, sku_w)

    n_pick = 5  # 布局实验固定人力（实验二档位的中位）
    logger.info("实验一（布局）：original vs abc_zoned，各 %d 次重复 ...", C.SIM_REPEATS)
    exp1_orig = run_experiment_arm(layout_orig, n_pick, base_seed, "exp1_original", orders=fixed_orders)
    exp1_abc = run_experiment_arm(layout_abc, n_pick, base_seed, "exp1_abc_zoned", orders=fixed_orders)

    logger.info("实验二（人力）：拣货员 %s 人，布局=abc_zoned，各 %d 次重复 ...",
                C.SIM_PICKER_LEVELS, C.SIM_REPEATS)
    exp2_arms = []
    for np_pick in C.SIM_PICKER_LEVELS:
        arm = run_experiment_arm(layout_abc, np_pick, base_seed, f"exp2_p{np_pick}",
                                 sku_ids=sku_ids, sku_weights=sku_w)
        exp2_arms.append(arm)
    tradeoff = tradeoff_curve_and_knee(exp2_arms)

    # 校准：用【原始布局臂】对比数据层 A——该臂现在读的正是本层交付的那份库位分配，
    # 与生成 outbound 的布局逐 SKU 相同（不再是按 ABC 标签的近似），同布局同尺度才可比；
    # 用 abc_zoned 臂会把「重排提速」误算成「仿真偏差」。CV/偏度跨尺度形状对照同理用原始臂。
    sim_runs_proxy = [{
        "line_pick_sec_mean": exp1_orig["metrics"]["line_pick_sec_mean"]["mean"],
        "line_pick_sec_cv": exp1_orig["metrics"]["line_pick_sec_cv"]["mean"],
        "fulfillment_cv": exp1_orig["metrics"]["fulfillment_cv"]["mean"],
        "fulfillment_skew": exp1_orig["metrics"]["fulfillment_skew"]["mean"],
    }]
    calib_a = calibrate_against_layer_a(sim_runs_proxy, warehouse_dir / "outbound_orders.csv")
    calib_olist = calibrate_against_olist(sim_runs_proxy, olist_dir / "clean_orders.csv")

    paths: dict[str, Path] = {}
    paths["exp1"] = out_dir / "exp1_layout.json"
    _write_json({"original": exp1_orig, "abc_zoned": exp1_abc,
                 "improvement": {
                     "walk_m_reduction_pct": round(
                         (1 - exp1_abc["metrics"]["total_walk_m"]["mean"]
                          / exp1_orig["metrics"]["total_walk_m"]["mean"]) * 100, 1),
                     "pick_sec_reduction_pct": round(
                         (1 - exp1_abc["metrics"]["avg_order_pick_sec"]["mean"]
                          / exp1_orig["metrics"]["avg_order_pick_sec"]["mean"]) * 100, 1),
                 }}, paths["exp1"])
    paths["exp2"] = out_dir / "exp2_staffing.json"
    _write_json({"arms": exp2_arms, "tradeoff": tradeoff}, paths["exp2"])
    paths["calibration"] = out_dir / "calibration.json"
    _write_json({"against_layer_a": calib_a, "against_olist_shape": calib_olist,
                 "adr": "0001（不做跨尺度绝对时长 KS）"}, paths["calibration"])

    # what-if 缓存：拣货员人数 → 指标（供看板滑块直接读，ADR-0010 同哲学）
    whatif = {str(a["n_pickers"]): {
        "avg_fulfillment_sec": a["metrics"]["avg_fulfillment_sec"],
        "picker_utilization": a["metrics"]["picker_utilization"],
        "daily_labor_cost": a["n_pickers"] * C.PICKER_DAILY_COST,
    } for a in exp2_arms}
    paths["whatif_staffing"] = out_dir / "whatif_staffing.json"
    _write_json(whatif, paths["whatif_staffing"])

    logger.info(
        "实验完成：布局重排降行走 %.1f%% / 降拣货时长 %.1f%%；人力拐点=%s 人；"
        "仿真拣货 %.1f s/行 vs 数据层A %.1f s/行",
        (1 - exp1_abc["metrics"]["total_walk_m"]["mean"] / exp1_orig["metrics"]["total_walk_m"]["mean"]) * 100,
        (1 - exp1_abc["metrics"]["avg_order_pick_sec"]["mean"] / exp1_orig["metrics"]["avg_order_pick_sec"]["mean"]) * 100,
        tradeoff["knee_at_pickers"],
        calib_a["sim"]["pick_sec_per_line_mean"], calib_a["layer_a"]["pick_sec_per_line_mean"],
    )
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = run_all_experiments()
    print("\n=== 数据层 F 产物 ===")
    for k, v in paths.items():
        print(f"  {k:18s} -> {v}")


if __name__ == "__main__":
    main()
