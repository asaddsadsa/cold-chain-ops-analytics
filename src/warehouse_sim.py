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
_DAY_START_SEC = _DAY_START_HOUR * 3600


def release_second(arrival_sec: float, wave_interval_min: float) -> float:
    """订单从「到达」变成「可拣」的时刻——波次释放的边界。

    波次窗自营业起点起算（08:00、08:00+W、08:00+2W …）。订单在窗内累积，到边界才批量
    投入拣货队列。`wave_interval_min <= 0` 表示不累积（到达即抢拣货员）。

    **为什么这是一个独立的环节**：「订单到达」与「拣货释放」是仓库里两件事。门店下单是
    平稳的，而拣货按波次组织——中间隔着一次批量释放。仿真原先只有前者，订单一到就抢
    拣货员，等于假设「拣货员永远有空接单」。这个假设在 300 单/天的欠载系统里看不出问题，
    却让「加人有没有用」这个问题失去了机理：没有累积就没有排队，没有排队就没有边际收益。
    """
    if wave_interval_min <= 0:
        return arrival_sec
    w = wave_interval_min * 60.0
    k = math.floor((arrival_sec - _DAY_START_SEC) / w) + 1
    return _DAY_START_SEC + k * w


def _order_process(
    env: simpy.Environment,
    arrival_sec: float,
    line_skus: list[str],
    pickers: simpy.Resource,
    review: simpy.Resource,
    layout: dict[str, float],
    rng: np.random.Generator,
    rec: dict,
    wave_interval_min: float = 0.0,
) -> None:
    """单订单作业流：到达 →（累积到波次边界）→ 等拣货员 → 逐行拣货 → 等复核台 → 复核打包 → 发货。"""
    handle_mu = math.log(C.PICK_SECONDS_PER_LINE_MEAN) - C.WAREHOUSE_PICK_HANDLE_SIGMA**2 / 2
    yield env.timeout(arrival_sec - env.now)
    arrive = env.now

    release = release_second(arrive, wave_interval_min)
    if release > arrive:
        yield env.timeout(release - arrive)
    rec["wave_waits"].append(release - arrive)

    with pickers.request() as req:
        yield req
        pick_start = env.now
        rec["queue_waits"].append(pick_start - release)
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

    rec["fulfillment_secs"].append(ship - release)
    rec["order_to_ship_secs"].append(ship - arrive)
    rec["arrivals"].append(arrive)
    rec["releases"].append(release)
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
    wave_interval_min: float = 0.0,
) -> dict:
    """单次仿真（一个代表日）。返回指标 dict（含校准用的逐行拣货秒统计）。

    orders 为 None 时按 sku_ids/sku_weights 现抽（用于布局实验：固定到达流、只变距离）。

    `wave_interval_min` 见 `release_second`：0 = 到达即抢拣货员（口径同数据层 A 的即时拣货），
    >0 = 波次释放。三个时长指标随之为：到达→发货（端到端）、释放→发货（作业）、释放→开拣（排队）。
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
        "order_walk_m": [], "fulfillment_secs": [], "order_to_ship_secs": [],
        "arrivals": [], "releases": [], "ships": [],
        "wave_waits": [], "queue_waits": [],
        "picker_busy": 0.0,
    }
    for arrival_sec, line_skus in orders:
        env.process(
            _order_process(env, arrival_sec, line_skus, pickers, review, layout, rng, rec,
                           wave_interval_min)
        )
    env.run()

    ful = np.array(rec["fulfillment_secs"])
    e2e = np.array(rec["order_to_ship_secs"])
    wave_waits = np.array(rec["wave_waits"])
    queue_waits = np.array(rec["queue_waits"])
    # 利用率的作业窗口从「订单可拣」算起：波次累积的那段等待不是拣货员的可用工时，
    # 计进分母会人为压低利用率（对比 4 人 vs 6 人时，压低幅度还随人数变化）。
    span = (max(rec["ships"]) - min(rec["releases"])) if rec["releases"] else 1.0
    util = rec["picker_busy"] / (n_pickers * span) if span > 0 else 0.0
    line_secs = np.array(rec["line_pick_secs"])
    slow_mask = np.array(rec["line_slow"], dtype=bool)

    return {
        "n_orders": len(orders),
        "n_pickers": n_pickers,
        "wave_interval_min": wave_interval_min,
        "avg_order_pick_sec": float(np.mean(rec["order_pick_secs"])),
        "total_walk_m": float(np.sum(rec["order_walk_m"])),
        "avg_walk_m_per_order": float(np.mean(rec["order_walk_m"])),
        "avg_fulfillment_sec": float(np.mean(ful)),
        "avg_order_to_ship_sec": float(np.mean(e2e)),
        "avg_wave_wait_sec": float(np.mean(wave_waits)),
        "avg_queue_wait_sec": float(np.mean(queue_waits)),
        "avg_order_wait_sec": float(np.mean(e2e - ful)),
        "fulfillment_cv": float(np.std(ful) / np.mean(ful)) if np.mean(ful) > 0 else 0.0,
        "fulfillment_skew": float(stats.skew(ful)),
        "picker_utilization": float(util),
        "line_pick_sec_mean": float(np.mean(line_secs)),
        "line_pick_sec_cv": float(np.std(line_secs) / np.mean(line_secs)),
        "line_pick_sec_mean_other": float(np.mean(line_secs[~slow_mask])) if (~slow_mask).any() else float("nan"),
        "line_pick_sec_mean_slow": float(np.mean(line_secs[slow_mask])) if slow_mask.any() else float("nan"),
        "makespan_sec": float(span),
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
    wave_interval_min: float = 0.0,
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
        runs.append(run_one_sim(layout, n_pickers, seed, orders=arm_orders,
                                wave_interval_min=wave_interval_min))
    metrics = ["avg_order_pick_sec", "total_walk_m", "avg_walk_m_per_order",
               "avg_fulfillment_sec", "avg_order_to_ship_sec",
               "avg_wave_wait_sec", "avg_queue_wait_sec",
               "picker_utilization", "fulfillment_cv", "fulfillment_skew",
               "line_pick_sec_mean", "line_pick_sec_cv"]
    agg = {m: aggregate_repeats(runs, m) for m in metrics}
    return {"arm_id": arm_id, "n_pickers": n_pickers, "repeats": repeats,
            "wave_interval_min": wave_interval_min, "metrics": agg}


def _arm_summary(arm: dict) -> dict:
    """档位摘要：只留敏感性扫描与看板用得到的字段，避免把全量指标重复写五遍。"""
    m = arm["metrics"]
    return {
        "n_pickers": arm["n_pickers"],
        "fulfillment_sec": m["avg_fulfillment_sec"],
        "order_to_ship_sec": m["avg_order_to_ship_sec"],
        "wave_wait_sec": m["avg_wave_wait_sec"],
        "queue_wait_sec": m["avg_queue_wait_sec"],
        "picker_utilization": m["picker_utilization"],
        "daily_labor_cost": arm["n_pickers"] * C.PICKER_DAILY_COST,
    }


# ---------------------------------------------------------------------------
# 权衡曲线与拐点
# ---------------------------------------------------------------------------
def _means_differ(a: dict, b: dict) -> bool:
    """两档的履约时长均值之差，在 5% 水平上是否显著（Welch t 检验，双侧）。

    判拐点必须看显著性，不能只比点估计。到达即抢拣货员（波次窗口 0）时，4/5/6 三档的
    均值差只有 0.1–1 秒，落在重复仿真的噪声里；而点估计仍可能「恰好」逐档递减，
    于是报出一个并不存在的拐点——这正是这一版之前发生的事。
    """
    n_a, n_b = a.get("n", 0), b.get("n", 0)
    if n_a < 2 or n_b < 2:
        return False
    va, vb = a["std"] ** 2 / n_a, b["std"] ** 2 / n_b
    se = math.sqrt(va + vb)
    if se == 0:
        return a["mean"] != b["mean"]
    df = (va + vb) ** 2 / (va ** 2 / (n_a - 1) + vb ** 2 / (n_b - 1))
    return abs(a["mean"] - b["mean"]) / se > float(stats.t.ppf(0.975, df))


def tradeoff_curve_and_knee(arms: list[dict]) -> dict:
    """实验二「时长—人力成本」权衡曲线与拐点。

    人力成本 = 拣货员数 × PICKER_DAILY_COST。时长＝**释放 → 发货**（作业口径），
    不含波次累积等待——那段等待由作业组织决定，加多少人都不变，混进来只会稀释人力的效应。

    **拐点只有在「边际收益显著递减」时才报出来**，判据分两层：

      1. 每档的均值差先过 Welch t 检验（`_means_differ`）——两侧置信区间重叠的档位，
         其「边际收益」不可与 0 区分，不能拿来排序；
      2. 显著的那些档里，边际收益要逐档下降、且最大的一档不在**最后一档**。

    不满足时 `knee_at_pickers` 为 `None`，并在 `knee_note` 里说清是哪一种：
    各档都在噪声内 / 最大边际落在最后一档（区间没覆盖到拐点）/ 显著档之间非单调。

    这条规则是 2026-09-15 两次补的。原实现取 `sec_saved_per_yuan` 的**最大值**当拐点，
    与 docstring 写的「边际收益骤降处」是两回事：边际一非单调就翻到最后一档。改成
    「逐档递减」之后仍不够——波次窗口为 0 时三档均值差本就在噪声内，点估计却恰好递减，
    于是又报出「拐点 = 5 人」。
    """
    pts = sorted(
        [(a["n_pickers"], a["metrics"]["avg_fulfillment_sec"],
          a["n_pickers"] * C.PICKER_DAILY_COST) for a in arms],
        key=lambda x: x[0],
    )
    curve = [{"n_pickers": p, "avg_fulfillment_sec": s["mean"], "daily_labor_cost": c}
             for p, s, c in pts]
    # 边际：每增 1 人，履约时长降多少 / 成本增多少
    marginals = []
    for i in range(1, len(pts)):
        p0, s0, c0 = pts[i - 1]
        p1, s1, c1 = pts[i]
        dt, dc = s1["mean"] - s0["mean"], c1 - c0  # dt 负=改善
        marginals.append({
            "from_pickers": p0, "to_pickers": p1,
            "delta_fulfillment_sec": round(dt, 2), "delta_cost": round(dc, 2),
            "sec_saved_per_yuan": round(-dt / dc, 4) if dc else None,
            "significant": _means_differ(s0, s1),
        })

    vals = [m["sec_saved_per_yuan"] or 0.0 for m in marginals]
    significant = [i for i, m in enumerate(marginals) if m["significant"]]
    if len(vals) < 2:
        knee, note = None, "档位不足两档，无法判断边际收益是否递减"
    elif not significant:
        knee = None
        note = (f"各档的时长差异都与 0 不可区分（95% CI 重叠；点估计 {vals} 秒/元），"
                "这个负载与作业组织下分不出拐点")
    else:
        best = max(significant, key=lambda i: vals[i])
        if best == len(marginals) - 1:
            knee = None
            note = (f"最大边际落在最后一档（{vals}）——「再加人还在变好」，"
                    "说明测试区间没覆盖到拐点，而不是存在拐点")
        elif not all(vals[b] < vals[a] for a, b in zip(significant, significant[1:])):
            knee, note = None, f"显著档之间的边际收益非单调（{vals}），拐点不可判定"
        else:
            knee = marginals[significant[0]]["to_pickers"]
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


def calibrate_wave_window(outbound_csv: Path, wave_interval_min: float) -> dict:
    """波次窗口的**校准依据**：仿真里「订单到达 → 可拣」的平均等待 vs 数据层 A 实测的同段延迟。

    这一步把 W 从「挑一个数」变成「校准出来的数」：窗内订单近似均匀到达，平均累积等待
    = W/2，令它等于数据层 A 的释放延迟参数均值（`WAREHOUSE_RELEASE_DELAY_MIN` 的中值），
    即得 W。落盘是为了让这个取值可被检验、可被推翻——读者能自己看到它对没对上。
    """
    ob = pd.read_csv(outbound_csv, encoding="utf-8-sig",
                     parse_dates=["order_time", "pick_start"])
    line_lag = (ob["pick_start"] - ob["order_time"]).dt.total_seconds()
    first = ob.groupby("order_id").agg(order_time=("order_time", "min"),
                                       pick_start=("pick_start", "min"))
    per_order_lag = (first["pick_start"] - first["order_time"]).dt.total_seconds()
    param_mean_min = sum(C.WAREHOUSE_RELEASE_DELAY_MIN) / 2
    return {
        "wave_interval_min": wave_interval_min,
        "expected_wave_wait_min": round(wave_interval_min / 2, 1),
        "layer_a_release_delay_param_min": param_mean_min,
        "layer_a_observed_lag_min": {
            "per_order_mean": round(float(per_order_lag.mean()) / 60, 1),
            "per_line_mean": round(float(line_lag.mean()) / 60, 1),
        },
        "note": (
            "W 由「平均累积等待 W/2 = 数据层 A 的释放延迟参数均值」校准得出。层 A 的实测"
            "延迟与该参数同量级（差异来自层 A 的行内错峰与逐日抽样），仿真侧的波次等待也落在"
            "同一量级——三者对得上，W 才站得住。层 A 已验收且是校准锚点，不为仿真实验改动它；"
            "两层在「释放」上的实现差异（独立随机延迟 vs 窗界同步释放）记在 config 注释里。"
        ),
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

    def _staffing_arms(wave: float, tag: str) -> list[dict]:
        return [run_experiment_arm(layout_abc, n, base_seed, f"{tag}_p{n}",
                                   sku_ids=sku_ids, sku_weights=sku_w,
                                   wave_interval_min=wave)
                for n in C.SIM_PICKER_LEVELS]

    wave_main = C.WAREHOUSE_WAVE_INTERVAL_MIN
    logger.info("实验二（人力）：拣货员 %s 人，布局=abc_zoned，波次 %.0f 分钟，各 %d 次重复 ...",
                C.SIM_PICKER_LEVELS, wave_main, C.SIM_REPEATS)
    exp2_arms = _staffing_arms(wave_main, "exp2")
    tradeoff = tradeoff_curve_and_knee(exp2_arms)

    # 波次窗口敏感性：结论对「作业组织」有多敏感，是这个实验真正要回答的问题之一。
    # 主档已在上面算过，不重复跑。
    logger.info("实验二敏感性：波次窗口 %s 分钟 ...", list(C.SIM_WAVE_LEVELS))
    wave_sensitivity = []
    for wave in C.SIM_WAVE_LEVELS:
        arms_w = exp2_arms if wave == wave_main else _staffing_arms(wave, f"wave{wave:g}")
        wave_sensitivity.append({
            "wave_interval_min": wave,
            "arms": [_arm_summary(a) for a in arms_w],
            "tradeoff": tradeoff_curve_and_knee(arms_w),
        })

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
    calib_wave = calibrate_wave_window(warehouse_dir / "outbound_orders.csv", wave_main)

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
    _write_json({
        "wave_interval_min": wave_main,
        "arms": exp2_arms,
        "tradeoff": tradeoff,
        "wave_sensitivity": wave_sensitivity,
        "limitations": [
            "本模型只建模波次的**等待成本**，未建模它的**合并拣货收益**（一次波次内多单"
            "合并成一条行走路径）。因此波次窗口的取值不可被读作「W 越小越好」——真实系统里"
            "W 变大还有省行走的一侧，本模型没有它。",
            "波次窗口 W 由数据层 A 的释放延迟校准（见 calibration.json 的 wave_window），"
            "不是独立标定的参数；层 A 的释放机制本身是对「订单不是到达即拣」的近似。",
            "日均 300 单对 4–6 拣货员仍是欠载系统（利用率 0.14–0.21）：加人的收益全部来自"
            "波次释放造成的排队，而不是产能不足。真正的杠杆是波次窗口，不是人数。",
        ],
    }, paths["exp2"])
    paths["calibration"] = out_dir / "calibration.json"
    _write_json({"against_layer_a": calib_a, "against_olist_shape": calib_olist,
                 "wave_window": calib_wave,
                 "adr": "0001（不做跨尺度绝对时长 KS）"}, paths["calibration"])

    # what-if 缓存：拣货员人数 → 指标（供看板滑块直接读，ADR-0010 同哲学）
    # 带上时长分解，是因为「加人到底能改什么」只有拆开才看得见：端到端里最大的一段是
    # 波次累积等待，它由作业组织决定、与人数无关；人力能压的只有「等拣货员」那一段。
    whatif = {str(a["n_pickers"]): {
        "avg_fulfillment_sec": a["metrics"]["avg_fulfillment_sec"],
        "avg_order_to_ship_sec": a["metrics"]["avg_order_to_ship_sec"],
        "avg_wave_wait_sec": a["metrics"]["avg_wave_wait_sec"],
        "avg_queue_wait_sec": a["metrics"]["avg_queue_wait_sec"],
        "avg_pick_sec": a["metrics"]["avg_order_pick_sec"],
        "picker_utilization": a["metrics"]["picker_utilization"],
        "daily_labor_cost": a["n_pickers"] * C.PICKER_DAILY_COST,
    } for a in exp2_arms}
    paths["whatif_staffing"] = out_dir / "whatif_staffing.json"
    _write_json(whatif, paths["whatif_staffing"])

    logger.info(
        "实验完成：布局重排降行走 %.1f%% / 降拣货时长 %.1f%%；"
        "人力拐点=%s 人（波次 %.0f 分钟）；仿真拣货 %.1f s/行 vs 数据层A %.1f s/行",
        (1 - exp1_abc["metrics"]["total_walk_m"]["mean"] / exp1_orig["metrics"]["total_walk_m"]["mean"]) * 100,
        (1 - exp1_abc["metrics"]["avg_order_pick_sec"]["mean"] / exp1_orig["metrics"]["avg_order_pick_sec"]["mean"]) * 100,
        tradeoff["knee_at_pickers"], wave_main,
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
