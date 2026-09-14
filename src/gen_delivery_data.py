"""数据层 D：配送情景三件套生成器（06 号票）。

一次运行产出成都冷链城配的完整配送情景：

  ① **配送订单**（`delivery_orders.csv`）：基于数据层 C 的真实 POI 坐标，生成
     90 天订单（每店每日泊松 1–3、固定种子），重量/体积按数据层 B 的 Olist 实测
     分布**形状**抽样、尺度按单批次冷链补货量标定（见 config 与本文件说明），
     时间窗落在 08:00–18:00（默认 2 小时），片区 R01–R08 空间聚簇（规则落盘）。
  ② **车辆与成本模型**（`vehicles.csv` + `tco_analysis.{json,md}`）：15 台 4.2 米
     冷链轻卡（柴油自购 / 纯电租赁双模式），输出双模式日总成本曲线、盈亏平衡里程、
     司机工资/能源价格/租金三档敏感性、自营单趟成本 vs 货拉拉对照表与模式建议。
  ③ **在途轨迹与异常**（`shipment_tracking.csv` + `shipment_anomalies.csv`）：
     逐运单轨迹采样（经纬度/时间/车速/车厢温度）与异常表，总体异常概率以数据层 B
     实测 Olist 真实延迟率标定；主动埋入四个异常靶点，供下游在途分析与周度复盘复现。

生成后输出质检摘要（JSON + Markdown）到 `data/processed/qc/`，定量验证四埋点成立。

**下游读取约定**：订单表的时间窗 `window_start` / `window_end` 是**当日分钟数**
（0 = 当日 00:00，便于 VRPTW 时间维直接做整数运算）；异常表的时间列则是**时间戳**。
里程/时长矩阵的行列索引为 `DC, P001…`，POI 表主键同名。

运行方式（项目根）：python -m src.gen_delivery_data
数据类别：情景假设（POI 坐标为真实观测，见 data_sources_ledger.md 第 2 节、ADR-0005）。
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 片区划分
# ---------------------------------------------------------------------------
def diagnose_unconstrained_clustering(
    poi_df: pd.DataFrame, k: int | None = None, min_size: int = 3
) -> dict:
    """实测一次**无约束空间聚簇**的片区规模，作为划分规则的排除依据。

    规则里「无约束聚簇会产生碎簇」这句话必须是可复现的测量结果而非事后叙述，
    故真跑 k-means++ 与 Ward 各一次，把**实测簇规模**写入 region_rule.json。
    `n_below_min_size` 是排除理由的量化形式：有多少个簇小到撑不起片区级异常率统计。
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.cluster.vq import kmeans2

    k = len(C.REGION_CODES) if k is None else k
    k = min(k, len(poi_df))  # 小样例表上 k 不得超过 POI 数（诊断是规则的量化依据，不是硬约束）
    x = poi_df[["lng", "lat"]].to_numpy(dtype=float)
    _, km_label = kmeans2(x, k, minit="++", seed=C.SEED_DELIVERY)
    # Ward 标签从 1 开始，bincount 后丢掉下标 0（空槽）再计簇规模
    ward_sizes = np.bincount(fcluster(linkage(x, "ward"), k, criterion="maxclust"), minlength=k + 1)[1:]
    km_sizes = np.bincount(km_label, minlength=k)
    sizes = {"kmeans_pp": sorted(int(v) for v in km_sizes), "ward": sorted(int(v) for v in ward_sizes)}
    return {
        "k": k,
        "cluster_sizes": sizes,
        "min_size_floor": min_size,
        "n_below_min_size": {name: int(sum(1 for v in s if v < min_size)) for name, s in sizes.items()},
    }


def assign_regions(poi_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """把 POI 划分为 R01–R08 片区，并返回可落盘的划分规则。

    划分规则（空间聚簇）：以**行政区**为空间单元聚合 POI——同一行政区地理连续，
    必然落在同一片区，且天然给出 8 个片区（实测成都主城区 + 青白江恰为 8 区）。
    片区编号按片区质心**经度升序**（由西向东），保证同一份 POI 表得到同一套编号。

    实测排除的替代方案：无约束空间聚簇（scipy k-means++ 与 Ward 层次聚类，k=8）
    均产生 1–3 个 POI 的碎簇并跨越行政区边界（49 个 POI 的真实空间结构是
    「青白江 1 个大簇 + 主城区 7 个中簇」，无约束聚簇会把青白江劈成两半同时合并
    两个主城区），片区规模不足以支撑片区级异常率统计，故不采用。该实验结论随
    规则一并落盘（region_rule.json），供复现者复核。

    返回 (带 region 列的 POI 表, 规则字典)。
    """
    df = poi_df.copy()
    agg = df.groupby("district").agg(
        n_poi=("poi_id", "size"), centroid_lng=("lng", "mean"), centroid_lat=("lat", "mean")
    )
    # 编号：质心经度升序（同经度再按纬度），保证确定性
    agg = agg.sort_values(["centroid_lng", "centroid_lat"], kind="stable")
    if len(agg) > len(C.REGION_CODES):
        raise ValueError(
            f"POI 覆盖 {len(agg)} 个行政区，超出片区码表容量 {len(C.REGION_CODES)}；"
            f"请扩充 config.REGION_CODES 或收敛数据层 C 的抓取范围"
        )
    codes = C.REGION_CODES[: len(agg)]
    code_of_district = dict(zip(agg.index, codes))
    df["region"] = df["district"].map(code_of_district)

    regions = {
        code: {
            "district": district,
            "n_poi": int(row.n_poi),
            "centroid_lng": round(float(row.centroid_lng), 6),
            "centroid_lat": round(float(row.centroid_lat), 6),
        }
        for district, code, row in (
            (d, c, agg.loc[d]) for d, c in code_of_district.items()
        )
    }
    rule = {
        "rule_version": "1.0",
        "data_category": "情景假设",
        "method": {
            "unit": "行政区（空间连续单元）",
            "label_rule": "质心经度升序编号（同经度按纬度）",
            "clustering_algorithm": "按行政区聚合（等价于带连续性约束的空间聚簇）",
            "rejected_alternative": (
                "无约束空间聚簇（k-means++ / Ward，k=8）：实测产生过小的碎簇并跨越行政区，"
                "片区规模撑不起片区级异常率统计，弃用。实测簇规模见同目录 "
                "unconstrained_clustering_diagnosis"
            ),
        },
        "unconstrained_clustering_diagnosis": diagnose_unconstrained_clustering(poi_df),
        "n_regions": len(agg),
        "n_poi": int(len(df)),
        "regions": regions,
        "poi_to_region": dict(zip(df["poi_id"], df["region"])),
    }
    return df, rule


# ---------------------------------------------------------------------------
# 需求分布：形状取自数据层 B 的 Olist 实测，尺度按情景锚点标定
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _olist_shape_params() -> dict:
    """读取数据层 B 产出的 Olist 经验分布（重量/体积对数正态 σ）。

    06 号票**只读不重算**：形状参数必须来自已落盘的实测产物，
    文件缺失即报错并给出手动生成指引，绝不凭记忆填数。
    """
    path = C.OLIST_WEIGHT_VOLUME_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 Olist 经验分布产物：{path}\n"
            f"请先运行数据层 B：python -m src.olist_clean -- 生成 "
            f"{path.name}（含 product_weight_g / product_volume_cm3 的 lognorm_shape_sigma）"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "weight_sigma": float(data["product_weight_g"]["lognorm_shape_sigma"]),
        "volume_sigma": float(data["product_volume_cm3"]["lognorm_shape_sigma"]),
        "source_n": {
            "weight": int(data["product_weight_g"]["n"]),
            "volume": int(data["product_volume_cm3"]["n"]),
        },
    }


def weight_shape_sigma() -> float:
    """订单重量分布形状（对数正态 σ），取自 Olist 商品重量实测。"""
    return _olist_shape_params()["weight_sigma"]


def volume_shape_sigma() -> float:
    """订单体积分布形状（对数正态 σ），取自 Olist 商品体积实测。"""
    return _olist_shape_params()["volume_sigma"]


def sample_store_lambdas(rng: np.random.Generator, n_stores: int) -> np.ndarray:
    """为每个门店抽取稳定的日订单泊松均值 λ ~ U(1,3)（读 config）。

    λ 按门店固定（而非逐日重抽），使各门店有稳定的需求强度画像，
    下游「片区订单量/车辆利用率」才有可解释的差异。
    """
    lo, hi = C.ORDER_POISSON_MEAN_RANGE
    return rng.uniform(lo, hi, size=n_stores)


def sample_daily_counts(rng: np.random.Generator, lambdas: np.ndarray) -> np.ndarray:
    """给定各门店 λ，抽取该日各门店订单数 ~ Poisson(λ)。"""
    return rng.poisson(lambdas)


def sample_demand(
    rng: np.random.Generator, n: int, median: float, sigma: float
) -> np.ndarray:
    """从「指定中位数的对数正态」抽样。

    对数正态的 exp(mu) 即中位数，故 mu = ln(median)；sigma 为形状参数
    （由调用方从 Olist 实测产物传入）。用中位数而非均值作锚点，是因为
    长尾分布下中位数对尺度标定更稳健、也更贴近「典型订单」的业务表述。
    """
    return median * np.exp(sigma * rng.standard_normal(n))


def make_receiving_waves(rng: np.random.Generator, n_stores: int) -> np.ndarray:
    """为每家门店抽取当日**收货波次起点**（当日分钟数，读 config）。

    一家店一天只有一波冷链收货窗口（真实的便利店 / 超市收货习惯），该店当日所有
    订单的送达窗都锚在这一波上。这是让 ADR-0004「单点一访 + 逐单判定时间窗达成率」
    自洽的前提：若同店各单的窗各自独立散布在整天的 08:00–18:00，一次到访不可能同时
    满足它们，那个 KPI 就没有意义了（10 号票评审实测逐单达成率被压到 35%）。
    """
    lo, hi = C.RECEIVING_WAVE_START_RANGE
    latest = int(C.TIME_WINDOW_OPEN[1] * 60) - int(C.DEFAULT_TIME_WINDOW_HOURS * 60) \
        - C.RECEIVING_WAVE_JITTER_MIN
    return rng.integers(lo, min(hi, latest) + 1, size=n_stores)


def make_time_windows(
    rng: np.random.Generator, wave_start: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """按各订单所属门店的**收货波次**生成送达时间窗（当日分钟数）。

    窗起 = 波次起点 + 抖动（0–RECEIVING_WAVE_JITTER_MIN 分钟），窗宽取默认小时数。
    抖动保留「同店各单的窗不完全相同」的真实感，也让「逐单判定」仍然是有效判定
    （到访必须落在各单自己的窗内，而不是落在一个人为放大的信封窗里）。
    """
    wave_start = np.asarray(wave_start, dtype=float)
    duration = C.DEFAULT_TIME_WINDOW_HOURS * 60
    start = wave_start + rng.integers(0, C.RECEIVING_WAVE_JITTER_MIN + 1, size=len(wave_start))
    return start.astype(int), (start + duration).astype(int)


def generate_orders(
    poi_df: pd.DataFrame, days: int | None = None, seed: int | None = None
) -> pd.DataFrame:
    """生成 90 天配送订单表（需求 19）。

    逐日按「各门店 λ → Poisson 订单数 → 逐单抽重量/体积/时间窗/优先级」生成；
    坐标直接取自 POI 表（真实观测），不做扰动。同种子完全可复现。
    """
    days = C.SIM_DAYS if days is None else days
    rng = np.random.default_rng(C.SEED_DELIVERY if seed is None else seed)
    lambdas = sample_store_lambdas(rng, len(poi_df))
    w_sigma, v_sigma = weight_shape_sigma(), volume_shape_sigma()
    pri_labels = [p for p, _ in C.ORDER_PRIORITY_WEIGHTS]
    pri_probs = [q for _, q in C.ORDER_PRIORITY_WEIGHTS]

    poi_ids = poi_df["poi_id"].to_numpy()
    frames = []
    start_date = pd.Timestamp(C.SIM_START_DATE)
    for d in range(days):
        counts = sample_daily_counts(rng, lambdas)
        n = int(counts.sum())
        if n == 0:
            continue
        day = start_date + pd.Timedelta(days=d)
        idx = np.repeat(np.arange(len(poi_df)), counts)  # 每单落在哪个门店
        # 门店当日收货波次 → 该店每张单的窗锚在波次上（同店各单窗口因此高度重叠）
        waves = make_receiving_waves(rng, len(poi_df))
        w_start, w_end = make_time_windows(rng, waves[idx])
        frame = pd.DataFrame(
            {
                "date": day,
                "poi_id": poi_ids[idx],
                "region": poi_df["region"].to_numpy()[idx],
                "lng": poi_df["lng"].to_numpy()[idx],
                "lat": poi_df["lat"].to_numpy()[idx],
                "weight_kg": np.round(
                    sample_demand(rng, n, C.ORDER_WEIGHT_MEDIAN_KG, w_sigma), 2
                ),
                "volume_m3": np.round(
                    sample_demand(rng, n, C.ORDER_VOLUME_MEDIAN_M3, v_sigma), 4
                ),
                "window_start": w_start,
                "window_end": w_end,
                "priority": rng.choice(pri_labels, size=n, p=pri_probs),
            }
        )
        # 当日内的稳定顺序：先按门店、再按时间窗起点
        frame = frame.sort_values(["poi_id", "window_start"], kind="stable").reset_index(drop=True)
        frame.insert(
            0, "order_id",
            [f"DO{day:%Y%m%d}{i:04d}" for i in range(len(frame))],
        )
        frames.append(frame)

    orders = pd.concat(frames, ignore_index=True)
    name_of = dict(zip(poi_df["poi_id"], poi_df["name"]))
    orders.insert(3, "poi_name", orders["poi_id"].map(name_of))
    return orders


# ---------------------------------------------------------------------------
# 车队与成本模型
# ---------------------------------------------------------------------------
#: 车牌字符集（现行民用号牌不含 I / O，避免与 1 / 0 混淆）
_PLATE_CHARS = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def _interleaved_modes() -> list[str]:
    """按 config 配比生成车辆动力序列，交错排布而非两种模式连号。"""
    if sum(C.FLEET_MODE_MIX.values()) != C.FLEET_SIZE:
        raise ValueError(
            f"FLEET_MODE_MIX 合计 {sum(C.FLEET_MODE_MIX.values())} ≠ FLEET_SIZE {C.FLEET_SIZE}"
        )
    remaining = dict(C.FLEET_MODE_MIX)
    modes: list[str] = []
    for _ in range(C.FLEET_SIZE):
        # 取「已用比例」最低的模式；并列时按模式名字典序取最大值，保证确定性
        mode = max(remaining, key=lambda m: (remaining[m] / C.FLEET_MODE_MIX[m], m))
        modes.append(mode)
        remaining[mode] -= 1
    return modes


def build_fleet(seed: int | None = None) -> pd.DataFrame:
    """生成 15 台 4.2 米冷链轻卡车辆表（需求 21）。

    车牌用派生种子生成（Faker 仅用于无关结论的文本，车牌同样不承载分析结论）；
    额定载重/容积与双模式成本参数一律读 config，禁止在本模块硬编码金额。
    """
    rng = np.random.default_rng((C.SEED_DELIVERY if seed is None else seed) + 1000)
    modes = _interleaved_modes()

    plates: list[str] = []
    while len(plates) < C.FLEET_SIZE:
        plate = "川A" + "".join(rng.choice(list(_PLATE_CHARS), size=5))
        if plate not in plates:
            plates.append(plate)

    rows = []
    for i, (vid, mode, plate) in enumerate(
        zip((f"V{i:02d}" for i in range(1, C.FLEET_SIZE + 1)), modes, plates)
    ):
        rows.append(
            {
                "vehicle_id": vid,
                "plate": plate,
                "mode": mode,
                "mode_label": "纯电租赁" if mode == "ev" else "柴油自购",
                "rated_payload_kg": C.RATED_PAYLOAD_T * 1000,
                "rated_volume_m3": C.RATED_VOLUME_M3,
                "fixed_cost_per_day": C.cost_fixed_per_day(mode),
                "cost_per_km": C.cost_per_km(mode),
                "alloc_weight": C.FLEET_ALLOC_WEIGHTS[i],
            }
        )
    return pd.DataFrame(rows)


def load_rate(weight_kg: float, rated_payload_kg: float) -> float:
    """满载率 = 实际载重 / 额定载重（需求 37 口径，按重量计）。"""
    return float(weight_kg) / float(rated_payload_kg)


def volume_rate(volume_m3: float, rated_volume_m3: float) -> float:
    """容积利用率 = 实际体积 / 额定容积（需求 37 的容积上限口径）。"""
    return float(volume_m3) / float(rated_volume_m3)


def exceeds_capacity(
    weight_kg: float, volume_m3: float, rated_payload_kg: float, rated_volume_m3: float
) -> bool:
    """载重或容积任一超限即判超载（VRPTW 两个容量约束的口径）。"""
    return bool(weight_kg > rated_payload_kg or volume_m3 > rated_volume_m3)


def huolala_cost(distance_km: float, n_stops: int) -> float:
    """货拉拉外包单趟报价（元）。

    计价规则读 config（起步价含前 5km → 6–BAND1_END 按元/km → 超出按区间中值 →
    超出免费点数按元/点），参数全部登记为情景假设（ADR-0009）。仅作决策对照，
    不进自营车队成本。
    """
    cost = C.HUOLALA_START_FEE
    if distance_km > C.HUOLALA_START_KM:
        band1_km = min(distance_km, C.HUOLALA_BAND1_END_KM) - C.HUOLALA_START_KM
        cost += band1_km * C.HUOLALA_RATE_6_25
    if distance_km > C.HUOLALA_BAND1_END_KM:
        cost += (distance_km - C.HUOLALA_BAND1_END_KM) * float(np.mean(C.HUOLALA_RATE_26_PLUS))
    cost += max(0, n_stops - C.HUOLALA_FREE_STOPS) * C.HUOLALA_EXTRA_STOP_FEE
    return round(float(cost), 2)


def _huolala_crossover_km(n_stops: int, mode: str = "diesel", hi: float = 400.0) -> float | None:
    """自营与货拉拉的单趟成本平价里程（km）；区间内无交点则返回 None。

    在同一行程画像（点位数固定）下扫里程网格找符号翻转，再二分细化。
    """
    f = lambda km: C.daily_total_cost(mode, km) - huolala_cost(km, n_stops)
    grid = np.linspace(0.0, hi, 401)
    vals = np.array([f(km) for km in grid])
    sign_change = np.where(np.diff(np.sign(vals)) != 0)[0]
    if len(sign_change) == 0:
        return None
    lo_b, hi_b = float(grid[sign_change[0]]), float(grid[sign_change[0] + 1])
    for _ in range(60):
        mid_b = (lo_b + hi_b) / 2
        if (f(lo_b) < 0) == (f(mid_b) < 0):
            lo_b = mid_b
        else:
            hi_b = mid_b
    return round((lo_b + hi_b) / 2, 1)


def build_tco_analysis(ref_km: float | None = None) -> dict:
    """构建双模式 TCO 成本模型全部结论（需求 21/22）。

    产出：① 双模式「日总成本 = 日固定 + 里程 × 变动」曲线；② 柴油自购 vs 纯电租赁
    的盈亏平衡里程；③ 司机工资 / 能源价格 / 租金三档 one-at-a-time 敏感性；
    ④ 自营（双模式）vs 货拉拉单趟成本对照表与模式建议。
    """
    ref_km = C.TCO_REFERENCE_DAILY_KM if ref_km is None else ref_km

    curves = {}
    for mode in ("diesel", "ev"):
        fixed = C.cost_fixed_per_day(mode)
        per_km = C.cost_per_km(mode)
        curves[mode] = {
            "daily_fixed": round(float(fixed), 2),
            "per_km": round(float(per_km), 4),
            "mileage_km": [float(k) for k in C.TCO_MILEAGE_GRID_KM],
            "daily_total_cost": [
                round(float(C.daily_total_cost(mode, k)), 2) for k in C.TCO_MILEAGE_GRID_KM
            ],
        }

    # 敏感性：每次只动一个参数，其余留 mid（one-at-a-time）
    sensitivity: dict[str, dict] = {}
    for param in C.SENSITIVITY_PARAMS:
        sensitivity[param] = {}
        for level in C.SENSITIVITY_LEVELS:
            kw = {p: "mid" for p in C.SENSITIVITY_PARAMS}
            kw[param] = level
            sensitivity[param][level] = {
                "diesel_daily_cost": round(float(C.daily_total_cost("diesel", ref_km, **kw)), 2),
                "ev_daily_cost": round(float(C.daily_total_cost("ev", ref_km, **kw)), 2),
                "breakeven_km": round(float(C.breakeven_km(**kw)), 2),
            }

    # 自营 vs 货拉拉对照：比较基准 = 单车单日 1 趟（日固定成本全额摊到该趟）
    comparison = []
    for prof in C.HUOLALA_REFERENCE_PROFILES:
        km, stops = float(prof["distance_km"]), int(prof["stops"])
        diesel = float(C.daily_total_cost("diesel", km))
        ev = float(C.daily_total_cost("ev", km))
        hl = huolala_cost(km, stops)
        best_self = min(diesel, ev)
        best_mode = "ev" if ev <= diesel else "diesel"
        tie_band = C.HUOLALA_TIE_BAND
        if best_self < hl * tie_band:
            verdict = "自营更省"
        elif hl < best_self * tie_band:
            verdict = "外包更省"
        else:
            verdict = "基本持平"
        comparison.append(
            {
                "profile": prof["profile"],
                "stops": stops,
                "distance_km": km,
                "self_diesel_cost": round(diesel, 2),
                "self_ev_cost": round(ev, 2),
                "self_best_mode": best_mode,
                "self_best_cost": round(best_self, 2),
                "huolala_cost": hl,
                "delta_pct_vs_huolala": round((best_self - hl) / hl * 100, 1),
                "verdict": verdict,
                "huolala_crossover_km": _huolala_crossover_km(stops, best_mode),
            }
        )

    breakeven = float(C.breakeven_km())
    daily_gap = float(C.daily_total_cost("diesel", ref_km) - C.daily_total_cost("ev", ref_km))
    recommendation = {
        "recommended_mode": "ev" if ref_km > breakeven else "diesel",
        "breakeven_km": round(breakeven, 2),
        "reference_daily_km": ref_km,
        "daily_saving_vs_diesel": round(daily_gap, 2),
        "rationale": (
            f"单车日均里程超过 {breakeven:.1f} km 时纯电租赁日总成本低于柴油自购"
            f"（纯电公里变动成本 {C.cost_per_km('ev'):.2f} 元/km 显著低于柴油 "
            f"{C.cost_per_km('diesel'):.2f} 元/km，代价是日固定成本高 "
            f"{C.cost_fixed_per_day('ev') - C.cost_fixed_per_day('diesel'):.0f} 元）。"
            f"代表里程 {ref_km:.0f} km 已越过盈亏平衡点，纯电每日省约 {daily_gap:.0f} 元；"
            f"反之短途轻载线路（≲ {breakeven:.0f} km）柴油更省。"
        ),
        "caveats": [
            "司机工资对两模式同额，相减抵消，不影响盈亏平衡里程；",
            "租金档位可令盈亏平衡里程为负（低租金时纯电在全里程段占优）；",
            "能源价格低/高两档为「油价档 × 能耗档」与「电价档」的独立取值组合，"
            "并非同一物理量的等比缩放，故盈亏平衡里程对能源档位不单调。",
            "对照表按单车单日 1 趟摊销日固定成本；一日多趟时自营成本进一步摊薄。",
        ],
        "comparison_basis": (
            "单车单日 1 趟、日固定成本全额摊入该趟；里程取**优化前**口径。"
            "模块二路径优化后单趟里程与成本都会下降，优化后的对照由 11 号票叠加。"
        ),
        "data_category": "情景假设（参数锚点见 config 与台账第 4 节，ADR-0009）",
    }

    return {
        "data_category": "情景假设",
        "source": "参数锚点=需求文档 2026-09 市场检索，登记于 data_sources_ledger.md 第 4 节（ADR-0009，URL 待补）",
        "reference_daily_km": ref_km,
        "curves": curves,
        "breakeven_km": {"km": round(breakeven, 2), "basis": "柴油自购 vs 纯电租赁 日总成本相等点"},
        "sensitivity": sensitivity,
        "huolala_comparison": comparison,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# 车辆装载率
# ---------------------------------------------------------------------------
def vehicle_daily_load(orders: pd.DataFrame, fleet: pd.DataFrame) -> pd.DataFrame:
    """按「车辆 × 日期」汇总当日承运量，给出满载率与容积利用率。

    口径与需求 37 一致：满载率 = 当日承运总重 / 额定载重；容积利用率同理。
    现状派车下一台车一日可承接多单，故按车日聚合而非按趟次——趟次级满载率
    由模块二的路由结果计算。
    """
    cols = ["vehicle_id", "rated_payload_kg", "rated_volume_m3"]
    merged = orders.merge(fleet[cols], on="vehicle_id", how="left")
    if merged["rated_payload_kg"].isna().any():
        unknown = set(merged.loc[merged["rated_payload_kg"].isna(), "vehicle_id"])
        raise ValueError(f"订单引用了车辆表中不存在的车辆：{sorted(unknown)}")
    daily = merged.groupby(["vehicle_id", "date"], as_index=False).agg(
        n_orders=("weight_kg", "size"),
        load_kg=("weight_kg", "sum"),
        volume_m3=("volume_m3", "sum"),
        rated_payload_kg=("rated_payload_kg", "first"),
        rated_volume_m3=("rated_volume_m3", "first"),
    )
    daily["load_rate"] = [
        load_rate(w, p) for w, p in zip(daily["load_kg"], daily["rated_payload_kg"])
    ]
    daily["volume_rate"] = [
        volume_rate(v, c) for v, c in zip(daily["volume_m3"], daily["rated_volume_m3"])
    ]
    return daily


def summarize_vehicle_load(daily: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """把车日装载率汇总到车辆级，并标记「长期低满载」（埋点④）。

    判定口径：车辆**平均**日满载率低于阈值即认定为长期低满载（长期状态，
    而非偶发单日低载），同时给出低于阈值的车日数作为稳定性佐证。
    """
    threshold = C.LOW_LOAD_VEHICLE_THRESHOLD if threshold is None else threshold
    g = daily.groupby("vehicle_id")["load_rate"]
    summary = pd.DataFrame(
        {
            "n_days": g.size(),
            "mean_load_rate": g.mean(),
            "median_load_rate": g.median(),
            "min_load_rate": g.min(),
            "max_load_rate": g.max(),
            "days_below_threshold": daily.assign(
                _below=daily["load_rate"] < threshold
            ).groupby("vehicle_id")["_below"].sum(),
        }
    ).reset_index()
    summary["is_low_load"] = summary["mean_load_rate"] < threshold
    return summary


# ---------------------------------------------------------------------------
# 在途异常与轨迹
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def olist_delay_rate() -> float:
    """数据层 B 实测的 Olist 真实延迟率——在途总体异常概率的标定基准。

    必须来自已落盘的实测产物；文件缺失即报错，绝不凭记忆填数（需求 23）。
    """
    path = C.OLIST_DELAY_RATE_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 Olist 延迟率产物：{path}\n"
            f"请先运行数据层 B：python -m src.olist_kpi -- 生成 {path.name}"
        )
    return float(json.loads(path.read_text(encoding="utf-8"))["delay_rate"])


def anomaly_weights(regions, is_friday) -> np.ndarray:
    """异常概率的**归一前权重**（片区倍率 × 周五倍率）。

    这是该权重的**唯一实现**——抽样（生成器）与单元测试都走这里。片区/周五倍率只做
    运单间的概率**重分配**，是否抬高全局水位由调用方决定（见 anomaly_probabilities）。
    """
    regions = np.asarray(regions)
    is_friday = np.asarray(is_friday, dtype=bool)
    weight = np.where(regions == C.HIGH_ANOMALY_REGION, C.ANOMALY_REGION_MULTIPLIER, 1.0)
    return weight * np.where(is_friday, C.ANOMALY_FRIDAY_OVERALL_MULTIPLIER, 1.0)


def anomaly_probabilities(regions, is_friday, base_rate: float) -> np.ndarray:
    """运单级**异常概率**（已归一）。

    按全样本权重均值归一，使**总体异常率严格等于 base_rate**（Olist 实测延迟率）：
    R07 与周五的高发是从其他运单身上重分配来的，不是凭空抬高的全局水位。
    调用方须传入**整批运单**的 region/is_friday，归一才成立。
    """
    weight = anomaly_weights(regions, is_friday)
    return weight * base_rate / weight.mean()


def anomaly_type_shares(is_friday: bool, is_rainy: bool) -> dict[str, float]:
    """给定「是否周五 / 是否雨日」时各异常类型的份额（合计 1）。

    周五与雨天抬高「晚点」份额、其余类型按比例摊薄——故周五/雨日的晚点率更高，
    而**异常构成之外的总量**不受影响（总量由 anomaly_probabilities 决定）。
    """
    shares = {t: s for t, s in C.ANOMALY_TYPE_SHARES}
    late_multiplier = 1.0
    if is_friday:
        late_multiplier *= C.ANOMALY_FRIDAY_LATE_MULTIPLIER
    if is_rainy:
        late_multiplier *= C.ANOMALY_RAIN_LATE_MULTIPLIER
    shares["晚点"] *= late_multiplier
    total = sum(shares.values())
    return {t: s / total for t, s in shares.items()}


def in_cold_chain_range(temp_c: float) -> bool:
    """车厢温度是否落在达标区间（闭区间，读 COLD_CHAIN_TEMP_RANGE）。"""
    lo, hi = C.COLD_CHAIN_TEMP_RANGE
    return bool(lo <= temp_c <= hi)


def temp_compliance_rate(temps) -> float:
    """温控达标率 = 达标采样数 / 总采样数（需求 37）。

    生产路径与单元测试共用本函数，避免「口径」出现第二份实现。
    """
    arr = np.asarray(temps, dtype=float)
    if arr.size == 0:
        return float("nan")
    lo, hi = C.COLD_CHAIN_TEMP_RANGE
    return float(np.mean((arr >= lo) & (arr <= hi)))


def generate_tracking_and_anomalies(
    orders: pd.DataFrame,
    fleet: pd.DataFrame,
    dist_df: pd.DataFrame,
    time_df: pd.DataFrame,
    dc_coord: tuple[float, float],
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成在途轨迹表与运单异常表（需求 23）。

    每张运单一条异常记录（含「无异常」），并作为该运单整车直送轨迹的骨架：
    计划发车 = 时间窗起点 − 行驶时长 − 缓冲（即常态准点），异常按类型分别作用于
    **出发时刻**（晚点/故障/温控波动）或**在途时长**（拥堵），到店卸货开门另计温升。

    轨迹按「DC → 该运单送达点」直送分段仿真，不依赖路线，避免与模块二循环依赖；
    直连里程/时长取自数据层 C 的真实路网矩阵。返回 (轨迹表, 异常表)。
    """
    rng = np.random.default_rng((C.SEED_DELIVERY if seed is None else seed) + 2000)
    base_rate = olist_delay_rate()
    df = orders.reset_index(drop=True)
    n = len(df)

    # ---- 日期维度：周五与雨天 ------------------------------------------------
    date = pd.to_datetime(df["date"])
    is_friday = (date.dt.weekday == 4).to_numpy()
    uniq_days = pd.Index(date.unique())
    day_rain_prob = np.array(
        [C.RAIN_PROB_BY_MONTH.get(pd.Timestamp(d).month, 0.0) for d in uniq_days]
    )
    rainy_of_day = dict(zip(uniq_days, rng.random(len(uniq_days)) < day_rain_prob))
    rainy = date.map(rainy_of_day).to_numpy()

    # ---- 异常判定：总体严格标定到 Olist 实测延迟率 ---------------------------
    region = df["region"].to_numpy()
    is_anom = rng.random(n) < anomaly_probabilities(region, is_friday, base_rate)

    type_names = [t for t, _ in C.ANOMALY_TYPE_SHARES]
    anomaly_type = np.full(n, "无异常", dtype=object)
    # 「周五 × 雨日」只有 4 种组合，逐组合抽样——份额模型只有 anomaly_type_shares 一份
    for fri in (False, True):
        for rain in (False, True):
            mask = (is_friday == fri) & (rainy == rain) & is_anom
            k = int(mask.sum())
            if not k:
                continue
            cum = np.cumsum([anomaly_type_shares(fri, rain)[t] for t in type_names])
            pick = (rng.random(k)[:, None] > cum).sum(axis=1)  # 分类抽样（逆变换）
            anomaly_type[mask] = np.array(type_names, dtype=object)[pick]

    delay_min = np.zeros(n)
    handling_min = np.zeros(n)
    for t in type_names:
        mask = anomaly_type == t
        k = int(mask.sum())
        if k:
            delay_min[mask] = rng.uniform(*C.ANOMALY_DELAY_MINUTES[t], size=k)
            handling_min[mask] = rng.uniform(*C.ANOMALY_HANDLING_MINUTES[t], size=k)

    # ---- 时刻推算 ------------------------------------------------------------
    travel_min = time_df.loc[df["poi_id"], "DC"].to_numpy(dtype=float)
    dc_dist_km = dist_df.loc[df["poi_id"], "DC"].to_numpy(dtype=float)
    window_start = df["window_start"].to_numpy(dtype=float)
    window_end = df["window_end"].to_numpy(dtype=float)
    # 出发类延误（晚点/故障/温控波动）推迟发车；途中类延误（拥堵）拉长在途时长
    transit_type = anomaly_type == "拥堵"
    depart_delay = np.where(transit_type, 0.0, delay_min)
    transit_delay = np.where(transit_type, delay_min, 0.0)

    planned_depart = window_start - travel_min - C.PLANNED_ARRIVAL_BUFFER_MIN
    actual_depart = planned_depart + depart_delay
    arrival = actual_depart + travel_min + transit_delay
    on_time = arrival <= window_end

    # ---- 车辆分派（现状派车：固定份额轮派，份额不均以复现埋点④）--------------
    vids = fleet["vehicle_id"].to_numpy()
    vehicle_id = vids[np.searchsorted(np.cumsum(C.FLEET_ALLOC_WEIGHTS), rng.random(n))]

    # ---- 轨迹采样 ------------------------------------------------------------
    dc_lng, dc_lat = float(dc_coord[0]), float(dc_coord[1])
    poi_lng = df["lng"].to_numpy(dtype=float)
    poi_lat = df["lat"].to_numpy(dtype=float)
    order_ids = df["order_id"].to_numpy()
    step = float(C.TRACKING_SAMPLE_MINUTES)

    seq_l, ord_l, veh_l, ts_l, lng_l, lat_l, spd_l, tmp_l = [], [], [], [], [], [], [], []
    t_min = np.empty(n)
    t_max = np.empty(n)
    t_rate = np.empty(n)
    for i in range(n):
        dur = max(arrival[i] - actual_depart[i], 1.0)
        offsets = np.append(np.arange(0.0, dur, step), dur)  # 末点严格落在到店时刻
        k = offsets.size
        frac = offsets / dur

        speed_mean = dc_dist_km[i] / (dur / 60.0)
        speed = np.clip(speed_mean * (1.0 + rng.normal(0.0, 0.15, k)), 0.0, None)

        temp = C.CABIN_TEMP_SETPOINT_C + rng.normal(0.0, C.CABIN_TEMP_NOISE_SIGMA_C, k)
        if anomaly_type[i] == "温控波动":
            temp += rng.uniform(*C.CABIN_TEMP_EXCURSION_RISE_C) * frac  # 全程逐步升温
        unload_start = max(0.0, dur - C.UNLOAD_MINUTES)
        door = np.clip((offsets - unload_start) / max(dur - unload_start, 1e-9), 0.0, 1.0)
        temp += rng.uniform(*C.CABIN_TEMP_DOOR_OPEN_RISE_C) * door  # 到点卸货开门温升

        seq_l.append(np.arange(k))
        ord_l.append(np.full(k, order_ids[i]))
        veh_l.append(np.full(k, vehicle_id[i]))
        ts_l.append(offsets + actual_depart[i])
        lng_l.append(dc_lng + (poi_lng[i] - dc_lng) * frac)
        lat_l.append(dc_lat + (poi_lat[i] - dc_lat) * frac)
        spd_l.append(speed)
        tmp_l.append(temp)
        t_min[i], t_max[i] = temp.min(), temp.max()
        t_rate[i] = temp_compliance_rate(temp)

    lengths = np.array([len(x) for x in ord_l])
    ts_min = np.concatenate(ts_l)
    day_repeat = np.repeat(date.to_numpy().astype("datetime64[ns]"), lengths)
    tracking = pd.DataFrame(
        {
            "order_id": np.concatenate(ord_l),
            "vehicle_id": np.concatenate(veh_l),
            "sample_seq": np.concatenate(seq_l),
            "ts_min": np.round(ts_min, 2),
            "timestamp": pd.to_datetime(day_repeat) + pd.to_timedelta(ts_min, unit="m"),
            "lng": np.round(np.concatenate(lng_l), 6),
            "lat": np.round(np.concatenate(lat_l), 6),
            "speed_kmh": np.round(np.concatenate(spd_l), 1),
            "cabin_temp_c": np.round(np.concatenate(tmp_l), 2),
        }
    )

    anomalies = pd.DataFrame(
        {
            "order_id": order_ids,
            "date": df["date"].to_numpy(),
            "weekday": date.dt.dayofweek.to_numpy(),
            "is_friday": is_friday,
            "rainy": rainy,
            "region": region,
            "poi_id": df["poi_id"].to_numpy(),
            "vehicle_id": vehicle_id,
            "anomaly_type": anomaly_type,
            "delay_min": np.round(delay_min, 1),
            "handling_min": np.round(handling_min, 1),
            "planned_departure": _minutes_to_ts(df["date"], planned_depart),
            "actual_departure": _minutes_to_ts(df["date"], actual_depart),
            "arrival_time": _minutes_to_ts(df["date"], arrival),
            "window_start": _minutes_to_ts(df["date"], window_start),
            "window_end": _minutes_to_ts(df["date"], window_end),
            "on_time": on_time,
            "temp_min_c": np.round(t_min, 2),
            "temp_max_c": np.round(t_max, 2),
            "temp_compliance_rate": np.round(t_rate, 4),
        }
    )
    return tracking, anomalies


def _minutes_to_ts(dates, minutes) -> pd.Series:
    """把「当日分钟数」还原为时间戳（跨 24:00 自动进位到次日）。"""
    base = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    return base + pd.to_timedelta(np.asarray(minutes, dtype=float), unit="m")


# ---------------------------------------------------------------------------
# 质检摘要
# ---------------------------------------------------------------------------
def two_proportion_z(n1: int, x1: int, n0: int, x0: int) -> float:
    """两比例 z 统计量（合并比例标准误）。

    用于判定埋点是否**统计显著**——埋点只有「建立起来」不够，还必须能被下游
    周度复盘以统计口径稳定复现，故质检摘要直接给出 z 值供断言。

    公开函数：11 号票的周度异常复盘复用同一实现，避免「同一套统计写两份」。
    """
    if n1 == 0 or n0 == 0:
        return float("nan")
    p1, p0 = x1 / n1, x0 / n0
    pooled = (x1 + x0) / (n1 + n0)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
    return float((p1 - p0) / se) if se > 0 else float("nan")


#: 5% 显著性水平下的双侧临界值（大样本正态近似）。
Z_CRITICAL_5PCT: float = 1.959964


def two_proportion_test(n1: int, x1: int, n0: int, x0: int) -> dict:
    """两比例检验的完整结论：两组比率、z、5% 显著性判定。

    供质检摘要与周度复盘共用，保证两处报的显著性是**同一个口径**。
    """
    z = two_proportion_z(n1, x1, n0, x0)
    return {
        "rate_1": round(x1 / n1, 4) if n1 else None,
        "rate_0": round(x0 / n0, 4) if n0 else None,
        "ratio": round((x1 / n1) / (x0 / n0), 2) if n0 and n1 and x0 else None,
        "n_1": int(n1),
        "n_0": int(n0),
        "z": None if math.isnan(z) else round(z, 2),
        "significant_at_5pct": None if math.isnan(z) else bool(abs(z) >= Z_CRITICAL_5PCT),
    }


def build_qc_summary(
    orders: pd.DataFrame,
    anomalies: pd.DataFrame,
    tracking: pd.DataFrame,
    load_daily: pd.DataFrame,
    load_summary: pd.DataFrame,
    fleet: pd.DataFrame,
    region_rule: dict,
) -> dict:
    """计算数据层 D 质检摘要：表规模、分布统计、四埋点验证值、完整性。

    全部统计量由生成产物直接计算——埋点回归测试（tests/）读取本摘要断言。
    """
    is_anom = anomalies["anomaly_type"] != "无异常"
    overall_rate = float(is_anom.mean())

    # 埋点①：片区异常率（HIGH_ANOMALY_REGION 显著高于其他片区）
    by_region = anomalies.assign(_a=is_anom).groupby("region")["_a"].mean()
    hi_region = C.HIGH_ANOMALY_REGION
    hi_mask = anomalies["region"] == hi_region
    hi_rate = float(by_region.get(hi_region, float("nan")))
    other_rate = float(by_region.drop(index=hi_region, errors="ignore").mean())
    region_z = two_proportion_z(
        int(hi_mask.sum()), int(is_anom[hi_mask].sum()),
        int((~hi_mask).sum()), int(is_anom[~hi_mask].sum()),
    )
    # 模型—数据一致性：落盘异常率必须忠实于 anomaly_probabilities 给出的期望。
    # 这是「抽样用的是不是同一份概率模型」的自证——若模型被复制成两份并走偏，这里立刻暴露。
    model_p = anomaly_probabilities(
        anomalies["region"].to_numpy(), anomalies["is_friday"].to_numpy(), olist_delay_rate()
    )
    model_by_region = pd.Series(model_p).groupby(anomalies["region"].to_numpy()).mean()
    model_by_region = model_by_region.reindex(by_region.index)
    n_by_region = anomalies.groupby("region").size().reindex(by_region.index)
    se_by_region = np.sqrt(model_by_region * (1 - model_by_region) / n_by_region)
    model_z_by_region = (by_region - model_by_region) / se_by_region

    # 埋点②③：周五 / 雨日的「晚点」率
    is_late = anomalies["anomaly_type"] == "晚点"
    fri_mask = anomalies["is_friday"]
    fri_late = float(is_late[fri_mask].mean())
    other_late = float(is_late[~fri_mask].mean())
    friday_z = two_proportion_z(
        int(fri_mask.sum()), int(is_late[fri_mask].sum()),
        int((~fri_mask).sum()), int(is_late[~fri_mask].sum()),
    )
    rain_mask = anomalies["rainy"]
    rain_late = float(is_late[rain_mask].mean())
    dry_late = float(is_late[~rain_mask].mean())
    rainy_z = two_proportion_z(
        int(rain_mask.sum()), int(is_late[rain_mask].sum()),
        int((~rain_mask).sum()), int(is_late[~rain_mask].sum()),
    )

    # 埋点④：部分车辆长期低满载
    low = load_summary[load_summary["is_low_load"]]
    high = load_summary[~load_summary["is_low_load"]]

    # 温控达标率（按采样点口径；复用需求 37 定义的 KPI 函数，不在此另写一份）
    temp_ok = temp_compliance_rate(tracking["cabin_temp_c"])

    # 完整性：外键、坐标范围、时间逻辑
    poi_ids = set(orders["poi_id"])
    fk_ok = bool(
        set(anomalies["poi_id"]).issubset(poi_ids)
        and set(anomalies["order_id"]) == set(orders["order_id"])
        and set(anomalies["vehicle_id"]).issubset(set(fleet["vehicle_id"]))
        and set(tracking["order_id"]).issubset(set(orders["order_id"]))
    )
    time_ok = bool(
        (anomalies["actual_departure"] >= anomalies["planned_departure"]).all()
        and (anomalies["arrival_time"] > anomalies["actual_departure"]).all()
        and (anomalies["window_end"] > anomalies["window_start"]).all()
    )
    mask_ok = bool(
        (anomalies["on_time"] == (anomalies["arrival_time"] <= anomalies["window_end"])).all()
    )
    coverage_ok = bool(
        (tracking.groupby("order_id")["sample_seq"].min() == 0).all()
        and (tracking.groupby("order_id")["sample_seq"].max() + 1 == tracking.groupby("order_id").size()).all()
    )
    coords_ok = bool(orders["lng"].between(103.0, 105.5).all() and orders["lat"].between(30.0, 31.5).all())

    start = pd.Timestamp(C.SIM_START_DATE)
    end = start + pd.Timedelta(days=int(orders["date"].nunique()) - 1)
    per_store_day = orders.groupby(["poi_id", "date"]).size()

    # 时间窗自洽性：「单点一访 + 逐单判定」要求同店当日各单的窗有公共交集
    def _has_common_intersection(g: pd.DataFrame) -> bool:
        return bool(g["window_start"].max() <= g["window_end"].min())

    multi = orders.groupby(["poi_id", "date"]).filter(lambda g: len(g) > 1)
    if len(multi):
        by_store_day = multi.groupby(["poi_id", "date"])
        n_ok = int(sum(_has_common_intersection(g) for _, g in by_store_day))
        n_multi = int(by_store_day.ngroups)
    else:
        n_ok, n_multi = 0, 0

    return {
        "tables": {
            "delivery_orders": len(orders),
            "shipment_anomalies": len(anomalies),
            "shipment_tracking": len(tracking),
            "vehicles": len(fleet),
            "vehicle_days": len(load_daily),
        },
        "date_range": {
            "start": str(start.date()),
            "end": str(end.date()),
            "days": int(orders["date"].nunique()),
        },
        "order_stats": {
            "per_day_mean": round(float(orders.groupby("date").size().mean()), 1),
            "per_store_day_mean": round(float(per_store_day.mean()), 3),
            "weight_kg": {
                "median": round(float(orders["weight_kg"].median()), 2),
                "mean": round(float(orders["weight_kg"].mean()), 2),
                "p95": round(float(orders["weight_kg"].quantile(0.95)), 2),
                "total_t_per_day": round(float(orders["weight_kg"].sum()) / orders["date"].nunique() / 1000, 2),
            },
            "volume_m3": {
                "median": round(float(orders["volume_m3"].median()), 4),
                "mean": round(float(orders["volume_m3"].mean()), 4),
            },
            "shape_source": {
                "weight_sigma": round(weight_shape_sigma(), 4),
                "volume_sigma": round(volume_shape_sigma(), 4),
                "source": str(C.OLIST_WEIGHT_VOLUME_JSON.relative_to(C.PROJECT_ROOT)),
            },
            "window": {
                "wave_start_range_min": list(C.RECEIVING_WAVE_START_RANGE),
                "jitter_max_min": C.RECEIVING_WAVE_JITTER_MIN,
                "n_multi_order_store_days": n_multi,
                "n_with_common_intersection": n_ok,
                "common_intersection_rate": round(n_ok / n_multi, 4) if n_multi else None,
                "note": (
                    "同店当日各单的窗锚在同一收货波次上，故公共交集普遍存在——"
                    "这是 ADR-0004「单点一访 + 逐单判定时间窗达成率」能同时成立的前提"
                ),
            },
        },
        "regions": {
            "n_regions": region_rule["n_regions"],
            "method": region_rule["method"]["unit"],
            "label_rule": region_rule["method"]["label_rule"],
            "poi_per_region": {k: v["n_poi"] for k, v in region_rule["regions"].items()},
            "order_share": {
                k: round(float(v), 4) for k, v in orders["region"].value_counts(normalize=True).items()
            },
        },
        "embedding_checks": {
            "region_anomaly_rate": {
                "high_region": hi_region,
                "high_region_rate": round(hi_rate, 4),
                "other_regions_rate": round(other_rate, 4),
                "ratio": round(hi_rate / other_rate, 2) if other_rate else None,
                "z": round(region_z, 2),
                "significant_at_5pct": bool(region_z > 1.96),
                "n_high_region": int(hi_mask.sum()),
                "per_region": {k: round(float(v), 4) for k, v in by_region.items()},
                "model_predicted_rate": {k: round(float(v), 4) for k, v in model_by_region.items()},
                "model_deviation_z": {k: round(float(v), 2) for k, v in model_z_by_region.items()},
                "max_abs_deviation_z": round(float(model_z_by_region.abs().max()), 2),
            },
            "friday_late_rate": {
                "friday": round(fri_late, 4),
                "other_weekdays": round(other_late, 4),
                "ratio": round(fri_late / other_late, 2) if other_late else None,
                "z": round(friday_z, 2),
                "significant_at_5pct": bool(friday_z > 1.96),
                "n_friday": int(fri_mask.sum()),
            },
            "rainy_late_rate": {
                "rainy": round(rain_late, 4),
                "dry": round(dry_late, 4),
                "ratio": round(rain_late / dry_late, 2) if dry_late else None,
                "z": round(rainy_z, 2),
                "significant_at_5pct": bool(rainy_z > 1.96),
                "n_rainy": int(rain_mask.sum()),
                "rainy_day_share": round(float(anomalies.groupby("date")["rainy"].first().mean()), 4),
            },
            "vehicle_low_load": {
                "threshold": C.LOW_LOAD_VEHICLE_THRESHOLD,
                "n_low_load": int(len(low)),
                "low_load_mean_rate": round(float(low["mean_load_rate"].mean()), 4) if len(low) else None,
                "other_mean_rate": round(float(high["mean_load_rate"].mean()), 4) if len(high) else None,
                "fleet_mean_rate": round(float(load_summary["mean_load_rate"].mean()), 4),
                "per_vehicle": [
                    {
                        "vehicle_id": r.vehicle_id,
                        "mean_load_rate": round(float(r.mean_load_rate), 4),
                        "is_low_load": bool(r.is_low_load),
                    }
                    for r in load_summary.itertuples()
                ],
            },
            "cold_chain": {
                "range_c": list(C.COLD_CHAIN_TEMP_RANGE),
                "compliance_rate": round(temp_ok, 4),
                "n_samples": int(len(tracking)),
                "n_out_of_range": int(round((1 - temp_ok) * len(tracking))),
                "per_shipment_mean": round(float(anomalies["temp_compliance_rate"].mean()), 4),
            },
        },
        "anomaly": {
            "overall_rate": round(overall_rate, 4),
            "olist_baseline_rate": round(olist_delay_rate(), 4),
            "baseline_source": str(C.OLIST_DELAY_RATE_JSON.relative_to(C.PROJECT_ROOT)),
            "by_type": {
                k: round(float(v), 4)
                for k, v in anomalies["anomaly_type"].value_counts(normalize=True).items()
            },
            "mean_handling_min_by_type": {
                k: round(float(v), 1)
                for k, v in anomalies.loc[is_anom].groupby("anomaly_type")["handling_min"].mean().items()
            },
            "on_time_rate": round(float(anomalies["on_time"].mean()), 4),
        },
        "integrity": {
            "fk_violations": 0 if fk_ok else 1,
            "time_order_violations": 0 if time_ok else 1,
            "ontime_flag_violations": 0 if mask_ok else 1,
            "tracking_coverage_violations": 0 if coverage_ok else 1,
            "coord_range_violations": 0 if coords_ok else 1,
        },
        "assumptions": [
            "整张配送订单表归类「情景假设」：POI 坐标为真实观测，需求量为泊松生成，"
            "重量/体积的分布**形状**（对数正态 σ）借自 Olist 实测、**尺度**按单批次冷链补货量标定（ADR-0005）",
            "时间窗锚定门店当日收货波次（config.RECEIVING_WAVE_*）：同店当日所有订单的送达窗"
            "落在同一波（08:30–15:00 之间取波次起点 + 0–10 分钟抖动），因此同店各单的窗有公共交集——"
            "这是 ADR-0004『单点一访 + 逐单判定时间窗达成率』能同时成立的前提（否则一次到访无法同时满足多单）",
            "重量/体积尺度锚点：单张订单中位 "
            f"{C.ORDER_WEIGHT_MEDIAN_KG:g} kg / {C.ORDER_VOLUME_MEDIAN_M3:g} m³"
            "（便利店/超市单批次冷链补货量），使 1.4t 载重约束在成都场景下非退化",
            "轨迹按「DC → 该运单送达点」直送分段仿真（不依赖路线，避免与模块二循环依赖）；"
            "坐标为两点间线性插值，仅用于速度/温度时序与温控达标率，**不得用于里程计算**",
            "「晚点」是异常表中的一个**异常类型标签**（含延误幅度），总体异常概率 = Olist 实测延迟率；"
            "模块二的时间窗达成率由路由计划到达时刻 + 本表 delay_min 判定，两者口径不同不可混用",
            "周五 / 雨天改变异常**构成**（抬高「晚点」份额），周五另叠加整体异常概率倍率"
            f"（×{C.ANOMALY_FRIDAY_OVERALL_MULTIPLIER}，午后拥堵与周末备货高峰）；"
            "片区倍率与周五倍率一起经全样本归一，故全局总体异常率仍严格等于 Olist 实测延迟率",
            f"雨天概率按月给定（{C.RAIN_PROB_BY_MONTH}），仅用于生成、不入 KPI 口径",
            "车厢温度越界有两个来源：到点卸货开门温升（日常主因）与「温控波动」异常（全程温升）",
            "现状派车按固定份额轮派（份额不均以复现埋点④），非真实调度结果；"
            "模块二将以其自身贪心基线与之对照",
        ],
    }


def _write_qc_outputs(qc: dict, qc_dir: Path) -> tuple[Path, Path]:
    """质检摘要落盘：JSON + Markdown（data/processed/qc/）。"""
    qc_dir.mkdir(parents=True, exist_ok=True)
    json_path = qc_dir / "delivery_qc.json"
    md_path = qc_dir / "delivery_qc.md"
    json_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")

    e = qc["embedding_checks"]
    md = [
        "# 数据层 D 质检摘要（配送情景）",
        "",
        f"- 数据窗口：{qc['date_range']['start']} ~ {qc['date_range']['end']}（{qc['date_range']['days']} 天）",
        f"- 表行数：{qc['tables']}",
        f"- 日均订单：{qc['order_stats']['per_day_mean']} 单；单店日均 "
        f"{qc['order_stats']['per_store_day_mean']} 单",
        f"- 时间窗锚定门店当日收货波次：{qc['order_stats']['window']['n_multi_order_store_days']} "
        f"个多单店日中 {qc['order_stats']['window']['n_with_common_intersection']} 个各单窗有公共交集"
        f"（{qc['order_stats']['window']['common_intersection_rate']:.1%}）",
        f"- 订单重量中位 {qc['order_stats']['weight_kg']['median']} kg / 体积中位 "
        f"{qc['order_stats']['volume_m3']['median']} m³；全网日运量 "
        f"{qc['order_stats']['weight_kg']['total_t_per_day']} t",
        f"- 片区划分：{qc['regions']['n_regions']} 个（{qc['regions']['method']}，"
        f"{qc['regions']['label_rule']}）",
        "",
        "## 埋点统计验证",
        "",
        f"1. **片区异常率**：{e['region_anomaly_rate']['high_region']} "
        f"{e['region_anomaly_rate']['high_region_rate']:.2%} vs 其他片区 "
        f"{e['region_anomaly_rate']['other_regions_rate']:.2%}，"
        f"比值 {e['region_anomaly_rate']['ratio']}",
        f"2. **周五晚点偏高**：周五 {e['friday_late_rate']['friday']:.2%} vs "
        f"其他日 {e['friday_late_rate']['other_weekdays']:.2%}，比值 {e['friday_late_rate']['ratio']}",
        f"3. **雨日晚点偏高**：雨日 {e['rainy_late_rate']['rainy']:.2%} vs "
        f"非雨日 {e['rainy_late_rate']['dry']:.2%}，比值 {e['rainy_late_rate']['ratio']}"
        f"（雨日占比 {e['rainy_late_rate']['rainy_day_share']:.1%}，情景假设）",
        f"4. **部分车辆长期低满载**：{e['vehicle_low_load']['n_low_load']} 台车平均日满载率低于 "
        f"{e['vehicle_low_load']['threshold']:.0%}，均值 "
        f"{e['vehicle_low_load']['low_load_mean_rate']:.1%} vs 其余 "
        f"{e['vehicle_low_load']['other_mean_rate']:.1%}（全队均值 "
        f"{e['vehicle_low_load']['fleet_mean_rate']:.1%}）",
        "",
        "## 在途异常与温控",
        "",
        f"- 总体异常率 {qc['anomaly']['overall_rate']:.2%}，标定基准（Olist 实测延迟率）"
        f"{qc['anomaly']['olist_baseline_rate']:.2%}",
        f"- 异常类型分布：{qc['anomaly']['by_type']}",
        f"- 平均异常处理时长：{qc['anomaly']['mean_handling_min_by_type']}",
        f"- 运单准点率（按本层计划时刻口径）：{qc['anomaly']['on_time_rate']:.2%}",
        f"- 温控达标率：{e['cold_chain']['compliance_rate']:.2%}"
        f"（{e['cold_chain']['n_out_of_range']}/{e['cold_chain']['n_samples']} 个采样越界，"
        f"达标区间 {e['cold_chain']['range_c']} ℃）",
        "",
        "## 完整性",
        "",
        f"- 外键违例 {qc['integrity']['fk_violations']}；时间逻辑违例 "
        f"{qc['integrity']['time_order_violations']}；准点标志违例 "
        f"{qc['integrity']['ontime_flag_violations']}；轨迹覆盖违例 "
        f"{qc['integrity']['tracking_coverage_violations']}；坐标越界 "
        f"{qc['integrity']['coord_range_violations']}",
        "",
        "## 登记假设",
        "",
    ]
    md += [f"- {a}" for a in qc["assumptions"]]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return json_path, md_path


def _write_tco_outputs(tco: dict, json_path: Path, md_path: Path) -> None:
    """TCO 分析落盘：JSON + Markdown 报告。"""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(tco, ensure_ascii=False, indent=2), encoding="utf-8")

    c, s, rec = tco["curves"], tco["sensitivity"], tco["recommendation"]
    md = [
        "# 车辆成本模型（双自营模式 + 外包对标）",
        "",
        f"- 数据类别：{tco['data_category']}；{tco['source']}",
        f"- 车队：{C.FLEET_SIZE} 台 4.2 米冷链轻卡，额定载重 {C.RATED_PAYLOAD_T} t / "
        f"容积 {C.RATED_VOLUME_M3} m³；动力配比 {C.FLEET_MODE_MIX}",
        "",
        "## 1. 双模式日总成本曲线",
        "",
        f"日总成本 = 日固定成本 + 日行驶里程 × 公里变动成本。代表里程 "
        f"{tco['reference_daily_km']:.0f} km。",
        "",
        "| 里程 (km) | " + " | ".join(f"{km:.0f}" for km in C.TCO_MILEAGE_GRID_KM) + " |",
        "|---|" + "---|" * len(C.TCO_MILEAGE_GRID_KM),
    ]
    md.append("| 柴油自购 (元) | " + " | ".join(f"{v:.0f}" for v in c["diesel"]["daily_total_cost"]) + " |")
    md.append("| 纯电租赁 (元) | " + " | ".join(f"{v:.0f}" for v in c["ev"]["daily_total_cost"]) + " |")
    md += [
        "",
        f"- 柴油：日固定 {c['diesel']['daily_fixed']:.0f} 元 + {c['diesel']['per_km']:.2f} 元/km",
        f"- 纯电：日固定 {c['ev']['daily_fixed']:.0f} 元 + {c['ev']['per_km']:.2f} 元/km",
        "",
        "## 2. 盈亏平衡里程",
        "",
        f"**{tco['breakeven_km']['km']:.1f} km/日**（{tco['breakeven_km']['basis']}）——"
        f"低于此里程柴油更省，高于此里程纯电更省。",
        "",
        "## 3. 三档敏感性（one-at-a-time，其余参数留中位）",
        "",
        "| 参数 | 档位 | 柴油日成本 | 纯电日成本 | 盈亏平衡里程 |",
        "|---|---|---|---|---|",
    ]
    for param in C.SENSITIVITY_PARAMS:
        for level in C.SENSITIVITY_LEVELS:
            r = s[param][level]
            md.append(
                f"| {param} | {level} | {r['diesel_daily_cost']:.0f} | "
                f"{r['ev_daily_cost']:.0f} | {r['breakeven_km']:.1f} |"
            )
    md += [
        "",
        "## 4. 自营 vs 货拉拉外包对照",
        "",
        f"比较基准：{rec['comparison_basis']}",
        "",
        "| 行程画像 | 点数 | 里程 | 自营柴油 | 自营纯电 | 货拉拉 | 最优自营 vs 货拉拉 | 结论 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in tco["huolala_comparison"]:
        md.append(
            f"| {r['profile']} | {r['stops']} | {r['distance_km']:.0f} km | "
            f"{r['self_diesel_cost']:.0f} | {r['self_ev_cost']:.0f} | {r['huolala_cost']:.0f} | "
            f"{r['delta_pct_vs_huolala']:+.1f}% | **{r['verdict']}** |"
        )
    md += [
        "",
        "## 5. 模式建议",
        "",
        f"**建议主力车型：{'纯电租赁' if rec['recommended_mode'] == 'ev' else '柴油自购'}**"
        f"（代表里程 {rec['reference_daily_km']:.0f} km/日）。",
        "",
        rec["rationale"],
        "",
        "### 注意事项",
        "",
    ]
    md += [f"- {x}" for x in rec["caveats"]]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------
def load_matrices(
    dist_csv: Path | None = None, time_csv: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取数据层 C 的真实路网矩阵（含 DC 行/列）。缺失时报错不降级。"""
    dist_csv = C.DIST_MATRIX_KM_CSV if dist_csv is None else Path(dist_csv)
    time_csv = C.TIME_MATRIX_MIN_CSV if time_csv is None else Path(time_csv)
    missing = [str(p) for p in (dist_csv, time_csv) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少数据层 C 路网矩阵：{missing}\n请先运行：python -m src.geo_poi 与 python -m src.geo_matrix"
        )
    return pd.read_csv(dist_csv, index_col=0), pd.read_csv(time_csv, index_col=0)


def load_poi_table(poi_csv: Path | None = None) -> pd.DataFrame:
    """读取数据层 C 的 POI 表。缺失时报错不降级。"""
    poi_csv = C.POI_CSV if poi_csv is None else Path(poi_csv)
    if not poi_csv.exists():
        raise FileNotFoundError(
            f"缺少 POI 表：{poi_csv}\n请先运行：python -m src.geo_poi"
        )
    return pd.read_csv(poi_csv)


def load_dc_coord(source_json: Path | None = None) -> tuple[float, float]:
    """读取数据层 C 落盘的 DC 坐标（matrix_source.json）。"""
    path = (C.GEO_DIR / "matrix_source.json") if source_json is None else Path(source_json)
    if not path.exists():
        raise FileNotFoundError(f"缺少矩阵来源文件：{path}（含 DC 坐标），请先运行 python -m src.geo_matrix")
    dc = json.loads(path.read_text(encoding="utf-8"))["dc"]
    return float(dc["lng"]), float(dc["lat"])


def generate_delivery_scenario(
    days: int | None = None,
    seed: int | None = None,
    out_dir: Path | None = None,
    qc_dir: Path | None = None,
) -> dict:
    """生成数据层 D 全部产物，返回各产物路径。"""
    out_dir = C.DELIVERY_DIR if out_dir is None else Path(out_dir)
    qc_dir = (C.PROCESSED_DIR / "qc") if qc_dir is None else Path(qc_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    poi = load_poi_table()
    regions, region_rule = assign_regions(poi)
    dist_df, time_df = load_matrices()
    dc_coord = load_dc_coord()
    fleet = build_fleet(seed=seed)

    orders = generate_orders(regions, days=days, seed=seed)
    tracking, anomalies = generate_tracking_and_anomalies(
        orders, fleet, dist_df, time_df, dc_coord, seed=seed
    )
    # 装载率按「订单量 × 现状派车分派」聚合：车辆分派来自异常表（同一批分派结果）
    assigned = orders.merge(anomalies[["order_id", "vehicle_id"]], on="order_id", how="left")
    load_daily = vehicle_daily_load(assigned, fleet)
    load_summary = summarize_vehicle_load(load_daily)
    tco = build_tco_analysis()
    qc = build_qc_summary(orders, anomalies, tracking, load_daily, load_summary, fleet, region_rule)

    paths = {
        "orders": out_dir / C.DELIVERY_ORDERS_CSV.name,
        "regions": out_dir / C.DELIVERY_REGIONS_CSV.name,
        "region_rule": out_dir / C.DELIVERY_REGION_RULE_JSON.name,
        "vehicles": out_dir / C.DELIVERY_VEHICLES_CSV.name,
        "tracking": out_dir / C.DELIVERY_TRACKING_CSV.name,
        "anomalies": out_dir / C.DELIVERY_ANOMALIES_CSV.name,
    }
    orders.to_csv(paths["orders"], index=False, encoding="utf-8-sig")
    regions.to_csv(paths["regions"], index=False, encoding="utf-8-sig")
    paths["region_rule"].write_text(
        json.dumps(region_rule, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fleet.to_csv(paths["vehicles"], index=False, encoding="utf-8-sig")
    tracking.to_csv(paths["tracking"], index=False, encoding="utf-8-sig")
    anomalies.to_csv(paths["anomalies"], index=False, encoding="utf-8-sig")
    load_daily.to_csv(out_dir / "vehicle_daily_load.csv", index=False, encoding="utf-8-sig")
    load_summary.to_csv(out_dir / "vehicle_load_summary.csv", index=False, encoding="utf-8-sig")
    _write_tco_outputs(tco, out_dir / C.DELIVERY_TCO_JSON.name, out_dir / C.DELIVERY_TCO_MD.name)
    qc_json, qc_md = _write_qc_outputs(qc, qc_dir)

    logger.info("配送订单 %d 行；异常 %d 行；轨迹 %d 行", len(orders), len(anomalies), len(tracking))
    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "qc_json": str(qc_json),
        "qc_md": str(qc_md),
        "tco_json": str(out_dir / C.DELIVERY_TCO_JSON.name),
        "tco_md": str(out_dir / C.DELIVERY_TCO_MD.name),
        "qc": qc,
        "tco": tco,
        "load_summary": load_summary,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = generate_delivery_scenario()
    qc = result["qc"]
    e = qc["embedding_checks"]
    print("\n=== 数据层 D 质检摘要 ===")
    print(f"  订单 {qc['tables']['delivery_orders']} 行 / 轨迹 {qc['tables']['shipment_tracking']} 行 / "
          f"车辆 {qc['tables']['vehicles']} 台，窗口 {qc['date_range']['start']}~{qc['date_range']['end']}")
    print(f"  ① 片区异常率：{e['region_anomaly_rate']['high_region']} "
          f"{e['region_anomaly_rate']['high_region_rate']:.2%} vs 其他 "
          f"{e['region_anomaly_rate']['other_regions_rate']:.2%}"
          f"（比值 {e['region_anomaly_rate']['ratio']}）")
    print(f"  ② 周五晚点率：{e['friday_late_rate']['friday']:.2%} vs "
          f"其他日 {e['friday_late_rate']['other_weekdays']:.2%}"
          f"（比值 {e['friday_late_rate']['ratio']}）")
    print(f"  ③ 雨日晚点率：{e['rainy_late_rate']['rainy']:.2%} vs "
          f"非雨日 {e['rainy_late_rate']['dry']:.2%}"
          f"（比值 {e['rainy_late_rate']['ratio']}）")
    print(f"  ④ 长期低满载车辆：{e['vehicle_low_load']['n_low_load']} 台，均值 "
          f"{e['vehicle_low_load']['low_load_mean_rate']:.1%} vs 其余 "
          f"{e['vehicle_low_load']['other_mean_rate']:.1%}")
    print(f"  总体异常率 {qc['anomaly']['overall_rate']:.2%}"
          f"（基准 {qc['anomaly']['olist_baseline_rate']:.2%}）；"
          f"温控达标率 {e['cold_chain']['compliance_rate']:.2%}")
    rec = result["tco"]["recommendation"]
    print(f"  TCO 建议：{'纯电租赁' if rec['recommended_mode'] == 'ev' else '柴油自购'}，"
          f"盈亏平衡里程 {rec['breakeven_km']:.1f} km/日")


if __name__ == "__main__":
    main()
