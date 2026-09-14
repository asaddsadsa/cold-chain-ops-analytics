"""数据层 A：仓内运营仿真数据生成器（02 号票）。

生成连续 90 天的仿真区域仓五张表（SKU 主数据、库位主数据、入库单、出库单、
库存与盘点），固定具名种子可复现，表间外键与时间逻辑自洽，并主动埋入四类
问题供下游分析定量复现：

  ① A 类高频 SKU 初始被安排在远离拣货台的库位（频率排序与距离排序完全反转）；
  ② P03 品类盘点差异率显著高于其他品类（18% vs 4%）；
  ③ 每日 14:00–16:00 拣货单位行耗时系统性拉长（整段时长 ×1.75–2.05）；
  ④ 收货环节可控比例晚到（18% 超过预约+30 分钟容差）。

生成后输出数据质检摘要（JSON + Markdown + 日志）到 processed 目录。

运行方式（项目根）：python -m src.gen_warehouse_data
数据类别：过程仿真（见 data_sources_ledger.md 第 2 节）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from src import config as C

# ---------------------------------------------------------------------------
# 生成器内部参数（仅本数据层使用的情景假设，统一登记在此处 + 台账）
# ---------------------------------------------------------------------------
_ZIPF_S = 0.8  # 需求频率 Zipf 指数：w_i = 1/rank^s
_GRID_ROWS, _GRID_COLS = 20, 40  # 库位网格 20 行 × 40 列 = 800 库位
_ROW_SPACING_M, _COL_SPACING_M = 3.0, 1.5  # 行/列间距（米）
_WALK_SPEED = C.WALK_SPEED_M_PER_SEC  # 行走速度（米/秒），读 config
_P03_DISCREPANCY_P = 0.18  # 埋点②：P03 盘点差异概率
_OTHER_DISCREPANCY_P = 0.04  # 其他品类盘点差异概率
_LATE_ARRIVAL_P = 0.18  # 埋点④：入库晚到概率
_LATE_MIN_RANGE = (35, 240)  # 晚到偏移（分钟）：超过 30 分钟容差
_ONTIME_MIN_RANGE = (-10, 25)  # 非晚到偏移（分钟）：容差内
#: 与数据层 F 共用的仓内生成参数一律读 config（`WAREHOUSE_*` 那组），本模块不再自存一份——
#: 两层各写一份时，「与数据层 F 互证」就只是注释里的一句话，没有任何东西强制它成立。
_PEAK_HOURS = C.SIM_PEAK_HOURS  # 订单到达双高峰 (9-11, 14-16)，与数据层 F 同源
_ORDER_RANGE = C.DAILY_OUTBOUND_RANGE  # 日订单量 [200, 400]
_QTY_RANGE = (1, 6)  # 单行数量 1–5 件
_REPLENISH_FACTOR, _REPLENISH_SAFETY = 1.15, 6  # 补货系数与安全量
_INITIAL_STOCK_DAYS = 12  # 期初库存 ≈ 12 天期望需求
_STOCKTAKE_PER_DAY = 50  # 每日循环盘点 SKU 数
_RELEASE_DELAY_MIN = C.WAREHOUSE_RELEASE_DELAY_MIN  # 订单释放到开始拣货的延迟（分钟），与数据层 F 同源
_LINE_STAGGER_MIN = (2, 6)  # 同单内逐行错峰（分钟）

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 主数据
# ---------------------------------------------------------------------------
def make_sku_master(rng: np.random.Generator) -> pd.DataFrame:
    """生成 SKU 主数据：500 个 SKU、品类 P01–P10、价格/重量/体积、需求频率与 ABC 初始标签。

    ABC 初始标签按需求频率累计占比登记（A≤70%、B≤90%、其余 C）；
    正式分类结果以模块一（3.7）基于出库行数的计算为准（ADR-0002）。
    """
    n = C.SKU_COUNT
    sku_id = np.array([f"SKU{i:04d}" for i in range(1, n + 1)])
    category = rng.choice(np.array(C.CATEGORY_CODES), size=n)

    # 需求频率：随机秩 + Zipf 权重（决定出库抽样概率与 ABC 标签）
    rank = rng.permutation(n) + 1
    weight = 1.0 / rank.astype(float) ** _ZIPF_S

    # 价格/重量/体积：对数正态，量纲贴近城配电商件
    unit_price = np.round(rng.lognormal(mean=np.log(60), sigma=0.8, size=n), 2)
    weight_kg = np.round(rng.lognormal(mean=np.log(2.0), sigma=0.9, size=n), 2)
    volume_m3 = np.round(
        np.clip(weight_kg / 250.0 * rng.uniform(0.6, 1.6, size=n), 0.001, 0.2), 4
    )

    # ABC 初始标签：按频率降序累计占比切分（阈值读 config）
    order = np.argsort(-weight)
    cum_share = np.cumsum(weight[order]) / weight.sum()
    abc = np.empty(n, dtype=object)
    a_cut, b_cut = C.ABC_THRESHOLDS
    abc[order] = np.where(cum_share <= a_cut, "A", np.where(cum_share <= b_cut, "B", "C"))

    df = pd.DataFrame(
        {
            "sku_id": sku_id,
            "category": category,
            "unit_price": unit_price,
            "weight_kg": weight_kg,
            "volume_m3": volume_m3,
            "demand_weight": np.round(weight / weight.sum(), 6),  # 归一化需求频率
            "abc_initial_label": abc,
        }
    )
    return df


def make_location_master() -> pd.DataFrame:
    """生成库位主数据：20×40 网格，行走距离 = 行距×行 + 列距×列（拣货台在原点）。

    分区：按行块划 Z1（最近）… Z4（最远）。距离由坐标计算，非随机。
    """
    rows = np.repeat(np.arange(1, _GRID_ROWS + 1), _GRID_COLS)
    cols = np.tile(np.arange(1, _GRID_COLS + 1), _GRID_ROWS)
    dist = rows * _ROW_SPACING_M + cols * _COL_SPACING_M
    zone = np.where(rows <= 5, "Z1", np.where(rows <= 10, "Z2", np.where(rows <= 15, "Z3", "Z4")))
    loc_id = np.array([f"L{i:04d}" for i in range(1, len(rows) + 1)])
    return pd.DataFrame(
        {
            "loc_id": loc_id,
            "row": rows,
            "col": cols,
            "zone": zone,
            "walk_dist_m": np.round(dist, 2),
        }
    )


def assign_sku_locations(
    sku_df: pd.DataFrame, loc_df: pd.DataFrame
) -> np.ndarray:
    """埋点①：A 类高频 SKU 初始安排在远离拣货台的库位。

    实现：在全部 800 库位中按距离等间隔采样 500 个（覆盖近端到远端全距离范围，
    而非挤在最远端），再把 SKU 频率降序与库位距离降序完全反转配对——最高频 SKU
    拿最远库位、最低频 SKU 拿最近库位。这样 A 类集中在远端、C 类集中在近端，
    为模块一的库位重排制造明确的改善空间（重排 = 把 A/C 库位对调），
    同时库容利用率（500/800）有意义、近端库位被低频 SKU 占用也更贴近真实仓。

    返回每个 SKU 的库位下标数组（loc_index_of_sku）。
    """
    n_sku = len(sku_df)
    n_loc = len(loc_df)
    # 库位按行走距离升序，等间隔采样 n_sku 个，覆盖 [最近, 最远] 全范围
    near_first = np.argsort(loc_df["walk_dist_m"].to_numpy())
    sampled_pos = np.linspace(0, n_loc - 1, n_sku).astype(int)
    loc_pool_near_first = near_first[sampled_pos]  # 选中库位下标，距离升序
    # SKU 按需求频率降序；与库位距离降序（[::-1]）一一配对 = 完全反转
    sku_by_freq = np.argsort(-sku_df["demand_weight"].to_numpy())
    loc_index_of_sku = np.empty(n_sku, dtype=int)
    loc_index_of_sku[sku_by_freq] = loc_pool_near_first[::-1]
    return loc_index_of_sku


# ---------------------------------------------------------------------------
# 补货计划（入库单）
# ---------------------------------------------------------------------------
def plan_inbound(
    rng: np.random.Generator,
    sku_df: pd.DataFrame,
    start: pd.Timestamp,
    faker: Faker,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """生成入库单表与「每日到货量矩阵」。

    每个 SKU 固定补货周期（3–7 天），到货量 = 周期内期望需求 × 1.15 + 安全量；
    埋点④：18% 的入库单实际到货晚于预约 + 30 分钟容差（晚到 35–240 分钟），
    其余在预约 ±容差内（-10 ~ +25 分钟）。

    返回：(入库单 DataFrame, arrivals[90, n_sku] 每日到货件数, initial_stock[n_sku])
    """
    n_sku = len(sku_df)
    share = sku_df["demand_weight"].to_numpy()
    # 全仓日期望件数 ≈ 日均订单 × 期望行/单 × 期望件/行
    exp_units_per_day = np.mean(_ORDER_RANGE) * 1.65 * np.mean(_QTY_RANGE)
    exp_daily = share * exp_units_per_day  # 每 SKU 日期望需求
    initial_stock = np.round(exp_daily * _INITIAL_STOCK_DAYS).astype(int) + 10

    operators = [faker.name() for _ in range(12)]  # Faker 仅生成操作员姓名（无关结论）
    arrivals = np.zeros((C.SIM_DAYS, n_sku), dtype=int)
    rows = []
    seq_by_day: dict[int, int] = {}

    for i in range(n_sku):
        cycle = int(rng.integers(3, 8))
        first_day = int(rng.integers(0, cycle))
        for d in range(first_day, C.SIM_DAYS, cycle):
            qty = int(round(exp_daily[i] * cycle * _REPLENISH_FACTOR)) + _REPLENISH_SAFETY
            arrivals[d, i] += qty
            # 预约到货：当日 08:00–12:00
            exp_ts = start + pd.Timedelta(days=d) + pd.Timedelta(minutes=int(rng.uniform(480, 720)))
            # 埋点④：晚到概率 18%
            if rng.random() < _LATE_ARRIVAL_P:
                offset_min = float(rng.uniform(*_LATE_MIN_RANGE))
            else:
                offset_min = float(rng.uniform(*_ONTIME_MIN_RANGE))
            act_ts = exp_ts + pd.Timedelta(minutes=offset_min)
            loc_i = -1  # 库位在调用方回填（避免循环依赖），先占位
            seq = seq_by_day.get(d, 0) + 1
            seq_by_day[d] = seq
            ymd = (start + pd.Timedelta(days=d)).strftime("%Y%m%d")
            rows.append(
                {
                    "inbound_id": f"IN{ymd}{seq:04d}",
                    "sku_id": sku_df["sku_id"].iat[i],
                    "sku_index": i,
                    "qty": qty,
                    "expected_arrival": exp_ts.floor("s"),
                    "actual_arrival": act_ts.floor("s"),
                    "loc_index": loc_i,
                    "operator": operators[int(rng.integers(len(operators)))],
                    "day": d,
                }
            )
    df = pd.DataFrame(rows)
    return df, arrivals, initial_stock


# ---------------------------------------------------------------------------
# 出库与盘点（逐日台账模拟）
# ---------------------------------------------------------------------------
def _minute_intensity() -> np.ndarray:
    """订单到达 NHPP 强度（分钟粒度，08:00–18:00 共 600 分钟），9–11 / 14–16 双高峰。

    强度值读 `config.WAREHOUSE_PEAK_INTENSITY`——与数据层 F **同一个参数**。这两层曾经
    各写一份且差一个数量级（A=1.8、F=10.0），而 ADR-0001 写明「到达过程与 14–16 点低谷
    埋点相互印证」，互证要求两边真的是同一个到达过程。
    """
    minutes = np.arange(C.WAREHOUSE_OPEN_HOURS[0] * 60, C.WAREHOUSE_OPEN_HOURS[1] * 60)
    hours = minutes // 60
    intensity = np.ones(len(minutes), dtype=float)
    for h0, h1 in _PEAK_HOURS:
        intensity[(hours >= h0) & (hours < h1)] = C.WAREHOUSE_PEAK_INTENSITY
    return intensity / intensity.sum()


def simulate_outbound_and_stocktake(
    rng: np.random.Generator,
    faker: Faker,
    sku_df: pd.DataFrame,
    loc_index_of_sku: np.ndarray,
    loc_df: pd.DataFrame,
    start: pd.Timestamp,
    arrivals: np.ndarray,
    initial_stock: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """逐日模拟出库单与循环盘点，维护库存台账（期初 + 到货 − 出库）。

    - 出库行仅从有库存的 SKU 按需求频率加权抽样（缺货 SKU 从抽样掩码剔除）；
    - 埋点③：拣货开始时刻在 [14,16) 的行，整段拣货时长 ×1.75–2.05；
    - 埋点②：盘点差异概率 P03=18%、其他=4%，差异幅度 ±1–5 件（亏多盈少）；
    - 盘点账面数取台账日末值；盘点只观察不回写台账（登记为假设）。

    返回：(出库单 DataFrame, 盘点 DataFrame, 台账统计 dict)
    """
    n_sku = len(sku_df)
    cat = sku_df["category"].to_numpy()
    share = sku_df["demand_weight"].to_numpy()
    sku_ids = sku_df["sku_id"].to_numpy()
    walk_sec = loc_df["walk_dist_m"].to_numpy()[loc_index_of_sku] / _WALK_SPEED  # 每 SKU 行走秒数
    loc_ids = loc_df["loc_id"].to_numpy()[loc_index_of_sku]
    handle_mu = float(np.log(C.PICK_SECONDS_PER_LINE_MEAN) - C.WAREHOUSE_PICK_HANDLE_SIGMA**2 / 2)

    pickers = [faker.name() for _ in range(20)]  # Faker 仅生成拣货员姓名（无关结论）
    minute_p = _minute_intensity()
    minute_grid = np.arange(C.WAREHOUSE_OPEN_HOURS[0] * 60, C.WAREHOUSE_OPEN_HOURS[1] * 60)

    stock = initial_stock.copy()
    outbound_days: list[pd.DataFrame] = []
    stocktake_days: list[pd.DataFrame] = []
    stats = {"min_stock": int(stock.min()), "stockout_transitions": 0, "order_counts": []}

    for d in range(C.SIM_DAYS):
        day_start = start + pd.Timedelta(days=d)
        stock += arrivals[d]  # 当日到货先入账

        # --- 当日订单与行 ---
        n_orders = int(rng.integers(_ORDER_RANGE[0], _ORDER_RANGE[1] + 1))
        stats["order_counts"].append(n_orders)
        order_min = np.sort(rng.choice(minute_grid, size=n_orders, p=minute_p))
        lines_per_order = rng.choice([1, 2, 3], size=n_orders, p=C.WAREHOUSE_LINES_PER_ORDER_P)
        n_lines = int(lines_per_order.sum())
        order_idx = np.repeat(np.arange(n_orders), lines_per_order)
        line_starts = np.repeat(np.cumsum(lines_per_order) - lines_per_order, lines_per_order)
        line_no = np.arange(n_lines) - line_starts + 1

        # --- 逐行抽 SKU（库存掩码 + 频率加权），顺序扣减保证日内一致 ---
        sku_sel = np.empty(n_lines, dtype=int)
        qty_sel = np.empty(n_lines, dtype=int)
        prev_stock = stock.copy()
        for k in range(n_lines):
            # 库存逐行扣减，掩码每行重建（只从有库存的 SKU 抽样）
            p_row = np.where(stock > 0, share, 0.0)
            s = p_row.sum()
            if s <= 0:  # 理论不可达：补货计划保证有库存
                p_row = np.ones(n_sku) / n_sku
            else:
                p_row = p_row / s
            i = int(rng.choice(n_sku, p=p_row))
            q = int(rng.integers(*_QTY_RANGE))
            q = min(q, int(stock[i]))
            stock[i] -= q
            sku_sel[k] = i
            qty_sel[k] = q
        stats["stockout_transitions"] += int(((stock <= 0) & (prev_stock > 0)).sum())
        stats["min_stock"] = min(stats["min_stock"], int(stock.min()))

        # --- 时间链（向量化）：下单 → 拣货开始/完成 → 复核 → 出库 ---
        order_sec = order_min[order_idx] * 60 + rng.integers(0, 60, n_lines)
        pick_start_sec = (
            order_sec
            + rng.uniform(_RELEASE_DELAY_MIN[0] * 60, _RELEASE_DELAY_MIN[1] * 60, n_lines)
            + (line_no - 1) * rng.uniform(_LINE_STAGGER_MIN[0] * 60, _LINE_STAGGER_MIN[1] * 60, n_lines)
        )
        handle = rng.lognormal(handle_mu, C.WAREHOUSE_PICK_HANDLE_SIGMA, n_lines) * (1 + 0.1 * (qty_sel - 1))
        duration = walk_sec[sku_sel] + handle
        # 埋点③：拣货开始落在 [14,16) 的行整段放大
        pick_hour = (pick_start_sec // 3600).astype(int)
        h0, h1 = C.PICK_SLOW_HOURS
        slow_mask = (pick_hour >= h0) & (pick_hour < h1)
        duration = duration * np.where(
            slow_mask, rng.uniform(*C.WAREHOUSE_SLOWDOWN_RANGE, n_lines), 1.0
        )
        pick_end_sec = pick_start_sec + duration
        check_sec = pick_end_sec + rng.triangular(*C.PACK_TRIANGULAR, n_lines)
        ship_sec = check_sec + rng.uniform(300, 2700, n_lines)

        def _ts(sec_arr: np.ndarray) -> pd.Series:
            return pd.Series(
                pd.to_datetime(day_start) + pd.to_timedelta(np.round(sec_arr), "s")
            )

        ymd = day_start.strftime("%Y%m%d")
        order_no = np.array([f"SO{ymd}{j:04d}" for j in range(1, n_orders + 1)])
        outbound_days.append(
            pd.DataFrame(
                {
                    "order_id": order_no[order_idx],
                    "line_no": line_no,
                    "sku_id": sku_ids[sku_sel],
                    "qty": qty_sel,
                    "order_time": _ts(order_sec).values,
                    "pick_start": _ts(pick_start_sec).values,
                    "pick_end": _ts(pick_end_sec).values,
                    "check_time": _ts(check_sec).values,
                    "ship_time": _ts(ship_sec).values,
                    "loc_id": loc_ids[sku_sel],
                    "picker": [pickers[int(x)] for x in rng.integers(0, len(pickers), n_lines)],
                }
            )
        )

        # --- 当日循环盘点（50 个 SKU，账面 = 台账日末值）---
        idx = rng.choice(n_sku, _STOCKTAKE_PER_DAY, replace=False)
        book = stock[idx].copy()
        is_p03 = cat[idx] == C.HIGH_DIFF_CATEGORY
        p_diff = np.where(is_p03, _P03_DISCREPANCY_P, _OTHER_DISCREPANCY_P)
        has_diff = rng.random(_STOCKTAKE_PER_DAY) < p_diff
        sign = np.where(rng.random(_STOCKTAKE_PER_DAY) < 0.7, -1, 1)  # 盘亏为主
        mag = rng.integers(1, 6, _STOCKTAKE_PER_DAY)
        delta = np.where(has_diff, sign * mag, 0)
        delta = np.maximum(delta, -book)  # 实盘不得为负；账面 0 时差异自然消去
        physical = book + delta
        diff_type = np.where(delta > 0, "盘盈", np.where(delta < 0, "盘亏", "无差异"))
        stocktake_days.append(
            pd.DataFrame(
                {
                    "date": day_start.date(),
                    "sku_id": sku_ids[idx],
                    "loc_id": loc_ids[idx],
                    "book_qty": book,
                    "physical_qty": physical,
                    "diff_qty": delta,
                    "diff_type": diff_type,
                }
            )
        )

    return pd.concat(outbound_days, ignore_index=True), pd.concat(
        stocktake_days, ignore_index=True
    ), stats


# ---------------------------------------------------------------------------
# 质检摘要
# ---------------------------------------------------------------------------
def build_qc_summary(
    sku_df: pd.DataFrame,
    loc_df: pd.DataFrame,
    inbound_df: pd.DataFrame,
    outbound_df: pd.DataFrame,
    stocktake_df: pd.DataFrame,
    loc_index_of_sku: np.ndarray,
    stats: dict,
    start: pd.Timestamp,
) -> dict:
    """计算质检摘要：行数、时间范围、四埋点统计验证值、完整性检查。

    全部统计量由生成产物直接计算——埋点回归测试（tests/）读取本摘要断言。
    """
    dist_of_sku = loc_df["walk_dist_m"].to_numpy()[loc_index_of_sku]
    abc = sku_df["abc_initial_label"].to_numpy()

    # 埋点①：A 类 vs C 类平均库位距离
    mean_dist_a = float(dist_of_sku[abc == "A"].mean())
    mean_dist_c = float(dist_of_sku[abc == "C"].mean())

    # 埋点②：P03 vs 其他品类盘点差异率
    cat_map = dict(zip(sku_df["sku_id"], sku_df["category"]))
    st_cat = stocktake_df["sku_id"].map(cat_map)
    st_diff = stocktake_df["diff_type"] != "无差异"
    p03_rate = float(st_diff[st_cat == C.HIGH_DIFF_CATEGORY].mean())
    other_rate = float(st_diff[st_cat != C.HIGH_DIFF_CATEGORY].mean())

    # 埋点③：14–16 点 vs 其余时段拣货秒/行
    pick_sec = (outbound_df["pick_end"] - outbound_df["pick_start"]).dt.total_seconds()
    pick_hour = outbound_df["pick_start"].dt.hour
    h0, h1 = C.PICK_SLOW_HOURS
    slow = (pick_hour >= h0) & (pick_hour < h1)
    sec_slow = float(pick_sec[slow].mean())
    sec_other = float(pick_sec[~slow].mean())

    # 埋点④：收货晚到率（实际 > 预约 + 30 分钟容差）
    late = (inbound_df["actual_arrival"] - inbound_df["expected_arrival"]) > pd.Timedelta(minutes=30)
    late_rate = float(late.mean())

    # 完整性：外键、时间单调、日订单量范围
    fk_ok = bool(
        outbound_df["sku_id"].isin(set(sku_df["sku_id"])).all()
        and outbound_df["loc_id"].isin(set(loc_df["loc_id"])).all()
        and inbound_df["sku_id"].isin(set(sku_df["sku_id"])).all()
        and stocktake_df["sku_id"].isin(set(sku_df["sku_id"])).all()
    )
    time_ok = bool(
        (
            (outbound_df["pick_end"] > outbound_df["pick_start"])
            & (outbound_df["check_time"] > outbound_df["pick_end"])
            & (outbound_df["ship_time"] > outbound_df["check_time"])
        ).all()
    )
    daily_orders = outbound_df.groupby(outbound_df["order_time"].dt.date)["order_id"].nunique()

    end = start + pd.Timedelta(days=C.SIM_DAYS - 1)
    return {
        "tables": {
            "sku_master": len(sku_df),
            "location_master": len(loc_df),
            "inbound_receipts": len(inbound_df),
            "outbound_orders": len(outbound_df),
            "inventory_stocktake": len(stocktake_df),
        },
        "date_range": {"start": str(start.date()), "end": str(end.date()), "days": C.SIM_DAYS},
        "daily_orders": {
            "min": int(daily_orders.min()),
            "max": int(daily_orders.max()),
            "mean": round(float(daily_orders.mean()), 1),
            "in_range_200_400": bool(daily_orders.between(*_ORDER_RANGE).all()),
        },
        "embedding_checks": {
            "abc_distance": {
                "mean_dist_A_m": round(mean_dist_a, 1),
                "mean_dist_C_m": round(mean_dist_c, 1),
                "ratio_A_over_C": round(mean_dist_a / mean_dist_c, 2),
                "n_A": int((abc == "A").sum()),
                "n_C": int((abc == "C").sum()),
            },
            "p03_discrepancy": {
                "p03_rate": round(p03_rate, 4),
                "other_rate": round(other_rate, 4),
                "ratio": round(p03_rate / other_rate, 2) if other_rate > 0 else None,
                "n_p03_counts": int((st_cat == C.HIGH_DIFF_CATEGORY).sum()),
                "n_other_counts": int((st_cat != C.HIGH_DIFF_CATEGORY).sum()),
            },
            "pick_slowdown_14_16": {
                "sec_per_line_14_16": round(sec_slow, 1),
                "sec_per_line_other": round(sec_other, 1),
                "ratio": round(sec_slow / sec_other, 2),
                "n_slow_lines": int(slow.sum()),
            },
            "late_arrival": {
                "late_rate": round(late_rate, 4),
                "ontime_rate": round(1 - late_rate, 4),
                "n_inbound": len(inbound_df),
            },
        },
        "integrity": {
            "fk_violations": 0 if fk_ok else 1,
            "time_order_violations": 0 if time_ok else 1,
            "min_stock": stats["min_stock"],
            "stockout_transitions": stats["stockout_transitions"],
        },
        "assumptions": [
            "盘点差异只记录、不回写库存台账（账面数保持 期初+入库−出库 的推算值）",
            "每 SKU 固定主库位（一 SKU 一库位），入库/出库/盘点同库位",
            "拣货 14–16 点低谷作用于整段时长（行走+操作），系数 1.75–2.05",
        ],
    }


def _write_qc_outputs(qc: dict, qc_dir: Path) -> tuple[Path, Path, Path]:
    """质检摘要落盘：JSON + Markdown + 日志（data/processed/qc/）。"""
    qc_dir.mkdir(parents=True, exist_ok=True)
    json_path = qc_dir / "warehouse_qc.json"
    md_path = qc_dir / "warehouse_qc.md"
    log_path = qc_dir / "gen_warehouse_data.log"

    json_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")

    e = qc["embedding_checks"]
    md_lines = [
        "# 数据层 A 质检摘要（过程仿真）",
        "",
        f"- 数据窗口：{qc['date_range']['start']} ~ {qc['date_range']['end']}（{qc['date_range']['days']} 天）",
        f"- 表行数：{qc['tables']}",
        f"- 日订单量：min {qc['daily_orders']['min']} / max {qc['daily_orders']['max']} / mean {qc['daily_orders']['mean']}"
        f"（范围内：{qc['daily_orders']['in_range_200_400']}）",
        "",
        "## 埋点统计验证",
        "",
        f"1. **A 类远库位**：A 类平均距离 {e['abc_distance']['mean_dist_A_m']} m vs "
        f"C 类 {e['abc_distance']['mean_dist_C_m']} m，比值 {e['abc_distance']['ratio_A_over_C']}",
        f"2. **P03 高盘差**：P03 差异率 {e['p03_discrepancy']['p03_rate']:.2%} vs "
        f"其他 {e['p03_discrepancy']['other_rate']:.2%}，比值 {e['p03_discrepancy']['ratio']}",
        f"3. **14–16 点拣货低谷**：低谷 {e['pick_slowdown_14_16']['sec_per_line_14_16']} 秒/行 vs "
        f"其余 {e['pick_slowdown_14_16']['sec_per_line_other']} 秒/行，比值 {e['pick_slowdown_14_16']['ratio']}",
        f"4. **收货晚到**：晚到率 {e['late_arrival']['late_rate']:.2%}（n={e['late_arrival']['n_inbound']}）",
        "",
        "## 完整性",
        "",
        f"- 外键违例：{qc['integrity']['fk_violations']}；时间逻辑违例：{qc['integrity']['time_order_violations']}",
        f"- 台账最低库存：{qc['integrity']['min_stock']}；缺货转换次数：{qc['integrity']['stockout_transitions']}",
        "",
        "## 登记假设",
        "",
    ]
    md_lines += [f"- {a}" for a in qc["assumptions"]]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(qc, ensure_ascii=False) + "\n")
    return json_path, md_path, log_path


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def generate_warehouse(
    seed: int | None = None,
    out_dir: Path | None = None,
    qc_dir: Path | None = None,
) -> dict[str, Path]:
    """生成五张表 + 质检摘要，返回全部产物路径。

    seed 默认读 config.SEED_WAREHOUSE；out_dir/qc_dir 默认读 config 路径
    （测试可传 tmp_path 隔离）。同种子重跑产物字节级一致。
    """
    seed = C.SEED_WAREHOUSE if seed is None else seed
    out_dir = Path(out_dir) if out_dir is not None else C.WAREHOUSE_DIR
    qc_dir = Path(qc_dir) if qc_dir is not None else C.PROCESSED_DIR / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    faker = Faker("zh_CN")
    Faker.seed(seed)
    faker.seed_instance(seed)

    start = pd.Timestamp(C.SIM_START_DATE)
    logger.info("开始生成仓内仿真数据：seed=%s, 窗口=%s 起 %d 天", seed, start.date(), C.SIM_DAYS)

    sku_df = make_sku_master(rng)
    loc_df = make_location_master()
    loc_index_of_sku = assign_sku_locations(sku_df, loc_df)

    inbound_df, arrivals, initial_stock = plan_inbound(rng, sku_df, start, faker)
    # 回填入库单库位（= SKU 主库位）
    loc_ids = loc_df["loc_id"].to_numpy()
    inbound_df["loc_id"] = loc_ids[loc_index_of_sku[inbound_df["sku_index"].to_numpy()]]
    inbound_df = inbound_df.drop(columns=["sku_index", "loc_index", "day"])

    outbound_df, stocktake_df, stats = simulate_outbound_and_stocktake(
        rng, faker, sku_df, loc_index_of_sku, loc_df, start, arrivals, initial_stock
    )

    # 落盘六张表（列序固定，保证字节级可复现）
    sku_out = sku_df.drop(columns=["demand_weight"])  # 需求频率为内部参数，不随表交付
    # 库位分配要**交付**：原始布局的定义参数是 SKU 需求频率（上面刚被丢掉那个），
    # 数据层 F 的实验一要拿「原始布局臂」与本层的 outbound 对比（两条证据链互证），
    # 交付实际分配后它读到的就是同一份布局本身，而不是按 ABC 标签的近似复刻。
    assignment = pd.DataFrame(
        {
            "sku_id": sku_df["sku_id"].to_numpy(),
            "loc_id": loc_ids[loc_index_of_sku],
        }
    )
    paths = {
        "sku_master": out_dir / "sku_master.csv",
        "location_master": out_dir / "location_master.csv",
        "sku_location_assignment": C.WAREHOUSE_SKU_LOCATION_CSV.name,
        "inbound_receipts": out_dir / "inbound_receipts.csv",
        "outbound_orders": out_dir / "outbound_orders.csv",
        "inventory_stocktake": out_dir / "inventory_stocktake.csv",
    }
    sku_out.to_csv(paths["sku_master"], index=False)
    loc_df.to_csv(paths["location_master"], index=False)
    assignment.to_csv(out_dir / C.WAREHOUSE_SKU_LOCATION_CSV.name, index=False)
    inbound_df.to_csv(paths["inbound_receipts"], index=False)
    outbound_df.to_csv(paths["outbound_orders"], index=False)
    stocktake_df.to_csv(paths["inventory_stocktake"], index=False)

    qc = build_qc_summary(
        sku_df, loc_df, inbound_df, outbound_df, stocktake_df, loc_index_of_sku, stats, start
    )
    qc_json, qc_md, qc_log = _write_qc_outputs(qc, qc_dir)
    paths.update({"qc_json": qc_json, "qc_md": qc_md, "qc_log": qc_log})

    e = qc["embedding_checks"]
    logger.info(
        "生成完成：出库 %d 行 / 入库 %d 行 / 盘点 %d 行；埋点验证 A/C距离比=%.2f, "
        "P03盘差比=%.2f, 低谷比=%.2f, 晚到率=%.2f%%",
        len(outbound_df), len(inbound_df), len(stocktake_df),
        e["abc_distance"]["ratio_A_over_C"], e["p03_discrepancy"]["ratio"],
        e["pick_slowdown_14_16"]["ratio"], e["late_arrival"]["late_rate"] * 100,
    )
    return paths


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    paths = generate_warehouse()
    print("\n=== 数据层 A 产物 ===")
    for k, v in paths.items():
        print(f"  {k:22s} -> {v}")


if __name__ == "__main__":
    main()
