"""模块一（上）：仓内运营 KPI + ABC 分类 + 库位重排（08 号票）。

数据源为**数据层 A 的仿真仓五表**（`data/warehouse/`），全部 KPI 按严格口径计算、
函数可独立调用、可单测：

  ① 仓内六 KPI：库存准确率（金额加权）、收货及时率（预约 +30 分钟容差）、
     拣货效率（行/人时）、订单平均履约时效、库容利用率、盘点差异率；
     并分品类 / 分时段下钻，定量复现数据层 A 埋入的 **P03 高盘差**与 **14–16 点拣货低谷**。
  ② ABC 分类：按**仿真仓出库行数**累计占比划分（阈值读 config），输出帕累托与 A 类清单；
     分类与库位体系同源，直接驱动重排（ADR-0002）。
  ③ 库位优化：建「库位—拣货台」距离模型，以 Σ(出库频次 × 库位距离) 为拣货行走成本，
     按重排不等式把 A 类重排至最近库位，输出重排前后成本与预计效率变化，
     并与数据层 F 实验一（SimPy 仿真）互相印证。

口径回归由单元测试承担（每个 KPI 至少一个手工算例），不产出 Excel 校验工作簿——
原需求 3.x 的「Excel 公式复算对照」已按用户要求整体去掉，见 ADR-0012。

运行方式（项目根）：python -m src.warehouse_kpi
数据类别：过程仿真（见 data_sources_ledger.md 第 2/3 节）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 取数
# ---------------------------------------------------------------------------
def load_warehouse_tables() -> dict[str, pd.DataFrame]:
    """读取数据层 A 的五张表。缺失时报错并给生成指引，不做静默降级。"""
    missing = [str(p) for p in C.WAREHOUSE_TABLES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少数据层 A 仿真仓表：{missing}\n请先运行：python -m src.gen_warehouse_data"
        )
    return {
        name: pd.read_csv(path, encoding="utf-8-sig")
        for name, path in C.WAREHOUSE_TABLES.items()
    }


# ---------------------------------------------------------------------------
# 六 KPI（主缝：每个都是可独立调用的纯函数）
# ---------------------------------------------------------------------------
def _stocktake_amounts(stocktake: pd.DataFrame, sku: pd.DataFrame) -> pd.DataFrame:
    """给盘点记录补上单价与金额列（库存准确率的唯一计算口径）。"""
    price = dict(zip(sku["sku_id"], sku["unit_price"]))
    unit_price = stocktake["sku_id"].map(price)
    if unit_price.isna().any():
        raise ValueError("盘点记录引用了 SKU 主数据中不存在的 SKU")
    out = pd.DataFrame(
        {
            "sku_id": stocktake["sku_id"].to_numpy(),
            "book_qty": stocktake["book_qty"].to_numpy(),
            "diff_qty": stocktake["diff_qty"].to_numpy(),
            "unit_price": unit_price.to_numpy(dtype=float),
        }
    )
    out["abs_diff_value"] = out["diff_qty"].abs() * out["unit_price"]
    out["book_value"] = out["book_qty"].abs() * out["unit_price"]
    return out


def _order_fulfillment_hours(outbound: pd.DataFrame) -> pd.Series:
    """订单级履约时长（小时）= 该订单**最后一行发货** − **第一行下单**（唯一口径）。

    数据层 A 逐行独立抽样，同一订单各行的 order_time / ship_time 并不相同
    （实测 26865 个订单里 13263 个是多行单，其中 13261 个 ship_time 不唯一）。
    订单在**最后一行发出**时才算履约完成，故取 max(ship_time) − min(order_time)；
    早先取「任意一行」会低估时效（0.884 h vs 正确值 0.985 h）。
    """
    per_order = outbound.groupby("order_id", as_index=False).agg(
        order_time=("order_time", "min"), ship_time=("ship_time", "max")
    )
    return (
        pd.to_datetime(per_order["ship_time"]) - pd.to_datetime(per_order["order_time"])
    ).dt.total_seconds() / 3600.0


def inventory_accuracy(
    stocktake: pd.DataFrame, sku: pd.DataFrame, by: str | None = None
) -> dict | pd.DataFrame:
    """库存准确率 = 1 − Σ|账实差异| × 单价 / Σ 账面数量 × 单价（金额加权，需求 3.7）。

    用金额而非件数加权：一件高值商品的错发比一件低值商品严重得多，
    件数口径会低估错发的影响。返回 {rate, abs_diff_value, book_value, n_records}。

    `by="date"` 时改为按盘点日期分组的 DataFrame（逐日序列，供驾驶舱趋势线读）——
    分组后**每一天内部仍按金额加权**再聚合，不是把逐日件数率简单平均。
    """
    amounts = _stocktake_amounts(stocktake, sku)
    if by is None:
        abs_diff_value = float(amounts["abs_diff_value"].sum())
        book_value = float(amounts["book_value"].sum())
        rate = 1.0 - abs_diff_value / book_value if book_value else float("nan")
        return {
            "rate": rate,
            "abs_diff_value": round(abs_diff_value, 2),
            "book_value": round(book_value, 2),
            "n_records": int(len(stocktake)),
        }
    if by != "date":
        raise ValueError(f"不支持的库存准确率下钻维度：{by}（仅支持 'date'）")
    amounts = amounts.assign(date=pd.to_datetime(stocktake["date"]).dt.date)
    grouped = amounts.groupby("date").agg(
        abs_diff_value=("abs_diff_value", "sum"),
        book_value=("book_value", "sum"),
        n_records=("abs_diff_value", "size"),
    )
    # 与 `by=None` 分支同款防护：某天账面金额为 0 时该天准确率无定义，记 NaN 而不是 inf
    grouped["rate"] = np.where(
        grouped["book_value"] > 0,
        1.0 - grouped["abs_diff_value"] / grouped["book_value"].replace(0, np.nan),
        np.nan,
    )
    out = grouped.reset_index()
    out["date"] = out["date"].astype(str)
    return out[["date", "rate", "abs_diff_value", "book_value", "n_records"]]


def _receipt_on_time(
    inbound: pd.DataFrame, tolerance_min: float | None = None
) -> tuple[pd.Series, pd.Series, float]:
    """收货及时性的判定：→（是否及时, 预约到货时刻, 所用容差分钟）。

    **口径只允许这一个实现。** 聚合版 `receipt_timeliness` 取掩码均值；逐日版 `daily_kpi`
    按预约到货日分组取均值——两处若各写一遍 `act <= exp + tol`，口径漂移时不会有东西变红
    （而逐日版正是驾驶舱趋势线与环比箭头的读数据来源）。

    容差读 config（默认 30 分钟）——不加容差会把「提前/小幅迟到」误判为不及时。
    """
    tol = C.RECEIPT_TOLERANCE_MIN if tolerance_min is None else tolerance_min
    expected = pd.to_datetime(inbound["expected_arrival"])
    actual = pd.to_datetime(inbound["actual_arrival"])
    return actual <= expected + pd.Timedelta(minutes=tol), expected, float(tol)


def _pooled_rate(rates: pd.Series, weights: pd.Series) -> float:
    """按**自然单位**加权的合并率 = Σ(各组率 × 组内单位数) / Σ组内单位数。

    它等价于「Σ分子 / Σ分母」——例如盘差率 = 差异记录数合计 / 盘点记录数合计。

    与之相对的「对各组比率取等权平均」是**另一个统计量**，回答的是「平均而言一个组如何」。
    两者在组内单位数不等时不同，而头条口径要的是前者：单组（如 P03）的率本来就是合并算的，
    拿它去比各组等权的均值，两侧不是同一种量。
    """
    total = float(weights.sum())
    return float((rates * weights).sum() / total) if total else float("nan")


def _pick_seconds(outbound: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """每行拣货工时（秒）与开工时刻。

    逐行计时（同单内各行错峰开工），故按**行**累计工时而非按订单去重，否则多行订单的工时
    会被重复计入分母、效率被低估。聚合版 `picking_efficiency` 用它的和；逐日版按 `pick_start`
    所在日分组。
    """
    start = pd.to_datetime(outbound["pick_start"])
    end = pd.to_datetime(outbound["pick_end"])
    return (end - start).dt.total_seconds(), start


def receipt_timeliness(inbound: pd.DataFrame, tolerance_min: float | None = None) -> dict:
    """收货及时率 = 实际到货 ≤ 预约到货 + 容差 的记录占比（需求 3.7）。

    判定见 `_receipt_on_time`（与逐日版同源）。
    """
    on_time, _, tol = _receipt_on_time(inbound, tolerance_min)
    return {
        "rate": float(on_time.mean()),
        "n_records": int(len(inbound)),
        "n_late": int((~on_time).sum()),
        "tolerance_min": float(tol),
    }


def picking_efficiency(outbound: pd.DataFrame) -> dict:
    """拣货效率（行/人时）= 出库总行数 / Σ 每行拣货工时（需求 3.7）。

    工时口径见 `_pick_seconds`（与逐日版同源）。
    """
    seconds, _ = _pick_seconds(outbound)
    lines = int(len(outbound))
    hours = float(seconds.sum()) / 3600.0
    return {
        "lines_per_hour": lines / hours if hours else float("nan"),
        "n_lines": lines,
        "picker_hours": round(hours, 3),
        "sec_per_line": float(seconds.mean()),
    }


def avg_fulfillment_hours(outbound: pd.DataFrame) -> dict:
    """订单平均履约时效（小时）——口径见 `_order_fulfillment_hours`（需求 3.7）。

    必须按订单聚合后再求均值：直接对行求均值会把多行订单的权重放大数倍。
    """
    hours = _order_fulfillment_hours(outbound)
    return {
        "hours": float(hours.mean()),
        "n_orders": int(len(hours)),
        "median_hours": float(hours.median()),
    }


def sku_location_map(outbound: pd.DataFrame) -> pd.Series:
    """从出库单还原「SKU → 主库位」映射（一 SKU 一库位，与数据层 A 的分配规则一致）。

    sku_master 不落 loc_id（库位是分配结果而非主数据），故从出库行还原；
    若同一 SKU 出现在多个库位则说明前提被破坏，直接报错而不是悄悄取一个。
    """
    n_loc = outbound.groupby("sku_id")["loc_id"].nunique()
    if (n_loc > 1).any():
        raise ValueError(f"存在多库位 SKU，库位模型前提不成立：{n_loc[n_loc > 1].index[:5].tolist()}")
    return outbound.groupby("sku_id")["loc_id"].first()


def location_utilization(assignment: pd.Series, loc: pd.DataFrame) -> dict:
    """库容利用率 = 已占用库位数 / 库位总数（需求 3.7）。

    `assignment` 为「SKU → 主库位」映射（见 sku_location_map）；一 SKU 一库位，
    故占用数 = 被映射到的不同库位数。
    """
    occupied = int(assignment.nunique())
    total = int(len(loc))
    return {
        "rate": occupied / total if total else float("nan"),
        "occupied": occupied,
        "total_locations": total,
    }


def stocktake_discrepancy(
    stocktake: pd.DataFrame, sku: pd.DataFrame, by: str | None = None
) -> dict | pd.DataFrame:
    """盘点差异率 = 差异记录数 / 盘点记录数；`by` 支持 "category" / "date" 下钻（需求 3.7）。

    差异 = diff_type 非「无差异」（盘亏 / 盘盈都算差异）。

    下钻维度：
      - `category`：按 SKU 品类，用于复现 P03 高盘差埋点；
      - `date`：按盘点日期（数据层 A 每日一轮循环盘点），即本数据能支撑的「分时段」下钻。
        盘点表**不含时刻**，故 14–16 点效率低谷不可能由盘差率复现——该低谷的载体是
        「拣货效率」按小时下钻（见 `picking_efficiency_by_hour`），两者不可互换。
    """
    is_diff = stocktake["diff_type"] != "无差异"
    if by is None:
        return {
            "rate": float(is_diff.mean()),
            "n_records": int(len(stocktake)),
            "n_diff": int(is_diff.sum()),
        }
    if by == "category":
        key = stocktake["sku_id"].map(dict(zip(sku["sku_id"], sku["category"])))
        key.name = "category"
    elif by == "date":
        key = pd.to_datetime(stocktake["date"]).dt.date
        key.name = "date"
        key = key.astype(str)
    else:
        raise ValueError(f"不支持的盘点差异下钻维度：{by}（仅支持 'category' / 'date'）")
    return is_diff.groupby(key).agg(rate="mean", n_records="size", n_diff="sum").reset_index()


def picking_efficiency_by_hour(outbound: pd.DataFrame) -> pd.DataFrame:
    """按拣货起始小时下钻的拣货时长（秒/行），用于复现 14–16 点效率低谷埋点。"""
    seconds = (
        pd.to_datetime(outbound["pick_end"]) - pd.to_datetime(outbound["pick_start"])
    ).dt.total_seconds()
    hour = pd.to_datetime(outbound["pick_start"]).dt.hour
    out = (
        pd.DataFrame({"hour": hour, "seconds": seconds})
        .groupby("hour", as_index=False)
        .agg(sec_per_line=("seconds", "mean"), n_lines=("seconds", "size"))
    )
    return out


def daily_kpi(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """仓内指标的**逐日**序列（驾驶舱趋势线与环比箭头的数据源）。

    一行一天，列为：库存准确率（金额加权，见 `inventory_accuracy`）、盘点差异率、
    收货及时率、拣货效率（行/人时）。

    「这一天」的定义逐指标写明，因为三张表的时间字段不是一回事：
      - 库存准确率 / 盘差率 → 盘点表的 `date`（数据层 A 每日一轮循环盘点）；
      - 收货及时率 → 收货表的**预约到货日**（排班口径看的是「该哪天到」，不是实际哪天到，
        否则晚到会被算到第二天去、当天的问题就消失了）；
      - 拣货效率 → 出库行的 `pick_start` 所在日。

    收货及时率与拣货效率的判定掩码取自 `_receipt_on_time` / `_pick_seconds`——与聚合版
    `receipt_timeliness` / `picking_efficiency` **同源**。逐日版原先把这两段口径各重写了一遍，
    而它正是驾驶舱趋势线的数据来源，两版漂移时没有任何东西会变红。
    """
    sku, outbound, stocktake, inbound = (
        tables["sku"], tables["outbound"], tables["stocktake"], tables["inbound"]
    )
    acc = inventory_accuracy(stocktake, sku, by="date").rename(columns={"rate": "inventory_accuracy"})
    acc = acc[["date", "inventory_accuracy", "abs_diff_value", "book_value", "n_records"]].rename(
        columns={"n_records": "n_stocktake_records"}
    )
    disc = stocktake_discrepancy(stocktake, sku, by="date").rename(
        columns={"rate": "stocktake_discrepancy_rate", "n_records": "n_discrepancy_records"}
    )

    on_time, expected, _ = _receipt_on_time(inbound)
    receipt = (
        pd.DataFrame({"date": expected.dt.date, "on_time": on_time})
        .groupby("date", as_index=False)
        .agg(receipt_timeliness_rate=("on_time", "mean"), n_receipts=("on_time", "size"))
    )
    receipt["date"] = receipt["date"].astype(str)

    seconds, start = _pick_seconds(outbound)
    pick = (
        pd.DataFrame({"date": start.dt.date, "seconds": seconds})
        .groupby("date", as_index=False)
        .agg(picking_seconds=("seconds", "sum"), n_pick_lines=("seconds", "size"))
    )
    pick["picking_lines_per_hour"] = pick["n_pick_lines"] / (pick["picking_seconds"] / 3600)
    pick["date"] = pick["date"].astype(str)

    out = acc.merge(disc, on="date", how="outer").merge(receipt, on="date", how="outer")
    out = out.merge(pick, on="date", how="outer")
    keep = ["date", "inventory_accuracy", "stocktake_discrepancy_rate", "receipt_timeliness_rate",
            "picking_lines_per_hour", "abs_diff_value", "book_value",
            "n_stocktake_records", "n_discrepancy_records",
            "n_receipts", "n_pick_lines", "picking_seconds"]
    return out[keep].sort_values("date", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# ABC 分类（数据源 = 仿真仓出库行数，ADR-0002）
# ---------------------------------------------------------------------------
def abc_classify(
    outbound: pd.DataFrame, thresholds: tuple[float, float] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """按出库行数做 ABC 分类。返回 (SKU 级分类表, 帕累托表, 摘要)。

    口径（ADR-0002）：以**仿真仓出库行数**为权重降序累计，累计占比 ≤ 阈值即归入该类。
    行数而非数量：拣货成本由行数驱动（每行一次行走 + 一次操作），
    与库位重排的「Σ(出库频次 × 距离)」口径同源。
    """
    a_cut, b_cut = C.ABC_THRESHOLDS if thresholds is None else thresholds
    if not 0 < a_cut < b_cut < 1:
        raise ValueError(f"ABC 阈值须满足 0 < A < B < 1，收到 {a_cut}, {b_cut}")
    counts = (
        outbound.groupby("sku_id", as_index=False)
        .size()
        .rename(columns={"size": "outbound_lines"})
        .sort_values(["outbound_lines", "sku_id"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    total = int(counts["outbound_lines"].sum())
    counts["share"] = counts["outbound_lines"] / total
    counts["cum_share"] = counts["share"].cumsum()
    # 用「上一条的累计占比」判类，保证每类至少覆盖到阈值本身
    prev_cum = counts["cum_share"].shift(fill_value=0.0)
    counts["abc_class"] = np.where(prev_cum < a_cut, "A", np.where(prev_cum < b_cut, "B", "C"))
    counts["rank"] = np.arange(1, len(counts) + 1)

    summary = {
        "thresholds": {"A": a_cut, "B": b_cut},
        "total_lines": total,
        "n_sku": int(len(counts)),
        "by_class": {
            cls: {
                "n_sku": int((counts["abc_class"] == cls).sum()),
                "sku_share": round(float((counts["abc_class"] == cls).mean()), 4),
                "line_share": round(float(counts.loc[counts["abc_class"] == cls, "share"].sum()), 4),
            }
            for cls in ("A", "B", "C")
        },
    }
    pareto = counts[["rank", "sku_id", "outbound_lines", "share", "cum_share", "abc_class"]].copy()
    return counts, pareto, summary


# ---------------------------------------------------------------------------
# 库位优化：库位—拣货台距离模型
# ---------------------------------------------------------------------------
def walk_cost(frequencies, distances) -> float:
    """拣货行走成本 = Σ(出库频次 × 库位距离)，单位：行·米（需求 34 口径）。

    传入 pandas Series 时**按索引对齐**再相乘。库位重排的前后对比正是「两个 Series
    索引相同、顺序不同」的场景，若按位置相乘会静默算出错误成本（实测踩到过）。
    """
    if isinstance(frequencies, pd.Series) and isinstance(distances, pd.Series):
        f_ser, d_ser = frequencies, distances.reindex(frequencies.index)
        if d_ser.isna().any():
            raise ValueError("距离序列缺少频次序列中的索引，无法对齐")
        f, d = f_ser, d_ser
    else:
        f, d = frequencies, distances
    f = np.asarray(f, dtype=float)
    d = np.asarray(d, dtype=float)
    if f.shape != d.shape:
        raise ValueError(f"频次与距离长度不一致：{f.shape} vs {d.shape}")
    return float((f * d).sum())


def optimize_slotting(
    frequencies: pd.Series, distances: pd.Series, loc_ids: pd.Series
) -> dict:
    """把 SKU 重排到现有库位，使 Σ(出库频次 × 库位距离) 最小（需求 34）。

    按**重排不等式**，Σ fᵢd_σ(i) 在「频次降序 ↔ 距离升序」配对时取最小，故该配对
    就是同一批库位下的最优解，无需迭代搜索。库位集合不变（不扩仓），只换占用关系。
    """
    order_by_freq = frequencies.sort_values(ascending=False, kind="stable").index
    available = distances.sort_values(ascending=True, kind="stable")
    if len(order_by_freq) != len(available):
        raise ValueError("SKU 数与可用库位数不一致，无法重排")
    new_dist = pd.Series(available.to_numpy(), index=order_by_freq)
    before = walk_cost(frequencies, distances)
    after = walk_cost(frequencies, new_dist)
    # 预计拣货效率：把行走成本折成「每行平均行走时间」，用 config 的仓内行走速度换算。
    # 这是本模块自有的效率口径（不借用数据层 F 的仿真结论）。
    line_total = float(frequencies.sum())
    walk_m_before = before / line_total if line_total else float("nan")
    walk_m_after = after / line_total if line_total else float("nan")
    sec_before = walk_m_before / C.WALK_SPEED_M_PER_SEC
    sec_after = walk_m_after / C.WALK_SPEED_M_PER_SEC
    assignment = pd.DataFrame(
        {
            "sku_id": order_by_freq.to_numpy(),
            "outbound_lines": frequencies.loc[order_by_freq].to_numpy(),
            "walk_dist_before_m": distances.loc[order_by_freq].to_numpy(),
            "walk_dist_after_m": new_dist.loc[order_by_freq].to_numpy(),
            "loc_id_after": loc_ids.loc[available.index].to_numpy(),
        }
    )
    return {
        "walk_cost_before": round(before, 1),
        "walk_cost_after": round(after, 1),
        "reduction_pct": round((before - after) / before * 100, 2) if before else None,
        "walk_m_per_line_before": round(walk_m_before, 2),
        "walk_m_per_line_after": round(walk_m_after, 2),
        "walk_sec_per_line_before": round(sec_before, 2),
        "walk_sec_per_line_after": round(sec_after, 2),
        "walk_sec_per_line_saved": round(sec_before - sec_after, 2),
        "walk_speed_m_per_sec": C.WALK_SPEED_M_PER_SEC,
        "assignment": assignment,
    }


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------
def compute_all_kpis(tables: dict[str, pd.DataFrame]) -> dict:
    """计算六 KPI + 两个下钻 + ABC + 库位重排（08 号票主体）。"""
    sku, loc = tables["sku"], tables["location"]
    outbound, stocktake, inbound = tables["outbound"], tables["stocktake"], tables["inbound"]
    sku_loc = sku_location_map(outbound)

    kpis = {
        "inventory_accuracy": inventory_accuracy(stocktake, sku),
        "receipt_timeliness": receipt_timeliness(inbound),
        "picking_efficiency": picking_efficiency(outbound),
        "avg_fulfillment_hours": avg_fulfillment_hours(outbound),
        "location_utilization": location_utilization(sku_loc, loc),
        "stocktake_discrepancy": stocktake_discrepancy(stocktake, sku),
    }
    by_category = stocktake_discrepancy(stocktake, sku, by="category")
    by_date = stocktake_discrepancy(stocktake, sku, by="date")
    by_hour = picking_efficiency_by_hour(outbound)
    abc_sku, pareto, abc_summary = abc_classify(outbound)
    # 库位重排：频次 = 出库行数，距离 = 该 SKU 当前库位的行走距离
    loc_dist = dict(zip(loc["loc_id"], loc["walk_dist_m"]))
    freq = abc_sku.set_index("sku_id")["outbound_lines"].astype(float).sort_index()
    dist = pd.Series({s: loc_dist[sku_loc[s]] for s in freq.index}, dtype=float)
    locs = pd.Series({s: sku_loc[s] for s in freq.index})
    slotting = optimize_slotting(freq, dist, locs)

    # 埋点复现：P03 盘差（分品类）与 14–16 点拣货低谷（分时段）。
    #
    # 两处都按**自然单位**合并（记录数 / 行数），而不是对各组比率取等权平均。理由是
    # 单组的率本来就是合并算出来的（P03 的 17.45% 就是 74/424），拿它去比一个「各组等权」
    # 的均值，两侧根本不是同一种量；而且等权平均会让记录少的组拿到与记录多的组一样的权重。
    # 数据层 A 的质检摘要用的正是合并口径，此前两者对不上就出在这里——见
    # `tests/test_warehouse_kpi.py::TestGeneratorQcVersusKpiReDerivation` 的对账。
    cat = by_category.set_index("category")
    other_cats = cat.drop(index=C.HIGH_DIFF_CATEGORY)
    p03_rate = float(cat.loc[C.HIGH_DIFF_CATEGORY, "rate"])
    other_rate = _pooled_rate(other_cats["rate"], other_cats["n_records"])

    h0, h1 = C.PICK_SLOW_HOURS
    hour = by_hour.set_index("hour")
    slow_mask = (hour.index >= h0) & (hour.index < h1)
    slow_sec = _pooled_rate(hour.loc[slow_mask, "sec_per_line"], hour.loc[slow_mask, "n_lines"])
    other_sec = _pooled_rate(hour.loc[~slow_mask, "sec_per_line"], hour.loc[~slow_mask, "n_lines"])

    return {
        "kpis": kpis,
        "drilldown": {"stocktake_by_category": by_category, "stocktake_by_date": by_date,
                      "picking_by_hour": by_hour},
        "daily": daily_kpi(tables),
        "abc": {"sku": abc_sku, "pareto": pareto, "summary": abc_summary},
        "slotting": slotting,
        "embedding_checks": {
            "p03_discrepancy": {
                "p03_rate": round(p03_rate, 4),
                "other_rate": round(other_rate, 4),
                "ratio": round(p03_rate / other_rate, 2) if other_rate else None,
            },
            "pick_slowdown_14_16": {
                "sec_per_line_14_16": round(slow_sec, 1),
                "sec_per_line_other": round(other_sec, 1),
                "ratio": round(slow_sec / other_sec, 2) if other_sec else None,
            },
        },
    }


def _write_outputs(result: dict, out_dir: Path) -> dict:
    """落盘 KPI、下钻、ABC 与库位重排。

    `out_dir` 可重定向（单测据此写进临时目录，不污染产物目录）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    kpis = result["kpis"]
    written: dict[str, Path] = {}

    def _json(name: str, payload) -> None:
        p = out_dir / name
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written[name] = p

    def _csv(name: str, frame: pd.DataFrame) -> None:
        p = out_dir / name
        frame.to_csv(p, index=False, encoding="utf-8-sig")
        written[name] = p

    _csv(C.WAREHOUSE_DAILY_CSV.name, result["daily"])

    _json(
        C.WAREHOUSE_KPI_JSON.name,
        {
            "data_category": "过程仿真",
            "source": "数据层 A 仿真仓五表（data/warehouse/）",
            "kpis": kpis,
            "abc": result["abc"]["summary"],
            "slotting": {k: v for k, v in result["slotting"].items() if k != "assignment"},
            "embedding_checks": result["embedding_checks"],
        },
    )
    drill = result["drilldown"]
    _csv(C.WAREHOUSE_DRILLDOWN_CATEGORY_CSV.name, drill["stocktake_by_category"])
    _csv(C.WAREHOUSE_DRILLDOWN_DATE_CSV.name, drill["stocktake_by_date"])
    _csv(C.WAREHOUSE_DRILLDOWN_HOUR_CSV.name, drill["picking_by_hour"])
    _csv(C.WAREHOUSE_ABC_CSV.name, result["abc"]["sku"])
    _csv(C.WAREHOUSE_ABC_PARETO_CSV.name, result["abc"]["pareto"])
    slot = result["slotting"]
    _json(
        C.WAREHOUSE_SLOTTING_JSON.name,
        {k: v for k, v in slot.items() if k != "assignment"},
    )
    _csv(C.WAREHOUSE_SLOTTING_CSV.name, slot["assignment"])

    return written


def _display_path(p) -> str:
    """报告里展示路径：项目内显示相对路径，项目外（如单测临时目录）只显示文件名。"""
    p = Path(p).resolve()
    root = C.PROJECT_ROOT.resolve()
    if p == root or root in p.parents:
        return str(p.relative_to(root))
    return p.name


def _write_report(result: dict, sim_exp1: dict | None, path: Path | None = None) -> Path:
    """生成模块一仓内侧 Markdown 报告（含与数据层 F 实验一的互证段落）。"""
    path = (C.WAREHOUSE_KPI_MD if path is None else Path(path))
    k, e, s = result["kpis"], result["embedding_checks"], result["slotting"]
    abc = result["abc"]["summary"]
    by_date = result["drilldown"]["stocktake_by_date"]["rate"]
    d = {
        "n_days": int(len(by_date)),
        "min": float(by_date.min()),
        "max": float(by_date.max()),
        "mean": float(by_date.mean()),
    }
    lines = [
        "# 模块一（上）：仓内运营分析报告",
        "",
        "- 数据源：数据层 A 仿真仓五表（过程仿真，90 天）",
        "",
        "## 一、仓内六 KPI（严格口径）",
        "",
        "| KPI | 取值 | 口径 |",
        "|---|---|---|",
        f"| 库存准确率 | {k['inventory_accuracy']['rate']:.4f} | "
        f"1 − Σ\\|账实差异\\|×单价 / Σ账面×单价（金额加权） |",
        f"| 收货及时率 | {k['receipt_timeliness']['rate']:.4f} | "
        f"实际到货 ≤ 预约 + {k['receipt_timeliness']['tolerance_min']:.0f} 分钟容差 |",
        f"| 拣货效率 | {k['picking_efficiency']['lines_per_hour']:.1f} 行/人时 | "
        f"出库总行数 / Σ逐行拣货工时 |",
        f"| 订单平均履约时效 | {k['avg_fulfillment_hours']['hours']:.2f} 小时 | "
        f"平均(发货 − 下单)，按订单聚合 |",
        f"| 库容利用率 | {k['location_utilization']['rate']:.4f} | "
        f"已占用库位 {k['location_utilization']['occupied']} / "
        f"{k['location_utilization']['total_locations']} |",
        f"| 盘点差异率 | {k['stocktake_discrepancy']['rate']:.4f} | "
        f"非「无差异」记录占比（盘亏 + 盘盈） |",
        "",
        "## 二、埋点复现（分品类 / 分时段下钻）",
        "",
        f"1. **{C.HIGH_DIFF_CATEGORY} 高盘差（盘差率按品类下钻）**："
        f"{e['p03_discrepancy']['p03_rate']:.2%} vs 其他品类 "
        f"{e['p03_discrepancy']['other_rate']:.2%}，比值 {e['p03_discrepancy']['ratio']}",
        f"2. **盘差率按日期下钻**：{d['n_days']} 天，日盘差率 "
        f"{d['min']:.2%}–{d['max']:.2%}（均值 {d['mean']:.2%}），未呈现系统性时段规律——"
        f"盘点表只到「日」粒度、**不含时刻**，故 14–16 点低谷不可能由盘差率复现",
        f"3. **14–16 点拣货低谷（拣货效率按小时下钻）**：低谷 "
        f"{e['pick_slowdown_14_16']['sec_per_line_14_16']} 秒/行 vs 其余时段 "
        f"{e['pick_slowdown_14_16']['sec_per_line_other']} 秒/行，"
        f"比值 {e['pick_slowdown_14_16']['ratio']}",
        "",
        "两个埋点分属**不同 KPI、不同下钻维度**：盘差看品类，低谷看时段（拣货效率口径）。"
        "票据把二者并列，但下钻维度不可互换。",
        "",
        "## 三、ABC 分类（数据源 = 仿真仓出库行数，ADR-0002）",
        "",
        f"- 阈值：累计占比**首次达到** {abc['thresholds']['A']:.0%} 之前（含跨过阈值的那一个）归 A，"
        f"达到 {abc['thresholds']['B']:.0%} 之前归 B，其余 C",
        "",
        "| 类别 | SKU 数 | SKU 占比 | 出库行数占比 |",
        "|---|---|---|---|",
    ]
    for cls in ("A", "B", "C"):
        d = abc["by_class"][cls]
        lines.append(f"| {cls} | {d['n_sku']} | {d['sku_share']:.1%} | {d['line_share']:.1%} |")
    lines += [
        "",
        "## 四、库位重排（Σ(出库频次 × 库位距离)）",
        "",
        f"- 重排前行走成本：**{s['walk_cost_before']:,.0f} 行·米**",
        f"- 重排后行走成本：**{s['walk_cost_after']:,.0f} 行·米**",
        f"- 降幅：**{s['reduction_pct']}%**",
        f"- 预计拣货效率（本模块自有口径，行走速度 {s['walk_speed_m_per_sec']} m/s）："
        f"每行平均行走 {s['walk_m_per_line_before']} m → {s['walk_m_per_line_after']} m，"
        f"折合 {s['walk_sec_per_line_before']} s → {s['walk_sec_per_line_after']} s/行"
        f"（**每行省 {s['walk_sec_per_line_saved']} s**）",
        "- 重排规则：按重排不等式，出库频次降序 ↔ 库位距离升序配对即为同批库位下的最优解",
        "",
    ]
    if sim_exp1:
        lines += [
            "### 与数据层 F 实验一互证",
            "",
            f"- SimPy 仿真（实验一）：原始布局单均行走 {sim_exp1['original_walk']:.1f} m → "
            f"ABC 分区 {sim_exp1['zoned_walk']:.1f} m（**降幅 {sim_exp1['walk_reduction_pct']}%**），"
            f"单均拣货时长 {sim_exp1['original_pick']:.1f} s → {sim_exp1['zoned_pick']:.1f} s"
            f"（降幅 {sim_exp1['pick_reduction_pct']}%）",
            "- 两条证据链方向一致、量级可比但**模型不同**：模块一是 90 天实际出库行数的确定性"
            "Σ(频次×距离)，实验一是代表日的离散事件仿真（含排队与人力耦合）。"
            "二者互证「A 类就近存放」的收益，不互相替代。",
            "",
        ]
    lines += [
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_sim_exp1() -> dict | None:
    """读取数据层 F 实验一结果供互证；缺失时返回 None（不阻断模块一）。"""
    path = C.PROCESSED_DIR / "sim" / "exp1_layout.json"
    if not path.exists():
        logger.warning("未找到数据层 F 实验一结果（%s），跳过互证段落", path)
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    o, z = data["original"]["metrics"], data["abc_zoned"]["metrics"]
    return {
        "original_walk": o["avg_walk_m_per_order"]["mean"],
        "zoned_walk": z["avg_walk_m_per_order"]["mean"],
        "walk_reduction_pct": round(
            (o["avg_walk_m_per_order"]["mean"] - z["avg_walk_m_per_order"]["mean"])
            / o["avg_walk_m_per_order"]["mean"] * 100, 2
        ),
        "original_pick": o["avg_order_pick_sec"]["mean"],
        "zoned_pick": z["avg_order_pick_sec"]["mean"],
        "pick_reduction_pct": round(
            (o["avg_order_pick_sec"]["mean"] - z["avg_order_pick_sec"]["mean"])
            / o["avg_order_pick_sec"]["mean"] * 100, 2
        ),
    }


def run_all(out_dir: Path | None = None) -> dict:
    """跑通模块一仓内侧全流程并落盘（`out_dir` 可重定向，供单测写临时目录）。"""
    out_dir = C.WAREHOUSE_KPI_DIR if out_dir is None else Path(out_dir)
    tables = load_warehouse_tables()
    result = compute_all_kpis(tables)
    written = _write_outputs(result, out_dir)
    report = _write_report(result, _load_sim_exp1(), out_dir / C.WAREHOUSE_KPI_MD.name)
    written[C.WAREHOUSE_KPI_MD.name] = report
    return {"result": result, "paths": written}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = run_all()
    k, e, s = out["result"]["kpis"], out["result"]["embedding_checks"], out["result"]["slotting"]
    print("\n=== 模块一（上）仓内 KPI ===")
    print(f"  库存准确率 {k['inventory_accuracy']['rate']:.4f}  |  "
          f"收货及时率 {k['receipt_timeliness']['rate']:.4f}  |  "
          f"拣货效率 {k['picking_efficiency']['lines_per_hour']:.1f} 行/人时")
    print(f"  平均履约时效 {k['avg_fulfillment_hours']['hours']:.2f} h  |  "
          f"库容利用率 {k['location_utilization']['rate']:.4f}  |  "
          f"盘点差异率 {k['stocktake_discrepancy']['rate']:.4f}")
    print(f"  埋点① {C.HIGH_DIFF_CATEGORY} 盘差 {e['p03_discrepancy']['p03_rate']:.2%} vs "
          f"其他 {e['p03_discrepancy']['other_rate']:.2%}（比值 {e['p03_discrepancy']['ratio']}）")
    print(f"  埋点② 14–16 点 {e['pick_slowdown_14_16']['sec_per_line_14_16']} 秒/行 vs "
          f"其余 {e['pick_slowdown_14_16']['sec_per_line_other']} 秒/行"
          f"（比值 {e['pick_slowdown_14_16']['ratio']}）")
    print(f"  库位重排行走成本 {s['walk_cost_before']:,.0f} → {s['walk_cost_after']:,.0f} 行·米"
          f"（降 {s['reduction_pct']}%）")


if __name__ == "__main__":
    main()
