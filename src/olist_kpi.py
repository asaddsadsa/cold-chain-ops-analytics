"""模块一（下）：Olist 真实履约 KPI（09 号票）。

基于 03 号票产出的 clean_orders，计算真实履约 KPI 并按客户州 / 商品重量段 /
下单月份三维下钻，输出延迟 Top 区域、重量段时效关系、延迟 vs 准时订单的评分交叉。

术语严格遵循 CONTEXT.md：Olist 侧叫「真实准时交付率」（实际送达 ≤ 预计送达），
绝不与配送侧「时间窗达成率」混用。

运行方式（项目根）：python -m src.olist_kpi
数据类别：真实观测（见 data_sources_ledger.md 第 1 节）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

logger = logging.getLogger(__name__)

# 重量段口径（登记台账）：订单级重量 = 该订单所有 item 重量之和（kg）
WEIGHT_BINS_KG = [0, 0.5, 1, 3, 10, float("inf")]
WEIGHT_LABELS = ["≤0.5kg", "0.5–1kg", "1–3kg", "3–10kg", ">10kg"]
MIN_STATE_ORDERS = 100  # 延迟 Top 区域榜的州最小订单数阈值（避免小样本噪声）
BAD_REVIEW_THRESHOLD = 2  # review_score ≤ 2 记为差评


# ---------------------------------------------------------------------------
# 纯 KPI 函数（可独立调用、可单测）
# ---------------------------------------------------------------------------
def real_ontime_rate(df: pd.DataFrame) -> float:
    """真实准时交付率 = 实际送达 ≤ 预计送达的订单占比（边界「相等」算准时）。

    is_late 已在 03 落盘（实际送达 > 预计送达）。
    """
    if len(df) == 0:
        return float("nan")
    return float((~df["is_late"]).mean())


def real_delay_rate(df: pd.DataFrame) -> float:
    """真实延迟率 = 1 − 真实准时交付率。"""
    if len(df) == 0:
        return float("nan")
    return float(df["is_late"].mean())


def add_order_weight(orders: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """给订单表附加订单级总重量（kg）与重量段标签。

    订单级重量 = 该订单所有 item 的 product_weight_g 之和 / 1000。
    无重量数据的订单段标为「未知」，不进重量时效分析。
    """
    df = orders.copy()
    item_w = items.copy()
    item_w["product_weight_g"] = pd.to_numeric(item_w["product_weight_g"], errors="coerce")
    order_weight_g = item_w.groupby("order_id")["product_weight_g"].sum()
    df["order_weight_kg"] = df["order_id"].map(order_weight_g) / 1000.0
    df["weight_segment"] = pd.cut(
        df["order_weight_kg"], bins=WEIGHT_BINS_KG, labels=WEIGHT_LABELS, right=True
    )
    df["weight_segment"] = df["weight_segment"].astype(object).where(
        df["order_weight_kg"].notna(), "未知"
    )
    return df


def _group_stats(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """按某维度分组统计：订单数 / 真实准时交付率 / 延迟率 / 平均履约天数。"""
    g = df.groupby(by, observed=True)
    out = pd.DataFrame(
        {
            "n_orders": g.size(),
            "ontime_rate": g.apply(lambda x: real_ontime_rate(x), include_groups=False),
            "delay_rate": g.apply(lambda x: real_delay_rate(x), include_groups=False),
            "avg_fulfillment_days": g["fulfillment_days"].mean(),
        }
    ).reset_index()
    out["ontime_rate"] = out["ontime_rate"].round(4)
    out["delay_rate"] = out["delay_rate"].round(4)
    out["avg_fulfillment_days"] = out["avg_fulfillment_days"].round(2)
    return out


def drilldown_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """按客户州下钻，延迟率降序（延迟 Top 区域）。仅纳入订单数 ≥ 阈值的州。"""
    stats = _group_stats(df, "customer_state")
    stats = stats[stats["n_orders"] >= MIN_STATE_ORDERS]
    return stats.sort_values("delay_rate", ascending=False).reset_index(drop=True)


def drilldown_by_weight(df: pd.DataFrame) -> pd.DataFrame:
    """按商品重量段下钻（剔除「未知」段），按重量段顺序排列，用于时效关系分析。"""
    d = df[df["weight_segment"] != "未知"]
    stats = _group_stats(d, "weight_segment")
    order = {lbl: i for i, lbl in enumerate(WEIGHT_LABELS)}
    stats["_ord"] = stats["weight_segment"].map(order)
    return stats.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def drilldown_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """按下单月份下钻，月份升序。"""
    d = df.copy()
    d["month"] = pd.to_datetime(d["order_purchase_timestamp"]).dt.to_period("M").astype(str)
    stats = _group_stats(d, "month")
    return stats.sort_values("month").reset_index(drop=True)


def weight_time_relationship(df: pd.DataFrame) -> pd.DataFrame:
    """重量段时效关系：各重量段平均履约/在途天数 + 单调性观察。"""
    d = df[df["weight_segment"] != "未知"]
    g = d.groupby("weight_segment", observed=True)
    out = pd.DataFrame(
        {
            "n_orders": g.size(),
            "avg_fulfillment_days": g["fulfillment_days"].mean(),
            "avg_transit_days": g["transit_days"].mean(),
        }
    ).reset_index()
    order = {lbl: i for i, lbl in enumerate(WEIGHT_LABELS)}
    out["_ord"] = out["weight_segment"].map(order)
    out = out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    out["avg_fulfillment_days"] = out["avg_fulfillment_days"].round(2)
    out["avg_transit_days"] = out["avg_transit_days"].round(2)
    return out


def review_cross_analysis(orders: pd.DataFrame, reviews: pd.DataFrame) -> dict:
    """延迟订单 vs 准时订单的评分分布交叉。

    一单可有多条评论 → 取订单均分；差评率 = review_score ≤ 2 的订单占比。
    返回 {延迟组/准时组: {订单数, 均分, 各分值占比, 差评率}}。
    """
    rv = reviews.copy()
    rv["review_score"] = pd.to_numeric(rv["review_score"], errors="coerce")
    rv = rv.dropna(subset=["review_score", "order_id"])
    order_score = rv.groupby("order_id")["review_score"].mean().rename("avg_review_score")

    df = orders.merge(order_score, on="order_id", how="inner")
    df["is_bad"] = df["avg_review_score"] <= BAD_REVIEW_THRESHOLD

    result: dict = {}
    for label, mask in (("late", df["is_late"]), ("on_time", ~df["is_late"])):
        sub = df[mask]
        n = len(sub)
        if n == 0:
            result[label] = {"n_orders": 0}
            continue
        # 订单均分四舍五入到整数后的分布（1–5）
        rounded = sub["avg_review_score"].round().clip(1, 5).astype(int)
        dist = (rounded.value_counts(normalize=True).sort_index() * 100).round(1).to_dict()
        result[label] = {
            "n_orders": n,
            "mean_review_score": round(float(sub["avg_review_score"].mean()), 3),
            "score_distribution_pct": {int(k): float(v) for k, v in dist.items()},
            "bad_review_rate": round(float(sub["is_bad"].mean()), 4),
        }
    result["late_vs_ontime_score_gap"] = round(
        result.get("on_time", {}).get("mean_review_score", 0)
        - result.get("late", {}).get("mean_review_score", 0),
        3,
    )
    return result


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------
def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_olist_kpi(
    clean_orders_csv: Path | None = None,
    items_csv: Path | None = None,
    reviews_csv: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """端到端计算 Olist 履约 KPI 并落盘 processed/olist_kpi/。返回产物路径。"""
    olist_dir = C.PROCESSED_DIR / "olist"
    clean_orders_csv = Path(clean_orders_csv) if clean_orders_csv else olist_dir / "clean_orders.csv"
    items_csv = Path(items_csv) if items_csv else olist_dir / "order_items_clean.csv"
    reviews_csv = Path(reviews_csv) if reviews_csv else C.RAW_OLIST_DIR / "olist_order_reviews_dataset.csv"
    out_dir = Path(out_dir) if out_dir else C.PROCESSED_DIR / "olist_kpi"
    out_dir.mkdir(parents=True, exist_ok=True)

    orders = pd.read_csv(clean_orders_csv)
    items = pd.read_csv(items_csv)
    reviews = pd.read_csv(reviews_csv)

    orders = add_order_weight(orders, items)

    paths: dict[str, Path] = {}

    # 总体 KPI
    overall = {
        "n_orders": int(len(orders)),
        "real_ontime_rate": round(real_ontime_rate(orders), 4),
        "real_delay_rate": round(real_delay_rate(orders), 4),
        "avg_fulfillment_days": round(float(orders["fulfillment_days"].mean()), 2),
        "avg_outbound_response_days": round(float(orders["outbound_response_days"].mean()), 2),
        "avg_transit_days": round(float(orders["transit_days"].mean()), 2),
    }
    paths["overall"] = out_dir / "overall_kpi.json"
    _write_json(overall, paths["overall"])

    # 三维下钻
    by_state = drilldown_by_state(orders)
    by_weight = drilldown_by_weight(orders)
    by_month = drilldown_by_month(orders)
    paths["by_state"] = out_dir / "drilldown_by_state.csv"
    paths["by_weight"] = out_dir / "drilldown_by_weight.csv"
    paths["by_month"] = out_dir / "drilldown_by_month.csv"
    by_state.to_csv(paths["by_state"], index=False)
    by_weight.to_csv(paths["by_weight"], index=False)
    by_month.to_csv(paths["by_month"], index=False)

    # 重量段时效关系
    wt = weight_time_relationship(orders)
    paths["weight_time"] = out_dir / "weight_time_relationship.csv"
    wt.to_csv(paths["weight_time"], index=False)

    # reviews 交叉
    rc = review_cross_analysis(orders, reviews)
    paths["review_cross"] = out_dir / "review_cross.json"
    _write_json(rc, paths["review_cross"])

    # Markdown 摘要
    md = [
        "# 模块一（下）：Olist 真实履约 KPI（真实观测）",
        "",
        f"- 订单数：{overall['n_orders']:,}",
        f"- **真实准时交付率：{overall['real_ontime_rate']:.2%}**（延迟率 {overall['real_delay_rate']:.2%}）",
        f"- 平均履约 {overall['avg_fulfillment_days']} 天 / 出库响应 {overall['avg_outbound_response_days']} 天 / 在途 {overall['avg_transit_days']} 天",
        "",
        f"## 延迟 Top 区域（州订单数 ≥ {MIN_STATE_ORDERS}）",
        "",
        "| 州 | 订单数 | 延迟率 | 真实准时交付率 | 平均履约天数 |",
        "|---|---|---|---|---|",
    ]
    for _, r in by_state.head(10).iterrows():
        md.append(
            f"| {r['customer_state']} | {r['n_orders']:,} | {r['delay_rate']:.2%} | "
            f"{r['ontime_rate']:.2%} | {r['avg_fulfillment_days']} |"
        )
    md += ["", "## 重量段时效关系", "", "| 重量段 | 订单数 | 平均履约天数 | 平均在途天数 |",
           "|---|---|---|---|"]
    for _, r in wt.iterrows():
        md.append(
            f"| {r['weight_segment']} | {r['n_orders']:,} | {r['avg_fulfillment_days']} | {r['avg_transit_days']} |"
        )
    md += [
        "",
        "## 延迟 vs 准时订单评分交叉",
        "",
        f"- 延迟订单：n={rc['late'].get('n_orders', 0):,}，均分 {rc['late'].get('mean_review_score')}，"
        f"差评率 {rc['late'].get('bad_review_rate', 0):.2%}",
        f"- 准时订单：n={rc['on_time'].get('n_orders', 0):,}，均分 {rc['on_time'].get('mean_review_score')}，"
        f"差评率 {rc['on_time'].get('bad_review_rate', 0):.2%}",
        f"- 准时组比延迟组评分高 {rc['late_vs_ontime_score_gap']} 分",
        "",
        "> 口径：重量段=订单内 item 重量之和(kg)；准时=实际送达≤预计送达（相等算准时）；差评=订单评论均分≤2。",
    ]
    (out_dir / "olist_kpi_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    paths["report_md"] = out_dir / "olist_kpi_report.md"

    logger.info(
        "Olist KPI 完成：n=%d，真实准时交付率 %.2f%%，延迟率 %.2f%%；延迟最高州=%s",
        overall["n_orders"], overall["real_ontime_rate"] * 100, overall["real_delay_rate"] * 100,
        by_state.iloc[0]["customer_state"] if len(by_state) else "N/A",
    )
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = run_olist_kpi()
    print("\n=== 模块一（下）产物 ===")
    for k, v in paths.items():
        print(f"  {k:16s} -> {v}")


if __name__ == "__main__":
    main()
