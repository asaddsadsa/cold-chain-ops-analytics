"""数据层 B：Olist 真实订单获取与清洗（03 号票）。

三段式获取（本地 raw → kagglehub 下载 → 报错给手动指引，严禁编造数据）+ 清洗
（时间转 datetime、剔除非 delivered、剔除时间逻辑矛盾、邮编多坐标聚合、关联
product/customers）+ 两项下游标定产物：商品重量/体积对数正态经验分布、Olist
真实延迟率。

运行方式（项目根）：python -m src.olist_clean
数据类别：真实观测（见 data_sources_ledger.md 第 1 节）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import config as C

logger = logging.getLogger(__name__)

# Olist 九表文件名（kagglehub dataset olistbr/brazilian-ecommerce）
OLIST_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# 五个时间戳列（orders 表）
TIME_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

_KAGGLE_DATASET = "olistbr/brazilian-ecommerce"


# ---------------------------------------------------------------------------
# 三段式获取
# ---------------------------------------------------------------------------
def _local_raw_complete(raw_dir: Path) -> bool:
    """检测本地 raw 目录九表是否齐全。"""
    return all((raw_dir / fn).exists() for fn in OLIST_FILES.values())


def _download_via_kagglehub(raw_dir: Path) -> bool:
    """尝试用 kagglehub 下载 Olist 数据集到 raw 目录。

    成功返回 True；任何异常（无凭证、网络失败、包缺失）返回 False，由调用方降级报错。
    """
    try:
        import kagglehub  # 延迟导入：无凭证环境也能让本地分支先跑通
    except ImportError:
        logger.warning("kagglehub 未安装，无法自动下载")
        return False
    try:
        logger.info("本地 raw 缺失，尝试 kagglehub 下载 %s ...", _KAGGLE_DATASET)
        path = kagglehub.dataset_download(_KAGGLE_DATASET)
        src = Path(path)
        raw_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for fn in OLIST_FILES.values():
            # kagglehub 下载目录可能平铺或带子目录，递归查找
            matches = list(src.rglob(fn))
            if matches:
                pd.read_csv(matches[0]).to_csv(raw_dir / fn, index=False)
                copied += 1
        logger.info("kagglehub 下载并拷入 %d/%d 表", copied, len(OLIST_FILES))
        return copied == len(OLIST_FILES)
    except Exception as exc:  # noqa: BLE001 —— 任何失败都降级为「报错给手动指引」
        logger.warning("kagglehub 下载失败：%s", exc)
        return False


def load_olist_raw(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """三段式获取 Olist 九表。

    1) 本地 raw 已存在九表 → 直接读；
    2) 缺失 → kagglehub 自动下载；
    3) 两者都失败 → 抛 RuntimeError，文案含手动下载 URL 与放置路径说明。

    严禁编造数据替代真实数据集。返回 {逻辑名: DataFrame}。
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else C.RAW_OLIST_DIR

    if _local_raw_complete(raw_dir):
        logger.info("本地 raw 九表齐全，直接读取：%s", raw_dir)
    elif _download_via_kagglehub(raw_dir):
        logger.info("kagglehub 下载成功")
    else:
        raise RuntimeError(
            "无法获取 Olist 数据集，且严禁用编造数据替代。\n"
            f"  期望路径：{raw_dir}\n"
            "  手动获取：\n"
            f"    1. 访问 https://www.kaggle.com/datasets/{_KAGGLE_DATASET}\n"
            "    2. 下载并解压，把以下 9 个 CSV 放入上述路径：\n"
            + "\n".join(f"       - {fn}" for fn in OLIST_FILES.values())
            + "\n    或配置 Kaggle 凭证后重试（kagglehub 会自动下载）。"
        )

    return {name: pd.read_csv(raw_dir / fn) for name, fn in OLIST_FILES.items()}


# ---------------------------------------------------------------------------
# 清洗（纯函数，可独立调用、可单测）
# ---------------------------------------------------------------------------
def aggregate_geolocation(geo: pd.DataFrame) -> pd.DataFrame:
    """同邮编多坐标聚合为中位数（坐标是连续量，中位数比均值抗离群）。

    需求措辞为「众数/均值」；连续坐标众数不适用，实现取中位数并登记台账。
    返回 [geolocation_zip_code_prefix, lat, lng]。
    """
    g = geo.dropna(subset=["geolocation_lat", "geolocation_lng"]).copy()
    agg = (
        g.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .reset_index()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})
    )
    return agg


def clean_orders(orders: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """清洗 orders 表，返回 (clean_orders, 分原因剔除统计)。

    规则（顺序执行，逐条计数，一行只计入第一个命中的原因）：
    1. 五个时间戳全部转 datetime；
    2. 剔除 order_status != 'delivered'；
    3. 剔除关键时间戳缺失（下单/实际送达/预计送达任一为空）；
    4. 剔除时间逻辑矛盾：实际送达早于下单、发货(carrier)早于下单、审单早于下单、
       实际送达早于发货。
    """
    reasons: dict[str, int] = {}
    df = orders.copy()
    n0 = len(df)
    reasons["input_rows"] = n0

    # 1. 时间转 datetime
    for col in TIME_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # 2. 仅保留 delivered
    mask_delivered = df["order_status"] == "delivered"
    reasons["dropped_not_delivered"] = int((~mask_delivered).sum())
    df = df[mask_delivered]

    # 3. 关键时间戳缺失（下单、实际送达、预计送达为延迟率/时效计算必需）
    key_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    mask_missing = df[key_cols].isna().any(axis=1)
    reasons["dropped_missing_timestamp"] = int(mask_missing.sum())
    df = df[~mask_missing]

    # 4. 时间逻辑矛盾
    purchase = df["order_purchase_timestamp"]
    delivered = df["order_delivered_customer_date"]
    carrier = df["order_delivered_carrier_date"]
    approved = df["order_approved_at"]
    contra = (
        (delivered < purchase)  # 实际送达早于下单
        | (carrier.notna() & (carrier < purchase))  # 发货早于下单
        | (approved.notna() & (approved < purchase))  # 审单早于下单
        | (carrier.notna() & (delivered < carrier))  # 实际送达早于发货
    )
    reasons["dropped_time_contradiction"] = int(contra.sum())
    df = df[~contra]

    reasons["output_rows"] = len(df)
    return df.reset_index(drop=True), reasons


def build_clean_orders(
    orders_clean: pd.DataFrame,
    customers: pd.DataFrame,
    geo_agg: pd.DataFrame,
) -> pd.DataFrame:
    """clean_orders 关联客户州与聚合坐标，并计算时效字段。

    输出列含：order_id、customer_state、五时间戳、履约总时长/出库响应/在途时长（天）、
    is_late（实际送达 > 预计送达）、lat/lng（按客户邮编前缀关联）。
    """
    df = orders_clean.merge(
        customers[["customer_id", "customer_zip_code_prefix", "customer_state"]],
        on="customer_id",
        how="left",
    )
    df = df.merge(
        geo_agg.rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix"}),
        on="customer_zip_code_prefix",
        how="left",
    )
    # 时效字段（天）
    df["fulfillment_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    df["outbound_response_days"] = (
        df["order_delivered_carrier_date"] - df["order_approved_at"]
    ).dt.total_seconds() / 86400
    df["transit_days"] = (
        df["order_delivered_customer_date"] - df["order_delivered_carrier_date"]
    ).dt.total_seconds() / 86400
    # 延迟标记：实际送达 > 预计送达
    df["is_late"] = df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    return df


def build_order_items_clean(
    order_items: pd.DataFrame, products: pd.DataFrame, orders_clean: pd.DataFrame
) -> pd.DataFrame:
    """order_items 关联 product 重量/体积，并仅保留属于 clean_orders 的订单行。"""
    valid_orders = set(orders_clean["order_id"])
    df = order_items[order_items["order_id"].isin(valid_orders)].copy()
    prod = products[
        ["product_id", "product_category_name", "product_weight_g",
         "product_length_cm", "product_height_cm", "product_width_cm"]
    ].copy()
    prod["product_volume_cm3"] = (
        prod["product_length_cm"] * prod["product_height_cm"] * prod["product_width_cm"]
    )
    df = df.merge(prod, on="product_id", how="left")
    return df


def compute_delay_rate(clean_orders_df: pd.DataFrame) -> dict:
    """Olist 真实延迟率 = 实际送达 > 预计送达的订单占比（与真实准时交付率互补）。"""
    n = len(clean_orders_df)
    late = int(clean_orders_df["is_late"].sum())
    on_time = n - late
    return {
        "n_orders": n,
        "n_late": late,
        "n_on_time": on_time,
        "delay_rate": round(late / n, 4) if n else None,
        "on_time_rate": round(on_time / n, 4) if n else None,
    }


def fit_weight_volume_distribution(order_items_clean: pd.DataFrame) -> dict:
    """对非空且 >0 的商品重量(g)与体积(cm³)拟合对数正态经验分布，保存参数+分位数。

    供数据层 D（配送情景）抽样使用。返回 {weight_g:{...}, volume_cm3:{...}}。
    """
    out: dict[str, dict] = {}
    for col in ("product_weight_g", "product_volume_cm3"):
        s = pd.to_numeric(order_items_clean[col], errors="coerce").dropna()
        s = s[s > 0]
        if len(s) < 10:
            out[col] = {"n": int(len(s)), "fitted": False}
            continue
        # lognorm.fit：s = shape, loc, scale；数据>0 时 loc≈0
        shape, loc, scale = stats.lognorm.fit(s, floc=0)
        qs = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95]).to_dict()
        out[col] = {
            "n": int(len(s)),
            "fitted": True,
            "lognorm_shape_sigma": round(float(shape), 4),
            "lognorm_scale_exp_mu": round(float(scale), 4),  # scale = exp(mu)
            "lognorm_mu": round(float(np.log(scale)), 4),
            "mean": round(float(s.mean()), 2),
            "std": round(float(s.std()), 2),
            "quantiles": {f"P{int(k*100)}": round(float(v), 2) for k, v in qs.items()},
        }
    return out


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_clean(
    raw_dir: Path | None = None, out_dir: Path | None = None
) -> dict[str, Path]:
    """端到端清洗入口：获取 → 清洗 → 关联 → 拟合分布 → 算延迟率 → 落盘。

    返回全部产物路径。out_dir 默认 processed/olist/（测试可传 tmp_path）。
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else C.RAW_OLIST_DIR
    out_dir = Path(out_dir) if out_dir is not None else C.PROCESSED_DIR / "olist"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_olist_raw(raw_dir)
    logger.info("九表读取完成：%s", {k: len(v) for k, v in raw.items()})

    geo_agg = aggregate_geolocation(raw["geolocation"])
    orders_clean, reasons = clean_orders(raw["orders"])
    clean_orders_df = build_clean_orders(orders_clean, raw["customers"], geo_agg)
    items_clean = build_order_items_clean(
        raw["order_items"], raw["products"], orders_clean
    )

    delay = compute_delay_rate(clean_orders_df)
    dist = fit_weight_volume_distribution(items_clean)

    # 落盘清洗结果
    paths: dict[str, Path] = {}
    paths["clean_orders"] = out_dir / "clean_orders.csv"
    paths["order_items_clean"] = out_dir / "order_items_clean.csv"
    clean_orders_df.to_csv(paths["clean_orders"], index=False)
    items_clean.to_csv(paths["order_items_clean"], index=False)

    # 落盘清洗统计（行数对比 + 剔除原因）
    clean_stats = {
        "before_after": {
            "orders_input": reasons["input_rows"],
            "orders_output": reasons["output_rows"],
            "order_items_input": len(raw["order_items"]),
            "order_items_output": len(items_clean),
        },
        "drop_reasons": {
            k: v for k, v in reasons.items() if k.startswith("dropped_")
        },
        "geolocation": {
            "raw_rows": len(raw["geolocation"]),
            "aggregated_zip_codes": len(geo_agg),
            "aggregation_method": "median（坐标连续量，中位数抗离群；众数不适用）",
        },
    }
    paths["clean_stats"] = out_dir / "clean_stats.json"
    _write_json(clean_stats, paths["clean_stats"])

    # 落盘两项下游标定产物
    paths["weight_volume_dist"] = out_dir / "weight_volume_distribution.json"
    _write_json(dist, paths["weight_volume_dist"])
    paths["delay_rate"] = out_dir / "olist_delay_rate.json"
    _write_json(delay, paths["delay_rate"])

    # Markdown 清洗报告
    md = [
        "# 数据层 B：Olist 清洗报告（真实观测）",
        "",
        "## 清洗前后行数",
        f"- orders：{reasons['input_rows']:,} → {reasons['output_rows']:,}",
        f"- order_items：{len(raw['order_items']):,} → {len(items_clean):,}",
        "",
        "## 剔除原因统计",
    ]
    md += [f"- {k}：{v:,}" for k, v in clean_stats["drop_reasons"].items()]
    md += [
        "",
        "## Olist 真实延迟率（供在途异常概率标定）",
        f"- 订单数：{delay['n_orders']:,}",
        f"- 延迟订单：{delay['n_late']:,}",
        f"- **真实延迟率：{delay['delay_rate']:.2%}**（真实准时交付率 {delay['on_time_rate']:.2%}）",
        "",
        "## 商品重量/体积经验分布（对数正态拟合，供数据层 D 抽样）",
    ]
    for col, d in dist.items():
        if d.get("fitted"):
            md.append(
                f"- {col}：n={d['n']:,}，μ={d['lognorm_mu']}，σ={d['lognorm_shape_sigma']}，"
                f"中位={d['quantiles']['P50']}，均值={d['mean']}"
            )
    (out_dir / "clean_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    paths["clean_report_md"] = out_dir / "clean_report.md"

    logger.info(
        "清洗完成：orders %d→%d，真实延迟率 %.2f%%，重量分布中位 %s g",
        reasons["input_rows"], reasons["output_rows"],
        delay["delay_rate"] * 100, dist["product_weight_g"]["quantiles"]["P50"],
    )
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = run_clean()
    print("\n=== 数据层 B 产物 ===")
    for k, v in paths.items():
        print(f"  {k:22s} -> {v}")


if __name__ == "__main__":
    main()
