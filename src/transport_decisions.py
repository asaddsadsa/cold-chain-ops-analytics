"""模块二（下）：周度异常复盘 + TCO 决策分析 + what-if 预计算（11 号票）。

补齐运输模块的三块**决策产出**（10 号票已交付路由优化核心）：

  ① **周度异常复盘**（需求 41）：按 ISO 周输出异常类型分布与典型异常案例清单，
     并以统计口径复现数据层 D 的三个埋点（周五晚点偏高 / R07 异常偏高 / 雨日晚点偏高）。
  ② **车队 TCO 决策分析**（需求 22/42）：在数据层 D 的成本模型之上叠加 10 号票的
     **优化前后**口径——双模式日/月总成本、盈亏平衡里程、自营（优化后）单趟成本 vs
     货拉拉外包对照、司机工资/能源价格/租金三档敏感性，并给出模式选择建议。
  ③ **what-if 预计算**（需求 46 / ADR-0010）：车辆数 5–20 共 16 档各跑一次代表日 VRPTW，
     结果缓存到 processed，供改善建议页滑块**瞬时切档**；超出网格的组合显式提示
     「需重跑预计算脚本」，不做隐式实时求解。

**与 10 号票的口径衔接**：本模块的「优化前/后」数字不重新求解，而是读 10 号票落盘的
`transport_kpi.json` 与 `trips.csv`——保证两份产物报的是**同一批数字**，评审可以互相核对。
what-if 各档是新求解（那本来就是新计算），其中「档位 = 车队规模」那一档会与 10 号票发表的
结果做一致性自检并如实报出差值（OR-Tools 带时限的启发式搜索不保证跨运行逐位复现）。

运行方式（项目根）：python -m src.transport_decisions
数据类别：情景假设（成本参数与订单需求）+ 真实观测（Olist 派生的异常标定），见台账。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src import gen_delivery_data as GD
from src import transport_optimize as TO

logger = logging.getLogger(__name__)

_L_PER_M3 = 1000

#: 埋点复现的三个靶点：key → (中文标签, 高发组掩码名, 事件掩码名)。
#: 标签与掩码定义**只此一份**——池化检验、逐周复现与报告渲染都从这里取，
#: 避免同一份「哪个靶点叫什么、怎么算」散落在三处而各自漂移。
_EMBEDDED_POINTS: dict[str, dict[str, str]] = {
    "friday_late": {
        "label": "周五「晚点」率 vs 其他工作日",
        "group": "is_friday", "event": "is_late",
    },
    "r07_anomaly": {
        "label": f"{C.HIGH_ANOMALY_REGION} 片区异常率 vs 其他片区",
        "group": "is_r07", "event": "is_anomaly",
    },
    "rainy_late": {
        "label": "雨日「晚点」率 vs 非雨日",
        "group": "is_rainy", "event": "is_late",
    },
}


def _point_flags(anomalies: pd.DataFrame) -> pd.DataFrame:
    """给异常表加上「分组 / 事件」两个布尔列，供两比例检验直接取用。

    列名即 `_EMBEDDED_POINTS` 里登记的掩码名，二者是同一份约定的两面。
    """
    df = anomalies.copy()
    df["is_late"] = df["anomaly_type"] == "晚点"
    df["is_anomaly"] = df["anomaly_type"] != "无异常"
    df["is_friday"] = df["is_friday"].astype(bool)
    df["is_rainy"] = df["rainy"].astype(bool)
    df["is_r07"] = df["region"] == C.HIGH_ANOMALY_REGION
    return df


def _point_test(df: pd.DataFrame, point: str) -> dict:
    """对某个靶点做两比例检验（高发组 vs 对照组），口径与质检摘要同源。"""
    spec = _EMBEDDED_POINTS[point]
    g, e = df[spec["group"]].to_numpy(bool), df[spec["event"]].to_numpy(bool)
    out = GD.two_proportion_test(int(g.sum()), int(e[g].sum()),
                                 int((~g).sum()), int(e[~g].sum()))
    out["label"] = spec["label"]
    out["reproduced"] = bool(out["significant_at_5pct"] and out["rate_1"] > out["rate_0"])
    return out


# ---------------------------------------------------------------------------
# 取数
# ---------------------------------------------------------------------------
def load_decision_context() -> dict:
    """读取决策分析所需的全部输入：10 号票产物 + 数据层 D 异常表。

    10 号票产物缺失即报错并给出运行指引，不降级、不重算——重算会让本模块报出的
    「优化后」与已发表的 `transport_kpi.json` 出现两个版本的数字。
    """
    required = {
        "kpi": C.TRANSPORT_KPI_JSON,
        "trips": C.TRANSPORT_TRIPS_CSV,
        "anomalies": C.DELIVERY_ANOMALIES_CSV,
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少 11 号票输入产物：{missing}\n"
            f"请先运行：python -m src.gen_delivery_data 与 python -m src.transport_optimize"
        )
    ctx = TO.load_transport_context()
    anomalies = pd.read_csv(required["anomalies"], encoding="utf-8-sig", parse_dates=["date"])
    return {
        **ctx,
        "kpi": json.loads(required["kpi"].read_text(encoding="utf-8")),
        "trips": pd.read_csv(required["trips"], encoding="utf-8-sig"),
        "anomalies": anomalies,
    }


# ---------------------------------------------------------------------------
# 一、周度异常复盘
# ---------------------------------------------------------------------------
def _week_label(dates: pd.Series) -> pd.Series:
    """ISO 周标签 `YYYY-Www`（跨年周按 ISO 年，避免把 12 月末归到下一年的第 1 周）。"""
    ic = pd.to_datetime(dates).dt.isocalendar()
    return ic["year"].astype(str) + "-W" + ic["week"].astype(str).str.zfill(2)


def weekly_anomaly_summary(anomalies: pd.DataFrame) -> pd.DataFrame:
    """按 ISO 周汇总异常类型分布与关键比率（需求 41「周度异常复盘」）。

    **分母是全部运单**，不是「有异常的运单」——把「无异常」排除在分母外会把异常率
    算高一截；`share_*` 与 `anomaly_rate` 因此都按运单数算。

    `mean_delay_min` / `mean_handling_min` 只在**有异常**的运单上取均值（无异常的延误
    恒为 0，混进来等于用「无异常占比」稀释延误强度，那是无意义的平均数）；
    `max_delay_min` 则取全周最大，用于快速定位最严重的一单。

    周五 / R07 两列同时给出**该周的**比率与 z 值：逐周样本量（周五约占 1/5、
    R07 约占 1/4）只能支撑量级判断，故 `*_replicates` 与 `min_group_n` 一并落盘，
    把「这周没测出显著」与「埋点不存在」区分开——总体性证据以池化检验为准
    （见 `embedded_point_reproduction`）。
    """
    df = _point_flags(anomalies)
    df["date"] = pd.to_datetime(df["date"])
    df["_week"] = _week_label(df["date"])
    types = [name for name, _ in C.ANOMALY_TYPE_SHARES]

    rows = []
    for week, grp in df.groupby("_week", sort=True):
        n = len(grp)
        anom = grp[grp["is_anomaly"]]
        row = {
            "week": week,
            "week_start": str(grp["date"].min().date()),
            "week_end": str(grp["date"].max().date()),
            "n_shipments": int(n),
            "n_anomalies": int(len(anom)),
            "anomaly_rate": float(len(anom) / n) if n else float("nan"),
            "late_rate": float((grp["anomaly_type"] == "晚点").mean()) if n else float("nan"),
            "mean_delay_min": float(anom["delay_min"].mean()) if len(anom) else 0.0,
            "max_delay_min": float(grp["delay_min"].max()) if n else 0.0,
            "mean_handling_min": float(anom["handling_min"].mean()) if len(anom) else 0.0,
            "temp_compliance_rate": float(grp["temp_compliance_rate"].mean()) if n else float("nan"),
        }
        for name in types + ["无异常"]:
            row[f"share_{name}"] = float((grp["anomaly_type"] == name).mean()) if n else float("nan")
        shares = {name: row[f"share_{name}"] for name in types}
        row["top_type"] = max(shares, key=shares.get) if any(shares.values()) else None
        row.update(_weekly_point_columns(grp))
        rows.append(row)

    out = pd.DataFrame(rows)
    out["anomaly_rate_pp_vs_overall"] = (
        out["anomaly_rate"] - float(df["is_anomaly"].mean())
    ) * 100
    return out


def _weekly_point_columns(grp: pd.DataFrame) -> dict:
    """单周的周五晚点 / R07 异常两列（含该周 z 与是否复现）。

    `grp` 必须是 `_point_flags` 处理过的切片，掩码列已在其中。
    """
    fri = _point_test(grp, "friday_late")
    r07 = _point_test(grp, "r07_anomaly")
    return {
        "friday_late_rate": fri["rate_1"],
        "other_late_rate": fri["rate_0"],
        "friday_late_z": fri["z"],
        "friday_late_replicates": fri["significant_at_5pct"],
        "r07_anomaly_rate": r07["rate_1"],
        "other_anomaly_rate": r07["rate_0"],
        "r07_anomaly_z": r07["z"],
        "r07_anomaly_replicates": r07["significant_at_5pct"],
        # 两组各自的运单数：逐周检验的功效取决于**较小的那一组**，把它落盘，
        # 「这周没测出显著」才有据可判（是埋点弱还是样本少），而不是靠打折估算
        "n_friday": int(grp["is_friday"].sum()),
        "n_r07": int(grp["is_r07"].sum()),
    }


def embedded_point_reproduction(anomalies: pd.DataFrame) -> dict:
    """把数据层 D 的埋点在**周度复盘口径**下复现一遍：池化检验 + 逐周复现计数。

    池化（全 90 天）检验是「埋点能否被下游稳定复现」的**主证据**；逐周计数回答的是
    另一个问题——「每周复盘时能不能当周看出来」。后者受单周样本量限制（周五约占
    1/5、R07 约占 1/4），故连同 `min_group_n` 一起报，不把「当周不显著」说成「埋点不存在」。
    """
    df = _point_flags(anomalies)
    df["_week"] = _week_label(pd.to_datetime(df["date"]))

    points = {key: _point_test(df, key) for key in _EMBEDDED_POINTS}

    weekly = weekly_anomaly_summary(df)
    replication = {}
    for key, col, n_col in (("friday_late", "friday_late_z", "n_friday"),
                            ("r07_anomaly", "r07_anomaly_z", "n_r07")):
        # 该周某组的样本为 0 时（例如小样例里没有 R07 的运单）z 为 None，列会退化成
        # object dtype，直接 .abs() 会抛错；统一按数值缺失处理，缺失即「没测出显著」
        z = pd.to_numeric(weekly[col], errors="coerce")
        sig = (z.abs() >= GD.Z_CRITICAL_5PCT).fillna(False)
        # 功效由**较小的一组**决定，按该周实际运单数算，不用比例估算
        smaller = np.minimum(weekly[n_col], weekly["n_shipments"] - weekly[n_col])
        replication[key] = {
            "n_weeks": int(len(weekly)),
            "n_significant": int(sig.sum()),
            "min_group_n": int(smaller.min()),
            "median_group_n": int(smaller.median()),
        }
    return {
        **points,
        "weekly_replication": {
            **replication,
            "note": (
                "逐周检验的功效受单周样本量限制（min_group_n 即最小的那个对比组运单数）；"
                "埋点是**总体性效应**，可复现证据以池化检验为准，逐周计数说明「每周复盘时"
                "能否当周看出」——当周不显著不等于埋点不存在"
            ),
        },
        "data_category": "过程仿真（数据层 D 埋点）+ 真实观测（总体水平由 Olist 延迟率标定）",
    }


def typical_anomaly_cases(anomalies: pd.DataFrame, per_type: int | None = None) -> pd.DataFrame:
    """典型异常案例清单：每类异常按**严重度**取前 N 条（需求 41）。

    严重度规则显式且不可择优：
      - 晚点 / 故障 / 拥堵 → `delay_min`（延误越长越严重）；
      - 温控波动 → `temp_max_c − 设定点`（温升越高越严重）。

    温控波动单列一套规则是有意的：它的延误被刻意限制在 0–10 分钟（见 config
    `ANOMALY_DELAY_MINUTES`），诊断价值在**温度**而不在时刻，用延误排序会把它排到末尾、
    让温度埋点在案例清单里消失。「无异常」不入选（它不是案例，是背景）。
    """
    per_type = C.ANOMALY_CASES_PER_TYPE if per_type is None else per_type
    df = anomalies[anomalies["anomaly_type"] != "无异常"].copy()
    if df.empty:
        return pd.DataFrame(
            columns=["order_id", "date", "region", "poi_id", "vehicle_id", "anomaly_type",
                     "delay_min", "handling_min", "temp_max_c", "severity", "severity_basis"]
        )
    is_temp = df["anomaly_type"] == "温控波动"
    df["severity"] = np.where(
        is_temp, df["temp_max_c"] - C.CABIN_TEMP_SETPOINT_C, df["delay_min"]
    )
    df["severity_basis"] = np.where(is_temp, "温升 (℃)", "延误 (min)")
    # 同级按 order_id 兜底，保证同一种子下案例清单逐条可复现
    df = df.sort_values(["anomaly_type", "severity", "order_id"],
                        ascending=[True, False, True], kind="stable")
    cols = ["order_id", "date", "region", "poi_id", "vehicle_id", "anomaly_type",
            "delay_min", "handling_min", "temp_max_c", "severity", "severity_basis"]
    return df.groupby("anomaly_type", sort=True).head(per_type)[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 二、TCO 决策分析
# ---------------------------------------------------------------------------
def mode_cost_before_after(trips: pd.DataFrame, n_vehicles_by_plan: dict | None = None) -> dict:
    """双模式「优化前 / 优化后」的日、月总成本（需求 22/42）。

    单趟成本 = 日固定成本 + 该趟里程 × 公里变动成本（纯函数读 config，无硬编码金额）。
    代表日**每台车恰好只跑 1 趟**，因此把日固定成本全额摊入该趟是**精确分摊**而非
    「按 1 趟摊销」的近似。该前提由「趟次数 == 用车数」判定，用车数来自 10 号票的
    KPI 产物（`n_vehicles_by_plan`）——**不能只看 trips.csv**：那里的 trip_id 是趟序号
    而非车牌，趟序号唯一与「一车一趟」是两回事，只看它会得出恒为真的假证据。
    三态如实落盘：True 精确 / False 一车多趟（单趟成本被高估）/ None 无法判定。
    **只要有任何一个方案缺用车数就判 None**——只对「恰好提供了用车数的那几个方案」下结论，
    会把「没校验」说成「已校验为精确」，那是三态里最坏的一种谎。
    """
    plans = sorted(trips["plan"].unique())
    given = {str(p): int(v) for p, v in (n_vehicles_by_plan or {}).items()}
    exact = (
        all(int((trips["plan"] == p).sum()) == given[p] for p in plans)
        if set(given) >= set(plans) else None
    )
    out: dict = {}
    for mode in ("diesel", "ev"):
        fixed = float(C.cost_fixed_per_day(mode))
        per_km = float(C.cost_per_km(mode))
        out[mode] = {}
        for plan in plans:
            sub = trips[trips["plan"] == plan]
            # 成本一律走 config 的成本模型（`daily_total_cost`），不在这里另写一份
            # 「固定 + 里程×变动」——口径只允许有一个实现
            day_total = float(C.daily_total_cost(mode, sub["distance_km"]).sum())
            n_orders = int(sub["n_orders"].sum())
            out[mode][plan] = {
                "day_total": round(day_total, 2),
                "day_per_order": round(day_total / n_orders, 2) if n_orders else None,
                "month_total": round(day_total * C.WORKDAYS_PER_MONTH, 2),
                "month_per_order": round(day_total / n_orders * C.WORKDAYS_PER_MONTH, 2)
                if n_orders else None,
                "n_trips": int(len(sub)),
                "n_orders": n_orders,
                "total_distance_km": round(float(sub["distance_km"].sum()), 2),
                "daily_fixed_per_vehicle": round(fixed, 2),
                "per_km": round(per_km, 4),
            }
        if "baseline" in plans and "optimized" in plans:
            b, o = out[mode]["baseline"], out[mode]["optimized"]
            out[mode]["saving"] = {
                "day": round(b["day_total"] - o["day_total"], 2),
                "month": round(b["month_total"] - o["month_total"], 2),
                "day_per_order": round(b["day_per_order"] - o["day_per_order"], 2),
                "month_per_order": round(
                    b["month_per_order"] - o["month_per_order"], 2
                ),
                # 节省率取正数并显式命名为「成本降幅」：本项目里程/成本类指标在对比表里
                # 用负数表示变好，两套符号混用会让人把「省了 13%」读成「贵了 13%」
                "cost_reduction_pct": round(
                    (b["day_total"] - o["day_total"]) / b["day_total"] * 100, 2
                ) if b["day_total"] else None,
            }
    return {
        "workdays_per_month": C.WORKDAYS_PER_MONTH,
        "per_trip_allocation_exact": exact,
        "n_vehicles_by_plan": {k: int(v) for k, v in (n_vehicles_by_plan or {}).items()} or None,
        "basis": C.TRANSPORT_OUTSOURCE_BASIS,
        **out,
    }


def outsource_comparison(trips: pd.DataFrame) -> dict:
    """自营单趟成本 vs 货拉拉外包报价，按**该趟自己的**里程与点数逐趟对照（需求 22/42）。

    用每趟真实里程/点数而不是固定行程画像，是「叠加优化后」的正确做法：优化把每趟里程
    压短了，自营单趟成本随之下降，外包的相对吸引力因此变化——固定画像（06 号票口径）
    看不到这一点，因为它把里程钉死。

    自营成本给出柴油/纯电两列并取较省者作 `self_cost`，`self_best_mode` 记录省的是哪种；
    这个 min 隐含「车队可按线路自由选模式」，是**自营竞争力的上界**，报告里已注明。
    """
    if trips.empty:
        raise ValueError("trips 为空，无法做外包对照")
    df = trips.copy()
    for mode in ("diesel", "ev"):
        df[f"self_{mode}_cost"] = C.daily_total_cost(mode, df["distance_km"])
    df["self_cost"] = df[["self_diesel_cost", "self_ev_cost"]].min(axis=1)
    # 模式选择走 `_cheaper_mode` 这一个口径，不在此另写一遍 np.where 版的同一规则
    df["self_best_mode"] = [
        _cheaper_mode(d, e) for d, e in zip(df["self_diesel_cost"], df["self_ev_cost"])
    ]
    df["huolala_cost"] = [
        GD.huolala_cost(float(km), int(stops))
        for km, stops in zip(df["distance_km"], df["n_stops"])
    ]
    df["delta_pct_vs_huolala"] = (df["self_cost"] - df["huolala_cost"]) / df["huolala_cost"] * 100
    band = C.HUOLALA_TIE_BAND
    df["verdict"] = [
        "自营更省" if s < h * band else ("外包更省" if h < s * band else "基本持平")
        for s, h in zip(df["self_cost"], df["huolala_cost"])
    ]
    keep = ["trip_id", "plan", "n_stops", "n_orders", "distance_km", "self_diesel_cost",
            "self_ev_cost", "self_cost", "self_best_mode", "huolala_cost",
            "delta_pct_vs_huolala", "verdict"]
    out = df[keep].reset_index(drop=True)

    by_plan: dict[str, dict] = {}
    for plan, sub in out.groupby("plan"):
        n = len(sub)
        cheap = int((sub["verdict"] == "自营更省").sum())
        out_cheap = int((sub["verdict"] == "外包更省").sum())
        by_plan[str(plan)] = {
            "n_trips": int(n),
            "total_self_cost": round(float(sub["self_cost"].sum()), 2),
            "total_self_diesel_cost": round(float(sub["self_diesel_cost"].sum()), 2),
            "total_self_ev_cost": round(float(sub["self_ev_cost"].sum()), 2),
            "total_huolala_cost": round(float(sub["huolala_cost"].sum()), 2),
            "n_trips_self_cheaper": cheap,
            "n_trips_outsource_cheaper": out_cheap,
            "n_trips_tie": int(n - cheap - out_cheap),
            "best_mode": _cheaper_mode(
                float(sub["self_diesel_cost"].sum()), float(sub["self_ev_cost"].sum())
            ),
        }
        tot_h = by_plan[str(plan)]["total_huolala_cost"]
        by_plan[str(plan)]["delta_pct_vs_huolala"] = round(
            (by_plan[str(plan)]["total_self_cost"] - tot_h) / tot_h * 100, 2
        ) if tot_h else None

    primary = "optimized" if "optimized" in by_plan else sorted(by_plan)[0]
    return {
        "basis": C.TRANSPORT_OUTSOURCE_BASIS,
        "primary_plan": primary,
        "per_trip": out,
        "by_plan": by_plan,
        "total_self_cost": by_plan[primary]["total_self_cost"],
        "total_huolala_cost": by_plan[primary]["total_huolala_cost"],
        "n_trips_self_cheaper": by_plan[primary]["n_trips_self_cheaper"],
        "n_trips_outsource_cheaper": by_plan[primary]["n_trips_outsource_cheaper"],
        "n_trips_tie": by_plan[primary]["n_trips_tie"],
    }


def _cheaper_mode(diesel_total: float, ev_total: float) -> str:
    return "ev" if ev_total <= diesel_total else "diesel"


def sensitivity_table(reference_km: float) -> dict:
    """三档敏感性（司机工资 / 能源价格 / 租金），one-at-a-time 变一个、其余留 mid。

    直接复用数据层 D 的成本模型（`gen_delivery_data.build_tco_analysis`），
    不在此另写一份——成本模型只允许有一个实现。
    """
    return GD.build_tco_analysis(reference_km)["sensitivity"]


def build_transport_tco(trips: pd.DataFrame, n_vehicles_by_plan: dict | None = None) -> dict:
    """运输侧 TCO 决策分析汇总（需求 22/42）。

    代表里程取**优化后**的平均单趟里程：优化把趟次里程压短后，「司机工资 / 能源 / 租金」
    三档敏感性应当以新的里程水平来评估，否则敏感性还在描述一个已经不存在的车队。
    """
    mode_costs = mode_cost_before_after(trips, n_vehicles_by_plan)
    outs = outsource_comparison(trips)
    ref_km = float(trips.loc[trips["plan"] == outs["primary_plan"], "distance_km"].mean())
    base = GD.build_tco_analysis(ref_km)
    breakeven = float(C.breakeven_km())
    recommended = _cheaper_mode(
        mode_costs["diesel"][outs["primary_plan"]]["day_total"],
        mode_costs["ev"][outs["primary_plan"]]["day_total"],
    )
    return {
        "data_category": "情景假设（成本参数锚点见 config 与台账第 4 节，ADR-0009）",
        "reference_daily_km": round(ref_km, 2),
        "reference_basis": (
            "代表日**优化后**的单趟平均里程——优化压缩了趟次里程，敏感性须按新的里程水平评估"
        ),
        "mode_costs": mode_costs,
        "curves": base["curves"],
        "breakeven_km": {"km": round(breakeven, 2),
                         "basis": "柴油自购 vs 纯电租赁 日总成本相等点"},
        "sensitivity": base["sensitivity"],
        "outsourcing": outs,
        "recommendation": {
            "recommended_mode": recommended,
            "basis": f"代表日优化后口径（{outs['primary_plan']}）",
            "breakeven_km": round(breakeven, 2),
            "daily_saving_vs_other_mode": round(
                abs(mode_costs["diesel"][outs["primary_plan"]]["day_total"]
                    - mode_costs["ev"][outs["primary_plan"]]["day_total"]), 2
            ),
            "outsourcing_verdict": (
                f"代表日 {outs['by_plan'][outs['primary_plan']]['n_trips']} 趟中，自营更省 "
                f"{outs['n_trips_self_cheaper']} 趟、外包更省 {outs['n_trips_outsource_cheaper']} 趟、"
                f"基本持平 {outs['n_trips_tie']} 趟；全外包需 "
                f"{outs['total_huolala_cost']:,.0f} 元 vs 自营 "
                f"{outs['total_self_cost']:,.0f} 元"
            ),
            "caveats": [
                "自营单趟成本按「每车每日 1 趟」全额摊入日固定成本；该前提在代表日成立"
                "（用车数 = 趟次数），趟次合并时会高估单趟成本；",
                "`self_cost` 取柴油/纯电较省者，隐含「车队可按线路自由选模式」，"
                "是自营竞争力的**上界**，实际单一模式车队应看对应模式那一列；",
                "司机工资对两模式同额、相减抵消，不影响盈亏平衡里程（06 号票已实测）；",
                "货拉拉计价为公开计价规则的情景假设，未含实际议价与高峰期加价。",
            ],
        },
    }


# ---------------------------------------------------------------------------
# 三、what-if 车辆数预计算（ADR-0010）
# ---------------------------------------------------------------------------
def whatif_gears() -> list[int]:
    """预计算的车辆数档位网格（config `WHATIF_VEHICLE_RANGE`，5–20 共 16 档）。"""
    lo, hi = C.WHATIF_VEHICLE_RANGE
    return list(range(int(lo), int(hi) + 1))


#: 每个档位记录的**优化后** KPI 字段集。不可行的档位也保留同一套键（值填 None），
#: 看板读缓存时不因某档无解而缺键。
_PLAN_FIELDS = (
    "n_vehicles_used", "total_distance_km", "time_window_rate", "n_orders_ontime",
    "n_orders_served", "n_orders_unserved", "load_rate_mean", "load_rate_max",
    "mileage_utilization", "cost", "solve_time_limit_sec",
)


def precompute_whatif(
    nodes: pd.DataFrame,
    dist_km: np.ndarray,
    time_min: np.ndarray,
    day_orders: pd.DataFrame,
    fleet: pd.DataFrame,
    delays: dict[str, float],
    *,
    levels: list[int] | None = None,
    time_limit_sec: float | None = None,
) -> dict:
    """车辆数各档在代表日各跑一次 VRPTW + 一次基线贪心，缓存 KPI（ADR-0010）。

    `n_vehicles` 只影响 OR-Tools 的车辆数上限；建模、目标函数、时限与 10 号票完全一致，
    因此档位 = 车队规模那一档可与已发表的优化结果对照（见 `consistency_check`）。

    车太少会**真的无解**（当日货量装不下或时间窗挤不进），此时 `feasible=False` 并记下
    原因——这是 what-if 的**结论本身**（「5 台车跑不了这一天」），不是错误，更不能拿
    一个够不到的解冒充可行。
    """
    levels = whatif_gears() if levels is None else list(levels)
    time_limit_sec = C.WHATIF_TIME_LIMIT_SEC if time_limit_sec is None else time_limit_sec
    rated_kg = float(fleet["rated_payload_kg"].iloc[0])
    rated_l = float(fleet["rated_volume_m3"].iloc[0]) * _L_PER_M3
    common = dict(
        rated_payload_kg=rated_kg,
        rated_volume_l=rated_l,
        service_min=C.TRANSPORT_STOP_SERVICE_MIN,
        depot_open_min=C.TRANSPORT_DEPOT_OPEN_MIN,
        max_route_min=C.TRANSPORT_MAX_ROUTE_MIN,
    )
    total_kg = float(nodes["weight_kg"].sum())

    gears: dict[str, dict] = {}
    for n in levels:
        rec: dict = {"n_vehicles_available": int(n), "feasible": False, "infeasible_reason": None}
        base_routes = TO.nearest_neighbor_routes(nodes, dist_km, time_min, max_vehicles=n, **common)
        rec["baseline"] = _plan_record(base_routes, nodes, day_orders, dist_km, fleet, delays,
                                       time_limit_sec=None)
        opt_routes = TO.solve_vrptw(nodes, dist_km, time_min, n_vehicles=n,
                                    time_limit_sec=time_limit_sec, **common)
        if opt_routes is None:
            # 容量下界能直接判定的，把原因写具体；判不出的（时间窗挤不下）如实说判不出
            capacity_floor = int(np.ceil(total_kg / rated_kg))
            rec["infeasible_reason"] = (
                f"{n} 台车在时限内未求得可行解：当日货量 {total_kg / 1000:.1f} t，"
                f"按额定 {rated_kg / 1000:.1f} t 至少需 {capacity_floor} 台车"
                + ("（**已低于载重下界**，必然无解）" if n < capacity_floor else
                   "（载重下界满足，无解来自容积或时间窗）")
            )
            # 不可行档位保留**同样的字段集**、值填 None：看板读缓存时键一定存在，
            # 不会因为某档无解就 KeyError——「无解」本身是要展示的结论，不是缺数据。
            rec.update({k: None for k in _PLAN_FIELDS})
        else:
            rec.update(_plan_record(opt_routes, nodes, day_orders, dist_km, fleet, delays,
                                    time_limit_sec=time_limit_sec))
            rec["feasible"] = True
        gears[str(n)] = rec
        logger.info("what-if 车辆数 %2d：%s", n,
                    "可行" if rec["feasible"] else f"不可行（{rec['infeasible_reason']}）")

    return {
        "grid": levels,
        "gears": gears,
        "time_limit_sec": time_limit_sec,
        "n_nodes": int(len(nodes)),
        "n_orders": int(len(day_orders)),
        "rated_payload_kg": rated_kg,
        "consistency_check": _consistency_check(gears),
    }


def _plan_record(
    routes: list[dict],
    nodes: pd.DataFrame,
    day_orders: pd.DataFrame,
    dist_km: np.ndarray,
    fleet: pd.DataFrame,
    delays: dict[str, float],
    *,
    time_limit_sec: float | None,
) -> dict:
    """某一档某一策略的 KPI 记录（字段与看板滑块所需一一对应）。"""
    k = TO.transport_kpis(routes, nodes, day_orders, dist_km, fleet, delays)
    return {
        "n_vehicles_used": int(k["n_vehicles"]),
        "total_distance_km": k["total_distance_km"],
        "time_window_rate": k["time_window"]["rate"],
        "n_orders_ontime": k["time_window"]["n_ontime"],
        "n_orders_served": k["time_window"]["n_orders"],
        "n_orders_unserved": k["time_window"]["n_unserved"],
        "load_rate_mean": k["load_rate"]["mean"],
        "load_rate_max": k["load_rate"]["max"],
        "mileage_utilization": k["mileage_utilization"],
        "cost": {m: {"total": k["cost"][m]["total"], "per_order": k["cost"][m]["per_order"]}
                 for m in ("diesel", "ev")},
        "solve_time_limit_sec": time_limit_sec,
    }


def _consistency_check(gears: dict[str, dict]) -> dict:
    """档位 = 车队规模时，预计算结果与 10 号票发表结果的差值。

    OR-Tools 带时限的启发式搜索按**挂钟**截断迭代，同参数多次求解**不保证同解**——
    本机对同一档（15 台）实测出现过 876.15 与 870.16 km 两个结果，相差 **0.68%**
    （4 次观测中 3 次同值）。故这里**记录**差值而不是断言相等，也不设硬阈值：
    差值在 1% 量级属搜索随机性，量级明显变大才说明两处建模已经走偏。
    """
    key = str(int(C.FLEET_SIZE))
    published = None
    if C.TRANSPORT_KPI_JSON.exists() and key in gears:
        kpi = json.loads(C.TRANSPORT_KPI_JSON.read_text(encoding="utf-8"))
        published = float(kpi["optimized"]["total_distance_km"])
    got = gears.get(key)
    if published is None or got is None or not got.get("feasible"):
        return {"gear": int(C.FLEET_SIZE), "distance_km": None, "published_distance_km": published,
                "distance_delta_pct": None,
                "note": "缺少可比对的一档（10 号票产物缺失或该档不可行）"}
    got_km = float(got["total_distance_km"])
    return {
        "gear": int(C.FLEET_SIZE),
        "distance_km": got_km,
        "published_distance_km": published,
        "distance_delta_pct": round((got_km - published) / published * 100, 3) if published else None,
        "note": (
            "档位=车队规模时的 what-if 解 vs 10 号票发表的优化解。两者建模同构、时限相同，"
            "但带时限的搜索按**挂钟**截断迭代，同参数多次求解不保证同解——本机实测同档出现过 "
            "876.15 / 870.16 km 两个结果（相差 0.68%）。故只记录差值、不设硬阈值："
            "1% 量级属搜索随机性，量级明显变大才说明两处建模已经走偏"
        ),
    }


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def _write_report(weekly: pd.DataFrame, repro: dict, tco: dict, whatif: dict,
                  cases: pd.DataFrame, out_path: Path) -> Path:
    mc = tco["mode_costs"]
    outs = tco["outsourcing"]
    lines = [
        "# 模块二（下）：周度异常复盘 + TCO 决策分析 + what-if 预计算",
        "",
        f"- 复盘周期：{weekly['week'].iloc[0]} ~ {weekly['week'].iloc[-1]}（{len(weekly)} 个 ISO 周）",
        f"- 运单总量：{int(weekly['n_shipments'].sum()):,} 单；总体异常率 "
        f"{weekly['n_anomalies'].sum() / weekly['n_shipments'].sum():.2%}",
        "- 数据类别：过程仿真（数据层 D 埋点）+ 情景假设（成本参数）+ 真实观测"
        "（异常总体水平由 Olist 实测延迟率标定）",
        "",
        "## 一、周度异常复盘",
        "",
        "周度表全列见 `anomaly_weekly.csv`。**分母是全部运单**（含无异常的），"
        "否则异常率会被算高；`mean_delay_min` 只在有异常的运单上取均值"
        "（无异常延误恒为 0，混进去等于用「无异常占比」稀释延误强度）。",
        "",
        "| 周 | 运单 | 异常数 | 异常率 | 晚点率 | 平均延误 | 最长延误 | 周五晚点率 | "
        f"{C.HIGH_ANOMALY_REGION} 异常率 | 温控达标 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in weekly.itertuples(index=False):
        lines.append(
            f"| {r.week} | {r.n_shipments} | {r.n_anomalies} | {r.anomaly_rate:.2%} | "
            f"{r.late_rate:.2%} | {r.mean_delay_min:.0f} min | {r.max_delay_min:.0f} min | "
            f"{r.friday_late_rate:.2%} | {r.r07_anomaly_rate:.2%} | {r.temp_compliance_rate:.1%} |"
        )
    lines += [
        "",
        "### 埋点复现（池化检验，90 天全量）",
        "",
        "| 靶点 | 高发组 | 对照组（池化） | 比值 | z | 5% 显著 | 复现 |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in _EMBEDDED_POINTS:
        p = repro[key]
        lines.append(
            f"| {p['label']} | {p['rate_1']:.2%}（n={p['n_1']:,}） | "
            f"{p['rate_0']:.2%}（n={p['n_0']:,}） | {p['ratio']} | {p['z']:+.2f} | "
            f"{'是' if p['significant_at_5pct'] else '否'} | {'成立' if p['reproduced'] else '未复现'} |"
        )
    lines += [
        "",
        "对照组取**池化率**（该组全部运单合并计算），与 z 检验的合并比例标准误同一个口径；"
        f"06 号票质检摘要里的「其他片区率」是 **7 个片区各自率的算术平均**，与本表的 R07 对照组 "
        "略有小数差（5.28% vs 5.29%），是定义不同、不是谁算错了。三处**z 值**完全一致。",
        "",
        f"- 逐周复现计数：周五晚点 {repro['weekly_replication']['friday_late']['n_significant']}"
        f"/{repro['weekly_replication']['friday_late']['n_weeks']} 周显著"
        f"（当周较小分组最少 {repro['weekly_replication']['friday_late']['min_group_n']} 单、"
        f"中位 {repro['weekly_replication']['friday_late']['median_group_n']} 单）、"
        f"{C.HIGH_ANOMALY_REGION} 异常 "
        f"{repro['weekly_replication']['r07_anomaly']['n_significant']}"
        f"/{repro['weekly_replication']['r07_anomaly']['n_weeks']} 周显著"
        f"（较小分组最少 {repro['weekly_replication']['r07_anomaly']['min_group_n']} 单）",
        f"- {repro['weekly_replication']['note']}",
        f"- 全量数值（含逐周 z、分组样本量）另落盘 `{C.TRANSPORT_ANOMALY_POINTS_JSON.name}`，"
        "看板与复现者可直接读取，不必从本报告的散文里摘数字",
        "",
        "### 典型异常案例",
        "",
        f"每条按**严重度**取前 {C.ANOMALY_CASES_PER_TYPE} 名（全量见 `anomaly_cases.csv`）。"
        "晚点/故障/拥堵按延误分钟排序；**温控波动按温升排序**——它的延误被刻意限制在 "
        "0–10 分钟（config `ANOMALY_DELAY_MINUTES`），诊断价值在温度而不在时刻，"
        "用延误排会把温度埋点挤出清单。",
        "",
        "| 类型 | 运单 | 日期 | 片区 | 车辆 | 严重度 | 延误 | 处理 | 最高温 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in cases.itertuples(index=False):
        lines.append(
            f"| {r.anomaly_type} | {r.order_id} | {str(r.date)[:10]} | {r.region} | {r.vehicle_id} | "
            f"{r.severity:.1f} {r.severity_basis} | {r.delay_min:.0f} min | "
            f"{r.handling_min:.0f} min | {r.temp_max_c:.1f} ℃ |"
        )
    lines += [
        "",
        "## 二、车队 TCO 决策分析",
        "",
        f"代表里程：**{tco['reference_daily_km']} km**（{tco['reference_basis']}）",
        "",
        "### 双模式日 / 月总成本（优化前 vs 优化后）",
        "",
        "| 模式 | 方案 | 日总成本 | 月总成本 | 单均日成本 | 趟次 | 里程 |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, label in (("diesel", "柴油自购"), ("ev", "纯电租赁")):
        for plan, plan_label in (("baseline", "优化前"), ("optimized", "优化后")):
            d = mc[mode][plan]
            lines.append(
                f"| {label} | {plan_label} | {d['day_total']:,.0f} 元 | {d['month_total']:,.0f} 元 | "
                f"{d['day_per_order']:.2f} 元/单 | {d['n_trips']} | {d['total_distance_km']:,.1f} km |"
            )
        s = mc[mode].get("saving")
        if s:
            lines.append(
                f"| {label} | **节省** | **{s['day']:,.0f} 元（成本降 {s['cost_reduction_pct']:.2f}%）** | "
                f"**{s['month']:,.0f} 元** | {s['day_per_order']:.2f} 元/单 | — | — |"
            )
    if mc["per_trip_allocation_exact"] is True:
        exact_note = (
            f"- 精确分摊前提**成立**（趟次数 = 用车数 "
            f"{mc['n_vehicles_by_plan']}），日固定成本全额摊入该趟是精确值而非近似"
        )
    elif mc["per_trip_allocation_exact"] is False:
        exact_note = (
            "- ⚠️ 精确分摊前提**不成立**（存在一车多趟），单趟自营成本被**高估**，"
            "下方逐趟对照会偏保守"
        )
    else:
        exact_note = "- 精确分摊前提**未判定**（未提供车队规模），单趟成本按每车每日 1 趟摊销"
    lines += [
        "",
        f"- 月口径 = 日 × {mc['workdays_per_month']} 工作日（`WORKDAYS_PER_MONTH`）",
        exact_note,
        f"- 盈亏平衡里程：**{tco['breakeven_km']['km']:.1f} km/日**（{tco['breakeven_km']['basis']}）",
        f"- 模式建议：**{tco['recommendation']['recommended_mode']}**（{tco['recommendation']['basis']}）",
        "",
        "### 自营（逐趟真实里程）vs 货拉拉外包",
        "",
        tco["recommendation"]["outsourcing_verdict"] + "。",
        "",
        "| 方案 | 趟次 | 自营合计 | 全外包合计 | 差额 | 自营更省 | 外包更省 | 持平 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for plan, blk in outs["by_plan"].items():
        lines.append(
            f"| {plan} | {blk['n_trips']} | {blk['total_self_cost']:,.0f} 元 | "
            f"{blk['total_huolala_cost']:,.0f} 元 | {blk['delta_pct_vs_huolala']:+.1f}% | "
            f"{blk['n_trips_self_cheaper']} | {blk['n_trips_outsource_cheaper']} | {blk['n_trips_tie']} |"
        )
    lines += ["", "注意事项："] + [f"- {c}" for c in tco["recommendation"]["caveats"]]
    lines += [
        "",
        "### 三档敏感性（one-at-a-time，变一个参数、其余留 mid）",
        "",
        "| 参数 | 档位 | 柴油日成本 | 纯电日成本 | 盈亏平衡里程 |",
        "|---|---|---|---|---|",
    ]
    for param, levels in tco["sensitivity"].items():
        for lv, d in levels.items():
            lines.append(
                f"| {param} | {lv} | {d['diesel_daily_cost']:,.0f} 元 | "
                f"{d['ev_daily_cost']:,.0f} 元 | {d['breakeven_km']:.1f} km |"
            )
    lines += [
        "",
        "## 三、what-if 车辆数预计算（ADR-0010）",
        "",
        f"网格：车辆数 **{whatif['grid'][0]}–{whatif['grid'][-1]}** 共 {len(whatif['grid'])} 档，"
        f"每档 {whatif['time_limit_sec']:.0f} 秒时限（代表日 {whatif['n_orders']} 单 / "
        f"{whatif['n_nodes']} 点）。看板滑块只读本缓存，**不实时求解**；"
        "超出网格的组合提示「需重跑预计算脚本」，不插值外推。",
        "",
        "| 车辆数 | 优化用车 | 总里程 | 时间窗达成率 | 满载率均值 | 柴油单均 | 纯电单均 | 未服务单 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for gear, rec in whatif["gears"].items():
        if not rec["feasible"]:
            lines.append(f"| {gear} | — | — | — | — | — | — | **不可行** |")
            continue
        lines.append(
            f"| {gear} | {rec['n_vehicles_used']} | {rec['total_distance_km']:,.1f} km | "
            f"{rec['time_window_rate']:.2%} | {rec['load_rate_mean']:.1%} | "
            f"{rec['cost']['diesel']['per_order']:.2f} 元 | {rec['cost']['ev']['per_order']:.2f} 元 | "
            f"{rec['n_orders_unserved']} |"
        )
    infeasible = [g for g, r in whatif["gears"].items() if not r["feasible"]]
    if infeasible:
        lines += ["", "不可行档位的原因："]
        lines += [f"- 车辆数 {g}：{whatif['gears'][g]['infeasible_reason']}" for g in infeasible]
    chk = whatif["consistency_check"]
    chk_km = "无解" if chk["distance_km"] is None else f"{chk['distance_km']:,.1f}"
    pub_km = "缺失" if chk["published_distance_km"] is None else f"{chk['published_distance_km']:,.1f}"
    delta = "—" if chk["distance_delta_pct"] is None else f"{chk['distance_delta_pct']:+.3f}"
    lines += [
        "",
        f"**一致性自检**：档位 = 车队规模（{chk['gear']} 台）时预计算里程 {chk_km} km vs "
        f"10 号票发表 {pub_km} km，差 {delta}%。{chk['note']}",
        "",
        "## 四、产物",
        "",
        f"- `{C.TRANSPORT_ANOMALY_WEEKLY_CSV.name}` / `{C.TRANSPORT_ANOMALY_CASES_CSV.name}`："
        "周度复盘表与典型案例清单",
        f"- `{C.TRANSPORT_ANOMALY_POINTS_JSON.name}`：埋点复现全量结论"
        "（三靶点池化检验 + 逐周复现计数与分组样本量）",
        f"- `{C.TRANSPORT_TCO_JSON.name}`：TCO 全量结论（曲线 / 盈亏平衡 / 外包逐趟 / 敏感性）",
        f"- `{C.TRANSPORT_WHATIF_JSON.name}`：what-if 16 档缓存（改善建议页滑块数据源）",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------
def run_all(
    out_dir: Path | None = None,
    *,
    whatif_time_limit_sec: float | None = None,
    levels: list[int] | None = None,
) -> dict:
    """跑通模块二（下）全流程并落盘。

    `levels` / `whatif_time_limit_sec` 仅用于测试压缩耗时，默认按 config 的 16 档与 30 秒。
    """
    ctx = load_decision_context()
    anomalies, trips = ctx["anomalies"], ctx["trips"]
    orders, fleet = ctx["orders"], ctx["vehicles"]
    dist_km, time_min = ctx["dist_km"], ctx["time_min"]

    logger.info("周度异常复盘：%d 单 / %d 周…", len(anomalies),
                anomalies["date"].dt.isocalendar()["week"].nunique())
    weekly = weekly_anomaly_summary(anomalies)
    repro = embedded_point_reproduction(anomalies)
    cases = typical_anomaly_cases(anomalies)

    logger.info("TCO 决策分析…")
    n_veh = {p: int(ctx["kpi"][p]["n_vehicles"]) for p in ("baseline", "optimized")
             if p in ctx["kpi"] and "n_vehicles" in ctx["kpi"][p]}
    tco = build_transport_tco(trips, n_veh)

    rep_day = pd.Timestamp(ctx["kpi"]["representative_day"])
    day_orders = orders[orders["date"] == rep_day].copy()
    nodes = TO.aggregate_demand(day_orders)
    sub_d, sub_t = TO.node_matrix(nodes, dist_km), TO.node_matrix(nodes, time_min)

    gear_list = whatif_gears() if levels is None else list(levels)
    limit = C.WHATIF_TIME_LIMIT_SEC if whatif_time_limit_sec is None else whatif_time_limit_sec
    logger.info("what-if 预计算：车辆数 %s 共 %d 档，每档 %.0f 秒…",
                f"{gear_list[0]}–{gear_list[-1]}", len(gear_list), limit)
    whatif = precompute_whatif(
        nodes, sub_d, sub_t, day_orders, fleet, ctx["order_delay_min"],
        levels=gear_list, time_limit_sec=limit,
    )
    whatif["representative_day"] = str(rep_day.date())
    whatif["data_category"] = "预计算结果（非实时求解，ADR-0010）"
    whatif["note"] = (
        "车辆数 what-if 为**离线预计算**缓存，看板滑块只切档、不实时求解；"
        "超出网格的组合需重跑预计算脚本"
    )

    out_dir = C.TRANSPORT_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out_dir / C.TRANSPORT_ANOMALY_WEEKLY_CSV.name, index=False, encoding="utf-8-sig")
    cases.to_csv(out_dir / C.TRANSPORT_ANOMALY_CASES_CSV.name, index=False, encoding="utf-8-sig")
    (out_dir / C.TRANSPORT_ANOMALY_POINTS_JSON.name).write_text(
        json.dumps(_jsonable(repro), ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    (out_dir / C.TRANSPORT_TCO_JSON.name).write_text(
        json.dumps(_tco_for_json(tco), ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    (out_dir / C.TRANSPORT_WHATIF_JSON.name).write_text(
        json.dumps(_jsonable(whatif), ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    report = _write_report(weekly, repro, tco, whatif, cases, out_dir / C.TRANSPORT_TCO_MD.name)
    return {
        "weekly": weekly, "reproduction": repro, "cases": cases, "tco": tco,
        "whatif": whatif, "anomalies": anomalies, "report": str(report),
    }


def _tco_for_json(tco: dict) -> dict:
    """TCO 结论的 JSON 形态：逐趟明细表转成记录数组（DataFrame 不可直接序列化）。

    CSV 落盘与 JSON 落盘用的是同一份 `per_trip`，故看板读 JSON、复现者读 CSV
    看到的逐趟数字必然一致，不存在「两个版本」。
    """
    out = _jsonable({k: v for k, v in tco.items() if k != "outsourcing"})
    outs = {k: v for k, v in tco["outsourcing"].items() if k != "per_trip"}
    out["outsourcing"] = _jsonable(outs)
    out["outsourcing"]["per_trip"] = _jsonable(
        tco["outsourcing"]["per_trip"].to_dict(orient="records")
    )
    return out


def _jsonable(obj):
    """把 DataFrame / numpy 标量转成可 json 序列化的结构（产物必须能被看板直接读）。"""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="模块二（下）：周度复盘 + TCO + what-if 预计算")
    parser.add_argument("--whatif-seconds", type=float, default=None,
                        help=f"what-if 每档求解时限（秒），默认 {C.WHATIF_TIME_LIMIT_SEC:.0f}")
    args = parser.parse_args()
    out = run_all(whatif_time_limit_sec=args.whatif_seconds)

    w, t = out["weekly"], out["tco"]
    print(f"\n=== 周度异常复盘：{len(w)} 周，共 {int(w['n_shipments'].sum()):,} 单 ===")
    for key, label in (("friday_late", "周五晚点"), ("r07_anomaly", f"{C.HIGH_ANOMALY_REGION} 异常"),
                       ("rainy_late", "雨日晚点")):
        p = out["reproduction"][key]
        print(f"  {label:8s} {p['rate_1']:.2%} vs {p['rate_0']:.2%}  z={p['z']:+.2f}  "
              f"{'显著' if p['significant_at_5pct'] else '不显著'}")
    print(f"\n=== TCO：代表里程 {t['reference_daily_km']} km ===")
    for mode, label in (("diesel", "柴油自购"), ("ev", "纯电租赁")):
        s = t["mode_costs"][mode]["saving"]
        print(f"  {label}：优化前 {t['mode_costs'][mode]['baseline']['day_total']:,.0f} 元/日 → "
              f"优化后 {t['mode_costs'][mode]['optimized']['day_total']:,.0f} 元/日 "
              f"（月省 {s['month']:,.0f} 元）")
    print(f"  盈亏平衡里程 {t['breakeven_km']['km']:.1f} km/日；"
          f"建议模式 {t['recommendation']['recommended_mode']}")
    print(f"  外包对照：{t['recommendation']['outsourcing_verdict']}")
    chk = out["whatif"]["consistency_check"]
    print(f"\n=== what-if：{len(out['whatif']['grid'])} 档预计算完成；"
          f"一致性自检 @{chk['gear']} 台 差值 {chk['distance_delta_pct']}% ===")
    print(f"  产物目录：{Path(out['report']).parent}")


if __name__ == "__main__":
    main()
