"""车队成本模型：自营双模式（柴油自购 / 纯电租赁）TCO + 货拉拉外包对标。

**一个概念一个家。** 收口前「车队 TCO」散在三处：成本原语住在 `config.py`（一个常量表），
货拉拉计价与 TCO 分析住在 `gen_delivery_data.py`（数据层 D 的**生成器**），外包逐趟对照与
敏感性住在 `transport_decisions.py`。于是改一个能源价格档位要动三个文件，而
`transport_decisions.sensitivity_table` 只是一行转发到生成器里的那份实现——interface 与
implementation 等高，是典型的 shallow。

**边界：`config` 声明数值，本模块定义模型。** 全部参数（司机工资 / 油价电价 / 租金 /
货拉拉计价规则 / 敏感性档位 / 里程网格）仍留在 `config.py`，那是本项目「唯一魔法数来源」
的约定；本模块只负责怎么用它们算。数据生成器不再持有计价规则。

数据类别：情景假设（成本参数锚点见 `data_sources_ledger.md` 第 4 节，ADR-0009）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C


# ---------------------------------------------------------------------------
# 成本原语：日固定成本 / 公里变动成本 / 日总成本 / 盈亏平衡里程
# ---------------------------------------------------------------------------
def fixed_per_day(
    mode: str,
    *,
    driver_wage: str = "mid",
    rent: str = "mid",
) -> float:
    """返回某自营模式在给定敏感性档位下的日固定成本（元/工作日）。

    mode: "diesel"（柴油自购）或 "ev"（纯电租赁）
    driver_wage: 司机工资档位 low/mid/high
    rent: 租金档位 low/mid/high（仅 ev 用；diesel 无租金，忽略）

    柴油日固定 = 折旧 + 保险 + 司机工资
    纯电日固定 = 租金 + 司机工资

    三个敏感性参数各档独立传入，支持 one-at-a-time 分析（变一个、其余留 mid）。
    """
    wage = C.DRIVER_WAGE_PER_DAY[driver_wage]
    if mode == "diesel":
        return C.DIESEL_FIXED_EX_WAGE + wage
    if mode == "ev":
        return C.EV_RENT_PER_DAY[rent] + wage
    raise ValueError(f"未知模式: {mode}（应为 'diesel' 或 'ev'）")


def per_km(mode: str, *, energy_price: str = "mid") -> float:
    """返回某自营模式在给定能源价格档位下的公里变动成本（元/km）。

    柴油 = 燃油 + 尿素 + 维保；纯电 = 电费。
    energy_price: 能源价格档位 low/mid/high（柴油作用于油价、纯电作用于电价）。
    """
    if mode == "diesel":
        return C.DIESEL_FUEL_PER_KM[energy_price] + C.DIESEL_UREA_PER_KM + C.DIESEL_MAINTENANCE_PER_KM
    if mode == "ev":
        return C.EV_ELEC_PER_KM[energy_price]
    raise ValueError(f"未知模式: {mode}（应为 'diesel' 或 'ev'）")


def daily_total(
    mode: str,
    daily_km: float,
    *,
    driver_wage: str = "mid",
    energy_price: str = "mid",
    rent: str = "mid",
) -> float:
    """日总成本 = 日固定成本 + 日行驶里程 × 公里变动成本（元）。

    用于绘制「日行驶里程—日总成本」曲线与求盈亏平衡里程（ADR-0009）。
    三个敏感性档位独立透传，支持 one-at-a-time 分析。
    """
    fixed = fixed_per_day(mode, driver_wage=driver_wage, rent=rent)
    var = per_km(mode, energy_price=energy_price)
    return fixed + daily_km * var


def breakeven_km(
    *,
    driver_wage: str = "mid",
    energy_price: str = "mid",
    rent: str = "mid",
) -> float:
    """柴油自购 vs 纯电租赁的盈亏平衡日里程（km）。

    令两模式日总成本相等求解里程：
    (固定_ev − 固定_diesel) / (变动_diesel − 变动_ev)
    低于此里程柴油更省，高于此里程纯电更省（因纯电公里变动成本低）。
    注：司机工资对两模式同额，相减抵消，不影响盈亏平衡里程；
    租金（ev 固定）与能源价格（两模式变动）影响结果。
    """
    fixed_diff = fixed_per_day("ev", driver_wage=driver_wage, rent=rent) - fixed_per_day(
        "diesel", driver_wage=driver_wage
    )
    var_diff = per_km("diesel", energy_price=energy_price) - per_km(
        "ev", energy_price=energy_price
    )
    if var_diff == 0:
        raise ZeroDivisionError("两模式公里变动成本相等，无盈亏平衡点")
    return fixed_diff / var_diff


def cheaper_mode(diesel_total: float, ev_total: float) -> str:
    """两模式中较省者。**模式选择只允许这一个实现**——散在各处的 `np.where` 版会漂移。"""
    return "ev" if ev_total <= diesel_total else "diesel"


# ---------------------------------------------------------------------------
# 货拉拉外包计价（仅作决策对照，不进自营车队成本；ADR-0009）
# ---------------------------------------------------------------------------
def huolala_cost(distance_km: float, n_stops: int) -> float:
    """货拉拉外包单趟报价（元）。

    计价规则读 config（起步价含前 5km → 6–BAND1_END 按元/km → 超出按区间中值 →
    超出免费点数按元/点），参数全部登记为情景假设（ADR-0009）。
    """
    cost = C.HUOLALA_START_FEE
    if distance_km > C.HUOLALA_START_KM:
        band1_km = min(distance_km, C.HUOLALA_BAND1_END_KM) - C.HUOLALA_START_KM
        cost += band1_km * C.HUOLALA_RATE_6_25
    if distance_km > C.HUOLALA_BAND1_END_KM:
        cost += (distance_km - C.HUOLALA_BAND1_END_KM) * float(np.mean(C.HUOLALA_RATE_26_PLUS))
    cost += max(0, n_stops - C.HUOLALA_FREE_STOPS) * C.HUOLALA_EXTRA_STOP_FEE
    return round(float(cost), 2)


def huolala_crossover_km(n_stops: int, mode: str = "diesel", hi: float = 400.0) -> float | None:
    """自营与货拉拉的单趟成本平价里程（km）；区间内无交点则返回 None。

    在同一行程画像（点位数固定）下扫里程网格找符号翻转，再二分细化。
    """
    def f(km: float) -> float:
        return daily_total(mode, km) - huolala_cost(km, n_stops)

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


# ---------------------------------------------------------------------------
# TCO 分析：曲线 + 盈亏平衡 + 三档敏感性 + 货拉拉对照 + 模式建议
# ---------------------------------------------------------------------------
def tco_analysis(ref_km: float | None = None) -> dict:
    """构建双模式 TCO 成本模型全部结论（需求 21/22）。

    产出：① 双模式「日总成本 = 日固定 + 里程 × 变动」曲线；② 柴油自购 vs 纯电租赁
    的盈亏平衡里程；③ 司机工资 / 能源价格 / 租金三档 one-at-a-time 敏感性；
    ④ 自营（双模式）vs 货拉拉单趟成本对照表与模式建议。
    """
    ref_km = C.TCO_REFERENCE_DAILY_KM if ref_km is None else ref_km

    curves = {}
    for mode in ("diesel", "ev"):
        curves[mode] = {
            "daily_fixed": round(float(fixed_per_day(mode)), 2),
            "per_km": round(float(per_km(mode)), 4),
            "mileage_km": [float(k) for k in C.TCO_MILEAGE_GRID_KM],
            "daily_total_cost": [
                round(float(daily_total(mode, k)), 2) for k in C.TCO_MILEAGE_GRID_KM
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
                "diesel_daily_cost": round(float(daily_total("diesel", ref_km, **kw)), 2),
                "ev_daily_cost": round(float(daily_total("ev", ref_km, **kw)), 2),
                "breakeven_km": round(float(breakeven_km(**kw)), 2),
            }

    # 自营 vs 货拉拉对照：比较基准 = 单车单日 1 趟（日固定成本全额摊到该趟）
    comparison = []
    for prof in C.HUOLALA_REFERENCE_PROFILES:
        km, stops = float(prof["distance_km"]), int(prof["stops"])
        diesel, ev = float(daily_total("diesel", km)), float(daily_total("ev", km))
        hl = huolala_cost(km, stops)
        best_self, best_mode = min(diesel, ev), cheaper_mode(diesel, ev)
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
                "huolala_crossover_km": huolala_crossover_km(stops, best_mode),
            }
        )

    breakeven = float(breakeven_km())
    diesel_at_ref = float(daily_total("diesel", ref_km))
    ev_at_ref = float(daily_total("ev", ref_km))
    daily_gap = diesel_at_ref - ev_at_ref
    recommendation = {
        # 模式选择走 `cheaper_mode` 这一个口径。原先这里写的是「参考里程是否越过了盈亏平衡点」，
        # 与 `cheaper_mode` 是两套规则，且在**恰好相等**处给出相反答案（这里判 diesel，
        # `cheaper_mode` 判 ev）——而本模块的 docstring 宣称模式选择只允许一个实现。
        "recommended_mode": cheaper_mode(diesel_at_ref, ev_at_ref),
        "breakeven_km": round(breakeven, 2),
        "reference_daily_km": ref_km,
        "daily_saving_vs_diesel": round(daily_gap, 2),
        "rationale": (
            f"单车日均里程超过 {breakeven:.1f} km 时纯电租赁日总成本低于柴油自购"
            f"（纯电公里变动成本 {per_km('ev'):.2f} 元/km 显著低于柴油 "
            f"{per_km('diesel'):.2f} 元/km，代价是日固定成本高 "
            f"{fixed_per_day('ev') - fixed_per_day('diesel'):.0f} 元）。"
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
        "source": "参数锚点=需求文档 2026-09 市场检索，登记于 data_sources_ledger.md 第 4 节（ADR-0009；具体 URL 已于 2026-09-15 逐页核验回填）",
        "reference_daily_km": ref_km,
        "curves": curves,
        "breakeven_km": {"km": round(breakeven, 2), "basis": "柴油自购 vs 纯电租赁 日总成本相等点"},
        "sensitivity": sensitivity,
        "huolala_comparison": comparison,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# 优化后口径：自营 vs 外包逐趟对照
# ---------------------------------------------------------------------------
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
        df[f"self_{mode}_cost"] = daily_total(mode, df["distance_km"])
    df["self_cost"] = df[["self_diesel_cost", "self_ev_cost"]].min(axis=1)
    df["self_best_mode"] = [
        cheaper_mode(d, e) for d, e in zip(df["self_diesel_cost"], df["self_ev_cost"])
    ]
    df["huolala_cost"] = [
        huolala_cost(float(km), int(stops))
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
            "best_mode": cheaper_mode(
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
