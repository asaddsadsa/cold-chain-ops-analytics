"""改善建议页（看板模块三 · 15 号票，需求 45）。

一屏回答「该改什么、值不值得改」：把 `report/pdca_report.md` 的 P→D→C→A 逐条做成
可交互版本——「问题诊断 → 改善动作 → 预期收益」表由 `data/processed/` 产物**自动汇总**，
每个数字都标出证据文件与数据类别（真实观测 / 学术基准 / 过程仿真 / 情景假设）；
再用四个 what-if 控件让人**亲手验证**改善逻辑是否成立：

  ① 车辆数 5–20 滑块：读 ADR-0010 的 16 档**预计算缓存**，瞬时切档（绝不实时求解）；
  ② ABC 累计占比阈值滑块：调 `warehouse_kpi.abc_classify` **实时重算**分类并联动帕累托；
  ③ 拣货员人数档位：联动 SimPy 仿真缓存（4/5/6 档）；
  ④ 动力模式切换 + 司机工资/能源价格/租金三档敏感性：联动 TCO 曲线与盈亏平衡里程
     （敏感性是 **one-at-a-time**，变一个、其余留 mid——页面上写死了这一点）。

两条纪律（照 12 号票基建）：

- **所有读数走 `src.dashboard.data` 或既有纯函数**，页面里没有任何硬编码的业务数字；
  算得慢的 ABC 重算用 `st.cache_data` 缓存，且**缓存键包含两个阈值**，否则拖动不生效。
- **本页为全量口径**：车辆 / 仓库 / 代表日的标量读数没有日期、片区、品类维度，
  全局筛选器不改变它们（不装作用户拖了筛选器而数字没动的样子）。

运行：在项目根执行 `streamlit run app.py`，左侧页面选择「改善建议」。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config as C
from src import costing
from src import warehouse_kpi as WK
from src.dashboard import charts as CH
from src.dashboard import components as UI
from src.dashboard import data as D
from src.dashboard import kpis, theme
from src.dashboard import page as P

# 本页真正吃到的产物：what-if 缓存、TCO、仿真档位、仓内 KPI、运输基线/优化、出库行、
# 异常埋点、Olist（真实观测标定基准）。delivery_orders / regions 是共享侧边栏与
# D.order_window() 的依赖，一并声明。
f = P.bootstrap(
    page_title="改善建议 · 区域仓配中心", page_icon="🛠️",
    title="改善建议",
    subtitle=lambda first, last: (
        f"区域冷链城配中心 · 连续 {C.SIM_DAYS} 天运营模拟（{first.date()} ~ {last.date()}）· "
        "「诊断→动作→收益」表由产物自动汇总，四个 what-if 控件读预计算缓存瞬时响应"
    ),
    applied="无——本页为全量口径，全局筛选不改变下列读数",
    note="本页绝大多数读数是车队/仓/代表日的**标量**（车队的里程与成本、仓的行走成本、"
         "代表日的 16 档预计算），没有日期、片区、品类维度，故日期范围 / 品类 / 配送区域三者"
         "在本页都不生效。动力模式与成本参数由下方「TCO 与动力模式」区块自己的控件选择",
    artifacts=(
        "transport_whatif", "transport_tco",
        "sim_whatif", "sim_exp2",
        "warehouse_kpi", "transport_kpi",
        "transport_anomaly_points",
        "outbound", "olist_overall",
        "delivery_orders", "regions",
    ),
)


# ---------------------------------------------------------------------------
# 纯函数：单位换算与展示格式（可独立单测，页面里不重写口径）
# ---------------------------------------------------------------------------
def _rel(path) -> str:
    """把产物路径显示成项目内相对路径（证据列用，便于按图索骥到文件）。"""
    return str(path.relative_to(C.PROJECT_ROOT))


def _pct(x, decimals: int = 2) -> str:
    return f"{x * 100:.{decimals}f}%"


def _num(x, decimals: int = 2) -> str:
    return f"{x:,.{decimals}f}"


def _dash(x, fmt: str = "{:,.2f}") -> str:
    """缺值（不可行档位填 None）显示为「—」，不显示 0——0 会被读成一个真实的零。"""
    return "—" if x is None else fmt.format(x)


def _tco_kwargs(param: str, level: str) -> dict:
    """把「敏感性参数 + 档位」翻成 `config` 成本函数的关键字参数。

    **one-at-a-time**：只有被选中的那个参数取 `level`，其余一律留 `mid`——
    这正是 `transport_tco.json::sensitivity` 的生成口径（见 `sensitivity_table` docstring），
    页面若把三个参数同时变，得到的是产物里**不存在**的组合。
    """
    kw = {"driver_wage": "mid", "energy_price": "mid", "rent": "mid"}
    if param not in kw:
        raise ValueError(f"未知敏感性参数：{param}（应为 {tuple(kw)}）")
    kw[param] = level
    return kw


@st.cache_data(show_spinner=False)
def _abc_recompute(a_cut: float, b_cut: float):
    """ABC 阈值实时重算（4.4 万行）。

    复用 `warehouse_kpi.abc_classify` 的**唯一口径**，不在页面里重写分类逻辑；
    缓存键 = `(a_cut, b_cut)`，两阈值一变缓存即失效——阈值不进键，滑块拖动就没反应。
    """
    return WK.abc_classify(D.artifact("outbound"), (float(a_cut), float(b_cut)))


# 一次性读入本页所有产物（缓存原语挂在 data.py 上，随产物 mtime 自动失效）
wh = D.warehouse_kpi()               # 仓内 KPI / ABC / 库位重排 / 埋点复现
trk = D.transport_kpi()              # 运输基线 vs 优化（代表日）
pts = D.artifact("transport_anomaly_points")   # 异常埋点池化检验 + 逐周复现
ol = D.artifact("olist_overall")     # Olist 真实履约（异常总体水平的标定基准）
tco = D.artifact("transport_tco")    # TCO 曲线 / 盈亏平衡 / 敏感性 / 建议
whatif = D.whatif_vehicles()         # 车辆数 16 档预计算缓存
exp2 = D.artifact("sim_exp2")        # 人力权衡曲线与拐点
staffing = D.artifact("sim_whatif")  # 人力档位瞬时切换缓存

DATA_CATEGORY_VOCAB = "真实观测 / 学术基准 / 过程仿真 / 情景假设"
st.caption(
    "数据类别词表（本页每行都会标注其一）：**" + DATA_CATEGORY_VOCAB + "**。"
    "过程仿真 = 程序生成的仿真仓/配送情景；情景假设 = 人为设定的成本参数与需求强度，"
    "其稳健性由三档敏感性承载；真实观测 = Olist 公开数据与高德真实路网。"
)

st.divider()

# ===========================================================================
# 一、问题诊断 → 改善动作 → 预期收益（自动汇总自产物）
# ===========================================================================
st.subheader("一、问题诊断 → 改善动作 → 预期收益")
st.caption(
    "每行的「关键读数 / 预期收益」都从 `data/processed/` 的结果文件里取，"
    "「证据文件」列给出该数字所在的文件——不在页面里硬编码任何业务数字。"
    "该表是 `report/pdca_report.md` 的 P→D→C 可交互版本，逐条对齐。"
)

_wh_path = _rel(C.WAREHOUSE_KPI_JSON)
_trk_path = _rel(C.TRANSPORT_KPI_JSON)
_pts_path = _rel(C.TRANSPORT_ANOMALY_POINTS_JSON)
_exp2_path = _rel(C.PROCESSED_DIR / "sim" / "exp2_staffing.json")
_ol_path = _rel(C.PROCESSED_DIR / "olist_kpi" / "overall_kpi.json")

_slow = wh.pick_slowdown
_p03 = wh.p03_discrepancy
_slot = wh.slotting
_tr_acc = wh.inventory_accuracy_rate
_b, _o = trk.baseline, trk.optimized
_bd, _od = _b.diesel_per_order, _o.diesel_per_order
_be, _oe = _b.ev_per_order, _o.ev_per_order
_fl, _r7 = pts["friday_late"], pts["r07_anomaly"]
_wr = pts["weekly_replication"]
_trade = exp2["tradeoff"]
_knee = _trade["knee_at_pickers"]
_m0, _m1 = _trade["marginals"][0], _trade["marginals"][1]

_diag_rows = [
    {
        "问题诊断": "拣货效率在 14–16 点存在时段低谷",
        "数据类别": "过程仿真",
        "证据文件": f"{_wh_path} → embedding_checks.pick_slowdown_14_16",
        "关键读数": f"14–16 点 {_slow.sec_per_line_14_16:.1f} 秒/行 vs 其余 "
                    f"{_slow.sec_per_line_other:.1f} 秒/行（{_slow.ratio:.2f}×）",
        "改善动作": "到货波次错峰 + 人力档位维持拐点（而非整体加人）",
        "预期收益": f"人力拐点 {_knee} 人：4→5 人每投入 1 元省 "
                    f"{_m0['sec_saved_per_yuan']:.4f} 秒，5→6 人降至 "
                    f"{_m1['sec_saved_per_yuan']:.4f} 秒",
        "收益证据文件": _exp2_path,
    },
    {
        "问题诊断": f"{C.HIGH_DIFF_CATEGORY} 品类盘点差异率显著偏高",
        "数据类别": "过程仿真",
        "证据文件": f"{_wh_path} → embedding_checks.p03_discrepancy",
        "关键读数": f"{C.HIGH_DIFF_CATEGORY} {_p03.rate:.2%} vs 其他品类 "
                    f"{_p03.other_rate:.2%}（{_p03.ratio:.2f}×）",
        "改善动作": f"{C.HIGH_DIFF_CATEGORY} 循环盘点加密、差异责任到人",
        "预期收益": f"守住库存准确率（金额加权口径）{_tr_acc:.4%}——"
                    f"{C.HIGH_DIFF_CATEGORY} 是主要缺口",
        "收益证据文件": f"{_wh_path} → kpis.inventory_accuracy",
    },
    {
        "问题诊断": "库位布局与出库频次不匹配，行走成本被浪费",
        "数据类别": "过程仿真",
        "证据文件": f"{_wh_path} → slotting",
        "关键读数": f"行走成本 {_slot.walk_cost_before:,.0f} → "
                    f"{_slot.walk_cost_after:,.0f} 行·米（−{_slot.reduction_pct:.2f}%）",
        "改善动作": "库位重排：出库频次降序 ↔ 库位距离升序配对（库位集合不变）",
        "预期收益": f"每行拣货行走省 {_slot.walk_sec_per_line_saved:.2f} 秒"
                    f"（{_slot.walk_m_per_line_before:.2f} → "
                    f"{_slot.walk_m_per_line_after:.2f} m）",
        "收益证据文件": f"{_wh_path} → slotting.walk_sec_per_line_*",
    },
    {
        "问题诊断": "人工就近派车下里程与成本偏高",
        "数据类别": trk.data_category,
        "证据文件": f"{_trk_path} → baseline / optimized",
        "关键读数": f"里程 {_b.total_distance_km:,.2f} → {_o.total_distance_km:,.2f} km；"
                    f"柴油单均 {_bd:.2f} → {_od:.2f} 元",
        "改善动作": "VRPTW 路径优化（载重+容积+时间窗+单 DC 往返，时限求解）",
        "预期收益": f"里程 −{(_b.total_distance_km - _o.total_distance_km) / _b.total_distance_km * 100:.2f}%、"
                    f"用车 {_b.n_vehicles} → {_o.n_vehicles} 台、"
                    f"纯电单均 −{(_be - _oe) / _be * 100:.2f}%",
        "收益证据文件": f"{_trk_path} → baseline_all_days（全 90 天口径）",
    },
    {
        "问题诊断": "在途异常在时间（周五）与空间（片区）上结构性偏高",
        "数据类别": f"{pts['data_category']}；总体水平由 Olist 标定",
        "证据文件": f"{_pts_path} → friday_late / r07_anomaly",
        "关键读数": f"周五「晚点」率 {_fl['rate_1']:.2%} vs {_fl['rate_0']:.2%}"
                    f"（z={_fl['z']:+.2f}）；{C.HIGH_ANOMALY_REGION} 异常率 "
                    f"{_r7['rate_1']:.2%} vs {_r7['rate_0']:.2%}（z={_r7['z']:+.2f}）",
        "改善动作": "周度异常复盘（ISO 周）+ 周五午后与埋点片区定向排班",
        "预期收益": f"{C.HIGH_ANOMALY_REGION} 异常逐周显著 "
                    f"{_wr['r07_anomaly']['n_significant']}/{_wr['r07_anomaly']['n_weeks']} 周、"
                    f"周五晚点 {_wr['friday_late']['n_significant']}/"
                    f"{_wr['friday_late']['n_weeks']} 周（当周即可发现）",
        "收益证据文件": f"{_pts_path} → weekly_replication；标定基准 {_ol_path}"
                        f"（延迟率 {ol['real_delay_rate']:.2%}）",
    },
    {
        "问题诊断": "人力与时长存在边际收益递减拐点",
        "数据类别": "过程仿真",
        "证据文件": f"{_exp2_path} → tradeoff",
        "关键读数": f"拐点 {_knee} 人；4/5/6 人履约 "
                    f"{_trade['curve'][0]['avg_fulfillment_sec']:.1f} / "
                    f"{_trade['curve'][1]['avg_fulfillment_sec']:.1f} / "
                    f"{_trade['curve'][2]['avg_fulfillment_sec']:.1f} 秒/单",
        "改善动作": "维持拐点档人数，先把预算投到时段性拥堵（错峰）而非加人",
        "预期收益": f"{_m1['from_pickers']}→{_m1['to_pickers']} 人每日多花 "
                    f"{_m1['delta_cost']:.0f} 元仅省 "
                    f"{abs(_m1['delta_fulfillment_sec']):.2f} 秒",
        "收益证据文件": _exp2_path,
    },
    {
        "问题诊断": "时间窗达成率**未**随优化改善（如实记录，不粉饰）",
        "数据类别": "过程仿真",
        "证据文件": f"{_trk_path} → baseline.time_window / optimized.time_window",
        "关键读数": f"基线 = 优化 = {_b.time_window_rate:.2%}（两方案都排在窗内，延误把同样那批点推出窗）",
        "改善动作": f"目标函数补准时项（已加 {C.TRANSPORT_DELAY_BUFFER_MIN:.0f} 分钟计划缓冲"
                    f" + 误点惩罚软上界）",
        "预期收益": f"距理论上限 {_o.time_window_ceiling_rate:.0%} 还差 "
                    f"{_o.gap_to_ceiling_pp:.2f} pp",
        "收益证据文件": f"{_trk_path} → optimized.time_window.ceiling",
    },
]
diag = pd.DataFrame(_diag_rows)
st.dataframe(diag, width="stretch", hide_index=True)

# 量化改善幅度（三条能折成同一量纲百分比的收益）
_gains = pd.DataFrame({
    "改善动作": ["库位重排", "VRPTW 路径优化（里程）", "VRPTW 路径优化（柴油单均成本）"],
    "幅度": [
        -_slot.reduction_pct,
        -(_b.total_distance_km - _o.total_distance_km) / _b.total_distance_km * 100,
        -(_bd - _od) / _bd * 100,
    ],
    "证据文件": [_wh_path, _trk_path, _trk_path],
})
UI.chart_block(
    CH.bar_chart(_gains["改善动作"], _gains["幅度"], label="改善幅度", unit="%", slot=2),
    caption=(
        "三项可折算为同一量纲（百分比）的改善幅度，取自上方表格同一批产物读数——"
        "负值表示成本/里程下降（越大越好的方向）。三个动作量纲不同，其余收益（异常率、"
        "秒/单、拐点）已在上表逐条给出，不硬凑进这张图。"
    ),
    table=_gains,
    table_label="改善幅度数据表",
)

st.divider()

# ===========================================================================
# 二、车辆数 what-if（5–20 滑块，读 16 档预计算缓存，瞬时切档）
# ===========================================================================
st.subheader("二、车辆数 what-if（读预计算缓存，瞬时切档）")
_grid = list(whatif.grid)
_lo, _hi = int(_grid[0]), int(_grid[-1])
st.caption(
    f"代表日 {whatif.representative_day}（{whatif.n_orders} 单 / {whatif.n_nodes} 个配送点）已按车辆数 "
    f"**{_lo}–{_hi}** 共 {len(_grid)} 档**离线预计算**（ADR-0010）。滑块只读缓存、"
    "**不实时求解**——车辆数对里程/成本的影响是非线性的，缓存里没有的组合不做插值外推。"
)
_n_default = C.FLEET_SIZE if _lo <= C.FLEET_SIZE <= _hi else _lo
n_veh = st.slider("车辆数（预计算档位，拖动即刻切档）", min_value=_lo, max_value=_hi,
                  value=_n_default, step=1,
                  help="档位 = 车队规模时与 10 号票发表解做过一致性自检，见下方说明")

res = whatif.gear(n_veh)
if res is None:
    # 超出网格：如实显示「需重跑预计算脚本」，不画一个够不到的 0
    st.warning(f"车辆数 {n_veh} 超出预计算网格 {_lo}–{_hi}，需重跑预计算脚本"
               "（ADR-0010：不做隐式实时求解）")
elif not res.feasible:
    # 不可行档位：显示原因，而不是画成 0（「跑不了这一天」本身就是结论）
    st.error(f"车辆数 {n_veh} 档不可行（预计算如实判为无解，非缺数据）：{res.infeasible_reason}")
else:
    _opt = res.optimized
    _cost_mode = f.cost_mode
    _mode_label = kpis.MODE_LABELS[_cost_mode]
    _po = _opt.cost_per_order[_cost_mode]
    UI.kpi_cards([
        kpis.Metric(key="wf_km", label="代表日优化后总里程", value=_opt.total_distance_km,
                    previous=None, unit="km", higher_is_better=False, decimals=1,
                    basis="该档 VRPTW 优化解的 Σ趟次里程（whatif_vehicles.json）"),
        kpis.Metric(key="wf_veh", label="实际用车数", value=float(_opt.n_vehicles_used),
                    previous=None, unit="台", higher_is_better=False, decimals=0,
                    basis="优化解实际启用的车辆数（≤ 车辆数上限）"),
        kpis.Metric(key="wf_tw", label="时间窗达成率", value=_opt.time_window_rate,
                    previous=None, unit="%", higher_is_better=True,
                    basis="逐单判定：Σ按时单 / Σ已服务单（实际送达口径）"),
        kpis.Metric(key="wf_load", label="满载率均值", value=_opt.load_rate_mean,
                    previous=None, unit="%", higher_is_better=True, decimals=1,
                    basis="该档全部趟次的实际载重 / 额定载重的平均"),
        kpis.Metric(key="wf_cost", label=f"单均成本（{_mode_label}）", value=_po,
                    previous=None, unit="元/单", higher_is_better=False,
                    basis=f"该档总成本 / 订单数；口径由侧边栏「成本口径」决定（当前：{_mode_label}）"),
    ])
    st.caption(
        f"未服务单 {_opt.n_orders_unserved} 单；车辆数上限 {n_veh} 台时实际用车 "
        f"{_opt.n_vehicles_used} 台。以上数字随滑块即时重算，读的都是缓存里的同一档记录。"
    )

# 16 档总览：图 + 表（缺值列如实显示「—」，不可行档位给出原因）
_gear_rows = []
for _n in _grid:
    _rec = whatif.gear(int(_n))
    _gopt = _rec.optimized          # 不可行档位为 None——值显示「—」，不显示 0
    _gear_rows.append({
        "车辆数": int(_n),
        "可行": "是" if _rec.feasible else "否",
        "优化用车": _dash(_gopt.n_vehicles_used if _gopt else None, "{:.0f}"),
        "总里程(km)": _dash(_gopt.total_distance_km if _gopt else None, "{:,.1f}"),
        "时间窗达成率": _dash(_gopt.time_window_rate if _gopt else None, "{:.2%}"),
        "满载率均值": _dash(_gopt.load_rate_mean if _gopt else None, "{:.1%}"),
        "柴油单均(元)": _dash(_gopt.cost_per_order["diesel"] if _gopt else None, "{:.2f}"),
        "纯电单均(元)": _dash(_gopt.cost_per_order["ev"] if _gopt else None, "{:.2f}"),
        "未服务单": _dash(_gopt.n_orders_unserved if _gopt else None, "{:.0f}"),
        "不可行原因": _rec.infeasible_reason or "",
    })
gears_tbl = pd.DataFrame(_gear_rows)
_feasible = gears_tbl[gears_tbl["可行"] == "是"]
_g_labels = [int(g) for g in _feasible["车辆数"]]

_r2 = st.columns(2)
with _r2[0]:
    UI.chart_block(
        CH.bar_chart(_g_labels, [float(x.replace(",", "")) for x in _feasible["总里程(km)"]],
                     label="优化后总里程", unit=" km",
                     highlight={str(n_veh): theme.series(1)}),
        caption=(
            "各可行档位（车辆数越多，求解器可动用的车越多）的优化后总里程。"
            "当前选中档以序列色高亮（强调，不是按值深浅编码）。不可行档位（"
            f"{[int(g) for g in gears_tbl.loc[gears_tbl['可行'] == '否', '车辆数']]}）"
            "没有数值，故不在图上——它们不是 0，是「跑不了这一天」，原因见下表。"
            "该图层级：`whatif_vehicles.json`。"
        ),
        table=gears_tbl,
        table_label="what-if 16 档数据表",
        height=340,
    )
with _r2[1]:
    _rate_x = _g_labels
    UI.chart_block(
        CH.dual_line_chart(
            _rate_x,
            {
                "时间窗达成率": [float(_feasible.iloc[i]["时间窗达成率"].rstrip("%")) / 100
                                 for i in range(len(_feasible))],
                "满载率均值": [float(_feasible.iloc[i]["满载率均值"].rstrip("%")) / 100
                               for i in range(len(_feasible))],
            },
            y_title="百分比 (%)", unit="%",
            slots={"时间窗达成率": 0, "满载率均值": 1},
        ),
        caption=(
            "两条序列同量纲（均为百分比），故共用一根 0–100% 的轴——不使用双 Y 轴。"
            "时间窗达成率随车辆数上升而趋稳（车辆足够后瓶颈转到时间窗与延误），"
            "满载率则先升后平：车多到一定程度，每车装的货变少。图例用于区分两个指标。"
            "该图层级：`whatif_vehicles.json`。"
        ),
        table=_feasible[["车辆数", "优化用车", "时间窗达成率", "满载率均值"]],
        table_label="时间窗达成率 / 满载率数据表",
        height=340,
    )

_chk = whatif.consistency_check
if _chk is not None:
    st.caption(
        f"**一致性自检**：档位 = 车队规模（{_chk.gear} 台）时，预计算里程 "
        f"{_chk.distance_km:,.2f} km vs 10 号票发表 {_chk.published_distance_km:,.2f} km，"
        f"差 {_chk.distance_delta_pct:+.3f}%。{_chk.note}"
    )

st.divider()

# ===========================================================================
# 三、ABC 累计占比阈值滑块（实时重算分类 + 联动帕累托）
# ===========================================================================
st.subheader("三、ABC 阈值实时重算（数据源 = 仿真仓出库行数）")
st.caption(
    f"ABC 分类按 ADR-0002 以**仿真仓出库行数**累计占比划分（不是 Olist 商品销量）。"
    f"默认阈值 A={C.ABC_THRESHOLDS[0]:.0%} / B={C.ABC_THRESHOLDS[1]:.0%}（`config.ABC_THRESHOLDS`）；"
    "拖动下方两个滑块会**实时重算**分类（复用 `warehouse_kpi.abc_classify` 的唯一口径），"
    "联动帕累托图与三类 SKU 数 / 出库行数占比。"
)
_c1 = st.columns(2)
with _c1[0]:
    a_cut = st.slider("A 类累计占比阈值", min_value=0.30, max_value=0.90,
                      value=float(C.ABC_THRESHOLDS[0]), step=0.01,
                      help="累计占比「尚未达到」该阈值者归入 A 类")
with _c1[1]:
    b_cut = st.slider("B 类累计占比阈值", min_value=0.40, max_value=0.99,
                      value=float(C.ABC_THRESHOLDS[1]), step=0.01,
                      help="A 之后、累计占比「尚未达到」该阈值者归入 B 类，其余 C 类")

# 阈值须满足 0 < A < B < 1；两滑块各自的范围独立，故在此显式钳制并提示，绝不抛错
if not (0 < a_cut < b_cut < 1):
    b_eff = round(a_cut + 0.01, 2)
    st.warning(f"A 阈值须小于 B 阈值；当前 A={a_cut:.2f} ≥ B={b_cut:.2f}，"
               f"已按 B = A + 0.01 = {b_eff:.2f} 计算并在下方标注。")
else:
    b_eff = b_cut

_count, _pareto, _summary = _abc_recompute(a_cut, b_eff)
_by = _summary["by_class"]
_default_by = wh.abc_by_class
_classes_tbl = pd.DataFrame([
    {
        "类别": cls,
        "SKU 数": _by[cls]["n_sku"],
        "SKU 占比": _pct(_by[cls]["sku_share"], 1),
        "出库行数占比": _pct(_by[cls]["line_share"], 2),
        f"默认阈值下 SKU 数（A={C.ABC_THRESHOLDS[0]:.0%}/B={C.ABC_THRESHOLDS[1]:.0%}）":
            _default_by[cls].n_sku,
    }
    for cls in ("A", "B", "C")
])
st.caption(
    f"当前阈值 A={a_cut:.2f} / B={b_eff:.2f}，共 {_summary['n_sku']} 个 SKU、"
    f"{_summary['total_lines']:,} 条出库行。A 类 "
    f"{_by['A']['n_sku']} 个 SKU（{_pct(_by['A']['sku_share'], 1)}）承担 "
    f"{_pct(_by['A']['line_share'])} 出库行数。阈值一变，三个数字与下方帕累托立即重算"
    f"（缓存键含两个阈值，拖动即生效）。"
)
st.dataframe(_classes_tbl, width="stretch", hide_index=True)

UI.chart_block(
    CH.cumulative_share_chart(
        _pareto["rank"].tolist(),
        (_pareto["share"] * 100).tolist(),
        (_pareto["cum_share"] * 100).tolist(),
    ),
    caption=(
        f"横轴为频次排名（1 起，共 {len(_pareto)} 个 SKU）：柱 = 各 SKU 出库行数占比，"
        "线 = 累计占比；两者同量纲，共用一根 0–100% 轴（不使用双 Y 轴）。"
        f"当前阈值 A={a_cut:.2f} / B={b_eff:.2f}。该图无日期/品类/片区维度，不受全局筛选影响。"
    ),
    table=_pareto[["rank", "sku_id", "outbound_lines", "share", "cum_share", "abc_class"]],
    table_label="ABC 帕累托数据表（全 SKU）",
    height=380,
)

st.divider()

# ===========================================================================
# 四、拣货员人数（联动仿真缓存档位，显示拐点）
# ===========================================================================
st.subheader("四、拣货员人数（联动仿真缓存档位）")
st.caption(
    "SimPy 离散事件仿真（数据层 F），代表日 "
    f"{C.SIM_N_ORDERS_PER_DAY} 单、每档重复 {C.SIM_REPEATS} 次。所有数字读自预计算缓存"
    "（`whatif_staffing.json`），切换档位只切缓存、不重算仿真，故瞬时响应（ADR-0010 同哲学）。"
)
_arms = sorted(staffing.keys(), key=lambda k: int(k))
_pick = st.select_slider("拣货员人数（预计算档位，切换不重算仿真）",
                         options=_arms, value=_arms[len(_arms) // 2])
_sel = staffing[_pick]
_staff_tbl = pd.DataFrame([{
    "拣货员人数": int(_a),
    "平均履约时长(秒/单)": staffing[_a]["avg_fulfillment_sec"]["mean"],
    "CI 下界": staffing[_a]["avg_fulfillment_sec"]["ci95_low"],
    "CI 上界": staffing[_a]["avg_fulfillment_sec"]["ci95_high"],
    "拣货员利用率": staffing[_a]["picker_utilization"]["mean"],
    "日人力成本(元)": staffing[_a]["daily_labor_cost"],
} for _a in _arms])

UI.chart_block(
    CH.bar_chart([f"{_a} 人" for _a in _arms],
                 [staffing[_a]["avg_fulfillment_sec"]["mean"] for _a in _arms],
                 label="平均订单履约时长", unit=" 秒/单",
                 highlight={f"{_pick} 人": theme.series(1)}),
    caption=(
        f"当前选中 {_pick} 人档（以序列色高亮该柱，属强调而非取值编码）：平均履约 "
        f"{_sel['avg_fulfillment_sec']['mean']:.1f} 秒/单（95% CI "
        f"{_sel['avg_fulfillment_sec']['ci95_low']:.1f}–{_sel['avg_fulfillment_sec']['ci95_high']:.1f}），"
        f"拣货员利用率 {_sel['picker_utilization']['mean']:.1%}，"
        f"日人力成本 {_sel['daily_labor_cost']:.0f} 元。"
        f"边际收益递减，**拐点在 {_knee} 人**：{_m0['from_pickers']}→{_m0['to_pickers']} 人"
        f"每投入 1 元省 {_m0['sec_saved_per_yuan']:.3f} 秒，{_m1['from_pickers']}→"
        f"{_m1['to_pickers']} 人降至 {_m1['sec_saved_per_yuan']:.3f} 秒。"
        "该图无日期/品类/片区维度，不受全局筛选影响。"
    ),
    table=_staff_tbl,
    table_label="人力档位数据表",
    height=320,
)

st.divider()

# ===========================================================================
# 五、动力模式 + 三档敏感性（联动 TCO 曲线与盈亏平衡里程）
# ===========================================================================
st.subheader("五、动力模式与三档敏感性（TCO 曲线 / 盈亏平衡里程）")
_mode_labels = list(kpis.MODE_LABELS.values())
_mode_keys = {v: k for k, v in kpis.MODE_LABELS.items()}
_rec_mode = tco["recommendation"]["recommended_mode"]
st.caption(
    f"TCO 建议模式为 **{kpis.MODE_LABELS[_rec_mode]}**（{tco['recommendation']['basis']}，"
    f"证据：`{_rel(C.TRANSPORT_TCO_JSON)}`）。下方「动力模式」切换 TCO 曲线与参考读数；"
    "「敏感性」是 **one-at-a-time**——只变被选中的一个参数、其余两个留在中位（mid），"
    "**不要把三个参数同时变**，那样得到的组合在产物里不存在。"
)
_mode_col, _param_col, _level_col = st.columns(3)
with _mode_col:
    _mode_label = st.radio("动力模式", _mode_labels,
                           index=_mode_labels.index(kpis.MODE_LABELS[_rec_mode]),
                           help="柴油自购（槽 0）/ 纯电租赁（槽 1），类别色按实体固定分配")
with _param_col:
    _PARAM_LABELS = {"driver_wage": "司机工资", "energy_price": "能源价格", "rent": "租金"}
    _param_key = st.selectbox("敏感性参数（一次只变一个）",
                              options=list(_PARAM_LABELS),
                              format_func=lambda k: _PARAM_LABELS[k],
                              index=0)
with _level_col:
    _LEVEL_LABELS = {"low": "低", "mid": "中", "high": "高"}
    _level = st.radio("档位", options=list(_LEVEL_LABELS),
                      format_func=lambda k: _LEVEL_LABELS[k], index=1, horizontal=True)

_mode_key = _mode_keys[_mode_label]
_mode_slot = 0 if _mode_key == "diesel" else 1  # 柴油永远槽 0、纯电永远槽 1
_kw = _tco_kwargs(_param_key, _level)           # one-at-a-time
_curve = tco["curves"][_mode_key]
_mileage = _curve["mileage_km"]
_ref_km = float(tco["reference_daily_km"])
_be_km = float(costing.breakeven_km(**_kw))
_cost_curve = [float(costing.daily_total(_mode_key, km, **_kw)) for km in _mileage]
_ref_cost = float(costing.daily_total(_mode_key, _ref_km, **_kw))

UI.kpi_cards([
    kpis.Metric(key="tco_fixed", label=f"日固定成本（{_mode_label}）",
                value=float(costing.fixed_per_day(
                    _mode_key, driver_wage=_kw["driver_wage"], rent=_kw["rent"])),
                previous=None, unit="元/日", higher_is_better=False,
                basis=f"costing.fixed_per_day（司机工资={_LEVEL_LABELS[_kw['driver_wage']]}、"
                      f"租金={_LEVEL_LABELS[_kw['rent']]}）"),
    kpis.Metric(key="tco_perkm", label=f"公里变动成本（{_mode_label}）",
                value=float(costing.per_km(_mode_key, energy_price=_kw["energy_price"])),
                previous=None, unit="元/km", higher_is_better=False, decimals=3,
                basis=f"costing.per_km（能源价格={_LEVEL_LABELS[_kw['energy_price']]}）"),
    kpis.Metric(key="tco_ref", label="参考里程处日成本",
                value=_ref_cost, previous=None, unit="元/日", higher_is_better=False,
                basis=f"costing.daily_total @ 参考里程 {_ref_km:.1f} km（tco.reference_daily_km）"),
    kpis.Metric(key="tco_be", label="盈亏平衡里程",
                value=_be_km, previous=None, unit="km/日", higher_is_better=True,
                basis="costing.breakeven_km（柴油自购 = 纯电租赁 日总成本相等点）"),
])

UI.chart_block(
    CH.tco_curve(_mode_label, _mileage, _cost_curve, slot=_mode_slot,
                      breakeven_km=_be_km, ref_km=_ref_km, ref_cost=_ref_cost),
    caption=(
        f"{_mode_label} 在当前敏感性设置（{_PARAM_LABELS[_param_key]} = {_LEVEL_LABELS[_level]}，"
        f"其余参数留 mid）下的日总成本随单车日行驶里程的曲线。参考线为盈亏平衡里程 "
        f"{_be_km:.2f} km/日（该点两模式日成本相等，取自 `costing.breakeven_km`）；"
        f"标记点为参考里程 {_ref_km:.2f} km 处的成本 {_ref_cost:,.2f} 元/日。"
        f"曲线与标注随「动力模式 / 敏感性参数 / 档位」三个控件即时重算。"
        "该图无日期/品类/片区维度，不受全局筛选影响。"
    ),
    table=pd.DataFrame({
        "单车日里程(km)": _mileage,
        f"{_mode_label}日总成本(元)": [round(c, 2) for c in _cost_curve],
    }),
    table_label="TCO 曲线数据表",
    height=340,
)

# 三档敏感性：选中参数在低/中/高三档下的两模式日成本（同量纲，共用一根轴）
_sens = tco["sensitivity"][_param_key]
_levels = ["low", "mid", "high"]
_sens_tbl = pd.DataFrame({
    "档位": [_LEVEL_LABELS[lv] for lv in _levels],
    "柴油自购日成本(元)": [round(float(_sens[lv]["diesel_daily_cost"]), 2) for lv in _levels],
    "纯电租赁日成本(元)": [round(float(_sens[lv]["ev_daily_cost"]), 2) for lv in _levels],
    "盈亏平衡里程(km)": [round(float(_sens[lv]["breakeven_km"]), 2) for lv in _levels],
})
UI.chart_block(
    CH.dual_line_chart(
        [_LEVEL_LABELS[lv] for lv in _levels],
        {
            "柴油自购": [float(_sens[lv]["diesel_daily_cost"]) for lv in _levels],
            "纯电租赁": [float(_sens[lv]["ev_daily_cost"]) for lv in _levels],
        },
        # 槽位写实而不是靠字典次序碰巧对上：下方文案宣称「柴油永远槽 0、纯电永远槽 1」，
        # 那句话得由代码保证，否则重排一次字典就静默变色
        slots={"柴油自购": 0, "纯电租赁": 1},
        y_title="日总成本 (元)", unit=" 元",
    ),
    caption=(
        f"**{_PARAM_LABELS[_param_key]}** 在低/中/高三档下，两模式的日总成本"
        f"（one-at-a-time：另两个参数固定在中位 mid，参考里程 {_ref_km:.2f} km）。"
        "两条序列同量纲，共用一根轴并配图例；柴油永远槽 0、纯电永远槽 1，不随控件重排颜色。"
        "该表直接读 `transport_tco.json → sensitivity`，与上方曲线是同一口径的两个切面。"
        + ("注意：司机工资对两模式同额、相减抵消，**不影响盈亏平衡里程**——"
           "三档的平衡里程因此相同（`costing.breakeven_km` 已注明）。"
           if _param_key == "driver_wage" else "")
    ),
    table=_sens_tbl,
    table_label="三档敏感性数据表",
    height=340,
)

with st.expander("完整敏感性表（三个参数 × 三档，one-at-a-time）"):
    _full_rows = []
    for _p, _plabel in _PARAM_LABELS.items():
        for _lv in _levels:
            _d = tco["sensitivity"][_p][_lv]
            _full_rows.append({
                "参数": _plabel, "档位": _LEVEL_LABELS[_lv],
                "柴油日成本(元)": round(float(_d["diesel_daily_cost"]), 2),
                "纯电日成本(元)": round(float(_d["ev_daily_cost"]), 2),
                "盈亏平衡里程(km)": round(float(_d["breakeven_km"]), 2),
            })
    st.dataframe(pd.DataFrame(_full_rows), width="stretch", hide_index=True)
    st.caption(
        "每一行都是「只变该参数到该档、其余留 mid」的结果（one-at-a-time），"
        "**不是**三个参数同时变化的组合。证据：`" + _rel(C.TRANSPORT_TCO_JSON) + " → sensitivity`。"
    )

st.caption(
    "**外包对照（决策参考，不进自营车队）**：" + tco["recommendation"]["outsourcing_verdict"] + "。"
    + "".join(" " + c for c in tco["recommendation"]["caveats"])
)

st.divider()

# ===========================================================================
# 六、数据来源与口径
# ===========================================================================
with st.expander("数据来源与口径（每个数字的追溯入口）"):
    st.markdown(
        f"""
| 区块 | 读数 | 产物文件 | 生成命令 | 数据类别 |
|---|---|---|---|---|
| 诊断→动作→收益表 | 埋点复现 / 行走成本 / ABC / 里程成本 / 异常埋点 / 拐点 | `{C.WAREHOUSE_KPI_JSON.name}`、`{C.TRANSPORT_KPI_JSON.name}`、`{C.TRANSPORT_ANOMALY_POINTS_JSON.name}`、`exp2_staffing.json` | `python -m src.warehouse_kpi` / `src.transport_optimize` / `src.transport_decisions` / `src.warehouse_sim` | 混合，逐行已标注 |
| 车辆数 what-if | 16 档里程 / 用车 / 时间窗 / 满载率 / 单均成本 | `{C.TRANSPORT_WHATIF_JSON.name}` | `python -m src.transport_decisions` | 预计算结果（非实时求解，ADR-0010） |
| ABC 阈值重算 | SKU 出库行数累计占比与三类占比 | `{C.WAREHOUSE_ABC_PARETO_CSV.name}`（口径）+ `{C.WAREHOUSE_TABLES['outbound'].name}`（实时重算输入） | `python -m src.warehouse_kpi` / `src.gen_warehouse_data` | 过程仿真（数据源=仿真仓出库行数，ADR-0002） |
| 拣货员人数 | 履约时长 / 利用率 / 日人力成本 / 拐点 | `whatif_staffing.json`（`{C.PROCESSED_DIR.name}/sim/`）、`exp2_staffing.json` | `python -m src.warehouse_sim` | 过程仿真 |
| 动力模式 + 敏感性 | TCO 曲线 / 盈亏平衡 / 参考里程成本 | `{C.TRANSPORT_TCO_JSON.name}` | `python -m src.transport_decisions` | 情景假设（成本参数锚点见台账第 4 节，ADR-0009） |
| 异常总体水平标定 | Olist 真实延迟率 | `overall_kpi.json`（`{C.PROCESSED_DIR.name}/olist_kpi/`） | `python -m src.olist_kpi` | 真实观测 |

- **本页为全量口径**：车辆 / 仓库 / 代表日的标量读数没有日期、片区、品类维度，全局筛选器
  不改变它们；页面已在此显式声明，不装作用户拖了筛选器而数字没动的样子。
- 三个 what-if 控件**全部读预计算缓存**（ADR-0010）：滑块只切档、不实时求解；超出网格或不可行
  的组合如实提示，**不插值、不外推、不画成 0**。
- 敏感性是 **one-at-a-time**（变一个、其余留 mid）；「诊断→动作→收益」表的每个数字都标出了
  证据文件与数据类别，可用工具列复制 `data/processed/` 路径直接核对。
- 本页不显示 0 或占位值来掩盖缺产物；产物不全时直接列出缺哪个文件、该跑哪条命令。
"""
    )
