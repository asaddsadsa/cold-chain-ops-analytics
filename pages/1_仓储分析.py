"""仓储分析页（看板模块三 · 13 号票，需求 44）。

一屏回答「仓内作业与真实履约哪里有问题」：逐日仓内指标、ABC 帕累托、分时段拣货效率、
库位出库频次热力图、盘点差异品类、Olist 真实履约时长与延迟州下钻、SimPy 仿真对照实验
（含拣货员人数即时切换）。全部读数来自 `data/processed/` 与 `data/warehouse/` 的产物，
页面内不硬编码任何业务数字，颜色一律取自 `src.dashboard.theme`。

运行：在项目根执行 `streamlit run app.py`，左侧页面选择「仓储分析」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src.dashboard import charts as CH
from src.dashboard import components as UI
from src.dashboard import data as D
from src.dashboard import kpis, theme
from src.dashboard import page as P

# 本页真正吃到的产物：仓内 KPI/日表与三张下钻、库位与出库、仿真对照实验、Olist 履约。
# delivery_orders / regions 是共享侧边栏与 D.order_window() 的依赖，一并声明。
f = P.bootstrap(
    page_title="仓储分析 · 区域仓配中心", page_icon="📦",
    title="仓储分析",
    subtitle=lambda first, last: (
        f"区域仓配中心 · 连续 {C.SIM_DAYS} 天运营模拟（{first.date()} ~ {last.date()}）· "
        "仓内指标来自仿真仓（数据层 A/F），履约对比来自 Olist 实测（数据层 B）"
    ),
    applied="日期范围（仅作用于「逐日仓内指标」）、品类（仅作用于「盘点差异品类」）",
    note="配送区域在本页不生效：仓内指标与 Olist 州下钻都没有片区（R01–R08）维度；"
         "成本口径属运输侧，与仓内无关。",
    artifacts=(
        "warehouse_kpi", "warehouse_daily",
        "warehouse_abc_pareto", "warehouse_picking_hour", "warehouse_stocktake_category",
        "location_master", "outbound",
        "sim_exp1", "sim_exp2", "sim_whatif",
        "olist_overall", "olist_state", "olist_orders",
        "delivery_orders", "regions",
    ),
)

kpi = D.warehouse_kpi()

# ---------------------------------------------------------------------------
# 一、逐日仓内指标（本页唯一吃「日期范围」筛选器的区块）
# ---------------------------------------------------------------------------
st.subheader("一、逐日仓内指标（日期范围筛选生效）")
wh_daily = D.artifact("warehouse_daily")
wh_win = kpis.slice_window(wh_daily, f.window)
st.caption(
    f"当前区间 {f.window.start.date()} ~ {f.window.end.date()}（{f.window.days} 天），"
    "直接截取自 daily_kpi.csv 的逐日序列——预设「最近 N 天」锚定数据末日，不是今天。"
    "品类筛选器不作用于本区块（日表无品类维度）；配送区域与仓内指标无关。"
)

_daily_cols = ["date", "picking_lines_per_hour", "stocktake_discrepancy_rate",
               "inventory_accuracy", "receipt_timeliness_rate"]
_daily_view = wh_win[_daily_cols].copy() if not wh_win.empty else wh_win
_row = st.columns(2)
with _row[0]:
    UI.chart_block(
        UI.metric_trend(wh_win, x="date", y="picking_lines_per_hour",
                        label="拣货效率", unit="行/小时", slot=0, decimals=1),
        caption="当日出库行数 / 当日拣货工时；与分时段低谷（第三部分）同一批作业记录。",
        table=_daily_view[["date", "picking_lines_per_hour"]],
        table_label="拣货效率数据表",
    )
with _row[1]:
    UI.chart_block(
        UI.metric_trend(wh_win, x="date", y="stocktake_discrepancy_rate",
                        label="盘点差异率", unit="%", slot=1),
        caption="逐记录口径：当日差异记录数 / 盘点记录数。与驾驶舱「库存准确率」（金额加权）"
                "是两个口径，不可互换。",
        table=_daily_view[["date", "stocktake_discrepancy_rate"]],
        table_label="盘点差异率数据表",
    )

st.divider()

# ---------------------------------------------------------------------------
# 二、ABC 帕累托（数据源 = 仿真仓出库行数，ADR-0002）
# ---------------------------------------------------------------------------
st.subheader("二、ABC 帕累托：出库行数贡献")
abc = D.artifact("warehouse_abc_pareto")
_abc_meta = kpi.abc_by_class
_a, _b, _c = (_abc_meta["A"], _abc_meta["B"], _abc_meta["C"])
UI.chart_block(
    CH.cumulative_share_chart(
        abc["rank"].tolist(),
        (abc["share"] * 100).tolist(),
        (abc["cum_share"] * 100).tolist(),
    ),
    caption=(
        "数据源为**仿真仓出库行数**（ADR-0002），不是 Olist 商品销量。"
        f"A 类 {_a.n_sku} 个 SKU（{_a.sku_share:.1%}）贡献 {_a.line_share:.1%} 出库行数，"
        f"B 类 {_b.n_sku} 个贡献 {_b.line_share:.1%}，"
        f"C 类 {_c.n_sku} 个仅贡献 {_c.line_share:.1%}。"
        "横轴为频次排名（1 起，共 500 个 SKU），柱=各 SKU 出库行数占比，线=累计占比；"
        "两者同量纲，共用一根 0–100% 轴（不使用双 Y 轴）。"
        "该图无日期/品类/片区维度，不受全局筛选影响。"
    ),
    table=abc[["rank", "sku_id", "outbound_lines", "share", "cum_share", "abc_class"]],
    table_label="ABC 帕累托数据表（500 SKU）",
    height=380,
)

st.divider()

# ---------------------------------------------------------------------------
# 三、分时段拣货效率（必须肉眼可见 14–16 点低谷）
# ---------------------------------------------------------------------------
st.subheader("三、分时段拣货效率（14–16 点低谷）")
hour_df = D.artifact("warehouse_picking_hour")
# 低谷时段来自配置（单一来源），不在此写死；highlight 只给这两根柱上状态色，其余用序列色
_slow_hours = set(range(C.PICK_SLOW_HOURS[0], C.PICK_SLOW_HOURS[1]))
_highlight = {str(h): theme.STATUS["warning"] for h in _slow_hours}
_slow = kpi.pick_slowdown
UI.chart_block(
    CH.bar_chart(hour_df["hour"].astype(str), hour_df["sec_per_line"],
                 label="单行拣货耗时", unit=" 秒/行", highlight=_highlight),
    caption=(
        "横轴为出库所处小时（8–18 点），纵轴为单行拣货耗时（秒/行）。"
        f"14–16 点两根柱以状态色（⚠️ 警示）标出：该时段 {_slow.sec_per_line_14_16:.1f} 秒/行，"
        f"其余时段 {_slow.sec_per_line_other:.1f} 秒/行，约 {_slow.ratio:.2f} 倍，是埋点低谷。"
        "状态色永远配图标与文字，不靠颜色单独表意；单序列图不放图例。"
        "该图无日期/品类/片区维度，不受全局筛选影响。"
    ),
    table=hour_df,
    table_label="分时段拣货效率数据表",
)

st.divider()

# ---------------------------------------------------------------------------
# 四、库位出库频次热力图（800 库位网格）
# ---------------------------------------------------------------------------
st.subheader("四、库位出库频次热力图（800 库位网格）")
freq = D.location_frequency()
_rows = range(int(freq["row"].min()), int(freq["row"].max()) + 1)
_cols = range(int(freq["col"].min()), int(freq["col"].max()) + 1)
# 先把网格补全：没有出库的库位补 0（不是空洞/空白），再交给热力图
_grid = (freq.pivot_table(index="row", columns="col", values="outbound_lines", aggfunc="sum")
         .reindex(index=list(_rows), columns=list(_cols)).fillna(0))
_n_zero = int((freq["outbound_lines"] == 0).sum())
UI.chart_block(
    CH.heatmap_grid(list(_grid.columns), list(_grid.index), _grid.values.tolist(),
                    x_title="库位列", y_title="库位行", color_title="出库行数"),
    caption=(
        f"{len(freq)} 个库位排成 {_grid.shape[0]} 行 × {_grid.shape[1]} 列的网格，"
        f"色深=该库位承载的出库行数（与 ABC、库位重排同源口径，ADR-0002）。"
        f"其中 {_n_zero} 个库位在本批出库中完全没有行，已补 0 显示而非留空洞。"
        "顺序编码只用单色相由浅到深，不用彩虹色阶。"
        "该图无日期/品类/片区维度，不受全局筛选影响。"
    ),
    table=freq[["loc_id", "row", "col", "zone", "walk_dist_m", "outbound_lines"]],
    table_label="库位出库频次数据表（800 库位）",
    height=470,
)

st.divider()

# ---------------------------------------------------------------------------
# 五、盘点差异 Top10 品类（P03 居前；吃「品类」筛选器）
# ---------------------------------------------------------------------------
st.subheader("五、盘点差异 Top10 品类")
_full_cat = D.artifact("warehouse_stocktake_category")
cat = _full_cat
if f.categories:
    cat = cat[cat["category"].isin(f.categories)]
cat = cat.sort_values("rate", ascending=False).head(10)
if cat.empty:
    st.info("当前「品类」筛选下没有盘点记录，无法绘制该图。")
else:
    # P03 埋点比值读 KPI 产物里已算好的那一份，不在页面重算——「同一口径两处实现」
    # 漂移时不会有东西变红（`warehouse_kpi.interpret` 已把它收进具名结构）。
    _p03 = kpi.p03_discrepancy
    UI.chart_block(
        CH.bar_chart(cat["category"], cat["rate"] * 100, label="盘点差异率", unit="%",
                     orientation="h",
                     highlight={C.HIGH_DIFF_CATEGORY: theme.STATUS["warning"]}),
        caption=(
            "按差异率降序取前 10 个品类，横向柱状图。"
            f"{C.HIGH_DIFF_CATEGORY}（埋点品类）以状态色标出：差异率 {_p03.rate:.2%}，"
            f"约为其余品类合计（{_p03.other_rate:.2%}）的 {_p03.ratio:.2f} 倍。"
            "口径：差异率 = 差异记录数 / 盘点记录数（逐记录）——两边的率都按**记录数**合并，"
            "不是对各品类比率取等权平均（那样记录少的品类会拿到与记录多的品类一样的权重，"
            "而且与单项的率不是同一种量）；"
            "与驾驶舱「库存准确率」（金额加权）是两个口径，不可互换。"
            "本图吃侧边栏「品类」筛选器。"
        ),
        table=cat[["category", "rate", "n_diff", "n_records"]],
        table_label="盘点差异品类数据表",
        height=380,
    )

st.divider()

# ---------------------------------------------------------------------------
# 六、Olist 真实履约时长分布 + 延迟州下钻
# ---------------------------------------------------------------------------
st.subheader("六、Olist 真实履约时长分布与延迟州下钻")
dur = D.artifact("olist_orders")
ov = D.artifact("olist_overall")
_vals = dur["fulfillment_days"]
_counts, _edges = np.histogram(_vals, bins=40)
_hist_tbl = pd.DataFrame({
    "区间下界(天)": _edges[:-1], "区间上界(天)": _edges[1:], "笔数": _counts,
})
UI.chart_block(
    CH.histogram(_vals, label="履约时长", unit="天", bins=40, x_title="履约时长（天）"),
    caption=(
        f"Olist 实测 {ov['n_orders']:,} 单的履约时长分布（下单→签收，天）。"
        f"平均 {ov['avg_fulfillment_days']:.2f} 天，其中在途 {ov['avg_transit_days']:.2f} 天、"
        f"出库响应 {ov['avg_outbound_response_days']:.2f} 天。"
        "该产物只取了逐单履约时长等列，没有日期与片区维度，**不受日期范围/配送区域筛选影响**。"
    ),
    table=_hist_tbl,
    table_label="履约时长分布数据表",
)

_state = D.artifact("olist_state").sort_values("delay_rate", ascending=False)
_big = _state.loc[_state["n_orders"].idxmax()]
_small = _state.loc[_state["n_orders"].idxmin()]
UI.chart_block(
    CH.bar_chart(_state["customer_state"], _state["delay_rate"] * 100,
                 label="延迟率", unit="%", orientation="h"),
    caption=(
        "各州延迟率（未在承诺日前送达的占比），按延迟率降序取前若干州，横向柱状图。"
        f"注意样本量差异很大：{_big['customer_state']} {int(_big['n_orders']):,} 单 vs "
        f"{_small['customer_state']} {int(_small['n_orders']):,} 单——小样本州的延迟率天然波动更大，"
        "读图时请结合 n_orders 判断，不要把尾部小州的排名当成稳定结论。"
        "州（customer_state）与配送片区（R01–R08）没有映射，本图不受配送区域筛选影响。"
    ),
    table=_state,
    table_label="各州延迟率数据表",
    height=560,
)

st.divider()

# ---------------------------------------------------------------------------
# 七、仿真对照实验 + 拣货员人数即时切换
# ---------------------------------------------------------------------------
st.subheader("七、仿真对照实验与人力档位")
st.caption(
    "同为仿真仓（数据层 F），代表日 300 单、每档重复 30 次。"
    "所有数字读自预计算缓存，切换档位只切缓存、不重算仿真，故瞬时响应（ADR-0010 同哲学）。"
)

exp1 = D.artifact("sim_exp1")
_orig = exp1["original"]["metrics"]
_zoned = exp1["abc_zoned"]["metrics"]
_imp = exp1["improvement"]
st.markdown("**实验一：原始布局 vs ABC 分区**（各 30 次重复，误差棒为 95% CI）")
_e1 = st.columns(2)
with _e1[0]:
    UI.chart_block(
        CH.bar_with_ci(["原始布局", "ABC 分区"],
                [_orig["avg_order_pick_sec"]["mean"], _zoned["avg_order_pick_sec"]["mean"]],
                [_orig["avg_order_pick_sec"]["ci95_low"], _zoned["avg_order_pick_sec"]["ci95_low"]],
                [_orig["avg_order_pick_sec"]["ci95_high"], _zoned["avg_order_pick_sec"]["ci95_high"]],
                label="平均订单拣货时长", unit=" 秒/单", slot=0),
        caption=(
            f"单均拣货时长从 {_orig['avg_order_pick_sec']['mean']:.1f} 秒降到 "
            f"{_zoned['avg_order_pick_sec']['mean']:.1f} 秒（降幅 {_imp['pick_sec_reduction_pct']:.1f}%）。"
            "误差棒为两臂各自的 95% CI；单序列（两臂是 x 轴类别），故不放图例。"
        ),
        table=pd.DataFrame({
            "布局": ["原始布局", "ABC 分区"],
            "平均订单拣货时长(秒/单)": [_orig["avg_order_pick_sec"]["mean"],
                                        _zoned["avg_order_pick_sec"]["mean"]],
            "CI 下界": [_orig["avg_order_pick_sec"]["ci95_low"],
                       _zoned["avg_order_pick_sec"]["ci95_low"]],
            "CI 上界": [_orig["avg_order_pick_sec"]["ci95_high"],
                       _zoned["avg_order_pick_sec"]["ci95_high"]],
        }),
        table_label="实验一·拣货时长数据表",
    )
with _e1[1]:
    UI.chart_block(
        CH.bar_with_ci(["原始布局", "ABC 分区"],
                [_orig["avg_walk_m_per_order"]["mean"], _zoned["avg_walk_m_per_order"]["mean"]],
                [_orig["avg_walk_m_per_order"]["ci95_low"], _zoned["avg_walk_m_per_order"]["ci95_low"]],
                [_orig["avg_walk_m_per_order"]["ci95_high"], _zoned["avg_walk_m_per_order"]["ci95_high"]],
                label="单均行走距离", unit=" 米/单", slot=2),
        caption=(
            f"单均行走距离从 {_orig['avg_walk_m_per_order']['mean']:.1f} 米降到 "
            f"{_zoned['avg_walk_m_per_order']['mean']:.1f} 米（降幅 {_imp['walk_m_reduction_pct']:.1f}%）。"
            "误差棒为 95% CI。"
        ),
        table=pd.DataFrame({
            "布局": ["原始布局", "ABC 分区"],
            "单均行走距离(米/单)": [_orig["avg_walk_m_per_order"]["mean"],
                                    _zoned["avg_walk_m_per_order"]["mean"]],
            "CI 下界": [_orig["avg_walk_m_per_order"]["ci95_low"],
                       _zoned["avg_walk_m_per_order"]["ci95_low"]],
            "CI 上界": [_orig["avg_walk_m_per_order"]["ci95_high"],
                       _zoned["avg_walk_m_per_order"]["ci95_high"]],
        }),
        table_label="实验一·行走距离数据表",
    )

st.markdown("**实验二 / 人力 what-if：拣货员人数档位**")
whatif = D.artifact("sim_whatif")
_arms = sorted(whatif.keys(), key=lambda k: int(k))
_pick = st.select_slider("拣货员人数（预计算档位，切换不重算仿真）",
                         options=_arms, value=_arms[len(_arms) // 2])
_sel = whatif[_pick]

exp2 = D.artifact("sim_exp2")
_trade = exp2["tradeoff"]
_wave_min = exp2["wave_interval_min"]
_m0, _m1 = _trade["marginals"][0], _trade["marginals"][1]


def _review_sec(s: dict) -> float:
    """复核段 = 作业时长（释放→发货）− 等拣货员 − 拣货。"""
    return s["avg_fulfillment_sec"]["mean"] - s["avg_queue_wait_sec"]["mean"] - s["avg_pick_sec"]["mean"]


# 时长按环节拆开，而不只看总高：端到端里最大的一段是**波次累积等待**，它由作业组织
# （波次窗口）决定、加多少人都不会动；人力能压的只有「等拣货员」那一段。不拆开，
# 「加人到底改了什么」在图上没有答案。
_staff_tbl = pd.DataFrame([{
    "拣货员人数": int(_a),
    "波次累积等待(秒)": whatif[_a]["avg_wave_wait_sec"]["mean"],
    "等拣货员(秒)": whatif[_a]["avg_queue_wait_sec"]["mean"],
    "拣货(秒)": whatif[_a]["avg_pick_sec"]["mean"],
    "复核(秒)": _review_sec(whatif[_a]),
    "端到端合计(秒)": whatif[_a]["avg_order_to_ship_sec"]["mean"],
    "拣货员利用率": whatif[_a]["picker_utilization"]["mean"],
    "日人力成本(元)": whatif[_a]["daily_labor_cost"],
} for _a in _arms])
_rev_lo, _rev_hi = _review_sec(whatif[_arms[0]]), _review_sec(whatif[_arms[-1]])

UI.chart_block(
    CH.stacked_bars(
        [f"{_a} 人" for _a in _arms],
        {
            "波次累积等待": [whatif[_a]["avg_wave_wait_sec"]["mean"] for _a in _arms],
            "等拣货员": [whatif[_a]["avg_queue_wait_sec"]["mean"] for _a in _arms],
            "拣货": [whatif[_a]["avg_pick_sec"]["mean"] for _a in _arms],
            "复核": [_review_sec(whatif[_a]) for _a in _arms],
        },
        unit=" 秒/单",
    ),
    caption=(
        f"时长按环节拆开堆叠（自下而上：波次累积等待 / 等拣货员 / 拣货 / 复核）。"
        f"当前选中 {_pick} 人档：端到端 {_sel['avg_order_to_ship_sec']['mean']:.0f} 秒/单，"
        f"其中作业段（释放→发货）{_sel['avg_fulfillment_sec']['mean']:.1f} 秒"
        f"（95% CI {_sel['avg_fulfillment_sec']['ci95_low']:.1f}–"
        f"{_sel['avg_fulfillment_sec']['ci95_high']:.1f}），"
        f"拣货员利用率 {_sel['picker_utilization']['mean']:.1%}，"
        f"日人力成本 {_sel['daily_labor_cost']:.0f} 元。"
        # 拐点有没有，由产物说了算（见 `warehouse_sim.tradeoff_curve_and_knee`）：
        # 只有各档均值差先过显著性检验、再逐档递减，才算有拐点。
        + (f"**边际收益递减，拐点在 {_trade['knee_at_pickers']} 人**："
           if _trade.get("knee_at_pickers") else
           "**各档时长差异与 0 不可区分——这个负载与作业组织下分不出拐点**：")
        + f"{_m0['from_pickers']}→{_m0['to_pickers']} 人每投入 1 元省 "
          f"{_m0['sec_saved_per_yuan']:.4f} 秒，"
          f"{_m1['from_pickers']}→{_m1['to_pickers']} 人 "
          f"{_m1['sec_saved_per_yuan']:.4f} 秒。"
        f"**最底下那段（波次累积等待约 {_sel['avg_wave_wait_sec']['mean']:.0f} 秒）加多少人都不动**"
        f"——它由波次窗口（当前 {_wave_min:.0f} 分钟）决定，是比人力大一个量级的杠杆。"
        f"复核段还从 {_rev_lo:.0f} 秒升到 {_rev_hi:.0f} 秒（{_arms[0]}→{_arms[-1]} 人）："
        f"拣货侧加到 {_arms[-1]} 人后，瓶颈已经转到 {C.SIM_REVIEW_STATIONS} 个复核台。"
        "该图无日期/品类/片区维度，不受全局筛选影响。"
    ),
    table=_staff_tbl,
    table_label="人力档位数据表",
    height=380,
)

st.divider()

# ---------------------------------------------------------------------------
# 八、数据来源与口径
# ---------------------------------------------------------------------------
with st.expander("数据来源与口径（每个数字的追溯入口）"):
    st.markdown(
        f"""
| 区块 | 读数 | 产物文件 | 生成命令 |
|---|---|---|---|
| 逐日仓内指标 | 库存准确率 / 盘差率 / 收货及时率 / 拣货效率 | `{C.WAREHOUSE_DAILY_CSV.name}` | `python -m src.warehouse_kpi` |
| ABC 帕累托 | SKU 出库行数占比与累计占比（ADR-0002） | `{C.WAREHOUSE_ABC_PARETO_CSV.name}` | `python -m src.warehouse_kpi` |
| 分时段拣货 | 各小时单行耗时与行数 | `{C.WAREHOUSE_DRILLDOWN_HOUR_CSV.name}` | `python -m src.warehouse_kpi` |
| 库位热力图 | 库位 row/col/zone + 出库行数 | `{C.WAREHOUSE_TABLES['location'].name}` + `{C.WAREHOUSE_TABLES['outbound'].name}` | `python -m src.gen_warehouse_data` |
| 盘点差异品类 | 各品类差异率与记录数 | `{C.WAREHOUSE_DRILLDOWN_CATEGORY_CSV.name}` | `python -m src.warehouse_kpi` |
| Olist 履约 | 逐单履约时长 / 州下钻 / 总体 KPI | `clean_orders.csv` + `drilldown_by_state.csv` + `overall_kpi.json` | `python -m src.olist_clean` / `python -m src.olist_kpi` |
| 仿真对照实验 | 实验一布局对照、人力 what-if 档位 | `exp1_layout.json` / `exp2_staffing.json` / `whatif_staffing.json` | `python -m src.warehouse_sim` |

- 仓内指标与仿真来自**仿真仓**（数据层 A/F）；Olist 对比来自**真实电商履约数据**（数据层 B）。
  两者口径不同，页面各处已分别标注，请勿混用。
- ABC 数据源按 ADR-0002 取**仿真仓出库行数**，不是 Olist 商品销量。
- 「盘点差异率」（逐记录）与「库存准确率」（金额加权）是两个口径，不可互换。
- 本页不显示 0 或占位值来掩盖缺产物；产物不全时直接列出缺哪个文件、该跑哪条命令。
"""
    )
