"""侧边栏全局筛选器（12 号票，需求 47）。

**一组筛选器，作用于整页**——不放在任何单张图的卡片里。四个控件：

| 控件 | 作用域 | 默认 |
|---|---|---|
| 日期范围（预设 + 自定义） | 所有时间序列与卡片的重算窗口 | 最近 30 天 |
| 品类 | 仓储侧（有品类维度的表） | 全部 |
| 配送区域 | 运输侧（有片区维度的表） | 全部 |
| 成本口径（纯电/柴油） | 单均成本与 TCO 相关读数 | 纯电（TCO 建议模式） |

两条必须说清的边界，都在页面上以 `scope_note` 显示，不藏在文档里：

1. **「最近 N 天」相对的是数据的最后一天，不是今天。** 这是一个 2026-06-01~08-29 的
   历史模拟窗口；按「今天」算会得到一段空区间，看起来像看板坏了。
2. **没有对应维度的页面，该筛选器不生效。** 例如配送订单表没有品类字段，运输页就吃不到
   品类筛选。与其假装生效（把「全部品类」当成一个筛过的结果展示），不如在页面上写明
   本页实际吃哪几个筛选器。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from src import config as C
from src.dashboard import data as D
from src.dashboard import kpis
from src.dashboard.kpis import Window

PRESETS: tuple[tuple[str, int | None], ...] = (
    ("最近 7 天", 7),
    ("最近 30 天", 30),
    ("最近 90 天", 90),
    ("自定义", None),
)


@dataclass(frozen=True)
class Filters:
    """一次渲染周期内生效的全局筛选条件。"""

    window: Window
    categories: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    cost_mode: str = "ev"
    label: str = ""

    @property
    def categories_label(self) -> str:
        return "全部品类" if not self.categories else "、".join(self.categories)

    @property
    def regions_label(self) -> str:
        return "全部片区" if not self.regions else "、".join(self.regions)

    def describe(self) -> str:
        """一行摘要：当前区间 + 品类 + 片区。`label` 未显式给定时由它派生。"""
        return (f"{self.window.start.date()} ~ {self.window.end.date()}（{self.window.days} 天）"
                f" ｜ {self.categories_label} ｜ {self.regions_label}")

    def scope_note(self, applied: str) -> str:
        """本页实际吃哪些筛选器。`applied` 形如 "日期范围、配送区域"。"""
        return f"本页生效的筛选器：**{applied}** ｜ 当前：{self.label or self.describe()}"


def _preset_window(last_day: pd.Timestamp, preset: str, custom: tuple) -> Window:
    days = dict(PRESETS)[preset]
    if days is None:
        start, end = pd.Timestamp(custom[0]), pd.Timestamp(custom[1])
        return Window(min(start, end), max(start, end))
    return Window(last_day - pd.Timedelta(days=days - 1), last_day)


def sidebar() -> Filters:
    """渲染侧边栏并返回本轮生效的筛选条件（页面顶部调用一次）。"""
    st.sidebar.markdown("### 全局筛选")

    # 「最近 N 天」锚在**数据的最后一天**，不是今天（见模块 docstring 边界 1）
    last_day = D.order_window()[1]
    st.sidebar.caption(f"数据窗口末日：{last_day.date()}（预设区间以此为锚点，非今天）")

    preset = st.sidebar.radio("日期范围", [p for p, _ in PRESETS], index=1,
                              horizontal=True, label_visibility="visible")
    if preset == "自定义":
        first_day = D.order_window()[0]
        picked = st.sidebar.date_input(
            "自定义区间", value=(last_day - pd.Timedelta(days=29), last_day),
            min_value=first_day, max_value=last_day,
        )
        if isinstance(picked, (list, tuple)) and len(picked) == 2:
            window = _preset_window(last_day, preset, picked)
        else:
            st.sidebar.warning("请选择起止两天；暂按最近 30 天展示。")
            window = _preset_window(last_day, "最近 30 天", ())
    else:
        window = _preset_window(last_day, preset, ())

    codes = list(C.REGION_CODES)
    regions_df = D.artifact("regions")
    if regions_df.shape[0]:
        codes = [c for c in regions_df["region"].tolist()] or codes
    regions = st.sidebar.multiselect("配送区域", codes, default=[], placeholder="全部片区")

    sku = D.artifact("sku_master")
    if sku.empty:
        st.sidebar.multiselect("品类", [], disabled=True,
                               help="需要数据层 A 的 sku_master（python -m src.gen_warehouse_data）")
        categories: tuple[str, ...] = ()
    else:
        cats = sorted(sku["category"].dropna().unique().tolist())
        categories = tuple(st.sidebar.multiselect("品类", cats, default=[], placeholder="全部品类"))

    tco = D.artifact("transport_tco")
    rec = (tco.get("recommendation") or {}).get("recommended_mode", "ev")
    mode_label = kpis.MODE_LABELS
    picked_mode = st.sidebar.radio(
        "成本口径", list(mode_label.values()),
        index=0 if rec == "ev" else 1,
        help="车队为 9 电 / 6 柴混编，「单均成本」必须写明按哪种模式计价（TCO 建议见运输页）",
    )
    cost_mode = "ev" if picked_mode == mode_label["ev"] else "diesel"

    st.sidebar.divider()
    if st.sidebar.button("刷新产物读数", help="脚本重跑出新产物后点此重读；不必重启看板"):
        D.refresh()

    f = Filters(
        window=window,
        categories=categories,
        regions=tuple(regions),
        cost_mode=cost_mode,
    )
    st.session_state["filters"] = f
    return f


def apply_regions(df: pd.DataFrame, f: Filters, col: str = "region") -> pd.DataFrame:
    """按配送区域筛选（空选 = 全部）。列不存在时原样返回——页面已用 `scope_note` 声明边界。"""
    if not f.regions or col not in df.columns:
        return df
    return df[df[col].isin(f.regions)]
