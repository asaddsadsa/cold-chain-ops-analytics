"""看板统一配色与图表模板（12 号票）。

配色方案**不是随手挑的**：八个类别色按固定顺序分配（永不循环、永不生成第 9 个），
顺序本身是色盲安全机制——它经**实际校验脚本**逐对判定，满足「相邻对 CVD ΔE ≥ 8、
常视觉 ΔE ≥ 15」的硬门槛，浅色/深色两套各自在**各自底色**上通过。

改动色值前必须先重跑校验，不能凭眼睛判断。校验用什么跑：Data Visualization 技能的
`validate_palette.js`（**不在本仓库内**，随技能分发）。当前这套值的复核记录：

```
node validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4,#008300,#4a3aa7,#e34948" --mode light
  → ALL CHECKS PASS；CVD 最差相邻对 ΔE 9.1（黄↔青，protan），常视觉最差 19.6（品红↔黄）
  → WARN 对比度：青 2.74 / 黄 2.11 / 品红 2.62 低于 3:1 → 触发缓解规则（见下）
node validate_palette.js "#3987e5,#d95926,#199e70,#c98500,#d55181,#008300,#9085e9,#e66767" --mode dark
  → ALL CHECKS PASS；CVD 最差相邻对 ΔE 8.4，常视觉最差 19.3，对比度全部 ≥ 3:1
```

参考实例（浅色 / 深色两列是同一批色相针对不同底色重新取的步阶，不是简单的明暗翻转）：

| 槽位 | 色相 | 浅色 | 深色 |
|---|---|---|---|
| 1 | 蓝 | `#2a78d6` | `#3987e5` |
| 2 | 橙 | `#eb6834` | `#d95926` |
| 3 | 青 | `#1baf7a` | `#199e70` |
| 4 | 黄 | `#eda100` | `#c98500` |
| 5 | 品红 | `#e87ba4` | `#d55181` |
| 6 | 绿 | `#008300` | `#008300` |
| 7 | 紫 | `#4a3aa7` | `#9085e9` |
| 8 | 红 | `#e34948` | `#e66767` |

浅色模式下青 / 黄 / 品红三槽对底色的对比度低于 3:1，因此**每张图都必须配表格视图**
（这是配色规范的「缓解规则」，不是可选项）；看板里由 `components.chart_block` 统一提供，
调用方**必须传 `table=`**——不传就等于把这条硬约束说成了废话。
"""

from __future__ import annotations

#: 类别色（身份编码）：按固定顺序分配，第 9 个序列应折叠为「其他」或改用小倍数图，
#: 绝不生成新色相——生成的色相在色盲模拟下必然与已有槽位撞车。
CATEGORICAL_LIGHT: tuple[str, ...] = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
CATEGORICAL_DARK: tuple[str, ...] = (
    "#3987e5", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
)

#: 顺序（连续）编码用的单色阶：一个色相、由浅到深。**禁止彩虹色阶**。
SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

#: 发散编码：蓝↔红两极 + **中性灰**中点（中点必须读作「什么都没有」，
#: 所以不能用任何一个色相；两极必须读作相反，所以不能两个都是冷色）。
DIVERGING = {"low": "#2a78d6", "mid": "#f0efec", "high": "#e34948"}

#: 状态色：**保留语义**，绝不拿来当「第 4 个序列」用。永远配图标 + 文字，不靠颜色单独表意。
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
STATUS_ICON = {"good": "✅", "warning": "⚠️", "serious": "🔶", "critical": "⛔"}

#: 图表铬色与墨色（浅 / 深）。网格与坐标轴一律**实线发丝线**，一档于底色，
#: 不用虚线——虚线会被读成「预测」或「阈值」。
CHROME: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb", "page": "#f9f9f7",
        "primary_ink": "#0b0b0b", "secondary_ink": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "delta_good": "#006300",
        "border": "rgba(11,11,11,0.10)",
    },
    "dark": {
        "surface": "#1a1a19", "page": "#0d0d0d",
        "primary_ink": "#ffffff", "secondary_ink": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "delta_good": "#0ca30c",
        "border": "rgba(255,255,255,0.10)",
    },
}

#: 中文字体栈：西文用系统无衬线，中文回落到各平台自带黑体。
#: 数字与标题同一套字体——**大号数字禁止用衬线/展示体**，也不用 `tabular-nums`
#: （等宽数字在大字号下会显得松散；只有需要竖向对齐的表格与刻度才用）。
FONT_STACK = (
    'system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", '
    '"Hiragino Sans GB", "Noto Sans CJK SC", sans-serif'
)

#: 打印（PDF）字体栈：同一批字体，但把中文黑体**提到系统字体之前**。
#: 屏幕渲染有系统级的 CJK 回退链，打印走无头浏览器时那条链不可靠——中文回退失败就是
#: 整页豆腐块，故此处显式前置。它和 `FONT_STACK` 都住在这里，报告层不再各存一份。
FONT_STACK_PRINT = (
    '"Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", '
    'system-ui, -apple-system, "Segoe UI", sans-serif'
)

TEMPLATE_NAME = "ckops"


def tokens(dark: bool = False) -> dict:
    """返回某个模式下的完整色板（类别色 / 顺序色 / 极性色 / 状态色 / 铬色）。"""
    mode = "dark" if dark else "light"
    return {
        "mode": mode,
        "categorical": CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT,
        "sequential": SEQUENTIAL_BLUE,
        "diverging": DIVERGING,
        "status": STATUS,
        **CHROME[mode],
    }


def series(i: int, dark: bool = False) -> str:
    """第 i 个类别槽位的颜色（**按实体固定分配，不随筛选后的次序重排**）。"""
    pal = CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT
    if i >= len(pal):
        raise ValueError(
            f"类别色只有 {len(pal)} 个槽位；第 {i + 1} 个序列应折叠为「其他」或改用小倍数图，"
            f"不能生成新色相（生成色在色盲模拟下必与已有槽位撞车）"
        )
    return pal[i]


def register_plotly_template(dark: bool = False) -> str:
    """注册并返回统一 Plotly 模板名（全局调用一次即可，重复调用会覆盖同名模板）。"""
    import plotly.graph_objects as go
    import plotly.io as pio

    t = tokens(dark)
    axis = {
        "showgrid": True,
        "gridcolor": t["grid"],
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": t["axis"],
        "linewidth": 1,
        "ticks": "outside",
        "tickcolor": t["axis"],
        "ticklen": 4,
        "tickfont": {"color": t["muted"], "size": 12},
        "title": {"font": {"color": t["secondary_ink"], "size": 13}},
    }
    template = go.layout.Template(
        layout={
            "colorway": list(t["categorical"]),
            "font": {"family": FONT_STACK, "color": t["primary_ink"], "size": 13},
            "paper_bgcolor": t["surface"],
            "plot_bgcolor": t["surface"],
            "margin": {"l": 64, "r": 24, "t": 48, "b": 56},
            "hoverlabel": {
                "bgcolor": t["surface"],
                "bordercolor": t["axis"],
                "font": {"family": FONT_STACK, "color": t["primary_ink"], "size": 12},
            },
            "hovermode": "x unified",
            "legend": {
                "orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "left", "x": 0,
                "font": {"color": t["secondary_ink"], "size": 12},
                "bgcolor": "rgba(0,0,0,0)",
            },
            "xaxis": axis,
            "yaxis": axis,
        }
    )
    pio.templates[TEMPLATE_NAME] = template
    pio.templates.default = TEMPLATE_NAME
    return TEMPLATE_NAME
