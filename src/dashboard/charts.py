"""看板图表构造器（13/14/15 号票）：把「该用哪种图、怎么上色」固化成函数。

`components.py` 负责页面骨架（卡片、区块、清单），这里只负责出 `Figure`。分开的理由：
图表的约束比页面布局多（单序列不放图例、多序列必须放、绝不双 Y 轴、顺序编码只用单色相、
状态色不冒充序列色），集中在一处才守得住。

三条硬约束（每条都对应一类真实会犯的错）：

1. **绝不双 Y 轴**。量纲不同的指标一律拆图或换算成同一量纲——帕累托图在这里就是把
   柱子也换成占比，让两条序列共用一根 0–100% 的轴（见 `cumulative_share_chart`）。
2. **类别色按实体固定分配、不循环、不生成第 9 个**。超出就折叠为「其余」或改用小倍数图。
3. **顺序编码只用单色相由浅到深**，不出现彩虹色阶。
"""

from __future__ import annotations

import plotly.graph_objects as go

from src.dashboard import theme


def bar_chart(labels, values, *, label: str, unit: str = "", slot: int = 0,
              decimals: int = 2, orientation: str = "v",
              highlight: dict | None = None, height: int = 320) -> go.Figure:
    """单序列柱状图（纵向/横向）。

    `highlight` 用来**强调**少数几根柱（如埋点品类），其余保持同一颜色——强调的正确做法是
    给少数上色、其余退到同一基色，而不是让每根柱按值深浅换色（那会把柱高在色相上再编码
    一遍，白白烧掉唯一的自由通道）。
    """
    color = theme.series(slot)
    colors = [highlight.get(str(v), color) if highlight else color for v in labels]
    if orientation == "h":
        trace = go.Bar(y=list(labels), x=list(values), orientation="h", name=label,
                       marker={"color": colors, "line": {"width": 0}},
                       hovertemplate="%{y}<br>" + label + " %{x:." + str(decimals) + "f}"
                       + unit + "<extra></extra>")
        fig = go.Figure(trace)
        fig.update_layout(xaxis={"title": unit}, yaxis={"title": "", "autorange": "reversed"})
    else:
        trace = go.Bar(x=list(labels), y=list(values), name=label,
                       marker={"color": colors, "line": {"width": 0}},
                       hovertemplate="%{x}<br>" + label + " %{y:." + str(decimals) + "f}"
                       + unit + "<extra></extra>")
        fig = go.Figure(trace)
        fig.update_layout(yaxis={"title": unit}, xaxis={"title": ""})
    fig.update_layout(showlegend=False, bargap=0.3, height=height,
                      margin={"l": 64, "r": 16, "t": 24, "b": 64})
    return fig


def cumulative_share_chart(labels, share_pct, cumulative_pct, *, height: int = 340,
                           share_label: str = "占比",
                           cumulative_label: str = "累计占比") -> go.Figure:
    """帕累托图：**两根序列都是百分数，共用一根 0–100% 的轴**。

    常见的帕累托图把「数量」画柱、「累计占比」画线，各自的 Y 轴分开——那是双 Y 轴，
    两个刻度的相对位置是任意的。这里把柱也换成**占比**，两条序列同量纲、一根轴就够，
    帕累托的读法（少数类别贡献多数）完整保留。
    """
    fig = go.Figure()
    fig.add_bar(x=list(labels), y=list(share_pct), name=share_label,
                marker={"color": theme.series(0), "line": {"width": 0}},
                hovertemplate="%{x}<br>" + share_label + " %{y:.2f}%<extra></extra>")
    fig.add_scatter(x=list(labels), y=list(cumulative_pct), name=cumulative_label,
                    mode="lines+markers",
                    line={"width": 2, "color": theme.series(1)},
                    marker={"size": 7, "color": theme.series(1)},
                    hovertemplate="%{x}<br>" + cumulative_label + " %{y:.2f}%<extra></extra>")
    fig.update_layout(
        yaxis={"title": "百分比 (%)", "range": [0, 105]},
        xaxis={"title": "", "type": "category"},
        bargap=0.3, height=height, hovermode="x unified",
        margin={"l": 64, "r": 16, "t": 44, "b": 64},
    )
    return fig


def heatmap_grid(x, y, z, *, x_title: str, y_title: str, color_title: str,
                 height: int = 430) -> go.Figure:
    """二维热力图（顺序编码：**单一色相由浅到深**，不用彩虹色阶）。

    相邻格之间留 1px 底缝以分隔色块，而不是给每格描边——描边会把图读成表格。
    """
    t = theme.tokens()
    steps = len(theme.SEQUENTIAL_BLUE) - 1
    fig = go.Figure(
        go.Heatmap(
            x=list(x), y=list(y), z=z,
            colorscale=[[i / steps, c] for i, c in enumerate(theme.SEQUENTIAL_BLUE)],
            colorbar={"title": {"text": color_title, "side": "right"},
                      "tickfont": {"color": t["muted"], "size": 11},
                      "outlinewidth": 0},
            hovertemplate=f"{y_title} %{{y}} ｜ {x_title} %{{x}}<br>{color_title} %{{z:.0f}}"
                          "<extra></extra>",
            xgap=1, ygap=1,
        )
    )
    fig.update_layout(height=height, margin={"l": 56, "r": 16, "t": 24, "b": 48},
                      xaxis={"title": x_title, "showgrid": False},
                      yaxis={"title": y_title, "showgrid": False, "autorange": "reversed"})
    return fig


def histogram(values, *, label: str, unit: str, bins: int = 40, slot: int = 0,
              height: int = 320, x_title: str | None = None) -> go.Figure:
    """单序列分布直方图。"""
    color = theme.series(slot)
    fig = go.Figure(
        go.Histogram(
            x=list(values), nbinsx=bins, name=label,
            marker={"color": color, "line": {"width": 0}},
            hovertemplate=f"{label} %{{x:.2f}}<br>笔数 %{{y}}<extra></extra>",
        )
    )
    fig.update_layout(
        showlegend=False, bargap=0.02, height=height,
        yaxis={"title": "笔数"}, xaxis={"title": x_title or unit},
        margin={"l": 64, "r": 16, "t": 24, "b": 48},
    )
    return fig


def pie_chart(labels, values, *, height: int = 340, max_slices: int = 6) -> go.Figure:
    """饼图：**只用于一眼看构成、且类别 ≤ 6 个**。超出的尾部折叠为「其余」。

    类别色只有 8 个槽位、不生成新色相，故这里主动封顶——硬塞第 7、8 种颜色只会让相邻
    扇区在色盲模拟下互相糊掉。占比接近的类别本就不该用饼图（该用柱），页面文案另行提示。
    """
    labels, values = list(labels), list(values)
    if len(labels) > max_slices:
        order = sorted(range(len(values)), key=lambda i: -values[i])
        keep = order[: max_slices - 1]
        kept_labels = [labels[i] for i in keep]
        kept_values = [values[i] for i in keep]
        kept_labels.append("其余")
        kept_values.append(sum(values) - sum(kept_values))
        labels, values = kept_labels, kept_values
    colors = [theme.series(i) for i in range(len(labels))]
    fig = go.Figure(
        go.Pie(
            labels=labels, values=values, hole=0.42,
            marker={"colors": colors,
                    "line": {"color": theme.tokens()["surface"], "width": 2}},
            textinfo="label+percent", textposition="outside",
            hovertemplate="%{label}<br>%{value:,.0f}（%{percent}）<extra></extra>",
        )
    )
    fig.update_layout(height=height, showlegend=False,
                      margin={"l": 24, "r": 24, "t": 24, "b": 24})
    return fig


def gauge(value: float, *, title: str, unit: str = "%", vmin: float = 0.0,
          vmax: float = 1.0, threshold: float | None = None,
          height: int = 260) -> go.Figure:
    """仪表盘：用于**单值**指标（一个数字就是全部信息）。

    颜色只承担「达标 / 需关注」这一层语义，且必定与数字、标题同时出现——不靠颜色单独表意。
    `threshold` 画出参考线，说明「好」的判据从哪来。
    """
    t = theme.tokens()
    scale = 100 if unit == "%" else 1
    color = theme.STATUS["good"] if (threshold is None or value >= threshold) \
        else theme.STATUS["warning"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * scale,
            number={"suffix": unit, "font": {"size": 30, "color": t["primary_ink"]}},
            gauge={
                "axis": {"range": [vmin * scale, vmax * scale],
                         "tickfont": {"color": t["muted"], "size": 11}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": t["grid"], "borderwidth": 0,
                "threshold": ({"line": {"color": t["secondary_ink"], "width": 2},
                               "thickness": 0.8, "value": threshold * scale}
                              if threshold is not None else None),
            },
            title={"text": title, "font": {"size": 13, "color": t["secondary_ink"]}},
        )
    )
    fig.update_layout(height=height, margin={"l": 24, "r": 24, "t": 48, "b": 12})
    return fig


def slot_bars(labels, values, slots, *, unit: str = "", decimals: int = 2) -> go.Figure:
    """每根柱各自绑定一个类别槽位的柱状图（**按实体固定上色**）。

    与 `bar_chart` 的分工：`bar_chart` 是单序列一个颜色（可用 `highlight` 强调少数几根）；
    这里每根柱代表**一个不同的实体**（如「优化前 / 优化后」两个方案），颜色是身份编码，
    故槽位由调用方按实体指定，且不随筛选后的次序重排。
    """
    fig = go.Figure(
        go.Bar(
            x=list(labels), y=list(values),
            marker={"color": [theme.series(s) for s in slots], "line": {"width": 0}},
            text=[f"{v:,.{decimals}f}" for v in values],
            textposition="outside", cliponaxis=False,
            hovertemplate=f"%{{x}}<br>%{{y:,.{decimals}f}}{unit}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False, bargap=0.35,
                      yaxis={"title": unit}, xaxis={"title": ""})
    return fig


def bar_with_ci(labels, means, lows, highs, *, label: str, unit: str = "",
                slot: int = 0, height: int = 300) -> go.Figure:
    """两臂对照柱状图 + 95% CI 误差棒（单序列，故不放图例）。

    `bar_chart` 不含误差棒，而「两臂各 30 次重复、报均值与 95% CI」是本项目对照实验的
    标准呈现方式（实验一、实验二同理），故把它作为图表词汇的一员放在这里，而不是让每个
    需要它的页面就地补一个。颜色仍取自 theme（同一序列一个颜色，不按臂换色）。
    """
    t = theme.tokens()
    color = theme.series(slot)
    fig = go.Figure(
        go.Bar(
            x=list(labels), y=list(means), name=label,
            error_y={
                "type": "data", "symmetric": False,
                "array": [h - m for h, m in zip(highs, means)],
                "arrayminus": [m - l for m, l in zip(means, lows)],
                "color": t["secondary_ink"], "thickness": 1.2, "width": 8,
            },
            marker={"color": color, "line": {"width": 0}},
            hovertemplate="%{x}<br>" + label + " %{y:.1f}" + unit + "<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False, bargap=0.35, height=height,
                      yaxis={"title": unit}, xaxis={"title": ""},
                      margin={"l": 56, "r": 16, "t": 24, "b": 40})
    return fig


def stacked_bars(labels, segments: dict[str, list[float]], *, unit: str = "",
                 decimals: int = 0, height: int = 340) -> go.Figure:
    """堆叠柱：把同一个总量的**组成部分**摞在一起，看成分，而不只看总量。

    用在实验二的时长分解上——端到端时长 = 波次累积等待 + 等拣货员 + 拣货 + 复核。
    摞起来才看得见「加人能压的只有其中一段」，而那一段恰好不是最大的一段；不拆开，
    「加人到底改了什么」在图上没有答案，只看到一个几乎不动的总高。

    `segments` 是**有序**的「成分名 → 各柱取值」，顺序即自下而上的堆叠次序，由调用方给定：
    谁在下谁先被读到，这件事不该交给字典以外的任何东西决定。多序列，故放图例。
    """
    fig = go.Figure()
    for i, (name, values) in enumerate(segments.items()):
        fig.add_bar(
            x=list(labels), y=list(values), name=name,
            marker={"color": theme.series(i), "line": {"width": 0}},
            hovertemplate="%{x}｜" + name + " %{y:." + str(decimals) + "f}" + unit
                          + "<extra></extra>",
        )
    fig.update_layout(
        barmode="stack", height=height, bargap=0.35,
        yaxis={"title": unit}, xaxis={"title": ""},
        legend={"orientation": "h", "yanchor": "top", "y": -0.12, "x": 0},
        margin={"l": 64, "r": 16, "t": 24, "b": 84},
    )
    return fig


def tco_curve(mode_label: str, mileage, cost, *, slot: int,
              breakeven_km: float, ref_km: float, ref_cost: float) -> go.Figure:
    """单条 TCO 曲线：里程—日总成本，标出盈亏平衡里程与参考里程处的成本。

    单序列故**不放图例**（标题即序列名）；颜色按实体固定分配（柴油永远槽 0、纯电永远槽 1），
    不因切换模式而重排。参考线与标注只用基建的铬色 token，不写十六进制色值。
    """
    t = theme.tokens()
    color = theme.series(slot)
    fig = go.Figure(
        go.Scatter(
            x=list(mileage), y=list(cost), mode="lines+markers", name=mode_label,
            line={"width": 2, "color": color}, marker={"size": 6, "color": color},
            hovertemplate="%{x:.0f} km<br>" + mode_label + " %{y:,.2f} 元/日<extra></extra>",
        )
    )
    fig.add_vline(
        x=float(breakeven_km), line={"color": t["secondary_ink"], "width": 1},
        annotation_text=f"盈亏平衡 {breakeven_km:.1f} km",
        annotation_position="top left",
        annotation_font={"color": t["secondary_ink"], "size": 11},
    )
    fig.add_annotation(
        x=ref_km, y=ref_cost, text=f"参考里程 {ref_km:.1f} km / {ref_cost:,.2f} 元",
        showarrow=True, arrowhead=2, ax=44, ay=-28,
        font={"color": t["secondary_ink"], "size": 12},
    )
    fig.update_layout(
        showlegend=False, height=340,
        title={"text": f"{mode_label} 日总成本曲线（TCO）",
               "font": {"size": 14, "color": t["primary_ink"]}, "x": 0},
        yaxis={"title": "日总成本 (元)"}, xaxis={"title": "单车日行驶里程 (km)"},
        margin={"l": 64, "r": 16, "t": 48, "b": 56},
    )
    return fig


def dual_line_chart(x, series: dict, *, y_title: str, unit: str = "",
                    slots: dict[str, int] | None = None,
                    height: int = 320) -> go.Figure:
    """多序列折线（**同量纲、共用一根轴**）。≥2 序列时必须有图例。

    量纲不同的曲线不要往这里塞——拆成小倍数图，别用双 Y 轴。

    `slots` 给定时按**实体固定槽位**取色（`{"柴油自购": 2, "纯电租赁": 3}`）；不给则按
    序列在字典里的次序从槽 0 起取。同一个页面里画多张图时**必须传 `slots`**：槽位是身份
    编码，不传的话「第 1 条序列」在这张图是柴油、在那张图是「整体异常率」，同一个蓝就有了
    两种读法——那正是配色规范要禁止的漂移。（此前页面只能事后逐条重写 trace 颜色来绕过，
    那是把构造器的漏洞补在调用方。）
    """
    fig = go.Figure()
    for i, (name, ys) in enumerate(series.items()):
        color = theme.series(slots[name] if slots else i)
        fig.add_scatter(x=list(x), y=list(ys), mode="lines+markers", name=name,
                        line={"width": 2, "color": color},
                        marker={"size": 6, "color": color},
                        hovertemplate="%{x}<br>" + name + " %{y:,.2f}" + unit + "<extra></extra>")
    fig.update_layout(
        showlegend=True, height=height,
        yaxis={"title": y_title}, xaxis={"title": ""},
        margin={"l": 64, "r": 16, "t": 44, "b": 48},
    )
    return fig
