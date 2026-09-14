"""驾驶舱 KPI 的窗口聚合与环比（12 号票）。

**纯函数，不 import Streamlit**——这样「筛选窗口一变、卡片数字怎么重算」这件事能被
单元测试直接断言，而不是只能靠在页面上肉眼看。看板层只负责把这里算出的数字画出来。

每个指标一律给出 `basis`（口径一句话），由卡片显示在数字下方。理由：驾驶舱的四张卡片
来自三个不同模块、四种分母，不写口径的话「满载率 49.8%」和「满载率 77.0%」看起来
必然有一个是错的——实际上它们是车队平均与单趟平均两个口径。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_DATE = "date"

#: 成本口径的显示名（**单一定义**：侧边栏、卡片标签、页面文案都从这里取，
#: 三处各写一份字典，改一处就会漂）。
MODE_LABELS: dict[str, str] = {"ev": "纯电租赁", "diesel": "柴油自购"}


@dataclass(frozen=True)
class Window:
    """一个闭区间的日期窗口，以及它的等长前置窗口（环比的对照区间）。"""

    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def days(self) -> int:
        return int((self.end - self.start).days) + 1

    def previous(self) -> "Window":
        """紧邻本窗口之前的**等长**窗口——环比必须与同长度区间比，否则是苹果比橘子。"""
        prev_end = self.start - pd.Timedelta(days=1)
        return Window(prev_end - pd.Timedelta(days=self.days - 1), prev_end)


def slice_window(df: pd.DataFrame, window: Window, date_col: str = _DATE) -> pd.DataFrame:
    """取窗口内的行（闭区间；`date` 列按日归一后比较，不受时间部分影响）。"""
    if df.empty:
        return df
    d = pd.to_datetime(df[date_col]).dt.normalize()
    mask = (d >= window.start.normalize()) & (d <= window.end.normalize())
    return df.loc[mask]


def inventory_accuracy_over(wh_daily: pd.DataFrame, window: Window) -> float | None:
    """窗口内库存准确率 = 1 − Σ|账实差异金额| / Σ账面金额（**金额加权**）。

    先各自求和再相除，而不是把逐日准确率取平均：金额加权是本 KPI 的定义（用件数或
    逐日平均都会让高值商品错发的严重性被稀释）。
    """
    sub = slice_window(wh_daily, window)
    if sub.empty or sub["book_value"].sum() == 0:
        return None
    return float(1 - sub["abs_diff_value"].sum() / sub["book_value"].sum())


def time_window_rate_over(tr_daily: pd.DataFrame, window: Window) -> float | None:
    """窗口内时间窗达成率 = Σ按时单 / Σ已服务单（逐单，ADR-0004）。"""
    sub = slice_window(tr_daily, window)
    served = int(sub["n_orders_served"].sum())
    if sub.empty or served == 0:
        return None
    return float(sub["n_ontime"].sum() / served)


def load_rate_over(tr_daily: pd.DataFrame, window: Window) -> float | None:
    """窗口内满载率 = 逐日满载率的**趟次加权平均**（分母为趟次数）。

    注意口径：这不是「所有趟次的平均满载率」，而是「逐日平均满载率按当日趟次加权」。
    两者在逐日趟次差异不大时非常接近，但不是同一个数；日表没有逐趟载重，故只能到这个
    精度上。产物的口径说明如实写的是加权方式，不写成「平均满载率」。
    """
    sub = slice_window(tr_daily, window)
    trips = float(sub["n_trips"].sum())
    if sub.empty or trips == 0:
        return None
    return float((sub["load_rate_mean"] * sub["n_trips"]).sum() / trips)


def cost_per_order_over(tr_daily: pd.DataFrame, window: Window, mode: str = "ev") -> float | None:
    """窗口内单均运输成本 = Σ当日总成本 / Σ当日订单数（分量 / 分母，不是逐日单均的平均）。"""
    sub = slice_window(tr_daily, window)
    orders = int(sub["n_orders_total"].sum())
    if sub.empty or orders == 0:
        return None
    return float(sub[f"{mode}_cost"].sum() / orders)


@dataclass(frozen=True)
class Metric:
    """一张 KPI 卡片的全部内容：值、格式、口径、以及「变大是好还是坏」。"""

    key: str
    label: str
    value: float | None
    previous: float | None
    unit: str
    basis: str
    higher_is_better: bool
    decimals: int = 2

    @property
    def delta(self) -> float | None:
        if self.value is None or self.previous is None:
            return None
        return self.value - self.previous

    @property
    def delta_pct(self) -> float | None:
        if self.delta is None or not self.previous:
            return None
        return self.delta / abs(self.previous) * 100

    @property
    def direction(self) -> str:
        """`up` / `down` / `flat`；无对比值时为 `na`。"""
        d = self.delta
        if d is None:
            return "na"
        if abs(d) < 1e-12:
            return "flat"
        return "up" if d > 0 else "down"

    @property
    def is_good(self) -> bool | None:
        """环比是好事还是坏事——由指标方向决定，不由箭头方向决定。"""
        d = self.delta
        if d is None or abs(d) < 1e-12:
            return None
        return (d > 0) == self.higher_is_better

    def format(self, v: float | None) -> str:
        if v is None:
            return "—"
        if self.unit == "%":
            return f"{v * 100:.{self.decimals}f}%"
        return f"{v:,.{self.decimals}f} {self.unit}".strip()


def build_metrics(
    wh_daily: pd.DataFrame,
    tr_daily: pd.DataFrame,
    window: Window,
    cost_mode: str = "ev",
) -> list[Metric]:
    """四张驾驶舱卡片的取值与环比（需求 43）。

    `cost_mode` 由 TCO 的建议模式决定（见 11 号票 `recommendation.recommended_mode`），
    并写进卡片标签——车队是 9 电 / 6 柴的混编，「单均成本」不写口径就是个含糊数。
    """
    prev = window.previous()
    mode_label = MODE_LABELS.get(cost_mode, cost_mode)
    return [
        Metric(
            key="inventory_accuracy",
            label="库存准确率",
            value=inventory_accuracy_over(wh_daily, window),
            previous=inventory_accuracy_over(wh_daily, prev),
            unit="%", decimals=2, higher_is_better=True,
            basis="金额加权：1 − Σ|账实差异金额| / Σ账面金额",
        ),
        Metric(
            key="time_window_rate",
            label="时间窗达成率",
            value=time_window_rate_over(tr_daily, window),
            previous=time_window_rate_over(tr_daily, prev),
            unit="%", decimals=2, higher_is_better=True,
            basis="逐单判定：Σ按时单 / Σ已服务单（基线派车口径，含在途延误）",
        ),
        Metric(
            key="load_rate",
            label="满载率",
            value=load_rate_over(tr_daily, window),
            previous=load_rate_over(tr_daily, prev),
            unit="%", decimals=1, higher_is_better=True,
            basis="逐日满载率按当日趟次加权（非逐趟平均，日表无逐趟载重）",
        ),
        Metric(
            key="cost_per_order",
            label=f"单均成本（{mode_label}）",
            value=cost_per_order_over(tr_daily, window, cost_mode),
            previous=cost_per_order_over(tr_daily, prev, cost_mode),
            unit="元/单", decimals=2, higher_is_better=False,
            basis="Σ当日总成本 / Σ当日订单数；全队双模式并列见运输分析页",
        ),
    ]
