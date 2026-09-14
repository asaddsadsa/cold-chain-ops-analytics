"""页面引导：四张页面各自复制的那套开头，收成一个入口。

**顺序是有讲究的，也正因为有讲究才值得收进来而不是各页抄一遍**：
`set_page_config` 必须是脚本里第一个 Streamlit 调用 → 注册 Plotly 模板 →
**先查产物再读产物**（缺产物要报错停下，不能先读出一个空表）→ 页头 → 侧边栏 →
上下文条（它把「本页实际吃哪几个筛选器」写在页面上，而不是留在文档里）。

先前这五步在 `app.py` 与三个页面里各写一遍。抄错的代价不是多几行，而是顺序被悄悄改掉——
例如把读产物提到 `require_artifacts` 之前，缺产物时页面不会停下，只会渲染出空图。
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from src.dashboard import components as UI
from src.dashboard import data as D
from src.dashboard import filters as F
from src.dashboard import theme
from src.dashboard.filters import Filters


def bootstrap(
    *,
    page_title: str,
    page_icon: str,
    title: str,
    subtitle: str | Callable[[pd.Timestamp, pd.Timestamp], str],
    applied: str,
    note: str | None = None,
    artifacts: tuple[str, ...] | None = None,
) -> Filters:
    """跑完页面开头那五步，返回本轮生效的筛选条件。

    `artifacts` 给定时只检查该子集（每张子页只声明自己真正要用的产物），不给则检查全部必需项。
    必须声明的一律声明：漏声明会让「缺哪个文件、跑哪条命令」的那段提示少一条。

    `subtitle` 传字符串就直接用；传零参以外的函数则收到 `(first, last)`——数据的日期跨度。
    需要先读产物才能拼出副标题的页面（例如运输页要显示代表日）用得上它，因为产物的读取必须
    发生在 `require_artifacts` **之后**。
    """
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    theme.register_plotly_template()
    D.require_artifacts(artifacts)
    first, last = D.order_window()
    UI.page_header(title, subtitle(first, last) if callable(subtitle) else subtitle)
    f = F.sidebar()
    UI.context_bar(f, applied, note)
    return f
