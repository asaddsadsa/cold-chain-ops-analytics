# 看板图表去掉 streamlit-echarts，统一用 Plotly

需求技术栈列了 `streamlit-echarts`。实现 01 号票实装依赖时发现：`streamlit-echarts 0.7.0` 是停更的孤儿包，仍用旧版 Streamlit 自定义组件加载机制，与 `streamlit 1.63` 运行时不兼容——`pip install` 能装上，但 `import` 即抛 `StreamlitAPIException: Component 'streamlit-echarts.streamlit_echarts' must be declared in pyproject.toml with asset_dir`。一个装得上却用不了的依赖不该留在锁文件里。

我们决定：**从 requirements.txt 移除 streamlit-echarts，看板全部图表统一用 Plotly**（已装、已验证导入成功，覆盖原 echarts 承担的仪表盘 / 饼图 / 热力图 / 帕累托双轴等全部图表类型）+ Folium 地图。这属于需求第 25 行授权的技术栈「减一个停更库、用等价库替代」，不删减任何图表 / KPI / 埋点。

## Considered Options

- 降级 Streamlit 到能跑 echarts 0.7.0 的旧版：旧 streamlit 会连锁要求旧 pandas，直接威胁 ADR-0006 钉死的 pandas 2.3.3；且旧 pandas 在 Python 3.14 上大概率没有 cp314 wheel，整套可能在 3.14 装不上。为保住一个孤儿图表库而动摇整个依赖基线，得不偿失。
- 引入 streamlit-echarts 的社区维护 fork：引入非官方依赖，长期维护与安全性不确定，仍可能与最新 streamlit 有兼容缝。

## Consequences

- 看板票（12–15）的图表实现一律基于 Plotly（配 Folium 地图），不再出现 echarts API。
- requirements.txt 与 spec 技术栈行已同步移除 streamlit-echarts；ADR-0006 中「全栈可装」的包列表不再包含它。
- Plotly 已具备全部所需图表能力，看板表现力无损。
