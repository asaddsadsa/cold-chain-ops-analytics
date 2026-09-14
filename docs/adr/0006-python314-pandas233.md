# 依赖基线锁定 Python 3.14.6 + pandas 2.3.3

本机默认解释器与项目 `.venv` 均为 Python 3.14.6。经 `pip install --dry-run` 解析验证，全技术栈在 3.14.6 下均可安装、关键二进制包都有 cp314 wheel（ortools 9.15、pandas、numpy 2.5、scipy 1.18、shapely、pyproj、pyogrio），其余（streamlit、simpy、plotly、folium、osmnx、kagglehub、faker）为纯 Python wheel。因此**无需降级到 3.12**，避免给复现者增加多版本环境负担。（注：原技术栈的 streamlit-echarts 已移除，见 ADR-0011；openpyxl 已移除，见 ADR-0012。）

pandas 锁定 **2.3.3** 而非 pip 默认解析到的 3.0.5。pandas 3.0 默认强制 Copy-on-Write、字符串列默认改用 PyArrow 后端，对一个以「可复现、工程稳」为最高诉求、含大量清洗/聚合/KPI 计算的工程是隐性风险源（原地修改 DataFrame 的写法行为会静默变化）。2.3.3 同样提供 cp314 wheel，生态与下游库（OR-Tools/geopandas）验证更充分。`requirements.txt` 固定全部依赖版本。

## Considered Options

- 降级到 Python 3.12：本机没有 3.12，徒增环境负担，且 3.14 已验证全栈可装。
- 用 pandas 3.0.5：跟随最新，但破坏性变更带来的不确定性不值得，本项目不靠 pandas 新特性。

## Consequences

- 可逆决策：实现期若发现某依赖强制要求 pandas 3.0，再上调并回归测试 CoW 影响。
