"""`dashboard/labels.py`：看板表格的列名读法。

这个测试存在的理由：一份映射供二三十张表共用，**漏一列不会有任何东西报错**——表里安安静静
露出一截英文列名，谁也不会去翻。所以这里不从映射本身自查（那只能证明它自洽），而是
**把四个页面真的跑一遍**，逐张表检查表头，再按 AST 守住「页面不得绕过收口点」。
"""

from __future__ import annotations

import ast

import pandas as pd
import pytest

from src import anomaly
from src import config as C
from src.dashboard import labels as L
from src.dashboard import components as UI

PAGES = sorted([C.PROJECT_ROOT / "app.py",
                *(C.PROJECT_ROOT / "pages").glob("*.py")])


def _pages_rendered():
    """跑一遍四个页面，产出 `[(页面名, [每张表的列名元组, …]), …]`。

    模块级缓存：一次 `AppTest` 跑一页要几秒，只为读列名跑两遍不值。
    """
    from streamlit.testing.v1 import AppTest

    out = []
    for page in PAGES:
        at = AppTest.from_file(str(page), default_timeout=180).run()
        assert not at.exception, [str(e.value) for e in at.exception]
        out.append((page.name, [tuple(map(str, d.value.columns)) for d in at.dataframe]))
    return out


@pytest.fixture(scope="module")
def rendered():
    return _pages_rendered()


class TestTheMappingItself:
    def test_renaming_only_touches_the_header(self):
        """换列名只动表头：值、顺序、条数都不变（否则就是借换名改了数）。"""
        df = pd.DataFrame({"date": [1, 2], "distance_km": [3.0, 4.0]})
        got = L.chinese_columns(df)
        assert list(got.columns) == ["日期", "里程 (km)"]
        assert got.equals(df.set_axis(got.columns, axis=1))

    def test_a_column_that_already_reads_as_chinese_is_left_alone(self):
        """页面自建的汇总表本来就是中文列名，还有 `ABC 类别`、`R07 片区异常率` 这类混写
        ——它们不该被要求再登记一次。"""
        df = pd.DataFrame({"片区": [1], "R07 片区异常率": [2.0], "ABC 类别": ["A"]})
        assert L.chinese_columns(df).equals(df)

    def test_an_unregistered_english_column_raises(self):
        """没登记就报错，不回退成英文表头——一个英文表头就是看板没做完的样子，
        静默放过它，它就一直在那儿（同 `data.py`「缺产物即报错，不回退」）。"""
        with pytest.raises(KeyError, match="COLUMN_LABELS"):
            L.chinese_columns(pd.DataFrame({"totally_new_column": [1]}))

    def test_every_anomaly_column_is_covered(self):
        """预警清单的列来自 `anomaly.SEVERITY_COLUMNS`；产物那边多一列，这里必须先补读法。"""
        assert set(anomaly.SEVERITY_COLUMNS) <= set(L.COLUMN_LABELS)

    def test_labels_are_not_empty_or_whitespace(self):
        assert all(v.strip() for v in L.COLUMN_LABELS.values())


class TestEveryRenderedTable:
    """对四个页面的真实渲染结果：表头必须已经是中文，且同一张表里不重名。"""

    def test_all_pages_render_without_error(self, rendered):
        assert len(rendered) == 4, [name for name, _ in rendered]
        # 防空转：真抓到表，下面几条断言才有意义（四个页面合计四十来张）
        assert sum(len(tables) for _, tables in rendered) > 30, rendered

    def test_no_table_shows_a_bare_english_header(self, rendered):
        offenders = [
            f"{page} 第 {i} 张表：{cols}"
            for page, tables in rendered
            for i, cols in enumerate(tables)
            if any(not L.is_readable(c) for c in cols)
        ]
        assert not offenders, "这些表格还在露英文列名：\n" + "\n".join(offenders)

    def test_no_table_repeats_a_header(self, rendered):
        """同一张表里两列读成同一个名字，读者就没法指认哪列是哪个数。
        （跨表重名是允许的：`趟次` 在一张表里是趟号、在另一张里是趟数，不会同框。）"""
        offenders = [
            f"{page} 第 {i} 张表：{sorted({c for c in cols if cols.count(c) > 1})}"
            for page, tables in rendered
            for i, cols in enumerate(tables)
            if len(set(cols)) != len(cols)
        ]
        assert not offenders, "这些表格有重复表头：\n" + "\n".join(offenders)


class TestPagesGoThroughTheHelper:
    """页面不得直接调 `st.dataframe`——绕过 `UI.data_table` 就是把产物列名端给读者。"""

    def test_no_direct_st_dataframe_in_pages(self):
        hits = []
        for page in PAGES:
            tree = ast.parse(page.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "dataframe"):
                    hits.append(f"{page.name}:{node.lineno}")
        assert not hits, "页面直接调了 st.dataframe（该用 UI.data_table）：" + "；".join(hits)

    def test_the_helper_itself_goes_through_the_mapping(self):
        src = (C.PROJECT_ROOT / "src" / "dashboard" / "components.py").read_text(encoding="utf-8")
        assert "L.chinese_columns(df)" in src, "UI.data_table 必须走统一映射"


def test_no_ad_hoc_renames_left_in_pages():
    """页面里不该再就地 `.rename(columns=…)` **造表头的读法**——列名的解释只此一处。

    聚合出来的列（`agg(["size","mean"])` 那种）允许就地取名，但它取的必须是**中文**；
    另有 `reset_index()` 产出的 `index` 需要改成产物里的名字（`index` → `plan`），
    那种目标名必须在 `COLUMN_LABELS` 里登记过，才轮得到统一映射接手。
    两条合起来就是：页面可以给列取名，但**不能自己发明一个英文表头**。
    """
    offenders = []
    for page in PAGES:
        for node in ast.walk(ast.parse(page.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "rename"):
                continue
            for kw in node.keywords:
                if kw.arg != "columns" or not isinstance(kw.value, ast.Dict):
                    continue
                for target in kw.value.values:
                    if not isinstance(target, ast.Constant):
                        continue
                    name = str(target.value)
                    if not L._CJK.search(name) and name not in L.COLUMN_LABELS:
                        offenders.append(f"{page.name}:{node.lineno} → {name!r}")
    assert not offenders, "页面自己发明了英文表头（该由 labels.py 统一给）：" + "；".join(offenders)
