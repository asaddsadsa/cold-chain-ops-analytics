"""取数层的 interface 测试（读侧收口后新增）。

三层，对应当初定下的测试面：

1. **登记表不变量**——key 唯一、路径是 `Path`、`hint` 指向的模块真实存在、
   四个页面与 `app.py` 声明/引用的每个 key 都已登记。
2. **解释器纯函数**——用 fixture 喂原始 dict，缺 key 一律抛错。这些产物的嵌套路径原先只在
   **页面渲染时**才被解引用，解错了是页面上的 KeyError；而页面按项目决策不做自动化测试。
   把解释写成纯函数，正是为了把这类错误搬进单测的射程。
3. **页面不直接读盘**守卫——现状零违规，固化成回归保护：页面绕过取数层就会绕过 mtime
   缓存键，产物更新后那一处不会自动失效。
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from src import config as C
from src.dashboard import data as D

PAGES = sorted((C.PROJECT_ROOT / "pages").glob("*.py")) + [C.PROJECT_ROOT / "app.py"]


# ---------------------------------------------------------------------------
# 1. 登记表不变量
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_keys_are_unique(self):
        keys = [a.key for a in D.ARTIFACTS]
        assert len(keys) == len(set(keys))

    def test_every_entry_has_a_path_and_a_hint(self):
        for a in D.ARTIFACTS:
            assert isinstance(a.path, Path), a.key
            assert a.hint.strip(), a.key

    def test_hint_names_a_module_that_exists(self):
        """`hint` 是缺产物时给用户抄的命令；它指向的模块必须真实存在。

        核对的是模块可被找到，不是「跑它能补上这个文件」——跨层顺序约束（例如
        `src.warehouse_kpi` 会读 `sim/exp1_layout.json`，缺了只 warning 跳过互证段）
        不在这条断言的范围里。
        """
        for a in D.ARTIFACTS:
            module = a.hint.removeprefix("python -m ").strip()
            assert module.startswith("src."), f"{a.key}: hint 应为 python -m src.xxx"
            assert importlib.util.find_spec(module) is not None, f"{a.key}: {module} 不存在"

    def test_optional_entry_declares_its_empty_structure(self):
        """「可选」必须在登记表里声明空结构，否则缺失时会退化成裸 None。"""
        for a in D.ARTIFACTS:
            if not a.required:
                assert a.empty is not None, a.key

    def test_read_kwargs_carry_the_parse_contract(self):
        """`parse_dates=["date"]` 是两张日表按 date 合并的前提，不是优化——它必须留在登记表里。"""
        for key in ("warehouse_daily", "transport_daily", "delivery_orders", "anomalies"):
            assert D.ARTIFACT_BY_KEY[key].read_kwargs.get("parse_dates") == ["date"]


def _page_artifact_keys(path: Path) -> set[str]:
    """页面里出现的产物 key：`D.require_artifacts((...))` 声明 + `D.artifact("literal")` 引用。

    用 AST 而不是正则：页面 docstring 里也有 `data/processed/` 字样，正则会误报。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else None
        if name == "require_artifacts" and node.args:
            for elt in ast.walk(node.args[0]):
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    keys.add(elt.value)
        if name == "artifact" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                keys.add(arg.value)
    return keys


class TestPagesDeclareOnlyRegisteredArtifacts:
    """登记表是契约：页面声明或引用的 key 必须都在里面。

    重构前 `missing_artifacts` 用 `a.key in set(keys)` 过滤——未登记的 key 被静默丢弃，
    运输页的 `"transport_comparison"` 就这样空转了很久（它从来不是任何 `Artifact.key`）。
    """

    def test_pages_declare_registered_keys(self):
        known = set(D.ARTIFACT_BY_KEY)
        for page in PAGES:
            unknown = _page_artifact_keys(page) - known
            assert not unknown, f"{page.name} 声明/引用了未登记的 key：{sorted(unknown)}"

    def test_every_registered_artifact_has_a_consumer(self):
        """反向检查：每条登记项都要有页面在用，避免登记表长成僵尸清单。"""
        used: set[str] = set()
        for page in PAGES:
            used |= _page_artifact_keys(page)
        # 侧边栏与派生视图消费的产物不直接出现在页面的字面量里
        used |= {"regions", "sku_master", "delivery_orders", "location_master", "outbound"}
        assert not (set(D.ARTIFACT_BY_KEY) - used), \
            f"没有消费者的登记项：{sorted(set(D.ARTIFACT_BY_KEY) - used)}"


class TestPageDoesNotReadDiskDirectly:
    """页面不得自行读盘——绕开取数层就绕过了 mtime 缓存键（data.py 模块 docstring 第 3 条）。"""

    BANNED = {"read_csv", "read_text", "read_bytes", "read_json", "load", "loads", "open"}

    def test_no_banned_io_calls_in_pages(self):
        for page in PAGES:
            tree = ast.parse(page.read_text(encoding="utf-8"))
            hits = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else (
                    func.id if isinstance(func, ast.Name) else None)
                # `st.cache_data` / `st.rerun` 之类不在此列；只盯文件读取
                if name in self.BANNED and not (
                    name == "load" and isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name) and func.value.id == "D"
                ):
                    hits.append(f"{page.name}:{node.lineno} {name}()")
            assert not hits, "页面直接读盘：" + "；".join(hits)


# ---------------------------------------------------------------------------
# 2. 解释器纯函数
# ---------------------------------------------------------------------------
def _transport_raw() -> dict:
    """最小 `transport_kpi.json` 形状（字段名与真实产物一致）。"""
    def plan(dist, veh, tw_rate, ceiling, gap, diesel, ev):
        return {
            "n_vehicles": veh, "total_distance_km": dist,
            "time_window": {"rate": tw_rate, "basis": "实际送达（计划到访 + 该点在途延误）",
                            "ceiling": {"ceiling_rate": ceiling}, "gap_to_ceiling_pp": gap},
            "cost": {"diesel": {"per_order": diesel}, "ev": {"per_order": ev}},
        }

    return {
        "data_category": "情景假设（需求）+ 学术基准（算法）",
        "representative_day": "2026-07-13",
        "baseline": plan(1170.14, 11, 0.7955, 1.0, 20.45, 39.26, 35.90),
        "optimized": plan(876.15, 10, 0.7955, 1.0, 20.45, 34.11, 32.01),
        "baseline_all_days": {"days": 90},
    }


class TestInterpretTransportKpi:
    def test_reads_both_plans_and_the_all_days_denominator(self):
        got = D.interpret_transport_kpi(_transport_raw())
        assert got.representative_day == "2026-07-13"
        assert got.all_days == 90
        assert (got.baseline.n_vehicles, got.optimized.n_vehicles) == (11, 10)
        assert got.baseline.total_distance_km == pytest.approx(1170.14)
        assert got.baseline.diesel_per_order == pytest.approx(39.26)
        assert got.optimized.ev_per_order == pytest.approx(32.01)
        assert got.baseline.time_window_basis.startswith("实际送达")

    def test_missing_field_raises_instead_of_returning_none(self):
        """缺 key 必须抛——返回 None 会让页面把它渲染成「—」，与真实的缺数据混为一谈。"""
        raw = _transport_raw()
        del raw["optimized"]["cost"]["ev"]
        with pytest.raises(KeyError, match="cost.ev"):
            D.interpret_transport_kpi(raw)


def _warehouse_raw() -> dict:
    return {
        "data_category": "过程仿真",
        "abc": {"by_class": {c: {"n_sku": n, "sku_share": 0.2, "line_share": 0.7}
                             for c, n in (("A", 141), ("B", 192), ("C", 167))}},
        "slotting": {"walk_cost_before": 3932220.0, "walk_cost_after": 1538212.5,
                     "reduction_pct": 60.88, "walk_m_per_line_before": 89.12,
                     "walk_m_per_line_after": 34.86, "walk_sec_per_line_saved": 45.22},
        "embedding_checks": {
            # 取值照抄真实产物（`kpi_overall.json`），便于对照阅读
            "pick_slowdown_14_16": {"sec_per_line_14_16": 168.4,
                                    "sec_per_line_other": 88.6, "ratio": 1.9},
            "p03_discrepancy": {"p03_rate": 0.1745, "other_rate": 0.0402, "ratio": 4.34},
        },
        "kpis": {"inventory_accuracy": {"rate": 0.99895}},
    }


class TestInterpretWarehouseKpi:
    def test_reads_abc_slotting_and_both_embedding_points(self):
        got = D.interpret_warehouse_kpi(_warehouse_raw())
        assert got.abc_by_class["A"].n_sku == 141
        assert got.slotting.walk_sec_per_line_saved == pytest.approx(45.22)
        assert got.pick_slowdown.ratio == pytest.approx(1.9)
        assert got.inventory_accuracy_rate == pytest.approx(0.99895)

    def test_p03_field_names_are_position_independent(self):
        """产物里叫 `p03_rate`（带品类前缀），读侧统一成 `rate`——改品类名不该波及读侧。"""
        got = D.interpret_warehouse_kpi(_warehouse_raw())
        assert got.p03_discrepancy.rate == pytest.approx(0.1745)

    def test_missing_embedding_point_raises(self):
        raw = _warehouse_raw()
        del raw["embedding_checks"]["p03_discrepancy"]
        with pytest.raises(KeyError, match="embedding_checks.p03_discrepancy"):
            D.interpret_warehouse_kpi(raw)


def _whatif_raw() -> dict:
    def plan(dist, veh):
        return {"n_vehicles_used": veh, "total_distance_km": dist, "time_window_rate": 0.7955,
                "n_orders_ontime": 105, "n_orders_served": 132, "n_orders_unserved": 0,
                "load_rate_mean": 0.77, "load_rate_max": 0.95, "mileage_utilization": 0.69,
                "cost": {"diesel": {"total": 4503.0, "per_order": 34.11},
                         "ev": {"total": 4226.0, "per_order": 32.01}},
                "solve_time_limit_sec": 30.0}

    def gear(n, feasible):
        """真实产物里档位记录是「扁平字段 + 一个嵌套的 baseline 兄弟键」，不可行档全填 None。"""
        if not feasible:
            return {"n_vehicles_available": n, "feasible": False,
                    "infeasible_reason": "载重下界不满足", "baseline": None,
                    "n_vehicles_used": None, "total_distance_km": None,
                    "time_window_rate": None, "n_orders_ontime": None,
                    "n_orders_served": None, "n_orders_unserved": None,
                    "load_rate_mean": None, "load_rate_max": None,
                    "mileage_utilization": None, "cost": None, "solve_time_limit_sec": 30.0}
        return {"n_vehicles_available": n, "feasible": True, "infeasible_reason": None,
                "baseline": plan(1170.14, 11), **plan(876.15, 10)}

    return {"grid": [5, 6, 9], "representative_day": "2026-07-13", "n_orders": 132,
            "n_nodes": 45, "time_zone": None,
            "gears": {"5": gear(5, False), "6": gear(6, False), "9": gear(9, True)},
            "consistency_check": {"gear": 15, "distance_km": 876.15,
                                  "published_distance_km": 876.15, "distance_delta_pct": 0.0,
                                  "note": "带时限搜索不保证同解"}}


class TestInterpretWhatIf:
    def test_feasible_gear_exposes_the_optimized_plan(self):
        got = D.interpret_whatif_vehicles(_whatif_raw())
        g9 = got.gear(9)
        assert g9.feasible and g9.optimized.total_distance_km == pytest.approx(876.15)
        assert g9.optimized.cost_per_order["ev"] == pytest.approx(32.01)

    def test_infeasible_gear_keeps_its_reason_and_has_no_plan(self):
        """不可行是结论本身——`optimized` 为 None，而不是一个零值的计划。"""
        g5 = D.interpret_whatif_vehicles(_whatif_raw()).gear(5)
        assert not g5.feasible
        assert g5.optimized is None
        assert g5.infeasible_reason

    def test_out_of_grid_returns_none_instead_of_extrapolating(self):
        got = D.interpret_whatif_vehicles(_whatif_raw())
        assert got.gear(7) is None and got.gear(99) is None and got.gear(4) is None

    def test_consistency_check_is_optional(self):
        raw = _whatif_raw()
        raw.pop("consistency_check")
        assert D.interpret_whatif_vehicles(raw).consistency_check is None


# ---------------------------------------------------------------------------
# 3. 缺产物与未知 key
# ---------------------------------------------------------------------------
class TestMissingAndUnknown:
    def test_unknown_key_is_rejected_not_silently_dropped(self):
        for call in (lambda: D.artifact("warehous_kpi"),
                     lambda: D.missing_artifacts(("nope",)),
                     lambda: D.require_artifacts(("nope",)),
                     lambda: D.describe("nope")):
            with pytest.raises(D.UnknownArtifact):
                call()

    def test_optional_artifact_missing_returns_declared_empty_structure(self, tmp_path):
        """可选产物缺失返回登记表声明的空结构，不是裸 None / {}。"""
        ghost = D.Artifact("ghost", tmp_path / "absent.csv", "python -m src.olist_clean",
                           "csv", required=False, empty=pd.DataFrame)
        got = ghost.value()
        assert isinstance(got, pd.DataFrame) and got.empty

    def test_required_artifact_missing_raises_with_hint(self, tmp_path):
        ghost = D.Artifact("ghost", tmp_path / "absent.csv", "python -m src.olist_clean")
        with pytest.raises(D.MissingArtifact) as exc:
            ghost.value()
        assert exc.value.missing[0].key == "ghost"
        assert "python -m src.olist_clean" in str(exc.value)

    def test_missing_artifacts_is_pure_and_checks_only_required(self, tmp_path, monkeypatch):
        ghost = D.Artifact("ghost", tmp_path / "absent.csv", "python -m src.olist_clean")
        optional = D.Artifact("opt", tmp_path / "absent2.csv", "python -m src.olist_clean",
                              required=False, empty=pd.DataFrame)
        monkeypatch.setattr(D, "ARTIFACTS", (ghost, optional))
        monkeypatch.setattr(D, "ARTIFACT_BY_KEY", {a.key: a for a in (ghost, optional)})
        assert [a.key for a in D.missing_artifacts()] == ["ghost"]


class TestMtimeCacheKey:
    """产物一被重写就自动重读——这是「看板数字与报告对不上」最常见的来源。"""

    def test_same_mtime_reads_once_and_new_mtime_reads_again(self, tmp_path, monkeypatch):
        csv = tmp_path / "x.csv"
        csv.write_text("a\n1\n", encoding="utf-8")
        D.st.cache_data.clear()

        calls = []
        real = pd.read_csv

        def counting(*args, **kwargs):
            calls.append(args[0])
            return real(*args, **kwargs)

        monkeypatch.setattr(pd, "read_csv", counting)
        assert D._read_csv(str(csv), D._stamp(csv)).shape == (1, 1)
        assert D._read_csv(str(csv), D._stamp(csv)).shape == (1, 1)
        assert len(calls) == 1, "同 mtime 的第二次调用应命中缓存"

        D._read_csv(str(csv), D._stamp(csv) + 1.0)
        assert len(calls) == 2, "mtime 变了必须重读"
        D.st.cache_data.clear()


# ---------------------------------------------------------------------------
# 4. 对真实产物：三个解释器在已发表数字上站得住
# ---------------------------------------------------------------------------
class TestRealArtifacts:
    """读盘缝：解释器对真实产物解出来的值，必须与台账已发表的数字一致。"""

    def test_transport_kpi_matches_the_ledger(self):
        got = D.transport_kpi()
        assert got.representative_day == "2026-07-13"
        assert got.baseline.total_distance_km > got.optimized.total_distance_km
        assert got.all_days == C.SIM_DAYS
        # 台账：优化后 876.15 km / 用车 10 台 / 柴油单均 34.11 元
        assert got.optimized.n_vehicles == 10
        assert got.optimized.diesel_per_order == pytest.approx(34.11, abs=0.005)

    def test_warehouse_kpi_abc_matches_the_simulation_warehouse(self):
        got = D.warehouse_kpi()
        assert got.abc_by_class["A"].n_sku == 141
        assert got.slotting.reduction_pct == pytest.approx(60.88, abs=0.01)

    def test_whatif_grid_covers_the_configured_range(self):
        got = D.whatif_vehicles()
        assert got.grid[0] == C.WHATIF_VEHICLE_RANGE[0]
        assert got.grid[-1] == C.WHATIF_VEHICLE_RANGE[1]
        assert all(g.feasible == (g.optimized is not None) for g in got.gears.values())

    def test_routes_geojson_maps_plan_to_registry_key(self):
        assert len(D.routes_geojson("baseline")["features"]) >= 1
        assert len(D.routes_geojson("optimized")["features"]) >= 1
        with pytest.raises(ValueError, match="未知路线方案"):
            D.routes_geojson("nope")

    def test_describe_reports_the_composite_category_verbatim(self):
        """一份产物可同时属于多类，读侧不得压成单一类别（CONTEXT.md「复合类别」）。"""
        info = D.describe("transport_kpi")
        assert info.exists and info.mtime > 0
        assert "+" in (info.category or ""), info.category
