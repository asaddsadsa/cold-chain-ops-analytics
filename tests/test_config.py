"""config 模块回归测试。

测试缝：spec「成本模型缝」——TCO 派生函数是纯函数，对手工算例断言。
只测外部行为（输入→输出数值 / 常量约定），不测实现细节。
"""

import pytest

from src import config as C


class TestSeedsAndPaths:
    """种子集中声明 + 路径相对化约定。"""

    def test_seeds_unique(self):
        # 各数据层种子必须互不相同，避免单层重跑污染其他层
        vals = list(C.SEEDS.values())
        assert len(vals) == len(set(vals))

    def test_seeds_registry_covers_layers(self):
        assert set(C.SEEDS) >= {"warehouse", "delivery", "sim"}

    def test_paths_relative_under_root(self):
        # 所有数据路径都应在项目根之下（相对化，无本机绝对路径泄漏）
        for p in (C.RAW_OLIST_DIR, C.GEO_DIR, C.PROCESSED_DIR, C.DELIVERY_DIR):
            assert C.PROJECT_ROOT in p.parents or p == C.PROJECT_ROOT

    def test_no_hardcoded_user_path(self):
        # config 源码不得出现本机用户目录字样
        src = (C.__file__)
        text = open(src, encoding="utf-8").read()
        assert "C:\\Users" not in text and "/Users/" not in text


class TestABCThresholds:
    def test_abc_thresholds_default(self):
        # 默认 A:B:C = 70:20:10（累计 70% / 90% 切点）
        assert C.ABC_THRESHOLDS == (0.70, 0.90)


class TestCostModel:
    """TCO 成本派生纯函数。中位锚点对齐 data_sources_ledger.md 第 4 节。

    三个敏感性参数（司机工资/能源价格/租金）各档独立传入，
    支持 one-at-a-time 分析（变一个、其余留 mid）。
    """

    def test_diesel_fixed_mid(self):
        # 柴油日固定 = 折旧62 + 保险22 + 工资269 = 353
        assert C.cost_fixed_per_day("diesel") == pytest.approx(353.0)

    def test_ev_fixed_mid(self):
        # 纯电日固定 = 租金115 + 工资269 = 384
        # 注：台账「≈385」为月除未取整(3000/26=115.4, 7000/26=269.2→384.6)；
        # config 忠实于文档显式的每日锚点 115+269=384。
        assert C.cost_fixed_per_day("ev") == pytest.approx(384.0)

    def test_diesel_per_km_mid(self):
        # 柴油公里变动 = 燃油0.96 + 尿素0.05 + 维保0.10 = 1.11
        assert C.cost_per_km("diesel") == pytest.approx(1.11)

    def test_ev_per_km_mid(self):
        assert C.cost_per_km("ev") == pytest.approx(0.44)

    def test_daily_total_cost_formula(self):
        # 日总成本 = 日固定 + 里程 × 公里变动
        km = 120.0
        expected = C.cost_fixed_per_day("diesel") + km * C.cost_per_km("diesel")
        assert C.daily_total_cost("diesel", km) == pytest.approx(expected)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            C.cost_fixed_per_day("hybrid")
        with pytest.raises(ValueError):
            C.cost_per_km("hybrid")

    def test_one_at_a_time_sensitivity(self):
        # 单变一个参数、其余留 mid：三个敏感性参数各自独立生效
        base = C.daily_total_cost("diesel", 100.0)
        # 仅调高司机工资 → 固定成本上升，变动不变
        assert C.daily_total_cost("diesel", 100.0, driver_wage="high") > base
        # 仅调高油价 → 变动成本上升
        assert C.daily_total_cost("diesel", 100.0, energy_price="high") > base
        # 仅调高租金对 diesel 无影响（柴油无租金）
        assert C.daily_total_cost("diesel", 100.0, rent="high") == pytest.approx(base)
        # 租金对 ev 有影响
        assert C.daily_total_cost("ev", 100.0, rent="high") > C.daily_total_cost("ev", 100.0)

    def test_breakeven_diesel_cheaper_below(self):
        # 纯电日固定更高、公里变动更低 → 存在唯一盈亏平衡里程；
        # 低于该里程柴油更省，高于该里程纯电更省。
        be = C.breakeven_km()
        assert be > 0
        assert C.daily_total_cost("ev", be * 2) < C.daily_total_cost("diesel", be * 2)
        assert C.daily_total_cost("ev", be / 2) > C.daily_total_cost("diesel", be / 2)

    def test_breakeven_is_intersection(self):
        # 盈亏平衡点处两模式日总成本相等
        be = C.breakeven_km()
        assert C.daily_total_cost("ev", be) == pytest.approx(C.daily_total_cost("diesel", be), rel=1e-6)

    def test_breakeven_independent_of_driver_wage(self):
        # 司机工资对两模式同额，相减抵消 → 盈亏平衡里程不受工资档位影响
        assert C.breakeven_km(driver_wage="low") == pytest.approx(C.breakeven_km(driver_wage="high"))

    def test_sensitivity_three_levels_ordered(self):
        # 三档敏感性：low < mid < high（工资、电价、租金均单调）
        for d in (C.DRIVER_WAGE_PER_DAY, C.EV_ELEC_PER_KM, C.EV_RENT_PER_DAY):
            assert d["low"] < d["mid"] < d["high"]


class TestFleetAndScenario:
    def test_fleet_params(self):
        assert C.FLEET_SIZE == 15
        assert C.RATED_PAYLOAD_T == pytest.approx(1.4)
        assert C.RATED_VOLUME_M3 == pytest.approx(18.0)

    def test_cold_chain_range(self):
        assert C.COLD_CHAIN_TEMP_RANGE == (2.0, 8.0)

    def test_region_and_category_codes(self):
        assert len(C.REGION_CODES) == 8 and C.HIGH_ANOMALY_REGION in C.REGION_CODES
        assert len(C.CATEGORY_CODES) == 10 and C.HIGH_DIFF_CATEGORY in C.CATEGORY_CODES

    def test_whatif_vehicle_range_matches_16_gears(self):
        # 5–20 共 16 档（ADR-0010）
        lo, hi = C.WHATIF_VEHICLE_RANGE
        assert hi - lo + 1 == 16

    def test_solomon_gate(self):
        assert C.SOLOMON_GAP_GATE == pytest.approx(0.10)
        assert C.SOLOMON_INSTANCES == ("C101", "R101", "RC101")
