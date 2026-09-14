"""config 模块回归测试。

config 只声明数值、不定义模型（成本模型住在 `src/costing.py`，其测试在
`tests/test_costing.py`）。这里测的是**常量约定**：种子唯一、路径相对化、
档位单调、门禁阈值取值。
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


class TestSensitivityLevels:
    """三档敏感性锚点的**单调性**。

    成本派生函数（日固定 / 公里变动 / 日总成本 / 盈亏平衡里程）已搬到 `src/costing.py`，
    它们的回归测试在 `tests/test_costing.py`。留在这里的只是**常量本身**的不变量——
    档位必须递增，否则敏感性分析的方向会反。
    """

    def test_three_levels_ordered(self):
        # low < mid < high（工资、电价、租金均单调）
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
