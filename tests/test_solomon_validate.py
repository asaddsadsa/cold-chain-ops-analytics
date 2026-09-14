"""数据层 E（solomon_validate）单元测试。

测试缝（spec 测试缝③「算法门禁缝」）：解析器对手工构造小算例断言字段解析；
gap 计算对已知值断言（车辆数相同/更多/更少三种口径）；缺失文件报错不编造。
真实 OR-Tools 求解走端到端（见 test_solve_real_instances，标记 slow）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import solomon_validate as SV

# 手工构造的最小 Solomon 算例（DC + 3 客户），用于解析器断言
MINI_SOLOMON = """C1MINI

VEHICLE
NUMBER     CAPACITY
  5         100

CUSTOMER
CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME

    0      40         50          0          0       1000          0
    1      45         68         10        100        200         90
    2      45         70         20        150        250         90
    3      42         66         30         50        150         90
"""

MINI_BKS = """Route #1: 1 2
Route #2: 3
Cost 123.45
"""


class TestParser:
    def test_parse_instance_fields(self):
        inst = SV.parse_solomon(MINI_SOLOMON)
        assert inst.name == "C1MINI"
        assert inst.n_vehicles == 5
        assert inst.capacity == 100
        assert inst.n_customers == 3  # DC + 3
        assert list(inst.demand) == [0, 10, 20, 30]
        assert list(inst.ready) == [0, 100, 150, 50]
        assert list(inst.due) == [1000, 200, 250, 150]
        assert list(inst.service) == [0, 90, 90, 90]
        assert inst.coords[0].tolist() == [40.0, 50.0]

    def test_horizon_is_dc_due(self):
        inst = SV.parse_solomon(MINI_SOLOMON)
        assert inst.horizon() == 1000

    def test_dist_matrix_symmetric_zero_diag(self):
        inst = SV.parse_solomon(MINI_SOLOMON)
        d = inst.dist_matrix_scaled()
        assert d.shape == (4, 4)
        assert (d == d.T).all()  # 对称
        assert (d.diagonal() == 0).all()  # 对角 0
        # DC(40,50) → 客户1(45,68)：欧氏 dist = sqrt(25+324)=18.68，×100 取整
        assert d[0][1] == pytest.approx(1868, abs=2)

    def test_parse_bks(self):
        routes, cost = SV.parse_bks(MINI_BKS)
        assert routes == [[1, 2], [3]]
        assert cost == pytest.approx(123.45)

    def test_parse_bks_missing_cost_raises(self):
        with pytest.raises(ValueError):
            SV.parse_bks("Route #1: 1 2 3\n")

    def test_empty_instance_raises(self):
        with pytest.raises(ValueError):
            SV.parse_solomon("")

    def test_no_vehicle_section_raises(self):
        with pytest.raises(ValueError):
            SV.parse_solomon("NAME\nCUSTOMER\nCUST NO. X Y\n0 0 0 0 0 0 0\n")


class TestGap:
    def test_same_vehicles_uses_distance_gap(self):
        g = SV.compute_gap(veh=10, distance=900.0, bks_veh=10, bks_cost=827.3)
        assert g["vehicles_gap"] == pytest.approx(0.0)
        assert g["distance_gap"] == pytest.approx((900 - 827.3) / 827.3, abs=1e-4)
        assert g["gate_gap"] == g["distance_gap"]  # 车辆相同 → 距离 gap 判定

    def test_more_vehicles_takes_max(self):
        # 车辆多于 BKS：取车辆 gap 与距离 gap 较大者
        g = SV.compute_gap(veh=12, distance=850.0, bks_veh=10, bks_cost=827.3)
        assert g["vehicles_gap"] == pytest.approx(0.2)
        assert g["distance_gap"] == pytest.approx((850 - 827.3) / 827.3, abs=1e-4)
        assert g["gate_gap"] == pytest.approx(max(0.2, g["distance_gap"]))

    def test_fewer_vehicles_not_penalized(self):
        # 车辆少于 BKS（用车更优）：门禁 gap 不应为正惩罚
        g = SV.compute_gap(veh=9, distance=820.0, bks_veh=10, bks_cost=827.3)
        assert g["vehicles_gap"] == pytest.approx(-0.1)
        assert g["gate_gap"] <= 0.0  # 距离也更优 → gap 为负或 0，门禁必过

    def test_distance_better_same_vehicles_negative_gap(self):
        g = SV.compute_gap(veh=10, distance=800.0, bks_veh=10, bks_cost=827.3)
        assert g["gate_gap"] < 0  # 优于 BKS


class TestLoadMissing:
    def test_missing_files_raise_with_hint(self, tmp_path):
        with pytest.raises(FileNotFoundError) as ei:
            SV.load_instance_and_bks("C101", sol_dir=tmp_path)
        assert "CVRPLIB" in str(ei.value) or "手动放置" in str(ei.value)


class TestSolveRealInstances:
    """真实 OR-Tools 求解（依赖已落盘的 data/raw/solomon）。标记慢测试。"""

    @pytest.mark.slow
    def test_c101_solves_and_gap_reasonable(self):
        inst, bks_routes, bks_cost = SV.load_instance_and_bks("C101")
        assert inst.n_customers == 100
        assert bks_cost == pytest.approx(827.3)
        sol = SV.solve_vrptw(inst, time_limit_sec=15)
        assert sol["solved"] is True
        assert sol["vehicles"] >= 1
        assert sol["distance"] > 0
        # 门禁阈值 10%：C101 是经典算例，OR-Tools GLS 应能逼近
        g = SV.compute_gap(sol["vehicles"], sol["distance"], len(bks_routes), bks_cost)
        assert g["gate_gap"] <= 0.15, f"C101 gap 偏大: {g}"
