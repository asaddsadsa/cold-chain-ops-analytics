"""数据层 E：Solomon VRPTW 标准算例算法验证（05 号票）。

本模块是进入成都场景优化的**算法门禁**：用与模块二完全一致的 OR-Tools 建模
（载重 + 时间窗 + 车辆数最小化 + 距离最小化）求解 C101/R101/RC101，
与从权威源下载的 best-known 最优解对比 gap%，gap>10% 不放行。

算例与 best-known 来源：PUC-Rio CVRPLIB（见 data/raw/solomon/SOURCE.md）。
**严禁凭记忆编造算例数据或最优解**：本地缺失时自动从该源取；取不到则报错并给手动放置说明。

运行方式（项目根）：python -m src.solomon_validate
数据类别：学术基准（见 data_sources_ledger.md 第 1 节）。
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import requests

from src import config as C

logger = logging.getLogger(__name__)

#: 距离缩放因子：OR-Tools 要求整数弧权重，Solomon 距离为浮点（BKS 带 1 位小数）
DIST_SCALE = 100

#: 手动放置说明（算例/BKS 缺失时报错用）
MANUAL_PLACE_HINT = """Solomon 算例或 best-known 缺失。请手动放置后重跑：
  目录：{dir}
  算例文件：{{C101,R101,RC101}}.txt  （标准 Solomon 格式：VEHICLE 段 + CUSTOMER 段）
  最优解文件：{{C101,R101,RC101}}.sol （每行 "Route #k: <客户序列>"，末行 "Cost <总距离>"）
  下载来源（本机可达）：PUC-Rio CVRPLIB
    https://galgos.inf.puc-rio.br/cvrplib/en/instances/2
    实例/解接口：/cvrplib/en/download/instance/{{380,397,420}} 与 /download/bks/{{380,397,420}}
  注意：GitHub 与 SINTEF TOP 官方页在本机网络不可达；best-known 必须来自可机读源，
  禁止凭记忆填写。"""


# ---------------------------------------------------------------------------
# 解析器
# ---------------------------------------------------------------------------
class SolomonInstance:
    """Solomon VRPTW 算例。节点 0 为 DC，1..n 为客户。"""

    def __init__(self, name, n_vehicles, capacity, coords, demand, ready, due, service):
        self.name = name
        self.n_vehicles = n_vehicles
        self.capacity = capacity
        self.coords = coords  # (n+1, 2) 浮点
        self.demand = demand  # (n+1,) 整数，DC 为 0
        self.ready = ready  # (n+1,)
        self.due = due  # (n+1,)
        self.service = service  # (n+1,)

    @property
    def n_customers(self) -> int:
        return len(self.demand) - 1

    def dist_matrix_scaled(self) -> np.ndarray:
        """欧氏距离矩阵 × DIST_SCALE 取整（OR-Tools 整数弧）。"""
        d = np.linalg.norm(self.coords[:, None, :] - self.coords[None, :, :], axis=2)
        return np.round(d * DIST_SCALE).astype(int)

    def horizon(self) -> int:
        """时间轴上界（DC 的 due date）。"""
        return int(self.due[0])


def parse_solomon(text: str) -> SolomonInstance:
    """解析标准 Solomon 格式文本。

    格式：首行算例名；VEHICLE 段「NUMBER CAPACITY」下一行为车辆数与容量；
    CUSTOMER 段每行 7 列：CUST NO. XCOORD YCOORD DEMAND READY DUE SERVICE。
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("空的 Solomon 算例文件")
    name = lines[0].split()[0]

    # VEHICLE 段：找 "NUMBER CAPACITY" 表头后的首个数据行
    n_vehicles = capacity = None
    for i, ln in enumerate(lines):
        if re.match(r"^NUMBER\s+CAPACITY", ln, re.I):
            nums = lines[i + 1].split()
            n_vehicles, capacity = int(nums[0]), int(nums[1])
            break
    if n_vehicles is None:
        raise ValueError(f"{name}: 未找到 VEHICLE 段（NUMBER CAPACITY 表头）")

    # CUSTOMER 段：表头「CUST NO. ... SERVICE TIME」之后，每行 7 个整数
    coords, demand, ready, due, service = [], [], [], [], []
    in_cust = False
    for ln in lines:
        if re.match(r"^CUST\s*NO\.", ln, re.I):
            in_cust = True
            continue
        if not in_cust:
            continue
        nums = ln.split()
        if len(nums) < 7:
            continue
        try:
            vals = [float(x) for x in nums[:7]]
        except ValueError:
            continue
        _, x, y, dm, rt, dd, st = vals
        coords.append((x, y))
        demand.append(int(dm))
        ready.append(int(rt))
        due.append(int(dd))
        service.append(int(st))

    if len(coords) < 2:
        raise ValueError(f"{name}: CUSTOMER 段解析到 {len(coords)} 行，期望 ≥2（DC+客户）")
    return SolomonInstance(
        name, n_vehicles, capacity,
        np.array(coords, dtype=float), np.array(demand, dtype=int),
        np.array(ready, dtype=int), np.array(due, dtype=int), np.array(service, dtype=int),
    )


def parse_bks(text: str) -> tuple[list[list[int]], float]:
    """解析 best-known 解文件。返回 (路线列表, 总距离)。"""
    routes = []
    cost = None
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = re.match(r"^Route\s*#\d+\s*:\s*(.+)$", ln, re.I)
        if m:
            routes.append([int(x) for x in m.group(1).split()])
            continue
        m2 = re.match(r"^Cost\s+([\d.]+)", ln, re.I)
        if m2:
            cost = float(m2.group(1))
    if cost is None:
        raise ValueError("BKS 文件缺少 'Cost <值>' 行")
    return routes, cost


# ---------------------------------------------------------------------------
# 获取（本地优先 → 缺失自动取 → 取不到报错，全程不编造）
# ---------------------------------------------------------------------------
#: 算例与 best-known 的权威来源：PUC-Rio CVRPLIB。
#: GitHub 与 SINTEF TOP 官方页在本机网络不可达，故只用这个源（见 MANUAL_PLACE_HINT）。
CVRPLIB_BASE = "https://galgos.inf.puc-rio.br/cvrplib/en/download"
CVRPLIB_INSTANCES_URL = "https://galgos.inf.puc-rio.br/cvrplib/en/instances/2"
#: 算例名 → CVRPLIB 实例编号
CVRPLIB_INSTANCE_ID: dict[str, int] = {"C101": 380, "R101": 397, "RC101": 420}


def ensure_instance_files(name: str, sol_dir: Path | None = None, *,
                          timeout: float = 30.0) -> Path:
    """确保算例与 best-known 就位：本地已有就不动，缺了才从 CVRPLIB 取。

    **本地优先是刻意的**：已经放好的文件不会被网络上的版本悄悄换掉，重跑结果才可比。
    「算法门禁」的严肃性在于基准来源可追溯，不在于必须手工搬运——从权威源自动取与手工
    放置得到的是同一份文件，而**编造算例或凭记忆填 BKS** 是本模块唯一禁止的事。因此这里
    的失败路径是抛错 + 给手动说明，不是退回一个「差不多」的替代品。
    """
    sol_dir = Path(sol_dir) if sol_dir is not None else C.RAW_SOLOMON_DIR
    inst_f, bks_f = sol_dir / f"{name}.txt", sol_dir / f"{name}.sol"
    if inst_f.exists() and bks_f.exists():
        return sol_dir

    iid = CVRPLIB_INSTANCE_ID.get(name)
    if iid is None:
        raise FileNotFoundError(
            f"未知算例 {name}，没有对应的 CVRPLIB 实例编号\n"
            + MANUAL_PLACE_HINT.format(dir=sol_dir)
        )
    logger.info("本地缺 %s，从 CVRPLIB 获取（实例编号 %d）...", name, iid)
    sol_dir.mkdir(parents=True, exist_ok=True)
    try:
        for kind, dest in (("instance", inst_f), ("bks", bks_f)):
            resp = requests.get(f"{CVRPLIB_BASE}/{kind}/{iid}", timeout=timeout)
            resp.raise_for_status()
            dest.write_text(resp.text, encoding="utf-8")
        src = sol_dir / "SOURCE.md"
        if not src.exists():
            src.write_text(
                f"# Solomon 算例来源\n\n全部算例与 best-known 取自 PUC-Rio CVRPLIB（{CVRPLIB_INSTANCES_URL}），\n"
                "由 `src.solomon_validate.ensure_instance_files` 按实例编号自动获取：\n\n"
                + "\n".join(f"- {n} → 实例编号 {i}" for n, i in sorted(CVRPLIB_INSTANCE_ID.items()))
                + "\n\n本目录已 gitignore；文件一旦落盘即不再重取，保证重跑可比。\n",
                encoding="utf-8",
            )
    except Exception as exc:
        for f in (inst_f, bks_f):  # 不留半截文件——否则下次会被当成「本地已有」
            f.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"{name} 获取失败：{type(exc).__name__}: {exc}\n"
            + MANUAL_PLACE_HINT.format(dir=sol_dir)
        ) from exc
    return sol_dir


def load_instance_and_bks(name: str, sol_dir: Path | None = None) -> tuple[SolomonInstance, list[list[int]], float]:
    """读取算例 + BKS。**纯读取**，不触网——缺失时抛 FileNotFoundError 并给手动放置说明。

    自动获取走 `ensure_instance_files`，由 `validate()` 在读取前调用。分开是为了让「文件
    到底在不在」这件事只有一个答案：读取函数只回答这个，取文件是另一件事。
    """
    sol_dir = Path(sol_dir) if sol_dir is not None else C.RAW_SOLOMON_DIR
    inst_f, bks_f = sol_dir / f"{name}.txt", sol_dir / f"{name}.sol"
    missing = [str(f) for f in (inst_f, bks_f) if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"{name} 缺失文件：{missing}\n" + MANUAL_PLACE_HINT.format(dir=sol_dir)
        )
    inst = parse_solomon(inst_f.read_text(encoding="utf-8", errors="ignore"))
    routes, cost = parse_bks(bks_f.read_text(encoding="utf-8", errors="ignore"))
    return inst, routes, cost


# ---------------------------------------------------------------------------
# OR-Tools 建模（与模块二同构：载重 + 时间窗 + 车辆数优先 + 距离最小）
# ---------------------------------------------------------------------------
def solve_vrptw(inst: SolomonInstance, time_limit_sec: float | None = None,
                solution_limit: int | None = None) -> dict:
    """OR-Tools 求解 VRPTW。返回 {vehicles, distance, scaled_distance, status, 策略}。

    目标（Solomon 惯例的双目标，按优先级）：
      1. 车辆数最小化——通过给每辆车设置远高于最长可能路线的固定成本实现；
      2. 在最少车辆下总距离最小。
    约束：载重上限、客户时间窗 [ready, due] + 服务时间、单车从 DC 出发返回 DC。
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    time_limit_sec = C.VRPTW_TIME_LIMIT_SEC if time_limit_sec is None else time_limit_sec
    n = len(inst.demand)  # 节点总数（含 DC）
    dist = inst.dist_matrix_scaled()

    # 车辆数必须用算例的车队上限（C101=25）。曾误写为 1 → 单车容量 200 装不下
    # 总需求 ~1458，数学上不可行、OR-Tools 返回 None（实测捕获的致命根因）。
    manager = pywrapcp.RoutingIndexManager(n, inst.n_vehicles, 0)  # n 节点、inst.n_vehicles 辆车、depot=0
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(fi, ti):
        return int(dist[manager.IndexToNode(fi)][manager.IndexToNode(ti)])

    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    # 载重维度
    def demand_cb(fi):
        return int(inst.demand[manager.IndexToNode(fi)])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    # 容量列表必须每辆车一项（长度 = 车辆数），否则 C++ 层 CHECK 失败直接终止进程：
    # "vehicles_ == vehicle_capacities.size() (25 vs. 1)"（实测捕获）。
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [int(inst.capacity)] * inst.n_vehicles, True, "Capacity"
    )

    # 时间维度：transit(i,j) = 距离 + i 的服务时间（到达 j 前已完成 i 的服务）
    def time_cb(fi, ti):
        i = manager.IndexToNode(fi)
        return int(dist[i][manager.IndexToNode(ti)] + inst.service[i] * DIST_SCALE)

    time_idx = routing.RegisterTransitCallback(time_cb)
    horizon_scaled = inst.horizon() * DIST_SCALE
    # slack_max 必须 = horizon：Solomon 时间窗允许车辆**早到后等待**（等待不产生
    # 距离成本但占用时间），等待在 routing 中由 slack 变量表达。slack_max=0 等于
    # 禁止等待，C101 这类紧时间窗算例会直接约束冲突 → 无可行解（实测捕获）。
    routing.AddDimension(time_idx, horizon_scaled, horizon_scaled, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    # 时间窗（按 Solomon 时间单位 × DIST_SCALE 对齐到同一整数尺度）
    for node in range(n):
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(
            int(inst.ready[node]) * DIST_SCALE, int(inst.due[node]) * DIST_SCALE
        )
    # depot 的 End 索引单独放开（NodeToIndex 只覆盖 Start），仅受 horizon 上界约束
    for v in range(routing.vehicles()):
        time_dim.CumulVar(routing.End(v)).SetRange(0, horizon_scaled)

    # 车辆数最小化：每辆车固定成本 >> 任何可行路线的距离成本
    max_route_cost = int(dist.max()) * (n + 1)
    routing.SetFixedCostOfAllVehicles(max_route_cost)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # 主停止条件是**解数**不是墙钟：routing 搜索没有随机源，墙钟是唯一让 gap 随机器负载
    # 漂移的东西（见 config.VRPTW_SOLUTION_LIMIT 的说明）。时限退居安全网。
    solution_limit = (C.VRPTW_SOLUTION_LIMIT if solution_limit is None
                      else int(solution_limit))
    params.solution_limit = solution_limit
    params.time_limit.FromSeconds(int(time_limit_sec))
    # 首解策略用 PARALLEL_CHEAPEST_INSERTION 而非 PATH_CHEAPEST_ARC：
    # 后者在紧时间窗算例（R101/RC101）上构造不出可行首解 → 整体无解（实测捕获）；
    # 前者对 C/R/RC 三类算例均稳健，且更贴合成都场景（紧时间窗冷链配送）。
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )

    sol = routing.SolveWithParameters(params)

    if sol is None:
        return {
            "vehicles": None, "distance": None, "scaled_distance": None,
            "solved": False,
            "first_solution_strategy": "PARALLEL_CHEAPEST_INSERTION",
            "local_search": "GUIDED_LOCAL_SEARCH",
            "time_limit_sec": time_limit_sec,
            "solution_limit": solution_limit,
        }

    routes, scaled_total = [], 0
    veh = 0
    for v in range(routing.vehicles()):
        route, i, rdist = [], routing.Start(v), 0
        while not routing.IsEnd(i):
            j = sol.Value(routing.NextVar(i))
            rdist += dist[manager.IndexToNode(i)][manager.IndexToNode(j)]
            # 仅当 j 非 End 索引时才计入路线：End 的 IndexToNode 对单 depot 返回 0，
            # 若一并 append 会让每条路线末尾多出幽灵 DC 节点，使空车被误判为已使用
            # （实测捕获：25 条路线求和 125 = 100 客户 + 25 个幽灵 0）。
            if not routing.IsEnd(j):
                route.append(manager.IndexToNode(j))
            i = j
        if route:  # 仅真实服务了客户的车辆才计入用车数
            veh += 1
            routes.append(route)
            scaled_total += rdist
    return {
        "vehicles": veh,
        "distance": round(scaled_total / DIST_SCALE, 2),
        "scaled_distance": int(scaled_total),
        "routes": routes,
        "solved": True,
        # 必须与实际使用的首解策略一致：早先这里回填成 PATH_CHEAPEST_ARC，
        # 与上面 params 里真正用的 PARALLEL_CHEAPEST_INSERTION 矛盾，
        # 会让落盘结果的「采用策略」字段失真（05 号票评审发现）
        "first_solution_strategy": "PARALLEL_CHEAPEST_INSERTION",
        "local_search": "GUIDED_LOCAL_SEARCH",
        "time_limit_sec": time_limit_sec,
        "solution_limit": solution_limit,
    }


# ---------------------------------------------------------------------------
# gap 与门禁
# ---------------------------------------------------------------------------
def compute_gap(veh: int, distance: float, bks_veh: int, bks_cost: float) -> dict:
    """计算与 best-known 的 gap。

    Solomon 是双目标（先车辆数、后距离），故分开报告并在车辆数不等时给出综合口径：
      - distance_gap：同尺度距离相对差（仅车辆数相等时可直接与 BKS 距离比）；
      - vehicles_gap：车辆数相对差；
      - gate_gap：门禁判定用的综合 gap——车辆数相等时取距离 gap；
        车辆数更多时，距离比较失去意义（路线结构不同），取 vehicles_gap 与
        distance_gap 的较大者，避免「少一辆车但距离暴涨」被掩盖。
    """
    vehicles_gap = (veh - bks_veh) / bks_veh if bks_veh else 0.0
    if veh == bks_veh:
        distance_gap = (distance - bks_cost) / bks_cost if bks_cost else 0.0
        gate_gap = distance_gap
        basis = "车辆数与 BKS 相同 → 按距离 gap 判定"
    elif veh > bks_veh:
        distance_gap = (distance - bks_cost) / bks_cost if bks_cost else 0.0
        gate_gap = max(vehicles_gap, distance_gap)
        basis = f"车辆数多于 BKS（{veh} > {bks_veh}）→ 取车辆 gap 与距离 gap 较大者"
    else:  # veh < bks_veh：用车更少，优于 BKS（BKS 可能是「最少车辆下的最短距离」）
        distance_gap = (distance - bks_cost) / bks_cost if bks_cost else 0.0
        gate_gap = min(0.0, distance_gap) if distance_gap < 0 else vehicles_gap
        basis = f"车辆数少于 BKS（{veh} < {bks_veh}）→ 用车更优，gap 取保守值"
    return {
        "vehicles_gap": round(float(vehicles_gap), 4),
        "distance_gap": round(float(distance_gap), 4),
        "gate_gap": round(float(gate_gap), 4),
        "basis": basis,
    }


def validate(
    sol_dir: Path | None = None,
    out_dir: Path | None = None,
    time_limit_sec: float | None = None,
    instances: tuple[str, ...] | None = None,
) -> dict:
    """求解三算例、计算 gap、执行门禁。结果落盘 processed/solomon/。

    返回 {results, gate_passed, gate_threshold, failures}。
    """
    sol_dir = Path(sol_dir) if sol_dir is not None else C.RAW_SOLOMON_DIR
    out_dir = Path(out_dir) if out_dir is not None else C.PROCESSED_DIR / "solomon"
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = instances or C.SOLOMON_INSTANCES
    gate = C.SOLOMON_GAP_GATE

    results = []
    for name in instances:
        ensure_instance_files(name, sol_dir)
        inst, bks_routes, bks_cost = load_instance_and_bks(name, sol_dir)
        bks_veh = len(bks_routes)
        logger.info("求解 %s（%d 客户，容量 %d，时限 %.0fs）...",
                    name, inst.n_customers, inst.capacity, time_limit_sec or C.VRPTW_TIME_LIMIT_SEC)
        sol = solve_vrptw(inst, time_limit_sec)
        if not sol["solved"]:
            results.append({"instance": name, "solved": False,
                            "bks": {"vehicles": bks_veh, "cost": bks_cost}})
            logger.error("%s 未求得可行解", name)
            continue
        gaps = compute_gap(sol["vehicles"], sol["distance"], bks_veh, bks_cost)
        rec = {
            "instance": name,
            "solved": True,
            "n_customers": inst.n_customers,
            "capacity": inst.capacity,
            "ours": {"vehicles": sol["vehicles"], "distance": sol["distance"]},
            "bks": {"vehicles": bks_veh, "cost": bks_cost},
            "gaps": gaps,
            "gate_pass": gaps["gate_gap"] <= gate,
            "strategy": {
                "first_solution": sol["first_solution_strategy"],
                "local_search": sol["local_search"],
                # 两个都记：主停止条件是解数，时限只是安全网。只记后者会让读的人
                # 以为这个解是跑 300 秒跑出来的（ADR-0015）。
                "solution_limit": sol["solution_limit"],
                "time_limit_sec": sol["time_limit_sec"],
            },
        }
        results.append(rec)
        logger.info(
            "%s: 我方 %d 车 / %.2f，BKS %d 车 / %.2f → gate_gap %.2f%%（%s）%s",
            name, sol["vehicles"], sol["distance"], bks_veh, bks_cost,
            gaps["gate_gap"] * 100, gaps["basis"],
            "✓ 通过" if rec["gate_pass"] else "✗ 超阈值",
        )

    failures = [r["instance"] for r in results if not r.get("gate_pass", False)]
    gate_passed = len(results) == len(instances) and not failures

    out = {
        "gate_threshold": gate,
        "gate_passed": gate_passed,
        "failures": failures,
        "bks_source": "PUC-Rio CVRPLIB（见 data/raw/solomon/SOURCE.md，检索 2026-09-14）",
        "results": results,
    }
    if not gate_passed:
        out["troubleshooting_required"] = (
            "gap 超过阈值或未求解成功：必须先排查建模与参数（载重/时间窗/服务时间/距离尺度/"
            "首解策略/局部搜索/时限），记录排查过程后方可进入运输优化（3.5.3 门禁）。"
        )
    out_f = out_dir / "solomon_validation.json"
    out_f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown 报告
    md = ["# 数据层 E：Solomon 算法门禁（学术基准）", "",
          f"- 门禁阈值：gate_gap ≤ {gate:.0%}",
          f"- **门禁结果：{'通过 ✅' if gate_passed else '未通过 ❌'}**",
          f"- best-known 来源：{out['bks_source']}", "",
          "| 算例 | 我方车辆/距离 | BKS车辆/距离 | 车辆gap | 距离gap | 门禁gap | 通过 |",
          "|---|---|---|---|---|---|---|"]
    for r in results:
        if not r.get("solved"):
            md.append(f"| {r['instance']} | 未求解 | {r['bks']['vehicles']}/{r['bks']['cost']} | - | - | - | ❌ |")
            continue
        g = r["gaps"]
        md.append(
            f"| {r['instance']} | {r['ours']['vehicles']}车/{r['ours']['distance']:.2f} | "
            f"{r['bks']['vehicles']}车/{r['bks']['cost']:.2f} | {g['vehicles_gap']:+.2%} | "
            f"{g['distance_gap']:+.2%} | **{g['gate_gap']:+.2%}** | {'✅' if r['gate_pass'] else '❌'} |"
        )
    md += ["", "## 求解策略", ""]
    for r in results:
        if r.get("solved"):
            s = r["strategy"]
            md.append(f"- {r['instance']}：首解 {s['first_solution']}，局部搜索 {s['local_search']}，时限 {s['time_limit_sec']:.0f}s")
            md.append(f"  - gap 判定口径：{r['gaps']['basis']}")
    if not gate_passed:
        md += ["", "## ⚠️ 门禁未通过", "", out["troubleshooting_required"]]
    (out_dir / "solomon_validation.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = validate()
    print(f"\n=== 数据层 E 门禁：{'通过' if out['gate_passed'] else '未通过'} ===")
    for r in out["results"]:
        if r.get("solved"):
            g = r["gaps"]
            print(f"  {r['instance']:6s} 我方 {r['ours']['vehicles']:2d}车/{r['ours']['distance']:8.2f}  "
                  f"BKS {r['bks']['vehicles']:2d}车/{r['bks']['cost']:8.2f}  gate_gap {g['gate_gap']:+.2%}")
        else:
            print(f"  {r['instance']:6s} 未求解")


if __name__ == "__main__":
    main()
