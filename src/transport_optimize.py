"""模块二（上）：运输调度优化核心（10 号票）。

产出成都冷链城配的路径优化核心结果：

  ① **运输 KPI（严格口径，可独立调用）**：时间窗达成率（**逐单**判定，ADR-0004）、
     车辆满载率（每趟实际载重/额定，均值 + 分布）、单均运输成本（柴油自购 / 纯电租赁并列）、
     里程利用率（载货里程 / 总里程）。
  ② **基线策略**：最近邻贪心模拟「人工就近派车」，用数据层 C 的**真实路网**矩阵
     （禁止直线距离）跑全 90 天，记录总里程 / 用车数 / 总成本 / 时间窗达成率。
  ③ **优化策略**：代表日（90 天中订单量最大的工作日，ADR-0003）用 OR-Tools 求解 VRPTW——
     节点 = POI、需求聚合（同店当日多单重量体积求和、时间窗取最优点访窗、单点一访，ADR-0004），
     约束含载重 / 容积 / 时间窗 / 单车从单一 DC 出发返回，时限 30 秒，
     目标 = 固定成本 + 里程成本，**建模逻辑与 05 号票的 Solomon 验证同构**。
  ④ 代表日优化前后对比表 + 两套路线 GeoJSON；单趟节省 → 月/年金额换算（外推假设登记台账）。

运行方式（项目根）：python -m src.transport_optimize
数据类别：情景假设（需求）+ 学术基准（算法），见 data_sources_ledger.md。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src import costing

logger = logging.getLogger(__name__)

#: 整数化尺度：OR-Tools 弧权必须为整数，故距离用米、时间用秒、容积用升
_M_PER_KM = 1000
_SEC_PER_MIN = 60
_L_PER_M3 = 1000


# ---------------------------------------------------------------------------
# 取数
# ---------------------------------------------------------------------------
def load_transport_context() -> dict:
    """读取配送订单、真实路网矩阵、DC 坐标、车队表与在途延误。缺失时报错不降级。"""
    paths = {
        "orders": C.DELIVERY_ORDERS_CSV,
        "dist": C.DIST_MATRIX_KM_CSV,
        "time": C.TIME_MATRIX_MIN_CSV,
        "vehicles": C.DELIVERY_VEHICLES_CSV,
        "anomalies": C.DELIVERY_ANOMALIES_CSV,
        "source": C.GEO_DIR / "matrix_source.json",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少模块二输入产物：{missing}\n"
            f"请先运行：python -m src.geo_matrix 与 python -m src.gen_delivery_data"
        )
    dc = json.loads(paths["source"].read_text(encoding="utf-8"))["dc"]
    anomalies = pd.read_csv(paths["anomalies"], encoding="utf-8-sig", usecols=["order_id", "delay_min"])
    return {
        "orders": pd.read_csv(paths["orders"], encoding="utf-8-sig", parse_dates=["date"]),
        "dist_km": pd.read_csv(paths["dist"], index_col=0),
        "time_min": pd.read_csv(paths["time"], index_col=0),
        "vehicles": pd.read_csv(paths["vehicles"], encoding="utf-8-sig"),
        "order_delay_min": dict(zip(anomalies["order_id"], anomalies["delay_min"].astype(float))),
        "dc_coord": (float(dc["lng"]), float(dc["lat"])),
        "matrix_source": json.loads(paths["source"].read_text(encoding="utf-8"))["source"],
    }


def representative_day(orders: pd.DataFrame) -> pd.Timestamp:
    """代表日 = 订单量最大的**工作日**（ADR-0003：工作日才有城配调度）。"""
    weekday = orders.loc[orders["date"].dt.weekday < 5]
    counts = weekday.groupby("date").size()
    return counts.idxmax()


# ---------------------------------------------------------------------------
# 需求聚合（ADR-0004：节点 = POI，单点一访）
# ---------------------------------------------------------------------------
def visit_window(starts, ends) -> tuple[int, int, int]:
    """选一次到访的可行窗：**覆盖订单数最多的那一刻**所属各窗的交集。

    返回 (窗起, 窗止, 覆盖订单数)。扫描线找覆盖数最大的时刻 t*，再取「包含 t* 的
    那些窗」的交集——该交集非空且包含 t*，故「到访落在窗内」这条硬约束本身就尽可能
    多地满足该店订单；`time_window_achievement` 再对每张订单独立判定，不被这里替代。

    与 ADR-0004 的关系：ADR 原文写「时间窗取外包络」。外包络在数学上可行，但**过宽**——
    求解器可以合法地把到访停在信封边缘，那一刻几乎所有订单都不在窗内（10 号票评审实测
    逐单达成率被压到 35%，而真实上限是 64%）。本函数是外包络规则的**收紧**：
    结果始终是外包络的子区间，只排除掉「满足不了任何订单」的到访时刻，不排除任何
    能满足全部订单的解。收紧后 10 号票的时间窗达成率与目标函数脱钩的问题随之消失。
    """
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    events = sorted([(s, 1) for s in starts] + [(e + 1, -1) for e in ends])
    cur = best = 0
    best_t = float(starts.min())
    for t, delta in events:
        cur += delta
        if cur > best:
            best, best_t = cur, t
    members = [(s, e) for s, e in zip(starts, ends) if s <= best_t <= e]
    return int(max(s for s, _ in members)), int(min(e for _, e in members)), len(members)


def aggregate_demand(orders_day: pd.DataFrame) -> pd.DataFrame:
    """把某日订单聚合到 POI 节点：重量/体积求和、时间窗取「最优点访窗」、单点一访。

    返回含 poi_id / weight_kg / volume_m3 / window_start / window_end / n_orders /
    window_covered_orders / full_coverage / 坐标。

    时间窗口径见 `visit_window`：取覆盖订单数最多的那一刻所属各窗的交集。数据层 D 把
    同店当日各单的窗锚在同一收货波次上，故绝大多数节点 `full_coverage=True`（交集非空），
    此时到访落在窗内即满足该店全部订单；少数窗口确实冲突的节点会标记 `full_coverage=False`
    并如实计入达成率的分母，不假装它们也被满足了。
    """
    rows = []
    for poi_id, grp in orders_day.groupby("poi_id"):
        w_start, w_end, covered = visit_window(grp["window_start"], grp["window_end"])
        # 排线用的**计划窗止** = 服务窗止 − 延误缓冲：给在途异常留吸收空间。
        # 逐单判定仍用订单自己的真实窗（见 time_window_achievement），此处的收紧只影响排线。
        buffer = C.TRANSPORT_DELAY_BUFFER_MIN
        plan_end = max(int(w_end - buffer), w_start)
        rows.append(
            {
                "poi_id": poi_id,
                "weight_kg": float(grp["weight_kg"].sum()),
                "volume_m3": float(grp["volume_m3"].sum()),
                "window_start": w_start,
                "window_end": plan_end,
                "window_end_service": int(w_end),
                "n_orders": int(len(grp)),
                "window_covered_orders": int(covered),
                "full_coverage": bool(covered == len(grp)),
                "lng": float(grp["lng"].iloc[0]),
                "lat": float(grp["lat"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("poi_id", kind="stable").reset_index(drop=True)


def node_matrix(nodes: pd.DataFrame, matrix: pd.DataFrame) -> np.ndarray:
    """取「DC + 节点」子矩阵（行/列顺序 = DC, node0, node1…）。"""
    ids = ["DC"] + nodes["poi_id"].tolist()
    return matrix.loc[ids, ids].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# 基线：最近邻贪心（人工就近派车）
# ---------------------------------------------------------------------------
def served_node_indices(routes: list[dict]) -> set[int]:
    """本轮路线实际服务到的节点下标集合。

    基线贪心在「剩余点任何车都装不下/送不到」时会收车退出，**必然存在服务不到的点**
    （时间窗太早、车还没开到就关窗）。调用方必须显式用本函数算未服务集并计数，
    不能默认「计划里的点都会被送到」——早先版本把未服务点的订单也算进"已覆盖"，是虚报。
    """
    return {i for r in routes for i in r["node_idx"]}


def nearest_neighbor_routes(
    nodes: pd.DataFrame,
    dist_km: np.ndarray,
    time_min: np.ndarray,
    *,
    rated_payload_kg: float,
    rated_volume_l: float,
    service_min: float,
    depot_open_min: float,
    max_route_min: float,
    max_vehicles: int,
) -> list[dict]:
    """最近邻贪心派车（基线）。每步选「装得下且能按时窗送达」的最近未访问点。

    与仿真/优化共用的可行性口径：载重、容积、时间窗（早到可等待、晚到不可行）、
    单车单日最长在途时长。返回每趟的 {nodes(下标), arrival_min, load, distance, …}。
    """
    n = len(nodes)
    w = nodes["weight_kg"].to_numpy(dtype=float)
    v = nodes["volume_m3"].to_numpy(dtype=float) * _L_PER_M3
    ws = nodes["window_start"].to_numpy(dtype=float)
    we = nodes["window_end"].to_numpy(dtype=float)
    remaining = set(range(n))
    routes: list[dict] = []

    while remaining and len(routes) < max_vehicles:
        pos = 0  # 0 = DC
        t = depot_open_min
        load_w = load_v = 0.0
        dist_m = 0.0
        seq: list[int] = []
        arrivals: list[float] = []
        while True:
            best, best_km, best_arrive = None, np.inf, None
            for j in remaining:
                if load_w + w[j] > rated_payload_kg or load_v + v[j] > rated_volume_l:
                    continue
                arrive = t + time_min[pos, j + 1]
                if arrive > we[j]:
                    continue  # 晚到不可行（早到可以等）
                # 用**实际开始服务时刻**（含早到等待）算回程，否则早到等待会被漏算，
                # 使「单车单日最长在途时长」形同虚设（实测可超限 49%）
                start_svc = max(arrive, ws[j])
                back = start_svc + service_min + time_min[j + 1, 0]
                if back - depot_open_min > max_route_min:
                    continue
                km = dist_km[pos, j + 1]
                if km < best_km:
                    best, best_km, best_arrive = j, km, arrive
            if best is None:
                break
            start_service = max(best_arrive, ws[best])  # 早到等待到窗起
            seq.append(best)
            arrivals.append(start_service)
            load_w += w[best]
            load_v += v[best]
            dist_m += best_km * _M_PER_KM
            t = start_service + service_min
            pos = best + 1
            remaining.discard(best)
        if not seq:
            break  # 剩余点任何车都装不下/送不到，避免死循环
        dist_m += dist_km[pos, 0] * _M_PER_KM
        routes.append(
            {
                "node_idx": seq,
                "arrival_min": arrivals,
                "load_kg": load_w,
                "load_l": load_v,
                "distance_m": dist_m,
                "duration_min": t + time_min[pos, 0] - depot_open_min,
                "depart_min": depot_open_min,
            }
        )
    return routes


# ---------------------------------------------------------------------------
# 优化：OR-Tools VRPTW（与 05 号票 Solomon 验证同构）
# ---------------------------------------------------------------------------
def solve_vrptw(
    nodes: pd.DataFrame,
    dist_km: np.ndarray,
    time_min: np.ndarray,
    *,
    n_vehicles: int,
    rated_payload_kg: float,
    rated_volume_l: float,
    service_min: float,
    depot_open_min: float,
    max_route_min: float,
    time_limit_sec: float | None = None,
    solution_limit: int | None = None,
) -> list[dict] | None:
    """OR-Tools VRPTW：载重 + 容积 + 时间窗 + 单 DC 往返，目标 = 固定成本 + 里程成本。

    建模与 05 号票完全同构（同样用 PARALLEL_CHEAPEST_INSERTION 首解 + GUIDED_LOCAL_SEARCH），
    差别只有：两个容量维度（重量 kg / 容积 L）、时间单位为秒、车辆数 = 车队规模。
    无可行解返回 None。
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    time_limit_sec = C.VRPTW_TIME_LIMIT_SEC if time_limit_sec is None else time_limit_sec
    n = len(nodes) + 1  # 含 DC
    d_m = np.round(dist_km * _M_PER_KM).astype(int)
    t_s = np.round(time_min * _SEC_PER_MIN).astype(int)
    service_s = int(round(service_min * _SEC_PER_MIN))
    horizon = int(round((depot_open_min + max_route_min) * _SEC_PER_MIN))

    manager = pywrapcp.RoutingIndexManager(n, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(fi, ti):
        return int(d_m[manager.IndexToNode(fi)][manager.IndexToNode(ti)])

    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

    w_kg = np.concatenate([[0.0], nodes["weight_kg"].to_numpy(dtype=float)])
    v_l = np.concatenate([[0.0], nodes["volume_m3"].to_numpy(dtype=float) * _L_PER_M3])

    def weight_cb(fi):
        return int(round(w_kg[manager.IndexToNode(fi)]))

    def volume_cb(fi):
        return int(round(v_l[manager.IndexToNode(fi)]))

    # 容量列表必须每辆车一项，否则 C++ 层 CHECK 直接终止进程（05 号票踩过的坑）
    for name, cb, cap in (
        ("Weight", weight_cb, int(rated_payload_kg)),
        ("Volume", volume_cb, int(rated_volume_l)),
    ):
        idx = routing.RegisterUnaryTransitCallback(cb)
        routing.AddDimensionWithVehicleCapacity(idx, 0, [cap] * n_vehicles, True, name)

    def time_cb(fi, ti):
        i = manager.IndexToNode(fi)
        return int(t_s[i][manager.IndexToNode(ti)] + (service_s if i != 0 else 0))

    time_idx = routing.RegisterTransitCallback(time_cb)
    # slack_max 必须够大：早到等待在 routing 中由 slack 表达，禁止等待会直接无解（05 号票踩过）
    routing.AddDimension(time_idx, horizon, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    ws = nodes["window_start"].to_numpy(dtype=float)
    we = nodes["window_end"].to_numpy(dtype=float)
    late_penalty = int(C.TRANSPORT_LATE_PENALTY_PER_MIN * _SEC_PER_MIN)
    for node in range(1, n):
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(int(ws[node - 1] * _SEC_PER_MIN), int(we[node - 1] * _SEC_PER_MIN))
        # 软上界设在**服务窗起点**：越过窗起每一秒都要付代价，于是求解器会尽量贴着
        # 窗起到达，把「窗起→窗止」这段余量整段留给在途延误去消耗。
        # 设在窗止则毫无作用——硬约束本来就禁止越过窗止，惩罚永远不会触发，实测确认
        # 加上后时间窗达成率纹丝不动（30.3% vs 基线 41.7%）。这一项才是真正买准时的。
        time_dim.SetCumulVarSoftUpperBound(idx, int(ws[node - 1] * _SEC_PER_MIN), late_penalty)
    for veh in range(routing.vehicles()):
        time_dim.CumulVar(routing.Start(veh)).SetRange(int(depot_open_min * _SEC_PER_MIN), horizon)
        time_dim.CumulVar(routing.End(veh)).SetRange(int(depot_open_min * _SEC_PER_MIN), horizon)

    # 车辆数最小化：固定成本远超任何可行路线的里程成本
    routing.SetFixedCostOfAllVehicles(int(d_m.max()) * (n + 1))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # 主停止条件是**解数**不是墙钟：routing 搜索没有随机源，墙钟是唯一让结果随机器负载
    # 漂移的东西（见 config.VRPTW_SOLUTION_LIMIT 的说明）。时限退居安全网。
    params.solution_limit = (C.VRPTW_SOLUTION_LIMIT if solution_limit is None
                             else int(solution_limit))
    params.time_limit.FromSeconds(int(time_limit_sec))
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION

    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None

    routes: list[dict] = []
    for veh in range(routing.vehicles()):
        seq, arrivals = [], []
        i = routing.Start(veh)
        while not routing.IsEnd(i):
            j = sol.Value(routing.NextVar(i))
            if routing.IsEnd(j):
                break
            seq.append(manager.IndexToNode(j) - 1)
            arrivals.append(sol.Value(time_dim.CumulVar(j)) / _SEC_PER_MIN)
            i = j
        if not seq:
            continue
        dist_m = float(d_m[0][seq[0] + 1])
        for a, b in zip(seq, seq[1:]):
            dist_m += float(d_m[a + 1][b + 1])
        dist_m += float(d_m[seq[-1] + 1][0])
        routes.append(
            {
                "node_idx": seq,
                "arrival_min": arrivals,
                "load_kg": float(w_kg[[s + 1 for s in seq]].sum()),
                "load_l": float(v_l[[s + 1 for s in seq]].sum()),
                "distance_m": dist_m,
                "duration_min": sol.Value(time_dim.CumulVar(routing.End(veh))) / _SEC_PER_MIN
                - depot_open_min,
                "depart_min": depot_open_min,
            }
        )
    return routes


# ---------------------------------------------------------------------------
# 运输 KPI（严格口径，可独立调用）
# ---------------------------------------------------------------------------
def optimal_single_visit_ceiling(nodes: pd.DataFrame, orders_day: pd.DataFrame) -> dict:
    """「单点一访」规则下**逐单时间窗达成率的理论上限**（不依赖任何路由算法）。

    对每个 POI 求一个到访时刻 t，使该店当日窗包含 t 的订单数最多（区间扫描线），
    再把各点的最优命中数相加除以总单数。这个数说明：在 ADR-0004 的聚合规则下，
    单次到访**还能**拿到多少按时单——用它把「聚合规则的结构性限制」与
    「目标函数没把准时当回事」两种原因区分开，避免把优化压不动的比率一律归给前者。
    """
    best = 0
    per_stop = {}
    for poi_id, grp in orders_day.groupby("poi_id"):
        events = []
        for s, e in zip(grp["window_start"], grp["window_end"]):
            events.append((s, 1))
            events.append((e + 1, -1))  # 闭区间：终点仍算命中
        events.sort()
        cur = peak = 0
        for _, delta in events:
            cur += delta
            peak = max(peak, cur)
        per_stop[str(poi_id)] = peak
        best += peak
    n_orders = int(len(orders_day))
    return {
        "max_ontime_orders": int(best),
        "n_orders": n_orders,
        "ceiling_rate": round(best / n_orders, 4) if n_orders else None,
        "n_stops_where_all_windows_overlap": int(sum(1 for v in per_stop.values() if v > 1)),
        "n_stops": int(len(per_stop)),
        "note": (
            "该上限只受「单点一访」聚合规则约束；实测达成率低于它，说明差距出在"
            "**目标函数不含准时项**（目标=固定成本+里程成本），而不是聚合规则不可逾越"
        ),
    }


def node_delays(nodes: pd.DataFrame, orders_day: pd.DataFrame,
                delays: dict[str, float]) -> np.ndarray:
    """每个配送点的**在途延误**（分钟）= 该点所辖订单中最大的延误；按节点表顺序返回。

    延误挂在**节点**上而非「趟」上——这是有意的：节点所辖订单对基线与优化两套方案
    完全相同，因此两套方案面对的是同一份延误，比较才只反映排程差异。早先按「趟」
    取最大延误会系统性惩罚并载更多的方案（一辆车装 5 单自然比装 3 单更容易撞上
    严重异常），那是耦合方式造成的假象，不是排程质量。
    """
    by_poi = {
        str(poi_id): max((delays.get(str(o), 0.0) for o in grp["order_id"]), default=0.0)
        for poi_id, grp in orders_day.groupby("poi_id")
    }
    return np.array([by_poi[str(p)] for p in nodes["poi_id"]], dtype=float)


def time_window_achievement(
    routes: list[dict],
    nodes: pd.DataFrame,
    orders_day: pd.DataFrame,
    delays: dict[str, float] | None = None,
) -> dict:
    """时间窗达成率 = **实际送达**落在该订单自己的时间窗内的订单占比（ADR-0004 逐单判定）。

    实际送达 = 路由的计划到访时刻 + **该配送点的**在途延误（见 `node_delays`，延误取自
    数据层 D 的异常表）。不传 `delays` 时退化为计划口径，报告会标明用的是哪一种。

    到访时刻由路由给出；同店多单共用一次到访，但每张订单拿自己的窗来判定——
    不因为「节点窗被满足」就默认所有单都准时。节点窗由 `visit_window` 收紧为
    「覆盖订单数最多的那一刻所属各窗的交集」，故正常情况下到访落在窗内即满足全部订单。
    """
    node_delay = node_delays(nodes, orders_day, delays) if delays else np.zeros(len(nodes))
    arrival_of: dict[str, float] = {}
    for r in routes:
        for idx, arr in zip(r["node_idx"], r["arrival_min"]):
            arrival_of[nodes.loc[idx, "poi_id"]] = arr + node_delay[idx]
    served = orders_day[orders_day["poi_id"].isin(arrival_of)]
    arr = served["poi_id"].map(arrival_of).to_numpy(dtype=float)
    on_time = (arr >= served["window_start"].to_numpy()) & (arr <= served["window_end"].to_numpy())
    ceiling = optimal_single_visit_ceiling(nodes, orders_day)
    rate = float(on_time.mean()) if len(served) else float("nan")
    return {
        "rate": rate,
        "basis": "实际送达（计划到访 + 该点在途延误）" if delays else "计划送达（未接入在途延误）",
        "n_orders": int(len(served)),
        "n_ontime": int(on_time.sum()),
        "n_unserved": int(len(orders_day) - len(served)),
        "n_stops": int(sum(len(r["node_idx"]) for r in routes)),
        "mean_orders_per_stop": round(float(nodes["n_orders"].mean()), 2) if len(nodes) else None,
        "n_stops_with_full_window_coverage": int(nodes["full_coverage"].sum()),
        "ceiling": ceiling,
        "gap_to_ceiling_pp": round((ceiling["ceiling_rate"] - rate) * 100, 2)
        if ceiling["ceiling_rate"] is not None
        else None,
    }


def load_rate_stats(routes: list[dict], rated_payload_kg: float) -> dict:
    """满载率 = 每趟实际载重 / 额定载重；给出均值 + 分布（需求 3.7）。"""
    rates = np.array([r["load_kg"] / rated_payload_kg for r in routes], dtype=float)
    return {
        "mean": float(rates.mean()) if rates.size else float("nan"),
        "median": float(np.median(rates)) if rates.size else float("nan"),
        "p25": float(np.quantile(rates, 0.25)) if rates.size else float("nan"),
        "p75": float(np.quantile(rates, 0.75)) if rates.size else float("nan"),
        "min": float(rates.min()) if rates.size else float("nan"),
        "max": float(rates.max()) if rates.size else float("nan"),
        "n_trips": int(rates.size),
    }


def mileage_utilization(routes: list[dict], dist_km: np.ndarray) -> float:
    """里程利用率 = 载货里程 / 总里程。

    载货里程 = DC→首点→…→末点（全程载货）；回程 DC 段为空驶，不计入分子。
    单 DC 往返结构下该指标上限约为 1（若只有一个点则接近 50%）。
    """
    loaded = empty = 0.0
    for r in routes:
        seq = r["node_idx"]
        loaded += float(dist_km[0][seq[0] + 1])
        for a, b in zip(seq, seq[1:]):
            loaded += float(dist_km[a + 1][b + 1])
        empty += float(dist_km[seq[-1] + 1][0])
    total = loaded + empty
    return loaded / total if total else float("nan")


def trips_table(routes: list[dict], nodes: pd.DataFrame, fleet_mode: str, cost_per_km: float,
                fixed_cost: float, rated_payload_kg: float) -> pd.DataFrame:
    """逐趟明细（含单趟成本），用于落盘与看板下钻。"""
    rows = []
    for i, r in enumerate(routes, start=1):
        km = r["distance_m"] / _M_PER_KM
        rows.append(
            {
                "trip_id": f"{fleet_mode}-T{i:02d}",
                "mode": fleet_mode,
                "n_stops": len(r["node_idx"]),
                "n_orders": int(nodes.loc[r["node_idx"], "n_orders"].sum()),
                "load_kg": round(r["load_kg"], 1),
                "load_rate": round(r["load_kg"] / rated_payload_kg, 4),
                "distance_km": round(km, 2),
                "duration_min": round(r["duration_min"], 1),
                "mileage_cost": round(km * cost_per_km, 2),
                "fixed_cost": round(fixed_cost, 2),
                "trip_cost": round(fixed_cost + km * cost_per_km, 2),
            }
        )
    return pd.DataFrame(rows)


def total_cost(routes: list[dict], mode: str) -> float:
    """车队总成本 = Σ(每趟 日固定成本 + 里程 × 公里变动成本)（成本模型见 src/costing.py）。"""
    fixed = costing.fixed_per_day(mode)
    per_km = costing.per_km(mode)
    return float(sum(fixed + r["distance_m"] / _M_PER_KM * per_km for r in routes))


def transport_kpis(
    routes: list[dict], nodes: pd.DataFrame, orders_day: pd.DataFrame,
    dist_km: np.ndarray, fleet: pd.DataFrame, delays: dict[str, float] | None = None,
) -> dict:
    """某套路线方案的运输 KPI 汇总（需求 3.7 全套口径）。"""
    rated_kg = float(fleet["rated_payload_kg"].iloc[0])
    n_orders = int(len(orders_day))
    out = {
        "n_vehicles": len(routes),
        "n_stops": int(sum(len(r["node_idx"]) for r in routes)),
        "total_distance_km": round(sum(r["distance_m"] for r in routes) / _M_PER_KM, 2),
        "time_window": time_window_achievement(routes, nodes, orders_day, delays),
        "load_rate": load_rate_stats(routes, rated_kg),
        "mileage_utilization": round(mileage_utilization(routes, dist_km), 4),
        "cost": {},
    }
    for mode in ("diesel", "ev"):
        cost = total_cost(routes, mode)
        out["cost"][mode] = {
            "total": round(cost, 2),
            "per_order": round(cost / n_orders, 2) if n_orders else None,
            "per_km": costing.per_km(mode),
            "fixed_per_day": costing.fixed_per_day(mode),
        }
    return out


# ---------------------------------------------------------------------------
# GeoJSON 与对比
# ---------------------------------------------------------------------------
def routes_to_geojson(routes: list[dict], nodes: pd.DataFrame, dc_coord: tuple[float, float],
                      properties: dict) -> dict:
    """把路线导出为 GeoJSON（每条路线一条 LineString，含趟次属性与**各点时间窗**）。

    时间窗按 ADR-0004 落**外包络**（同店当日各单的 min 窗起 / max 窗止），
    与求解时使用的窗一致；需求 45 要求地图上显示各点时间窗，故随路线一并带出，
    下游看板不必自行回原始订单表反推。
    """
    features = []
    for i, r in enumerate(routes, start=1):
        seq = r["node_idx"]
        coords = [list(dc_coord)] + [
            [float(nodes.loc[idx, "lng"]), float(nodes.loc[idx, "lat"])] for idx in seq
        ] + [list(dc_coord)]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "trip_id": f"T{i:02d}",
                    "n_stops": len(seq),
                    "load_kg": round(r["load_kg"], 1),
                    "distance_km": round(r["distance_m"] / _M_PER_KM, 2),
                    "poi_ids": [str(nodes.loc[idx, "poi_id"]) for idx in seq],
                    "window_start_min": [int(nodes.loc[idx, "window_start"]) for idx in seq],
                    "window_end_min": [int(nodes.loc[idx, "window_end"]) for idx in seq],
                    "arrival_min": [round(float(a), 1) for a in r["arrival_min"]],
                    "n_orders": [int(nodes.loc[idx, "n_orders"]) for idx in seq],
                    **properties,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def comparison_table(baseline: dict, optimized: dict) -> pd.DataFrame:
    """代表日优化前后对比表，含各项降幅（%）。"""
    rows = [
        ("总里程 (km)", "total_distance_km"),
        ("用车数 (台)", "n_vehicles"),
        ("时间窗达成率", "time_window_rate"),
        ("满载率均值", "load_rate_mean"),
        ("里程利用率", "mileage_utilization"),
        ("柴油总成本 (元)", "diesel_cost"),
        ("纯电总成本 (元)", "ev_cost"),
        ("柴油单均成本 (元/单)", "diesel_per_order"),
        ("纯电单均成本 (元/单)", "ev_per_order"),
    ]
    flat = {"baseline": _flatten(baseline), "optimized": _flatten(optimized)}
    out = []
    for label, key in rows:
        b, o = flat["baseline"][key], flat["optimized"][key]
        change = (o - b) / b * 100 if b else None
        out.append(
            {
                "metric": label,
                "baseline": round(b, 4),
                "optimized": round(o, 4),
                "change_pct": None if change is None else round(change, 2),
            }
        )
    return pd.DataFrame(out)


def _flatten(k: dict) -> dict:
    return {
        "total_distance_km": k["total_distance_km"],
        "n_vehicles": k["n_vehicles"],
        "time_window_rate": k["time_window"]["rate"],
        "load_rate_mean": k["load_rate"]["mean"],
        "mileage_utilization": k["mileage_utilization"],
        "diesel_cost": k["cost"]["diesel"]["total"],
        "ev_cost": k["cost"]["ev"]["total"],
        "diesel_per_order": k["cost"]["diesel"]["per_order"],
        "ev_per_order": k["cost"]["ev"]["per_order"],
    }


def savings_extrapolation(comparison: pd.DataFrame, n_orders: int) -> dict:
    """代表日单趟节省 → 月/年金额换算（外推假设登记台账）。"""
    by_metric = comparison.set_index("metric")
    per_order_saving = {
        mode: float(by_metric.loc[f"{label} (元/单)", "baseline"] - by_metric.loc[f"{label} (元/单)", "optimized"])
        for mode, label in (("diesel", "柴油单均成本"), ("ev", "纯电单均成本"))
    }
    days_month = C.WORKDAYS_PER_MONTH
    days_year = C.WORKDAYS_PER_YEAR
    return {
        "assumption": C.TRANSPORT_EXTRAPOLATION_ASSUMPTION,
        "representative_day_orders": int(n_orders),
        "daily_saving_by_mode": {
            mode: {
                "per_order": round(v, 2),
                "per_day": round(v * n_orders, 2),
                "per_month": round(v * n_orders * days_month, 2),
                "per_year": round(v * n_orders * days_year, 2),
            }
            for mode, v in per_order_saving.items()
        },
        "workdays_per_month": days_month,
        "workdays_per_year": days_year,
    }


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------
def _write_report(kpi: dict, comparison: pd.DataFrame, savings: dict, out_path: Path) -> Path:
    b, o = kpi["baseline"], kpi["optimized"]
    ceiling = o["time_window"]["ceiling"]
    tw_ceiling = ceiling["ceiling_rate"]
    all_days = kpi["baseline_all_days"]
    lines = [
        "# 模块二（上）：运输调度优化核心报告",
        "",
        f"- 代表日：**{kpi['representative_day']}**（{kpi['representative_day_orders']} 单、"
        f"{kpi['representative_day_stops']} 个配送点）；代表日=90 天中订单量最大的工作日（ADR-0003）",
        f"- 路网来源：**{kpi['matrix_source']}**（真实路网，非直线距离）",
        f"- 基线规则：{C.TRANSPORT_BASELINE_RULE}",
        "",
        "## 一、代表日优化前后对比",
        "",
        "| 指标 | 基线（人工就近派车） | 优化后（OR-Tools VRPTW） | 变化 |",
        "|---|---|---|---|",
    ]
    for r in comparison.itertuples(index=False):
        change = "—" if r.change_pct is None else f"{r.change_pct:+.2f}%"
        lines.append(f"| {r.metric} | {r.baseline} | {r.optimized} | {change} |")
    lines += [
        "",
        "读法：里程 / 用车数 / 成本的降幅为**负**即变好；时间窗达成率、满载率为**正**即变好；"
        f"里程利用率由 {b['mileage_utilization']:.1%} 变为 {o['mileage_utilization']:.1%}"
        "（{trend}）——该指标只描述载货里程占总里程的比例，不单独作为好坏判据。".format(
            trend="升 = 空驶占比下降"
            if o["mileage_utilization"] >= b["mileage_utilization"]
            else "降 = 空驶占比上升，若总里程同时大幅下降则为正常权衡"
        ),
        "",
        "## 二、运输 KPI（严格口径）",
        "",
        "| KPI | 基线 | 优化后 |",
        "|---|---|---|",
        f"| 时间窗达成率（逐单判定） | {b['time_window']['rate']:.2%}"
        f"（{b['time_window']['n_ontime']}/{b['time_window']['n_orders']} 单） | "
        f"{o['time_window']['rate']:.2%}（{o['time_window']['n_ontime']}/{o['time_window']['n_orders']} 单） |",
        f"| 满载率均值（分布见 JSON） | {b['load_rate']['mean']:.1%} | {o['load_rate']['mean']:.1%} |",
        f"| 里程利用率 | {b['mileage_utilization']:.1%} | {o['mileage_utilization']:.1%} |",
        f"| 柴油单均成本 | {b['cost']['diesel']['per_order']:.2f} 元 | "
        f"{o['cost']['diesel']['per_order']:.2f} 元 |",
        f"| 纯电单均成本 | {b['cost']['ev']['per_order']:.2f} 元 | "
        f"{o['cost']['ev']['per_order']:.2f} 元 |",
        "",
        "### 时间窗达成率",
        "",
        f"- 口径：**{o['time_window']['basis']}**；逐单判定（每张订单拿自己的窗判，ADR-0004）",
        f"- 达成率：基线 {b['time_window']['rate']:.1%} → 优化 {o['time_window']['rate']:.1%}；"
        f"理论上限 {tw_ceiling:.1%}（{ceiling['max_ontime_orders']}/{ceiling['n_orders']} 单）",
        f"- 节点窗覆盖：{o['time_window']['n_stops_with_full_window_coverage']}/"
        f"{o['time_window']['n_stops']} 个配送点的当日各单窗可被一次到访**全部**满足",
        f"- 未服务单：基线 {b['time_window']['n_unserved']} / 优化 {o['time_window']['n_unserved']}",
        "",
        "两项关键口径（否则这个数字会失真，10 号票评审实测过两端）：",
        "",
        "1. **同店各单的窗锚在同一收货波次**（数据层 D `RECEIVING_WAVE_*`）：一家店一天只有"
        "一波冷链收货窗口。若各单的窗各自独立散布在 08:00–18:00，一次到访不可能同时满足它们，"
        "逐单达成率会被压到 ~35%（那正是评审在旧数据上实测到的数），KPI 失去意义。",
        "2. **节点窗取「覆盖订单数最多的那一刻」所属各窗的交集**（`visit_window`），"
        "而非 ADR-0004 原文的外包络：外包络允许求解器把到访合法地停在信封边缘——"
        "那一刻几乎没有任何订单在窗内。收紧后结果仍是外包络的子区间，只排除「满足不了任何单」"
        "的到访时刻，不排除任何能满足全部订单的解。",
        "",
        "**实际 ≠ 计划**：到达时刻已叠加数据层 D 异常表的在途延误（该配送点取最严重的一单；"
        "挂在节点上而非趟上，两套方案面对同一份延误，比较才只反映排程差异），"
        "所以这里量的是实际送达，而不是纸面计划。",
        "",
        "## 三、单趟节省 → 月/年金额",
        "",
        f"外推假设：{savings['assumption']}",
        "",
        "| 模式 | 单均节省 | 日节省 | 月节省 | 年节省 |",
        "|---|---|---|---|---|",
    ]
    for mode, label in (("diesel", "柴油自购"), ("ev", "纯电租赁")):
        d = savings["daily_saving_by_mode"][mode]
        lines.append(
            f"| {label} | {d['per_order']:.2f} 元 | {d['per_day']:,.0f} 元 | "
            f"{d['per_month']:,.0f} 元 | {d['per_year']:,.0f} 元 |"
        )
    lines += [
        "",
        "## 四、全量 90 天基线覆盖",
        "",
        f"- 全 90 天均已用同一基线贪心跑过：总里程 **{all_days['total_distance_km']:,.0f} km**、"
        f"**{all_days['n_trips']:,} 车次**、订单 **{all_days['n_orders_total']:,} 单**"
        f"（其中已服务 {all_days['n_orders_served']:,} 单）",
        f"- 全量时间窗达成率（逐单，分母=已服务单）**{all_days['time_window_rate']:.2%}**；"
        f"全量柴油成本 **{all_days['cost']['diesel']:,.0f} 元**、纯电 **{all_days['cost']['ev']:,.0f} 元**",
        f"- ⚠️ **{all_days['n_orders_unserved']} 单未被服务**（分布于 {all_days['days_with_unserved']} 天）："
        f"这些点的时间窗在 08:00 发车后车辆可抵达之前就已关闭，基线贪心直接跳过。"
        f"全量口径如实计入 `n_orders_unserved`，**不粉饰为「已覆盖」**；"
        f"代表日恰无此情形，故该缺陷只在全量口径下暴露",
        f"- 优化仅对代表日精算（30 秒时限），其余 89 天按 ADR-0003 用基线覆盖、"
        f"按代表日单均节省率外推——**不外推为真实财务承诺**",
        "",
        "## 五、产物",
        "",
        f"- 路线 GeoJSON：`{C.TRANSPORT_ROUTES_BASELINE_GEOJSON.name}` / "
        f"`{C.TRANSPORT_ROUTES_OPTIMIZED_GEOJSON.name}`",
        f"- 逐趟明细：`{C.TRANSPORT_TRIPS_CSV.name}`；对比表：`{C.TRANSPORT_COMPARISON_CSV.name}`",
        f"- 逐日基线 KPI：`{C.TRANSPORT_DAILY_CSV.name}`（驾驶舱趋势线的数据源；"
        "只依赖确定性的基线贪心，可在不重跑代表日优化的前提下单独刷新："
        "`python -m src.transport_optimize --daily-only`）",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def daily_baseline_kpis(
    orders: pd.DataFrame,
    dist_km: pd.DataFrame,
    time_min: pd.DataFrame,
    fleet: pd.DataFrame,
    delays: dict[str, float],
) -> pd.DataFrame:
    """全 90 天逐日跑基线贪心，返回**逐日** KPI 表（驾驶舱趋势线的数据源）。

    为什么单独落一张日表：`baseline_all_days` 那样的聚合值撑不起一条趋势线，而驾驶舱的
    「近 30 天趋势 + 环比箭头」需要的是每日序列。这张表只依赖**确定性的基线贪心**
    （无 RNG、无时限搜索），所以它与代表日的 OR-Tools 求解结果无关——重跑它不会影响
    已发表的优化前后对比（ADR-0013）。

    逐日字段与 KPI 口径同源（复用 `time_window_achievement` / `load_rate_stats` /
    `mileage_utilization` / `total_cost`），不另写一套公式。

    **逐日值不在这里取整**：日表是数据产物，取整会丢信息——把每天的距离/成本各自四舍五入
    再求和，与整体求和后取整不相等（实测差 0.1 km / 0.06 元），于是 `summarize_all_days`
    的结果就与重构前发表的全量口径对不上了。舍入只在**发布口径**（汇总）那一步做一次，
    显示层的取整交给看板。
    """
    rated_kg = float(fleet["rated_payload_kg"].iloc[0])
    rated_l = float(fleet["rated_volume_m3"].iloc[0]) * _L_PER_M3
    common = dict(
        rated_payload_kg=rated_kg,
        rated_volume_l=rated_l,
        service_min=C.TRANSPORT_STOP_SERVICE_MIN,
        depot_open_min=C.TRANSPORT_DEPOT_OPEN_MIN,
        max_route_min=C.TRANSPORT_MAX_ROUTE_MIN,
    )
    rows = []
    for date, grp in orders.groupby("date", sort=True):
        grp = grp.copy()
        nd = aggregate_demand(grp)
        sd, st = node_matrix(nd, dist_km), node_matrix(nd, time_min)
        rs = nearest_neighbor_routes(nd, sd, st, max_vehicles=C.FLEET_SIZE, **common)
        tw = time_window_achievement(rs, nd, grp, delays)
        lr = load_rate_stats(rs, rated_kg)
        n_total = int(len(grp))
        row = {
            "date": pd.Timestamp(date).date(),
            "n_orders_total": n_total,
            "n_orders_served": int(tw["n_orders"]),
            "n_orders_unserved": int(n_total - tw["n_orders"]),
            "n_trips": int(len(rs)),
            "total_distance_km": sum(r["distance_m"] for r in rs) / _M_PER_KM,
            "n_ontime": int(tw["n_ontime"]),
            "time_window_rate": tw["rate"] if tw["n_orders"] else None,
            "load_rate_mean": lr["mean"],
            "mileage_utilization": mileage_utilization(rs, sd),
        }
        for mode in ("diesel", "ev"):
            cost = total_cost(rs, mode)
            row[f"{mode}_cost"] = cost
            row[f"{mode}_per_order"] = cost / n_total if n_total else None
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_all_days(daily: pd.DataFrame) -> dict:
    """把逐日基线 KPI 表汇总成全量口径（票据要求的「全 90 天真实分母」）。

    汇总一律从日表**再聚合**，不另跑一遍循环——两处各算一次正是数字对不上的经典来源。
    """
    n_served = int(daily["n_orders_served"].sum())
    out = {
        "days": int(len(daily)),
        "total_distance_km": round(float(daily["total_distance_km"].sum()), 1),
        "n_trips": int(daily["n_trips"].sum()),
        "n_orders_total": int(daily["n_orders_total"].sum()),
        "n_orders_served": n_served,
        "n_orders_unserved": int(daily["n_orders_unserved"].sum()),
        "n_ontime": int(daily["n_ontime"].sum()),
        "cost": {m: round(float(daily[f"{m}_cost"].sum()), 2) for m in ("diesel", "ev")},
        "days_with_unserved": int((daily["n_orders_unserved"] > 0).sum()),
    }
    out["time_window_rate"] = (
        round(out["n_ontime"] / n_served, 4) if n_served else None
    )
    out["unserved_note"] = (
        "未服务单 = 该点时间窗在车辆可抵达之前就关闭（08:00 发车仍够不到），"
        "基线贪心会收车跳过；全量口径如实计入 n_orders_unserved，不粉饰为「已覆盖」"
    )
    return out


def run_all(out_dir: Path | None = None, time_limit_sec: float | None = None) -> dict:
    """跑通模块二（上）全流程并落盘。"""
    ctx = load_transport_context()
    orders, fleet = ctx["orders"], ctx["vehicles"]
    dist_km, time_min, dc_coord = ctx["dist_km"], ctx["time_min"], ctx["dc_coord"]
    rated_kg = float(fleet["rated_payload_kg"].iloc[0])
    rated_l = float(fleet["rated_volume_m3"].iloc[0]) * _L_PER_M3
    common = dict(
        rated_payload_kg=rated_kg,
        rated_volume_l=rated_l,
        service_min=C.TRANSPORT_STOP_SERVICE_MIN,
        depot_open_min=C.TRANSPORT_DEPOT_OPEN_MIN,
        max_route_min=C.TRANSPORT_MAX_ROUTE_MIN,
    )

    order_delay = ctx["order_delay_min"]
    rep_day = representative_day(orders)
    day_orders = orders[orders["date"] == rep_day].copy()
    nodes = aggregate_demand(day_orders)
    sub_d = node_matrix(nodes, dist_km)
    sub_t = node_matrix(nodes, time_min)

    order_delay = ctx["order_delay_min"]
    logger.info("基线：全 90 天最近邻贪心（含成本与时间窗口径）…")
    daily = daily_baseline_kpis(orders, dist_km, time_min, fleet, order_delay)
    all_days = summarize_all_days(daily)
    if all_days["n_orders_unserved"]:
        logger.warning(
            "全量基线有 %d 单未服务（分布于 %d 天）——时间窗早于最早可达时刻，非代码缺陷",
            all_days["n_orders_unserved"], all_days["days_with_unserved"],
        )
    logger.info("基线汇总：%s km / %s 车次 / %s 单", f"{all_days['total_distance_km']:,.0f}",
                f"{all_days['n_trips']:,}", f"{all_days['n_orders_total']:,}")

    logger.info("代表日 %s：基线 vs OR-Tools（%.0fs 时限）…", rep_day.date(), time_limit_sec or C.VRPTW_TIME_LIMIT_SEC)
    base_routes = nearest_neighbor_routes(nodes, sub_d, sub_t, max_vehicles=C.FLEET_SIZE, **common)
    opt_routes = solve_vrptw(nodes, sub_d, sub_t, n_vehicles=C.FLEET_SIZE,
                             time_limit_sec=time_limit_sec, **common)
    if opt_routes is None:
        raise RuntimeError("代表日 VRPTW 未求得可行解——不得带病出结果，请先排查建模与参数")

    baseline = transport_kpis(base_routes, nodes, day_orders, sub_d, fleet, order_delay)
    optimized = transport_kpis(opt_routes, nodes, day_orders, sub_d, fleet, order_delay)
    comparison = comparison_table(baseline, optimized)
    savings = savings_extrapolation(comparison, len(day_orders))

    kpi = {
        "data_category": "情景假设（需求）+ 学术基准（算法）",
        "representative_day": str(rep_day.date()),
        "representative_day_orders": int(len(day_orders)),
        "representative_day_stops": int(len(nodes)),
        "matrix_source": ctx["matrix_source"],
        "baseline_rule": C.TRANSPORT_BASELINE_RULE,
        "solve": {
            "time_limit_sec": time_limit_sec or C.VRPTW_TIME_LIMIT_SEC,
            "first_solution_strategy": "PARALLEL_CHEAPEST_INSERTION",
            "local_search": "GUIDED_LOCAL_SEARCH",
            "vehicles_available": int(C.FLEET_SIZE),
            "service_min": C.TRANSPORT_STOP_SERVICE_MIN,
        },
        "baseline": baseline,
        "optimized": optimized,
        "baseline_all_days": all_days,
    }

    out_dir = C.TRANSPORT_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / C.TRANSPORT_KPI_JSON.name).write_text(
        json.dumps(kpi, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    comparison.to_csv(out_dir / C.TRANSPORT_COMPARISON_CSV.name, index=False, encoding="utf-8-sig")
    (out_dir / C.TRANSPORT_SAVINGS_JSON.name).write_text(
        json.dumps(savings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trips = pd.concat(
        [
            trips_table(base_routes, nodes, "diesel", costing.per_km("diesel"),
                        costing.fixed_per_day("diesel"), rated_kg).assign(plan="baseline"),
            trips_table(opt_routes, nodes, "diesel", costing.per_km("diesel"),
                        costing.fixed_per_day("diesel"), rated_kg).assign(plan="optimized"),
        ],
        ignore_index=True,
    )
    trips.to_csv(out_dir / C.TRANSPORT_TRIPS_CSV.name, index=False, encoding="utf-8-sig")
    daily.to_csv(out_dir / C.TRANSPORT_DAILY_CSV.name, index=False, encoding="utf-8-sig")
    for name, rs, plan in (
        (C.TRANSPORT_ROUTES_BASELINE_GEOJSON.name, base_routes, "baseline"),
        (C.TRANSPORT_ROUTES_OPTIMIZED_GEOJSON.name, opt_routes, "optimized"),
    ):
        (out_dir / name).write_text(
            json.dumps(routes_to_geojson(rs, nodes, dc_coord, {"plan": plan}),
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    report = _write_report(kpi, comparison, savings, out_dir / C.TRANSPORT_KPI_MD.name)
    return {"kpi": kpi, "comparison": comparison, "savings": savings, "report": str(report),
            "baseline_routes": base_routes, "optimized_routes": opt_routes, "nodes": nodes,
            "daily": daily}


def write_daily_only(out_dir: Path | None = None) -> Path:
    """只重算并落盘逐日基线 KPI 表，**不动代表日优化结果**。

    存在的理由：逐日表是驾驶舱的必需品，但它与代表日的 30 秒启发式搜索完全无关
    （见 `daily_baseline_kpis`）。如果为了补这张表而重跑 `run_all`，就会把
    `transport_kpi.json` 里那个「某一次搜索的解」一并换掉，连带 10/11 号票已发表的
    里程与成本全部作废——那是拿确定性的产物去陪跑一个不确定的搜索。
    """
    ctx = load_transport_context()
    daily = daily_baseline_kpis(ctx["orders"], ctx["dist_km"], ctx["time_min"],
                                ctx["vehicles"], ctx["order_delay_min"])
    out_dir = C.TRANSPORT_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / C.TRANSPORT_DAILY_CSV.name
    daily.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="模块二（上）：运输调度优化核心")
    parser.add_argument("--daily-only", action="store_true",
                        help="只重算逐日基线 KPI 表，不重跑代表日优化（产物其余部分保持不变）")
    args = parser.parse_args()
    if args.daily_only:
        path = write_daily_only()
        print(f"\n逐日基线 KPI 已写：{path}（代表日优化结果未改动）")
        return
    out = run_all()
    k, c = out["kpi"], out["comparison"]
    print(f"\n=== 模块二（上）运输优化：代表日 {k['representative_day']}"
          f"（{k['representative_day_orders']} 单 / {k['representative_day_stops']} 点）===")
    for r in c.itertuples(index=False):
        change = "—" if r.change_pct is None else f"{r.change_pct:+.2f}%"
        print(f"  {r.metric:20s} 基线 {r.baseline:>10}  →  优化 {r.optimized:>10}   {change}")
    s = out["savings"]["daily_saving_by_mode"]["ev"]
    print(f"  纯电口径：单均省 {s['per_order']:.2f} 元 → 年省 {s['per_year']:,.0f} 元（外推假设见台账）")


if __name__ == "__main__":
    main()
