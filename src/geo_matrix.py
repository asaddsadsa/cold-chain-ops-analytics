"""数据层 C-2：路网距离/时间矩阵构建（04 号票）。

对「DC + 全部 POI」节点用高德距离测量接口（type=1 驾车）按目的地逐列批量请求
（origins 单次上限 100 对，官方文档 2026-09-13 核实），构建对称完整的
dist_matrix_km.csv 与 time_matrix_min.csv，永久缓存——之后所有运行直接读缓存。

离线降级：无密钥或持续失败 → OSMnx 下载成都路网 + networkx 最短路计算矩阵，
结果标记来源 OSM。两种路径都失败才报错，报错说明阻塞点与手动放置格式。

运行方式（项目根）：python -m src.geo_matrix
数据类别：真实观测（高德）/ 真实观测-标记 OSM（降级），见 ADR-0008。
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.geo_poi import (
    AMAP_GEOCODE_URL,
    MissingKeyError,
    RateLimiter,
    amap_get,
    geocode_dc,
    get_amap_key,
)

logger = logging.getLogger(__name__)

AMAP_DISTANCE_URL = "https://restapi.amap.com/v3/distance"
ORIGINS_BATCH_LIMIT = 100  # 官方文档：origins 支持 100 个坐标对（2026-09-13 核实）
DRIVING_SPEED_KMH = 30.0  # OSM 降级时的城区平均车速假设（登记台账）
MATRIX_SOURCE_JSON = "matrix_source.json"


def load_nodes(poi_csv: Path | None = None) -> tuple[list[str], np.ndarray]:
    """读 POI 缓存并返回节点表：DC 占第 0 位（节点 id 'DC'），其后为 POI。

    返回 (node_ids, coords[n,2] (lng,lat))。DC 坐标从 matrix_source.json 缓存读取
    （由 fetch_matrix 写入），缓存缺失时用 config 降级坐标——地理编码的正式
    调用在 fetch_matrix 中执行并登记来源。
    """
    poi_csv = Path(poi_csv) if poi_csv is not None else C.POI_CSV
    if not poi_csv.exists():
        raise FileNotFoundError(
            f"POI 缓存不存在：{poi_csv}。请先运行 python -m src.geo_poi"
        )
    poi = pd.read_csv(poi_csv)
    node_ids = ["DC", *poi["poi_id"].tolist()]
    coords = np.column_stack([poi["lng"].to_numpy(), poi["lat"].to_numpy()])
    return node_ids, coords


def _request_distance_column(
    origins: np.ndarray, dest: tuple[float, float], key: str, limiter: RateLimiter
) -> tuple[np.ndarray, np.ndarray]:
    """一次批量请求：多 origins → 单 destination 的驾车距离(km)/时间(min)。

    origins 超过批量上限时分批。返回 (km数组, min数组)。
    """
    km = np.zeros(len(origins))
    minutes = np.zeros(len(origins))
    n_batches = math.ceil(len(origins) / ORIGINS_BATCH_LIMIT)
    for b in range(n_batches):
        chunk = origins[b * ORIGINS_BATCH_LIMIT : (b + 1) * ORIGINS_BATCH_LIMIT]
        origins_str = "|".join(f"{lng:.6f},{lat:.6f}" for lng, lat in chunk)
        data = amap_get(
            {
                "_url": AMAP_DISTANCE_URL,
                "origins": origins_str,
                "destination": f"{dest[0]:.6f},{dest[1]:.6f}",
                "type": "1",  # 驾车导航距离（考虑路况）
            },
            key,
            limiter,
        )
        if data.get("status") != "1":
            raise ConnectionError(
                f"距离测量业务失败：{data.get('info')}（infocode={data.get('infocode')}）"
            )
        results = data.get("results") or []
        if len(results) != len(chunk):
            raise ConnectionError(
                f"距离测量返回数量异常：期望 {len(chunk)}，实得 {len(results)}"
            )
        for i, r in enumerate(results):
            km[b * ORIGINS_BATCH_LIMIT + i] = float(r["distance"]) / 1000.0
            minutes[b * ORIGINS_BATCH_LIMIT + i] = float(r["duration"]) / 60.0
    return km, minutes


def build_matrix_amap(
    node_ids: list[str], coords: np.ndarray, dc_coord: tuple[float, float], key: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """高德路径：按目的地逐列批量请求，构建对称完整矩阵。

    上三角直接取自 API（含路况的驾车距离不对称属正常，取请求方向值），
    下三角用转置补齐后强制对称化：d[i,j] = d[j,i] = 实测值（按列请求已覆盖
    全部 i→j 方向，实际两次方向取均值以保证严格对称，口径登记台账）。
    对角线 = 0。
    """
    n = len(node_ids)
    all_coords = np.vstack([[dc_coord], coords])  # 节点 0 = DC
    limiter = RateLimiter()
    raw_km = np.zeros((n, n))
    raw_min = np.zeros((n, n))

    for j in range(n):  # 每列一个目的地 → n 次批量请求（ADR-0008）
        dest = tuple(all_coords[j])
        origins = np.delete(all_coords, j, axis=0)
        km_col, min_col = _request_distance_column(origins, dest, key, limiter)
        idx = [k for k in range(n) if k != j]
        raw_km[idx, j] = km_col
        raw_min[idx, j] = min_col
        logger.info("矩阵列 %d/%d（%s）完成", j + 1, n, node_ids[j])

    # 对称化：正反两向实测取均值（口径登记台账）；对角 0
    km_sym = (raw_km + raw_km.T) / 2.0
    min_sym = (raw_min + raw_min.T) / 2.0
    np.fill_diagonal(km_sym, 0.0)
    np.fill_diagonal(min_sym, 0.0)
    dist_df = pd.DataFrame(np.round(km_sym, 3), index=node_ids, columns=node_ids)
    time_df = pd.DataFrame(np.round(min_sym, 2), index=node_ids, columns=node_ids)
    return dist_df, time_df


def build_matrix_osm(
    node_ids: list[str], coords: np.ndarray, dc_coord: tuple[float, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """降级路径：OSMnx 下载成都路网 + networkx 最短路，标记来源 OSM。

    时间 = 距离 / 城区平均车速假设（30 km/h，登记台账为情景假设）。
    """
    import networkx as nx
    import osmnx as ox

    logger.info("OSMnx 降级：下载成都路网 ...")
    G = ox.graph_from_place("Chengdu, Sichuan, China", network_type="drive")
    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)
    all_coords = np.vstack([[dc_coord], coords])
    # 每个节点吸附到最近路网节点
    nearest = ox.distance.nearest_nodes(G, all_coords[:, 0], all_coords[:, 1])
    n = len(node_ids)
    dist = np.zeros((n, n))
    mins = np.zeros((n, n))
    for i in range(n):
        lengths = nx.single_source_dijkstra_path_length(
            G, nearest[i], weight="length"
        )
        times = nx.single_source_dijkstra_path_length(
            G, nearest[i], weight="travel_time"
        )
        for j in range(n):
            if i != j:
                dist[i, j] = lengths.get(nearest[j], np.nan) / 1000.0
                mins[i, j] = times.get(nearest[j], np.nan) / 60.0
    # NaN（不可达对）用直线距离×绕行系数 1.3 兜底并记录
    nan_mask = np.isnan(dist) & ~np.eye(n, dtype=bool)
    if nan_mask.any():
        logger.warning("OSM 不可达对 %d 个，用直线×1.3 兜底", int(nan_mask.sum() // 2))
        lat0 = float(np.mean(all_coords[:, 1]))
        for i, j in zip(*np.where(nan_mask)):
            dlat = (all_coords[i, 1] - all_coords[j, 1]) * 111.32
            dlng = (all_coords[i, 0] - all_coords[j, 0]) * 111.32 * math.cos(math.radians(lat0))
            dist[i, j] = math.hypot(dlat, dlng) * 1.3
            mins[i, j] = dist[i, j] / DRIVING_SPEED_KMH * 60.0
    np.fill_diagonal(dist, 0.0)
    np.fill_diagonal(mins, 0.0)
    dist_df = pd.DataFrame(np.round(dist, 3), index=node_ids, columns=node_ids)
    time_df = pd.DataFrame(np.round(mins, 2), index=node_ids, columns=node_ids)
    return dist_df, time_df


def _validate_matrix(df: pd.DataFrame, node_ids: list[str]) -> None:
    """矩阵完整性校验：维度、对称、对角 0、非负、无 NaN。"""
    n = len(node_ids)
    assert df.shape == (n, n), f"维度错误 {df.shape} != {(n, n)}"
    assert list(df.index) == node_ids and list(df.columns) == node_ids, "节点序不一致"
    assert np.allclose(df.to_numpy(), df.to_numpy().T), "矩阵不对称"
    assert (np.diag(df.to_numpy()) == 0).all(), "对角线非 0"
    assert (df.to_numpy() >= 0).all(), "存在负值"
    assert not df.isna().any().any(), "存在 NaN"


def fetch_matrix(
    poi_csv: Path | None = None,
    dist_csv: Path | None = None,
    time_csv: Path | None = None,
    key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构建（或读缓存）距离/时间矩阵。缓存命中且维度匹配时零网络请求。

    降级链：高德（有密钥）→ OSMnx（无密钥或持续失败）→ 报错（含手动放置格式说明）。
    来源标记写 matrix_source.json。
    """
    dist_csv = Path(dist_csv) if dist_csv is not None else C.DIST_MATRIX_KM_CSV
    time_csv = Path(time_csv) if time_csv is not None else C.TIME_MATRIX_MIN_CSV
    poi_csv = Path(poi_csv) if poi_csv is not None else C.POI_CSV

    node_ids, coords = load_nodes(poi_csv)
    n = len(node_ids)

    # 缓存命中判定：两文件存在且维度 = 当前节点数
    if dist_csv.exists() and time_csv.exists():
        d = pd.read_csv(dist_csv, index_col=0)
        t = pd.read_csv(time_csv, index_col=0)
        if d.shape == (n, n) and t.shape == (n, n):
            logger.info("矩阵缓存命中（%d×%d），跳过网络请求", n, n)
            return d, t
        logger.warning("缓存维度 %s 与节点数 %d 不匹配，重建", d.shape, n)

    # DC 坐标（地理编码 → 降级 config）
    source_meta = {"n_nodes": n, "built_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    try:
        if key is None:
            key = get_amap_key()
        dc_lng, dc_lat, dc_source = geocode_dc(key)
        source_meta["dc"] = {"lng": dc_lng, "lat": dc_lat, "source": dc_source}
        try:
            dist_df, time_df = build_matrix_amap(node_ids, coords, (dc_lng, dc_lat), key)
            source_meta["source"] = "amap"
        except ConnectionError as exc:
            logger.warning("高德矩阵构建持续失败（%s），降级 OSMnx", exc)
            dist_df, time_df = build_matrix_osm(node_ids, coords, (dc_lng, dc_lat))
            source_meta["source"] = "osm"
            source_meta["osm_speed_kmh_assumption"] = DRIVING_SPEED_KMH
    except MissingKeyError:
        logger.warning("无 AMAP_KEY，降级 OSMnx")
        dc_lng, dc_lat = C.DC_FALLBACK_LNG, C.DC_FALLBACK_LAT
        source_meta["dc"] = {"lng": dc_lng, "lat": dc_lat, "source": "fallback_config_no_key"}
        try:
            dist_df, time_df = build_matrix_osm(node_ids, coords, (dc_lng, dc_lat))
            source_meta["source"] = "osm"
            source_meta["osm_speed_kmh_assumption"] = DRIVING_SPEED_KMH
        except Exception as exc:  # noqa: BLE001 —— OSM 也失败才允许报错
            raise RuntimeError(
                "高德（无密钥）与 OSMnx（下载/最短路失败）两条路径均不可用。\n"
                f"阻塞点：{exc}\n"
                "手动放置矩阵格式说明：\n"
                f"  1. {dist_csv.name}：CSV，首列与首行为节点 id（DC,P001,...，顺序一致），"
                "值=驾车里程 km，对称、对角 0、无 NaN；\n"
                f"  2. {time_csv.name}：同结构，值=驾车时间 min；\n"
                f"  3. 放入 {dist_csv.parent}/ 后重跑本脚本（缓存命中即采用）。"
            ) from exc

    _validate_matrix(dist_df, node_ids)
    _validate_matrix(time_df, node_ids)

    dist_csv.parent.mkdir(parents=True, exist_ok=True)
    dist_df.to_csv(dist_csv)
    time_df.to_csv(time_csv)
    (dist_csv.parent / MATRIX_SOURCE_JSON).write_text(
        json.dumps(source_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "矩阵落盘：%d×%d，来源=%s（dist %s / time %s）",
        n, n, source_meta["source"], dist_csv.name, time_csv.name,
    )
    return dist_df, time_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dist_df, time_df = fetch_matrix()
    src = json.loads((C.DIST_MATRIX_KM_CSV.parent / MATRIX_SOURCE_JSON).read_text(encoding="utf-8"))
    print(f"\n=== 数据层 C-2 产物 ===")
    print(f"矩阵: {dist_df.shape[0]}×{dist_df.shape[1]}  来源: {src['source']}  DC: {src['dc']}")
    print(f"里程范围: {dist_df.to_numpy()[dist_df.to_numpy() > 0].min():.1f} – "
          f"{dist_df.to_numpy().max():.1f} km")
    print(f"时间范围: {time_df.to_numpy()[time_df.to_numpy() > 0].min():.1f} – "
          f"{time_df.to_numpy().max():.1f} min")


if __name__ == "__main__":
    main()
