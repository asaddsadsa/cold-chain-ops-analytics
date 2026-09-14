"""数据层 C-1：高德 POI 抓取 + DC 地理编码（04 号票）。

抓取成都主城区与青白江的连锁超市 / 便利店 / 快递驿站配送点 40–60 个：
分页抓取、每次调用间隔 ≥0.3s（限速）、按行政区控制空间分散度、去重后
永久缓存 data/geo/poi.csv——已存在缓存时不发起任何网络请求。

DC 坐标：高德地理编码「成都国际铁路港」，失败用 config 青白江城区坐标并登记。

密钥读取顺序：环境变量 AMAP_KEY → 项目根 .env → 都读不到时抛 MissingKeyError
（调用方据此停下来询问用户；绝不硬编码、绝不编造坐标）。

运行方式（项目根）：python -m src.geo_poi
数据类别：真实观测（见 data_sources_ledger.md 第 1 节、ADR-0008）。
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

from src import config as C

logger = logging.getLogger(__name__)

AMAP_PLACE_URL = "https://restapi.amap.com/v3/place/text"
AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"

#: 检索词 × 行政区配额：主城区 7 区每区最多 5 个 + 青白江 14 个 ≈ 49 个（40–60 区间）
SEARCH_KEYWORDS: tuple[str, ...] = ("连锁超市", "便利店", "快递驿站")
DISTRICT_QUOTA: dict[str, int] = {
    "青白江区": 14,  # DC 所在区，配送腹地，配额最大
    "锦江区": 5,
    "青羊区": 5,
    "金牛区": 5,
    "武侯区": 5,
    "成华区": 5,
    "郫都区": 5,
    "龙泉驿区": 5,
}
PAGE_SIZE = 20  # 高德单页上限 25，取 20 留余量


class MissingKeyError(RuntimeError):
    """AMAP_KEY 不可用。调用方应据此询问用户，而非降级或编造。"""


def _key_from_streamlit_secrets() -> str:
    """从 `st.secrets` 取 key（仅云端部署用）。

    Streamlit Community Cloud 上不能放 `.env` 文件，密钥走 App settings → Secrets。
    此处**按需惰性导入** streamlit：数据层脚本不该为了读一个环境变量就背上整个看板依赖，
    本地/CI 运行时这段直接短路返回空串。导入失败（没装 streamlit）同样返回空串，
    不改变原有的「读不到就报 MissingKeyError」语义。
    """
    try:
        import streamlit as st
    except Exception:  # pragma: no cover - 未安装 streamlit 的环境
        return ""
    try:
        return str(st.secrets.get("AMAP_KEY", "")).strip()
    except Exception:  # 无 secrets 文件 / 无运行时上下文
        return ""


def get_amap_key(project_root: Path | None = None) -> str:
    """环境变量 `AMAP_KEY` → 项目根 `.env` → `st.secrets`（云端）→ 抛 MissingKeyError。

    顺序是有意的：本地开发用 `.env`，CI 用环境变量，云端用 Secrets；三者都不落盘密钥、
    不打印密钥。**绝不硬编码**——`get_amap_key` 是全文唯一读取点，提交前扫描据此判定。
    """
    key = os.environ.get("AMAP_KEY", "").strip()
    if key:
        return key
    root = Path(project_root) if project_root is not None else C.PROJECT_ROOT
    env_file = root / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("AMAP_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    key = _key_from_streamlit_secrets()
    if key:
        return key
    raise MissingKeyError(
        "未找到高德密钥：环境变量 AMAP_KEY 为空、项目根 .env 缺失或为空、st.secrets 中也没有。"
        "请设置 AMAP_KEY 后重试（模板见 .env.example；Streamlit Cloud 上改用 App settings → Secrets），"
        "或由用户手动提供。"
    )


class RateLimiter:
    """最小间隔限速器：保证相邻两次调用间隔 ≥ min_interval 秒。"""

    def __init__(self, min_interval: float = C.AMAP_MIN_INTERVAL_SEC):
        self.min_interval = min_interval
        self.last_call: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self.last_call is not None:
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self.last_call = time.monotonic()


#: 可重试的高德业务错误码（瞬时性，退避后重试有意义）。
#: 10019/10020/10021 = QPS 超限（CUQPS/CKQPS/CIQPS_HAS_EXCEEDED_THE_LIMIT）；
#: 10014 = QPS 超限；10015 = 瞬时并发超限；30001 = 引擎响应数据错误（瞬时）。
#: 不在表内的业务失败（如 INVALID_PARAMS、无权限）不重试，直接返回交调用方判断。
RETRYABLE_AMAP_INFOCODES = frozenset({"10014", "10015", "10019", "10020", "10021", "30001"})


def amap_get(
    params: dict,
    key: str,
    limiter: RateLimiter | None = None,
    session: requests.Session | None = None,
) -> dict:
    """带限速 + 指数退避重试（1/2/4s×3）的高德 GET。返回解析后的 JSON dict。

    两类错误都会退避重试：
      1. 网络层异常（连接失败/超时/HTTP 5xx）；
      2. 可重试业务错误（status != '1' 且 infocode 在 RETRYABLE_AMAP_INFOCODES 内，
         典型为 QPS 瞬时超限）——实测高德个人 key 在 0.3s 限速下仍偶发 QPS 超限，
         退避重试可兜住；不可重试的业务失败（参数错误等）直接返回让调用方判断。
    重试用尽仍失败：网络异常抛 ConnectionError，业务错误返回最后一次响应。
    """
    limiter = limiter or RateLimiter()
    session = session or requests
    url = params.pop("_url", AMAP_PLACE_URL)
    payload = {**params, "key": key}
    last_exc: Exception | None = None
    last_data: dict | None = None
    for attempt, backoff in enumerate((0, *C.AMAP_RETRY_BACKOFF_SEC)):
        if backoff:
            time.sleep(backoff)
        limiter.wait()
        try:
            resp = session.get(url, params=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 —— 网络层异常统一退避重试
            last_exc = exc
            logger.warning("高德请求失败（第 %d 次）：%s", attempt + 1, exc)
            continue
        # 业务层：QPS 等可重试错误码 → 退避重试
        if data.get("status") == "1" or str(data.get("infocode")) not in RETRYABLE_AMAP_INFOCODES:
            return data
        last_data = data
        logger.warning(
            "高德业务可重试错误（第 %d 次）：%s/%s", attempt + 1, data.get("info"), data.get("infocode")
        )
    if last_exc is not None and last_data is None:
        raise ConnectionError(f"高德请求连续失败（已按 1/2/4s 退避重试 3 次）：{last_exc}")
    return last_data  # 业务错误重试用尽，返回最后一次响应交调用方判断


def search_district_pois(
    district: str,
    quota: int,
    key: str,
    limiter: RateLimiter | None = None,
    session: requests.Session | None = None,
) -> list[dict]:
    """按行政区抓取至多 quota 个 POI：关键词轮询 × 分页，坐标去重。

    **按区直接检索**（city=区名 + citylimit=true），而非「全市搜索后按 adname 过滤」。
    后者有实测缺陷（2026-09-13）：高德按相关性排序，远郊区（如青白江）在全市搜索的
    前几页几乎排不上，全市 225 条结果里青白江仅 2 条；按区检索同样条件可得 75 条。
    adname 过滤保留作为兜底（citylimit 对个别非标准区名可能失效时剔除跨区结果）。
    """
    limiter = limiter or RateLimiter()
    collected: dict[tuple[float, float], dict] = {}
    for kw in SEARCH_KEYWORDS:
        if len(collected) >= quota:
            break
        page = 1
        while len(collected) < quota and page <= 3:  # 每关键词最多 3 页，防配额浪费
            data = amap_get(
                {
                    "_url": AMAP_PLACE_URL,
                    "keywords": kw,
                    "city": district,  # 按区检索（关键修复）
                    "citylimit": "true",
                    "offset": PAGE_SIZE,
                    "page": page,
                    "extensions": "base",
                },
                key,
                limiter,
                session,
            )
            if data.get("status") != "1":
                logger.warning(
                    "POI 搜索业务失败：%s/%s → %s", district, kw, data.get("info")
                )
                break
            pois = data.get("pois") or []
            if not pois:
                break
            for p in pois:
                if len(collected) >= quota:
                    break
                # 行政区过滤（返回可能含邻区结果）
                if district not in str(p.get("adname", "")):
                    continue
                try:
                    lng_s, lat_s = str(p["location"]).split(",")
                    lng, lat = round(float(lng_s), 6), round(float(lat_s), 6)
                except (ValueError, KeyError):
                    continue
                if (lng, lat) in collected:  # 坐标去重
                    continue
                collected[(lng, lat)] = {
                    "name": p.get("name", ""),
                    "lng": lng,
                    "lat": lat,
                    "district": str(p.get("adname", "")),
                    "address": str(p.get("address", "")),
                    "keyword": kw,
                }
            page += 1
    return list(collected.values())


def geocode_dc(key: str, limiter: RateLimiter | None = None, session=None) -> tuple[float, float, str]:
    """DC 坐标：地理编码「成都国际铁路港」，失败降级 config 青白江城区坐标。

    返回 (lng, lat, source)，source ∈ {amap_geocode, fallback_config}（登记台账）。
    """
    try:
        data = amap_get(
            {"_url": AMAP_GEOCODE_URL, "address": "成都国际铁路港", "city": "成都"},
            key,
            limiter,
            session,
        )
        geos = data.get("geocodes") or []
        if data.get("status") == "1" and geos:
            lng_s, lat_s = str(geos[0]["location"]).split(",")
            return round(float(lng_s), 6), round(float(lat_s), 6), "amap_geocode"
    except ConnectionError as exc:
        logger.warning("地理编码失败：%s", exc)
    logger.warning(
        "地理编码未命中，降级青白江城区坐标 (%s, %s)，登记台账",
        C.DC_FALLBACK_LNG, C.DC_FALLBACK_LAT,
    )
    return C.DC_FALLBACK_LNG, C.DC_FALLBACK_LAT, "fallback_config"


def build_poi_table(raw_pois: list[dict]) -> pd.DataFrame:
    """POI 列表 → 标准表：poi_id（P001…，按名称排序保证可复现）、六字段齐全。"""
    df = pd.DataFrame(raw_pois)
    if df.empty:
        return pd.DataFrame(columns=["poi_id", "name", "lng", "lat", "district", "address"])
    df = df.sort_values(["district", "name"]).reset_index(drop=True)
    df.insert(0, "poi_id", [f"P{i:03d}" for i in range(1, len(df) + 1)])
    return df[["poi_id", "name", "lng", "lat", "district", "address"]]


def fetch_pois(
    out_csv: Path | None = None,
    key: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """抓取（或读缓存）POI 表。缓存命中时不发起任何网络请求（04 号票验收项）。

    缓存有效性：文件存在且行数在 POI_TARGET_RANGE 内。
    """
    out_csv = Path(out_csv) if out_csv is not None else C.POI_CSV
    lo, hi = C.POI_TARGET_RANGE
    if out_csv.exists():
        cached = pd.read_csv(out_csv)
        if lo <= len(cached) <= hi:
            logger.info("POI 缓存命中（%d 个），跳过网络请求", len(cached))
            return cached
        logger.warning("缓存行数 %d 不在 [%d,%d]，重新抓取", len(cached), lo, hi)

    if key is None:
        key = get_amap_key()  # 读不到抛 MissingKeyError → 调用方询问用户

    limiter = RateLimiter()
    all_pois: list[dict] = []
    for district, quota in DISTRICT_QUOTA.items():
        got = search_district_pois(district, quota, key, limiter=limiter, session=session)
        logger.info("行政区 %s：抓到 %d/%d", district, len(got), quota)
        all_pois.extend(got)

    df = build_poi_table(all_pois)
    if not (lo <= len(df) <= hi):
        raise RuntimeError(
            f"POI 抓取数量 {len(df)} 不在目标区间 [{lo},{hi}]——"
            "可能是配额受限或关键词命中不足；不编造坐标。请检查 AMAP_KEY 配额后重跑，"
            "或手动放置符合格式的 poi.csv（列：poi_id,name,lng,lat,district,address）。"
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("POI 表落盘：%s（%d 个）", out_csv, len(df))
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        key = get_amap_key()
    except MissingKeyError as exc:
        raise SystemExit(str(exc))
    limiter = RateLimiter()
    df = fetch_pois(key=key)
    lng, lat, source = geocode_dc(key, limiter)
    print(f"\n=== 数据层 C-1 产物 ===\nPOI: {len(df)} 个 -> {C.POI_CSV}")
    print(f"DC 坐标: ({lng}, {lat})  来源: {source}")
    print(df.groupby("district").size().to_string())


if __name__ == "__main__":
    main()
