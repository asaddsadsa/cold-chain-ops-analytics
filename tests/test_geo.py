"""数据层 C（geo_poi / geo_matrix）单元测试。

网络调用全部用 fake session / monkeypatch 隔离——不消耗真实配额、不依赖网络。
覆盖：缓存命中零请求、限速间隔、矩阵对称性/对角0/维度、密钥读取顺序、POI 去重排序。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import geo_matrix as GM
from src import geo_poi as GP


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    """记录每次调用；按 URL 返回预置响应。"""

    def __init__(self, responder):
        self.responder = responder
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        return FakeResponse(self.responder(url, params))


def make_place_responder(pois_by_district: dict[str, list[dict]]):
    """构造 place/text 响应器：按 city/district 返回 POI 列表。"""

    def responder(url, params):
        if "geocode" in url:
            return {"status": "1", "geocodes": [{"location": "104.25,30.88"}]}
        kw = params.get("keywords", "")
        district = next(iter(pois_by_district), "青白江区")
        pool = pois_by_district.get(district, [])
        # 简化：所有关键词返回同一池（测试只关心数量/去重/过滤）
        return {"status": "1", "pois": pool, "count": str(len(pool))}

    return responder


def _poi(name, lng, lat, district="青白江区"):
    return {
        "name": name,
        "location": f"{lng},{lat}",
        "adname": district,
        "address": f"{district}{name}路1号",
    }


# ---------------------------------------------------------------------------
# 密钥读取顺序
# ---------------------------------------------------------------------------
class TestAmapGetRetry:
    """amap_get 的退避重试：网络异常 + QPS 业务错误码（实测高德个人 key 偶发 QPS 超限）。"""

    def test_retries_on_qps_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(GP.time, "sleep", lambda *_: None)  # 退避不拖慢测试
        calls = {"n": 0}

        def responder(url, params):
            calls["n"] += 1
            if calls["n"] <= 2:  # 前两次 QPS 超限
                return {"status": "0", "info": "CUQPS_HAS_EXCEEDED_THE_LIMIT", "infocode": "10019"}
            return {"status": "1", "pois": [], "count": "0"}

        session = FakeSession(responder)
        out = GP.amap_get({"_url": GP.AMAP_PLACE_URL}, "k", GP.RateLimiter(min_interval=0), session)
        assert out["status"] == "1"
        assert calls["n"] == 3  # 退避重试后成功

    def test_non_retryable_business_error_returns_immediately(self, monkeypatch):
        monkeypatch.setattr(GP.time, "sleep", lambda *_: None)
        calls = {"n": 0}

        def responder(url, params):
            calls["n"] += 1
            return {"status": "0", "info": "INVALID_PARAMS", "infocode": "20001"}

        session = FakeSession(responder)
        out = GP.amap_get({"_url": GP.AMAP_PLACE_URL}, "k", GP.RateLimiter(min_interval=0), session)
        assert out["status"] == "0"
        assert calls["n"] == 1  # 参数错误不重试

    def test_qps_exhausts_retries_returns_last(self, monkeypatch):
        monkeypatch.setattr(GP.time, "sleep", lambda *_: None)
        calls = {"n": 0}

        def responder(url, params):
            calls["n"] += 1
            return {"status": "0", "info": "CUQPS_HAS_EXCEEDED_THE_LIMIT", "infocode": "10019"}

        session = FakeSession(responder)
        out = GP.amap_get({"_url": GP.AMAP_PLACE_URL}, "k", GP.RateLimiter(min_interval=0), session)
        # 重试用尽（1 初始 + 3 退避 = 4 次），返回最后一次业务响应而非抛异常
        assert calls["n"] == 4
        assert out["infocode"] == "10019"


class TestKeyResolution:
    def test_env_var_first(self, monkeypatch):
        monkeypatch.setenv("AMAP_KEY", "env_key_123")
        assert GP.get_amap_key() == "env_key_123"

    def test_dotenv_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AMAP_KEY", raising=False)
        (tmp_path / ".env").write_text('AMAP_KEY="dotenv_key_456"\n', encoding="utf-8")
        assert GP.get_amap_key(project_root=tmp_path) == "dotenv_key_456"

    def test_missing_raises(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AMAP_KEY", raising=False)
        with pytest.raises(GP.MissingKeyError):
            GP.get_amap_key(project_root=tmp_path)


# ---------------------------------------------------------------------------
# 限速器
# ---------------------------------------------------------------------------
class TestRateLimiter:
    def test_min_interval_enforced(self):
        limiter = GP.RateLimiter(min_interval=0.05)
        t0 = time.monotonic()
        limiter.wait()
        limiter.wait()  # 第二次必须等满间隔
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.05

    def test_first_call_no_wait(self):
        limiter = GP.RateLimiter(min_interval=5.0)
        t0 = time.monotonic()
        limiter.wait()
        assert time.monotonic() - t0 < 0.5  # 首次不等待


# ---------------------------------------------------------------------------
# POI 抓取
# ---------------------------------------------------------------------------
class TestPoiTable:
    def test_sorted_and_deduped_ids(self):
        raw = [
            {"name": "b店", "lng": 104.2, "lat": 30.7, "district": "青白江区", "address": "x"},
            {"name": "a店", "lng": 104.3, "lat": 30.8, "district": "锦江区", "address": "y"},
        ]
        df = GP.build_poi_table(raw)
        assert list(df["poi_id"]) == ["P001", "P002"]
        assert list(df.columns) == ["poi_id", "name", "lng", "lat", "district", "address"]
        # 按 district+name 排序：锦江区 a店 在前
        assert df.iloc[0]["name"] == "a店"

    def test_empty_table_has_columns(self):
        df = GP.build_poi_table([])
        assert list(df.columns) == ["poi_id", "name", "lng", "lat", "district", "address"]
        assert len(df) == 0


class TestFetchPoisCache:
    def test_cache_hit_zero_network(self, tmp_path, monkeypatch):
        # 预置合法缓存（行数在区间内）
        poi_csv = tmp_path / "poi.csv"
        n = (C.POI_TARGET_RANGE[0] + C.POI_TARGET_RANGE[1]) // 2
        df = pd.DataFrame(
            {
                "poi_id": [f"P{i:03d}" for i in range(1, n + 1)],
                "name": [f"店{i}" for i in range(n)],
                "lng": np.linspace(104.0, 104.5, n),
                "lat": np.linspace(30.5, 30.9, n),
                "district": ["青白江区"] * n,
                "address": ["addr"] * n,
            }
        )
        df.to_csv(poi_csv, index=False)

        # 若发起网络请求或读密钥则失败
        def boom(*a, **k):
            raise AssertionError("缓存命中不应触发网络/密钥读取")

        monkeypatch.setattr(GP, "get_amap_key", boom)
        out = GP.fetch_pois(out_csv=poi_csv)
        assert len(out) == n  # 直接返回缓存

    def test_search_filters_district_and_dedups(self):
        pool = [
            _poi("店1", 104.10, 30.10),
            _poi("店1", 104.10, 30.10),  # 同坐标重复
            _poi("店2", 104.11, 30.11),
            _poi("外区店", 104.12, 30.12, district="新都区"),  # 非目标区，应过滤
        ]
        session = FakeSession(make_place_responder({"青白江区": pool}))
        got = GP.search_district_pois(
            "青白江区", quota=10, key="k",
            limiter=GP.RateLimiter(min_interval=0), session=session,
        )
        names = {g["name"] for g in got}
        assert names == {"店1", "店2"}  # 去重 + 行政区过滤


class TestGeocodeDc:
    def test_geocode_success(self):
        session = FakeSession(make_place_responder({}))
        lng, lat, source = GP.geocode_dc("k", GP.RateLimiter(min_interval=0), session)
        assert source == "amap_geocode"
        assert (lng, lat) == (104.25, 30.88)

    def test_geocode_fallback_on_empty(self):
        def responder(url, params):
            return {"status": "1", "geocodes": []}

        session = FakeSession(responder)
        lng, lat, source = GP.geocode_dc("k", GP.RateLimiter(min_interval=0), session)
        assert source == "fallback_config"
        assert (lng, lat) == (C.DC_FALLBACK_LNG, C.DC_FALLBACK_LAT)


# ---------------------------------------------------------------------------
# 距离矩阵
# ---------------------------------------------------------------------------
class TestDistanceColumn:
    def test_batch_parses_results(self, monkeypatch):
        # monkeypatch amap_get 返回 3 origins → 1 dest 的结果
        def fake_amap_get(params, key, limiter, session=None):
            origins = params["origins"].split("|")
            return {
                "status": "1",
                "results": [
                    {"distance": str(1000 * (i + 1)), "duration": str(60 * (i + 1))}
                    for i in range(len(origins))
                ],
            }

        monkeypatch.setattr(GM, "amap_get", fake_amap_get)
        origins = np.array([[104.0, 30.0], [104.1, 30.1], [104.2, 30.2]])
        km, mins = GM._request_distance_column(origins, (104.5, 30.5), "k", GP.RateLimiter(0))
        assert list(km) == [1.0, 2.0, 3.0]  # 米→公里
        assert list(mins) == [1.0, 2.0, 3.0]  # 秒→分钟

    def test_batch_splits_over_limit(self, monkeypatch):
        calls = []

        def fake_amap_get(params, key, limiter, session=None):
            calls.append(len(params["origins"].split("|")))
            n = calls[-1]
            return {"status": "1", "results": [{"distance": "1000", "duration": "60"}] * n}

        monkeypatch.setattr(GM, "amap_get", fake_amap_get)
        origins = np.random.default_rng(0).uniform(104, 105, size=(150, 2))
        GM._request_distance_column(origins, (104.5, 30.5), "k", GP.RateLimiter(0))
        assert calls == [100, 50]  # 超过批量上限 100 → 分两批


class TestValidateMatrix:
    def test_valid_passes(self):
        ids = ["DC", "P001", "P002"]
        m = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
        GM._validate_matrix(pd.DataFrame(m, index=ids, columns=ids), ids)

    def test_asymmetric_fails(self):
        ids = ["DC", "P001"]
        m = np.array([[0, 1], [2, 0]], dtype=float)
        with pytest.raises(AssertionError):
            GM._validate_matrix(pd.DataFrame(m, index=ids, columns=ids), ids)

    def test_wrong_dim_fails(self):
        ids = ["DC", "P001", "P002"]
        m = np.zeros((2, 2))
        with pytest.raises(AssertionError):
            GM._validate_matrix(pd.DataFrame(m, index=["DC", "P001"], columns=["DC", "P001"]), ids)


class TestMatrixCache:
    def test_cache_hit_zero_network(self, tmp_path, monkeypatch):
        # 预置 POI 缓存 + 维度匹配的矩阵缓存
        n_poi = 5
        poi_csv = tmp_path / "poi.csv"
        pd.DataFrame(
            {
                "poi_id": [f"P{i:03d}" for i in range(1, n_poi + 1)],
                "name": [f"店{i}" for i in range(n_poi)],
                "lng": np.linspace(104.0, 104.4, n_poi),
                "lat": np.linspace(30.5, 30.8, n_poi),
                "district": ["青白江区"] * n_poi,
                "address": ["a"] * n_poi,
            }
        ).to_csv(poi_csv, index=False)

        n = n_poi + 1  # +DC
        ids = ["DC", *[f"P{i:03d}" for i in range(1, n_poi + 1)]]
        m = np.ones((n, n))
        np.fill_diagonal(m, 0)
        dist_csv, time_csv = tmp_path / "dist.csv", tmp_path / "time.csv"
        pd.DataFrame(m, index=ids, columns=ids).to_csv(dist_csv)
        pd.DataFrame(m * 2, index=ids, columns=ids).to_csv(time_csv)

        def boom(*a, **k):
            raise AssertionError("矩阵缓存命中不应触发网络/密钥")

        monkeypatch.setattr(GM, "get_amap_key", boom)
        monkeypatch.setattr(GM, "build_matrix_amap", boom)
        d, t = GM.fetch_matrix(poi_csv=poi_csv, dist_csv=dist_csv, time_csv=time_csv)
        assert d.shape == (n, n) and t.shape == (n, n)

    def test_load_nodes_dc_first(self, tmp_path):
        poi_csv = tmp_path / "poi.csv"
        pd.DataFrame(
            {
                "poi_id": ["P001", "P002"],
                "name": ["a", "b"],
                "lng": [104.0, 104.1],
                "lat": [30.5, 30.6],
                "district": ["青白江区", "青白江区"],
                "address": ["x", "y"],
            }
        ).to_csv(poi_csv, index=False)
        ids, coords = GM.load_nodes(poi_csv)
        assert ids == ["DC", "P001", "P002"]  # DC 占第 0 位
        assert coords.shape == (2, 2)  # coords 只含 POI（DC 单独传）

    def test_load_nodes_missing_poi_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            GM.load_nodes(tmp_path / "nope.csv")
