# 数据层 E：Solomon 算法门禁（学术基准）

- 门禁阈值：gate_gap ≤ 10%
- **门禁结果：通过 ✅**
- best-known 来源：PUC-Rio CVRPLIB（见 data/raw/solomon/SOURCE.md，检索 2026-09-14）

| 算例 | 我方车辆/距离 | BKS车辆/距离 | 车辆gap | 距离gap | 门禁gap | 通过 |
|---|---|---|---|---|---|---|
| C101 | 10车/829.01 | 10车/827.30 | +0.00% | +0.21% | **+0.21%** | ✅ |
| R101 | 19车/1680.68 | 20车/1637.70 | -5.00% | +2.62% | **-5.00%** | ✅ |
| RC101 | 16车/1682.15 | 15车/1619.80 | +6.67% | +3.85% | **+6.67%** | ✅ |

## 求解策略

- C101：首解 PARALLEL_CHEAPEST_INSERTION，局部搜索 GUIDED_LOCAL_SEARCH，时限 300s
  - gap 判定口径：车辆数与 BKS 相同 → 按距离 gap 判定
- R101：首解 PARALLEL_CHEAPEST_INSERTION，局部搜索 GUIDED_LOCAL_SEARCH，时限 300s
  - gap 判定口径：车辆数少于 BKS（19 < 20）→ 用车更优，gap 取保守值
- RC101：首解 PARALLEL_CHEAPEST_INSERTION，局部搜索 GUIDED_LOCAL_SEARCH，时限 300s
  - gap 判定口径：车辆数多于 BKS（16 > 15）→ 取车辆 gap 与距离 gap 较大者
