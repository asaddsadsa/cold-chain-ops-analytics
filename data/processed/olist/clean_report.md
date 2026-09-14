# 数据层 B：Olist 清洗报告（真实观测）

## 清洗前后行数
- orders：99,441 → 96,282
- order_items：112,650 → 109,953

## 剔除原因统计
- dropped_not_delivered：2,963
- dropped_missing_timestamp：8
- dropped_time_contradiction：188

## Olist 真实延迟率（供在途异常概率标定）
- 订单数：96,282
- 延迟订单：7,823
- **真实延迟率：8.13%**（真实准时交付率 91.87%）

## 商品重量/体积经验分布（对数正态拟合，供数据层 D 抽样）
- product_weight_g：n=109,927，μ=6.689，σ=1.3282，中位=700.0，均值=2090.6
- product_volume_cm3：n=109,935，μ=8.8457，σ=1.2797，中位=6460.0，均值=15179.8
