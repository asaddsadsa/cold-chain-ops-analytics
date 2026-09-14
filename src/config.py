"""集中式配置：全项目唯一「魔法数」来源。

下游所有数据层与模块脚本一律 import 本模块读取种子、路径、参数，
禁止各自硬编码。调整成本参数 / 阈值 / 种子只改这一处。

设计依据：
- 种子体系（每数据层独立具名）：见 CONTEXT.md「随机种子体系」、ADR 决策 Q4
- 路径相对化（禁止本机绝对路径）：见可复现纪律
- 车辆与成本参数锚点：见 data_sources_ledger.md 第 4 节、ADR-0009
- ABC 阈值：见 CONTEXT.md「ABC 分类」、ADR-0002
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# 路径常量：全部基于项目根的相对路径，禁止硬编码本机绝对路径
# ---------------------------------------------------------------------------
# 本文件位于 <root>/src/config.py，项目根 = 上一级
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_OLIST_DIR = RAW_DIR / "olist"
RAW_SOLOMON_DIR = RAW_DIR / "solomon"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
GEO_DIR = DATA_DIR / "geo"
DELIVERY_DIR = DATA_DIR / "delivery"
PROCESSED_DIR = DATA_DIR / "processed"

# 关键产物文件名（下游票按此读写，统一约定）
POI_CSV = GEO_DIR / "poi.csv"
DIST_MATRIX_KM_CSV = GEO_DIR / "dist_matrix_km.csv"
TIME_MATRIX_MIN_CSV = GEO_DIR / "time_matrix_min.csv"

REPORT_DIR = PROJECT_ROOT / "report"
TESTS_DIR = PROJECT_ROOT / "tests"


# ---------------------------------------------------------------------------
# 随机种子：每数据层独立具名，支持单层重跑不污染其他层（ADR Q4）
# ---------------------------------------------------------------------------
SEED_WAREHOUSE: int = 42  # 数据层 A：仓内运营仿真数据
SEED_OLIST: int = 43  # 数据层 B：Olist 清洗（如有随机聚合）
SEED_DELIVERY: int = 44  # 数据层 D：配送订单 / 车辆 / 在途异常
SEED_SIM: int = 45  # 数据层 F：SimPy 仓内作业仿真基础种子
# 仿真重复实验由 SEED_SIM 派生子种子（见 ADR / 07 号票），不在此另设全局种子。

#: 统一登记表，便于审计「种子是否集中声明」
SEEDS: dict[str, int] = {
    "warehouse": SEED_WAREHOUSE,
    "olist": SEED_OLIST,
    "delivery": SEED_DELIVERY,
    "sim": SEED_SIM,
}


# ---------------------------------------------------------------------------
# 场景设定（见 data_sources_ledger.md 第 2 节、ADR-0005）
# ---------------------------------------------------------------------------
SIM_DAYS: int = 90  # 连续运营天数
# 90 天窗口起始日期（数据层 A/D/F 共用同一窗口，结束 = 起始 + SIM_DAYS - 1）
SIM_START_DATE: str = "2026-06-01"
SKU_COUNT: int = 500  # 仿真仓 SKU 数
LOCATION_COUNT: int = 800  # 库位数
DAILY_OUTBOUND_RANGE: tuple[int, int] = (200, 400)  # 日均出库订单数区间
CATEGORY_CODES: tuple[str, ...] = tuple(f"P{i:02d}" for i in range(1, 11))  # P01–P10
HIGH_DIFF_CATEGORY: str = "P03"  # 盘点差异率埋点品类（数据层 A）
PICK_SLOW_HOURS: tuple[int, int] = (14, 16)  # 拣货效率低谷时段 [14:00,16:00)
REGION_CODES: tuple[str, ...] = tuple(f"R{i:02d}" for i in range(1, 9))  # R01–R08
HIGH_ANOMALY_REGION: str = "R07"  # 异常率埋点片区（数据层 D）

# ---------------------------------------------------------------------------
# 数据层 A 与数据层 F **共用**的仓内生成参数（情景假设，登记台账）
#
# 为什么集中在这里：这两层的关系是「互相印证」，而互证只有在两边真的用同一组参数时才成立。
# 先前两层各写一份私有副本（都注明「与数据层 A / 数据层 F 一致」），但没有任何东西强制它们
# 一致——只改一边，「互证」就退化成一句一致性声明，而写着一致的那行注释还留在原处。
# ---------------------------------------------------------------------------
#: 每单 1/2/3 行的概率（数据层 A 生成出库行、数据层 F 生成到达订单流）。
WAREHOUSE_LINES_PER_ORDER_P: tuple[float, float, float] = (0.50, 0.35, 0.15)
#: 14–16 点低谷整段拣货时长放大系数（数据层 A 的埋点③，数据层 F 的对照实验同源）。
WAREHOUSE_SLOWDOWN_RANGE: tuple[float, float] = (1.75, 2.05)
#: 拣货操作耗时对数正态 sigma（均值见 `PICK_SECONDS_PER_LINE_MEAN`）。
WAREHOUSE_PICK_HANDLE_SIGMA: float = 0.4
#: 仓内营业时段 [起, 止)（小时）：数据层 A 的下单窗、数据层 F 的到达窗口与秒轴共用。
WAREHOUSE_OPEN_HOURS: tuple[int, int] = (8, 18)
#: 订单到达的 NHPP 高峰/平峰强度比（平峰基准 1.0），高峰时段见 `SIM_PEAK_HOURS`。
#:
#: **这个值曾经两层各一份，而且差一个数量级**：数据层 A 用 1.8、数据层 F 用 10.0。
#: 统一到 1.8（层 A 的取值），理由有三：
#:   ① ADR-0001 把数据层 A 定为校准锚点、并写明「到达过程与 14–16 点低谷埋点相互印证」——
#:      让锚点去迁就仿真，方向是反的；
#:   ② 物理上这是**订单到达**（门店下单），不是「波次释放」。10h 窗口里 4 个高峰小时承载
#:      87% 的订单，等于一天有 6 小时几乎空转；1.8 对应 54.5%，是正常的下单节奏；
#:   ③ 10.0 的注释写着「波次拣货到达集中度」——那是拿参数制造负载起伏，好让人力实验有东西
#:      可看。选择参数以让实验显示预期结论，正是本项目纪律要禁止的事。
#:
#: 当时那句「代价是实验二拐点不再成立」只对了一半：拐点消失的真正原因不是强度，而是
#: 下游缺了「释放」这一环（见 `WAREHOUSE_RELEASE_DELAY_MIN`）。补上之后拐点回来了，
#: 而且不再靠人造参数。
WAREHOUSE_PEAK_INTENSITY: float = 1.8

#: 订单**释放**到开始拣货的延迟（分钟）——「订单到达」与「拣货开始」之间隔着的那一段。
#:
#: 数据层 A 一直有这个参数（私有常量 `_RELEASE_DELAY_MIN = (10, 40)`），数据层 F 则完全
#: 没有：仿真里订单一到就抢拣货员，等于假设「拣货员随时有空接单」。这个假设在 300 单/天
#: 的欠载系统里不显眼，却让实验二失去了机理——没有累积就没有排队，没有排队，加人自然
#: 换不来任何改善。提到 config 是为了让两层**共用一个数**，而不是各自记着「与另一层一致」。
WAREHOUSE_RELEASE_DELAY_MIN: tuple[float, float] = (10.0, 40.0)

#: 数据层 F 的**波次窗口**（分钟）：订单在窗内累积，到窗界才批量投入拣货队列。
#:
#: 取值是从锚点校准出来的，不是挑的：层 A 实测「下单 → 开始拣货」平均等待 ≈ 1500 秒
#: （单口径）／1615 秒（行口径），即约 25 分钟；波次窗内订单近似均匀到达，平均累积等待
#: = W/2，于是 W = 2 × 25 = 50 分钟——正是 `WAREHOUSE_RELEASE_DELAY_MIN` 的均值。
#:
#: **两层实现不同，这是刻意的**：层 A 用「每单独立的随机延迟」近似这件事，本层用「窗内
#: 累积、窗界同步释放」。差异在于前者不产生排队（延迟与人数无关），后者产生——而「加人
#: 有没有用」恰恰取决于排队。层 A 已验收、是校准锚点，其五张表与全部下游 KPI 都建在现状
#: 之上；为了仿真实验去改锚点，方向是反的。所以：**参数共享，实现各异，差异记录在此**。
#:
#: 本模型**只建模了波次的等待成本，没建模它的合并拣货收益**（一次波次内多单合并成一条
#: 行走路径）。因此这里的取值**不能**被读作「W 越小越好」——真实系统里 W 变大还有省行走
#: 的一侧，本模型没有它。见 `data_sources_ledger.md` 与 `exp2_staffing.json` 的 limitations。
WAREHOUSE_WAVE_INTERVAL_MIN: float = 2.0 * (sum(WAREHOUSE_RELEASE_DELAY_MIN) / 2.0)

# DC 设定：成都青白江物流聚集区（教学模拟，非真实企业设施）
DC_FALLBACK_LNG: float = 104.2510  # 青白江城区坐标（高德地理编码失败时降级用）
DC_FALLBACK_LAT: float = 30.8780

# 冷链城配设定（ADR-0005）
COLD_CHAIN_TEMP_RANGE: tuple[float, float] = (2.0, 8.0)  # 车厢温度达标区间 ℃

# POI 抓取规模
POI_TARGET_RANGE: tuple[int, int] = (40, 60)
AMAP_MIN_INTERVAL_SEC: float = 0.3  # 高德调用最小间隔（限速）

# 配送订单生成
ORDER_POISSON_MEAN_RANGE: tuple[float, float] = (1.0, 3.0)  # 每店每日订单泊松均值
TIME_WINDOW_OPEN: tuple[int, int] = (8, 18)  # 送达时间窗落在 08:00–18:00
DEFAULT_TIME_WINDOW_HOURS: float = 2.0  # 默认 2 小时窗
#: 门店当日**收货波次**（情景假设）：一家店一天只有一波冷链收货窗口，该店当日
#: 所有订单的送达窗都落在这一波里，只加小幅抖动以保留「逐单判定」的颗粒度。
#: 这是让 ADR-0004「单点一访 + 逐单判定时间窗达成率」自洽的前提——若同店各单的窗
#: 各自独立散布在 08:00–18:00，一次到访根本不可能同时满足它们，该 KPI 会失去意义。
RECEIVING_WAVE_START_RANGE: tuple[int, int] = (510, 900)  # 波次起点取值范围（当日分钟数 08:30–15:00）
RECEIVING_WAVE_JITTER_MIN: int = 10  # 各订单窗起相对波次起点的抖动上限（分钟）
ORDER_PRIORITY_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("高", 0.25), ("中", 0.50), ("低", 0.25),
)  # 配送优先级分布（情景假设；高=生鲜/药品时效件）

# ---------------------------------------------------------------------------
# 数据层 D：配送情景（06 号票；见 data_sources_ledger.md 第 2/4 节、ADR-0005）
# ---------------------------------------------------------------------------
DELIVERY_ORDERS_CSV = DELIVERY_DIR / "delivery_orders.csv"
DELIVERY_REGIONS_CSV = DELIVERY_DIR / "regions.csv"
DELIVERY_REGION_RULE_JSON = DELIVERY_DIR / "region_rule.json"
DELIVERY_VEHICLES_CSV = DELIVERY_DIR / "vehicles.csv"
DELIVERY_TRACKING_CSV = DELIVERY_DIR / "shipment_tracking.csv"
DELIVERY_ANOMALIES_CSV = DELIVERY_DIR / "shipment_anomalies.csv"
DELIVERY_TCO_JSON = DELIVERY_DIR / "tco_analysis.json"
DELIVERY_TCO_MD = DELIVERY_DIR / "tco_analysis.md"

#: Olist 实测的「商品重量/体积」经验分布（数据层 B 产出，06 号票只读不重算）。
#: 06 号票按此文件的**分布形状**（对数正态 sigma）抽样，尺度另按下方情景锚点标定。
OLIST_WEIGHT_VOLUME_JSON = PROCESSED_DIR / "olist" / "weight_volume_distribution.json"
#: Olist 实测真实延迟率（数据层 B 产出），作为在途**总体异常概率**的标定基准。
OLIST_DELAY_RATE_JSON = PROCESSED_DIR / "olist" / "olist_delay_rate.json"

#: 单张配送订单（= 该店一次冷链接货批次）的重量 / 体积中位锚点。
#: 情景假设：便利店/超市单批次冷链补货量（中位 ~40kg / ~0.3m³）。
#: 分布**形状**（对数正态 sigma ≈ 1.33/1.28，即长尾重货）取自 Olist 商品实测，
#: **尺度**不沿用 Olist（巴西电商单件商品中位仅 0.8kg，直接沿用会使 1.4t 载重约束
#: 完全失效、满载率 KPI 退化为 ~0.3%），故按冷链城配单批次补货量重新标定，见 ADR-0005
#: 「分布形状借自 Olist」条款与台账登记。
ORDER_WEIGHT_MEDIAN_KG: float = 40.0
ORDER_VOLUME_MEDIAN_M3: float = 0.30

#: 在途异常：总体异常概率 = 数据层 B 实测 Olist 真实延迟率（8.13%），逐日逐片区按倍率调整。
ANOMALY_TYPE_SHARES: tuple[tuple[str, float], ...] = (
    ("拥堵", 0.35), ("晚点", 0.35), ("温控波动", 0.20), ("故障", 0.10),
)  # 异常类型份额（情景假设，合计 1.0；「无异常」为剩余概率）
#: 异常词表的两个哨兵（与 `HIGH_DIFF_CATEGORY` / `HIGH_ANOMALY_REGION` 同一模式：
#: 领域词表的值集中在 config，禁止在代码里散落字面量）。
#: 「无异常」不是异常类型之一，而是背景——凡按异常聚合处都要显式排除它，
#: 否则异常率会被自身的分母稀释。
NO_ANOMALY: str = "无异常"
#: 温控波动的延误被刻意限制在 0–10 分钟（见 `ANOMALY_DELAY_MINUTES`），
#: 故它的严重度按**温升**判而非延误（规则见 `src/anomaly.py::with_severity`）。
TEMP_ANOMALY_TYPE: str = "温控波动"
ANOMALY_REGION_MULTIPLIER: float = 2.8  # 埋点②：HIGH_ANOMALY_REGION 异常概率倍率
#: 埋点①：周五午后拥堵叠加周末备货高峰——整体异常概率与「晚点」份额同时抬升。
#: 实测依据：仅抬高晚点份额时（倍率 2.2）周五/其他日晚点率之比仅 1.24、约 1.4σ，
#: 下游周度复盘无法稳定复现，故按业务机理补充整体倍率并加强晚点份额倍率。
ANOMALY_FRIDAY_OVERALL_MULTIPLIER: float = 1.4  # 周五整体异常概率倍率
ANOMALY_FRIDAY_LATE_MULTIPLIER: float = 3.5  # 周五「晚点」类异常份额倍率
ANOMALY_RAIN_LATE_MULTIPLIER: float = 3.0  # 埋点③：雨日「晚点」类异常份额倍率
#: 各异常类型的延误时长（分钟，均匀分布上下界）。
#: 「拥堵」计为**在途**延误（拉长行驶时长、表现为车速下降）；「晚点/故障/温控波动」计为
#: **发车**延误（推迟出发）。「温控波动」只造成轻微发车延误（0–10 分钟），其诊断价值在温度而非时刻。
ANOMALY_DELAY_MINUTES: dict[str, tuple[float, float]] = {
    "拥堵": (30.0, 180.0), "晚点": (60.0, 240.0), "故障": (90.0, 240.0), "温控波动": (0.0, 10.0),
}
#: 异常处理时长（分钟，均匀分布上下界）；「无异常」记为 0。
ANOMALY_HANDLING_MINUTES: dict[str, tuple[float, float]] = {
    "拥堵": (2.0, 15.0), "晚点": (5.0, 30.0), "故障": (45.0, 180.0), "温控波动": (10.0, 60.0),
}

#: 雨天概率：按月份给定的情景假设（成都 6–9 月雨季；雨天概率仅用于生成，不入 KPI 口径）。
RAIN_PROB_BY_MONTH: dict[int, float] = {6: 0.35, 7: 0.42, 8: 0.35, 9: 0.28}

#: 车厢温度：设定点与正常波动带宽；越界由「温控波动」异常与到点卸货开门共同造成。
#: 达标区间读 COLD_CHAIN_TEMP_RANGE（[2,8]℃）；设定点取区间中点。
CABIN_TEMP_SETPOINT_C: float = 5.0
CABIN_TEMP_NOISE_SIGMA_C: float = 0.7  # 正常波动标准差
#: 「温控波动」异常时相对设定点的**温升幅度**（℃，均匀分布）——取值区间保证越界（>8℃）。
CABIN_TEMP_EXCURSION_RISE_C: tuple[float, float] = (3.5, 6.5)
#: 到点卸货开门造成的温升（℃，均匀分布，仅作用于轨迹末段）——日常越界的主要来源。
CABIN_TEMP_DOOR_OPEN_RISE_C: tuple[float, float] = (0.5, 4.5)
UNLOAD_MINUTES: int = 10  # 轨迹末段卸货时长（分钟）

#: 在途轨迹采样间隔（分钟）。轨迹按「DC → 该运单送达点」直送分段仿真（不依赖路线，
#: 避免与模块二循环依赖）；直连里程/时长取数据层 C 真实路网矩阵。
TRACKING_SAMPLE_MINUTES: int = 5
#: 计划发车相对时间窗起点的缓冲（分钟）：计划到达 = 窗起点 - 缓冲，即常态准点。
PLANNED_ARRIVAL_BUFFER_MIN: float = 10.0

#: 现状派车（情景）：按固定份额轮派当日订单。份额刻意不均（主力车高、备用车低），
#: 以复现埋点④「部分车辆长期低满载」。权重个数 = FLEET_SIZE，和为 1。
FLEET_ALLOC_WEIGHTS: tuple[float, ...] = (
    0.105, 0.095, 0.090, 0.085, 0.080, 0.075, 0.070, 0.065,
    0.060, 0.055, 0.050, 0.030, 0.028, 0.026, 0.086,
)
#: 埋点④阈值：日装载率长期低于此值即认定为「长期低满载」车辆。
LOW_LOAD_VEHICLE_THRESHOLD: float = 0.35

#: TCO 曲线里程网格（km/日）与对照表用的代表行程参数。
TCO_MILEAGE_GRID_KM: tuple[float, ...] = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 150.0, 200.0, 250.0)
TCO_REFERENCE_DAILY_KM: float = 60.0  # 模式建议与对照表用的单车日均里程


# ---------------------------------------------------------------------------
# 车队规模（见 data_sources_ledger.md 第 4 节）
# ---------------------------------------------------------------------------
FLEET_SIZE: int = 15  # 车辆总数
RATED_PAYLOAD_T: float = 1.4  # 额定载重（吨）—— 4.2 米蓝牌冷链轻卡统一取值
RATED_VOLUME_M3: float = 18.0  # 额定容积（立方米）
#: 车队动力配比（情景假设）：成都新能源城配车推广 + 以租代购成本优势，纯电为主。
#: 合计必须 = FLEET_SIZE；柴油自购车用于长里程/补能不便线路。
FLEET_MODE_MIX: dict[str, int] = {"ev": 9, "diesel": 6}


# ---------------------------------------------------------------------------
# ABC 分类阈值（见 CONTEXT.md「ABC 分类」、ADR-0002）
# 数据源 = 仿真仓出库行数累计占比，不是 Olist
# ---------------------------------------------------------------------------
ABC_THRESHOLDS: tuple[float, float] = (0.70, 0.90)  # 累计占比「尚未达到」阈值者归入该类（见 warehouse_kpi.abc_classify）


# ---------------------------------------------------------------------------
# 模块一：仓储运营分析（08 号票）
# ---------------------------------------------------------------------------
WAREHOUSE_KPI_DIR = PROCESSED_DIR / "warehouse_kpi"
WAREHOUSE_KPI_JSON = WAREHOUSE_KPI_DIR / "kpi_overall.json"
WAREHOUSE_KPI_MD = WAREHOUSE_KPI_DIR / "warehouse_kpi_report.md"
WAREHOUSE_DRILLDOWN_CATEGORY_CSV = WAREHOUSE_KPI_DIR / "drilldown_stocktake_category.csv"
WAREHOUSE_DRILLDOWN_DATE_CSV = WAREHOUSE_KPI_DIR / "drilldown_stocktake_date.csv"
WAREHOUSE_DRILLDOWN_HOUR_CSV = WAREHOUSE_KPI_DIR / "drilldown_picking_hour.csv"
#: 仓内指标**逐日**序列（库存准确率 / 收货及时率 / 拣货效率 / 盘差率）。
#: 驾驶舱的近 30 天趋势与环比箭头读它——聚合值撑不起一条趋势线。
WAREHOUSE_DAILY_CSV = WAREHOUSE_KPI_DIR / "daily_kpi.csv"
WAREHOUSE_ABC_CSV = WAREHOUSE_KPI_DIR / "abc_classification.csv"
WAREHOUSE_ABC_PARETO_CSV = WAREHOUSE_KPI_DIR / "abc_pareto.csv"
WAREHOUSE_SLOTTING_JSON = WAREHOUSE_KPI_DIR / "slotting_before_after.json"
WAREHOUSE_SLOTTING_CSV = WAREHOUSE_KPI_DIR / "slotting_optimized_assignment.csv"

#: 模块一 KPI 表路径（数据层 A 产出，只读）
WAREHOUSE_TABLES: dict[str, Path] = {
    "sku": WAREHOUSE_DIR / "sku_master.csv",
    "location": WAREHOUSE_DIR / "location_master.csv",
    "inbound": WAREHOUSE_DIR / "inbound_receipts.csv",
    "outbound": WAREHOUSE_DIR / "outbound_orders.csv",
    "stocktake": WAREHOUSE_DIR / "inventory_stocktake.csv",
}

#: 库位分配（SKU → 库位）。数据层 A 产出、数据层 F 读取。
#: **为什么要交付它**：数据层 F 的「原始布局臂」要和数据层 A 的 outbound 对比（两条证据链
#: 互证），而原始布局的定义参数是 SKU 需求频率——它当时被当作内部参数丢掉了，仿真器只能拿
#: `abc_initial_label` 近似复刻。交付实际分配后，仿真器读到的就是同一份布局本身。
WAREHOUSE_SKU_LOCATION_CSV = WAREHOUSE_DIR / "sku_location_assignment.csv"

RECEIPT_TOLERANCE_MIN: float = 30.0  # 收货及时率容差（分钟），需求 3.x 硬口径


# ---------------------------------------------------------------------------
# 车辆与成本参数锚点（全部情景假设；锚点=需求文档 2026-09 市场检索，URL 待补）
# 见 data_sources_ledger.md 第 4 节、ADR-0009。单位：人民币元。
# ---------------------------------------------------------------------------
# 三档敏感性：低 / 中 / 高。中位=锚点值，低/高=围绕中位的合理区间端点。
# 司机工资、能源价格（油价/电价）、租金为三个敏感性参数。

#: 司机工资（元/工作日）。中位 269（月薪 7000 / 26 工作日）。
DRIVER_WAGE_PER_DAY: dict[str, float] = {"low": 231.0, "mid": 269.0, "high": 385.0}
# 低≈6000/月、高≈10000/月，对应招聘市场 6–10K 区间端点。

#: 柴油自购：日固定成本拆解（元/工作日），不含司机工资（单独敏感性）。
DIESEL_DEPRECIATION_PER_DAY: float = 62.0  # 购车 12 万、5 年直线、残值 20%
DIESEL_INSURANCE_PER_DAY: float = 22.0  # 营运保险约 7000 元/年
DIESEL_FIXED_EX_WAGE: float = DIESEL_DEPRECIATION_PER_DAY + DIESEL_INSURANCE_PER_DAY  # 84

#: 柴油公里变动成本拆解（元/km）。
DIESEL_FUEL_PER_KM: dict[str, float] = {"low": 0.90, "mid": 0.96, "high": 1.20}
# 百公里 11–14L × 柴油 7.5–8.5 元/L；低/高为能耗×油价组合端点。
DIESEL_UREA_PER_KM: float = 0.05
DIESEL_MAINTENANCE_PER_KM: float = 0.10

#: 纯电租赁：日固定成本拆解（元/工作日），租金含保险保养维修。
EV_RENT_PER_DAY: dict[str, float] = {"low": 77.0, "mid": 115.0, "high": 154.0}
# 中位 3000 元/月 ≈115；低≈以租代购 2000/月；高为偏高市场报价。
EV_FIXED_EX_WAGE_KEYS: tuple[str, ...] = ("rent",)  # 纯电固定成本=租金(+司机工资)

#: 纯电公里变动成本（元/km），即电价敏感性。
EV_ELEC_PER_KM: dict[str, float] = {"low": 0.30, "mid": 0.44, "high": 0.55}
# 约 37 度/百公里 × 快充 1.2–1.5 / 谷电 0.7–0.8 元/度。

#: 货拉拉外包对标计价（仅作决策对照，不进自营车队；ADR-0009）。
HUOLALA_START_FEE: float = 90.0  # 起步价，含前 5km
HUOLALA_START_KM: float = 5.0
HUOLALA_RATE_6_25: float = 5.0  # 6–25km 元/km
HUOLALA_BAND1_END_KM: float = 25.0  # 第一段计价里程上界（km）
HUOLALA_RATE_26_PLUS: tuple[float, float] = (3.5, 4.8)  # 26km 以上元/km 区间
HUOLALA_EXTRA_STOP_FEE: float = 38.0  # 超出 8 个点后，元/点
HUOLALA_FREE_STOPS: int = 8
#: 台账第 4 节的货拉拉城配实例锚点（≈300 元/趟 @14 点），**仅作量级对照、未进模型**
#: ——计价模型的产出见 `data/delivery/tco_analysis.md` 的对照表。
HUOLALA_CITY_INSTANCE_FEE: float = 300.0
#: 外包对照用的代表行程画像（情景假设：短途轻载 / 中程标准 / 长程多点）。
#: 与计价参数同属情景假设，统一放 config，禁止散落在生成模块里。
HUOLALA_REFERENCE_PROFILES: tuple[dict, ...] = (
    {"profile": "短途轻载", "stops": 6, "distance_km": 25.0},
    {"profile": "中程标准", "stops": 12, "distance_km": 60.0},
    {"profile": "长程多点", "stops": 18, "distance_km": 120.0},
)

#: 工作日折算
WORKDAYS_PER_MONTH: int = 26
WORKDAYS_PER_YEAR: int = 26 * 12

#: 三个敏感性参数名（看板/报告按此联动，ADR-0009 / 11 号票）
SENSITIVITY_PARAMS: tuple[str, ...] = ("driver_wage", "energy_price", "rent")
SENSITIVITY_LEVELS: tuple[str, ...] = ("low", "mid", "high")


# ---------------------------------------------------------------------------
# 仿真分布参数（数据层 F；见 data_sources_ledger.md 第 3 节）
# ---------------------------------------------------------------------------
PICK_SECONDS_PER_LINE_MEAN: float = 12.0  # 拣货耗时对数正态均值（秒/行，情景假设）
PACK_TRIANGULAR: tuple[float, float, float] = (30.0, 60.0, 120.0)  # 复核打包三角分布(秒)
WALK_SPEED_M_PER_SEC: float = 1.2  # 仓内行走速度（米/秒，情景假设）
SIM_PEAK_HOURS: tuple[tuple[int, int], ...] = ((9, 11), (14, 16))  # 非齐次泊松双高峰
SIM_REPEATS: int = 30  # 每档重复仿真次数（报告均值 + 95% CI）
SIM_PICKER_LEVELS: tuple[int, ...] = (4, 5, 6)  # 实验二人力档位
SIM_REVIEW_STATIONS: int = 3  # 复核台数量（固定资源）
PICKER_DAILY_COST: float = 230.0  # 拣货员日成本（元/人·日，情景假设，登记台账）
# 权衡曲线人力成本轴 = 拣货员数 × PICKER_DAILY_COST；用于实验二「时长—人力成本」拐点分析

# SimPy 代表日规模（情景假设，登记台账）
# 注意：300 单/天 × ~100s/单 ≈ 3 万秒工作量，对 4–6 拣货员（10h 窗口=14.4 万秒/人）
# 属欠载系统。这个张力是真的，**但它此前被误读成「所以人力实验没内容」**——真正缺的是
# 「释放」那一环：订单一到就抢拣货员，等于假设拣货员随时有空接单，没有累积就没有排队，
# 加人自然换不来任何改善。补上波次释放（`WAREHOUSE_WAVE_INTERVAL_MIN`）后，系统仍欠载，
# 但波次窗内的瞬时负载会造成排队，人力档位重新可分——且不再需要人造的到达强度。
SIM_N_ORDERS_PER_DAY: int = 300  # 代表日订单数（数据层 A 日均 200–400 中值）

#: 实验二的**波次窗口敏感性档位**（分钟）。0 = 不累积（到达即抢拣货员），作对照用；
#: 其余为校准值 `WAREHOUSE_WAVE_INTERVAL_MIN` 的 1/2、1、2、4 倍。
#: 用几何级数而不是随手挑几个数：要回答的是「结论对 W 有多敏感」，那得让 W 跨一个量级。
SIM_WAVE_LEVELS: tuple[float, ...] = (0.0, 25.0, 50.0, 100.0, 200.0)


# ---------------------------------------------------------------------------
# 优化与门禁（模块二 / 数据层 E）
# ---------------------------------------------------------------------------
#: OR-Tools 求解的**主停止条件**：累计找到这么多个解就停。
#:
#: 为什么不是「跑 N 秒」：那是**墙钟**，同一个输入在负载不同的时刻会在不同的迭代次数处
#: 停下，结果随之漂移。本仓库实测过这件事——同一天的代表日求解两次差 0.7%（876.15 vs
#: 870.16 km），而报告里的数字是写死的，于是「报告与产物一致」的测试会随机变红。
#: OR-Tools 的 routing 搜索本身**没有随机源**（参数表里根本没有 random_seed，GLS 的
#: 扰动序列是确定的），唯一的不确定性就是墙钟；换成基于解数的条件，结果即可复现。
#: 实测 C101 在 200 个解处的距离与 30 秒时限的**完全相同**（829.01），耗时反而从 30 秒
#: 降到约 12 秒。
VRPTW_SOLUTION_LIMIT: int = 200
#: 墙钟**安全网**：正常情况下 solver 会在远早于此的时刻达到 `VRPTW_SOLUTION_LIMIT`。
#: 它只用来防止某个算例长时间凑不满解数而挂住。产物里记录实际耗时，一旦接近本值就说明
#: 本次结果又不可复现了，该调大解数上限或排查建模，而不是把这个值调小。
VRPTW_TIME_LIMIT_SEC: float = 300.0
SOLOMON_GAP_GATE: float = 0.10  # Solomon 算法门禁：gap>10% 不放行（ADR / 05 号票）
SOLOMON_INSTANCES: tuple[str, ...] = ("C101", "R101", "RC101")
WHATIF_VEHICLE_RANGE: tuple[int, int] = (5, 20)  # what-if 车辆数预计算档位（ADR-0010）

# ---------------------------------------------------------------------------
# 模块二（上）：运输调度优化核心（10 号票）
# ---------------------------------------------------------------------------
TRANSPORT_DIR = PROCESSED_DIR / "transport"
TRANSPORT_KPI_JSON = TRANSPORT_DIR / "transport_kpi.json"
TRANSPORT_KPI_MD = TRANSPORT_DIR / "transport_kpi_report.md"
TRANSPORT_COMPARISON_CSV = TRANSPORT_DIR / "baseline_vs_optimized.csv"
TRANSPORT_TRIPS_CSV = TRANSPORT_DIR / "trips.csv"
TRANSPORT_ROUTES_BASELINE_GEOJSON = TRANSPORT_DIR / "routes_baseline.geojson"
TRANSPORT_ROUTES_OPTIMIZED_GEOJSON = TRANSPORT_DIR / "routes_optimized.geojson"
TRANSPORT_SAVINGS_JSON = TRANSPORT_DIR / "savings_extrapolation.json"
#: 全 90 天**基线贪心**的逐日 KPI（时间窗达成率 / 满载率 / 单均成本 / 里程…）。
#: 驾驶舱的「近 30 天趋势 + 环比箭头」读它——聚合值撑不起一条趋势线。
#: 该表只依赖确定性的基线贪心，与代表日的 OR-Tools 求解无关（ADR-0013）。
TRANSPORT_DAILY_CSV = TRANSPORT_DIR / "daily_baseline_kpis.csv"

#: 单点卸货服务时长（分钟）——冷链城配到点开门卸货，情景假设，登记台账。
TRANSPORT_STOP_SERVICE_MIN: float = 15.0
#: DC 最早发车时刻（当日分钟数）：08:00。
TRANSPORT_DEPOT_OPEN_MIN: float = 8 * 60
#: 单车单日最长在途时长（分钟）：10 小时工作窗，超出即不可行（情景假设）。
TRANSPORT_MAX_ROUTE_MIN: float = 10 * 60
#: 计划延误缓冲（分钟）：排线时要求**提前**这么多分钟送达，给在途异常留出吸收空间。
#: 数据层 D 的异常表会给实际到达叠加 0–240 分钟的延误；不预留缓冲的计划在延误面前
#: 会成批破窗（10 号票实测：接入实际延误后优化方案 15.2% vs 基线 34.1%——
#: 优化压低了成本，却把原本靠长路线"顺带"攒下的时间余量也压没了）。
TRANSPORT_DELAY_BUFFER_MIN: float = 30.0
#: 误点惩罚（元/分钟，折成里程当量后进 OR-Tools 目标）：越过**服务窗起点**每一分钟的代价。
#: 只靠延误缓冲仍不够——目标纯成本导向时求解器把「早到」当浪费，排出的计划在真实
#: 在途延误面前比朴素贪心更脆（实测优化 30.3% vs 基线 41.7%）。软上界设在窗止无效
#: （硬约束本就禁止越界，惩罚永不触发），必须设在**窗起**才能逼出「贴窗起到达」，
#: 把窗起→窗止整段余量留给延误消耗。取值量级：300 元/分钟 ≈ 0.3 km 里程当量。
TRANSPORT_LATE_PENALTY_PER_MIN: float = 300.0
#: 基线「人工就近派车」的贪心规则说明（落盘进 KPI JSON，供复现者核对口径）。
TRANSPORT_BASELINE_RULE: str = (
    "最近邻贪心：从 DC 出发，每步选择「载重/容积仍装得下且能按时窗送达」的最近未访问点；"
    "无可行点时回 DC 收车并派下一辆。模拟『人工就近派车』的现有做法。"
)
#: 代表日单均节省率外推为月/年金额时的假设（登记台账）。
TRANSPORT_EXTRAPOLATION_ASSUMPTION: str = (
    "代表日=90 天中订单量最大的工作日（ADR-0003）。月/年金额 = 代表日**单均成本节省** × "
    "代表日订单数 × 月/年配送天数；即隐含假设「全年每个配送日的订单强度与单均节省率都等于"
    "代表日」。属情景假设，不外推为真实财务承诺。"
)

# ---------------------------------------------------------------------------
# 模块二（下）：周度异常复盘 + TCO 决策分析 + what-if 预计算（11 号票）
# ---------------------------------------------------------------------------
TRANSPORT_ANOMALY_WEEKLY_CSV = TRANSPORT_DIR / "anomaly_weekly.csv"
TRANSPORT_ANOMALY_CASES_CSV = TRANSPORT_DIR / "anomaly_cases.csv"
TRANSPORT_ANOMALY_POINTS_JSON = TRANSPORT_DIR / "anomaly_points.json"
TRANSPORT_TCO_JSON = TRANSPORT_DIR / "transport_tco.json"
TRANSPORT_TCO_MD = TRANSPORT_DIR / "transport_tco_report.md"
TRANSPORT_WHATIF_JSON = TRANSPORT_DIR / "whatif_vehicles.json"

#: 典型案例清单每类异常最多取几条（报告与看板展示用）。
ANOMALY_CASES_PER_TYPE: int = 3

#: 「自营 vs 外包基本持平」判定带宽：两者单趟成本相差在此比例以内即判持平。
#: 取 2% 是为了不让计价规则的小数级差异被渲染成结论性的「谁更省」。
HUOLALA_TIE_BAND: float = 0.98

#: what-if 每档 VRPTW 的墙钟安全网（秒），语义同 `VRPTW_TIME_LIMIT_SEC`：正常路径由
#: `VRPTW_SOLUTION_LIMIT` 决定何时停，这个值只兜住「凑不满解数」的意外。
#: ADR-0010 定车辆数 5–20 共 16 档离线预计算，每档按代表日规模跑一次——一次性成本换
#: 看板滑块零等待。
WHATIF_TIME_LIMIT_SEC: float = 300.0

#: 外包对照的计价基准（写进产物，供复现者核对口径）。
TRANSPORT_OUTSOURCE_BASIS: str = (
    "自营单趟成本 = 该趟实际里程 × 公里变动成本 + 日固定成本。代表日每台车恰好只跑 1 趟"
    "（用车数 = 趟次数，10 号票产物可验），故日固定成本全额摊入该趟是**精确分摊**而非近似；"
    "若将来一趟变多趟，该口径会高估单趟成本，模块会在产物里标注。"
    "货拉拉按 config 计价规则对同一里程/点数报价。仅作决策对照，不进自营车队成本（ADR-0009）。"
)

# 高德距离矩阵：批量接口按目的地逐列请求（ADR-0008）
AMAP_RETRY_BACKOFF_SEC: tuple[int, ...] = (1, 2, 4)  # 指数退避，至多 3 次


# ---------------------------------------------------------------------------
# 本模块只声明数值，不定义模型。
#
# 车队成本模型（日固定成本 / 公里变动成本 / 日总成本 / 盈亏平衡里程 / 货拉拉计价 /
# TCO 分析 / 自营 vs 外包对照）住在 `src/costing.py`——它读的就是上面这些常量。
# 之所以分开：`config` 是「唯一魔法数来源」，而成本模型是**行为**。混在一个文件里会让
# 「改一个参数」与「改一条公式」看起来是同一件事，也让「成本模型住哪」这个问题没有答案
# （收口前它散在本文件、gen_delivery_data 与 transport_decisions 三处）。
# ---------------------------------------------------------------------------

