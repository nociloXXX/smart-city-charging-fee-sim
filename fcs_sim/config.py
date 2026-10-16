"""全局参数配置。

复现规模：8x8 方格路网 / 5000 辆电动汽车 / 8 座快充站 / IEEE 33 节点配电网。
所有可调参数集中在这里，方便做敏感性分析和场景对比。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# 充电站布局：(路网行, 路网列, IEEE33 节点号)
STATION_LAYOUT: Tuple[Tuple[int, int, int], ...] = (
    (1, 1, 6),
    (1, 5, 10),
    (2, 3, 14),
    (3, 6, 17),
    (4, 1, 20),
    (5, 4, 24),
    (6, 2, 26),
    (6, 6, 30),
)

# 服务费档位（元/kWh），模块四按"提高/降低一个等级"离散调整
FEE_LEVELS: Tuple[float, ...] = (1.40, 1.45, 1.50, 1.55, 1.60)


def tou_price(n_slots: int = 96, slot_min: int = 15) -> np.ndarray:
    """分时电价（元/kWh）：00:00-08:00 谷 0.35，10:00-15:00 与 18:00-21:00 峰 1.05，其余平 0.65。"""
    hour = np.arange(n_slots) * slot_min / 60.0
    price = np.full(n_slots, 0.65)
    price[hour < 8.0] = 0.35
    peak = ((hour >= 10.0) & (hour < 15.0)) | ((hour >= 18.0) & (hour < 21.0))
    price[peak] = 1.05
    return price


def make_fee_levels(fee_min: float, fee_max: float, step: float = 0.05) -> Tuple[float, ...]:
    """按上下限生成服务费档位。"""
    n = max(1, round((fee_max - fee_min) / step))
    return tuple(round(fee_min + i * step, 2) for i in range(n + 1))


def build_config(**overrides) -> "Config":
    """由键值覆盖构造 Config，便于命令行 / 图形界面传参。"""
    cfg = Config()
    for key, value in overrides.items():
        if value is None:
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


@dataclass
class Config:
    """仿真参数集合。"""

    # ---------------- 路网 ----------------
    rows: int = 8
    cols: int = 8
    spacing_km: float = 2.0            # 相邻路口间距，区域约 14 x 14 km
    free_speed_kmh: float = 40.0       # 自由流车速
    peak_speed_factor: float = 0.70    # 早晚高峰速度折减系数

    # ---------------- 车辆与出行链 ----------------
    n_ev: int = 5000
    battery_range: Tuple[float, float] = (24.0, 24.0)      # 电池容量 kWh（论文：Nissan Leaf 24 kWh）
    consumption_range: Tuple[float, float] = (0.16, 0.22)  # 百公里电耗 kWh/km
    init_soc_range: Tuple[float, float] = (0.30, 0.80)     # 初始 SOC
    soc_trigger: float = 0.25          # 里程焦虑阈值（论文：SOC < 25%）
    charge_soc_gain: float = 0.75      # 单次快充补充的 SOC（论文：30 min 补 75% SOC）
    charge_duration_min: float = 30.0  # 单次快充时长（论文：30 min）
    time_value: float = 30.0           # 时间价值 元/h
    seed: int = 20261014
    chain_weights: Tuple[float, float, float] = (0.60, 0.25, 0.15)
    public_share: float = 0.10         # 依赖公共快充的车队比例（其余车辆有私人桩）
    home_depart_mean: float = 7.5      # 家出发时刻（小时）
    home_depart_sd: float = 0.8
    work_dwell: float = 8.0            # 工作地停留时长 h
    mall_dwell: float = 1.5            # 商场停留时长 h

    # ---------------- 充电站 ----------------
    station_layout: Tuple[Tuple[int, int, int], ...] = STATION_LAYOUT
    pile_power_kw: float = 80.0        # 快充单桩恒定功率（论文：80 kW）
    piles_per_station: int = 4         # 每站桩数
    fee_levels: Tuple[float, ...] = FEE_LEVELS
    fee_init: float = 1.50             # 初始服务费（元/kWh，取区间中部，留出上下调空间）
    include_energy_price: bool = True  # 选站成本中是否计入充电电费
    random_choice: bool = True         # True 时用 Logit 概率选择（论文的用户决策模型）
    logit_theta: float = 0.30          # Logit 尺度参数 元^-1

    # ---------------- 配电网 ----------------
    base_load_scale: float = 0.45      # 基础负荷缩放（基准算例负荷偏重，缩放后留出电压裕度）
    v_lower: float = 0.93              # 电压带下限（论文：0.93），低于则上调服务费
    v_upper: float = 1.07              # 电压带上限（论文：1.07）
    v_light: float = 0.96              # 电压良好阈值，高于则可下调服务费
    delta_v_penalty: float = 5.0       # 越带电压偏差惩罚系数 K_dV（论文 K_ΔV）
    criticality_mode: str = "degree"   # degree（节点度数）| load_share | mixed

    # ---------------- 迭代 ----------------
    max_iter: int = 8
    vdini_tol: float = 5e-3            # VDINI 相对变化收敛阈值
    wait_smoothing: float = 0.5        # 等待时间的迭代平滑系数
    fee_damping: float = 0.4           # 服务费调整阻尼（0~1，越小越稳）

    # ---------------- 时间离散 ----------------
    n_slots: int = 96
    slot_min: int = 15

    # ---------------- 派生量 ----------------
    @property
    def slot_hours(self) -> float:
        return self.slot_min / 60.0

    @property
    def n_stations(self) -> int:
        return len(self.station_layout)

    @property
    def station_buses(self) -> Tuple[int, ...]:
        return tuple(b for _, _, b in self.station_layout)

    def fee_index(self, fee: float) -> int:
        """把连续服务费映射到最接近的档位下标。"""
        return int(np.argmin([abs(fee - lv) for lv in self.fee_levels]))
