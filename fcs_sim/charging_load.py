"""模块二：充电负荷预测。

把模块一的充电决策叠加到 96 x N_stations 的功率矩阵上：
    充电时长 = 所需电量 / 快充功率
每座站按"桩数 + 先到先服务"排队，车辆到达时若无空闲桩则等待。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from .config import Config
from .traffic_sim import ChargeDecision


@dataclass
class ChargeLoadResult:
    """模块二输出。"""

    power_kw: np.ndarray                     # 96 x N_stations，各站各时段充电功率
    station_energy_kwh: np.ndarray           # 各站日充电量（电网侧，按 80 kW x 30 min 计）
    station_charge_energy_kwh: np.ndarray    # 各站日充电量（电池侧，按补充的 SOC 计）
    station_peak_kw: np.ndarray              # 各站峰值功率
    station_utilization: np.ndarray          # 各站桩利用率（充电量 / 桩容量）
    wait_by_station: np.ndarray              # 各站平均排队时长（分钟）
    avg_wait_min: float
    max_wait_min: float
    total_events: int
    balked: int                              # 到站过晚、当天无法完成充电的车辆数
    unserved_kwh: float = 0.0                # 跨日截断未计入的充电量

    @property
    def total_energy_kwh(self) -> float:
        return float(self.station_energy_kwh.sum())


def simulate_charging_load(
    cfg: Config,
    decisions: Sequence[ChargeDecision],
    n_stations: int,
) -> ChargeLoadResult:
    """按站排队并把充电功率叠加到时间轴上。"""
    slots = cfg.n_slots
    power = np.zeros((slots, n_stations))
    charge_energy_by_station = np.zeros(n_stations)
    busy_until = np.full((n_stations, cfg.piles_per_station), -1.0)  # 各桩空闲时刻（分钟）
    waits = np.zeros(len(decisions))
    balked = 0
    served = 0
    unserved = 0.0
    day_end = slots * cfg.slot_min

    # 需求守恒：所有触发充电需求的车辆都会完成充电（可能排队等待），
    # 等待时长通过迭代反馈影响下一轮的站点选择。

    order = sorted(range(len(decisions)), key=lambda k: decisions[k].arrive_hour)
    for pos, k in enumerate(order):
        dec = decisions[k]
        arrival = dec.arrive_hour * 60.0
        duration = cfg.charge_duration_min
        if arrival >= day_end:
            balked += 1
            continue

        piles = busy_until[dec.station]
        idx = int(np.argmin(piles))                # 最早空闲的桩
        start = max(arrival, float(piles[idx]))
        if start + duration > day_end:             # 当天充不完，按可充部分计入
            duration = max(0.0, day_end - start)
            balked += 1
            unserved += dec.energy_kwh * (cfg.charge_duration_min - duration) / cfg.charge_duration_min
        served += 1
        charge_energy_by_station[dec.station] += dec.energy_kwh
        piles[idx] = start + duration

        dec.wait_min = start - arrival
        dec.start_min = start
        dec.duration_min = duration
        waits[pos] = dec.wait_min

        # 把功率摊到各时段的边界上
        remain, cur = duration, start
        while remain > 1e-9 and cur < day_end - 1e-9:
            slot = int(cur // cfg.slot_min)
            slot_end = (slot + 1) * cfg.slot_min
            seg = min(remain, slot_end - cur)
            power[slot, dec.station] += cfg.pile_power_kw * seg / cfg.slot_min
            remain -= seg
            cur += seg

    station_energy = power.sum(axis=0) * cfg.slot_hours
    station_peak = power.max(axis=0)
    capacity = cfg.piles_per_station * cfg.pile_power_kw * 24.0
    utilization = station_energy / capacity

    wait_by_station = np.zeros(n_stations)
    for s in range(n_stations):
        w = [waits[i] for i, d in enumerate(decisions) if d.station == s]
        wait_by_station[s] = float(np.mean(w)) if w else 0.0

    return ChargeLoadResult(
        power_kw=power,
        station_energy_kwh=station_energy,
        station_charge_energy_kwh=charge_energy_by_station,
        station_peak_kw=station_peak,
        station_utilization=utilization,
        wait_by_station=wait_by_station,
        avg_wait_min=float(waits.mean()) if len(waits) else 0.0,
        max_wait_min=float(waits.max()) if len(waits) else 0.0,
        total_events=served,
        balked=balked,
        unserved_kwh=float(unserved),
    )
