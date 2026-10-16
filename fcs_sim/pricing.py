"""模块四：服务费动态调整。

规则：电压偏差超过阈值的节点，其充电站服务费上调一个档位；
电压质量良好、负荷较轻的节点，服务费下调一个档位；
服务费始终约束在 [rho1, rho4] = [1.4, 1.6] 元/kWh 区间内。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .config import Config


@dataclass
class FeeAdjustment:
    """单个站点的一次调整记录。"""

    station: int
    bus: int
    v_min: float
    action: str
    fee_before: float
    fee_after: float


def adjust_fees(
    cfg: Config,
    fees: Sequence[float],
    vm: np.ndarray,
    station_buses: Sequence[int],
    energies: Sequence[float],
) -> Tuple[List[float], List[FeeAdjustment]]:
    """按电压偏差调整各站服务费，并保持用户总充电费用不变（论文约束）。

    先按电压判据对各站做"上调/下调一个档位"的离散动作，再对服务费向量做一次
    整体平移，使加权平均服务费回到初始值 fee_init（总费用约束），最后夹在
    [rho1, rho4] 区间内。整体平移不改变站点之间的相对价差。
    """
    levels = list(cfg.fee_levels)
    min_f, max_f = levels[0], levels[-1]
    step = levels[1] - levels[0]
    new_fees = [float(f) for f in fees]
    target_fees = [float(f) for f in fees]
    details: List[FeeAdjustment] = []

    for s, bus in enumerate(station_buses):
        v_series = vm[:, bus]
        v_min = float(np.nanmin(v_series))
        action = "保持"

        if not np.isfinite(v_min):
            action = "保持"
        elif v_min < cfg.v_lower and fees[s] < max_f:
            action = "上调"
            target_fees[s] = fees[s] + step
        elif v_min > cfg.v_light and fees[s] > min_f:
            action = "下调"
            target_fees[s] = fees[s] - step

        details.append(
            FeeAdjustment(
                station=s,
                bus=bus,
                v_min=v_min,
                action=action,
                fee_before=float(fees[s]),
                fee_after=float(new_fees[s]),
            )
        )

    # 阻尼：朝目标档位走一步，避免 bang-bang 振荡
    new_fees = [f + cfg.fee_damping * (t - f) for f, t in zip(fees, target_fees)]

    energy = np.asarray(list(energies), dtype=float)
    if energy.sum() > 0:
        for _ in range(6):
            weighted_avg = float(np.dot(new_fees, energy) / energy.sum())
            shift = cfg.fee_init - weighted_avg
            if abs(shift) < 1e-4:
                break
            new_fees = list(np.clip(np.asarray(new_fees) + shift, min_f, max_f))
    new_fees = [round(f, 2) for f in new_fees]
    for detail, fee in zip(details, new_fees):
        detail.fee_after = fee
    return new_fees, details


def uniform_tou_fees(cfg: Config, fee: float) -> List[float]:
    """对照组辅助：全网统一定价（按分时电价聚类，不做分站定制）。"""
    return [fee for _ in range(cfg.n_stations)]


def adjust_fees_uniform(
    cfg: Config,
    fees: Sequence[float],
    vm: np.ndarray,
) -> List[float]:
    """对照组：全网统一定价策略。

    与定制策略使用同一套电压判据，但只依据整个配电网的最低电压统一升降，
    所有站点始终保持同一档位，因此无法把负荷从薄弱节点引导到健康节点。
    """
    levels = list(cfg.fee_levels)
    idx = cfg.fee_index(fees[0])
    v_min = float(np.nanmin(vm))
    if np.isfinite(v_min) and v_min < cfg.v_lower:
        idx = min(len(levels) - 1, idx + 1)
    elif np.isfinite(v_min) and v_min > cfg.v_light:
        idx = max(0, idx - 1)
    return [levels[idx]] * cfg.n_stations
