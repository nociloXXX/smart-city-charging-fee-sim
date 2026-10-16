"""模块一：路网交通仿真。

生成电动汽车出行需求（出发地、目的地、出发时刻），按出行链行驶，
当 SOC 低于阈值时按"综合充电成本最低"选择快充站，输出每辆车的充电决策。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import Config
from .road_network import RoadNetwork

# 出行链定义：家 -> ... -> 家
CHAINS: Dict[str, Tuple[str, ...]] = {
    "H_W_M_H": ("home", "work", "mall", "home"),
    "H_M_H": ("home", "mall", "home"),
    "H_W_H": ("home", "work", "home"),
}


@dataclass
class EV:
    """一辆电动汽车的出行属性。"""

    idx: int
    battery_kwh: float
    consumption: float        # kWh/km
    init_soc: float
    home: int
    work: int
    mall: int
    chain: str
    depart_hour: float        # 从家出发时刻（小时）
    public_user: bool = True  # 是否依赖公共快充（其余车辆假设有私人桩）


@dataclass
class ChargeDecision:
    """一次充电决策记录。"""

    ev: int
    station: int
    arrive_hour: float        # 到达充电站时刻（小时）
    energy_kwh: float         # 充电量
    soc_before: float
    soc_after: float
    detour_km: float
    detour_min: float
    cost: float               # 综合充电成本（元）
    leg: int
    origin: int
    dest: int
    wait_min: float = 0.0     # 排队等待时长，由模块二回填
    start_min: float = 0.0    # 开始充电时刻（分钟），由模块二回填
    duration_min: float = 0.0 # 充电时长（分钟），由模块二回填


def build_regions(cfg: Config, net: RoadNetwork) -> Dict[str, Tuple[int, ...]]:
    """划分居住区 / 工作区（CBD）/ 商业区（商场）节点。"""
    rows, cols = net.rows, net.cols
    center = [(i, j) for i in range(rows // 2 - 1, rows // 2 + 1) for j in range(cols // 2 - 1, cols // 2 + 1)]
    malls = [
        (max(0, rows // 2 - 3), max(0, cols // 2 + 2)),
        (min(rows - 1, rows // 2 + 3), max(0, cols // 2 - 2)),
        (max(0, rows // 2 - 3), max(0, cols // 2 - 3)),
        (min(rows - 1, rows // 2 + 2), min(cols - 1, cols // 2 + 3)),
    ]
    return {
        "home": tuple(range(net.n_nodes)),                       # 居住区：全区域
        "work": tuple(net.node(i, j) for i, j in center),        # 工作地：中心 2x2
        "mall": tuple(net.node(i, j) for i, j in malls),         # 商场：4 个外围商业节点
    }


def generate_evs(cfg: Config, net: RoadNetwork, rng: np.random.Generator) -> List[EV]:
    """生成 N 辆车的出行需求：初始电量、OD、出行链、出发时刻。"""
    regions = build_regions(cfg, net)
    chains = list(CHAINS.keys())
    weights = np.array(cfg.chain_weights, dtype=float)
    weights = weights / weights.sum()
    evs: List[EV] = []
    for idx in range(cfg.n_ev):
        chain = chains[int(rng.choice(len(chains), p=weights))]
        depart = float(rng.normal(cfg.home_depart_mean, cfg.home_depart_sd))
        public_user = bool(rng.random() < cfg.public_share)
        evs.append(
            EV(
                idx=idx,
                battery_kwh=float(rng.uniform(*cfg.battery_range)),
                consumption=float(rng.uniform(*cfg.consumption_range)),
                init_soc=float(rng.uniform(*cfg.init_soc_range)),
                home=int(rng.choice(regions["home"])),
                work=int(rng.choice(regions["work"])),
                mall=int(rng.choice(regions["mall"])),
                chain=chain,
                depart_hour=float(np.clip(depart, 5.0, 11.0)),
                public_user=public_user,
            )
        )
    return evs


def _station_cost(
    cfg: Config,
    net: RoadNetwork,
    ev: EV,
    soc: float,
    hour: float,
    origin: int,
    dest: int,
    station: int,
    fee: float,
    price: float,
    wait_min: float,
) -> Tuple[float, float, float]:
    """计算前往某充电站的综合充电成本（元）。

    综合充电成本 = 服务费 x 充电量 + 绕路时间成本 + 绕路耗电成本（+ 充电电费，可开关）
    """
    direct = net.distance(origin, dest)
    via = net.distance(origin, station) + net.distance(station, dest)
    detour_km = max(0.0, via - direct)
    direct_t = net.travel_time_min(origin, dest, hour)
    via_t = net.travel_time_min(origin, station, hour) + net.travel_time_min(station, dest, hour)
    detour_min = max(0.0, via_t - direct_t)

    energy_kwh = cfg.charge_soc_gain * ev.battery_kwh
    detour_kwh = detour_km * ev.consumption

    cost = fee * energy_kwh
    cost += cfg.time_value / 60.0 * (detour_min + wait_min)
    cost += price * detour_kwh
    if cfg.include_energy_price:
        cost += price * energy_kwh
    return cost, detour_km, detour_min


def choose_station(
    cfg: Config,
    net: RoadNetwork,
    ev: EV,
    soc: float,
    hour: float,
    origin: int,
    dest: int,
    stations: Sequence[int],
    fees: Sequence[float],
    waits: Sequence[float],
    price: float,
    rng: np.random.Generator,
) -> Tuple[int, float, float, float]:
    """按最小综合成本选择充电站；random_choice=True 时使用 Logit 随机效用。"""
    costs, detours, detour_ts = [], [], []
    for s_idx, station in enumerate(stations):
        c, d_km, d_min = _station_cost(
            cfg, net, ev, soc, hour, origin, dest, station, fees[s_idx], price, waits[s_idx]
        )
        costs.append(c)
        detours.append(d_km)
        detour_ts.append(d_min)

    if cfg.random_choice:
        c = np.asarray(costs)
        p = np.exp(-cfg.logit_theta * (c - c.min()))
        p = p / p.sum()
        s_idx = int(rng.choice(len(costs), p=p))
    else:
        s_idx = int(np.argmin(costs))
    return s_idx, costs[s_idx], detours[s_idx], detour_ts[s_idx]


def simulate_traffic(
    cfg: Config,
    net: RoadNetwork,
    evs: Sequence[EV],
    stations: Sequence[int],
    fees: Sequence[float],
    waits: Sequence[float],
    price: np.ndarray,
    rng: np.random.Generator,
) -> List[ChargeDecision]:
    """按出行链逐段行驶，触发充电需求时选站并记录充电决策。"""
    slot_min = cfg.slot_min
    decisions: List[ChargeDecision] = []

    for ev in evs:
        place_node = {"home": ev.home, "work": ev.work, "mall": ev.mall}
        seq = CHAINS[ev.chain]
        soc = ev.init_soc
        t = ev.depart_hour

        for k in range(len(seq) - 1):
            origin, dest = place_node[seq[k]], place_node[seq[k + 1]]
            dist = net.distance(origin, dest)
            tt = net.travel_time_min(origin, dest, t)
            soc -= dist * ev.consumption / ev.battery_kwh
            t_arr = t + tt / 60.0
            legs_left = len(seq) - 2 - k

            # 里程焦虑：SOC 低于阈值，或剩余电量不足以完成后续行程
            remaining_km = sum(
                net.distance(place_node[seq[k + j]], place_node[seq[k + j + 1]])
                for j in range(legs_left)
            )
            insufficient = soc * ev.battery_kwh < 1.1 * remaining_km * ev.consumption
            if ev.public_user and legs_left > 0 and (soc < cfg.soc_trigger or insufficient):
                slot = int(np.clip(t_arr * 60.0 // slot_min, 0, cfg.n_slots - 1))
                s_idx, cost, detour_km, detour_min = choose_station(
                    cfg, net, ev, soc, t_arr, origin, dest, stations, fees, waits, price[slot], rng
                )
                energy = cfg.charge_soc_gain * ev.battery_kwh
                arrive = t_arr + detour_min / 60.0
                decisions.append(
                    ChargeDecision(
                        ev=ev.idx,
                        station=s_idx,
                        arrive_hour=arrive,
                        energy_kwh=energy,
                        soc_before=soc,
                        soc_after=min(1.0, soc + cfg.charge_soc_gain),
                        detour_km=detour_km,
                        detour_min=detour_min,
                        cost=cost,
                        leg=k + 1,
                        origin=origin,
                        dest=dest,
                    )
                )
                # 绕路耗电 + 充电补能
                soc -= detour_km * ev.consumption / ev.battery_kwh
                soc = min(1.0, soc + cfg.charge_soc_gain)
                # 继续前往目的地
                charge_min = cfg.charge_duration_min
                t_arr = arrive + (waits[s_idx] + charge_min + 10.0) / 60.0
                t_arr += net.travel_time_min(stations[s_idx], dest, t_arr) / 60.0
                t_arr += 10.0 / 60.0

            # 下一段旅程前的停留
            nxt = seq[k + 1]
            if nxt == "work":
                t_arr += cfg.work_dwell
            elif nxt == "mall":
                t_arr += cfg.mall_dwell
            t = t_arr

    return decisions
