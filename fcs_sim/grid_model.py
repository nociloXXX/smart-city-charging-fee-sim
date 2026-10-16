"""模块三：配电网时序潮流计算（IEEE 33 节点）+ VDINI 指标。

基础负荷按母线类型叠加典型日曲线，充电负荷叠加到充电站所在母线，
逐时段调用 pandapower 潮流计算，得到 96 x 33 的电压矩阵。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandapower as pp
import warnings

from .config import Config


# IEEE 33（Baran & Wu）标准算例数据，直接写死在代码里，打包后不依赖 pandapower 的 JSON 数据文件。
# 32 个负荷节点（bus 1..32），bus 0 为电源点；37 条支路，其中 5 条联络开关常态断开。
_CASE33_LOADS = [
    (1, 0.1, 0.06), (2, 0.09, 0.04), (3, 0.12, 0.08), (4, 0.06, 0.03),
    (5, 0.06, 0.02), (6, 0.2, 0.1), (7, 0.2, 0.1), (8, 0.06, 0.02),
    (9, 0.06, 0.02), (10, 0.045, 0.03), (11, 0.06, 0.035), (12, 0.06, 0.035),
    (13, 0.12, 0.08), (14, 0.06, 0.01), (15, 0.06, 0.02), (16, 0.06, 0.02),
    (17, 0.09, 0.04), (18, 0.09, 0.04), (19, 0.09, 0.04), (20, 0.09, 0.04),
    (21, 0.09, 0.04), (22, 0.09, 0.05), (23, 0.42, 0.2), (24, 0.42, 0.2),
    (25, 0.06, 0.025), (26, 0.06, 0.025), (27, 0.06, 0.02), (28, 0.12, 0.07),
    (29, 0.2, 0.6), (30, 0.15, 0.07), (31, 0.21, 0.1), (32, 0.06, 0.04),
]
_CASE33_LINES = [
    (0, 1, 0.0922, 0.047), (1, 2, 0.493, 0.2511), (2, 3, 0.366, 0.1864),
    (3, 4, 0.3811, 0.1941), (4, 5, 0.819, 0.707), (5, 6, 0.1872, 0.6188),
    (6, 7, 0.7114, 0.2351), (7, 8, 1.03, 0.74), (8, 9, 1.044, 0.74),
    (9, 10, 0.1966, 0.065), (10, 11, 0.3744, 0.1238), (11, 12, 1.468, 1.155),
    (12, 13, 0.5416, 0.7129), (13, 14, 0.591, 0.526), (14, 15, 0.7463, 0.545),
    (15, 16, 1.289, 1.721), (16, 17, 0.732, 0.574), (1, 18, 0.164, 0.1565),
    (18, 19, 1.5042, 1.3554), (19, 20, 0.4095, 0.4784), (20, 21, 0.7089, 0.9373),
    (2, 22, 0.4512, 0.3083), (22, 23, 0.898, 0.7091), (23, 24, 0.896, 0.7011),
    (5, 25, 0.203, 0.1034), (25, 26, 0.2842, 0.1447), (26, 27, 1.059, 0.9337),
    (27, 28, 0.8042, 0.7006), (28, 29, 0.5075, 0.2585), (29, 30, 0.9744, 0.963),
    (30, 31, 0.3105, 0.3619), (31, 32, 0.341, 0.5302),
]
_CASE33_TIES = [
    (20, 7, 2.0, 2.0), (8, 14, 2.0, 2.0), (11, 21, 2.0, 2.0),
    (17, 32, 0.5, 0.5), (24, 28, 0.5, 0.5),
]


def build_case33bw():
    """构造 IEEE 33（Baran & Wu）配电网，等价于 pandapower 的 case33bw()。"""
    net = pp.create_empty_network(name="case33bw", sn_mva=10.0)
    for i in range(33):
        pp.create_bus(net, vn_kv=12.66, name=f"bus{i}")
    pp.create_ext_grid(net, bus=0, vm_pu=1.0, va_degree=0.0)
    for bus, p_mw, q_mvar in _CASE33_LOADS:
        pp.create_load(net, bus=bus, p_mw=p_mw, q_mvar=q_mvar, name=f"load{bus}")
    for from_bus, to_bus, r_ohm, x_ohm in _CASE33_LINES:
        pp.create_line_from_parameters(
            net, from_bus, to_bus, length_km=1.0, r_ohm_per_km=r_ohm,
            x_ohm_per_km=x_ohm, c_nf_per_km=0.0, max_i_ka=1000.0
        )
    for from_bus, to_bus, r_ohm, x_ohm in _CASE33_TIES:
        pp.create_line_from_parameters(
            net, from_bus, to_bus, length_km=1.0, r_ohm_per_km=r_ohm,
            x_ohm_per_km=x_ohm, c_nf_per_km=0.0, max_i_ka=1000.0, in_service=False
        )
    return net


class DistributionGrid:
    """IEEE 33 节点辐射状配电网 + 充电负荷接口。"""

    def __init__(
        self,
        cfg: Config,
        station_road_degree: Optional[Sequence[int]] = None,
    ) -> None:
        self.cfg = cfg
        self.n_slots = cfg.n_slots
        self.net = build_case33bw()
        self.n_bus = len(self.net.bus)

        # 原始负荷（每母线一条），先记录下来才能逐时段改写
        self.base_rows = list(self.net.load.index)
        self.base_buses = self.net.load.bus.astype(int).to_numpy()
        self.base_p0 = self.net.load.p_mw.to_numpy(dtype=float)
        self.base_q0 = self.net.load.q_mvar.to_numpy(dtype=float)

        self.res_profile, self.com_profile = self._daily_profiles()
        self.base_p, self.base_q = self._base_load_matrix()

        # 充电站 -> 母线，追加 8 条可控负荷
        self.station_buses = list(cfg.station_buses)
        n_base_loads = len(self.base_rows)
        for s, bus in enumerate(self.station_buses):
            pp.create_load(self.net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"EV_station_{s + 1}")
        self.ev_rows = list(range(n_base_loads, len(self.net.load)))

        self.bus_names = {int(b): f"bus{b}" for b in self.net.bus.index}
        self.road_degree = list(station_road_degree) if station_road_degree else None
        self.criticality = self.criticality_weights(cfg.criticality_mode)

    # ---------------- 基础负荷 ----------------
    def _daily_profiles(self) -> Tuple[np.ndarray, np.ndarray]:
        """居民 / 商业典型日负荷曲线（峰值归一化到 1）。"""
        h = np.arange(self.n_slots) * self.cfg.slot_min / 60.0

        def gauss(mu: float, sigma: float) -> np.ndarray:
            return np.exp(-0.5 * ((h - mu) / sigma) ** 2)

        residential = 0.30 + 0.45 * gauss(7.8, 1.6) + 0.80 * gauss(19.5, 2.2)
        commercial = 0.25 + 0.75 * gauss(13.0, 3.0) + 0.35 * gauss(10.0, 2.0) + 0.30 * gauss(20.0, 2.5)
        return residential / residential.max(), commercial / commercial.max()

    def _base_load_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        P = np.zeros((self.n_slots, self.n_bus))
        Q = np.zeros_like(P)
        for bus, p0, q0 in zip(self.base_buses, self.base_p0, self.base_q0):
            profile = self.com_profile if bus % 3 == 0 else self.res_profile
            P[:, bus] += p0 * cfg.base_load_scale * profile
            Q[:, bus] += q0 * cfg.base_load_scale * profile
        return P, Q

    # ---------------- 节点关键度 ----------------
    def _grid_degree(self) -> np.ndarray:
        """配电网络拓扑节点度数：该节点连接的线路数。"""
        deg = np.zeros(self.n_bus)
        in_service = self.net.line["in_service"].to_numpy()
        for arr in (self.net.line["from_bus"].to_numpy()[in_service],
                    self.net.line["to_bus"].to_numpy()[in_service]):
            for bus in arr:
                deg[int(bus)] += 1
        return deg

    def criticality_weights(self, mode: str = "degree") -> np.ndarray:
        """节点关键度：节点度数 / 负荷占比 / 二者混合（论文公式 N_i）。"""
        w = np.zeros(self.n_bus)
        if mode in ("load_share", "mixed"):
            share = self.base_p.mean(axis=0)
            if share.sum() > 0:
                w += share / share.sum()
        if mode in ("degree", "mixed"):
            deg = self._grid_degree()
            if deg.sum() > 0:
                w = 0.5 * w + 0.5 * deg / deg.sum() if mode == "mixed" else deg / deg.sum()
        if w.sum() <= 0:
            w = np.ones(self.n_bus) / self.n_bus
        return w / w.sum()

    # ---------------- 潮流计算 ----------------
    def run_profile(self, charging_mw: np.ndarray) -> Dict[str, np.ndarray]:
        """逐时段潮流：返回电压矩阵、网损、收敛标志。"""
        cfg = self.cfg
        assert charging_mw.shape == (self.n_slots, len(self.station_buses))
        P = self.base_p.copy()
        P[:, self.station_buses] += charging_mw

        vm = np.full((self.n_slots, self.n_bus), np.nan)
        losses = np.zeros(self.n_slots)
        converged = np.zeros(self.n_slots, dtype=bool)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for t in range(self.n_slots):
                self.net.load.loc[self.base_rows, "p_mw"] = P[t, self.base_buses]
                self.net.load.loc[self.base_rows, "q_mvar"] = self.base_q[t, self.base_buses]
                self.net.load.loc[self.ev_rows, "p_mw"] = charging_mw[t]
                self.net.load.loc[self.ev_rows, "q_mvar"] = 0.0
                try:
                    pp.runpp(self.net, numba=False)
                    vm[t] = self.net.res_bus.vm_pu.to_numpy()
                    losses[t] = float(self.net.res_line.pl_mw.sum())
                    converged[t] = True
                except Exception:  # 严重越限时潮流可能不收敛
                    continue
        return {"vm_pu": vm, "loss_mw": losses, "converged": converged}

    # ---------------- VDINI ----------------
    def vdini_series(self, vm: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """VDINI(t) = sum_i N_i * dV_i，电压越带时偏差乘惩罚系数 K_dV。"""
        w = self.criticality if weights is None else weights
        vm = np.nan_to_num(vm, nan=1.0)
        dev = np.abs(vm - 1.0)
        in_band = (vm >= self.cfg.v_lower) & (vm <= self.cfg.v_upper)
        dev = np.where(in_band, dev, dev * self.cfg.delta_v_penalty)
        return dev @ w

    def summary(self, vm: np.ndarray) -> Dict[str, float]:
        vd = self.vdini_series(vm)
        return {
            "vdini_daily_sum": float(vd.sum() * self.cfg.slot_hours),
            "vdini_peak": float(vd.max()),
            "v_min": float(np.nanmin(vm)),
            "violation_slots": int((np.nanmin(vm, axis=1) < self.cfg.v_lower).sum()),
        }
