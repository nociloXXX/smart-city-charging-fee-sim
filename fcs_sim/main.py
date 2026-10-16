"""主仿真脚本：串联模块一~模块四的闭环。

流程：
    1) 初始化服务费为固定值（1.6 元/kWh）；
    2) 循环（最多 K 次）：
       a. 模块一 交通仿真 -> 车辆充电决策
       b. 模块二 负荷预测 -> 充电负荷矩阵
       c. 模块三 潮流计算 -> 节点电压与 VDINI
       d. VDINI 相对变化小于阈值则跳出
       e. 否则模块四 调整服务费，进入下一轮
    3) 输出最终服务费向量、充电负荷矩阵、电压矩阵与 VDINI 收敛曲线
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from . import report
from .charging_load import ChargeLoadResult, simulate_charging_load
from .config import Config, build_config, make_fee_levels, tou_price
from .grid_model import DistributionGrid
from .pricing import adjust_fees, adjust_fees_uniform
from .road_network import RoadNetwork
from .traffic_sim import ChargeDecision, generate_evs, simulate_traffic


def _default_outdir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "results"
    return Path(__file__).resolve().parents[1] / "results"


@dataclass
class ScenarioResult:
    """一次完整评估（某一组服务费方案）的结果。"""

    name: str
    fees: List[float]
    decisions: List[ChargeDecision]
    load: ChargeLoadResult
    vm: np.ndarray                 # 96 x 33 电压矩阵
    loss_mw: np.ndarray
    vdini: np.ndarray              # 96 个时段的 VDINI
    converged: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)


def evaluate(
    cfg: Config,
    net: RoadNetwork,
    grid: DistributionGrid,
    evs,
    stations: Sequence[int],
    fees: Sequence[float],
    price: np.ndarray,
    waits: Sequence[float],
    name: str,
) -> ScenarioResult:
    """跑一次"模块一 -> 模块二 -> 模块三"。"""
    rng = np.random.default_rng(cfg.seed)   # 每轮用同一随机种子，结果只随服务费变化
    decisions = simulate_traffic(cfg, net, evs, stations, fees, waits, price, rng)
    load = simulate_charging_load(cfg, decisions, cfg.n_stations)
    res = grid.run_profile(load.power_kw / 1000.0)
    vm = res["vm_pu"]
    vdini = grid.vdini_series(vm)
    vmin_slots = np.nanmin(vm, axis=1)

    metrics = {
        "vdini_daily": float(np.nansum(vdini) * cfg.slot_hours),
        "vdini_peak": float(np.nanmax(vdini)),
        "v_min": float(np.nanmin(vm)),
        "violation_slots": int(np.nansum(vmin_slots < cfg.v_lower)),
        "ev_events": int(len(decisions)),
        "balked": int(load.balked),
        "charge_energy_kwh": float(load.station_charge_energy_kwh.sum()),
        "grid_energy_kwh": float(load.total_energy_kwh),
        "peak_load_kw": float(load.power_kw.sum(axis=1).max()),
        "avg_wait_min": float(load.avg_wait_min),
        "max_wait_min": float(load.max_wait_min),
        "loss_mwh": float(np.nansum(res["loss_mw"]) * cfg.slot_hours),
        "user_cost": float(sum(d.cost for d in decisions)),
        "mean_fee": float(np.mean(fees)),
    }
    charge_energy = load.station_charge_energy_kwh
    if charge_energy.sum() > 0:
        metrics["weighted_mean_fee"] = float(np.dot(np.asarray(fees, dtype=float), charge_energy) / charge_energy.sum())
    else:
        metrics["weighted_mean_fee"] = float(np.mean(fees))
    return ScenarioResult(
        name=name,
        fees=[float(f) for f in fees],
        decisions=decisions,
        load=load,
        vm=vm,
        loss_mw=res["loss_mw"],
        vdini=vdini,
        converged=res["converged"],
        metrics=metrics,
    )


def run_simulation(cfg: Config | None = None, verbose: bool = True) -> Dict:
    cfg = cfg or Config()
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)

    net = RoadNetwork(cfg.rows, cfg.cols, cfg.spacing_km, cfg.free_speed_kmh, cfg.peak_speed_factor)
    stations = [net.node(i, j) for (i, j, _) in cfg.station_layout]
    road_degrees = [net.degree(n) for n in stations]
    grid = DistributionGrid(cfg, station_road_degree=road_degrees)
    price = tou_price(cfg.n_slots, cfg.slot_min)
    evs = generate_evs(cfg, net, rng)

    if verbose:
        print(f"[初始化] 路网 {cfg.rows}x{cfg.cols}，EV {len(evs)} 辆，充电站 {cfg.n_stations} 座，"
              f"单站 {cfg.piles_per_station} 桩 x {cfg.pile_power_kw:.0f} kW")

    zero_waits = np.zeros(cfg.n_stations)

    # ---- 对照组一：固定服务费，不做任何动态调整 ----
    baseline = evaluate(cfg, net, grid, evs, stations, [cfg.fee_init] * cfg.n_stations,
                        price, zero_waits, "固定服务费")
    if verbose:
        print(f"[对照组] 固定服务费 {cfg.fee_init:.2f} 元/kWh  "
              f"VDINI={baseline.metrics['vdini_daily']:.4f}  v_min={baseline.metrics['v_min']:.4f}  "
              f"越限时段={baseline.metrics['violation_slots']}")

    # ---- 对照组二：全网统一定价（同一套电压判据，但所有站点同档升降） ----
    uni_fees = [cfg.fee_init] * cfg.n_stations
    uni_waits = np.zeros(cfg.n_stations)
    uniform: ScenarioResult | None = None
    for it in range(1, cfg.max_iter + 1):
        uniform = evaluate(cfg, net, grid, evs, stations, uni_fees, price, uni_waits,
                           f"统一{it}")
        new_uni = adjust_fees_uniform(cfg, uni_fees, uniform.vm)
        if verbose:
            print(f"  [统一调整] 第 {it} 轮: 全网 {new_uni[0]:.2f} 元/kWh  "
                  f"VDINI={uniform.metrics['vdini_daily']:.4f}  v_min={uniform.metrics['v_min']:.4f}")
        if np.allclose(new_uni, uni_fees):
            break
        uni_fees = new_uni
        uni_waits = cfg.wait_smoothing * uni_waits + (1 - cfg.wait_smoothing) * uniform.load.wait_by_station
    assert uniform is not None

    # ---- 定制策略：服务费随电压偏差动态调整 ----
    fees = [cfg.fee_init] * cfg.n_stations
    waits = np.zeros(cfg.n_stations)
    history: List[Dict] = []
    proposed: ScenarioResult | None = None
    best: ScenarioResult | None = None
    prev_vdini = None

    for it in range(1, cfg.max_iter + 1):
        proposed = evaluate(cfg, net, grid, evs, stations, fees, price, waits, f"迭代{it}")
        if best is None or proposed.metrics["vdini_daily"] < best.metrics["vdini_daily"]:
            best = proposed
        history.append({"iter": it, **{k: v for k, v in proposed.metrics.items()},
                        **{f"fee_s{i + 1}": fees[i] for i in range(cfg.n_stations)}})
        if verbose:
            print(f"  第 {it} 轮: 服务费={np.round(fees, 3).tolist()}  "
                  f"VDINI={proposed.metrics['vdini_daily']:.4f}  v_min={proposed.metrics['v_min']:.4f}  "
                  f"充电量={proposed.metrics['charge_energy_kwh'] / 1000:.2f} MWh")

        changed = abs(proposed.metrics["vdini_daily"] - prev_vdini) / prev_vdini if prev_vdini else 1.0
        if prev_vdini is not None and changed < cfg.vdini_tol:
            if verbose:
                print(f"  -> VDINI 相对变化 {changed:.5f} < {cfg.vdini_tol}，收敛退出")
            break
        prev_vdini = proposed.metrics["vdini_daily"]

        new_fees, details = adjust_fees(
            cfg, fees, proposed.vm, cfg.station_buses, proposed.load.station_charge_energy_kwh
        )
        if verbose:
            acts = " ".join(f"S{d.station + 1}:{d.action}" for d in details)
            print(f"     模块四调整: {acts}")
        waits = cfg.wait_smoothing * waits + (1 - cfg.wait_smoothing) * proposed.load.wait_by_station
        if np.allclose(new_fees, fees):
            if verbose:
                print("  -> 服务费向量不再变化，收敛退出")
            fees = new_fees
            break
        fees = new_fees

    assert proposed is not None and best is not None
    proposed = best

    results = {
        "cfg": cfg,
        "net": net,
        "stations": stations,
        "road_degrees": road_degrees,
        "price": price,
        "evs": evs,
        "history": history,
        "proposed": proposed,
        "baseline": baseline,
        "uniform": uniform,
        "elapsed": time.time() - t0,
    }
    if verbose:
        print(f"[完成] 总耗时 {results['elapsed']:.1f} s")
    return results


def run_with(params: Dict, outdir: Path, plot: bool = True) -> Path:
    """由参数字典构建配置、运行仿真并把结果写入 outdir。"""
    cfg = build_config(**params)
    outdir.mkdir(parents=True, exist_ok=True)
    results = run_simulation(cfg)
    if plot:
        report.write_reports(results, outdir)
    return outdir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="快充负荷服务费定制策略仿真")
    parser.add_argument("--out", type=str, default=str(_default_outdir()),
                        help="结果输出目录")
    parser.add_argument("--n-ev", type=int, default=None, help="电动汽车数量")
    parser.add_argument("--public-share", type=float, default=None, help="公共快充比例")
    parser.add_argument("--battery", type=float, default=None, help="电池容量 kWh")
    parser.add_argument("--soc-trigger", type=float, default=None, help="里程焦虑 SOC 阈值")
    parser.add_argument("--pile-power", type=float, default=None, help="单桩功率 kW")
    parser.add_argument("--piles", type=int, default=None, help="每站桩数")
    parser.add_argument("--fee-min", type=float, default=None, help="服务费下限 元/kWh")
    parser.add_argument("--fee-max", type=float, default=None, help="服务费上限 元/kWh")
    parser.add_argument("--base-scale", type=float, default=None, help="基础负荷缩放")
    parser.add_argument("--v-lower", type=float, default=None, help="电压下限 p.u.")
    parser.add_argument("--max-iter", type=int, default=None, help="最大迭代轮数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--no-plots", action="store_true", help="只输出数据，不出图")
    args = parser.parse_args(argv)

    params: Dict = {}
    if args.fee_min is not None and args.fee_max is not None:
        params["fee_levels"] = make_fee_levels(args.fee_min, args.fee_max)
        params["fee_init"] = round((args.fee_min + args.fee_max) / 2, 2)
    flag_map = {
        "n_ev": "n_ev",
        "public_share": "public_share",
        "soc_trigger": "soc_trigger",
        "pile_power_kw": "pile_power",
        "piles_per_station": "piles",
        "base_load_scale": "base_scale",
        "v_lower": "v_lower",
        "max_iter": "max_iter",
        "seed": "seed",
    }
    for key, attr in flag_map.items():
        value = getattr(args, attr, None)
        if value is not None:
            params[key] = value
    if args.battery is not None:
        params["battery_range"] = (args.battery, args.battery)

    outdir = Path(args.out)
    run_with(params, outdir, plot=not args.no_plots)
    if not args.no_plots:
        print(f"[输出] CSV 与图件已写入 {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
