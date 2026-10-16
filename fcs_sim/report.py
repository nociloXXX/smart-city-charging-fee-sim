"""结果后处理：CSV 数据表与图件。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _setup() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["legend.frameon"] = False


def _hours(cfg, n_slots: int | None = None) -> np.ndarray:
    n = n_slots or cfg.n_slots
    return np.arange(n) * cfg.slot_min / 60.0


def _write_csv(path: Path, header: List[str], rows: List[List]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------------------------------------------- 数据表
def save_tables(results: Dict, outdir: Path) -> None:
    cfg = results["cfg"]
    proposed = results["proposed"]
    baseline = results["baseline"]
    uniform = results["uniform"]
    hours = _hours(cfg)

    # 1) 迭代收敛过程
    cols = ["iter", "mean_fee", "vdini_daily", "vdini_peak", "v_min", "violation_slots",
            "ev_events", "balked", "charge_energy_kwh", "grid_energy_kwh", "peak_load_kw",
            "avg_wait_min", "max_wait_min",
            "loss_mwh", "user_cost"] + [f"fee_s{i + 1}" for i in range(cfg.n_stations)]
    rows = [[h.get(c, "") for c in cols] for h in results["history"]]
    _write_csv(outdir / "convergence_history.csv", cols, rows)

    # 2) 充电负荷矩阵（定制策略 / 固定服务费）
    header = ["hour"] + [f"station{i + 1}_kw" for i in range(cfg.n_stations)] + ["total_kw"]
    for tag, res in (("proposed", proposed), ("baseline", baseline)):
        rows = []
        for t, hr in enumerate(hours):
            row = [round(float(hr), 3)] + [round(float(v), 3) for v in res.load.power_kw[t]]
            row.append(round(float(res.load.power_kw[t].sum()), 3))
            rows.append(row)
        _write_csv(outdir / f"charging_load_{tag}.csv", header, rows)

    # 3) 电压矩阵（定制策略）
    rows = []
    for t, hr in enumerate(hours):
        rows.append([round(float(hr), 3)] + [round(float(v), 5) for v in proposed.vm[t]])
    _write_csv(outdir / "voltage_matrix_proposed.csv",
               ["hour"] + [f"bus{b + 1}" for b in range(proposed.vm.shape[1])], rows)

    # 4) 站点汇总
    cols = ["station", "road_node", "grid_bus", "fee_proposed", "fee_baseline",
            "energy_kwh_proposed", "energy_kwh_baseline", "peak_kw_proposed", "peak_kw_baseline",
            "utilization_proposed", "utilization_baseline", "wait_min_proposed", "wait_min_baseline",
            "vmin_proposed", "vmin_baseline"]
    rows = []
    for s in range(cfg.n_stations):
        bus = cfg.station_buses[s]
        rows.append([
            s + 1, results["stations"][s], bus,
            proposed.fees[s], baseline.fees[s],
            round(float(proposed.load.station_energy_kwh[s]), 2),
            round(float(baseline.load.station_energy_kwh[s]), 2),
            round(float(proposed.load.station_peak_kw[s]), 2),
            round(float(baseline.load.station_peak_kw[s]), 2),
            round(float(proposed.load.station_utilization[s]), 4),
            round(float(baseline.load.station_utilization[s]), 4),
            round(float(proposed.load.wait_by_station[s]), 2),
            round(float(baseline.load.wait_by_station[s]), 2),
            round(float(np.nanmin(proposed.vm[:, bus])), 5),
            round(float(np.nanmin(baseline.vm[:, bus])), 5),
        ])
    _write_csv(outdir / "station_summary.csv", cols, rows)

    # 5) 全天 VDINI 曲线
    rows = [[round(float(hr), 3),
             round(float(proposed.vdini[t]), 6),
             round(float(baseline.vdini[t]), 6),
             round(float(uniform.vdini[t]), 6)]
            for t, hr in enumerate(hours)]
    _write_csv(outdir / "vdini_series.csv", ["hour", "proposed", "baseline", "uniform"], rows)

    # 6) 情景对比总表
    keys = ["vdini_daily", "vdini_peak", "v_min", "violation_slots", "ev_events", "balked",
            "charge_energy_kwh", "grid_energy_kwh", "peak_load_kw", "avg_wait_min", "max_wait_min", "loss_mwh",
            "user_cost", "weighted_mean_fee", "mean_fee"]
    rows = [[k, proposed.metrics[k], baseline.metrics[k], uniform.metrics[k]] for k in keys]
    _write_csv(outdir / "scenario_compare.csv", ["指标", "定制策略", "固定服务费", "统一调整"], rows)

    # 7) 充电决策样例
    rows = [[d.ev, d.station + 1, round(d.arrive_hour, 3), round(d.energy_kwh, 2),
             round(d.soc_before, 3), round(d.detour_km, 2), round(d.detour_min, 2),
             round(d.wait_min, 2), round(d.cost, 2)]
            for d in proposed.decisions[:500]]
    _write_csv(outdir / "charging_decisions_sample.csv",
               ["ev_id", "station", "arrive_hour", "energy_kwh", "soc_before",
                "detour_km", "detour_min", "wait_min", "cost_yuan"], rows)


# ---------------------------------------------------------------- 图件
def plot_convergence(results: Dict, outdir: Path) -> None:
    cfg = results["cfg"]
    hist = results["history"]
    base = results["baseline"].metrics
    iters = [h["iter"] for h in hist]
    vdini = [h["vdini_daily"] for h in hist]
    vmin = [h["v_min"] for h in hist]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(iters, vdini, "o-", color="#1f77b4", label="定制策略 VDINI")
    ax.axhline(base["vdini_daily"], ls="--", color="#d62728", label="固定服务费 VDINI")
    ax.set_xlabel("迭代轮次")
    ax.set_ylabel("VDINI（全天累计）")
    ax.set_title("VDINI 收敛过程")
    ax2 = ax.twinx()
    ax2.plot(iters, vmin, "s--", color="#2ca02c", label="最低电压")
    ax2.axhline(cfg.v_lower, ls=":", color="gray")
    ax2.set_ylabel("全网最低电压 / p.u.")
    ax.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig1_convergence.png")
    plt.close(fig)


def plot_fees(results: Dict, outdir: Path) -> None:
    cfg = results["cfg"]
    hist = results["history"]
    iters = [h["iter"] for h in hist]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})
    for s in range(cfg.n_stations):
        ax.plot(iters, [h[f"fee_s{s + 1}"] for h in hist], "o-", ms=4, label=f"站{s + 1}")
    ax.set_xlabel("迭代轮次")
    ax.set_ylabel("服务费 / (元/kWh)")
    ax.set_title("各站服务费调整轨迹")
    ax.legend(ncol=2, fontsize=8)

    final = results["proposed"].fees
    ax2.bar([f"S{s + 1}" for s in range(cfg.n_stations)], final, color="#4c72b0")
    ax2.axhline(cfg.fee_levels[0], ls=":", color="gray")
    ax2.axhline(cfg.fee_levels[-1], ls=":", color="gray")
    ax2.set_ylabel("最终服务费 / (元/kWh)")
    ax2.set_title("最终服务费向量")
    ax2.set_ylim(min(cfg.fee_levels) - 0.02, max(cfg.fee_levels) + 0.02)
    fig.tight_layout()
    fig.savefig(outdir / "fig2_service_fee.png")
    plt.close(fig)


def plot_load(results: Dict, outdir: Path) -> None:
    cfg = results["cfg"]
    hours = _hours(cfg)
    proposed, baseline = results["proposed"], results["baseline"]

    fig = plt.figure(figsize=(11.0, 7.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(hours, proposed.load.power_kw.sum(axis=1) / 1000, color="#1f77b4", label="定制策略")
    ax.plot(hours, baseline.load.power_kw.sum(axis=1) / 1000, color="#d62728", ls="--", label="固定服务费")
    ax.set_ylabel("全网充电负荷 / MW")
    ax.set_xlabel("时刻 / h")
    ax.set_title("充电负荷曲线对比")
    ax.legend()

    for s in range(cfg.n_stations):
        row = 1 if s < 4 else 2
        axs = fig.add_subplot(gs[row, s % 4])
        axs.plot(hours, proposed.load.power_kw[:, s], color="#1f77b4")
        axs.plot(hours, baseline.load.power_kw[:, s], color="#d62728", ls="--", lw=1)
        axs.set_title(f"站{s + 1}", fontsize=9)
        axs.tick_params(labelsize=7)
    fig.suptitle("各站充电功率（蓝=定制策略，红虚线=固定服务费）", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_charging_load.png")
    plt.close(fig)


def plot_voltage(results: Dict, outdir: Path) -> None:
    cfg = results["cfg"]
    hours = _hours(cfg)
    proposed, baseline = results["proposed"], results["baseline"]
    vmin_p = np.nanmin(proposed.vm, axis=1)
    vmin_b = np.nanmin(baseline.vm, axis=1)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax.plot(hours, vmin_p, color="#1f77b4", label="定制策略")
    ax.plot(hours, vmin_b, color="#d62728", ls="--", label="固定服务费")
    ax.axhline(cfg.v_lower, color="gray", ls=":", label=f"电压下限 {cfg.v_lower}")
    ax.set_xlabel("时刻 / h")
    ax.set_ylabel("全网最低电压 / p.u.")
    ax.set_title("全天最低电压")
    ax.legend(fontsize=8)

    peak_slot = int(np.nanargmin(vmin_b))
    buses = np.arange(1, proposed.vm.shape[1] + 1)
    ax2.plot(buses, proposed.vm[peak_slot], "o-", ms=4, color="#1f77b4", label="定制策略")
    ax2.plot(buses, baseline.vm[peak_slot], "s--", ms=4, color="#d62728", label="固定服务费")
    ax2.axhline(cfg.v_lower, color="gray", ls=":")
    ax2.set_xlabel("节点编号")
    ax2.set_ylabel("电压 / p.u.")
    ax2.set_title(f"最低电压时段（{hours[peak_slot]:.2f} h）节点电压分布")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_voltage.png")
    plt.close(fig)


def plot_vdini(results: Dict, outdir: Path) -> None:
    cfg = results["cfg"]
    hours = _hours(cfg)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.plot(hours, results["proposed"].vdini, color="#1f77b4", label="定制策略")
    ax.plot(hours, results["baseline"].vdini, color="#d62728", ls="--", label="固定服务费")
    ax.plot(hours, results["uniform"].vdini, color="#2ca02c", ls=":", label="全网统一调整")
    ax.fill_between(hours, results["proposed"].vdini, results["baseline"].vdini,
                    where=results["proposed"].vdini < results["baseline"].vdini,
                    color="#1f77b4", alpha=0.15)
    ax.set_xlabel("时刻 / h")
    ax.set_ylabel("VDINI(t)")
    ax.set_title("全天电压偏差-关键度指标 VDINI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig5_vdini.png")
    plt.close(fig)


def plot_stations(results: Dict, outdir: Path) -> None:
    cfg = results["cfg"]
    proposed, baseline = results["proposed"], results["baseline"]
    x = np.arange(cfg.n_stations)
    labels = [f"S{s + 1}" for s in range(cfg.n_stations)]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    w = 0.38
    axes[0].bar(x - w / 2, proposed.load.station_peak_kw, w, label="定制策略", color="#1f77b4")
    axes[0].bar(x + w / 2, baseline.load.station_peak_kw, w, label="固定服务费", color="#d62728")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("站峰值功率 / kW")
    axes[0].set_title("站点峰值负荷")
    axes[0].legend(fontsize=8)

    axes[1].bar(x - w / 2, proposed.load.station_utilization * 100, w, color="#1f77b4")
    axes[1].bar(x + w / 2, baseline.load.station_utilization * 100, w, color="#d62728")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("桩利用率 / %")
    axes[1].set_title("站点利用率")

    axes[2].bar(x - w / 2, proposed.load.wait_by_station, w, color="#1f77b4")
    axes[2].bar(x + w / 2, baseline.load.wait_by_station, w, color="#d62728")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("平均等待 / min")
    axes[2].set_title("站均排队时长")
    fig.tight_layout()
    fig.savefig(outdir / "fig6_station_compare.png")
    plt.close(fig)


def plot_network(results: Dict, outdir: Path) -> None:
    cfg = results["cfg"]
    net = results["net"]
    stations = results["stations"]
    fees = results["proposed"].fees
    energy = results["proposed"].load.station_energy_kwh

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for i in range(net.rows):
        for j in range(net.cols):
            x, y = net.xy(net.node(i, j))
            if j + 1 < net.cols:
                x2, y2 = net.xy(net.node(i, j + 1))
                ax.plot([x, x2], [y, y2], color="#b0b0b0", lw=1.0)
            if i + 1 < net.rows:
                x2, y2 = net.xy(net.node(i + 1, j))
                ax.plot([x, x2], [y, y2], color="#b0b0b0", lw=1.0)
    scatter = ax.scatter([net.xy(s)[0] for s in stations], [net.xy(s)[1] for s in stations],
                         s=energy / energy.max() * 700 + 60, c=fees, cmap="coolwarm",
                         vmin=min(cfg.fee_levels), vmax=max(cfg.fee_levels),
                         edgecolors="k", zorder=3)
    for s, node in enumerate(stations):
        x, y = net.xy(node)
        ax.annotate(f"S{s + 1}\n{fees[s]:.2f}", (x, y), textcoords="offset points",
                    xytext=(9, 6), fontsize=8)
    ax.set_title("路网与充电站（点大小=日充电量，颜色=最终服务费）")
    ax.set_xlabel("x / km")
    ax.set_ylabel("y / km")
    ax.set_aspect("equal")
    fig.colorbar(scatter, ax=ax, shrink=0.8, label="服务费 / (元/kWh)")
    fig.tight_layout()
    fig.savefig(outdir / "fig7_road_network.png")
    plt.close(fig)


# ---------------------------------------------------------------- 汇总
def write_summary(results: Dict, outdir: Path) -> None:
    cfg = results["cfg"]
    p, b, u = results["proposed"], results["baseline"], results["uniform"]
    lines = [
        "快充负荷服务费定制策略仿真结果",
        "=" * 44,
        f"路网 {cfg.rows}x{cfg.cols}（间距 {cfg.spacing_km} km），"
        f"EV {cfg.n_ev} 辆，充电站 {cfg.n_stations} 座，"
        f"单站 {cfg.piles_per_station} 桩 x {cfg.pile_power_kw:.0f} kW",
        f"配电网：IEEE 33 节点，基础负荷缩放 {cfg.base_load_scale}",
        f"服务费区间 [{min(cfg.fee_levels)}, {max(cfg.fee_levels)}] 元/kWh，"
        f"迭代 {len(results['history'])} 轮，耗时 {results['elapsed']:.1f} s",
        "",
        f"{'指标':<22}{'定制策略':>12}{'固定服务费':>12}{'统一调整':>12}",
        "-" * 58,
    ]
    rows = [
        ("VDINI 全天累计", "vdini_daily", "{:.4f}"),
        ("VDINI 峰值", "vdini_peak", "{:.4f}"),
        ("全网最低电压 (p.u.)", "v_min", "{:.4f}"),
        ("电压越限时段数", "violation_slots", "{:.0f}"),
        ("充电事件数", "ev_events", "{:.0f}"),
        ("放弃充电车辆数", "balked", "{:.0f}"),
        ("日充电量-电池侧 (kWh)", "charge_energy_kwh", "{:.0f}"),
        ("日充电量-电网侧 (kWh)", "grid_energy_kwh", "{:.0f}"),
        ("全网峰值负荷 (kW)", "peak_load_kw", "{:.1f}"),
        ("平均排队 (min)", "avg_wait_min", "{:.2f}"),
        ("最长排队 (min)", "max_wait_min", "{:.2f}"),
        ("网损电量 (kWh)", "loss_mwh", "{:.1f}"),
        ("用户综合成本 (元)", "user_cost", "{:.0f}"),
        ("加权平均服务费 (元/kWh)", "weighted_mean_fee", "{:.3f}"),
    ]
    for label, key, fmt in rows:
        lines.append(f"{label:<22}{fmt.format(p.metrics[key]):>12}"
                     f"{fmt.format(b.metrics[key]):>12}{fmt.format(u.metrics[key]):>12}")
    lines += [
        "",
        "最终服务费向量（站1..站8）：" + ", ".join(f"{f:.2f}" for f in p.fees),
        "固定服务费向量（对照组）：  " + ", ".join(f"{f:.2f}" for f in b.fees),
    ]
    (outdir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def write_reports(results: Dict, outdir: Path) -> None:
    _setup()
    outdir.mkdir(parents=True, exist_ok=True)
    save_tables(results, outdir)
    plot_convergence(results, outdir)
    plot_fees(results, outdir)
    plot_load(results, outdir)
    plot_voltage(results, outdir)
    plot_vdini(results, outdir)
    plot_stations(results, outdir)
    plot_network(results, outdir)
    write_summary(results, outdir)
