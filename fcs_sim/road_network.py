"""路网拓扑：8x8 方格路网与全源最短路。"""

from __future__ import annotations

import math

import numpy as np


class RoadNetwork:
    """方格路网。节点按 行*cols+列 编号，边为相邻路口的双向路段。"""

    def __init__(
        self,
        rows: int,
        cols: int,
        spacing_km: float,
        free_speed_kmh: float,
        peak_speed_factor: float,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.n_nodes = rows * cols
        self.spacing_km = spacing_km
        self.free_speed_kmh = free_speed_kmh
        self.peak_speed_factor = peak_speed_factor
        self.dist_km, self.time_min = self._all_pairs_shortest_path()

    # ---------- 编号与坐标 ----------
    def node(self, i: int, j: int) -> int:
        return i * self.cols + j

    def coords(self, node: int) -> tuple[int, int]:
        return divmod(node, self.cols)

    def xy(self, node: int) -> tuple[float, float]:
        i, j = self.coords(node)
        return j * self.spacing_km, i * self.spacing_km

    def degree(self, node: int) -> int:
        """节点度数（连接的边数）：内部 4，边 3，角 2。"""
        i, j = self.coords(node)
        deg = 4
        if i in (0, self.rows - 1):
            deg -= 1
        if j in (0, self.cols - 1):
            deg -= 1
        return deg

    # ---------- 最短路 ----------
    def _all_pairs_shortest_path(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.n_nodes
        big = 1e9
        dist = np.full((n, n), big)
        time = np.full((n, n), big)
        np.fill_diagonal(dist, 0.0)
        np.fill_diagonal(time, 0.0)

        leg_min = self.spacing_km / self.free_speed_kmh * 60.0
        for i in range(self.rows):
            for j in range(self.cols):
                a = self.node(i, j)
                for di, dj in ((0, 1), (1, 0)):
                    ii, jj = i + di, j + dj
                    if ii < self.rows and jj < self.cols:
                        b = self.node(ii, jj)
                        dist[a, b] = dist[b, a] = self.spacing_km
                        time[a, b] = time[b, a] = leg_min

        # Floyd-Warshall
        for k in range(n):
            dist = np.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])
            time = np.minimum(time, time[:, k : k + 1] + time[k : k + 1, :])
        return dist, time

    def distance(self, a: int, b: int) -> float:
        return float(self.dist_km[a, b])

    def travel_time_min(self, a: int, b: int, hour: float) -> float:
        """给定出发时刻的行驶时间（分钟），含高峰拥堵折减。"""
        return float(self.time_min[a, b]) * self.speed_penalty(hour)

    def speed_penalty(self, hour: float) -> float:
        def bump(mu: float, sigma: float) -> float:
            return math.exp(-0.5 * ((hour - mu) / sigma) ** 2)

        congestion = min(1.0, bump(8.0, 1.2) + bump(18.0, 1.5))
        return 1.0 - (1.0 - self.peak_speed_factor) * congestion
