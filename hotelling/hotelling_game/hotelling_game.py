# hotelling_game.py - Main game state and logic for Hotelling's model

import numpy as np
import graph_tool.all as gt

from hotelling.algorithms.node import Node
from hotelling.hotelling_game.cost_functions import BaseRevenueFunction
from hotelling.generators.graph_generators import (
    generate_line_graph,
    generate_star_graph,
    generate_random_tree,
    generate_grid_graph,
)

from hotelling.algorithms.branch_bound import BBTree, BBHeap
from hotelling.dash_panel.dash_plot import make_figure
from hotelling.hotelling_game.game_evaluation import evaluate_sellers
from hotelling.relaxed.mirror_descent import run_mirror_descent
from hotelling.relaxed.rl_module import run_rl
from hotelling.relaxed.utils import check_nash_deterministic


class HotellingGame:
    def __init__(
        self, graph=None, initial_M=3, rf=None, max_iter=1000, cache_size=200000
    ):
        self.graph = graph
        self.sellers_set = set()
        self.M = initial_M
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.base_cost = rf.base_cost if rf else BaseRevenueFunction().base_cost
        self.rf = rf or BaseRevenueFunction(base_cost=self.base_cost)
        self.is_nash = True  # Default for exact
        self.last_result = None

    def update_parameters(self, M=None, base_cost=None, max_iter=None, cache_size=None):
        if M is not None:
            self.M = M
            # Remove excess sellers if any
            if len(self.sellers_set) > M:
                self.sellers_set.clear()
        if base_cost is not None:
            self.base_cost = base_cost
            self.rf = BaseRevenueFunction(base_cost=self.base_cost)
        if max_iter is not None:
            self.max_iter = max_iter
        if cache_size is not None:
            self.cache_size = cache_size

    def set_graph(self, generator_name, **params):
        print(params.get("tree_seed"))
        generators = {
            "line": lambda: generate_line_graph(int(params.get("line_n") or 6)),
            "star": lambda: generate_star_graph(int(params.get("star_n_leaves") or 3)),
            "random_tree": lambda: generate_random_tree(
                int(params.get("tree_n") or 6),
                seed=int(params.get("tree_seed")) if params.get("tree_seed") else None,
            ),
            "grid": lambda: generate_grid_graph(
                int(params.get("grid_rows") or 3), int(params.get("grid_cols") or 3)
            ),
        }
        self.graph = generators[generator_name]()
        self.sellers_set = set()  # Reset sellers on new graph

    def reset_sellers(self):
        self.sellers_set.clear()

    def toggle_seller(self, node_id):
        node_id = int(node_id)
        if node_id in self.sellers_set:
            self.sellers_set.remove(node_id)
            return f"Removed seller {node_id}"
        elif len(self.sellers_set) >= self.M:
            return f"Cannot add seller {node_id}: max {self.M} sellers reached"
        else:
            self.sellers_set.add(node_id)
            return f"Added seller {node_id}"

    def run_branch_and_bound(self):
        if self.graph is None:
            raise ValueError("Graph not set")
        bbt = BBHeap(
            self.graph, self.M, self.rf, verbose=False, cache_maxsize=self.cache_size
        )

        bbt.run(max_iterations=self.max_iter)
        self.sellers_set = set(int(x) for x in bbt.occupation)
        # Verify Nash for exact method
        dist_u_v = self._compute_distance_matrix()
        d_u = np.ones(self.graph.num_vertices())
        x_list = [np.zeros(self.graph.num_vertices()) for _ in range(self.M)]
        for i, pos in enumerate(self.sellers_set):
            x_list[i][pos] = 1.0
        self.is_nash, _ = check_nash_deterministic(x_list, dist_u_v, d_u)
        run_stats = bbt.run_stat
        return f"B&B completed, found {len(self.sellers_set)} sellers.\n Running statistics: {run_stats}"

    def _compute_distance_matrix(self):
        n = self.graph.num_vertices()
        dist_u_v = np.zeros((n, n))
        for u in range(n):
            dists = gt.shortest_distance(
                self.graph, source=self.graph.vertex(u), weights=self.graph.ep["weight"]
            )
            for v in range(n):
                dist_u_v[u, v] = dists[self.graph.vertex(v)]
        return dist_u_v

    def run_mirror_descent(self, T=200, eta=0.2, beta=0.05, eps_H=0.05):
        if self.graph is None:
            raise ValueError("Graph not set")
        dist_u_v = self._compute_distance_matrix()
        d_u = np.ones(self.graph.num_vertices())  # uniform demand
        result = run_mirror_descent(
            self.graph, dist_u_v, d_u, m=self.M, T=T, eta=eta, beta=beta, eps_H=eps_H
        )
        positions = [np.argmax(x) for x in result["x"]]
        self.sellers_set = set(positions)
        self.is_nash = result["nash"]
        self.last_result = result
        return f"Mirror Descent completed, found {len(self.sellers_set)} sellers. Nash: {result['nash']}"

    def run_rl(self, T=200, eta=0.1, entropy_weight=0.05, eps_H=0.05, seed=42):
        if self.graph is None:
            raise ValueError("Graph not set")
        dist_u_v = self._compute_distance_matrix()
        d_u = np.ones(self.graph.num_vertices())  # uniform demand
        result = run_rl(
            self.graph,
            dist_u_v,
            d_u,
            m=self.M,
            T=T,
            eta=eta,
            entropy_weight=entropy_weight,
            eps_H=eps_H,
            seed=seed,
        )
        positions = [np.argmax(x) for x in result["x"]]
        self.sellers_set = set(positions)
        self.is_nash = result["nash"]
        self.last_result = result
        return f"RL completed, found {len(self.sellers_set)} sellers. Nash: {result['nash']}"

    def compute_stats(self):
        if not self.graph or not self.sellers_set:
            return "Sellers Statistics\nNo sellers"

        positions = evaluate_sellers(
            self.graph,
            self.sellers_set,
            self.M,
            "weight",
            self.rf,
            extended_return=True,
        )
        assert isinstance(positions, dict)

        num_sellers = positions.get("num_sellers", {})
        revenues = positions.get("revenues", {})
        # Sort by revenue ascending
        sorted_nids = sorted(self.sellers_set, key=lambda nid: revenues.get(nid, 0))
        stats_lines = []
        min_rps = float("inf")
        for nid in sorted_nids:
            k = num_sellers.get(nid, 0)
            rev = revenues.get(nid, 0)
            rps = rev / max(k, 1)
            min_rps = min(min_rps, rps)
            stats_lines.append(
                f"Node {nid}: k={k}, rev={rev:.3f},\n       $/seller={rps:.3f}\n"
            )

        nash_status = "Yes" if self.is_nash else "No"
        stats_text = (
            f"Sellers Statistics\nMinimal Revenue: {min_rps:.3f}\nNash Equilibrium: {nash_status}\n"
            + "\n".join(stats_lines)
        )
        return stats_text

    def make_figure(self, show_labels=True, redraw: int = 0):
        return make_figure(
            self.graph, self.sellers_set, self.M, self.rf, show_labels, redraw=redraw
        )

    def to_dict(self):
        return {
            "sellers_set": list(self.sellers_set),
            "M": self.M,
            "base_cost": self.base_cost,
            "rf": {"base_cost": self.rf.base_cost},
        }
