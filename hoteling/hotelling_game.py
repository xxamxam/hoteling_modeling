# hotelling_game.py - Main game state and logic for Hotelling's model

import numpy as np
from hoteling.algorithms.branch_bound import BaseRevenueFunction, Node
from hoteling.generators.graph_generators import (
    generate_line_graph, generate_star_graph, generate_random_tree,
    generate_grid_graph
)
from hoteling.algorithms.branch_bound import BBTree, BBHeap
from hoteling.dash_panel.dash_plot import make_figure
from hoteling.game_evaluation import evaluate_sellers

class HotellingGame:
    def __init__(self, graph=None, initial_M=3, rf=None, max_iter=1000, cache_size=200000):
        self.graph = graph
        self.sellers_set = set()
        self.M = initial_M
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.base_cost = rf.base_cost if rf else BaseRevenueFunction().base_cost
        self.rf = rf or BaseRevenueFunction(base_cost=self.base_cost)

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
                seed=int(params.get("tree_seed")) if params.get("tree_seed") else None
            ),
            "grid": lambda: generate_grid_graph(
                int(params.get("grid_rows") or 3),
                int(params.get("grid_cols") or 3)
            )
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

    def run_branch_and_bound(self, callback=None):
        if self.graph is None:
            raise ValueError("Graph not set")
        bbt = BBHeap(self.graph,
                     self.M,
                     self.rf,
                     verbose=False,
                     cache_maxsize=self.cache_size
                     )

        bbt.run(max_iterations=self.max_iter, callback=callback)
        self.sellers_set = set(int(x) for x in bbt.occupation)
        run_stats = bbt.run_stat
        return f"B&B completed, found {len(self.sellers_set)} sellers.\n Running statistics: {run_stats}"

    def compute_stats(self):
        if not self.graph or not self.sellers_set:
            return "Sellers Statistics\nNo sellers"

        positions = evaluate_sellers(self.graph, self.sellers_set, self.M, "weight", self.rf, extended_return=True)
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
            stats_lines.append(f"Node {nid}: k={k}, rev={rev:.3f},\n       $/seller={rps:.3f}\n")

        stats_text = f"Sellers Statistics\nMinimal Revenue: {min_rps:.3f}\n" + "\n".join(stats_lines)
        return stats_text

    def make_figure(self, show_labels=True, redraw: int =0):
        return make_figure(self.graph,
                           self.sellers_set,
                           self.M,
                           self.rf,
                           show_labels,
                           redraw=redraw
                           )

    def to_dict(self):
        return {
            "sellers_set": list(self.sellers_set),
            "M": self.M,
            "base_cost": self.base_cost,
            "rf": {"base_cost": self.rf.base_cost}
        }
