# simple_interactive_dash.py - Minimal interactive Dash app for Hotelling's model

import numpy as np
import plotly.graph_objects as go

from graph_tool.draw import sfdp_layout
import dash
from dash import Dash, dcc, html, Input, Output, State
from hoteling.branch_bound import BaseRevenueFunction, Node
from hoteling.dash_panel.dash_html import create_layout
from hoteling.dash_panel.dash_plot import make_figure


def create_minimal_dash_app(g, initial_M=3, rf=None):
    if rf is None:
        rf = BaseRevenueFunction()

    app = Dash(__name__)

    app.layout = create_layout(g, initial_M, rf)

    @app.callback(
        [
            Output("line-params", "style"),
            Output("star-params", "style"),
            Output("tree-params", "style"),
            Output("grid-params", "style")
        ],
        Input("generator-dropdown", "value")
    )
    def toggle_params_visibility(chosen_generator):
        styles = [{"display": "none"} for _ in range(4)]
        if chosen_generator == "line":
            styles[0] = {"display": "block"}
        elif chosen_generator == "star":
            styles[1] = {"display": "block"}
        elif chosen_generator == "random_tree":
            styles[2] = {"display": "block"}
        elif chosen_generator == "grid":
            styles[3] = {"display": "block"}
        return styles

    @app.callback(
        Output("graph", "figure"),
        Output("sellers", "data"),
        Output("click-info", "children"),
        Output("stats-display", "children"),
        Output("rf", "data"),
        Input("graph", "clickData"),
        Input("M-input", "value"),
        Input("base-input", "value"),
        Input("recompute-btn", "n_clicks"),
        Input("bb-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
        Input("generate-btn", "n_clicks"),
        State("max-iter-input", "value"),
        State("cache-input", "value"),
        State("generator-dropdown", "value"),
        State("line-n", "value"),
        State("star-n-leaves", "value"),
        State("tree-n", "value"),
        State("tree-seed", "value"),
        State("grid-rows", "value"),
        State("grid-cols", "value"),
        State("sellers", "data"),
        State("rf", "data"),
        prevent_initial_call=True,
    )
    def update_graph(click, m_val, base_val, recompute, bb, reset, generate, max_iter_input, cache_input, generator, line_n, star_n_leaves, tree_n, tree_seed, grid_rows, grid_cols, sellers_data, rf_data):
        global g  # allow modify global g for simplicity in this simple version

        sellers_set = set(sellers_data or [])
        current_m = m_val if m_val else initial_M

        rf_new = BaseRevenueFunction(base_cost=base_val or rf.base_cost)
        rf_data = {"base_cost": rf_new.base_cost}

        # If M decreased, clear sellers if exceeding
        if len(sellers_set) > current_m:
            sellers_set.clear()

        # Get triggered component
        trigger = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None

        if trigger == "generate-btn":
            # Generate new graph
            params = {
                "line_n": line_n or 6,
                "star_n_leaves": star_n_leaves or 3,
                "tree_n": tree_n or 6,
                "tree_seed": tree_seed or None,
                "grid_rows": grid_rows or 3,
                "grid_cols": grid_cols or 3
            }
            from hoteling.graph_generators import (
                generate_line_graph, generate_star_graph, generate_random_tree,
                generate_grid_graph
            )
            generators = {
                "line": lambda p: generate_line_graph(int(p["line_n"])),
                "star": lambda p: generate_star_graph(int(p["star_n_leaves"])),
                "random_tree": lambda p: generate_random_tree(int(p["tree_n"]), seed=int(p["tree_seed"]) if p["tree_seed"] else None),
                "grid": lambda p: generate_grid_graph(int(p["grid_rows"]), int(p["grid_cols"]))
            }
            try:
                g = generators[generator](params)
                sellers_set = set()
                info_text = f"Generated new {generator} graph"
            except Exception as e:
                info_text = f"Error generating graph: {str(e)}"
                fig = make_figure(g, sellers_set, current_m, rf_new)
                return fig, list(sellers_set), info_text, "Error", rf_data

        elif trigger == "reset-btn":
            sellers_set.clear()
            info_text = "Sellers reset"

        elif trigger == "bb-btn":
            from hoteling.branch_bound import BBTree
            max_iter = max_iter_input or 1000
            cache_size = cache_input or 200000
            bbt = BBTree(g, current_m, rf_new, verbose=False, cache_maxsize=cache_size)
            bbt.run(max_iterations=max_iter)
            sellers_set = set(int(x) for x in bbt.occupation)
            info_text = f"B&B completed, found {len(sellers_set)} sellers"

        elif trigger == "graph":
            nid = click["points"][0]["customdata"]
            nid = int(nid)
            if nid in sellers_set:
                sellers_set.remove(nid)
                info_text = f"Removed seller {nid}"
            else:
                if len(sellers_set) >= current_m:
                    info_text = f"Cannot add seller {nid}: max {current_m} sellers reached"
                else:
                    sellers_set.add(nid)
                    info_text = f"Added seller {nid}"

        else:
            info_text = "Updated parameters"

        fig = make_figure(g, sellers_set, current_m, rf_new)

        # Stats
        positions = Node.get_positions(g, sellers_set, current_m, "weight", rf_new, extended_return=True)
        num_sellers = positions.get("num_sellers", {})
        revenues = positions.get("revenues", {})
        if sellers_set:
            # Sort by revenue ascending
            sorted_nids = sorted(sellers_set, key=lambda nid: revenues.get(nid, 0))
            stats_lines = []
            for nid in sorted_nids:
                k = num_sellers.get(nid, 0)
                rev = revenues.get(nid, 0)
                rps = rev / max(k, 1)
                stats_lines.append(f"Node {nid}: k={k}, rev={rev:.3f}, $/seller={rps:.3f}")
            min_rev = min(revenues.values())
            stats_text = f"Sellers Statistics\nMinimal Revenue: {min_rev:.3f}\n" + "\n".join(stats_lines)
        else:
            stats_text = "Sellers Statistics\nNo sellers"

        return fig, list(sellers_set), info_text, stats_text, rf_data

    return app




# Example:
if __name__ == "__main__":
    from hoteling.graph_generators import generate_line_graph
    from hoteling.branch_bound import BaseRevenueFunction
    g = generate_line_graph(6)
    rf = BaseRevenueFunction(base_cost=10)
    app = create_minimal_dash_app(g, initial_M=3, rf=rf)
    app.run(debug=True)
