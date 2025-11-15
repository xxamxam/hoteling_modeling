# simple_interactive_dash.py - Minimal interactive Dash app for Hotelling's model

from dash import Dash, Input, Output, State
import dash
from hoteling.branch_bound import BaseRevenueFunction
from hoteling.dash_panel.dash_html import create_layout
from hoteling.dash_panel.hotelling_game import HotellingGame


def create_minimal_dash_app(g, initial_M=3, rf=None):
    app = Dash(__name__)

    if rf is None:
        rf = BaseRevenueFunction()

    game = HotellingGame(graph=g, initial_M=initial_M, rf=rf)
    app.layout = create_layout(game)

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
        State("show-labels-checkbox", "value"),
        prevent_initial_call=True,
    )
    def update_graph(click, m_val, base_val, recompute, bb, reset, generate,
                     max_iter_input, cache_input, generator, line_n, star_n_leaves,
                     tree_n, tree_seed, grid_rows, grid_cols, sellers_data, rf_data,
                     show_labels_checkbox):
        # Get triggered component
        trigger = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None
        show_labels = "show" in (show_labels_checkbox or [])

        game.update_parameters(M=m_val, base_cost=base_val,
                               max_iter=max_iter_input, cache_size=cache_input)

        if trigger == "generate-btn":
            params = {
                "line_n": line_n, "star_n_leaves": star_n_leaves,
                "tree_n": tree_n, "tree_seed": tree_seed,
                "grid_rows": grid_rows, "grid_cols": grid_cols
            }
            try:
                game.set_graph(generator, **params)
                info_text = f"Generated new {generator} graph"
            except Exception as e:
                info_text = f"Error generating graph: {str(e)}"

        elif trigger == "reset-btn":
            game.reset_sellers()
            info_text = "Sellers reset"

        elif trigger == "bb-btn":
            try:
                info_text = game.run_branch_and_bound()
            except Exception as e:
                info_text = f"B&B error: {str(e)}"

        elif trigger == "graph":
            nid = click["points"][0]["customdata"] if click else None
            if nid is not None:
                info_text = game.toggle_seller(nid)
            else:
                info_text = "Clicked, but no data"
        else:
            info_text = "Updated parameters"

        fig = game.make_figure(show_labels=show_labels)
        stats_text = game.compute_stats()
        game_state = game.to_dict()

        return fig, list(game.sellers_set), info_text, stats_text, game_state["rf"]

    return app


# Example:
if __name__ == "__main__":
    from hoteling.graph_generators import generate_line_graph
    from hoteling.branch_bound import BaseRevenueFunction
    g = generate_line_graph(6)
    rf = BaseRevenueFunction(base_cost=10)
    app = create_minimal_dash_app(g, initial_M=3, rf=rf)
    app.run(debug=True)
