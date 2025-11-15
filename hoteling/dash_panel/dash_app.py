# simple_interactive_dash.py - Minimal interactive Dash app for Hotelling's model

from dash import Dash, Input, Output, State
import dash

from hoteling.algorithms.branch_bound import BaseRevenueFunction
from hoteling.dash_panel.dash_html import create_layout
from hoteling.hotelling_game import HotellingGame


def create_dash_app(g, initial_M=3, rf=None):
    app = Dash(__name__, )

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
        Output("bb-progress-modal", "style"),
        Output("bb-progress-graph", "figure"),
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
        State("upload-graph", "contents"),
        State("upload-graph", "filename"),
        State("show-labels-checkbox", "value"),
        Input("load-file-btn", "n_clicks"),
        Input("bb-stop-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_graph(click, m_val, base_val, recompute, bb, reset, generate,
                     max_iter_input, cache_input, generator, line_n, star_n_leaves,
                     tree_n, tree_seed, grid_rows, grid_cols, sellers_data, rf_data,
                     upload_contents, upload_filename, show_labels_checkbox, load_file, bb_stop):
        # Get triggered component
        trigger = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None
        import plotly.graph_objects as go
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

        elif trigger == "recompute-btn":
            info_text = "Redrawn"

        elif trigger == "reset-btn":
            game.reset_sellers()
            info_text = "Sellers reset"

        elif trigger == "bb-btn":
            from hoteling.algorithms.branch_bound import BBCallback
            callback = BBCallback(num_updates=10)
            try:
                info_text = game.run_branch_and_bound(callback)
            except Exception as e:
                info_text = f"B&B error: {str(e)}"

        elif trigger == "load-file-btn":
            if upload_contents is not None:
                import base64
                import tempfile
                import os
                try:
                    content_string = upload_contents.split(',')[1]
                    content_decoded = base64.b64decode(content_string)
                    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as temp_file:
                        temp_file.write(content_decoded.decode('utf-8'))
                        temp_path = temp_file.name
                    from hoteling.generators.graph_reader import read_graph_from_file
                    g_loaded = read_graph_from_file(temp_path)
                    game.graph = g_loaded
                    game.sellers_set = set()
                    info_text = f"Loaded graph from file: {upload_filename or 'file'}"
                    os.unlink(temp_path)
                except Exception as e:
                    info_text = f"Error loading graph from file: {str(e)}"
            else:
                info_text = "No file uploaded"

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

        # Determine modal state
        if trigger == "bb-btn":
            fig_progress = go.Figure()
            if 'callback' in locals() and callback.best_values:
                iterations = list(range(len(callback.best_values)))
                fig_progress.add_trace(go.Scatter(x=iterations, y=callback.best_values, mode="lines+markers", name="Best Value"))
                fig_progress.update_layout(title="B&B Convergence", xaxis_title="Iteration", yaxis_title="Best Revenue")
        else:
            fig_progress = go.Figure()

        style_modal = {
            "position": "fixed",
            "top": "15%" if trigger == "bb-btn" else "20%",  # But perhaps simplify to one
            "left": "15%" if trigger == "bb-btn" else "20%",
            "width": "70%" if trigger == "bb-btn" else "60%",
            "height": "70%" if trigger == "bb-btn" else "40%",
            "backgroundColor": "white",
            "border": "1px solid black",
            "zIndex": 1000,
            "padding": "10px",
            "display": "block" if trigger == "bb-btn" else "none"
        }

        return fig, list(game.sellers_set), info_text, stats_text, game_state["rf"], style_modal, fig_progress

    return app
