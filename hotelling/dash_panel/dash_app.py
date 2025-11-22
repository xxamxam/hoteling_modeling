# simple_interactive_dash.py - Minimal interactive Dash app for Hotelling's model

from dash import Dash, Input, Output, State
import dash
import plotly.graph_objects as go

from hotelling.algorithms.branch_bound import BaseRevenueFunction
from hotelling.dash_panel.dash_html import create_layout
from hotelling.dash_panel.dash_plot import make_strategy_figure
from hotelling.hotelling_game.hotelling_game import HotellingGame


def create_dash_app(g, initial_M=3, rf=None):
    app = Dash(
        __name__,
    )

    if rf is None:
        rf = BaseRevenueFunction()

    game = HotellingGame(graph=g, initial_M=initial_M, rf=rf)
    app.layout = create_layout(game)

    @app.callback(
        [
            Output("line-params", "style"),
            Output("star-params", "style"),
            Output("tree-params", "style"),
            Output("grid-params", "style"),
        ],
        Input("generator-dropdown", "value"),
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
        [
            Output("exact-params", "style"),
            Output("md-params", "style"),
            Output("rl-params", "style"),
        ],
        Input("method-dropdown", "value"),
    )
    def toggle_method_params_visibility(chosen_method):
        styles = [{"display": "none"} for _ in range(3)]
        if chosen_method == "exact":
            styles[0] = {"display": "block"}
        elif chosen_method == "md":
            styles[1] = {"display": "block"}
        elif chosen_method == "rl":
            styles[2] = {"display": "block"}
        return styles

    @app.callback(
        Output("convergence-container", "style"),
        Input("method-dropdown", "value"),
    )
    def toggle_convergence_visibility(chosen_method):
        if chosen_method in ["md", "rl"]:
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        Output("iteration-slider", "max"),
        Input("method-dropdown", "value"),
        Input("md-T-input", "value"),
        Input("rl-T-input", "value"),
    )
    def update_slider_max(method, md_T, rl_T):
        if method == "md":
            return max(0, md_T - 1) if md_T else 199
        elif method == "rl":
            return max(0, rl_T - 1) if rl_T else 199
        return 199

    @app.callback(
        Output("player-checklist", "options"),
        Output("player-checklist", "value"),
        Input("M-input", "value"),
    )
    def update_player_checklist(M):
        if M is None or M < 1:
            M = 3
        options = [{"label": f"Player {i + 1}", "value": i} for i in range(M)]
        value = list(range(M))  # Default to all selected
        return options, value

    @app.callback(
        Output("strategy-graph", "figure"),
        Output("strategy-bars", "figure"),
        Input("iteration-slider", "value"),
        Input("player-checklist", "value"),
        State("method-dropdown", "value"),
    )
    def update_strategy_graph(step, selected_players, method):
        try:
            if (
                method in ["md", "rl"]
                and hasattr(game, "last_result")
                and game.last_result
                and game.graph
            ):
                strategies = game.last_result["log"]
                effective_step = min(step, len(strategies[0]) - 1) if strategies else 0
                fig = make_strategy_figure(
                    game.graph,
                    strategies,
                    effective_step,
                    selected_players=selected_players,
                )
                # Create bar chart
                bars_fig = go.Figure()
                node_ids = list(range(game.graph.num_vertices()))
                for i, strategy_log in enumerate(strategies):
                    y = strategy_log[effective_step]
                    bars_fig.add_trace(
                        go.Bar(
                            x=[f"Node {nid}" for nid in node_ids],
                            y=y,
                            name=f"Player {i + 1}",
                        )
                    )
                bars_fig.update_layout(
                    barmode="group",
                    title=f"Strategy Probabilities at Step {effective_step}",
                    xaxis_title="Nodes",
                    yaxis_title="Probability",
                )
                return fig, bars_fig
        except Exception as e:
            print(f"Error in update_strategy_graph: {e}")
        return go.Figure(), go.Figure()

    @app.callback(
        Output("graph", "figure"),
        Output("sellers", "data"),
        Output("click-info", "children"),
        Output("stats-display", "children"),
        Output("nash-status", "children"),
        Output("nash-status", "style"),
        Output("convergence-graph", "figure"),
        Output("rf", "data"),
        Output("iteration-slider", "value"),
        Input("graph", "clickData"),
        Input("M-input", "value"),
        Input("base-input", "value"),
        Input("recompute-btn", "n_clicks"),
        Input("run-method-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
        Input("generate-btn", "n_clicks"),
        State("method-dropdown", "value"),
        State("max-iter-input", "value"),
        State("cache-input", "value"),
        State("md-T-input", "value"),
        State("md-eta-input", "value"),
        State("md-beta-input", "value"),
        State("md-eps-H-input", "value"),
        State("rl-T-input", "value"),
        State("rl-eta-input", "value"),
        State("rl-entropy-weight-input", "value"),
        State("rl-eps-H-input", "value"),
        State("rl-seed-input", "value"),
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
        prevent_initial_call=True,
    )
    def update_graph(
        click,
        m_val,
        base_val,
        recompute,
        run_method,
        reset,
        generate,
        method,
        max_iter_input,
        cache_input,
        md_T,
        md_eta,
        md_beta,
        md_eps_H,
        rl_T,
        rl_eta,
        rl_entropy_weight,
        rl_eps_H,
        rl_seed,
        generator,
        line_n,
        star_n_leaves,
        tree_n,
        tree_seed,
        grid_rows,
        grid_cols,
        sellers_data,
        rf_data,
        upload_contents,
        upload_filename,
        show_labels_checkbox,
        load_file,
    ):
        # Get triggered component
        trigger = (
            dash.callback_context.triggered[0]["prop_id"].split(".")[0]
            if dash.callback_context.triggered
            else None
        )
        show_labels = "show" in (show_labels_checkbox or [])
        game.update_parameters(
            M=m_val, base_cost=base_val, max_iter=max_iter_input, cache_size=cache_input
        )

        if trigger == "generate-btn":
            params = {
                "line_n": line_n,
                "star_n_leaves": star_n_leaves,
                "tree_n": tree_n,
                "tree_seed": tree_seed,
                "grid_rows": grid_rows,
                "grid_cols": grid_cols,
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

        elif trigger == "run-method-btn":
            try:
                if method == "exact":
                    info_text = game.run_branch_and_bound()
                elif method == "md":
                    info_text = game.run_mirror_descent(
                        T=md_T, eta=md_eta, beta=md_beta, eps_H=md_eps_H
                    )
                elif method == "rl":
                    info_text = game.run_rl(
                        T=rl_T,
                        eta=rl_eta,
                        entropy_weight=rl_entropy_weight,
                        eps_H=rl_eps_H,
                        seed=rl_seed,
                    )
                else:
                    info_text = "Unknown method"
            except Exception as e:
                info_text = f"Method error: {str(e)}"

        elif trigger == "load-file-btn":
            if upload_contents is not None:
                import base64
                import tempfile
                import os

                try:
                    content_string = upload_contents.split(",")[1]
                    content_decoded = base64.b64decode(content_string)
                    with tempfile.NamedTemporaryFile(
                        mode="w+", delete=False, suffix=".txt"
                    ) as temp_file:
                        temp_file.write(content_decoded.decode("utf-8"))
                        temp_path = temp_file.name
                    from hotelling.generators.graph_reader import read_graph_from_file

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

        fig = game.make_figure(show_labels=show_labels, redraw=recompute)
        stats_text = game.compute_stats()
        game_state = game.to_dict()

        nash_text = "Nash Equilibrium: Yes" if game.is_nash else "Nash Equilibrium: No"
        nash_style = {
            "width": "100%",
            "padding": "10px",
            "borderRadius": "5px",
            "fontSize": "14px",
            "fontWeight": "bold",
            "textAlign": "center",
            "marginBottom": "10px",
            "backgroundColor": "#4CAF50" if game.is_nash else "#F44336",
            "color": "white",
        }

        # Create convergence figure if available
        if game.last_result and "entropy" in game.last_result:
            data = []
            for i, entropy_list in enumerate(game.last_result["entropy"]):
                data.append(
                    go.Scatter(
                        x=list(range(len(entropy_list))),
                        y=entropy_list,
                        mode="lines",
                        name=f"Player {i + 1} Entropy",
                    )
                )
            conv_fig = go.Figure(data=data)
            conv_fig.update_layout(
                title="Entropy Convergence",
                xaxis_title="Iteration",
                yaxis_title="Entropy",
            )
        else:
            conv_fig = go.Figure()

        # Set slider value to final iteration if method was run
        if trigger == "run-method-btn" and method in ["md", "rl"] and game.last_result:
            slider_value = len(game.last_result["log"][0]) - 1
        else:
            slider_value = dash.no_update

        return (
            fig,
            list(game.sellers_set),
            info_text,
            stats_text,
            nash_text,
            nash_style,
            conv_fig,
            game_state["rf"],
            slider_value,
        )

    return app
