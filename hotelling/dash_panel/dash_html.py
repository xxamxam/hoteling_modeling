# dash_html.py - HTML layout for the Dash app
# simple_interactive_dash.py - Minimal interactive Dash app for Hotelling's model

import numpy as np
import plotly.graph_objects as go

from graph_tool.draw import sfdp_layout
import dash
from dash import Dash, dcc, html, Input, Output, State
from hotelling.algorithms.branch_bound import BaseRevenueFunction, Node


from dash import dcc, html
from .dash_plot import make_figure

# Layout constants
LEFT_PANEL_WIDTH_PERCENT = 20
RIGHT_PANEL_WIDTH_PERCENT = 15


def create_show_labels_checklist():
    return html.Div(
        style={"position": "absolute", "top": "10px", "right": "10px"},
        children=[
            dcc.Checklist(
                id="show-labels-checkbox",
                options=[{"label": "Show Labels", "value": "show"}],
                value=["show"],
            )
        ],
    )


def create_layout(game):
    # Initial state using game
    sellers = list(game.sellers_set)
    fig = game.make_figure()
    stats_text = game.compute_stats()

    return html.Div(
        [
            # Stores
            dcc.Store(id="sellers", data=sellers),
            dcc.Store(id="rf", data={"base_cost": game.base_cost}),
            dcc.Store(id="current-graph", data=None),  # Not used for now
            # Layout: left 20%, right 15%, middle remaining
            html.Div(
                style={"display": "flex", "height": "100vh"},
                children=[
                    # Left: Controls
                    html.Div(
                        style={
                            "width": f"{LEFT_PANEL_WIDTH_PERCENT}%",
                            "height": "100%",
                            "display": "flex",
                            "flexDirection": "column",
                        },
                        children=[
                            html.Div(
                                style={"padding": "10px", "overflowY": "auto"},
                                children=[
                                    html.H3("Graph Generation"),
                                    dcc.Dropdown(
                                        id="generator-dropdown",
                                        options=[
                                            {"label": "Line Graph", "value": "line"},
                                            {"label": "Star Graph", "value": "star"},
                                            {
                                                "label": "Random Tree",
                                                "value": "random_tree",
                                            },
                                            {"label": "Grid Graph", "value": "grid"},
                                        ],
                                        value="line",  # default
                                        style={"marginBottom": "10px"},
                                    ),
                                    html.Div(
                                        id="line-params",
                                        children=[
                                            dcc.Input(
                                                id="line-n",
                                                type="number",
                                                min=2,
                                                value=6,
                                                placeholder="Number of vertices",
                                                style={
                                                    "width": "100%",
                                                    "marginTop": "5px",
                                                },
                                            )
                                        ],
                                        style={"display": "block"},
                                    ),
                                    html.Div(
                                        id="star-params",
                                        children=[
                                            dcc.Input(
                                                id="star-n-leaves",
                                                type="number",
                                                min=2,
                                                value=3,
                                                placeholder="Number of leaves",
                                                style={
                                                    "width": "100%",
                                                    "marginTop": "5px",
                                                },
                                            )
                                        ],
                                        style={"display": "none"},
                                    ),
                                    html.Div(
                                        id="tree-params",
                                        children=[
                                            dcc.Input(
                                                id="tree-n",
                                                type="number",
                                                min=2,
                                                value=6,
                                                placeholder="Number of vertices",
                                                style={
                                                    "width": "100%",
                                                    "marginTop": "5px",
                                                },
                                            ),
                                            dcc.Input(
                                                id="tree-seed",
                                                type="number",
                                                value="0",
                                                placeholder="Random seed",
                                                style={
                                                    "width": "100%",
                                                    "marginTop": "5px",
                                                },
                                            ),
                                        ],
                                        style={"display": "none"},
                                    ),
                                    html.Div(
                                        id="grid-params",
                                        children=[
                                            dcc.Input(
                                                id="grid-rows",
                                                type="number",
                                                min=2,
                                                value=3,
                                                placeholder="Rows",
                                                style={
                                                    "width": "100%",
                                                    "marginTop": "5px",
                                                },
                                            ),
                                            dcc.Input(
                                                id="grid-cols",
                                                type="number",
                                                min=2,
                                                value=3,
                                                placeholder="Columns",
                                                style={
                                                    "width": "100%",
                                                    "marginTop": "5px",
                                                },
                                            ),
                                        ],
                                        style={"display": "none"},
                                    ),
                                    html.Button(
                                        "Generate Graph",
                                        id="generate-btn",
                                        style={
                                            "width": "100%",
                                            "marginTop": "10px",
                                            "padding": "10px",
                                            "backgroundColor": "#4CAF50",
                                            "color": "white",
                                            "border": "none",
                                            "borderRadius": "5px",
                                            "fontSize": "14px",
                                            "cursor": "pointer",
                                        },
                                    ),
                                    html.Hr(),
                                    html.Div(
                                        [
                                            dcc.Upload(
                                                id="upload-graph",
                                                children=html.Div(
                                                    [
                                                        "Drag and Drop or Click to select a Graph File"
                                                    ],
                                                    style={
                                                        "textAlign": "center",
                                                        "margin": "10px",
                                                    },
                                                ),
                                                multiple=False,
                                                style={
                                                    "border": "2px dashed #aaa",
                                                    "borderRadius": "5px",
                                                },
                                            ),
                                            html.Button(
                                                "Load Graph from File",
                                                id="load-file-btn",
                                                style={
                                                    "width": "100%",
                                                    "padding": "8px",
                                                    "backgroundColor": "#2196F3",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "5px",
                                                    "fontSize": "14px",
                                                    "cursor": "pointer",
                                                    "marginTop": "5px",
                                                },
                                            ),
                                        ]
                                    ),
                                    html.Hr(),
                                    html.Button(
                                        "Redraw",
                                        id="recompute-btn",
                                        style={
                                            "width": "100%",
                                            "padding": "8px",
                                            "backgroundColor": "#2196F3",
                                            "color": "white",
                                            "border": "none",
                                            "borderRadius": "5px",
                                            "fontSize": "14px",
                                            "cursor": "pointer",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Div(
                                        style={"marginBottom": "10px"},
                                        children=[
                                            html.Label(
                                                "Total Sellers (M)",
                                                style={
                                                    "fontSize": "14px",
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                            dcc.Input(
                                                id="M-input",
                                                type="number",
                                                value=game.M,
                                                min=1,
                                                style={
                                                    "width": "100%",
                                                    "padding": "8px",
                                                    "border": "1px solid #ccc",
                                                    "borderRadius": "4px",
                                                    "fontSize": "14px",
                                                },
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={"marginBottom": "10px"},
                                        children=[
                                            html.Label(
                                                "Base Cost",
                                                style={
                                                    "fontSize": "14px",
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                            dcc.Input(
                                                id="base-input",
                                                type="number",
                                                value=game.base_cost,
                                                step=0.1,
                                                style={
                                                    "width": "100%",
                                                    "padding": "8px",
                                                    "border": "1px solid #ccc",
                                                    "borderRadius": "4px",
                                                    "fontSize": "14px",
                                                },
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={"marginBottom": "10px"},
                                        children=[
                                            html.Label(
                                                "Solution Method",
                                                style={
                                                    "fontSize": "14px",
                                                    "fontWeight": "bold",
                                                },
                                            ),
                                            dcc.Dropdown(
                                                id="method-dropdown",
                                                options=[
                                                    {
                                                        "label": "Exact (Branch & Bound)",
                                                        "value": "exact",
                                                    },
                                                    {
                                                        "label": "Mirror Descent",
                                                        "value": "md",
                                                    },
                                                    {
                                                        "label": "Reinforcement Learning",
                                                        "value": "rl",
                                                    },
                                                ],
                                                value="exact",
                                                style={"marginTop": "5px"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="exact-params",
                                        children=[
                                            html.Div(
                                                style={"marginBottom": "10px"},
                                                children=[
                                                    html.Label(
                                                        "B&B Max Iterations",
                                                        style={
                                                            "fontSize": "14px",
                                                            "fontWeight": "bold",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="max-iter-input",
                                                        type="number",
                                                        min=10,
                                                        value=game.max_iter,
                                                        step=10,
                                                        style={
                                                            "width": "100%",
                                                            "padding": "8px",
                                                            "border": "1px solid #ccc",
                                                            "borderRadius": "4px",
                                                            "fontSize": "14px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"marginBottom": "10px"},
                                                children=[
                                                    html.Label(
                                                        "B&B Cache Size",
                                                        style={
                                                            "fontSize": "14px",
                                                            "fontWeight": "bold",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="cache-input",
                                                        type="number",
                                                        min=1000,
                                                        value=game.cache_size,
                                                        step=1000,
                                                        style={
                                                            "width": "100%",
                                                            "padding": "8px",
                                                            "border": "1px solid #ccc",
                                                            "borderRadius": "4px",
                                                            "fontSize": "14px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                        ],
                                        style={"display": "block"},
                                    ),
                                    html.Div(
                                        id="md-params",
                                        children=[
                                            html.Div(
                                                style={"marginBottom": "10px"},
                                                children=[
                                                    html.Label(
                                                        "MD Iterations (T)",
                                                        style={
                                                            "fontSize": "14px",
                                                            "fontWeight": "bold",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="md-T-input",
                                                        type="number",
                                                        min=10,
                                                        value=200,
                                                        step=10,
                                                        style={
                                                            "width": "100%",
                                                            "padding": "8px",
                                                            "border": "1px solid #ccc",
                                                            "borderRadius": "4px",
                                                            "fontSize": "14px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"marginBottom": "10px"},
                                                children=[
                                                    html.Label(
                                                        "MD Learning Rate (eta)",
                                                        style={
                                                            "fontSize": "14px",
                                                            "fontWeight": "bold",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="md-eta-input",
                                                        type="number",
                                                        value=0.2,
                                                        step=0.01,
                                                        style={
                                                            "width": "100%",
                                                            "padding": "8px",
                                                            "border": "1px solid #ccc",
                                                            "borderRadius": "4px",
                                                            "fontSize": "14px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"marginBottom": "10px"},
                                                children=[
                                                    html.Label(
                                                        "MD Entropy Penalty (beta)",
                                                        style={
                                                            "fontSize": "14px",
                                                            "fontWeight": "bold",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="md-beta-input",
                                                        type="number",
                                                        value=0.05,
                                                        step=0.01,
                                                        style={
                                                            "width": "100%",
                                                            "padding": "8px",
                                                            "border": "1px solid #ccc",
                                                            "borderRadius": "4px",
                                                            "fontSize": "14px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"marginBottom": "10px"},
                                                children=[
                                                    html.Label(
                                                        "MD Entropy Threshold (eps_H)",
                                                        style={
                                                            "fontSize": "14px",
                                                            "fontWeight": "bold",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="md-eps-H-input",
                                                        type="number",
                                                        value=0.05,
                                                        step=0.01,
                                                        style={
                                                            "width": "100%",
                                                            "padding": "8px",
                                                            "border": "1px solid #ccc",
                                                            "borderRadius": "4px",
                                                            "fontSize": "14px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                        ],
                                        style={"display": "none"},
                                    ),
                                    html.Div(
                                        id="rl-params",
                                        children=[
                                            html.Div(
                                                style={"marginBottom": "10px"},
                                                children=[
                                                    html.Label(
                                                        "RL Iterations (T)",
                                                        style={
                                                            "fontSize": "14px",
                                                            "fontWeight": "bold",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="rl-T-input",
                                                        type="number",
                                                        min=10,
                                                        value=200,
                                                        step=10,
                                                        style={
                                                            "width": "100%",
                                                            "padding": "8px",
                                                            "border": "1px solid #ccc",
                                                            "borderRadius": "4px",
                                                            "fontSize": "14px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"marginBottom": "10px"},
                                                children=[
                                                    html.Label(
                                                        "RL Learning Rate (eta)",
                                                        style={
                                                            "fontSize": "14px",
                                                            "fontWeight": "bold",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="rl-eta-input",
                                                        type="number",
                                                        value=0.1,
                                                        step=0.01,
                                                        style={
                                                            "width": "100%",
                                                            "padding": "8px",
                                                            "border": "1px solid #ccc",
                                                            "borderRadius": "4px",
                                                            "fontSize": "14px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"marginBottom": "10px"},
                                                children=[
                                                    html.Label(
                                                        "RL Entropy Weight",
                                                        style={
                                                            "fontSize": "14px",
                                                            "fontWeight": "bold",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="rl-entropy-weight-input",
                                                        type="number",
                                                        value=0.05,
                                                        step=0.01,
                                                        style={
                                                            "width": "100%",
                                                            "padding": "8px",
                                                            "border": "1px solid #ccc",
                                                            "borderRadius": "4px",
                                                            "fontSize": "14px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"marginBottom": "10px"},
                                                children=[
                                                    html.Label(
                                                        "RL Entropy Threshold (eps_H)",
                                                        style={
                                                            "fontSize": "14px",
                                                            "fontWeight": "bold",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="rl-eps-H-input",
                                                        type="number",
                                                        value=0.05,
                                                        step=0.01,
                                                        style={
                                                            "width": "100%",
                                                            "padding": "8px",
                                                            "border": "1px solid #ccc",
                                                            "borderRadius": "4px",
                                                            "fontSize": "14px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"marginBottom": "10px"},
                                                children=[
                                                    html.Label(
                                                        "RL Seed",
                                                        style={
                                                            "fontSize": "14px",
                                                            "fontWeight": "bold",
                                                        },
                                                    ),
                                                    dcc.Input(
                                                        id="rl-seed-input",
                                                        type="number",
                                                        value=42,
                                                        style={
                                                            "width": "100%",
                                                            "padding": "8px",
                                                            "border": "1px solid #ccc",
                                                            "borderRadius": "4px",
                                                            "fontSize": "14px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                        ],
                                        style={"display": "none"},
                                    ),
                                    html.Button(
                                        "Run Method",
                                        id="run-method-btn",
                                        style={
                                            "width": "100%",
                                            "padding": "10px",
                                            "backgroundColor": "#FF9800",
                                            "color": "white",
                                            "border": "none",
                                            "borderRadius": "5px",
                                            "fontSize": "14px",
                                            "cursor": "pointer",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Div(
                                        id="nash-status",
                                        style={
                                            "width": "100%",
                                            "padding": "10px",
                                            "borderRadius": "5px",
                                            "fontSize": "14px",
                                            "fontWeight": "bold",
                                            "textAlign": "center",
                                            "marginBottom": "10px",
                                            "backgroundColor": "#f0f0f0",
                                            "color": "#666",
                                        },
                                        children="Nash Equilibrium: Unknown",
                                    ),
                                    html.Button(
                                        "Reset",
                                        id="reset-btn",
                                        style={
                                            "width": "100%",
                                            "padding": "8px",
                                            "backgroundColor": "#F44336",
                                            "color": "white",
                                            "border": "none",
                                            "borderRadius": "5px",
                                            "fontSize": "14px",
                                            "cursor": "pointer",
                                        },
                                    ),
                                ],
                            ),
                            html.Div(
                                id="click-info",
                                style={
                                    "height": "30%",
                                    "backgroundColor": "#f5f5f5",
                                    "padding": "10px",
                                    "overflowY": "auto",
                                    "fontSize": "12px",
                                },
                            ),
                            html.Div(
                                style={"padding": "10px"},
                                children=[
                                    dcc.Checklist(
                                        id="show-labels-checkbox",
                                        options=[
                                            {"label": "Show Labels", "value": "show"}
                                        ],
                                        value=["show"],
                                    )
                                ],
                            ),
                        ],
                    ),
                    # Middle: Graph
                    html.Div(
                        style={
                            "flex": "1",
                            "padding": "10px",
                            "display": "flex",
                            "flexDirection": "column",
                        },
                        children=[
                            dcc.Graph(id="graph", figure=fig, style={"flex": "1"}),
                            html.Div(
                                id="convergence-container",
                                style={"display": "none", "marginTop": "10px"},
                                children=[
                                    html.Div(
                                        style={"height": "300px"},
                                        children=[
                                            dcc.Graph(
                                                id="convergence-graph",
                                                style={"height": "100%"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={"marginTop": "10px", "display": "flex"},
                                        children=[
                                            html.Div(
                                                style={"flex": "1"},
                                                children=[
                                                    html.Label("Iteration Step"),
                                                    dcc.Slider(
                                                        id="iteration-slider",
                                                        min=0,
                                                        max=999,  # Allow up to large T
                                                        value=199,
                                                        step=1,
                                                        marks={
                                                            0: "0",
                                                            200: "200",
                                                            400: "400",
                                                            600: "600",
                                                            800: "800",
                                                            999: "999",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={
                                                    "width": "200px",
                                                    "marginLeft": "20px",
                                                },
                                                children=[
                                                    html.Label("Filter Players"),
                                                    dcc.Checklist(
                                                        id="player-checklist",
                                                        options=[],  # Will be set dynamically
                                                        value=[],  # Default to all
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={"height": "600px", "marginTop": "10px"},
                                        children=[
                                            dcc.Graph(
                                                id="strategy-graph",
                                                style={"height": "100%"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={"height": "300px", "marginTop": "10px"},
                                        children=[
                                            dcc.Graph(
                                                id="strategy-bars",
                                                style={"height": "100%"},
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Right: Stats and Filters
                    html.Div(
                        style={
                            "width": f"{RIGHT_PANEL_WIDTH_PERCENT}%",
                            "padding": "10px",
                            "height": "100%",
                            "display": "flex",
                            "flexDirection": "column",
                        },
                        children=[
                            html.Pre(
                                id="stats-display",
                                children=stats_text,
                                style={
                                    "border": "1px solid #ddd",
                                    "padding": "5px",
                                    "flex": "1",
                                    "overflowY": "auto",
                                },
                            ),
                        ],
                    ),
                ],
            ),
        ]
    )
