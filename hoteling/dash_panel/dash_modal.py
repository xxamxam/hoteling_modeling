# dash_modal.py - Modal components for the Dash app

from dash import html, dcc


def create_bb_progress_modal():
    """
    Create the B&B progress modal/floating div.
    """
    return html.Div(
        id="bb-progress-modal",
        children=[
            html.H4("Branch & Bound Progress", style={"margin": "5px"}),
            dcc.Graph(id="bb-progress-graph", style={"height": "calc(100% - 60px)", "width": "calc(100% - 10px)"}),
            html.Button("Close", id="bb-stop-btn", style={"position": "absolute", "top": "5px", "right": "5px"}),
        ],
        style={
            "position": "fixed",
            "top": "15%",
            "left": "15%",
            "width": "70%",
            "height": "70%",
            "backgroundColor": "white",
            "border": "1px solid black",
            "zIndex": 1000,
            "padding": "10px",
            "display": "none"
        }
    )


def create_graph_placeholder():
    """
    Placeholder for graph in modal, to be replaced by dcc.Graph when needed.
    """
    pass
