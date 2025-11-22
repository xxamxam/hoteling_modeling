# dash_plot.py - Plotting functions for Dash app

import numpy as np
import plotly.graph_objects as go

from graph_tool.draw import sfdp_layout

from hotelling.hotelling_game.game_evaluation import evaluate_sellers
from hotelling.algorithms.branch_bound import BaseRevenueFunction

# Global for layout persistence
chart_pos_prop = None
chart_node_ids = None
chart_xy = None
chart_edges = None
current_g_vertices = -1


def gt_export_layout(g, pos_prop=None):
    if pos_prop is None:
        pos_prop = sfdp_layout(g)
    node_ids = [int(v) for v in g.vertices()]
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    xy = np.empty((len(node_ids), 2), float)
    for nid, i in id2idx.items():
        xy[i] = pos_prop[g.vertex(nid)].a
    edges = [(int(e.source()), int(e.target())) for e in g.edges()]
    return node_ids, xy, edges, id2idx, pos_prop


def gt_build_plotly(fig, node_ids, xy, edges, labels, colors, sizes):
    ex, ey = [], []
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    for u, v in edges:
        i0, i1 = id2idx[u], id2idx[v]
        ex += [xy[i0][0], xy[i1][0], None]
        ey += [xy[i0][1], xy[i1][1], None]

    fig.add_trace(
        go.Scatter(
            x=ex,
            y=ey,
            mode="lines",
            line=dict(color="#9e9e9e", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[xy[i][0] for i in range(len(node_ids))],
            y=[xy[i][1] for i in range(len(node_ids))],
            mode="markers+text",
            marker=dict(color=colors, size=sizes),
            text=labels,
            textposition="top center",
            customdata=node_ids,
            hovertemplate="%{text}",
        )
    )

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=10),
    )
    return fig


def make_labels_colors_sizes(g, positions, sellers_set, node_ids, show_labels=True):
    num_sellers = positions.get("num_sellers", {})
    revenues = positions.get("revenues", {})

    labels, colors, sizes = [], [], []
    for nid in node_ids:
        if nid in sellers_set:
            k = num_sellers.get(nid, 1)
            rev = revenues.get(nid, 0.0)
            labels.append(f"id:{nid}<br>k:{k}<br>$/k:{rev / k:.2f}")
            colors.append("#e67e22")
            sizes.append(18)
        else:
            labels.append(f"id:{nid}" if show_labels else "")
            colors.append("#546e7a")
            sizes.append(14)
    return labels, colors, sizes


def make_strategy_figure(
    g, strategies, step, show_labels=True, redraw: int = 0, selected_players=None
):
    global chart_pos_prop, chart_node_ids, chart_xy, chart_edges, current_g_vertices

    # Always compute layout since fixed per g; force recalc if g changed or none
    if chart_pos_prop is None or id(g) != current_g_vertices or redraw:
        chart_node_ids, chart_xy, chart_edges, id2idx, chart_pos_prop = (
            gt_export_layout(g)
        )
        current_g_vertices = id(g)
    else:
        # Reuse
        pass

    fig = go.Figure()

    # Edges
    ex, ey = [], []
    id2idx = {nid: i for i, nid in enumerate(chart_node_ids)}
    for u, v in chart_edges:
        i0, i1 = id2idx[u], id2idx[v]
        ex += [chart_xy[i0][0], chart_xy[i1][0], None]
        ey += [chart_xy[i0][1], chart_xy[i1][1], None]

    fig.add_trace(
        go.Scatter(
            x=ex,
            y=ey,
            mode="lines",
            line=dict(color="#9e9e9e", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Get strategies at step
    num_players = len(strategies)
    effective_step = min(step, len(strategies[0]) - 1) if strategies else 0
    probs_at_step = [strategy_log[effective_step] for strategy_log in strategies]

    # For each node, find the player with max prob and the prob
    node_colors = []
    node_sizes = []
    labels = []
    for nid in chart_node_ids:
        player_probs = [probs[nid] for probs in probs_at_step]
        if selected_players:
            # Filter to selected players
            filtered_probs = [
                player_probs[i] for i in selected_players if i < len(player_probs)
            ]
            if filtered_probs:
                max_prob = max(filtered_probs)
                max_player = selected_players[filtered_probs.index(max_prob)]
            else:
                max_prob = 0
                max_player = 0
        else:
            max_prob = max(player_probs)
            max_player = player_probs.index(max_prob)
        # Color based on max player
        base_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        base = base_colors[max_player % len(base_colors)]
        intensity = int(max_prob * 255)
        color = f"rgb({base[0] * intensity // 255}, {base[1] * intensity // 255}, {base[2] * intensity // 255})"
        node_colors.append(color)
        node_sizes.append(14 + max_prob * 20)
        if show_labels:
            prob_text = "<br>".join(
                [f"P{i + 1}: {p:.2f}" for i, p in enumerate(player_probs)]
            )
            labels.append(f"id:{nid}<br>{prob_text}")
        else:
            labels.append("")

    fig.add_trace(
        go.Scatter(
            x=[chart_xy[id2idx[nid]][0] for nid in chart_node_ids],
            y=[chart_xy[id2idx[nid]][1] for nid in chart_node_ids],
            mode="markers+text",
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line=dict(width=1, color="black"),
            ),
            text=labels,
            textposition="top center",
            customdata=chart_node_ids,
            hovertemplate="%{text}",
            showlegend=False,
        )
    )

    # Add legend traces for reference
    for i in range(num_players):
        base = base_colors[i % len(base_colors)]
        color = f"rgb({base[0]}, {base[1]}, {base[2]})"
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=color, size=10),
                name=f"Player {i + 1}",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=f"Stochastic Strategies at Step {step}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=40, b=10),
    )
    return fig


def make_figure(g, sellers_set, M=5, rf=None, show_labels=True, redraw: int = 0):
    global chart_pos_prop, chart_node_ids, chart_xy, chart_edges, current_g_vertices

    if rf is None:
        rf = BaseRevenueFunction()
    # Always compute layout since fixed per g; force recalc if g changed or none
    if chart_pos_prop is None or id(g) != current_g_vertices or redraw:
        chart_node_ids, chart_xy, chart_edges, id2idx, chart_pos_prop = (
            gt_export_layout(g)
        )
        current_g_vertices = id(g)
    else:
        # Reuse
        pass

    # Positions
    positions = evaluate_sellers(g, sellers_set, M, "weight", rf, extended_return=True)

    # Labels etc.
    labels, colors, sizes = make_labels_colors_sizes(
        g, positions, sellers_set, chart_node_ids, show_labels
    )
    fig = go.Figure()
    gt_build_plotly(fig, chart_node_ids, chart_xy, chart_edges, labels, colors, sizes)
    return fig
