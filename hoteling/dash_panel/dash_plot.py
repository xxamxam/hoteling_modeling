# dash_plot.py - Plotting functions for Dash app

import numpy as np
import plotly.graph_objects as go

from graph_tool.draw import sfdp_layout


# Global for layout persistence
chart_pos_prop = None
chart_node_ids = None
chart_xy = None
chart_edges = None
current_g_vertices = 0


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

    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#9e9e9e", width=1), hoverinfo="skip", showlegend=False))

    fig.add_trace(go.Scatter(
        x=[xy[i][0] for i in range(len(node_ids))],
        y=[xy[i][1] for i in range(len(node_ids))],
        mode="markers+text",
        marker=dict(color=colors, size=sizes),
        text=labels,
        textposition="top center",
        customdata=node_ids,
        hovertemplate="%{text}",
    ))

    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=700)
    return fig


def make_labels_colors_sizes(g, positions, sellers_set, node_ids):
    num_sellers = positions.get("num_sellers", {})
    revenues = positions.get("revenues", {})

    labels, colors, sizes = [], [], []
    for nid in node_ids:
        if nid in sellers_set:
            k = num_sellers.get(nid, 1)
            rev = revenues.get(nid, 0.0)
            labels.append(f"id:{nid}<br>k:{k}<br>${rev:.2f}")
            colors.append("#e67e22")
            sizes.append(18)
        else:
            labels.append(f"id:{nid}")
            colors.append("#546e7a")
            sizes.append(14)
    return labels, colors, sizes


def make_figure(g, sellers_set, M=5, rf=None):
    global chart_pos_prop, chart_node_ids, chart_xy, chart_edges, current_g_vertices

    if rf is None:
        from hoteling.branch_bound import BaseRevenueFunction
        rf = BaseRevenueFunction()

    # Always compute layout since fixed per g; force recalc if g changed or none
    if chart_pos_prop is None or g.num_vertices() != current_g_vertices:
        chart_node_ids, chart_xy, chart_edges, id2idx, chart_pos_prop = gt_export_layout(g)
        current_g_vertices = g.num_vertices()
    else:
        # Reuse
        pass

    # Positions
    from hoteling.branch_bound import Node
    positions = Node.get_positions(g, sellers_set, M, "weight", rf, extended_return=True)

    # Labels etc.
    labels, colors, sizes = make_labels_colors_sizes(g, positions, sellers_set, chart_node_ids)
    fig = go.Figure()
    gt_build_plotly(fig, chart_node_ids, chart_xy, chart_edges, labels, colors, sizes)
    return fig
