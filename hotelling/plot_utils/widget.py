# pip install plotly ipywidgets
import numpy as np
import plotly.graph_objects as go
import ipywidgets as W
from IPython.display import display, clear_output
from graph_tool.draw import sfdp_layout

from hotelling.algorithms.branch_bound import BBTree, Node
from hotelling.hotelling_game.cost_functions import BaseRevenueFunction
from hotelling.hotelling_game.game_evaluation import evaluate_sellers


def gt_export_layout(g, pos_prop=None):
    if pos_prop is None:
        pos_prop = sfdp_layout(g)
    node_ids = [int(v) for v in g.vertices()]
    # быстрее индексировать по int->index через массив:
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    xy = np.empty((len(node_ids), 2), float)
    for nid, i in id2idx.items():
        xy[i] = pos_prop[g.vertex(nid)].a
    edges = [(int(e.source()), int(e.target())) for e in g.edges()]
    return node_ids, xy, edges, id2idx, pos_prop


def gt_build_plotly(fig, node_ids, xy, edges, labels, colors, sizes, show_labels=True):
    # edges
    ex, ey = [], []
    # используем id2idx для O(1) доступа:
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    for u, v in edges:
        i0, i1 = id2idx[u], id2idx[v]
        x0, y0 = xy[i0]
        x1, y1 = xy[i1]
        ex += [x0, x1, None]
        ey += [y0, y1, None]
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
    # nodes

    labels_html = [s.replace("\n", "<br>") for s in labels]
    fig.add_trace(
        go.Scatter(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="markers+text",
            marker=dict(color=colors, size=sizes, line=dict(color="#263238", width=1)),
            text=labels_html,
            textposition="top center",
            hovertemplate="%{text}",
            showlegend=False,
        )
    )

    fig.data[1].customdata = np.array(node_ids)
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=700,
        clickmode="event+select",
    )
    # подсветка выбранной точки
    # fig.update_traces(marker=dict(line=dict(size=2, color="#000")), selector=1)
    return fig


def make_labels_colors_sizes(g, positions: dict, sellers_set, node_ids):
    nearest_set = positions.get("nearest_set")
    num_sellers = positions.get(
        "num_sellers", {}
    )  # может быть {Vertex:...} или {int:...}
    revenues = positions.get("revenues", {})

    # нормализуем ключи в int
    def k2i(k):
        return int(k) if not isinstance(k, int) else k

    ns_int = {k2i(k): int(v) for k, v in num_sellers.items()}
    rv_int = {k2i(k): float(v) for k, v in revenues.items()}

    labels, colors, sizes = [], [], []
    for nid in node_ids:
        if nid in sellers_set:
            k = ns_int.get(nid, 1)
            rev = rv_int.get(nid, 0.0)
            rps = rev / max(k, 1)
            labels.append(f"id:{nid}\n  #:{k}\n  $:{rps:.2f}")
            colors.append("#e67e22")
            sizes.append(16)
        else:
            if nearest_set is not None:
                v = g.vertex(nid)
                nb = [int(u) for u in nearest_set[v]]
            else:
                nb = []
            labels.append(f"id:{nid}\n {nb}")
            colors.append("#546e7a")
            sizes.append(12)
    return labels, colors, sizes


def compute_min_revenue(positions2, sellers_set):
    """
    Возвращает (min_revenue_per_seller, details_str).
    positions2: ожидается, что содержит 'num_sellers' и 'revenues'.
    sellers_set: множество id узлов, отмеченных как продавцы.
    """
    num_sellers = positions2.get("num_sellers", {})
    revenues = positions2.get("revenues", {})

    # Нормализуем ключи: могут быть Vertex
    def k2i(k):
        return int(k) if not isinstance(k, int) else k

    ns_int = {k2i(k): int(v) for k, v in num_sellers.items()}
    rv_int = {k2i(k): float(v) for k, v in revenues.items()}

    per_seller = []
    lines = []
    for nid in sorted(sellers_set):
        k = ns_int.get(nid, 0)
        rev = rv_int.get(nid, 0.0)
        rps = rev / max(k, 1) if k > 0 else 0.0
        per_seller.append(rps)
        lines.append(f"node {nid}: sellers={k}, revenue={rev:.6g}, $/seller={rps:.6g}")

    if per_seller:
        m = min(per_seller)
        header = f"<b>Min $/seller:</b> {m:.6g}"
    else:
        m = 0.0
        header = "<b>Min $/seller:</b> — (no sellers)"
    details = "<br>".join(lines) if lines else "No seller nodes selected."
    html = f"{header}<br><br>{details}"
    return m, html


# --- Панель в ноутбуке ---
def panel_gt_interactive(g, revenue_function: BaseRevenueFunction, pos_prop=None):
    # состояние
    node_ids, xy, edges, id2idx, pos_prop = gt_export_layout(g, pos_prop)
    sellers_set = set()
    current_M = 1
    max_iterations = 400
    show_labels = True
    base_cost = 10
    revenue_function.base_cost = base_cost
    # виджеты
    out_plot = W.Output()
    out_stats = W.Output()  # правая панель статистики

    btn_reset = W.Button(description="Reset")
    btn_equil = W.Button(description="Find equilibrium", button_style="primary")
    num_M = W.BoundedIntText(value=current_M, min=1, max=10**9, step=1, description="M")
    max_iter_btn = W.BoundedIntText(
        value=max_iterations, min=1, max=10**6, step=1, description="Max iters"
    )
    base_cost_btn = W.BoundedFloatText(
        value=base_cost, min=0, max=10**6, step=1, description="Base cost C"
    )
    cb_labels = W.Checkbox(value=True, description="Show labels")

    info = W.HTML(value="Click a node to toggle seller.")
    info_box = W.HTML(  # обертка с рамкой/фоном по желанию
        value=f"<div style='font-family:monospace; font-size:13px;'>{info.value}</div>"
    )

    stats_header = W.HTML(value="<h3>Seller stats</h3>")

    def render():
        if len(sellers_set) > current_M:
            # update num_M
            s = len(sellers_set)
            num_M.value = s

        positions2 = evaluate_sellers(
            g, sellers_set, current_M, "weight", revenue_function, extended_return=True
        )
        assert isinstance(positions2, dict)
        labels, colors, sizes = make_labels_colors_sizes(
            g, positions2, sellers_set, node_ids
        )
        fig = go.Figure()
        gt_build_plotly(
            fig, node_ids, xy, edges, labels, colors, sizes, show_labels=show_labels
        )
        with out_plot:
            clear_output(wait=True)
            fw = go.FigureWidget(fig)
            scatter = fw.data[1]

            def on_click(trace, points, state):
                if not points.point_inds:
                    return
                idx = points.point_inds[0]
                v_id = int(scatter.customdata[idx])
                if v_id in sellers_set:
                    sellers_set.remove(v_id)
                else:
                    sellers_set.add(v_id)
                render()

            scatter.on_click(on_click)
            display(fw)

        # обновить правую панель (минимальный заработок продавца)
        _, html = compute_min_revenue(positions2, sellers_set)
        with out_stats:
            clear_output(wait=True)
            # делаем «большое» окно: можно обернуть в Box с фиксированной высотой и прокруткой
            box = W.HTML(
                value=f"<div style='font-family:monospace; font-size:14px;'>{html}</div>"
            )
            display(box)

    def on_reset(_):
        nonlocal node_ids, xy, edges, id2idx, pos_prop, sellers_set
        sellers_set.clear()
        # пересчитать раскладку, чтобы «поменялись позиции»
        pos_prop = sfdp_layout(g)
        node_ids, xy, edges, id2idx, pos_prop = gt_export_layout(g, pos_prop)
        render()

    def on_equilibrium(_):
        btn_equil.disabled = True
        info.value = "Running B&B..."
        try:
            bb_tree = BBTree(g, current_M, revenue_function)
            bb_tree.run(max_iterations=max_iterations)
            # bb_tree.run_stat
            sellers_set.clear()
            sellers_set.update(int(x) for x in bb_tree.occupation)
            # best = run_branch_and_bound_equilibrium(g, current_M, revenue_function)  # реализуйте
            # sellers_set.clear(); sellers_set.update(int(x) for x in best)
        finally:
            btn_equil.disabled = False
            info.value = "Done."
            render()

    def on_M_change(change):
        nonlocal current_M
        current_M = int(change["new"])
        render()

    def on_labels_change(change):
        nonlocal show_labels
        show_labels = bool(change["new"])
        render()

    def on_max_iter_change(change):
        nonlocal max_iterations
        max_iterations = int(change["new"])
        render()

    def on_base_cost_change(change):
        nonlocal base_cost, revenue_function
        base_cost = float(change["new"])
        revenue_function.base_cost = base_cost
        render()

    # wire
    btn_reset.on_click(on_reset)
    btn_equil.on_click(on_equilibrium)
    num_M.observe(on_M_change, names="value")
    max_iter_btn.observe(on_max_iter_change, names="value")
    cb_labels.observe(on_labels_change, names="value")
    base_cost_btn.observe(on_base_cost_change, names="value")

    # нужно добавить info
    controls = W.HBox(
        [btn_reset, btn_equil, num_M, max_iter_btn, base_cost_btn, cb_labels]
    )
    # Сделаем правое окно крупным: фиксированная ширина и прокрутка
    stats_box = W.VBox(
        [stats_header, out_stats],
        layout=W.Layout(width="420px", height="720px", overflow="auto"),
    )
    left_box = W.VBox([controls, out_plot], layout=W.Layout(flex="1 1 auto"))
    ui = W.HBox([left_box, stats_box], layout=W.Layout(width="100%"))
    # display(W.VBox([controls, out_plot]))
    # первый рендер
    render()
    display(ui)
