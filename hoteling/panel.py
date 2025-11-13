# pip install gradio plotly networkx
import gradio as gr
import plotly.graph_objects as go
import networkx as nx
import numpy as np

# ---- ВАШИ зависимости/функции ----
# g: ваш граф; pos: словарь {node_id: (x,y)} фиксированных координат
# Node.get_positions(...) и revenue_function(...) должны быть доступны

def build_positions(g, sellers_pos, M, revenue_function):
    # Обёртка вокруг Node.get_positions с extended_return=True
    return Node.get_positions(g, sellers_pos, M, "weight", revenue_function, extended_return=True)

def make_figure(g, pos, positions2, sellers_pos):
    # Plotly: рисуем рёбра и узлы
    x_nodes = []; y_nodes = []; labels = []; colors = []; sizes = []; ids = []
    num_sellers = positions2.get("num_sellers", {})
    revenues    = positions2.get("revenues", {})
    nearest_set = positions2.get("nearest_set", None)

    # Узлы
    for v in g.nodes():
        x, y = pos[v]
        ids.append(v)
        x_nodes.append(x); y_nodes.append(y)
        if v in sellers_pos:
            k   = int(num_sellers.get(v, 0))
            rev = float(revenues.get(v, 0.0))
            rps = rev / max(k, 1) if k > 0 else 0.0
            labels.append(f"id={v}<br>#sellers={k}<br>$/seller={rps:.2f}")
            colors.append("#e67e22")
            sizes.append(18)
        else:
            nb = [int(u) for u in (nearest_set[v] if nearest_set is not None else [])]
            labels.append(f"id={v}<br>nearest={nb}")
            colors.append("#546e7a")
            sizes.append(14)

    # Рёбра
    edge_x = []; edge_y = []
    for u, w in g.edges():
        x0, y0 = pos[u]; x1, y1 = pos[w]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="#9e9e9e", width=1), hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes, mode="markers", marker=dict(color=colors, size=sizes, line=dict(color="#263238", width=1)),
        text=labels, hovertemplate="%{text}", showlegend=False
    ))
    fig.update_layout(
        dragmode="lasso",  # позволит прямоугольное/лассо-выделение
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10), height=700
    )
    # Сохраняем ids и (x,y) в customdata для обратного маппинга индексов
    fig.data[1].customdata = np.array(ids)
    return fig

# ---- Gradio callbacks ----
def init_app():
    # Предполагаем, что у вас есть фиксированный pos (чтобы граф не "скакал")
    # Если нет, посчитайте один раз ранее (например, sfdp_layout из graph_tool или nx.spring_layout)
    sellers_pos = set()
    positions2 = build_positions(g, sellers_pos, M, revenue_function)
    fig = make_figure(g, pos, positions2, sellers_pos)
    return fig, list(sellers_pos)

def on_select(selection: gr.SelectData, sellers,  # sellers: список id текущих продавцов
              ):
    # selection.index для ScatterPlot даёт индексы выбранных точек; для Plot может возвращать bbox — поддержим оба случая
    touched = set()
    if hasattr(selection, "index") and selection.index is not None:
        # Может быть int, tuple или список — нормализуем
        idx = selection.index
        if isinstance(idx, (list, tuple, np.ndarray)):
            for i in idx:
                if isinstance(i, (list, tuple)) and len(i) == 2:
                    # диапазон (min,max) для линейного графика — игнорируем
                    continue
                touched.add(int(i))
        elif isinstance(idx, (int, np.integer)):
            touched.add(int(idx))
    # Если select вернул value с подмножеством точек
    if hasattr(selection, "value") and selection.value is not None and isinstance(selection.value, list):
        # Можно попытаться разобрать value как x-координаты — но надёжнее опираться на index
        pass

    # Преобразуем индексы точек (в порядке узлов) в node_id через customdata недоступно напрямую из Gradio;
    # поэтому просто считаем, что порядок узлов стабилен и index совпадает с порядком добавления.
    # Альтернатива: хранить map index->node_id в gr.State.
    # Для надёжности держим map_index_to_id в state.
    new_sellers = set(sellers)
    for i in touched:
        # защита
        if i < 0 or i >= len(node_order):
            continue
        v = node_order[i]
        if v in new_sellers: new_sellers.remove(v)
        else: new_sellers.add(v)

    positions2 = build_positions(g, new_sellers, M, revenue_function)
    fig = make_figure(g, pos, positions2, new_sellers)
    return fig, list(new_sellers)

def on_equilibrium(sellers):
    # Запускаем ваш B&B для нахождения равновесного множества (реализуйте функцию)
    best_set = run_branch_and_bound_equilibrium(g, M, revenue_function)
    best_sellers = set(best_set)
    positions2 = build_positions(g, best_sellers, M, revenue_function)
    fig = make_figure(g, pos, positions2, best_sellers)
    return fig, list(best_sellers)

# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("Interactive graph: click/drag to toggle sellers; use Find equilibrium to run B&B.")
    plot = gr.Plot()
    sellers_state = gr.State([])
    # Сохраним стабильный порядок узлов для маппинга index->id
    node_order = list(g.nodes())  # доступен в замыкании

    init_btn = gr.Button("Init / Reset")
    init_btn.click(fn=init_app, inputs=None, outputs=[plot, sellers_state])

    # .select: доступен у gr.Plot/gr.ScatterPlot; обработчик получает gr.SelectData
    plot.select(fn=on_select, inputs=[sellers_state], outputs=[plot, sellers_state])

    eq_btn = gr.Button("Find equilibrium")
    eq_btn.click(fn=on_equilibrium, inputs=[sellers_state], outputs=[plot, sellers_state])

# ---- Запуск сервера ----
port = 7860
print(f"Starting interface on http://0.0.0.0:{port}")
demo.launch(server_name="0.0.0.0", server_port=port)

