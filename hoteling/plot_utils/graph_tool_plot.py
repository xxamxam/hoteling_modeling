from graph_tool.draw import graph_draw, sfdp_layout

def plot_sellers(positions: dict):

    nearest_set = positions["nearest_set"]
    num_sellers = positions["num_sellers"]   # dict: Vertex -> int/float
    revenues    = positions["revenues"]      # dict: Vertex -> float

    # Подписи
    v_label = g.new_vertex_property("string")
    for v in g.vertices():
        if int(v) in num_sellers:
            print(v)
            k = int(num_sellers[v])
            rev = float(revenues[v])
            rps = rev / max(k, 1)
            v_label[v] = f"{int(v)}:\n#: {k}\n$: {rps:.2f}"
        else:
            nb = [int(u) for u in nearest_set[v]]
            v_label[v] = f"{int(v)}:\n{nb}"

    # Цвета
    is_seller = g.new_vertex_property("bool")
    v_color   = g.new_vertex_property("string")
    for v in g.vertices():
        flag = bool(int(v) in num_sellers)
        is_seller[v] = flag
        v_color[v]   = "#e67e22" if flag else "#546e7a"   # оранжевый / стальной

    # Размеры под текст
    v_size = g.new_vertex_property("double")
    for v in g.vertices():
        lines = v_label[v].split("\n")
        v_size[v] = 6 + 2.0 * max(len(t) for t in lines) + 5.0 * (len(lines) - 1)

    # Позиции
    pos = positions.get("pos")
    if pos is None:
        pos = sfdp_layout(g)

    # Рисуем
    graph_draw(
        g,
        pos=pos,
        vertex_text=v_label,
        vertex_fill_color=v_color,
        # vertex_text_position=-1,
        vertex_size=v_size,
        vertex_font_size=12,
        text_wrap=True,         # важный параметр для переноса
        text_wrap_width=18,     # подберите под ваш шрифт/масштаб
        output_size=(1000, 800),
        output="graph_with_ids.png",
    )
