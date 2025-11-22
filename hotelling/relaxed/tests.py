# hotelling_lib/tests.py

import graph_tool.all as gt
import numpy as np
from hotelling.relaxed import (
    run_mirror_descent,
    run_rl,
    compare_modules,
)
from math import log2


def cycle_graph(n):
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    for i in range(n):
        g.add_edge(i, (i + 1) % n)
    return g


def balanced_tree(r, h):
    g = gt.Graph(directed=False)
    if h < 0:
        return g
    total_nodes = sum(r**i for i in range(h + 1))
    g.add_vertex(total_nodes)

    current_level_start = 0
    for level in range(h):
        nodes_in_level = r**level
        next_level_start = current_level_start + nodes_in_level
        child_index = next_level_start
        for parent in range(current_level_start, next_level_start):
            for _ in range(r):
                if child_index < total_nodes:
                    g.add_edge(parent, child_index)
                    child_index += 1
        current_level_start = next_level_start
    return g


def star_graph(n):
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    center = 0
    for i in range(1, n):
        g.add_edge(center, i)
    return g


def erdos_renyi_graph(n, p, seed=42):
    np.random.seed(seed)
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < p:
                g.add_edge(i, j)
    return g


def generate_test_graphs(size_limit=10):
    graphs = {}
    # 1. Цикл
    graphs["Cycle"] = cycle_graph(size_limit)
    # 2. Полный граф
    graphs["Complete"] = gt.complete_graph(size_limit)
    # 3. Звезда
    graphs["Star"] = star_graph(size_limit)
    # 4. Дерево
    graphs["BalancedTree"] = balanced_tree(r=2, h=int(log2(size_limit)))
    # 5. Случайный граф Эрдёша–Реньи
    graphs["ErdosRenyi"] = erdos_renyi_graph(size_limit, 0.3, seed=42)
    return graphs


def test_on_graphs(m=2, T=200, size_limit=10):
    graphs = generate_test_graphs(size_limit=size_limit)
    for name, G in graphs.items():
        print(f"\n=== Тест на графе {name} (|V|={G.num_vertices()}) ===")
        dist_u_v = gt.shortest_distance(G).get_2d_array(range(G.num_vertices()))
        d_u = np.ones(G.num_vertices()) / G.num_vertices()

        results = compare_modules(
            G,
            dist_u_v,
            d_u,
            runners=[
                lambda G, D, U: run_mirror_descent(G, D, U, m=m, T=T),
                lambda G, D, U: run_rl(G, D, U, m=m, T=T),
            ],
            labels=["Mirror Descent", "RL"],
        )
