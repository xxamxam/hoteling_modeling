# Requires: graph-tool (C++ backend)
import math
from collections import defaultdict
from typing import Dict, Union

from graph_tool import Graph
import numpy as np

# import graph_tool as gt
import graph_tool.search as gts

from typing import Dict

from hotelling.hotelling_game.cost_functions import BaseRevenueFunction


def nearest_source_labels(g: Graph, weight_ep, sources, source_ids=None, eps=1e-12):
    """
    g: graph_tool.Graph
    weight_ep: EdgePropertyMap('double') with nonnegative weights
    sources: list[gt.Vertex] — реальные источники
    source_ids: optional list[int] — стабильные ID источников (для tie-break)
    eps: float — допуск сравнения расстояний

    Returns:
        dist_vp: VertexPropertyMap('double') — расстояния до ближайших источников
        nearest_vp: VertexPropertyMap('int64_t') — одна метка ближайшего источника (min sid при ничьей)
        near_set_vp: VertexPropertyMap('object') — set[int] из всех ближайших источников (для деления спроса)
    """
    # Карты расстояний и множеств
    dist_vp = g.new_vertex_property("double", val=math.inf)
    near_set_vp = g.new_vertex_property("object")
    for v in g.vertices():
        near_set_vp[v] = set()

    # Метки источников
    if source_ids is None:
        source_ids = list(range(len(sources)))
    sid_of = {s: sid for s, sid in zip(sources, source_ids)}

    # Добавляем супер-источник s* и нулевые рёбра к источникам
    s_star = g.add_vertex()
    zero_edges = []
    for s in sources:
        e = g.add_edge(s_star, s)
        weight_ep[e] = 0.0
        zero_edges.append(e)

    # Инициализация
    dist_vp[s_star] = 0.0
    near_set_vp[s_star] = set()

    class NearestSetVisitor(gts.DijkstraVisitor):
        def __init__(self, dist_vp, near_set_vp, eps):
            self.dist = dist_vp
            self.near = near_set_vp
            self.eps = eps

        def edge_relaxed(self, e):
            # Строгое улучшение: перезаписываем множество ближайших
            u = e.source()
            v = e.target()
            if u == s_star:
                self.near[v] = {sid_of[v]}  # s* -> источник v
            else:
                self.near[v] = set(self.near[u])  # наследуем множество из u

        def edge_not_relaxed(self, e):
            # Ничья: cand == dist[v] в пределах eps -> объединяем множества
            u = e.source()
            v = e.target()
            cand = self.dist[u] + weight_ep[e]
            dv = self.dist[v]
            if abs(cand - dv) <= self.eps:
                if u == s_star:
                    self.near[v].add(sid_of[v])
                else:
                    self.near[v] |= self.near[u]

        def examine_vertex(self, u):
            if u == s_star:
                self.dist[u] = 0.0

    # Запуск Дейкстры из s*
    gts.dijkstra_search(
        g,
        weight_ep,
        source=s_star,
        visitor=NearestSetVisitor(dist_vp, near_set_vp, eps),
        dist_map=dist_vp,
    )

    # Удаляем временные элементы
    for e in zero_edges:
        g.remove_edge(e)
    g.remove_vertex(s_star)

    # Одна детерминированная метка по min sid
    nearest_vp = g.new_vertex_property("int64_t", val=-1)
    for v in g.vertices():
        if near_set_vp[v]:
            nearest_vp[v] = min(near_set_vp[v])

    return dist_vp, nearest_vp, near_set_vp


def get_revenue(g, distances, nearest_set, revenue_function: BaseRevenueFunction):
    vertice_revenue = defaultdict(float)
    for v in g.vertices():
        v = int(v)
        nearest_sellers = nearest_set[v]

        dist = distances[v]
        vertex_value = revenue_function(dist)

        num_sellers = len(nearest_sellers)
        for seller in nearest_sellers:
            vertice_revenue[seller] += vertex_value / num_sellers
    return vertice_revenue


def bin_search(revenues: Dict, M, tol=1e-10):
    """
    param:
    revenues: Dict[Vertice, float]
    M: number of agents
    """
    revenue_values = np.array(list(revenues.values()))

    def f(lambd):
        return np.floor(revenue_values // lambd).sum() - M

    assert M >= len(revenue_values)
    left = 1e-12
    right = (
        np.min(revenue_values) + tol
    )  # at least one agent per selling position -- constrain
    if M == len(revenue_values):
        return {k: 1 for k in revenues.keys()}, right

    if f(right) < 0:
        while right - left > tol:
            mid = (right + left) / 2
            f_val = f(mid)
            if f_val >= 0:
                left = mid
            else:
                right = mid
    lambd = right

    # setting up sellers
    num_sellers = {k: 1 for k, v in revenues.items()}
    num_setted_sellers = sum(num_sellers.values())
    assert num_setted_sellers <= M
    if num_setted_sellers < M:
        remain_budgets = {
            k: revenues[k] - num_sellers[k] * lambd for k in revenues.keys()
        }
        while num_setted_sellers < M:
            pos = max(remain_budgets.items(), key=lambda x: x[1])[0]
            num_setted_sellers += 1
            num_sellers[pos] += 1
            remain_budgets[pos] -= lambd
    assert sum(num_sellers.values()) == M, (
        f"error. Get {num_sellers=}, while \
        {sum(num_sellers.values())} != {M}.\n{lambd=}  {revenues}"
    )
    # max_rev = max(revenues[k]/num_sellers[k] for k in revenues.keys())
    # delta = max_rev - lambd
    return num_sellers, lambd


def evaluate_sellers(
    graph,
    occupied_vertices,
    M,
    weights_name,
    revenue_function,
    tol=1e-12,
    extended_return=False,
) -> Union[float, Dict]:
    if len(occupied_vertices) == 0:
        if extended_return:
            return {}
        else:
            return 0.0

    weight_ep = graph.ep[weights_name]
    sources = [graph.vertex(v) for v in occupied_vertices]

    # compute nearest sellers for buyers
    distances, _, nearest_set = nearest_source_labels(
        graph, weight_ep, sources, source_ids=list(occupied_vertices)
    )

    # compute revenues per Vertice
    revenues = get_revenue(graph, distances, nearest_set, revenue_function)

    # find minimal revenue per seller on graph
    num_sellers, lambd = bin_search(revenues, M, tol=tol)
    if extended_return:
        return {
            "num_sellers": num_sellers,
            "lambd": lambd,
            "nearest_set": nearest_set,
            "distances": distances,
            "revenues": revenues,
        }
    else:
        return lambd
