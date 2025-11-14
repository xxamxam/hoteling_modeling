# Requires: graph-tool (C++ backend)
import math
from collections import defaultdict
import dataclasses
from dataclasses import dataclass

from typing import Dict, Union
from graph_tool import Graph
import numpy as np

# import graph_tool as gt
import graph_tool.search as gts

from cachetools import LRUCache
from enum import Enum
from typing import Dict, Tuple, FrozenSet



def nearest_source_labels(g, weight_ep, sources, source_ids=None, eps=1e-12):
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
                self.near[v] = { sid_of[v] }     # s* -> источник v
            else:
                self.near[v] = set(self.near[u]) # наследуем множество из u

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
        g, weight_ep, source=s_star,
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


def get_revenue(g, distances, nearest_set, revenue_function):
    vertice_revenue = defaultdict(float)
    for v in g.vertices():
        v = int(v)
        nearest_sellers = nearest_set[v]
        dist = distances[v]
        vertex_value = revenue_function(dist)
        assert vertex_value >= 0.0, "Revenue function must be non-negative"

        num_sellers = len(nearest_set[v])
        for seller in nearest_sellers:
            vertice_revenue[seller] += vertex_value/num_sellers
    return vertice_revenue


def bin_search(revenues: Dict, M, tol=1e-10):
    """
    param:
    revenues: Dict[Vertice, float]
    M: number of agents
    """
    revenue_values = np.array(list(revenues.values()))

    def f(lambd):
        return np.floor(revenue_values//lambd).sum() - M
    
    assert M >= len(revenue_values)
    left = 1e-12
    right = np.min(revenue_values) + tol  # чтобы у каждой вершины был как минимум один агент -- constrain
    if M == len(revenue_values):
        return {k: 1 for k in revenues.keys()}, right

    if f(right) < 0:
        while right - left > tol:
            mid = (right + left)/2
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
        remain_budgets = {k: revenues[k] - num_sellers[k] * lambd for k in revenues.keys()}
        while num_setted_sellers < M:
            pos = max(remain_budgets.items(), key=lambda x: x[1])[0]
            num_setted_sellers += 1
            num_sellers[pos] += 1
            remain_budgets[pos] -= lambd
    assert sum(num_sellers.values()) == M, f"error. Get {num_sellers=}, while \
        {sum(num_sellers.values())} != {M}.\n{lambd=}  {revenues}"
    return num_sellers, lambd


# classes for B&B method


class Status(str, Enum):
    UNEXPANDED = "UNEXPANDED"
    EXPANDED = "EXPANDED"
    FULLY_EXPANDED = "FULLY_EXPANDED"


@dataclass
class Node:
    """
    parameters:
    g: GraphWrapper
    M: number of agents
    occupied_vertices: set of occupied vertices
    cache: cached values for set value
    """
    graph: Graph
    weights_name: str = "weight"
    M: int = 0
    occupied_vertices: set = dataclasses.field(default_factory=set)
    vertices: set = dataclasses.field(default_factory=set)

    parent: 'Node' = None
    status: Status = Status.UNEXPANDED

    children: list = dataclasses.field(default_factory=list)
    value: float = 0.0
    bound: float = 0.0
    
    def __post_init__(self,):
        if len(self.vertices) == 0:
            self.vertices = {int(v) for v in self.graph.vertices()}

    @staticmethod
    def get_positions(graph,
                      occupied_vertices,
                      M,
                      weights_name,
                      revenue_function,
                      tol = 1e-12,
                      extended_return=False) -> Union[float, Dict]:

        if len(occupied_vertices) == 0:
            if extended_return:
                return {}
            else:
                return 0.0

        weight_ep = graph.ep[weights_name]
        sources = [graph.vertex(v) for v in occupied_vertices]
        
        # compute nearest sellers for buyers
        distances, nearest, nearest_set = nearest_source_labels(
            graph, weight_ep, sources, source_ids=list(occupied_vertices)
        )

        # compute revenues per Vertice
        revenues = get_revenue(
            graph, distances, nearest_set, revenue_function
        )

        # find minimal revenue per seller on graph
        num_sellers, lambd = bin_search(revenues, M, tol=tol)
        if extended_return:
            return {"num_sellers": num_sellers,
                    "lambd": lambd,
                    "nearest_set": nearest_set,
                    "distances": distances,
                    "revenues": revenues
                    }
        else:
            return lambd
    
    def compute_value(self, revenue_function, tol):
        lambd = Node.get_positions(
            self.graph,
            self.occupied_vertices,
            self.M,
            self.weights_name,
            revenue_function,
            tol=tol,
            extended_return=False
            )
        self.value = lambd
        return self.value
    
    def get_actions(self):
        return self.vertices - self.occupied_vertices
    

class BaseRevenueFunction:
    def __init__(self, base_cost: float):
        self.base_cost = base_cost

    def __call__(self, distance: float) -> float:
        return max(0.0, self.base_cost - distance)


class MyCache:
    def __init__(self, maxsize=200_000):
        self.cache = LRUCache(maxsize=maxsize)

    def put(self, S, value):
        self.cache[frozenset(S)] = (value)

    def get(self, S):
        return self.cache.get(frozenset(S))


@dataclass
class runStatistics:
    open_nodes: int = 0
    reused_nodes: int = 0
    rejected_nodes: int = 0

class BBTree:
    def __init__(self, graph: Graph, M:int, revenue_function: BaseRevenueFunction, cache_maxsize=200_000, tol = 1e-12, verbose = False):
        self.graph = graph
        self.M = M
        self.revenue_function = revenue_function
        self.tol = tol
        self.root = Node(graph=graph, M=M, occupied_vertices=set())
        self.root.value = self.root.compute_value(revenue_function, tol=self.tol)

        self.best_value = 0
        self.occupation = None
        self.verbose = verbose
        # set cache
        self.cache = MyCache(maxsize=cache_maxsize)

        # statistcs
        self.run_stat = runStatistics()

    def expand_node(self, node: Node):
        if len(node.occupied_vertices) >= self.M:
            node.status = Status.FULLY_EXPANDED
            return  # cannot expand further

        actions = node.get_actions()
        if self.verbose:
            print(f"base: Vertices: {node.occupied_vertices} val: {node.value}")
        for action in actions:

            new_occupied = set(node.occupied_vertices)
            new_occupied.add(action)
            if self.cache.get(new_occupied) is not None:
                self.run_stat.reused_nodes += 1
                continue
            
            child_node = Node(
                graph=node.graph,
                M=node.M,
                occupied_vertices=new_occupied,
                vertices=node.vertices,
                parent=node
            )

            child_node.bound = child_node.value = child_node.compute_value(self.revenue_function, self.tol)
            self.cache.put(new_occupied, child_node.value)

            if child_node.value >= node.value - 2 * self.tol:
                if self.verbose:
                    print(f"new node: Vertices: {new_occupied} val: {child_node.value}")
                self.run_stat.open_nodes += 1
                node.children.append(child_node)
                node.bound = max(node.bound, child_node.value)

                # update best value
                if child_node.value > self.best_value:
                    self.best_value = child_node.value
                    self.occupation = new_occupied
            else:
                self.run_stat.rejected_nodes += 1

        node.status = Status.EXPANDED

    def select_node_to_expand(self, node: Node):
        # select child with highest value
        cur_node = node
        while True:
            if len(cur_node.children) == 0:
                break
            cur_node = max(cur_node.children, key=lambda x: x.bound if x.status != Status.FULLY_EXPANDED else -math.inf)
        return cur_node
    def backpropagate(self, node: Node):
        cur_node = node
        # если не смогли расширить -- то делать тут нечего
        if len(cur_node.children) == 0:
            cur_node.status = Status.FULLY_EXPANDED

        while cur_node.parent is not None:
            parent = cur_node.parent
            if all(child.status == Status.FULLY_EXPANDED for child in parent.children):
                parent.status = Status.FULLY_EXPANDED
            parent.bound = max(parent.bound, cur_node.bound)
            cur_node = parent
        
    def run(self, max_iterations=1000):
        for _ in range(max_iterations):
            node_to_expand = self.select_node_to_expand(self.root)
            self.expand_node(node_to_expand)
            self.backpropagate(node_to_expand)

