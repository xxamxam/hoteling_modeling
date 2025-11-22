# Requires: graph-tool (C++ backend)
import math
from dataclasses import dataclass

from graph_tool import Graph
from heapdict import heapdict

from cachetools import LRUCache

from hotelling.algorithms.node import Node, Status, TreeNode
from hotelling.hotelling_game.cost_functions import BaseRevenueFunction


class MyCache:
    def __init__(self, maxsize=200_000):
        self.cache = LRUCache(maxsize=maxsize)

    def put(self, S, value):
        self.cache[frozenset(S)] = value

    def get(self, S):
        return self.cache.get(frozenset(S))


@dataclass
class runStatistics:
    open_nodes: int = 0
    reused_nodes: int = 0
    rejected_nodes: int = 0


class BBHeap:
    def __init__(
        self,
        graph: Graph,
        M: int,
        revenue_function: BaseRevenueFunction,
        cache_maxsize=200_000,
        tol=1e-12,
        verbose=False,
    ):
        self.graph = graph
        self.M = M
        self.revenue_function = revenue_function
        self.tol = tol

        # set cache
        self.cache = MyCache(maxsize=cache_maxsize)
        self.queue = heapdict()

        root_node = Node(graph=graph, M=M, occupied_vertices=set())
        value = root_node.compute_value(revenue_function, tol=self.tol)

        self.queue[root_node] = -value

        self.best_value = 0
        self.occupation = set()
        self.verbose = verbose

        # statistcs
        self.run_stat = runStatistics()

    def expand_node(self, node: Node, node_value: float):
        if len(node.occupied_vertices) >= self.M:
            # cannot expand further
            return

        actions = node.get_actions()

        if self.verbose:
            print(
                f"base: Vertices: {node.occupied_vertices}\
                    val: {node_value}"
            )
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
            )

            child_value = child_node.compute_value(self.revenue_function, self.tol)
            self.cache.put(new_occupied, child_value)
            if self.verbose:
                print(
                    f"new node: Vertices: {new_occupied} val: {child_value}.\
                       better: {child_value >= node_value - 2 * self.tol}"
                )

            if child_value >= node_value - 2 * self.tol:
                if self.verbose:
                    print(
                        f"new node: Vertices: {new_occupied} \
                           val: {child_value}"
                    )
                self.run_stat.open_nodes += 1
                self.queue[child_node] = -child_value  # priority = -value

                # update best value
                if child_value > self.best_value:
                    self.best_value = child_value
                    self.occupation = new_occupied
            else:
                self.run_stat.rejected_nodes += 1

    def select_node_to_expand(
        self,
    ):
        next_item, minus_value = self.queue.popitem()
        return next_item, -minus_value

    def run(self, max_iterations=1000):
        for _ in range(max_iterations):
            node_to_expand, value = self.select_node_to_expand()
            self.expand_node(node_to_expand, value)

            if len(self.queue) == 0:
                # if there is no more items to explore
                break


class BBTree:
    def __init__(
        self,
        graph: Graph,
        M: int,
        revenue_function: BaseRevenueFunction,
        cache_maxsize=200_000,
        tol=1e-12,
        verbose=False,
    ):
        self.graph = graph
        self.M = M
        self.revenue_function = revenue_function
        self.tol = tol
        self.root = TreeNode(graph=graph, M=M, occupied_vertices=set())
        self.root.value = self.root.compute_value(revenue_function, tol=self.tol)

        self.best_value = 0
        self.occupation = None
        self.verbose = verbose
        # set cache
        self.cache = MyCache(maxsize=cache_maxsize)

        # statistcs
        self.run_stat = runStatistics()

    def expand_node(self, node: TreeNode):
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

            child_node = TreeNode(
                graph=node.graph,
                M=node.M,
                occupied_vertices=new_occupied,
                vertices=node.vertices,
                parent=node,
            )

            child_node.bound = child_node.value = child_node.compute_value(
                self.revenue_function, self.tol
            )
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

    def select_node_to_expand(self, node: TreeNode):
        # select child with highest value
        cur_node = node
        while True:
            if len(cur_node.children) == 0:
                break
            cur_node = max(
                cur_node.children,
                key=lambda x: x.bound
                if x.status != Status.FULLY_EXPANDED
                else -math.inf,
            )
        return cur_node

    def backpropagate(self, node: TreeNode):
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
