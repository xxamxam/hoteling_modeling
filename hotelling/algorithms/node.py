# classes for B&B method


import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import List, Union

from graph_tool import Graph
from hotelling.hotelling_game.cost_functions import BaseRevenueFunction
from hotelling.hotelling_game.game_evaluation import evaluate_sellers


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

    def __hash__(self) -> int:
        return hash(frozenset(self.occupied_vertices))

    def __post_init__(
        self,
    ):
        if len(self.vertices) == 0:
            self.vertices = {int(v) for v in self.graph.vertices()}

    def compute_value(self, revenue_function: BaseRevenueFunction, tol: float) -> float:
        lambd = evaluate_sellers(
            self.graph,
            self.occupied_vertices,
            self.M,
            self.weights_name,
            revenue_function,
            tol=tol,
            extended_return=False,
        )

        assert isinstance(lambd, float)
        return lambd

    def get_actions(self):
        return self.vertices - self.occupied_vertices


@dataclass
class TreeNode(Node):
    """
    The same as Node, but with additional fields
    """

    parent: Union["TreeNode", None] = None
    status: Status = Status.UNEXPANDED

    children: List["TreeNode"] = dataclasses.field(default_factory=list)
    value: float = -float("inf")
    bound: float = -float("inf")
