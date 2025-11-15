"""
Graph generation utilities for Hotelling Game Library.
Provides functions to generate common graph topologies and random seller placement.
"""

import random
import numpy as np
from typing import List, Optional, Tuple
import graph_tool as gt
from graph_tool import Graph


def generate_line_graph(n: int) -> Graph:
    """
    Generate a line graph with n vertices.

    Args:
        n: Number of vertices

    Returns:
         with line topology (0-1-2-...-(n-1))
    """
    if n < 1:
        raise ValueError("Number of vertices must be at least 1")

    g = Graph(directed=False)

    w = g.new_edge_property("double")
    g.edge_properties["weight"] = w

    # Add vertices
    vertices = [g.add_vertex() for _ in range(n)]

    # Add edges
    for i in range(n - 1):
        ed = g.add_edge(vertices[i], vertices[i + 1])
        g.edge_properties["weight"][ed] = 1.0

    return g


def generate_star_graph(n_leaves: int) -> Graph:
    """
    Generate a star graph with n leaves.

    Args:
        n_leaves: Number of leaf vertices (total vertices = n_leaves + 1)

    Returns:
         with star topology (center connected to all leaves)
    """
    if n_leaves < 1:
        raise ValueError("Number of leaves must be at least 1")

    g = Graph(directed=False)

    w = g.new_edge_property("double")
    g.edge_properties["weight"] = w

    # Add vertices: center (index 0) + leaves
    vertices = [g.add_vertex() for _ in range(n_leaves + 1)]

    # Connect center to all leaves
    center = vertices[0]
    for leaf in vertices[1:]:
        ed = g.add_edge(center, leaf)
        g.edge_properties["weight"][ed] = 1.0

    return g


def generate_random_tree(n: int, seed: Optional[int] = 0) -> Graph:
    """
    Generate a random tree with n vertices.

    Args:
        n: Number of vertices
        seed: Random seed for reproducibility

    Returns:
         with random tree topology
    """
    if n < 1:
        raise ValueError("Number of vertices must be at least 1")

    # Create a random generator with the seed
    seed = seed if seed == 0 else random.randint(1, 1_000_000)

    rng = np.random.default_rng(seed)

    # random.Random(seed) if seed is not None else random.Random()

    g = Graph(directed=False)

    w = g.new_edge_property("double")
    g.edge_properties["weight"] = w

    # Add vertices
    vertices = [g.add_vertex() for _ in range(n)]

    # Generate random tree using Prufer code or similar
    if n == 1:
        return g
    elif n == 2:
        ed = g.add_edge(vertices[0], vertices[1])
        g.edge_properties["weight"][ed] = 1.0
        return g

    # Use a simple random tree generation algorithm
    # Start with vertex 0, then add vertices one by one
    connected = {0}
    unconnected = set(range(1, n))

    while unconnected:
        # Pick random unconnected vertex
        new_vertex = rng.choice(list(unconnected))
        # Connect to random connected vertex
        connected_vertex = rng.choice(list(connected))

        ed = g.add_edge(vertices[new_vertex], vertices[connected_vertex])
        g.edge_properties["weight"][ed] = 1.0
        connected.add(new_vertex)
        unconnected.remove(new_vertex)

    return g


def generate_grid_graph(rows: int, cols: int) -> Graph:
    """
    Generate a grid graph (Manhattan-style) with rows x cols vertices.

    Args:
        rows: Number of rows
        cols: Number of columns

    Returns:
         with grid topology
    """
    if rows < 1 or cols < 1:
        raise ValueError("Rows and columns must be at least 1")

    g = Graph(directed=False)

    w = g.new_edge_property("double")
    g.edge_properties["weight"] = w
    # Add vertices
    n_vertices = rows * cols
    vertices = [g.add_vertex() for _ in range(n_vertices)]

    # Helper function to get vertex index
    def get_index(r: int, c: int) -> int:
        return r * cols + c

    # Add edges
    for r in range(rows):
        for c in range(cols):
            current = get_index(r, c)

            # Connect right
            if c + 1 < cols:
                right = get_index(r, c + 1)
                ed = g.add_edge(vertices[current], vertices[right])
                g.edge_properties["weight"][ed] = 1.0

            # Connect down
            if r + 1 < rows:
                down = get_index(r + 1, c)
                ed = g.add_edge(vertices[current], vertices[down])
                g.edge_properties["weight"][ed] = 1.0

    return (g)


def place_sellers_randomly(graph: Graph, k: int, seed: Optional[int] = None) -> List[int]:
    """
    Randomly place k sellers on a graph.

    Args:
        graph: Graph to place sellers on
        k: Number of sellers to place
        seed: Random seed for reproducibility

    Returns:
        List of vertex indices where sellers are placed
    """
    if k < 1:
        raise ValueError("Number of sellers must be at least 1")
    if k > graph.num_vertices:
        raise ValueError(f"Cannot place {k} sellers on graph with {graph.num_vertices} vertices")

    if seed is not None:
        random.seed(seed)

    # Randomly select k distinct vertices
    vertices = list(range(graph.num_vertices))
    sellers = random.sample(vertices, k)

    return sorted(sellers)


def place_sellers_spaced(graph: Graph, k: int) -> List[int]:
    """
    Place k sellers with maximum spacing on a graph.

    Args:
        graph: Graph to place sellers on
        k: Number of sellers to place

    Returns:
        List of vertex indices where sellers are placed
    """
    if k < 1:
        raise ValueError("Number of sellers must be at least 1")
    if k > graph.num_vertices:
        raise ValueError(f"Cannot place {k} sellers on graph with {graph.num_vertices} vertices")

    n = graph.num_vertices

    if k == 1:
        return [n // 2]  # Place in the middle
    elif k == n:
        return list(range(n))  # Place on all vertices

    # Simple spacing algorithm: divide the vertex range
    sellers = []
    for i in range(k):
        pos = int(i * (n - 1) / (k - 1))
        sellers.append(pos)

    return sellers


# Convenience functions for common use cases
def create_line_graph(n: int) -> Graph:
    """Create a line graph with n vertices."""
    return generate_line_graph(n)


def create_star_graph(n_leaves: int) -> Graph:
    """Create a star graph with n leaves."""
    return generate_star_graph(n_leaves)


def create_random_tree(n: int, seed: Optional[int] = None) -> Graph:
    """Create a random tree with n vertices."""
    return generate_random_tree(n, seed)


def create_grid_graph(rows: int, cols: int) -> Graph:
    """Create a grid graph with given dimensions."""
    return generate_grid_graph(rows, cols)


def random_sellers(graph: Graph, k: int, seed: Optional[int] = None) -> List[int]:
    """Randomly place k sellers on a graph."""
    return place_sellers_randomly(graph, k, seed)


def spaced_sellers(graph: Graph, k: int) -> List[int]:
    """Place k sellers with spacing on a graph."""
    return place_sellers_spaced(graph, k)


# Convenience functions that create HotellingGame instances directly
def create_line_game(n: int, config=None) -> 'HotellingGame':
    """Create a HotellingGame with a line graph."""
    from ..src.htglib.api import HotellingGame
    graph = generate_line_graph(n)
    return HotellingGame.from_graph(graph, config)


def create_star_game(n_leaves: int, config=None) -> 'HotellingGame':
    """Create a HotellingGame with a star graph."""
    from ..src.htglib.api import HotellingGame
    graph = generate_star_graph(n_leaves)
    return HotellingGame.from_graph(graph, config)


def create_random_tree_game(n: int, seed=None, config=None) -> 'HotellingGame':
    """Create a HotellingGame with a random tree."""
    from ..src.htglib.api import HotellingGame
    graph = generate_random_tree(n, seed)
    return HotellingGame.from_graph(graph, config)


def create_grid_game(rows: int, cols: int, config=None) -> 'HotellingGame':
    """Create a HotellingGame with a grid graph."""
    from ..src.htglib.api import HotellingGame
    graph = generate_grid_graph(rows, cols)
    return HotellingGame.from_graph(graph, config)
