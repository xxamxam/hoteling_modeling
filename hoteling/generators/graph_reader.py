from graph_tool import Graph

def read_graph_from_file(file_path):
    """
    Read Graph as graph_tool Graph from file_path.

    # Assuming the file format is:
    # First line: number of vertices n
    # Next n lines: edges in the format "u v weight"
    Args:
        file_path: Path to the graph file
    """
    g = Graph(directed=False)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        n = int(lines[0].strip())
        vertices = [g.add_vertex() for _ in range(n)]  # Add vertices
        for line in lines[1:]:
            u, v, weight = map(float, line.strip().split())
            ed = g.add_edge(vertices[int(u)], vertices[int(v)])
            if "weight" not in g.edge_properties:
                w = g.new_edge_property("double")
                g.edge_properties["weight"] = w
            g.edge_properties["weight"][ed] = weight
    return g
    