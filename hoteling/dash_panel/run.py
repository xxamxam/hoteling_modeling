import fire

from hoteling.dash_panel.dash_app import create_dash_app
from hoteling.generators.graph_generators import generate_line_graph
from hoteling.algorithms.branch_bound import BaseRevenueFunction


def main(port: int = 8050):
    g = generate_line_graph(6)
    rf = BaseRevenueFunction(base_cost=10)
    app = create_dash_app(g, initial_M=3, rf=rf)
    app.run(debug=True, port=port)
    

if __name__ == "__main__":
    fire.Fire(main)
