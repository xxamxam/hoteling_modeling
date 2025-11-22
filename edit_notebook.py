import json

# Load the notebook
with open("notebooks/relaxed_demo.ipynb", "r") as f:
    nb = json.load(f)

# Update cells to remove networkx and use graph_tool
new_cells = []
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_lines = cell["source"]
        source_text = "".join(source_lines)
        new_source = []
        if (
            "BalancedTree" in source_text
            and "md_res = run_mirror_descent" in source_text
        ):
            # Balanced tree MD cell
            new_source.append("import graph_tool.all as gt")
            new_source.append("import numpy as np")
            new_source.append("from hotelling.relaxed import run_mirror_descent")
            new_source.append(
                "from hotelling.relaxed.utils import plot_entropy_dynamics, visualize_strategies, visualize_strategy_evolution, visualize_final_deterministic"
            )
            new_source.append("# данные")
            new_source.append('G = test_graphs["BalancedTree"]')
            new_source.append(
                "dist_u_v = gt.shortest_distance(G).get_2d_array(range(G.num_vertices()))"
            )
            new_source.append("d_u = np.ones(G.num_vertices()) / G.num_vertices()")
            new_source.append("# запуск")
            new_source.append("T = 10000")
            new_source.append(
                "md_res = run_mirror_descent(G, dist_u_v, d_u, m=4, T=T,eta = 0.5, beta=100)"
            )
            new_source.append(
                'plot_entropy_dynamics(md_res["entropy"], title="Mirror Descent: энтропия")'
            )
            new_source.append(
                'visualize_final_deterministic(G, md_res["x"], title="Mirror Descent: финальные детерминированные стратегии")'
            )
            new_source.append(
                'visualize_strategies(G, md_res["x"], title="Mirror Descent: финальные стратегии")'
            )
            new_source.append(
                'visualize_strategy_evolution(G, md_res["log"], every=T//5, title="Mirror Descent: эволюция")'
            )
            new_source.append('print("=== Mirror Descent ===")')
            new_source.append('print("Nash:", md_res["nash"])')
            new_source.append('for msg in md_res["nash_messages"]:')
            new_source.append('    print(" -", msg)')
        elif "lattice" in source_text and "md_res = run_mirror_descent" in source_text:
            # Grid MD cell
            new_source.append("import graph_tool.all as gt")
            new_source.append("import numpy as np")
            new_source.append("from hotelling.relaxed import run_mirror_descent")
            new_source.append(
                "from hotelling.relaxed.utils import plot_entropy_dynamics, visualize_strategies, visualize_strategy_evolution, visualize_final_deterministic"
            )
            new_source.append("# данные")
            new_source.append("G = gt.lattice((3,4))")
            new_source.append(
                "dist_u_v = gt.shortest_distance(G).get_2d_array(range(G.num_vertices()))"
            )
            new_source.append("d_u = np.ones(G.num_vertices()) / G.num_vertices()")
            new_source.append("# запуск")
            new_source.append("T = 10000")
            new_source.append(
                "md_res = run_mirror_descent(G, dist_u_v, d_u, m=2, T=T,eta = 0.01, beta=1)"
            )
            new_source.append(
                'plot_entropy_dynamics(md_res["entropy"], title="Mirror Descent: энтропия")'
            )
            new_source.append(
                'visualize_final_deterministic(G, md_res["x"], title="Mirror Descent: финальные детерминированные стратегии")'
            )
            new_source.append(
                'visualize_strategies(G, md_res["x"], title="Mirror Descent: финальные стратегии")'
            )
            new_source.append(
                'visualize_strategy_evolution(G, md_res["log"], every=T//5, title="Mirror Descent: эволюция")'
            )
            new_source.append('print("=== Mirror Descent ===")')
            new_source.append('print("Nash:", md_res["nash"])')
            new_source.append('for msg in md_res["nash_messages"]:')
            new_source.append('    print(" -", msg)')
        elif "run_rl" in source_text and "BalancedTree" in source_text:
            # Balanced tree RL cell
            new_source.append("from hotelling.relaxed import run_rl")
            new_source.append("T = 100000")
            new_source.append(
                "rl_res = run_rl(G, dist_u_v, d_u, m=4, T=T, eta=0.05, entropy_weight=0, seed = 20)"
            )
            new_source.append(
                'plot_entropy_dynamics(rl_res["entropy"], "RL: энтропия")'
            )
            new_source.append(
                'visualize_final_deterministic(G, rl_res["x"], title="RL: финальные детерминированные стратегии")'
            )
            new_source.append(
                'visualize_strategies(G, rl_res["x"], "RL: финальные стратегии")'
            )
            new_source.append(
                'visualize_strategy_evolution(G, rl_res["log"], every=T//5, title="RL: эволюция")'
            )
            new_source.append('print("=== RL ===")')
            new_source.append('print("Positions:", rl_res["determinized"])')
            new_source.append('print("Nash:", rl_res["nash"])')
            new_source.append('for msg in rl_res["nash_messages"]:')
            new_source.append('    print(" -", msg)')
        elif "run_rl" in source_text and "lattice" in source_text:
            # Grid RL cell
            new_source.append("from hotelling.relaxed import run_rl")
            new_source.append("T = 600000")
            new_source.append(
                "rl_res = run_rl(G, dist_u_v, d_u, m=2, T=T, eta=0.05, entropy_weight=0)"
            )
            new_source.append(
                'plot_entropy_dynamics(rl_res["entropy"], "RL: энтропия")'
            )
            new_source.append(
                'visualize_final_deterministic(G, rl_res["x"], title="RL: финальные детерминированные стратегии")'
            )
            new_source.append(
                'visualize_strategies(G, rl_res["x"], "RL: финальные стратегии")'
            )
            new_source.append(
                'visualize_strategy_evolution(G, rl_res["log"], every=T//5, title="RL: эволюция")'
            )
            new_source.append('print("=== RL ===")')
            new_source.append('print("Positions:", rl_res["determinized"])')
            new_source.append('print("Nash:", rl_res["nash"])')
            new_source.append('for msg in rl_res["nash_messages"]:')
            new_source.append('    print(" -", msg)')
        else:
            # Other cells, keep as is but remove networkx
            for line in source_lines:
                if "import networkx as nx" in line:
                    continue
                new_source.append(line)
        cell["source"] = new_source
    new_cells.append(cell)

nb["cells"] = new_cells

# Save the notebook
with open("notebooks/relaxed_demo.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
