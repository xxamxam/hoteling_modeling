import matplotlib.pyplot as plt
import numpy as np
import graph_tool.all as gt


def plot_entropy_dynamics(entropy_log, title="Динамика энтропии", eps_H=0.05):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, entropies in enumerate(entropy_log):
        ax.plot(entropies, label=f"Фирма {i + 1}")
    ax.axhline(eps_H, color="gray", linestyle="--", label=f"Порог энтропии = {eps_H}")
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Энтропия")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def visualize_strategies(G, x_list, title="Финальные стратегии фирм"):
    n = G.num_vertices()
    pos = get_positions(G, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    # Draw edges
    for e in G.edges():
        u, v = int(e.source()), int(e.target())
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], "gray", zorder=1)
    # Draw nodes
    for node in range(n):
        ax.scatter(*pos[node], s=300, color="lightgray", zorder=2)
        ax.text(
            pos[node][0],
            pos[node][1],
            str(node),
            ha="center",
            va="center",
            fontsize=12,
            zorder=4,
        )
    colors = ["red", "blue", "green", "purple"]
    for i, x in enumerate(x_list):
        for j in range(n):
            ax.scatter(
                *pos[j],
                s=500 * x[j],
                color=colors[i % len(colors)],
                alpha=0.5,
                label=f"Фирма {i + 1}" if j == 0 else "",
            )
    plt.title(title)
    plt.legend()
    ax.set_aspect("equal")
    plt.show()


def visualize_strategy_evolution(G, strategy_log, every=20, title="Эволюция стратегий"):
    n = G.num_vertices()
    pos = get_positions(G, seed=42)
    colors = ["red", "blue", "green", "purple"]
    n_iters = len(strategy_log[0])
    for t in range(0, n_iters, every):
        fig, ax = plt.subplots(figsize=(8, 6))
        # Draw edges
        for e in G.edges():
            u, v = int(e.source()), int(e.target())
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], "gray", zorder=1)
        # Draw nodes
        for node in range(n):
            ax.scatter(*pos[node], s=300, color="lightgray", zorder=2)
            ax.text(
                pos[node][0],
                pos[node][1],
                str(node),
                ha="center",
                va="center",
                fontsize=12,
                zorder=4,
            )
        for i, log in enumerate(strategy_log):
            x_t = log[t]
            for j in range(n):
                ax.scatter(
                    *pos[j],
                    s=500 * x_t[j],
                    color=colors[i % len(colors)],
                    alpha=0.5,
                    label=f"Фирма {i + 1}" if j == 0 else "",
                )
        plt.title(f"{title}, итерация {t}")
        plt.legend()
        ax.set_aspect("equal")
        plt.show()


def compute_payoffs_deterministic(positions, dist_u_v, d_u):
    """
    positions: list[int] длиной m — детерминированные вершины фирм
    dist_u_v: (n, n) матрица кратчайших расстояний
    d_u: (n,) веса вершин (распределение потребителей)
    Возвращает: payoffs (m,)
    """
    m = len(positions)
    n = dist_u_v.shape[0]
    # матрица расстояний фирм до каждой вершины
    D = np.stack([dist_u_v[:, v] for v in positions], axis=1)  # (n, m)
    # минимальные расстояния по вершинам
    dmin = D.min(axis=1, keepdims=True)  # (n, 1)
    winners = D == dmin  # (n, m) булево
    tie_counts = winners.sum(axis=1)  # (n,)
    # доля каждой вершины для каждой фирмы (с учётом деления при ничьей)
    share = winners.astype(float) / tie_counts[:, None]  # (n, m)
    # итоговые доходы: сумма весов вершин, умноженных на долю этой фирмы
    payoffs = (d_u[:, None] * share).sum(axis=0)  # (m,)
    return payoffs


def check_nash_deterministic(x_list, dist_u_v, d_u):
    """
    x_list: list[np.ndarray] — вероятности по вершинам, берем argmax как детерминированную позицию
    Возвращает: (is_nash: bool, messages: list[str])
    """
    m = len(x_list)
    n = dist_u_v.shape[0]
    positions = [int(np.argmax(x)) for x in x_list]
    base_payoffs = compute_payoffs_deterministic(positions, dist_u_v, d_u)
    messages = []
    is_nash = True

    for i in range(m):
        best = base_payoffs[i]
        improvable = False
        for w in range(n):
            if w == positions[i]:
                continue
            alt_positions = positions.copy()
            alt_positions[i] = w
            alt_payoffs = compute_payoffs_deterministic(alt_positions, dist_u_v, d_u)
            if alt_payoffs[i] > best + 1e-12:
                messages.append(
                    f"Фирма {i + 1}: улучшение {best:.6f} → {alt_payoffs[i]:.6f} при переходе в вершину {w}"
                )
                is_nash = False
                improvable = True
                break
        if not improvable:
            messages.append(f"Фирма {i + 1}: стратегия устойчива — равновесие Нэша.")
    return is_nash, messages


# hotelling_lib/utils.py


def extract_positions_from_strategies(x_list):
    """
    Преобразует список стратегий (np.ndarray) в список позиций (int) через argmax.
    Ожидает one-hot или вероятности; берёт вершину максимума.
    """
    return [int(np.argmax(x)) for x in x_list]


def get_positions(G, seed=None):
    """Get positions for graph nodes using ARF layout (planar-aware force-directed).

    ARF (Adaptive Resolution Force-directed) layout produces high-quality graph
    drawings with minimal edge crossings and good planar-like embeddings.

    Args:
        G: graph_tool Graph object
        seed: int, optional seed for reproducible results
    """
    # Set numpy random seed for reproducible results if seed is provided
    if seed is not None:
        np.random.seed(seed)

    # Use ARF layout - provides excellent planar-like embeddings with minimal crossings
    pos = gt.sfdp_layout(G)

    # Convert to dict
    pos_dict = {}
    for i in range(G.num_vertices()):
        pos_dict[i] = (pos[i][0], pos[i][1])
    return pos_dict


def visualize_final_deterministic(
    G, x_list, title="Финальные детерминированные стратегии", colors=None
):
    """
    Визуализирует конечное детерминированное размещение фирм:
    - одна точка на вершине графа для каждой фирмы,
    - цветом отмечается фирма.
    """
    positions_indices = extract_positions_from_strategies(x_list)
    n = G.num_vertices()
    node_list = list(range(n))
    positions = [node_list[i] for i in positions_indices]
    pos = get_positions(G, seed=42)
    if colors is None:
        colors = ["red", "blue", "green", "purple", "orange", "cyan"]

    fig, ax = plt.subplots(figsize=(8, 6))
    # Draw edges
    for e in G.edges():
        u, v = int(e.source()), int(e.target())
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], "gray", zorder=1)
    # Draw nodes
    for node in range(n):
        ax.scatter(*pos[node], s=300, color="lightgray", zorder=2)
        ax.text(
            pos[node][0],
            pos[node][1],
            str(node),
            ha="center",
            va="center",
            fontsize=12,
            zorder=4,
        )
    # Draw firms
    for i, node in enumerate(positions):
        ax.scatter(
            *pos[node],
            s=800,
            color=colors[i % len(colors)],
            label=f"Фирма {i + 1}",
            zorder=3,
        )
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def visualize_modules_final(
    G, results, title="Финальное размещение фирм по модулям", colors=None
):
    """
    Строит мульти-панельную визуализацию финального детерминированного размещения для нескольких модулей.
    results: dict[name] -> dict с ключом 'x' (список стратегий фирм)
    """
    if colors is None:
        colors = ["red", "blue", "green", "purple", "orange", "cyan"]
    n = G.num_vertices()
    pos = get_positions(G, seed=42)
    node_list = list(range(n))

    k = len(results)
    rows = 2 if k > 2 else 1
    cols = (k + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (name, res) in zip(axes, results.items()):
        x_list = res["x"]
        positions_indices = extract_positions_from_strategies(x_list)
        positions = [node_list[i] for i in positions_indices]
        # Draw edges
        for e in G.edges():
            u, v = int(e.source()), int(e.target())
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], "gray", zorder=1)
        # Draw nodes
        for node in range(n):
            ax.scatter(*pos[node], s=300, color="lightgray", zorder=2)
            ax.text(
                pos[node][0],
                pos[node][1],
                str(node),
                ha="center",
                va="center",
                fontsize=12,
                zorder=4,
            )
        for i, node in enumerate(positions):
            ax.scatter(
                *pos[node],
                s=800,
                color=colors[i % len(colors)],
                label=f"Фирма {i + 1}",
                zorder=3,
            )
        ax.set_title(name)
        ax.legend()
        ax.set_aspect("equal")
    # пустые оси (если модулей меньше, чем слотов)
    for ax in axes[len(results) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
