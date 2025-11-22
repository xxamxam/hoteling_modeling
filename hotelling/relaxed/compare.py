# hotelling_lib/compare.py

import matplotlib.pyplot as plt


def compare_modules(G, dist_u_v, d_u, runners, labels=None):
    """
    runners: list of callables (G, dist_u_v, d_u, ...) -> result dict
    labels: optional list of names to override result['name']
    Each result dict must contain keys: 'x', 'entropy', 'nash', 'nash_messages', 'name'
    """
    results = []
    for idx, run in enumerate(runners):
        res = run(G, dist_u_v, d_u)
        if labels and idx < len(labels):
            res["name"] = labels[idx]
        results.append(res)

    # Plot entropy dynamics
    plt.figure(figsize=(10, 6))
    for res in results:
        name = res["name"]
        for i, ent in enumerate(res["entropy"]):
            plt.plot(ent, label=f"{name} фирма {i + 1}")
    plt.title("Сравнение динамики энтропии по модулям")
    plt.xlabel("Итерация/Эпоха")
    plt.ylabel("Энтропия стратегии")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Nash check reports
    print("=== Проверка равновесия Нэша ===")
    for res in results:
        print(f"[{res['name']}] Nash: {res['nash']}")
        for msg in res["nash_messages"]:
            print(" -", msg)

    return results
