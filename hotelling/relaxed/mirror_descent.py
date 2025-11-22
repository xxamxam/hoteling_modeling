# hotelling_lib/mirror_descent.py

import numpy as np
import graph_tool.all as gt
from hotelling.relaxed.utils import check_nash_deterministic


def _soft_choice(costs):
    logits = -costs
    exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
    return probs  # (m, n)


def _expected_profit_vector(v_others, dist_u_v, d_u):
    n = dist_u_v.shape[0]
    profits = []
    for v in range(n):
        costs = np.array(
            [dist_u_v[:, v]] + [dist_u_v[:, vo] for vo in v_others]
        )  # (m, n)
        probs = _soft_choice(costs)
        profits.append(np.sum(probs[0] * d_u))
    return np.array(profits)  # длина n


def _entropy(x):
    return -np.sum(x * np.log(x + 1e-12))


def _md_step_with_entropy(x, values, eta, beta):
    # Mirror Descent на симплексе (KL) с штрафом на энтропию:
    # maximize <x, values> - beta * H(x)
    # update: x^+ ∝ x * exp(eta * (values + beta * (log x + 1)))
    eff = values + beta * (np.log(x + 1e-12) + 1.0)
    logits = np.log(x + 1e-12) + eta * eff
    z = np.exp(logits - np.max(logits))
    return z / np.sum(z)


def _determinize_if_low_entropy(x, eps_H):
    H = _entropy(x)
    if H < eps_H:
        v = int(np.argmax(x))
        det = np.zeros_like(x)
        det[v] = 1.0
        return det, H, True
    return x, H, False


def run_mirror_descent(G, dist_u_v, d_u, m=2, T=200, eta=0.2, beta=0.05, eps_H=0.05):
    """
    eta  — шаг MD
    beta — коэффициент штрафа на энтропию (больше → более детерминированные стратегии)
    """
    n = G.num_vertices()
    x_list = [np.ones(n) / n for _ in range(m)]

    entropy_log = [[] for _ in range(m)]
    strategy_log = [[] for _ in range(m)]

    for t in range(T):
        for i in range(m):
            v_others = [int(np.argmax(x_list[j])) for j in range(m) if j != i]
            values = _expected_profit_vector(v_others, dist_u_v, d_u)
            x_list[i] = _md_step_with_entropy(x_list[i], values, eta, beta)
            H = _entropy(x_list[i])
            entropy_log[i].append(H)
            strategy_log[i].append(x_list[i].copy())

    # автоматическая детерминизация
    x_det = []
    det_flags = []
    for i in range(m):
        x_i, H, det = _determinize_if_low_entropy(x_list[i], eps_H)
        x_det.append(x_i)
        det_flags.append(det)

    is_nash, messages = check_nash_deterministic(x_det, dist_u_v, d_u)

    return {
        "x": x_det,
        "entropy": entropy_log,
        "log": strategy_log,
        "nash": is_nash,
        "nash_messages": messages,
        "determinized": det_flags,
        "name": "MirrorDescent(+Entropy)",
    }
