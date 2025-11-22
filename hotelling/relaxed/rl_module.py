# hotelling_lib/rl_module.py

import numpy as np
import graph_tool.all as gt
from hotelling.relaxed.utils import check_nash_deterministic


def _soft_choice(costs):
    logits = -costs
    exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
    return probs


def _entropy(x):
    return -np.sum(x * np.log(x + 1e-12))


def _determinize_if_low_entropy(x, eps_H):
    H = _entropy(x)
    if H < eps_H:
        v = int(np.argmax(x))
        det = np.zeros_like(x)
        det[v] = 1.0
        return det, H, True
    return x, H, False


def run_rl(
    G, dist_u_v, d_u, m=2, T=200, eta=0.1, entropy_weight=0.05, eps_H=0.05, seed=42
):
    rng = np.random.RandomState(seed)
    n = G.num_vertices()
    theta = [np.zeros(n) for _ in range(m)]

    entropy_log = [[] for _ in range(m)]
    strategy_log = [[] for _ in range(m)]

    def sample_action_logits(th):
        probs = np.exp(th - np.max(th))
        probs = probs / np.sum(probs)
        a = rng.choice(len(th), p=probs)
        return a, probs

    for t in range(T):
        actions = []
        probs_list = []
        for i in range(m):
            a_i, probs_i = sample_action_logits(theta[i])
            actions.append(a_i)
            probs_list.append(probs_i)
            entropy_log[i].append(_entropy(probs_i))
            strategy_log[i].append(probs_i.copy())

        # вычисление наград как ожидаемых долей потребителей
        costs = np.array([dist_u_v[:, a] for a in actions])  # (m, n)
        assign_probs = _soft_choice(costs)
        rewards = np.sum(assign_probs * d_u, axis=1)  # (m,)

        # REINFORCE с энтропийной регуляризацией
        for i in range(m):
            probs = probs_list[i]
            grad_log = -probs
            grad_log[actions[i]] += 1.0
            # энтропийный градиент для логитов softmax: приближённо -log(probs) - 1
            grad_entropy = -(np.log(probs + 1e-12) + 1.0)
            theta[i] += eta * (rewards[i] * grad_log + entropy_weight * grad_entropy)

    # детерминизация
    x_det = []
    det_flags = []
    for i in range(m):
        x_i, H, det = _determinize_if_low_entropy(strategy_log[i][-1], eps_H)
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
        "name": "RL",
    }
