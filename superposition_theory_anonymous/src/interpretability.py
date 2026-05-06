
import numpy as np
from itertools import combinations


def support_coherence(W, support):
    W_S = W[:, support]
    G_S = W_S.T @ W_S
    np.fill_diagonal(G_S, 0.0)
    if G_S.size == 0:
        return 0.0
    return np.max(np.abs(G_S))


def is_identifiable_support(W, support, s):
    threshold = 1.0 / (2 * s - 1)
    mu_S = support_coherence(W, support)
    return mu_S < threshold


def interpretability_index_exact(W, s):
    K = W.shape[1]
    if s > K:
        return 0.0

    threshold = 1.0 / (2 * s - 1)
    total = 0
    identifiable = 0

    for support in combinations(range(K), s):
        total += 1
        mu_S = support_coherence(W, list(support))
        if mu_S < threshold:
            identifiable += 1

    return identifiable / total if total > 0 else 0.0


def interpretability_index_sampled(W, s, n_samples=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    K = W.shape[1]
    if s > K:
        return 0.0

    threshold = 1.0 / (2 * s - 1)
    identifiable = 0

    for _ in range(n_samples):
        support = rng.choice(K, size=s, replace=False)
        mu_S = support_coherence(W, support)
        if mu_S < threshold:
            identifiable += 1

    return identifiable / n_samples


def interpretability_index(W, s, n_samples=10000, rng=None):
    from math import comb
    K = W.shape[1]

    if comb(K, s) <= 50000:
        return interpretability_index_exact(W, s)
    else:
        return interpretability_index_sampled(W, s, n_samples=n_samples, rng=rng)
