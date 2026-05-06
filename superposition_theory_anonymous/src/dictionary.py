
import numpy as np


def random_dictionary(d, K, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    W = rng.standard_normal((d, K))
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    W = W / norms
    return W


def simplex_frame(d):
    K = d + 1
    W = np.zeros((d, K))
    for k in range(K):
        if k < d:
            W[k, k] = 1.0
    centroid = np.mean(W, axis=1, keepdims=True)
    W = W - centroid
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    W = W / norms
    return W


def mutual_coherence(W):
    G = W.T @ W
    K = G.shape[0]
    np.fill_diagonal(G, 0.0)
    return np.max(np.abs(G))


def coherence_matrix(W):
    G = np.abs(W.T @ W)
    np.fill_diagonal(G, 0.0)
    return G


def minimize_coherence(d, K, n_iter=2000, lr=0.01, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    W = random_dictionary(d, K, rng)

    if K <= d:
        U, _, Vt = np.linalg.svd(W, full_matrices=False)
        W = U @ Vt
        W = W / np.linalg.norm(W, axis=0, keepdims=True)
        return W

    n_phase1 = int(n_iter * 0.6)
    lr1 = lr * 2.0
    for it in range(n_phase1):
        G = W.T @ W
        np.fill_diagonal(G, 0.0)
        grad = 4.0 * W @ G
        W -= lr1 * grad / (K * K)
        norms = np.linalg.norm(W, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        W = W / norms

    n_phase2 = n_iter - n_phase1
    p_exp = 8
    lr2 = lr * 0.5
    for it in range(n_phase2):
        G = W.T @ W
        np.fill_diagonal(G, 0.0)
        weight = np.abs(G) ** (p_exp - 2) * G
        grad = p_exp * W @ weight
        W -= lr2 * grad / (K * np.max(np.abs(weight)) + 1e-12)
        norms = np.linalg.norm(W, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        W = W / norms

    return W


def generate_clustered_dictionary(d, K_star_dict, m, epsilon=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    K_star = K_star_dict.shape[1]
    K = m * K_star
    W = np.zeros((d, K))

    for i in range(K_star):
        base = K_star_dict[:, i]
        for j in range(m):
            perturbation = rng.standard_normal(d)
            perturbation -= np.dot(perturbation, base) * base
            perturbation_norm = np.linalg.norm(perturbation)
            if perturbation_norm > 1e-12:
                perturbation = perturbation / perturbation_norm * epsilon
            vec = base + perturbation
            vec /= np.linalg.norm(vec)
            W[:, i * m + j] = vec

    return W
