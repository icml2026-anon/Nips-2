
import numpy as np
from scipy.linalg import eigh, sqrtm


def _whiten(H, K):
    d, n = H.shape
    mean_h = np.mean(H, axis=1, keepdims=True)
    Hc = H - mean_h
    Sigma = (Hc @ Hc.T) / n

    eigenvalues, eigenvectors = eigh(Sigma)
    idx = np.argsort(eigenvalues)[::-1][:K]
    lam = eigenvalues[idx]
    lam = np.maximum(lam, 1e-12)
    U = eigenvectors[:, idx]

    D_inv_sqrt = np.diag(1.0 / np.sqrt(lam))
    Z = D_inv_sqrt @ U.T @ Hc

    W_whiten = U @ np.diag(np.sqrt(lam))

    return Z, W_whiten


def _cumulant_slice(Z, M_target):
    K, n = Z.shape
    quadratic = np.sum((M_target @ Z) * Z, axis=0)
    Q = (Z * quadratic[np.newaxis, :]) @ Z.T / n
    Q -= np.eye(K) * np.trace(M_target) + M_target + M_target.T
    return Q


def _jade_diag(matrices, n_iter=200, tol=1e-10):
    K = matrices[0].shape[0]
    R = np.eye(K)

    for iteration in range(n_iter):
        max_off = 0.0
        for p in range(K):
            for q in range(p + 1, K):
                g11 = 0.0
                g12 = 0.0
                g22 = 0.0
                for M in matrices:
                    Rp = R[:, p]
                    Rq = R[:, q]
                    h = np.array([
                        Rp @ M @ Rp - Rq @ M @ Rq,
                        Rp @ M @ Rq + Rq @ M @ Rp
                    ])
                    g11 += h[0] * h[0]
                    g12 += h[0] * h[1]
                    g22 += h[1] * h[1]

                ton = g11 - g22
                toff = 2.0 * g12
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff) + 1e-30)

                max_off = max(max_off, abs(theta))

                c = np.cos(theta)
                s = np.sin(theta)
                Rp_new = c * R[:, p] + s * R[:, q]
                Rq_new = -s * R[:, p] + c * R[:, q]
                R[:, p] = Rp_new
                R[:, q] = Rq_new

        if max_off < tol:
            break

    return R


def _extract_directions(R, W_whiten, K):
    W_hat = W_whiten @ R
    norms = np.linalg.norm(W_hat, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    W_hat = W_hat / norms
    return W_hat


def empirical_cumulant_matrix_fast(H):
    d, n = H.shape
    mean_h = np.mean(H, axis=1, keepdims=True)
    Hc = H - mean_h
    Sigma = (Hc @ Hc.T) / n

    V = np.zeros((n, d * d))
    for i in range(n):
        V[i] = np.outer(Hc[:, i], Hc[:, i]).ravel()
    M4 = (V.T @ V) / n

    Sigma_kron = np.kron(Sigma, Sigma)
    vec_Sigma = Sigma.ravel()
    outer_Sigma = np.outer(vec_Sigma, vec_Sigma)

    P_Sigma_kron = np.zeros((d * d, d * d))
    for i in range(d):
        for k in range(d):
            P_Sigma_kron[i * d:(i + 1) * d, k * d:(k + 1) * d] = (
                Sigma[i, :].reshape(-1, 1) * Sigma[:, k].reshape(1, -1)
            ).T

    gaussian_correction = Sigma_kron + outer_Sigma + P_Sigma_kron
    M_hat = M4 - gaussian_correction
    return M_hat


def cbfr(H, K, use_fast=True):
    d, n = H.shape

    Z, W_whiten = _whiten(H, K)

    cumulant_matrices = []
    for i in range(K):
        for j in range(i, K):
            M_target = np.zeros((K, K))
            M_target[i, j] = 1.0
            M_target[j, i] = 1.0
            Q = _cumulant_slice(Z, M_target)
            Q = (Q + Q.T) / 2.0
            cumulant_matrices.append(Q)

    R = _jade_diag(cumulant_matrices, n_iter=300)

    W_hat = _extract_directions(R, W_whiten, K)

    eigenvalues = np.zeros(K)
    for i in range(K):
        M_target = np.zeros((K, K))
        M_target[i, i] = 1.0
        Q = _cumulant_slice(Z, M_target)
        r = R[:, i]
        eigenvalues[i] = r @ Q @ r

    return W_hat, eigenvalues


def recovery_error(W_true, W_hat):
    from scipy.optimize import linear_sum_assignment

    K = W_true.shape[1]
    K_hat = W_hat.shape[1]
    K_min = min(K, K_hat)

    cost = np.zeros((K, K_hat))
    for i in range(K):
        for j in range(K_hat):
            err_pos = np.linalg.norm(W_true[:, i] - W_hat[:, j])
            err_neg = np.linalg.norm(W_true[:, i] + W_hat[:, j])
            cost[i, j] = min(err_pos, err_neg)

    row_ind, col_ind = linear_sum_assignment(cost)

    errors = cost[row_ind, col_ind]
    max_error = np.max(errors) if len(errors) > 0 else np.inf
    mean_error = np.mean(errors) if len(errors) > 0 else np.inf

    return max_error, mean_error, errors
