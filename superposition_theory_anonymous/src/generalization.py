
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def generate_binary_task(W, s, n, sigma=0.1, B=1.0, rng=None, u_star=None):
    if rng is None:
        rng = np.random.default_rng()

    d, K = W.shape

    if u_star is None:
        u_star = rng.standard_normal(d)
        u_star /= np.linalg.norm(u_star)

    p = min(s / K, 1.0)
    Z = rng.binomial(1, p, size=(K, n)).astype(np.float64)
    E = rng.exponential(0.5, size=(K, n))
    Phi = np.clip(Z * E, 0.0, B)

    H0 = W @ Phi

    scores = u_star @ H0
    y = (scores > 0).astype(np.float64)

    noise = rng.standard_normal((d, n)) * sigma
    H = (H0 + noise).T

    return H, y, u_star


def measure_generalization_gap(H_train, y_train, H_test, y_test):
    clf = LogisticRegression(
        penalty="l2", C=1.0, max_iter=1000, solver="lbfgs"
    )
    clf.fit(H_train, y_train)

    prob_train = clf.predict_proba(H_train)
    prob_test = clf.predict_proba(H_test)

    labels = np.unique(np.concatenate([y_train, y_test]))
    if len(labels) < 2:
        return 0.0, 0.0, 0.0

    train_loss = log_loss(y_train, prob_train, labels=labels)
    test_loss = log_loss(y_test, prob_test, labels=labels)
    gap = abs(test_loss - train_loss)

    return gap, train_loss, test_loss


def theoretical_gen_bound(s, n, B=1.0, L=1.0, delta=0.05):
    term1 = 2 * L * B * np.sqrt(3 * s / (2 * n))
    term2 = L * B * np.sqrt(3 * s * np.log(2 / delta) / n)
    return term1 + term2
