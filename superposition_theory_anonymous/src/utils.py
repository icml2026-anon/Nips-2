
import os
import numpy as np
import yaml


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def set_seed(seed):
    np.random.seed(seed)


def load_config(path=None):
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "configs", "default.yaml"
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def tau_coefficient(d, K, s):
    if K <= 1:
        return 0.0
    return (K - d) * (2 * s - 1) ** 2 / (d * (K - 1))


def critical_dimension(s):
    return (2 * s - 1) ** 2


def max_features(d, s):
    d_star = critical_dimension(s)
    if d_star <= d:
        return np.inf
    return int(np.floor(d * (d_star - 1) / (d_star - d)))


def welch_bound(d, K):
    if K <= d:
        return 0.0
    return np.sqrt((K - d) / (d * (K - 1)))


def donoho_elad_threshold(s):
    return 1.0 / (2 * s - 1)


def asymptotic_plateau(s, K_star):
    if K_star == np.inf or K_star == 0:
        return 1.0
    result = 1.0
    for i in range(s):
        result *= (1.0 - i / K_star)
    return result
