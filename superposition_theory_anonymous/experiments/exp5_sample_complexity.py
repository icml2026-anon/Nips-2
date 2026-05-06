
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import ensure_results_dir, load_config
from src.dictionary import minimize_coherence
from src.slr_model import SLRModel
from src.cbfr import cbfr, recovery_error


def find_min_n(W, s, sigma, B, K, eps_target, n_range, n_trials, seed):
    d = W.shape[0]
    lambda_params = np.linspace(1.0, 3.0, K)
    model = SLRModel(W, s, sigma=sigma, B=B,
                     activation="bernoulli_exponential",
                     lambda_params=lambda_params)

    lo, hi = n_range
    best_n = None

    n_candidates = np.unique(np.logspace(np.log10(lo), np.log10(hi), 20).astype(int))

    for n_val in n_candidates:
        errors = []
        for trial in range(n_trials):
            rng = np.random.default_rng(seed + trial)
            H = model.sample(int(n_val), rng=rng)
            try:
                W_hat, _ = cbfr(H, K, use_fast=True)
                max_err, _, _ = recovery_error(W, W_hat)
                errors.append(max_err)
            except Exception:
                errors.append(np.inf)

        mean_err = np.nanmean(errors)
        if mean_err <= eps_target:
            best_n = int(n_val)
            break

    return best_n


def run_exp5(config=None):
    if config is None:
        config = load_config()
    cfg = config["exp5_sample_complexity"]
    results_dir = ensure_results_dir()

    eta = cfg["eta"]
    s = cfg["s"]
    sigma = cfg["sigma"]
    d_values = cfg["d_values"]
    eps_target = cfg["eps_target"]
    n_range = tuple(cfg["n_search_range"])
    n_trials = cfg["n_trials"]

    print("=" * 60)
    print("Experiment 5: Sample Complexity Scaling")
    print("=" * 60)

    results = {}

    print("\n--- Part A: n_min vs K*d ---")
    Kd_values = []
    n_min_values = []
    recovery_curves = {}

    for d in d_values:
        K = int(eta * d)
        Kd = K * d
        print(f"\n  d={d}, K={K}, Kd={Kd}, eps={eps_target}")

        rng = np.random.default_rng(config["seed"])
        W = minimize_coherence(d, K, n_iter=3000, lr=0.02, rng=rng)

        n_min = find_min_n(W, s, sigma, 1.0, K, eps_target, n_range, n_trials,
                           config["seed"])

        Kd_values.append(Kd)
        n_min_values.append(n_min)
        print(f"    n_min = {n_min}")

        lambda_params = np.linspace(1.0, 3.0, K)
        model = SLRModel(W, s, sigma=sigma, B=1.0,
                         activation="bernoulli_exponential",
                         lambda_params=lambda_params)

        n_sweep = np.unique(np.logspace(
            np.log10(n_range[0]), np.log10(n_range[1]), 15
        ).astype(int))
        errs_mean = []
        errs_std = []
        for n_val in n_sweep:
            trial_errs = []
            for trial in range(n_trials):
                rng_t = np.random.default_rng(config["seed"] + trial)
                H = model.sample(int(n_val), rng=rng_t)
                try:
                    W_hat, _ = cbfr(H, K, use_fast=True)
                    max_err, _, _ = recovery_error(W, W_hat)
                    trial_errs.append(max_err)
                except Exception:
                    trial_errs.append(np.nan)
            errs_mean.append(np.nanmean(trial_errs))
            errs_std.append(np.nanstd(trial_errs))

        recovery_curves[d] = {
            "n_values": n_sweep,
            "error_mean": np.array(errs_mean),
            "error_std": np.array(errs_std),
        }

    results["part_a"] = {
        "Kd_values": np.array(Kd_values),
        "n_min_values": np.array([v if v is not None else np.nan for v in n_min_values]),
        "d_values": np.array(d_values),
    }

    print("\n--- Part B: n_min vs 1/eps^2 ---")
    d_fixed = cfg["d_fixed"]
    K_fixed = cfg["K_fixed"]
    eps_values = cfg["eps_values"]

    rng = np.random.default_rng(config["seed"])
    W_fixed = minimize_coherence(d_fixed, K_fixed, n_iter=3000, lr=0.02, rng=rng)

    inv_eps2_values = []
    n_min_eps_values = []

    for eps in eps_values:
        inv_eps2 = 1.0 / eps ** 2
        print(f"\n  d={d_fixed}, K={K_fixed}, eps={eps}, 1/eps^2={inv_eps2:.1f}")

        n_min = find_min_n(W_fixed, s, sigma, 1.0, K_fixed, eps, n_range,
                           n_trials, config["seed"])
        inv_eps2_values.append(inv_eps2)
        n_min_eps_values.append(n_min)
        print(f"    n_min = {n_min}")

    results["part_b"] = {
        "eps_values": np.array(eps_values),
        "inv_eps2_values": np.array(inv_eps2_values),
        "n_min_values": np.array([v if v is not None else np.nan for v in n_min_eps_values]),
    }

    data_path = os.path.join(results_dir, "exp5_results.npz")
    save_dict = {}
    for key, res in results.items():
        for k, v in res.items():
            if isinstance(v, np.ndarray):
                save_dict[f"{key}_{k}"] = v
    np.savez(data_path, **save_dict)
    print(f"Data saved: {data_path}")

    return results


if __name__ == "__main__":
    run_exp5()
