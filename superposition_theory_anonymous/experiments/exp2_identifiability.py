
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import ensure_results_dir, load_config
from src.dictionary import random_dictionary, mutual_coherence, minimize_coherence
from src.slr_model import SLRModel
from src.cbfr import cbfr, recovery_error


def run_exp2(config=None):
    if config is None:
        config = load_config()
    cfg = config["exp2_identifiability"]
    results_dir = ensure_results_dir()

    d = cfg["d"]
    K_values = cfg["K_values"]
    s = cfg["s"]
    sigma = cfg["sigma"]
    n_values = cfg["n_values"]
    n_trials = cfg["n_trials"]

    print("=" * 60)
    print("Experiment 2: CBFR Identifiability")
    print("=" * 60)

    all_results = {}

    print("\n--- Part A: Non-Gaussian (Bernoulli-Exponential) activations ---")

    for K in K_values:
        print(f"\n  d={d}, K={K}, s={s}")
        max_errors = {n_val: [] for n_val in n_values}
        mean_errors = {n_val: [] for n_val in n_values}

        for trial in range(n_trials):
            rng = np.random.default_rng(config["seed"] + trial)

            W = minimize_coherence(d, K, n_iter=3000, lr=0.02, rng=rng)
            mu = mutual_coherence(W)
            print(f"    Trial {trial+1}/{n_trials}, mu(W)={mu:.4f}")

            lambda_params = np.linspace(1.0, 3.0, K)

            model = SLRModel(W, s, sigma=sigma, B=1.0,
                             activation="bernoulli_exponential",
                             lambda_params=lambda_params)

            for n_val in n_values:
                H = model.sample(n_val, rng=rng)
                try:
                    W_hat, evals = cbfr(H, K, use_fast=True)
                    max_err, mean_err, _ = recovery_error(W, W_hat)
                except Exception as e:
                    print(f"      n={n_val}: CBFR failed ({e})")
                    max_err = np.nan
                    mean_err = np.nan

                max_errors[n_val].append(max_err)
                mean_errors[n_val].append(mean_err)

                if not np.isnan(max_err):
                    print(f"      n={n_val}: max_err={max_err:.4f}, mean_err={mean_err:.4f}")

        all_results[f"nongaussian_K{K}"] = {
            "n_values": np.array(n_values),
            "max_errors_mean": np.array([np.nanmean(max_errors[n]) for n in n_values]),
            "max_errors_std": np.array([np.nanstd(max_errors[n]) for n in n_values]),
            "mean_errors_mean": np.array([np.nanmean(mean_errors[n]) for n in n_values]),
            "mean_errors_std": np.array([np.nanstd(mean_errors[n]) for n in n_values]),
        }

    print("\n--- Part B: Gaussian activations (non-identifiability test) ---")
    K_gauss = K_values[0]
    rng = np.random.default_rng(config["seed"])
    W = minimize_coherence(d, K_gauss, n_iter=3000, lr=0.02, rng=rng)

    gauss_errors = []
    for n_val in n_values:
        trial_errors = []
        for trial in range(n_trials):
            rng_trial = np.random.default_rng(config["seed"] + 1000 + trial)
            model = SLRModel(W, s, sigma=sigma, B=3.0,
                             activation="gaussian")
            H = model.sample(n_val, rng=rng_trial)
            try:
                W_hat, _ = cbfr(H, K_gauss, use_fast=True)
                max_err, _, _ = recovery_error(W, W_hat)
            except Exception:
                max_err = np.nan
            trial_errors.append(max_err)
        gauss_errors.append(np.nanmean(trial_errors))
        print(f"  n={n_val}: Gaussian max_err={np.nanmean(trial_errors):.4f}")

    all_results["gaussian"] = {
        "n_values": np.array(n_values),
        "max_errors_mean": np.array(gauss_errors),
    }

    data_path = os.path.join(results_dir, "exp2_results.npz")
    save_dict = {}
    for key, res in all_results.items():
        for k, v in res.items():
            save_dict[f"{key}_{k}"] = v
    np.savez(data_path, **save_dict)
    print(f"Data saved: {data_path}")

    return all_results


if __name__ == "__main__":
    run_exp2()
