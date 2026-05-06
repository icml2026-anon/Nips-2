
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import ensure_results_dir, load_config, donoho_elad_threshold
from src.dictionary import random_dictionary, mutual_coherence, minimize_coherence
from src.generalization import (
    generate_binary_task, measure_generalization_gap, theoretical_gen_bound,
)


def run_exp4(config=None):
    if config is None:
        config = load_config()
    cfg = config["exp4_generalization"]
    results_dir = ensure_results_dir()

    s_default = cfg["s"]
    B = cfg["B"]
    sigma = cfg["sigma"]
    n_trials = cfg["n_trials"]
    n_test = cfg["n_test"]

    print("=" * 60)
    print("Experiment 4: Generalization Bound")
    print("=" * 60)

    results = {}

    print("\n--- Part A: Vary d, fixed eta and s ---")
    d_values = cfg["d_values"]
    eta = cfg["eta"]
    n_train_values = cfg["n_train_values"]

    for d in d_values:
        K = int(eta * d)
        print(f"\n  d={d}, K={K}, s={s_default}, eta={eta}")
        gaps_by_n = {n_tr: [] for n_tr in n_train_values}

        for trial in range(n_trials):
            rng = np.random.default_rng(config["seed"] + trial)
            W = random_dictionary(d, K, rng)

            u_star = rng.standard_normal(d)
            u_star /= np.linalg.norm(u_star)

            for n_tr in n_train_values:
                rng_train = np.random.default_rng(config["seed"] + trial * 1000 + n_tr)
                H_train, y_train, _ = generate_binary_task(
                    W, s_default, n_tr, sigma=sigma, B=B, rng=rng_train,
                    u_star=u_star
                )
                rng_test = np.random.default_rng(config["seed"] + trial + 10000)
                H_test, y_test, _ = generate_binary_task(
                    W, s_default, n_test, sigma=sigma, B=B, rng=rng_test,
                    u_star=u_star
                )

                gap, _, _ = measure_generalization_gap(H_train, y_train, H_test, y_test)
                gaps_by_n[n_tr].append(gap)

        gap_means = np.array([np.mean(gaps_by_n[n]) for n in n_train_values])
        gap_stds = np.array([np.std(gaps_by_n[n]) for n in n_train_values])
        results[f"vary_d_{d}"] = {
            "d": d, "K": K, "s": s_default,
            "n_train_values": np.array(n_train_values),
            "gap_mean": gap_means,
            "gap_std": gap_stds,
        }
        print(f"    Gaps: {dict(zip(n_train_values, np.round(gap_means, 4)))}")

    print("\n--- Part B: Vary s, fixed d ---")
    d_fixed = cfg["d_fixed"]
    s_values = cfg["s_values"]

    for s in s_values:
        K = int(eta * d_fixed)
        print(f"\n  d={d_fixed}, K={K}, s={s}, eta={eta}")
        gaps_by_n = {n_tr: [] for n_tr in n_train_values}

        for trial in range(n_trials):
            rng = np.random.default_rng(config["seed"] + trial + 5000)
            W = random_dictionary(d_fixed, K, rng)

            u_star = rng.standard_normal(d_fixed)
            u_star /= np.linalg.norm(u_star)

            for n_tr in n_train_values:
                rng_train = np.random.default_rng(config["seed"] + trial * 1000 + n_tr + 5000)
                H_train, y_train, _ = generate_binary_task(
                    W, s, n_tr, sigma=sigma, B=B, rng=rng_train,
                    u_star=u_star
                )
                rng_test = np.random.default_rng(config["seed"] + trial + 15000)
                H_test, y_test, _ = generate_binary_task(
                    W, s, n_test, sigma=sigma, B=B, rng=rng_test,
                    u_star=u_star
                )
                gap, _, _ = measure_generalization_gap(H_train, y_train, H_test, y_test)
                gaps_by_n[n_tr].append(gap)

        gap_means = np.array([np.mean(gaps_by_n[n]) for n in n_train_values])
        gap_stds = np.array([np.std(gaps_by_n[n]) for n in n_train_values])
        results[f"vary_s_{s}"] = {
            "d": d_fixed, "K": K, "s": s,
            "n_train_values": np.array(n_train_values),
            "gap_mean": gap_means,
            "gap_std": gap_stds,
        }
        print(f"    Gaps: {dict(zip(n_train_values, np.round(gap_means, 4)))}")

    data_path = os.path.join(results_dir, "exp4_results.npz")
    save_dict = {}
    for key, res in results.items():
        for k, v in res.items():
            if isinstance(v, (np.ndarray, float, int)):
                save_dict[f"{key}_{k}"] = v
    np.savez(data_path, **save_dict)
    print(f"Data saved: {data_path}")

    return results


if __name__ == "__main__":
    run_exp4()
