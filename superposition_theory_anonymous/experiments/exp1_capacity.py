
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    ensure_results_dir, set_seed, load_config,
    tau_coefficient, welch_bound, donoho_elad_threshold,
)
from src.dictionary import random_dictionary, mutual_coherence, minimize_coherence
from src.interpretability import interpretability_index


def _K_from_tau(d, s, tau_target):
    c = (2 * s - 1) ** 2
    denom = tau_target * d - c
    if abs(denom) < 1e-12:
        return None
    K_val = d * (tau_target - c) / denom
    K_int = int(np.round(K_val))
    if K_int <= d:
        return None
    return K_int


def run_exp1(config=None):
    if config is None:
        config = load_config()
    cfg = config["exp1_capacity"]
    results_dir = ensure_results_dir()

    d_values = cfg["d_values"]
    s_values = cfg["s_values"]
    n_dicts = min(cfg.get("n_random_dicts", 20), 20)
    n_support_samples = cfg.get("n_support_samples", 10000)

    tau_targets = np.concatenate([
        np.linspace(0.05, 0.9, 10),
        np.linspace(0.92, 1.08, 9),
        np.linspace(1.15, 2.5, 8),
    ])

    print("=" * 60)
    print("Experiment 1: Capacity Phase Transition")
    print("=" * 60)
    print(f"  tau sweep: {len(tau_targets)} points in [{tau_targets[0]:.2f}, {tau_targets[-1]:.2f}]")
    print(f"  n_dicts={n_dicts}, n_support_samples={n_support_samples}")

    all_results = {}

    for d in d_values:
        for s in s_values:
            print(f"\n--- d={d}, s={s} ---")
            threshold = donoho_elad_threshold(s)

            taus = []
            interp_means = []
            interp_stds = []
            coherence_means = []
            seen_K = set()

            for tau_t in tau_targets:
                K = _K_from_tau(d, s, tau_t)
                if K is None or K <= d or K > min(200, 20 * d):
                    continue
                if K in seen_K:
                    continue
                seen_K.add(K)

                tau = tau_coefficient(d, K, s)

                interp_vals = []
                coh_vals = []

                for trial in range(n_dicts):
                    rng = np.random.default_rng(config["seed"] + trial)
                    W = minimize_coherence(d, K, n_iter=5000, lr=0.01, rng=rng)
                    mu = mutual_coherence(W)
                    coh_vals.append(mu)
                    I = interpretability_index(W, s, n_samples=n_support_samples, rng=rng)
                    interp_vals.append(I)

                interp_means.append(np.mean(interp_vals))
                interp_stds.append(np.std(interp_vals))
                coherence_means.append(np.mean(coh_vals))
                taus.append(tau)

                status = "FEASIBLE" if tau < 1 else "IMPOSSIBLE"
                print(f"  K={K:4d}, tau={tau:.3f} [{status}], "
                      f"I={np.mean(interp_vals):.3f}+/-{np.std(interp_vals):.3f}, "
                      f"mu={np.mean(coh_vals):.4f} (thresh={threshold:.4f})")

            if not taus:
                print(f"  SKIPPED (no valid K values)")
                continue

            order = np.argsort(taus)
            all_results[(d, s)] = {
                "etas": np.array(taus)[order],
                "taus": np.array(taus)[order],
                "interp_means": np.array(interp_means)[order],
                "interp_stds": np.array(interp_stds)[order],
                "coherence_means": np.array(coherence_means)[order],
            }

    data_path = os.path.join(results_dir, "exp1_results.npz")
    np.savez(data_path, **{
        f"d{d}_s{s}_{k}": v
        for (d, s), res in all_results.items()
        for k, v in res.items()
    })
    print(f"Data saved: {data_path}")

    return all_results


if __name__ == "__main__":
    run_exp1()
