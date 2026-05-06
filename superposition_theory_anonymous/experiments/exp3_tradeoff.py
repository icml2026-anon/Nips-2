
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import (
    ensure_results_dir, load_config,
    tau_coefficient, critical_dimension, max_features, asymptotic_plateau,
)
from src.dictionary import (
    random_dictionary, mutual_coherence, minimize_coherence,
    generate_clustered_dictionary,
)
from src.interpretability import interpretability_index


def run_exp3(config=None):
    if config is None:
        config = load_config()
    cfg = config["exp3_tradeoff"]
    results_dir = ensure_results_dir()
    n_dicts = cfg["n_random_dicts"]
    n_support_samples = cfg["n_support_samples"]

    print("=" * 60)
    print("Experiment 3: Interpretability-Efficiency Tradeoff")
    print("=" * 60)

    results = {}

    sub_cfg = cfg["subcritical"]
    d, s = sub_cfg["d"], sub_cfg["s"]
    d_star = critical_dimension(s)
    K_star = max_features(d, s)
    eta_min, eta_max = sub_cfg["eta_range"]
    eta_steps = sub_cfg["eta_steps"]

    print(f"\n--- Sub-critical: d={d}, s={s}, d*={d_star}, K*={K_star} ---")
    print(f"    d > d*: {d} > {d_star} = True => full interpretability expected")

    etas_sub = np.linspace(eta_min, eta_max, eta_steps)
    interp_sub_mean = []
    interp_sub_std = []

    for eta in etas_sub:
        K = max(int(np.round(eta * d)), d + 1)
        tau = tau_coefficient(d, K, s)
        vals = []
        rng = np.random.default_rng(config["seed"])
        for _ in range(n_dicts):
            W = random_dictionary(d, K, rng)
            I = interpretability_index(W, s, n_samples=n_support_samples, rng=rng)
            vals.append(I)
        interp_sub_mean.append(np.mean(vals))
        interp_sub_std.append(np.std(vals))
        print(f"  eta={eta:.2f}, K={K}, tau={tau:.4f}, I={np.mean(vals):.4f}")

    results["subcritical"] = {
        "d": d, "s": s, "d_star": d_star,
        "etas": etas_sub,
        "interp_mean": np.array(interp_sub_mean),
        "interp_std": np.array(interp_sub_std),
    }

    sup_cfg = cfg["supercritical"]
    d, s = sup_cfg["d"], sup_cfg["s"]
    d_star = critical_dimension(s)
    K_star_val = max_features(d, s)
    eta_star = K_star_val / d if K_star_val != np.inf else np.inf
    eta_min, eta_max = sup_cfg["eta_range"]
    eta_steps = sup_cfg["eta_steps"]

    print(f"\n--- Super-critical: d={d}, s={s}, d*={d_star}, K*={K_star_val} ---")
    if eta_star != np.inf:
        print(f"    d <= d*: {d} <= {d_star} = True => decay beyond eta*={eta_star:.2f}")
    else:
        print(f"    d <= d*: {d} <= {d_star}, K*=inf")

    etas_sup = np.linspace(eta_min, eta_max, eta_steps)
    interp_sup_mean = []
    interp_sup_std = []

    for eta in etas_sup:
        K = max(int(np.round(eta * d)), d + 1)
        tau = tau_coefficient(d, K, s)
        vals = []
        rng = np.random.default_rng(config["seed"])
        for _ in range(n_dicts):
            W = minimize_coherence(d, K, n_iter=2000, lr=0.015, rng=rng)
            I = interpretability_index(W, s, n_samples=n_support_samples, rng=rng)
            vals.append(I)
        interp_sup_mean.append(np.mean(vals))
        interp_sup_std.append(np.std(vals))
        print(f"  eta={eta:.2f}, K={K}, tau={tau:.4f}, I={np.mean(vals):.4f}")

    results["supercritical"] = {
        "d": d, "s": s, "d_star": d_star,
        "K_star": K_star_val, "eta_star": eta_star,
        "etas": etas_sup,
        "interp_mean": np.array(interp_sup_mean),
        "interp_std": np.array(interp_sup_std),
    }

    plat_cfg = cfg["plateau"]
    d_plat, s_plat = plat_cfg["d"], plat_cfg["s"]
    d_star_plat = critical_dimension(s_plat)
    K_star_plat = max_features(d_plat, s_plat)
    m_values = plat_cfg["m_values"]

    print(f"\n--- Asymptotic Plateau: d={d_plat}, s={s_plat}, d*={d_star_plat}, K*={K_star_plat} ---")

    rng = np.random.default_rng(config["seed"])
    if K_star_plat == np.inf or K_star_plat > 500:
        K_base = min(d_plat, 20)
    else:
        K_base = max(d_plat + 1, int(0.6 * K_star_plat))
    W_base = minimize_coherence(d_plat, K_base, n_iter=10000, lr=0.01, rng=rng)
    mu_base = mutual_coherence(W_base)
    de_thresh = 1.0 / (2 * s_plat - 1)
    print(f"  K_base={K_base}, mu(W_base)={mu_base:.4f}, DE_thresh={de_thresh:.4f}, OK={mu_base < de_thresh}")

    plateau_etas = []
    plateau_interp_mean = []
    plateau_interp_std = []
    theoretical_plat = asymptotic_plateau(s_plat, K_base)

    for m in m_values:
        K = m * K_base
        eta = K / d_plat
        vals = []
        for trial in range(min(n_dicts, 10)):
            rng_trial = np.random.default_rng(config["seed"] + trial)
            if m == 1:
                W_test = W_base.copy()
            else:
                W_test = generate_clustered_dictionary(
                    d_plat, W_base, m, epsilon=0.02, rng=rng_trial
                )
            I = interpretability_index(W_test, s_plat, n_samples=n_support_samples, rng=rng_trial)
            vals.append(I)
        plateau_etas.append(eta)
        plateau_interp_mean.append(np.mean(vals))
        plateau_interp_std.append(np.std(vals))
        print(f"  m={m}, K={K}, eta={eta:.1f}, I={np.mean(vals):.4f}")

    print(f"  Theoretical plateau lower bound: {theoretical_plat:.4f}")

    results["plateau"] = {
        "d": d_plat, "s": s_plat, "d_star": d_star_plat,
        "K_base": K_base,
        "m_values": np.array(m_values),
        "etas": np.array(plateau_etas),
        "interp_mean": np.array(plateau_interp_mean),
        "interp_std": np.array(plateau_interp_std),
        "theoretical_plateau": theoretical_plat,
    }

    data_path = os.path.join(results_dir, "exp3_results.npz")
    save_dict = {}
    for regime, res in results.items():
        for k, v in res.items():
            if isinstance(v, (np.ndarray, float, int)):
                save_dict[f"{regime}_{k}"] = v
    np.savez(data_path, **save_dict)
    print(f"Data saved: {data_path}")

    return results


if __name__ == "__main__":
    run_exp3()
