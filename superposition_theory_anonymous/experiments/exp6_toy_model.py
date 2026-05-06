
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import ensure_results_dir, load_config, tau_coefficient
from src.dictionary import mutual_coherence
from src.interpretability import interpretability_index


def coherence_penalty(weight):
    W = weight / (weight.norm(dim=0, keepdim=True) + 1e-12)
    G = W.T @ W
    K = G.shape[0]
    mask = 1.0 - torch.eye(K, device=G.device)
    abs_G = torch.abs(G) * mask
    beta = 20.0
    return torch.logsumexp(beta * abs_G.view(-1), dim=0) / beta


class ToyBottleneckModel(nn.Module):

    def __init__(self, K, d, importance=None):
        super().__init__()
        self.K = K
        self.d = d
        self.W = nn.Parameter(torch.randn(d, K) * (1.0 / d ** 0.5))
        self.b_dec = nn.Parameter(torch.zeros(K))
        if importance is None:
            importance = torch.ones(K)
        self.register_buffer("importance", importance)

    def forward(self, x):
        h = x @ self.W.T
        h = torch.relu(h)
        x_hat = h @ self.W + self.b_dec
        return x_hat, h

    def get_dictionary(self):
        W = self.W.detach().cpu().numpy().copy()
        norms = np.linalg.norm(W, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        W = W / norms
        return W


def generate_sparse_data(K, s, n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    X = np.zeros((n, K), dtype=np.float32)
    for i in range(n):
        support = rng.choice(K, size=s, replace=False)
        X[i, support] = rng.exponential(1.0, size=s)
    return X


def train_toy_model(K, d, s, n_train=50000, n_epochs=500, lr=1e-3,
                    batch_size=512, seed=42, importance_decay=0.0,
                    lambda_coh=0.0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if importance_decay > 0:
        importance = torch.tensor(
            [(1 - importance_decay) ** i for i in range(K)], dtype=torch.float32
        )
    else:
        importance = torch.ones(K)

    model = ToyBottleneckModel(K, d, importance=importance).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    X_train = generate_sparse_data(K, s, n_train, rng=rng)
    X_tensor = torch.tensor(X_train, device=device)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for (batch_x,) in loader:
            x_hat, h = model(batch_x)
            recon_loss = ((batch_x - x_hat) ** 2 * model.importance.unsqueeze(0)).mean()
            loss = recon_loss
            if lambda_coh > 0:
                loss = loss + lambda_coh * coherence_penalty(model.W)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += recon_loss.item() * batch_x.shape[0]
        scheduler.step()

    W_learned = model.get_dictionary()
    final_loss = total_loss / n_train
    return W_learned, final_loss


def _run_one_condition(d, s, eta_values, n_train, n_epochs, n_trials,
                       importance_decay, lambda_coh, base_seed, tag,
                       batch_size=4096):
    etas_out, taus_out = [], []
    I_means, I_stds = [], []
    mu_means, mu_stds = [], []
    losses = []

    for eta in eta_values:
        K = max(int(np.round(eta * d)), 2)
        tau = tau_coefficient(d, K, s)
        print(f"\n  [{tag}] eta={eta:.2f}, K={K}, tau={tau:.4f}")

        trial_Is, trial_mus, trial_losses = [], [], []

        for trial in range(n_trials):
            seed = base_seed + trial * 100

            W_learned, loss = train_toy_model(
                K, d, s,
                n_train=n_train, n_epochs=n_epochs,
                lr=1e-3, batch_size=batch_size, seed=seed,
                importance_decay=importance_decay,
                lambda_coh=lambda_coh,
            )

            mu = mutual_coherence(W_learned)
            rng_eval = np.random.default_rng(seed + 999)
            n_samp = 3000 if K <= 40 else 2000
            I_val = interpretability_index(W_learned, s, n_samples=n_samp, rng=rng_eval)

            trial_Is.append(I_val)
            trial_mus.append(mu)
            trial_losses.append(loss)

            print(f"    trial {trial+1}/{n_trials}: loss={loss:.4f}, "
                  f"mu={mu:.4f}, I={I_val:.4f}")

        etas_out.append(eta)
        taus_out.append(tau)
        I_means.append(np.mean(trial_Is))
        I_stds.append(np.std(trial_Is))
        mu_means.append(np.mean(trial_mus))
        mu_stds.append(np.std(trial_mus))
        losses.append(np.mean(trial_losses))

    return {
        "etas": np.array(etas_out),
        "taus": np.array(taus_out),
        "interp_mean": np.array(I_means),
        "interp_std": np.array(I_stds),
        "coherence_mean": np.array(mu_means),
        "coherence_std": np.array(mu_stds),
        "train_loss": np.array(losses),
    }


def run_exp6(config=None):
    if config is None:
        config = load_config()
    results_dir = ensure_results_dir()

    cfg = config.get("exp6_toy_model", {})
    d = cfg.get("d", 16)
    s = cfg.get("s", 3)
    eta_values = cfg.get("eta_values",
        [0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0])
    n_train = cfg.get("n_train", 50000)
    n_epochs = cfg.get("n_epochs", 800)
    n_trials = cfg.get("n_trials", 5)
    importance_decay = cfg.get("importance_decay", 0.0)
    lambda_coh = cfg.get("lambda_coh", 1.0)
    batch_size = cfg.get("batch_size", 4096)
    base_seed = config.get("seed", 42)

    print("=" * 60)
    print("Experiment 6: Toy Trained Model Superposition")
    print("=" * 60)
    print(f"  d={d}, s={s}, n_train={n_train}, n_epochs={n_epochs}, bs={batch_size}")
    print(f"  lambda_coh={lambda_coh}")

    results = {"d": d, "s": s, "eta_values": np.array(eta_values),
               "lambda_coh": lambda_coh}

    print("\n>>> Condition A: No coherence regularisation (lambda=0)")
    cond_a = _run_one_condition(
        d, s, eta_values, n_train, n_epochs, n_trials,
        importance_decay, lambda_coh=0.0, base_seed=base_seed, tag="base",
        batch_size=batch_size)
    for k, v in cond_a.items():
        results[f"base_{k}"] = v
    for k, v in cond_a.items():
        results[k] = v

    print(f"\n>>> Condition B: Coherence regularisation (lambda={lambda_coh})")
    cond_b = _run_one_condition(
        d, s, eta_values, n_train, n_epochs, n_trials,
        importance_decay, lambda_coh=lambda_coh, base_seed=base_seed, tag="reg",
        batch_size=batch_size)
    for k, v in cond_b.items():
        results[f"reg_{k}"] = v

    data_path = os.path.join(results_dir, "exp6_results.npz")
    np.savez(data_path, **results)
    print(f"\nData saved: {data_path}")

    return results


if __name__ == "__main__":
    run_exp6()
