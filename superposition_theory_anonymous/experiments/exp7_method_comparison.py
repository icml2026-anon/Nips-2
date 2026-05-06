
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import ensure_results_dir, load_config
from src.dictionary import minimize_coherence
from src.slr_model import SLRModel
from src.cbfr import cbfr, recovery_error


def recover_pca(H, K, **kw):
    from sklearn.decomposition import PCA
    n, d = H.shape
    k = min(K, d)
    pca = PCA(n_components=k, random_state=42)
    pca.fit(H)
    W_hat = pca.components_.T
    if K > k:
        W_hat = np.hstack([W_hat, np.zeros((d, K - k))])
    norms = np.linalg.norm(W_hat, axis=0, keepdims=True)
    W_hat = W_hat / np.maximum(norms, 1e-12)
    return W_hat


def recover_fastica(H, K, **kw):
    from sklearn.decomposition import FastICA
    n, d = H.shape
    k = min(K, d)
    ica = FastICA(
        n_components=k,
        algorithm="parallel",
        whiten="unit-variance",
        fun="logcosh",
        max_iter=1000,
        tol=1e-4,
        random_state=42,
    )
    ica.fit(H)
    W_hat = ica.mixing_.copy()
    if K > k:
        W_hat = np.hstack([W_hat, np.zeros((d, K - k))])
    norms = np.linalg.norm(W_hat, axis=0, keepdims=True)
    W_hat = W_hat / np.maximum(norms, 1e-12)
    return W_hat


def recover_ksvd(H, K, **kw):
    from ksvd import ApproximateKSVD
    n, d = H.shape
    s = kw.get("s", 2)
    aksvd = ApproximateKSVD(
        n_components=K,
        max_iter=50,
        tol=1e-6,
        transform_n_nonzero_coefs=s + 1,
    )
    aksvd.fit(H)
    W_hat = aksvd.components_.T
    norms = np.linalg.norm(W_hat, axis=0, keepdims=True)
    W_hat = W_hat / np.maximum(norms, 1e-12)
    return W_hat


def recover_online_dl(H, K, **kw):
    from sklearn.decomposition import DictionaryLearning
    n, d = H.shape
    s = kw.get("s", 2)
    dl = DictionaryLearning(
        n_components=K,
        alpha=0.1,
        max_iter=200,
        transform_algorithm="omp",
        transform_n_nonzero_coefs=s,
        random_state=42,
    )
    dl.fit(H)
    W_hat = dl.components_.T
    norms = np.linalg.norm(W_hat, axis=0, keepdims=True)
    W_hat = W_hat / np.maximum(norms, 1e-12)
    return W_hat


class AnthropicSAE(nn.Module):

    def __init__(self, d_in: int, n_features: int):
        super().__init__()
        self.d_in = d_in
        self.n_features = n_features

        self.W_dec = nn.Parameter(torch.randn(d_in, n_features) * 0.01)
        self.W_enc = nn.Parameter(torch.randn(n_features, d_in) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        with torch.no_grad():
            self.W_enc.copy_(self.W_dec.T.clone())

    def forward(self, x):
        x_centered = x - self.b_dec
        z_pre = x_centered @ self.W_enc.T + self.b_enc
        f = torch.relu(z_pre)
        x_hat = f @ self.W_dec.T + self.b_dec
        return x_hat, f

    @torch.no_grad()
    def renorm_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)


def recover_sae(H, K, **kw):
    n, d = H.shape
    l1_coeff = kw.get("l1_coeff", 0.005)
    n_epochs = kw.get("n_epochs", 1000)
    lr = kw.get("lr", 3e-4)
    batch_size = min(512, n)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnthropicSAE(d, K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    H_tensor = torch.tensor(H, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(H_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model.train()
    for epoch in range(n_epochs):
        for (batch_x,) in loader:
            x_hat, f = model(batch_x)
            mse = ((batch_x - x_hat) ** 2).mean()
            l1 = l1_coeff * f.abs().mean()
            loss = mse + l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.renorm_decoder()

    W_hat = model.W_dec.detach().cpu().numpy()
    norms = np.linalg.norm(W_hat, axis=0, keepdims=True)
    W_hat = W_hat / np.maximum(norms, 1e-12)
    return W_hat


class TopKSAE(nn.Module):

    def __init__(self, d_in: int, n_features: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.n_features = n_features
        self.k = k

        self.W_dec = nn.Parameter(torch.randn(d_in, n_features) * 0.01)
        self.W_enc = nn.Parameter(torch.randn(n_features, d_in) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        with torch.no_grad():
            self.W_enc.copy_(self.W_dec.T.clone())

    def forward(self, x):
        x_centered = x - self.b_dec
        z_pre = x_centered @ self.W_enc.T + self.b_enc
        topk_vals, topk_idx = torch.topk(z_pre, self.k, dim=-1)
        f = torch.zeros_like(z_pre)
        f.scatter_(-1, topk_idx, torch.relu(topk_vals))
        x_hat = f @ self.W_dec.T + self.b_dec
        return x_hat, f

    @torch.no_grad()
    def renorm_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)


def recover_topk_sae(H, K, **kw):
    n, d = H.shape
    s = kw.get("s", 2)
    n_epochs = kw.get("n_epochs", 1000)
    lr = kw.get("lr", 3e-4)
    batch_size = min(512, n)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TopKSAE(d, K, k=s).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    H_tensor = torch.tensor(H, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(H_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model.train()
    for epoch in range(n_epochs):
        for (batch_x,) in loader:
            x_hat, f = model(batch_x)
            loss = ((batch_x - x_hat) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.renorm_decoder()

    W_hat = model.W_dec.detach().cpu().numpy()
    norms = np.linalg.norm(W_hat, axis=0, keepdims=True)
    W_hat = W_hat / np.maximum(norms, 1e-12)
    return W_hat


class GatedSAE(nn.Module):

    def __init__(self, d_in: int, n_features: int):
        super().__init__()
        self.d_in = d_in
        self.n_features = n_features

        self.W_dec = nn.Parameter(torch.randn(d_in, n_features) * 0.01)
        self.W_gate = nn.Parameter(torch.randn(n_features, d_in) * 0.01)
        self.b_gate = nn.Parameter(torch.zeros(n_features))
        self.r_mag = nn.Parameter(torch.zeros(n_features))
        self.b_mag = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        with torch.no_grad():
            self.W_gate.copy_(self.W_dec.T.clone())

    def forward(self, x):
        x_centered = x - self.b_dec
        pi_gate = x_centered @ self.W_gate.T + self.b_gate
        W_mag = self.W_gate * torch.exp(self.r_mag).unsqueeze(1)
        f_mag = torch.relu(x_centered @ W_mag.T + self.b_mag)
        gate_mask = (pi_gate > 0).float()
        f = gate_mask * f_mag
        x_hat = f @ self.W_dec.T + self.b_dec
        return x_hat, f, pi_gate

    @torch.no_grad()
    def renorm_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)


def recover_gated_sae(H, K, **kw):
    n, d = H.shape
    l1_coeff = kw.get("l1_coeff", 0.005)
    n_epochs = kw.get("n_epochs", 1000)
    lr = kw.get("lr", 3e-4)
    batch_size = min(512, n)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedSAE(d, K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    H_tensor = torch.tensor(H, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(H_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model.train()
    for epoch in range(n_epochs):
        for (batch_x,) in loader:
            x_hat, f, pi_gate = model(batch_x)
            mse = ((batch_x - x_hat) ** 2).mean()
            l1 = l1_coeff * pi_gate.abs().mean()
            loss = mse + l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.renorm_decoder()

    W_hat = model.W_dec.detach().cpu().numpy()
    norms = np.linalg.norm(W_hat, axis=0, keepdims=True)
    W_hat = W_hat / np.maximum(norms, 1e-12)
    return W_hat


class JumpReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, theta, bandwidth):
        mask = (z > theta).float()
        ctx.save_for_backward(z, theta, mask)
        ctx.bandwidth = bandwidth
        return z * mask

    @staticmethod
    def backward(ctx, grad_output):
        z, theta, mask = ctx.saved_tensors
        eps = ctx.bandwidth
        grad_z = grad_output * mask
        sig = torch.sigmoid((z - theta) / eps)
        kernel = sig * (1.0 - sig) / eps
        grad_theta = -(grad_output * z * kernel).sum(dim=0)
        return grad_z, grad_theta, None


class HeavisideSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z, theta, bandwidth):
        ctx.save_for_backward(z, theta)
        ctx.bandwidth = bandwidth
        return (z > theta).float()

    @staticmethod
    def backward(ctx, grad_output):
        z, theta = ctx.saved_tensors
        eps = ctx.bandwidth
        sig = torch.sigmoid((z - theta) / eps)
        kernel = sig * (1.0 - sig) / eps
        grad_theta = -(grad_output * kernel).sum(dim=0)
        return None, grad_theta, None


class JumpReLUSAE(nn.Module):

    def __init__(self, d_in: int, n_features: int, bandwidth: float = 0.1):
        super().__init__()
        self.d_in = d_in
        self.n_features = n_features
        self.bandwidth = bandwidth

        self.W_dec = nn.Parameter(torch.randn(d_in, n_features) * 0.01)
        self.W_enc = nn.Parameter(torch.randn(n_features, d_in) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.log_theta = nn.Parameter(torch.full((n_features,), -6.0))

        with torch.no_grad():
            self.W_enc.copy_(self.W_dec.T.clone())

    @property
    def theta(self):
        return torch.exp(self.log_theta)

    def forward(self, x):
        x_centered = x - self.b_dec
        z_pre = x_centered @ self.W_enc.T + self.b_enc
        theta = self.theta
        f = JumpReLU.apply(z_pre, theta, self.bandwidth)
        l0 = HeavisideSTE.apply(z_pre, theta, self.bandwidth).sum(dim=-1).mean()
        x_hat = f @ self.W_dec.T + self.b_dec
        return x_hat, f, l0

    @torch.no_grad()
    def renorm_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)


def recover_jumprelu_sae(H, K, **kw):
    n, d = H.shape
    s = kw.get("s", 2)
    l0_coeff = kw.get("l0_coeff", 1e-3)
    n_epochs = kw.get("n_epochs", 1000)
    lr = kw.get("lr", 3e-4)
    batch_size = min(512, n)
    bandwidth = kw.get("bandwidth", 0.1)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JumpReLUSAE(d, K, bandwidth=bandwidth).to(device)
    main_params = [p for name, p in model.named_parameters()
                   if name != "log_theta"]
    optimizer = optim.Adam([
        {"params": main_params, "lr": lr},
        {"params": [model.log_theta], "lr": lr * 10},
    ], betas=(0.9, 0.999))

    H_tensor = torch.tensor(H, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(H_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    model.train()
    for epoch in range(n_epochs):
        for (batch_x,) in loader:
            x_hat, f, l0 = model(batch_x)
            mse = ((batch_x - x_hat) ** 2).mean()
            loss = mse + l0_coeff * l0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.renorm_decoder()

    W_hat = model.W_dec.detach().cpu().numpy()
    norms = np.linalg.norm(W_hat, axis=0, keepdims=True)
    W_hat = W_hat / np.maximum(norms, 1e-12)
    return W_hat


def recover_cbfr(H, K, **kw):
    W_hat, _ = cbfr(H.T, K, use_fast=True)
    return W_hat


METHODS = [
    ("CBFR",          recover_cbfr),
    ("SAE",           recover_sae),
    ("TopK SAE",      recover_topk_sae),
    ("Gated SAE",     recover_gated_sae),
    ("JumpReLU SAE",  recover_jumprelu_sae),
    ("FastICA",       recover_fastica),
    ("K-SVD",         recover_ksvd),
    ("Online DL",     recover_online_dl),
    ("PCA",           recover_pca),
]


def run_exp7(config=None):
    if config is None:
        config = load_config()
    results_dir = ensure_results_dir()

    cfg = config.get("exp7_comparison", {})
    d = cfg.get("d", 10)
    K = cfg.get("K", 8)
    s = cfg.get("s", 2)
    sigma = cfg.get("sigma", 0.05)
    n_values = cfg.get("n_values", [1000, 2000, 5000, 10000, 20000, 50000])
    n_trials = cfg.get("n_trials", 8)
    sae_epochs = cfg.get("sae_epochs", 500)

    print("=" * 60)
    print("Experiment 7: Feature Recovery Method Comparison")
    print("=" * 60)
    print(f"  d={d}, K={K}, s={s}, sigma={sigma}")
    print(f"  Methods: {[m for m, _ in METHODS]}")

    method_names = [m for m, _ in METHODS]
    all_results = {m: {"max_err_mean": [], "max_err_std": [],
                       "mean_err_mean": [], "mean_err_std": [],
                       "time_mean": [], "time_std": []}
                   for m in method_names}

    for n_val in n_values:
        print(f"\n  n = {n_val}")
        method_errors = {m: {"max": [], "mean": [], "time": []} for m in method_names}

        for trial in range(n_trials):
            rng = np.random.default_rng(config.get("seed", 42) + trial)
            W = minimize_coherence(d, K, n_iter=3000, lr=0.01, rng=rng)
            lambda_params = np.linspace(1.0, 3.0, K)
            model = SLRModel(W, s, sigma=sigma, B=1.0,
                             activation="bernoulli_exponential",
                             lambda_params=lambda_params)
            H = model.sample(n_val, rng=rng).T

            for method_name, method_fn in METHODS:
                try:
                    t0 = time.perf_counter()
                    W_hat = method_fn(H, K, s=s, n_epochs=sae_epochs)
                    elapsed = time.perf_counter() - t0
                    max_err, mean_err, _ = recovery_error(W, W_hat)
                except Exception as e:
                    max_err, mean_err, elapsed = np.nan, np.nan, np.nan

                method_errors[method_name]["max"].append(max_err)
                method_errors[method_name]["mean"].append(mean_err)
                method_errors[method_name]["time"].append(elapsed)

            if trial == 0:
                errs = ", ".join(
                    f"{m}={np.nanmean(method_errors[m]['max']):.3f}"
                    for m in method_names
                )
                print(f"    trial 1: {errs}")

        for m in method_names:
            all_results[m]["max_err_mean"].append(
                np.nanmean(method_errors[m]["max"]))
            all_results[m]["max_err_std"].append(
                np.nanstd(method_errors[m]["max"]))
            all_results[m]["mean_err_mean"].append(
                np.nanmean(method_errors[m]["mean"]))
            all_results[m]["mean_err_std"].append(
                np.nanstd(method_errors[m]["mean"]))
            all_results[m]["time_mean"].append(
                np.nanmean(method_errors[m]["time"]))
            all_results[m]["time_std"].append(
                np.nanstd(method_errors[m]["time"]))

        for m in method_names:
            print(f"    {m:10s}: max_err={all_results[m]['max_err_mean'][-1]:.4f} "
                  f"± {all_results[m]['max_err_std'][-1]:.4f}")

    for m in method_names:
        for k in all_results[m]:
            all_results[m][k] = np.array(all_results[m][k])

    save_dict = {"n_values": np.array(n_values), "d": d, "K": K, "s": s}
    for m in method_names:
        safe_m = m.replace(" ", "_")
        for k, v in all_results[m].items():
            save_dict[f"{safe_m}_{k}"] = v
    data_path = os.path.join(results_dir, "exp7_results.npz")
    np.savez(data_path, **save_dict)
    print(f"\nData saved: {data_path}")

    return all_results


if __name__ == "__main__":
    run_exp7()
