
import numpy as np


class SLRModel:

    def __init__(self, W, s, sigma=0.1, B=1.0, activation="bernoulli_exponential",
                 lambda_params=None):
        self.W = np.array(W, dtype=np.float64)
        self.d, self.K = self.W.shape
        self.s = s
        self.sigma = sigma
        self.B = B
        self.activation = activation

        if lambda_params is None:
            self.lambda_params = np.ones(self.K) * 2.0
        else:
            self.lambda_params = np.array(lambda_params, dtype=np.float64)

        norms = np.linalg.norm(self.W, axis=0)
        assert np.allclose(norms, 1.0, atol=1e-6), "Dictionary columns must be unit-norm."

    def sample_activations(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        if self.activation == "bernoulli_exponential":
            p = min(self.s / self.K, 1.0)
            Z = rng.binomial(1, p, size=(self.K, n)).astype(np.float64)
            E = rng.exponential(1.0 / self.lambda_params[:, None],
                                size=(self.K, n))
            Phi = Z * E
            Phi = np.clip(Phi, 0.0, self.B)

        elif self.activation == "bernoulli_uniform":
            p = min(self.s / self.K, 1.0)
            Z = rng.binomial(1, p, size=(self.K, n)).astype(np.float64)
            U = rng.uniform(0.0, self.B, size=(self.K, n))
            Phi = Z * U

        elif self.activation == "gaussian":
            Phi = rng.standard_normal((self.K, n)) * 0.3
            Phi = np.clip(Phi, -self.B, self.B)

        else:
            raise ValueError(f"Unknown activation type: {self.activation}")

        return Phi

    def sample(self, n, rng=None, return_phi=False):
        if rng is None:
            rng = np.random.default_rng()

        Phi = self.sample_activations(n, rng)
        H = self.W @ Phi

        if self.sigma > 0:
            noise = rng.standard_normal((self.d, n)) * self.sigma
            H = H + noise

        if return_phi:
            return H, Phi
        return H

    def excess_kurtosis(self, n_samples=100000, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        Phi = self.sample_activations(n_samples, rng)
        kurtosis = np.zeros(self.K)
        for k in range(self.K):
            phi_k = Phi[k, :]
            mu = np.mean(phi_k)
            var = np.var(phi_k)
            if var < 1e-12:
                kurtosis[k] = 0.0
            else:
                m4 = np.mean((phi_k - mu) ** 4)
                kurtosis[k] = m4 / var ** 2 - 3.0
        return kurtosis

    def effective_kurtosis(self, n_samples=100000, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        Phi = self.sample_activations(n_samples, rng)
        lambda_k = np.zeros(self.K)
        for k in range(self.K):
            phi_k = Phi[k, :]
            mu = np.mean(phi_k)
            var = np.var(phi_k)
            if var < 1e-12:
                lambda_k[k] = 0.0
            else:
                m4 = np.mean((phi_k - mu) ** 4)
                kappa = m4 / var ** 2 - 3.0
                lambda_k[k] = kappa * var ** 2
        return lambda_k
