
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import ensure_results_dir, load_config
from src.cbfr import cbfr
from src.dictionary import mutual_coherence


MINI_CORPUS = """
The sun rose slowly over the mountains, casting long golden shadows across the valley below.
Scientists recently discovered a new species of deep-sea fish near hydrothermal vents in the Pacific Ocean.
The ancient library contained thousands of manuscripts written in languages long forgotten by scholars.
She walked into the room and immediately noticed something was different about the arrangement of furniture.
Economic forecasters predict a moderate slowdown in global trade over the next three quarters.
The protein folding problem remained one of the great unsolved challenges in computational biology.
Quantum entanglement allows particles to instantaneously affect each other regardless of the distance separating them.
The orchestra began with a soft pianissimo passage that gradually swelled into a magnificent fortissimo.
In medieval Europe, guilds controlled the production and sale of goods within city boundaries.
The algorithm sorted one million integers in under two seconds using a modified merge sort.
Rain fell steadily on the empty streets as the last bus departed from the terminal.
She had studied mathematics for twenty years, yet still found number theory endlessly surprising.
The constitutional amendment required approval from three-quarters of the state legislatures.
Astronomers detected an unusual radio signal emanating from a galaxy cluster four billion light-years away.
The chef carefully balanced the acidity of the lemon with the richness of the butter sauce.
Children learn language through a combination of imitation, correction, and innate grammatical intuition.
The archaeological excavation revealed pottery shards dating back to the Bronze Age settlement.
His argument rested on three premises, two of which were empirically questionable at best.
The neural network achieved ninety-two percent accuracy on the image classification benchmark.
Climate models suggest that Arctic ice sheets will diminish significantly by mid-century.
The treaty established a demilitarized zone along the river delta separating the two nations.
Philosophy has long grappled with the relationship between consciousness and physical brain processes.
The stock market reacted sharply to the unexpected announcement from the central bank governor.
Her novel explored the psychological consequences of long-term isolation in confined spaces.
The supply chain disruption affected manufacturers in seventeen countries across four continents.
Gravitational waves were first directly detected by the LIGO observatory in September 2015.
The enzyme catalyzed the reaction by lowering the activation energy required for the bond to break.
Migratory birds navigate using a combination of magnetic fields, star patterns, and landmarks.
The parliament debated the tax reform bill for six hours before reaching a procedural compromise.
Dark matter constitutes approximately twenty-seven percent of the total energy content of the universe.
The pianist practiced the same twelve-bar passage for three hours until every note felt natural.
Gene editing technologies have opened new possibilities for treating hereditary diseases at the molecular level.
The harbor town depended entirely on fishing and canning for its economic livelihood.
Bayesian inference updates prior beliefs in light of new evidence according to Bayes theorem.
The painting depicted a pastoral scene with shepherds and their flocks in an idealized landscape.
Machine learning models trained on biased data tend to reproduce and amplify those biases.
The surgeon performed the delicate procedure using robotic assistance and real-time imaging guidance.
In thermodynamics, entropy is a measure of the disorder or randomness of a system.
The diplomatic negotiations collapsed after one side introduced unexpected new demands.
She designed the bridge using computer-aided simulations to ensure it could withstand seismic forces.
The literary critic argued that postmodern fiction deliberately destabilizes narrative authority.
Antibiotic resistance has emerged as one of the most pressing public health crises of our era.
The telescope captured images of protoplanetary disks forming around newly born stars.
The judge instructed the jury to consider only the evidence presented during the trial.
Meteorologists warned of an approaching hurricane with sustained winds of one hundred forty miles per hour.
The immune system recognizes pathogens through a sophisticated system of molecular recognition.
Corporate governance reforms aimed to improve transparency and accountability in executive decision-making.
The excavation site showed evidence of continuous human habitation for over ten thousand years.
Sparse coding suggests that the brain represents information using a small number of active neurons.
The politician delivered a passionate speech about education reform to an enthusiastic crowd.
Water molecules form hydrogen bonds that give liquid water its unusually high boiling point.
The documentary filmmaker spent two years embedded with a nomadic tribe in central Mongolia.
Differential equations describe how physical quantities change continuously over time and space.
The restoration project aims to return the wetlands to their original ecological function.
She debugged the segmentation fault by tracing the pointer arithmetic through three nested functions.
The central bank raised interest rates to combat the highest inflation in four decades.
Plate tectonics explains the movement of continents and the formation of mountain ranges and ocean basins.
The award-winning novel blended magical realism with a gritty portrayal of urban poverty.
Signal processing algorithms filter out noise from measurements to extract the underlying signal.
The trade delegation arrived hoping to renegotiate the terms of the existing bilateral agreement.
Optical illusions demonstrate how the visual cortex actively constructs our perception of reality.
The marathon runner maintained a steady pace of five minutes per kilometer through the hilly course.
Formal verification methods prove the correctness of software by exhaustively checking all possible states.
The historian argued that contingency, not structural forces, determined the outcome of the war.
Cell division is regulated by a complex network of proteins that prevent uncontrolled growth.
The auction house estimated the painting's value at between four and six million dollars.
Tidal forces cause the gradual deceleration of Earth's rotation over geological timescales.
The compiler optimized the inner loop by vectorizing operations and eliminating redundant memory accesses.
Medieval Islamic scholars preserved and extended Greek mathematical knowledge during the European Dark Ages.
The central paradox of quantum mechanics is that measurement disturbs the system being measured.
Her research demonstrated that microplastics accumulate in marine food chains at alarming concentrations.
The conductor gestured for a diminuendo as the strings approached the lyrical second theme.
Statistical mechanics connects the microscopic properties of atoms to macroscopic thermodynamic quantities.
The prime minister called a snap election after losing a critical parliamentary confidence vote.
Crystallography uses the diffraction of X-rays to determine the atomic structure of materials.
The architect incorporated passive cooling techniques to reduce the building's energy consumption by forty percent.
Recursive algorithms solve problems by decomposing them into smaller instances of the same problem.
The biodiversity of tropical rainforests far exceeds that of any other terrestrial ecosystem.
She translated the cuneiform tablets, revealing an administrative record of barley distribution.
The peer review process requires independent experts to critically evaluate submitted manuscripts.
In Roman law, the principle of innocent until proven guilty was well established.
Photosynthesis converts carbon dioxide and water into glucose using energy from sunlight.
The geologist examined the rock strata, reading the history of ancient climate changes.
Probabilistic graphical models represent conditional independence relationships among random variables.
The novelist structured the book as a series of interconnected vignettes spanning three generations.
Superconductors conduct electricity with zero resistance below a critical temperature.
The peace treaty ended decades of conflict but left several territorial disputes unresolved.
Information theory quantifies the minimum number of bits needed to transmit a message without error.
The laboratory grew cultures of the bacteria to test the effectiveness of the new antibiotic.
Comparative anatomy reveals homologous structures across species that share common evolutionary ancestors.
The fiscal deficit widened as government spending on infrastructure and social programs increased.
The detective followed a chain of circumstantial evidence that ultimately led to the warehouse.
Renormalization group theory explains how physical laws transform under changes of scale.
The museum exhibited artifacts from the Silk Road trade routes connecting East and West.
Emotional regulation involves cognitive strategies that modify the intensity and duration of feelings.
The shipping container revolutionized global logistics by standardizing cargo transport across modes.
Fluorescent proteins have become indispensable tools for visualizing cellular processes in living organisms.
The referendum on independence produced a narrow majority in favor, triggering a constitutional crisis.
Gradient descent iteratively adjusts model parameters in the direction that minimizes the loss function.
The drought persisted for eighteen months, forcing farmers to abandon thousands of acres of cropland.
Laser interferometry can detect displacements smaller than a thousandth of a proton diameter.
The treaty of Westphalia in 1648 established the modern system of sovereign nation-states.
In graph theory, a Hamiltonian path visits every vertex in a graph exactly once.
The telescope resolved individual stars in the globular cluster at the edge of the Milky Way.
Viral vectors deliver therapeutic genetic material into target cells during gene therapy procedures.
The expedition crossed the polar ice cap using skis, dog sleds, and satellite navigation.
Computational fluid dynamics simulates the flow of air around aircraft wings to optimize aerodynamics.
The poet employed enjambment to create tension between syntactic and metrical boundaries.
Macroeconomic models attempt to capture the aggregate behavior of millions of individual actors.
The sonata alternated between the major and relative minor to create emotional contrast.
Neurotransmitters diffuse across the synaptic cleft to bind receptors on the postsynaptic neuron.
The civil engineer assessed the load-bearing capacity of the aging suspension bridge.
Set theory provides the foundation upon which modern mathematics is formally constructed.
The painting's impasto technique created a tactile surface that caught and reflected light differently from each angle.
Geothermal energy harnesses heat from within the Earth to generate electricity and provide heating.
The constitutional court struck down the legislation as incompatible with fundamental rights.
Protein synthesis occurs on ribosomes, where messenger RNA is translated into amino acid sequences.
The historian reconstructed the daily life of medieval peasants from manorial court records.
Abstract algebra studies algebraic structures such as groups, rings, fields, and modules.
The pharmaceutical company conducted phase three trials in twelve countries with four thousand participants.
Plate boundaries are zones of intense geological activity including earthquakes and volcanic eruptions.
The documentary examined how social media algorithms amplify polarization in political discourse.
Stellar nucleosynthesis produces chemical elements heavier than hydrogen and helium inside stars.
The museum conservation team used infrared reflectography to examine the underdrawing of the painting.
Random forests reduce overfitting by averaging predictions from many independently trained decision trees.
The maritime patrol discovered an unauthorized vessel operating within the exclusive economic zone.
Morphological analysis examines the internal structure of words and the rules governing their formation.
The carbon cycle describes the movement of carbon atoms through the atmosphere, biosphere, and geosphere.
The defendant's attorney filed a motion to suppress evidence obtained without a proper warrant.
Attentional mechanisms in neural networks allow models to focus on relevant parts of the input.
The glacier retreated several kilometers over the past century due to rising temperatures.
Epidemiological studies establish statistical associations between risk factors and disease outcomes.
Photonic integrated circuits manipulate light on a chip to perform computation and communication.
The senate committee examined the proposed regulatory framework for artificial intelligence systems.
Functional magnetic resonance imaging measures changes in blood oxygenation as a proxy for neural activity.
The harbor master coordinated the arrival and departure of over two hundred vessels per day.
Variational inference approximates complex posterior distributions using simpler parametric families.
The folklore of the region preserved oral traditions dating back centuries before written records.
Molecular dynamics simulations model the time evolution of atomic systems at femtosecond resolution.
The tax code's complexity creates opportunities for both legal avoidance and illegal evasion.
The spacecraft used a gravity assist maneuver around Jupiter to gain velocity toward Saturn.
Feminist literary criticism examines how gender shapes the production and reception of texts.
The bridge deck was prestressed using high-tensile steel tendons to counteract bending forces.
Epigenetic modifications alter gene expression without changing the underlying DNA sequence.
The international tribunal ruled on the maritime boundary dispute according to international law.
Active learning strategies select the most informative unlabeled examples for human annotation.
The novelist drew on her own experience as a refugee to give the protagonist psychological depth.
Topological insulators conduct electricity on their surfaces while remaining insulating in their bulk.
The orchestration called for an unusually large brass section to achieve the desired sonic power.
Selection bias occurs when the sample used in a study is not representative of the target population.
The volcano had remained dormant for over two centuries before erupting without warning.
Transfer learning adapts models pretrained on large datasets to specific downstream tasks.
The penal system came under criticism for its disproportionate impact on marginalized communities.
Membrane proteins regulate the transport of ions and molecules across the cell membrane.
The cartographer used satellite imagery and ground surveys to produce a detailed topographic map.
The legislative session addressed regulatory reforms to the financial services and energy sectors.
Hamiltonian mechanics reformulates classical mechanics using energy rather than force as the primary concept.
The social scientist examined how institutional trust shapes civic participation and collective action.
Ultrafast spectroscopy probes chemical reactions on timescales of femtoseconds and attoseconds.
The contractor completed the foundation work three weeks ahead of schedule due to favorable weather.
Contrastive learning trains representations by pushing similar examples together and dissimilar ones apart.
The medieval trade guilds enforced quality standards and controlled entry into skilled occupations.
The surgeon repaired the torn ligament using a minimally invasive arthroscopic technique.
Cosmic inflation explains the large-scale homogeneity and flatness of the observable universe.
The court interpreted the statute narrowly, limiting its application to the specific facts at hand.
Biochemical pathways convert nutrients into the energy and building blocks required for cellular function.
The urban planner proposed transforming the abandoned industrial site into mixed-use residential space.
Adversarial examples fool neural network classifiers with small perturbations invisible to human eyes.
The novelist's unreliable narrator forced readers to actively question the truthfulness of events.
Seismograph recordings revealed multiple distinct fault ruptures during the sequence of earthquakes.
The regulation established maximum permissible concentrations for twelve classes of industrial pollutants.
Gauge invariance is a fundamental symmetry principle underlying all modern theories of particle physics.
The trade agreement reduced tariffs on agricultural goods while maintaining protections for manufacturing.
Synaptic plasticity enables the brain to strengthen or weaken connections based on patterns of activity.
Spectroscopic observations of exoplanet atmospheres can reveal the presence of water vapor and methane.
The ethics committee reviewed protocols for studies involving human subjects and genetic data.
Optimization algorithms search for parameter configurations that minimize the objective function.
The opera premiered to mixed reviews, with critics divided over the unconventional staging choices.
Remote sensing satellites measure vegetation indices, surface temperatures, and ocean salinity globally.
The physicist derived the equations governing wave propagation in an inhomogeneous medium.
The diplomat navigated complex multilateral negotiations to achieve consensus on the climate agreement.
Multivariate statistics analyzes relationships among multiple variables simultaneously.
The forensic analyst extracted DNA from trace evidence to identify suspects in the cold case.
Network effects create winner-take-all dynamics in markets for digital platforms and social networks.
The restoration of the fresco required painstaking work to consolidate the fragile plaster surface.
Topological data analysis uses persistent homology to extract shape features from high-dimensional datasets.
The ecologist studied predator-prey dynamics in the Serengeti ecosystem over three decades.
The central bank's forward guidance influenced long-term interest rates and investment decisions.
Atmospheric chemistry models simulate the photochemical reactions that determine ozone concentrations.
The novelist wove multiple narrative threads together, revealing their connections only in the final chapter.
Electroweak unification demonstrates that electromagnetism and the weak nuclear force are manifestations of a single force.
The environmental impact assessment evaluated the effects on biodiversity, water quality, and air pollution.
"""


class _CausalSelfAttention(nn.Module):

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=-1)
        def split_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scale = self.head_dim ** -0.5
        att = (q @ k.transpose(-2, -1)) * scale
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class _MLP(nn.Module):

    def __init__(self, n_embd: int):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=True)
        self._captured: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.c_fc(x))
        self._captured = h.detach()
        return self.c_proj(h)


class _Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, eps: float = 1e-5):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=eps)
        self.attn = _CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd, eps=eps)
        self.mlp = _MLP(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MinimalGPT2(nn.Module):

    def __init__(self, vocab_size=50257, n_positions=1024,
                 n_embd=768, n_layer=12, n_head=12):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.h = nn.ModuleList([_Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, eps=1e-5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        return self.ln_f(x)

    @classmethod
    def from_pretrained(cls, model_name: str = "gpt2") -> "MinimalGPT2":
        from huggingface_hub import hf_hub_download

        print(f"  Downloading {model_name} config …")
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        with open(config_path) as f:
            cfg = json.load(f)

        n_embd = cfg["n_embd"]
        n_head = cfg["n_head"]
        n_layer = cfg["n_layer"]
        vocab_size = cfg["vocab_size"]
        n_positions = cfg["n_positions"]

        print(f"  Downloading {model_name} weights (~548 MB, cached after first run) …")
        try:
            weight_path = hf_hub_download(repo_id=model_name,
                                          filename="model.safetensors")
            from safetensors.torch import load_file as sf_load
            sd_hf = sf_load(weight_path)
        except Exception:
            weight_path = hf_hub_download(repo_id=model_name,
                                          filename="pytorch_model.bin")
            sd_hf = torch.load(weight_path, map_location="cpu")

        model = cls(vocab_size, n_positions, n_embd, n_layer, n_head)
        sd = model.state_dict()

        probe_key = "wte.weight"
        if probe_key not in sd_hf:
            probe_key = "transformer.wte.weight"
        prefix = "transformer." if probe_key.startswith("transformer.") else ""

        sd["wte.weight"] = sd_hf[f"{prefix}wte.weight"]
        sd["wpe.weight"] = sd_hf[f"{prefix}wpe.weight"]
        sd["ln_f.weight"] = sd_hf[f"{prefix}ln_f.weight"]
        sd["ln_f.bias"] = sd_hf[f"{prefix}ln_f.bias"]

        transposed_suffixes = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        copied_suffixes = [
            "ln_1.weight", "ln_1.bias",
            "ln_2.weight", "ln_2.bias",
            "attn.c_attn.bias", "attn.c_proj.bias",
            "mlp.c_fc.bias", "mlp.c_proj.bias",
        ]

        for i in range(n_layer):
            for suf in transposed_suffixes:
                hf_key = f"{prefix}h.{i}.{suf}"
                our_key = f"h.{i}.{suf}"
                sd[our_key] = sd_hf[hf_key].T.contiguous()
            for suf in copied_suffixes:
                hf_key = f"{prefix}h.{i}.{suf}"
                our_key = f"h.{i}.{suf}"
                sd[our_key] = sd_hf[hf_key]

        model.load_state_dict(sd)
        print(f"  GPT-2 ({n_layer}L / d={n_embd}) loaded successfully.")
        return model


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


def extract_mlp_activations(
    model: MinimalGPT2,
    tokenizer,
    corpus: str,
    layer_idx: int = 8,
    seq_len: int = 128,
    max_sequences: int = 50,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    model.eval().to(device)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokens = tokenizer.encode(corpus)
    print(f"  Corpus tokenised: {len(tokens)} tokens total.")

    mlp_hook = model.h[layer_idx].mlp
    activations = []

    n_seqs = min(max_sequences, len(tokens) // seq_len)
    with torch.no_grad():
        for i in range(n_seqs):
            chunk = tokens[i * seq_len: (i + 1) * seq_len]
            ids = torch.tensor([chunk], dtype=torch.long, device=device)
            model(ids)
            act = mlp_hook._captured
            activations.append(act.squeeze(0).cpu().numpy())

    H = np.concatenate(activations, axis=0)
    print(f"  Extracted activations: {H.shape} from layer {layer_idx}")
    return H


def pca_reduce(H: np.ndarray, d: int):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=d, random_state=42)
    H_pca = pca.fit_transform(H)
    print(f"  PCA {H.shape[1]}→{d}: cumulative var = "
          f"{pca.explained_variance_ratio_.sum():.3f}")
    return H_pca, pca.components_, pca.explained_variance_ratio_


def _jade_diag_vec(matrices: list, n_iter: int = 150, tol: float = 1e-8) -> np.ndarray:
    K = matrices[0].shape[0]
    C = np.stack(matrices, axis=0).copy()
    R = np.eye(K)

    for _ in range(n_iter):
        max_off = 0.0
        for p in range(K):
            for q in range(p + 1, K):
                Rp = R[:, p]
                Rq = R[:, q]

                CRp = C @ Rp
                CRq = C @ Rq

                h1 = CRp @ Rp - CRq @ Rq
                h2 = CRp @ Rq + CRq @ Rp

                g11 = float(h1 @ h1)
                g12 = float(h1 @ h2)
                g22 = float(h2 @ h2)

                ton  = g11 - g22
                toff = 2.0 * g12
                theta = 0.5 * np.arctan2(
                    toff, ton + np.sqrt(ton ** 2 + toff ** 2) + 1e-30
                )
                max_off = max(max_off, abs(theta))

                if abs(theta) < tol:
                    continue

                c, s = np.cos(theta), np.sin(theta)

                Rp_new =  c * R[:, p] + s * R[:, q]
                Rq_new = -s * R[:, p] + c * R[:, q]
                R[:, p] = Rp_new
                R[:, q] = Rq_new

        if max_off < tol:
            break

    return R


def cbfr_fast(H: np.ndarray, K: int) -> tuple:
    from src.cbfr import _whiten, _cumulant_slice

    Z, W_whiten = _whiten(H, K)

    matrices = []
    for i in range(K):
        for j in range(i, K):
            M = np.zeros((K, K))
            M[i, j] = 1.0
            if i != j:
                M[j, i] = 1.0
            matrices.append(_cumulant_slice(Z, M))

    R = _jade_diag_vec(matrices)

    W_hat = W_whiten @ R
    norms = np.linalg.norm(W_hat, axis=0, keepdims=True)
    W_hat /= np.maximum(norms, 1e-12)

    sources = R.T @ Z
    kurtoses = np.mean(sources ** 4, axis=1) / (np.mean(sources ** 2, axis=1) ** 2 + 1e-12) - 3.0

    return W_hat, kurtoses


class _CoherenceSAE(nn.Module):

    def __init__(self, d_in: int, n_features: int):
        super().__init__()
        self.W_dec = nn.Parameter(torch.randn(d_in, n_features) * 0.01)
        self.W_enc = nn.Parameter(torch.randn(n_features, d_in) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec.T.clone())

    def forward(self, x: torch.Tensor):
        x_c = x - self.b_dec
        f = torch.relu(x_c @ self.W_enc.T + self.b_enc)
        x_hat = f @ self.W_dec.T + self.b_dec
        return x_hat, f

    @torch.no_grad()
    def renorm_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)


def _coherence_penalty(W: torch.Tensor) -> torch.Tensor:
    W_n = W / (W.norm(dim=0, keepdim=True) + 1e-12)
    G = W_n.T @ W_n
    K = G.shape[0]
    mask = 1.0 - torch.eye(K, device=G.device)
    abs_G = G.abs() * mask
    beta = 20.0
    return torch.logsumexp(beta * abs_G.view(-1), dim=0) / beta


def train_sae(
    H: np.ndarray,
    K: int,
    lambda_l1: float = 0.005,
    lambda_coh: float = 0.0,
    n_epochs: int = 1500,
    lr: float = 3e-4,
    seed: int = 42,
    label: str = "SAE",
) -> np.ndarray:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, d = H.shape

    model = _CoherenceSAE(d, K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    H_t = torch.tensor(H, dtype=torch.float32, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(H_t),
        batch_size=min(512, n), shuffle=True, drop_last=False,
    )

    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        for (bx,) in loader:
            x_hat, f = model(bx)
            mse = ((bx - x_hat) ** 2).mean()
            l1 = lambda_l1 * f.abs().mean()
            coh = lambda_coh * _coherence_penalty(model.W_dec) if lambda_coh > 0 else 0.0
            loss = mse + l1 + coh
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.renorm_decoder()

        if (epoch + 1) % 500 == 0:
            print(f"  [{label}] epoch {epoch+1}/{n_epochs}  "
                  f"mse={mse.item():.4f}  l1={l1.item():.4f}  "
                  f"time={time.perf_counter()-t0:.1f}s")

    W_hat = model.W_dec.detach().cpu().numpy()
    norms = np.linalg.norm(W_hat, axis=0, keepdims=True)
    W_hat = W_hat / np.maximum(norms, 1e-12)
    return W_hat


def gram_matrix_abs(W: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    W_n = W / np.maximum(norms, 1e-12)
    return np.abs(W_n.T @ W_n)


def off_diagonal_coherences(W: np.ndarray) -> np.ndarray:
    G = gram_matrix_abs(W)
    K = G.shape[0]
    idx = np.triu_indices(K, k=1)
    return G[idx]


def activation_sparsity(H: np.ndarray, W: np.ndarray, threshold: float = 0.01) -> float:
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    W_n = W / np.maximum(norms, 1e-12)
    A = H @ W_n
    A = np.maximum(A, 0.0)
    return float((A > threshold).mean())


def run_exp8(config=None):
    if config is None:
        config = load_config()
    results_dir = ensure_results_dir()

    cfg = config.get("exp8_llm_activations", {})
    layer_idx   = cfg.get("layer_idx",   8)
    d_pca       = cfg.get("d_pca",       64)
    K           = cfg.get("K",           48)

    if K >= d_pca:
        raise ValueError(
            f"CBFR requires K < d_pca (got K={K}, d_pca={d_pca}). "
            f"Increase d_pca or decrease K in configs/default.yaml."
        )
    seq_len     = cfg.get("seq_len",     128)
    max_seqs    = cfg.get("max_seqs",    50)
    sae_epochs  = cfg.get("sae_epochs",  1500)
    lambda_l1   = cfg.get("lambda_l1",   0.005)
    lambda_coh  = cfg.get("lambda_coh",  0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Experiment 8: CBFR on Real LLM (GPT-2 small) Activations")
    print("=" * 60)
    print(f"  device={device}  layer={layer_idx}  d_pca={d_pca}  K={K}")

    print("\n[1/5] Loading GPT-2 small …")
    model = MinimalGPT2.from_pretrained("gpt2")
    tokenizer = _load_tokenizer()

    print(f"\n[2/5] Extracting layer-{layer_idx} post-GELU activations …")
    H_raw = extract_mlp_activations(
        model, tokenizer, MINI_CORPUS,
        layer_idx=layer_idx, seq_len=seq_len,
        max_sequences=max_seqs, device=device,
    )

    print(f"\n[3/5] PCA reduction to d={d_pca} + unit-variance normalisation …")
    H_pca, pca_comps, evr = pca_reduce(H_raw, d_pca)
    pca_component_std = H_pca.std(axis=0, keepdims=True)
    H_pca = H_pca / np.maximum(pca_component_std, 1e-12)
    n_samples = H_pca.shape[0]
    eig_ratio = (pca_component_std.max() / pca_component_std.min()) ** 2
    print(f"  {n_samples} token-position samples in d={d_pca} space (K={K}, η={K/d_pca:.2f})")
    print(f"  Pre-normalised: all components unit-variance  (raw eigenvalue ratio was {eig_ratio:.1f}×)")

    print(f"\n[4/5] Running CBFR (K={K}, vectorised JADE, unit-variance input) …")
    t0 = time.perf_counter()
    W_cbfr, kurtoses = cbfr_fast(H_pca.T, K)
    print(f"  CBFR done in {time.perf_counter()-t0:.1f}s  "
          f"μ(W)={mutual_coherence(W_cbfr):.4f}  "
          f"mean|κ|={np.abs(kurtoses).mean():.4f}")

    print(f"\n[5/5] Training SAE variants ({sae_epochs} epochs each) …")
    t1 = time.perf_counter()
    W_sae = train_sae(H_pca, K,
                      lambda_l1=lambda_l1, lambda_coh=0.0,
                      n_epochs=sae_epochs, label="L1-SAE")
    print(f"  L1-SAE done in {time.perf_counter()-t1:.1f}s  "
          f"μ(W)={mutual_coherence(W_sae):.4f}")

    t2 = time.perf_counter()
    W_coh = train_sae(H_pca, K,
                      lambda_l1=lambda_l1, lambda_coh=lambda_coh,
                      n_epochs=sae_epochs, label="Coh-SAE")
    print(f"  Coh-SAE done in {time.perf_counter()-t2:.1f}s  "
          f"μ(W)={mutual_coherence(W_coh):.4f}")

    print("\n── Summary ──────────────────────────────────────────────")
    for name, W in [("CBFR", W_cbfr), ("L1-SAE", W_sae), ("Coh-SAE", W_coh)]:
        ods = off_diagonal_coherences(W)
        sp  = activation_sparsity(H_pca, W)
        print(f"  {name:8s}: max_μ={mutual_coherence(W):.4f}  "
              f"mean_μ={ods.mean():.4f}  sparsity={sp:.4f}")

    data_path = os.path.join(results_dir, "exp8_results.npz")
    np.savez(
        data_path,
        W_cbfr=W_cbfr, W_sae=W_sae, W_coh=W_coh,
        H_pca=H_pca, kurtoses=kurtoses,
        d_pca=d_pca, K=K, layer_idx=layer_idx,
    )
    print(f"\nData saved: {data_path}")


    return {
        "W_cbfr": W_cbfr, "W_sae": W_sae, "W_coh": W_coh,
        "H_pca": H_pca, "kurtoses": kurtoses,
    }


if __name__ == "__main__":
    run_exp8()
