# Superposition Theory Experiments

Numerical experiments for validating theoretical results on neural feature superposition.

## Requirements

```bash
pip install -r requirements.txt
```

## Experiments

| Experiment | Script |
|---|---|
| Capacity Phase Transition | `experiments/exp1_capacity.py` |
| CBFR Identifiability | `experiments/exp2_identifiability.py` |
| Interpretability-Efficiency Tradeoff | `experiments/exp3_tradeoff.py` |
| Generalization Bound | `experiments/exp4_generalization.py` |
| Sample Complexity Scaling | `experiments/exp5_sample_complexity.py` |
| Toy Trained Model | `experiments/exp6_toy_model.py` |
| Method Comparison | `experiments/exp7_method_comparison.py` |
| LLM Activations | `experiments/exp8_llm_activations.py` |

## Run All Experiments

```bash
python experiments/run_all.py
```

Or run individually:

```bash
python experiments/exp1_capacity.py
python experiments/exp2_identifiability.py
python experiments/exp3_tradeoff.py
python experiments/exp4_generalization.py
python experiments/exp5_sample_complexity.py
python experiments/exp6_toy_model.py
python experiments/exp7_method_comparison.py
python experiments/exp8_llm_activations.py
```

Results are saved to `results/`.

## Project Structure

```
src/
  slr_model.py
  dictionary.py
  interpretability.py
  cbfr.py
  generalization.py
  utils.py
experiments/
  exp1_capacity.py
  exp2_identifiability.py
  exp3_tradeoff.py
  exp4_generalization.py
  exp5_sample_complexity.py
  exp6_toy_model.py
  exp7_method_comparison.py
  exp8_llm_activations.py
  run_all.py
configs/
  default.yaml
```
