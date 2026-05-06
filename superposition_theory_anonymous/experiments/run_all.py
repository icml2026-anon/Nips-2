
import os
import sys
import time
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import ensure_results_dir, load_config


def _run_one(args):
    name, module_name, func_name, config = args
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    start = time.time()
    try:
        module = __import__(module_name)
        func = getattr(module, func_name)
        func(config)
        elapsed = time.time() - start
        return name, True, elapsed, None
    except Exception as e:
        import traceback
        elapsed = time.time() - start
        tb = traceback.format_exc()
        return name, False, elapsed, tb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially instead of in parallel")
    args = parser.parse_args()

    ensure_results_dir()
    config = load_config(args.config)

    experiments = [
        ("Experiment 1: Capacity Phase Transition", "exp1_capacity", "run_exp1"),
        ("Experiment 2: CBFR Identifiability", "exp2_identifiability", "run_exp2"),
        ("Experiment 3: Interpretability-Efficiency Tradeoff", "exp3_tradeoff", "run_exp3"),
        ("Experiment 4: Generalization Bound", "exp4_generalization", "run_exp4"),
        ("Experiment 5: Sample Complexity Scaling", "exp5_sample_complexity", "run_exp5"),
    ]

    total_start = time.time()
    tasks = [(name, mod, func, config) for name, mod, func in experiments]

    if args.sequential:
        print("Running 5 experiments SEQUENTIALLY...\n")
        results = [_run_one(t) for t in tasks]
    else:
        print(f"Running 5 experiments in PARALLEL ({mp.cpu_count()} CPUs)...\n")
        with mp.Pool(processes=min(5, mp.cpu_count())) as pool:
            results = pool.map(_run_one, tasks)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, success, elapsed, tb in results:
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {name} -- {elapsed:.1f}s")
        if tb:
            print(f"         {tb.strip().splitlines()[-1]}")
    print(f"\n  Total wall time: {total_elapsed:.1f}s")
    print(f"  Results saved to: {os.path.join(os.path.dirname(__file__), '..', 'results')}")


if __name__ == "__main__":
    main()
