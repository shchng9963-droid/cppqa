"""RQ4 robustness to sampler budget (num_reads).
   We re-run CP-PQA on fsp_med_v2 with num_reads in {20, 80, 320}, 5 seeds each,
   to show that the gains come from the algorithm rather than from the sampler
   getting more compute.
"""
import sys, os, time, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.problems import load_problem
from src.baselines import run_cppqa
from src.quality import hypervolume

SEEDS = list(range(10))
BUDGETS = [20, 80, 160, 320]


def main():
    problem = load_problem('fsp_med_v2', data_dir='benchmarks')
    rows = [['budget', 'seed', 'archive', 'hv', 'time_s']]
    for B in BUDGETS:
        for seed in SEEDS:
            t0 = time.time()
            X, F = run_cppqa(problem, seed=seed, n_rounds=8,
                             num_reads=B, decompose=False)
            dt = time.time() - t0
            F = np.array(F) if F else np.zeros((0, problem.n_obj))
            ref = np.max(F, axis=0) + 1.0 if len(F) else None
            hv = float(hypervolume(F, ref)) if len(F) else 0.0
            rows.append([B, seed, len(F), f"{hv:.4f}", f"{dt:.2f}"])
            print(f"budget={B:4d} seed={seed} |A|={len(F):3d} HV={hv:.3f} t={dt:.1f}s",
                  flush=True)
    out = os.path.join(os.path.dirname(__file__), 'results', 'budget_sweep.csv')
    with open(out, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print(f"\nWritten {out}")


if __name__ == '__main__':
    main()
