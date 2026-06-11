from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from neal import SimulatedAnnealingSampler

from experiment_common import DATA_DIR, PROBLEMS, QUICK_PROBLEMS, TIER1_RESULTS_DIR, parse_problem_list
from src.encoders import FeasibilityEncoder, PenaltyEncoder, decode
from src.problems import load_problem

ENCODERS = {
    'penalty_lam1': lambda: PenaltyEncoder(lam=1.0),
    'penalty_lam10': lambda: PenaltyEncoder(lam=10.0),
    'penalty_lam100': lambda: PenaltyEncoder(lam=100.0),
    'fpe': lambda: FeasibilityEncoder(base_lambda=1.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--weights', type=int, default=12)
    ap.add_argument('--reads', type=int, default=50)
    ap.add_argument('--problems', type=str, default='')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', type=str, default=os.path.join(TIER1_RESULTS_DIR, 'fpe_stress_tier1.csv'))
    args = ap.parse_args()

    sampler = SimulatedAnnealingSampler()
    problems = parse_problem_list(args.problems, QUICK_PROBLEMS if args.quick else PROBLEMS)
    seeds = list(range(2 if args.quick else args.seeds))
    rows = []

    for pname in problems:
        problem = load_problem(pname, data_dir=DATA_DIR)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for label, factory in ENCODERS.items():
                total = 0
                feasible = 0
                best_ok = 0
                n_free_total = 0
                for wid in range(args.weights):
                    weights = rng.dirichlet(np.ones(problem.n_obj))
                    enc = factory()
                    bqm, _vmap, fixed = enc.encode(problem, weights)
                    n_free_total += len(bqm.variables)
                    if len(bqm.variables) == 0:
                        x = decode({}, problem.n_vars, fixed)
                        _, viol = problem.evaluate(x)
                        total += 1
                        feasible += int(viol == 0)
                        best_ok += int(viol == 0)
                        continue
                    ss = sampler.sample(bqm, num_reads=args.reads, seed=seed + 1000 * wid)
                    samples = list(ss.samples())
                    energies = list(ss.record.energy)
                    best_feasible_here = 0
                    for s in samples:
                        x = decode(dict(s), problem.n_vars, fixed)
                        _, viol = problem.evaluate(x)
                        total += 1
                        if viol == 0:
                            feasible += 1
                    if samples:
                        best = dict(samples[int(np.argmin(energies))])
                        x_best = decode(best, problem.n_vars, fixed)
                        _, viol_best = problem.evaluate(x_best)
                        best_feasible_here = int(viol_best == 0)
                    best_ok += best_feasible_here
                row = {
                    'problem': pname,
                    'encoder': label,
                    'seed': seed,
                    'feasible_rate': feasible / max(total, 1),
                    'best_sample_feasible_rate': best_ok / max(args.weights, 1),
                    'n_weights': args.weights,
                    'reads_per_weight': args.reads,
                    'n_samples': total,
                    'mean_free_vars': n_free_total / max(args.weights, 1),
                }
                rows.append(row)
                print(
                    f"[tier1-fpe] {pname:14s} {label:14s} s{seed} "
                    f"feas={row['feasible_rate']:.3f} best={row['best_sample_feasible_rate']:.3f} "
                    f"free={row['mean_free_vars']:.1f}",
                    flush=True,
                )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['problem', 'encoder', 'seed'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {args.out}', flush=True)


if __name__ == '__main__':
    main()
