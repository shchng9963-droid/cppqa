from __future__ import annotations

import argparse
import os

from experiment_common import (
    PROBLEMS,
    QUICK_PROBLEMS,
    TIER1_RESULTS_DIR,
    collect_problem_runs,
    parse_problem_list,
    rows_for_problem,
    summarize_rows,
    write_csv,
)

VARIANTS = {
    'strictND': {'CP-PQA': {'use_curator': False}},
    'eps001_cap32': {'CP-PQA': {'use_curator': True, 'curator_eps': 0.01, 'curator_capacity': 32}},
    'eps002_cap64': {'CP-PQA': {'use_curator': True, 'curator_eps': 0.02, 'curator_capacity': 64}},
    'eps002_cap128': {'CP-PQA': {'use_curator': True, 'curator_eps': 0.02, 'curator_capacity': 128}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--problems', type=str, default='')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', type=str, default=os.path.join(TIER1_RESULTS_DIR, 'curator_capacity_tier1.csv'))
    ap.add_argument('--summary-out', type=str, default=os.path.join(TIER1_RESULTS_DIR, 'curator_capacity_summary_tier1.csv'))
    args = ap.parse_args()

    problems = parse_problem_list(args.problems, QUICK_PROBLEMS if args.quick else PROBLEMS)
    seeds = list(range(2 if args.quick else args.seeds))
    rows = []

    for pname in problems:
        variant_rows = []
        for variant_name, overrides in VARIANTS.items():
            methods = ['CP-PQA']
            _problem, archives, runtimes, feasrates = collect_problem_runs(
                pname, methods, seeds, method_overrides=overrides, log_prefix=f'[tier1-curator:{variant_name}] '
            )
            per_rows = rows_for_problem(pname, methods, seeds, archives, runtimes, feasrates)
            for row in per_rows:
                row['variant'] = variant_name
            variant_rows.extend(per_rows)
        rows.extend(variant_rows)

    fieldnames = ['problem', 'variant', 'method', 'seed', 'HV', 'IGD', 'SP', 'size', 'runtime', 'feasrate']
    write_csv(args.out, rows, fieldnames)
    summary = summarize_rows(rows, group_key='variant')
    write_csv(args.summary_out, summary, list(summary[0].keys()) if summary else ['problem', 'label'])
    print(f'Saved {args.out}', flush=True)
    print(f'Saved {args.summary_out}', flush=True)


if __name__ == '__main__':
    main()
