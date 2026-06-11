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

ATPS_METHODS = ['CP-PQA', 'CP-PQA-random', 'MOQA-v0', 'CQHA-MEI']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--problems', type=str, default='')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', type=str, default=os.path.join(TIER1_RESULTS_DIR, 'atps_practical_tier1.csv'))
    ap.add_argument('--summary-out', type=str, default=os.path.join(TIER1_RESULTS_DIR, 'atps_practical_summary_tier1.csv'))
    args = ap.parse_args()

    problems = parse_problem_list(args.problems, QUICK_PROBLEMS if args.quick else PROBLEMS)
    seeds = list(range(2 if args.quick else args.seeds))

    rows = []
    for pname in problems:
        _problem, archives, runtimes, feasrates = collect_problem_runs(pname, ATPS_METHODS, seeds, log_prefix='[tier1-atps] ')
        rows.extend(rows_for_problem(pname, ATPS_METHODS, seeds, archives, runtimes, feasrates))

    fieldnames = ['problem', 'method', 'seed', 'HV', 'IGD', 'SP', 'size', 'runtime', 'feasrate']
    write_csv(args.out, rows, fieldnames)
    summary = summarize_rows(rows, group_key='method')
    write_csv(args.summary_out, summary, list(summary[0].keys()) if summary else ['problem', 'label'])
    print(f'Saved {args.out}', flush=True)
    print(f'Saved {args.summary_out}', flush=True)


if __name__ == '__main__':
    main()
