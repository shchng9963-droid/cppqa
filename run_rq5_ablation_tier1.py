from __future__ import annotations

import argparse
import os

import numpy as np

from experiment_common import (
    PROBLEMS,
    QUICK_PROBLEMS,
    TIER1_RESULTS_DIR,
    budget_for,
    collect_problem_runs,
    metric_row,
    parse_problem_list,
    reference_point,
    summarize_rows,
    union_pareto,
    write_csv,
)

VARIANTS = ['full', 'no-FPE', 'no-ATPS', 'no-Curator', 'no-SHD']
DISPLAY_NAMES = {
    'nrp_small_v2': 'NRP-20',
    'nrp_med_v2': 'NRP-50',
    'nrp_large_v2': 'NRP-100',
    'nrp_xlarge_v2': 'NRP-200',
    'nrp_xxlarge_v2': 'NRP-500',
    'fsp_small_v2': 'FSP-20',
    'fsp_med_v2': 'FSP-50',
    'fsp_large_v2': 'FSP-100',
    'fsp_xlarge_v2': 'FSP-200',
    'fsp_xxlarge_v2': 'FSP-500',
}


def variant_overrides(name: str, budget: dict) -> dict:
    base = {
        'n_rounds': budget['atps_rounds'],
        'num_reads': budget['qa_reads'],
        'base_lambda': 1.0,
        'max_size': 64,
        'decompose': budget['decompose'],
        'use_penalty': False,
        'random_weights': False,
        'use_curator': True,
        'curator_eps': 0.02,
        'curator_capacity': 64,
    }
    if name == 'no-FPE':
        base['use_penalty'] = True
    elif name == 'no-ATPS':
        base['random_weights'] = True
    elif name == 'no-Curator':
        base['use_curator'] = False
    elif name == 'no-SHD':
        base['decompose'] = False
    return base


def rows_for_problem_variants(problem_name: str, variants: list[str], seeds: list[int], archives: dict, runtimes: dict, feasrates: dict) -> list[dict]:
    all_fs = []
    for variant in variants:
        for seed in seeds:
            fs = archives.get(variant, {}).get(seed, [])
            if fs:
                all_fs.append(np.asarray(fs, dtype=float))
    if not all_fs:
        return []

    ref = reference_point(all_fs)
    pf = union_pareto(all_fs)
    rows = []
    for variant in variants:
        for seed in seeds:
            if seed not in archives.get(variant, {}):
                continue
            row = metric_row(
                problem_name,
                'CP-PQA',
                seed,
                archives[variant][seed],
                runtimes[variant][seed],
                feasrates[variant][seed],
                ref,
                pf,
            )
            row['label'] = variant
            rows.append(row)
    return rows


def render_table(summary_rows: list[dict]) -> str:
    by_problem = {}
    for row in summary_rows:
        by_problem.setdefault(row['problem'], {})[row['label']] = row

    lines = [
        r'\begin{table*}[t]\centering\footnotesize',
        r'\caption{RQ5 pillar ablation across the ten benchmarks.}',
        r'\label{tab:rq5-ablation}',
        r'\begin{tabular}{l|cc|cc|cc|cc|cc|}',
        r'\toprule',
        r' & \multicolumn{2}{c|}{full} & \multicolumn{2}{c|}{no-FPE} & \multicolumn{2}{c|}{no-ATPS} & \multicolumn{2}{c|}{no-Curator} & \multicolumn{2}{c|}{no-SHD} \\',
        r'Instance & HV & $|A|$ & HV & $|A|$ & HV & $|A|$ & HV & $|A|$ & HV & $|A|$ \\',
        r'\midrule',
    ]
    for problem in PROBLEMS:
        row = [DISPLAY_NAMES[problem]]
        for variant in VARIANTS:
            stats = by_problem.get(problem, {}).get(variant)
            if stats is None:
                row.extend(['--', '--'])
                continue
            row.append(f"{float(stats['HV_mean']):.0f}")
            row.append(f"{float(stats['size_mean']):.1f}")
        lines.append(' & '.join(row) + r' \\')
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\par\medskip',
        r'\footnotesize\emph{Notes.} Each cell reports the mean over 10 seeds. \texttt{full} is CP-PQA with FPE, ATPS, and Curator under the same tier1 budget protocol used elsewhere in the paper. \texttt{no-SHD} disables the decomposition safety net while keeping the same remaining settings. HV uses a tier1-internal reference over the five variants and is not directly comparable with \cref{tab:rq2-hv}.',
        r'\end{table*}',
    ])
    return '\n'.join(lines) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--problems', type=str, default='')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', type=str, default=os.path.join(TIER1_RESULTS_DIR, 'rq5_ablation_tier1.csv'))
    ap.add_argument('--summary-out', type=str, default=os.path.join(TIER1_RESULTS_DIR, 'rq5_ablation_summary_tier1.csv'))
    ap.add_argument('--table-out', type=str, default='')
    args = ap.parse_args()

    problems = parse_problem_list(args.problems, QUICK_PROBLEMS if args.quick else PROBLEMS)
    seeds = list(range(2 if args.quick else args.seeds))
    rows = []

    for pname in problems:
        problem_archives = {}
        problem_runtimes = {}
        problem_feasrates = {}
        for variant_name in VARIANTS:
            overrides = {'CP-PQA': variant_overrides(variant_name, budget_for(pname))}
            _problem, archives, runtimes, feasrates = collect_problem_runs(
                pname,
                ['CP-PQA'],
                seeds,
                method_overrides=overrides,
                log_prefix=f'[tier1-rq5:{variant_name}] ',
            )
            problem_archives[variant_name] = archives['CP-PQA']
            problem_runtimes[variant_name] = runtimes['CP-PQA']
            problem_feasrates[variant_name] = feasrates['CP-PQA']
        rows.extend(rows_for_problem_variants(pname, VARIANTS, seeds, problem_archives, problem_runtimes, problem_feasrates))

    fieldnames = ['problem', 'label', 'method', 'seed', 'HV', 'IGD', 'SP', 'size', 'runtime', 'feasrate']
    write_csv(args.out, rows, fieldnames)
    summary = summarize_rows(rows, group_key='label')
    write_csv(args.summary_out, summary, list(summary[0].keys()) if summary else ['problem', 'label'])
    print(f'Saved {args.out}', flush=True)
    print(f'Saved {args.summary_out}', flush=True)

    if args.table_out:
        os.makedirs(os.path.dirname(args.table_out), exist_ok=True)
        with open(args.table_out, 'w') as f:
            f.write(render_table(summary))
        print(f'Saved {args.table_out}', flush=True)


if __name__ == '__main__':
    main()
