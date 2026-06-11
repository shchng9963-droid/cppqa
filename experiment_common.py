from __future__ import annotations

import csv
import os
import time
import traceback
from typing import Iterable

import numpy as np

from src.baselines import (
    run_cqha_mei,
    run_cppqa,
    run_ibea,
    run_moqa_v0,
    run_moead,
    run_nsga2,
    run_smsemoa,
)
from src.problems import load_problem
from src.quality import (
    feasibility_rate,
    hypervolume,
    inverted_generational_distance,
    reference_point,
    spacing,
    union_pareto,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'benchmarks')
RESULTS_DIR = os.path.join(ROOT, 'results')
TIER1_RESULTS_DIR = os.path.join(RESULTS_DIR, 'tier1')
os.makedirs(TIER1_RESULTS_DIR, exist_ok=True)

PROBLEMS = [
    'nrp_small_v2', 'nrp_med_v2', 'nrp_large_v2', 'nrp_xlarge_v2', 'nrp_xxlarge_v2',
    'fsp_small_v2', 'fsp_med_v2', 'fsp_large_v2', 'fsp_xlarge_v2', 'fsp_xxlarge_v2',
]
QUICK_PROBLEMS = ['nrp_small_v2', 'fsp_small_v2']
DEFAULT_METHODS = ['MOQA-v0', 'CQHA-MEI', 'CP-PQA', 'NSGA-II', 'IBEA', 'SMS-EMOA']
DEFAULT_SEEDS = list(range(10))


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def parse_problem_list(spec: str | None, default: list[str] | None = None) -> list[str]:
    if spec is None or not str(spec).strip():
        return list(default or PROBLEMS)
    return [p.strip() for p in str(spec).split(',') if p.strip()]


def budget_for(problem_name: str) -> dict:
    if 'xxlarge' in problem_name:
        decompose = problem_name.startswith('nrp')
        return dict(qa_weights=6, qa_reads=150, ea_pop=80, ea_gen=80, atps_rounds=8, decompose=decompose)
    if 'xlarge' in problem_name:
        decompose = problem_name.startswith('nrp')
        return dict(qa_weights=8, qa_reads=180, ea_pop=80, ea_gen=100, atps_rounds=10, decompose=decompose)
    if 'small' in problem_name:
        return dict(qa_weights=10, qa_reads=150, ea_pop=60, ea_gen=80, atps_rounds=10, decompose=False)
    if 'med' in problem_name:
        return dict(qa_weights=10, qa_reads=200, ea_pop=80, ea_gen=100, atps_rounds=12, decompose=False)
    decompose = problem_name.startswith('nrp')
    return dict(qa_weights=8, qa_reads=200, ea_pop=80, ea_gen=100, atps_rounds=10, decompose=decompose)


def run_method(method: str, problem, seed: int, budget: dict, **overrides):
    if method == 'MOQA-v0':
        return run_moqa_v0(problem, n_weights=budget['qa_weights'], num_reads=budget['qa_reads'], seed=seed)
    if method == 'CQHA-MEI':
        return run_cqha_mei(problem, n_weights=budget['qa_weights'], num_reads=budget['qa_reads'], max_size=64, seed=seed)
    if method == 'CP-PQA':
        kwargs = dict(
            n_rounds=budget['atps_rounds'],
            num_reads=budget['qa_reads'],
            base_lambda=1.0,
            max_size=64,
            seed=seed,
            decompose=budget['decompose'],
            use_curator=True,
            curator_eps=0.02,
            curator_capacity=64,
        )
        kwargs.update(overrides)
        return run_cppqa(problem, **kwargs)
    if method == 'CP-PQA-random':
        kwargs = dict(
            n_rounds=budget['atps_rounds'],
            num_reads=budget['qa_reads'],
            base_lambda=1.0,
            max_size=64,
            seed=seed,
            decompose=budget['decompose'],
            random_weights=True,
            use_curator=True,
            curator_eps=0.02,
            curator_capacity=64,
        )
        kwargs.update(overrides)
        return run_cppqa(problem, **kwargs)
    if method == 'CP-PQA-penalty':
        kwargs = dict(
            n_rounds=budget['atps_rounds'],
            num_reads=budget['qa_reads'],
            base_lambda=1.0,
            max_size=64,
            seed=seed,
            decompose=budget['decompose'],
            use_penalty=True,
            use_curator=True,
            curator_eps=0.02,
            curator_capacity=64,
        )
        kwargs.update(overrides)
        return run_cppqa(problem, **kwargs)
    if method == 'CP-PQA-strictND':
        kwargs = dict(
            n_rounds=budget['atps_rounds'],
            num_reads=budget['qa_reads'],
            base_lambda=1.0,
            max_size=64,
            seed=seed,
            decompose=budget['decompose'],
            use_curator=False,
        )
        kwargs.update(overrides)
        return run_cppqa(problem, **kwargs)
    if method == 'NSGA-II':
        return run_nsga2(problem, pop_size=budget['ea_pop'], n_gen=budget['ea_gen'], seed=seed)
    if method == 'IBEA':
        return run_ibea(problem, pop_size=budget['ea_pop'], n_gen=budget['ea_gen'], seed=seed)
    if method == 'SMS-EMOA':
        return run_smsemoa(problem, pop_size=budget['ea_pop'], n_gen=budget['ea_gen'], seed=seed)
    if method == 'MOEA/D':
        return run_moead(problem, pop_size=budget['ea_pop'], n_gen=budget['ea_gen'], seed=seed)
    raise ValueError(method)


def should_skip(problem_name: str, method: str) -> bool:
    if ('xlarge' in problem_name or 'xxlarge' in problem_name) and method in ('IBEA', 'MOEA/D'):
        return True
    if method == 'CQHA-MEI' and problem_name.startswith('nrp') and ('xlarge' in problem_name or 'xxlarge' in problem_name):
        return True
    if method == 'NSGA-II' and problem_name.startswith('fsp_') and ('xlarge' in problem_name or 'xxlarge' in problem_name):
        return True
    return False


def run_with_timing(method: str, problem_name: str, seed: int, **overrides):
    problem = load_problem(problem_name, data_dir=DATA_DIR)
    budget = budget_for(problem_name)
    t0 = time.time()
    xs, fs = run_method(method, problem, seed=seed, budget=budget, **overrides)
    runtime = time.time() - t0
    feas = feasibility_rate(xs, problem) if xs else 0.0
    return problem, xs, fs, runtime, feas


def metric_row(problem_name: str, method: str, seed: int, fs, runtime: float, feas: float, ref, pf) -> dict:
    if not fs:
        return dict(problem=problem_name, method=method, seed=seed, HV=0.0, IGD=float('inf'), SP=0.0, size=0, runtime=runtime, feasrate=feas)
    fs_a = np.asarray(fs, dtype=float)
    hv = hypervolume(fs_a, ref)
    igd = inverted_generational_distance(fs_a, pf) if len(pf) else float('inf')
    sp = spacing(fs_a)
    return dict(problem=problem_name, method=method, seed=seed, HV=hv, IGD=igd, SP=sp, size=len(fs), runtime=runtime, feasrate=feas)


def collect_problem_runs(problem_name: str, methods: Iterable[str], seeds: Iterable[int], method_overrides: dict | None = None, log_prefix: str = ''):
    method_overrides = method_overrides or {}
    problem = load_problem(problem_name, data_dir=DATA_DIR)
    archives = {m: {} for m in methods}
    runtimes = {m: {} for m in methods}
    feasrates = {m: {} for m in methods}
    print(f"{log_prefix}=== {problem_name} (n={problem.n_vars}, K={problem.n_obj}, dens={problem.constraint_density():.3f}) ===", flush=True)
    for method in methods:
        if should_skip(problem_name, method):
            print(f"{log_prefix}  {method:12s} SKIP (runtime guard)", flush=True)
            continue
        for seed in seeds:
            t0 = time.time()
            try:
                xs, fs = run_method(method, problem, seed=seed, budget=budget_for(problem_name), **method_overrides.get(method, {}))
                runtime = time.time() - t0
                feas = feasibility_rate(xs, problem) if xs else 0.0
            except Exception as exc:
                runtime = time.time() - t0
                xs, fs, feas = [], [], 0.0
                print(f"{log_prefix}  FAIL {method}/{problem_name}/s{seed}: {exc}", flush=True)
                traceback.print_exc()
            archives[method][seed] = fs
            runtimes[method][seed] = runtime
            feasrates[method][seed] = feas
            print(f"{log_prefix}  {method:12s} s{seed} |A|={len(fs):3d} feas={feas:.2f} t={runtime:6.2f}s", flush=True)
    return problem, archives, runtimes, feasrates


def rows_for_problem(problem_name: str, methods: Iterable[str], seeds: Iterable[int], archives: dict, runtimes: dict, feasrates: dict):
    all_fs = []
    for method in methods:
        for seed in seeds:
            fs = archives.get(method, {}).get(seed, [])
            if fs:
                all_fs.append(np.asarray(fs, dtype=float))
    if not all_fs:
        return []
    ref = reference_point(all_fs)
    pf = union_pareto(all_fs)
    rows = []
    for method in methods:
        for seed in seeds:
            if seed not in archives.get(method, {}):
                continue
            rows.append(metric_row(problem_name, method, seed, archives[method][seed], runtimes[method][seed], feasrates[method][seed], ref, pf))
    return rows


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_parent(path)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict], group_key: str = 'method') -> list[dict]:
    buckets = {}
    for row in rows:
        key = (row['problem'], row[group_key])
        buckets.setdefault(key, []).append(row)
    summary = []
    for (problem_name, label), items in buckets.items():
        def finite_vals(name):
            vals = [float(x[name]) for x in items]
            if name == 'IGD':
                vals = [v for v in vals if np.isfinite(v)] or [float('inf')]
            return np.asarray(vals, dtype=float)
        def ms(name):
            vals = finite_vals(name)
            return float(np.nanmean(vals)), float(np.nanstd(vals))
        hv_m, hv_s = ms('HV')
        igd_m, igd_s = ms('IGD')
        sp_m, sp_s = ms('SP')
        sz_m, sz_s = ms('size')
        rt_m, rt_s = ms('runtime')
        fr_m, fr_s = ms('feasrate')
        summary.append(dict(
            problem=problem_name,
            label=label,
            HV_mean=hv_m, HV_std=hv_s,
            IGD_mean=igd_m, IGD_std=igd_s,
            SP_mean=sp_m, SP_std=sp_s,
            size_mean=sz_m, size_std=sz_s,
            runtime_mean=rt_m, runtime_std=rt_s,
            feas_mean=fr_m, feas_std=fr_s,
        ))
    return summary
