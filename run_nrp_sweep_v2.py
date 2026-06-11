"""RQ9 sweep v2: CP-PQA HV saturation in N_R on NRP, 10 seeds.

This sweep does NOT include MOQA-v0 because the previous tab_nrp_sat
mixed a 5-seed sweep MOQA run (200 reads, 16 weights) with the main
tab_rq2_hv 10-seed MOQA run (per-problem budget), producing inconsistent
HV values across the two tables. We now keep tab_nrp_sat scoped strictly
to CP-PQA saturation; the MOQA-v0 ranking lives in tab_rq2_hv.

Outputs results/nrp_round_sweep_v2.csv with columns
problem,config,seed,archive,HV,time_s.
"""
import sys, os, time, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.problems import load_problem
from src.baselines import run_cppqa
from src.quality import hypervolume, reference_point

PROBLEMS = ['nrp_large_v2', 'nrp_xlarge_v2', 'nrp_xxlarge_v2']
SEEDS = list(range(10))
ROUND_VALUES = [10, 20, 40, 80]
NUM_READS = 200

rows = [['problem','config','seed','archive','HV','time_s']]
for pname in PROBLEMS:
    problem = load_problem(pname, data_dir='benchmarks')
    decompose = pname in ('nrp_xlarge_v2', 'nrp_xxlarge_v2')
    bag = {}
    for nr in ROUND_VALUES:
        for seed in SEEDS:
            t0 = time.time()
            _, F = run_cppqa(problem, n_rounds=nr, num_reads=NUM_READS,
                              base_lambda=1.0, max_size=64, seed=seed,
                              decompose=decompose,
                              use_curator=True, curator_eps=0.02, curator_capacity=64)
            bag[(f'CP-PQA-R{nr}', seed)] = (np.asarray(F) if F else np.zeros((0,problem.n_obj)), time.time()-t0)
            print(f"{pname} CP-PQA-R{nr} s{seed} |A|={len(F)} t={time.time()-t0:.1f}s", flush=True)
    all_fs = [F for (F,_) in bag.values() if len(F)>0]
    ref = reference_point(all_fs)
    for (cfg, seed), (F, dt) in bag.items():
        hv = float(hypervolume(F, ref)) if len(F) else 0.0
        rows.append([pname, cfg, seed, len(F), f"{hv:.6e}", f"{dt:.2f}"])

out = os.path.join(os.path.dirname(__file__), 'results', 'nrp_round_sweep_v2.csv')
with open(out, 'w', newline='') as f:
    csv.writer(f).writerows(rows)
print('Saved', out)
