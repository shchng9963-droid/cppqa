# CP-PQA: Constraint-Preserving Pareto Quantum-Ready Annealing

Reference implementation and reproduction scripts for **CP-PQA**, a QUBO-level,
quantum-ready annealing framework for constraint-dense multi-objective optimization,
evaluated on two Search-Based Software Engineering problem families — the Next
Release Problem (NRP) and multi-objective Feature Selection (FSP).

> **Backend note.** All experiments use **simulated annealing** (`dwave-neal`) as a
> reproducible QA-style proxy. No quantum hardware is used and no quantum-speedup
> claim is made; the framework is defined at the QUBO level.

The three pillars:
- **FPE** — Feasibility-Preserving Encoder with a closed-form group-local penalty `λ★` (`src/encoders.py`).
- **ATPS** — Adaptive Tchebycheff Pareto Search with a coverage-gap finder (`src/atps.py`).
- **Curator** — online ε-Pareto archive with crowding-distance eviction (`src/archive_curator.py`).

## Layout
```
src/                     core method
  encoders.py            FPE + Tchebycheff QUBO encoding
  atps.py                adaptive coverage-gap scheduler
  archive_curator.py     ε-Pareto curator
  decomposition.py       optional spectral (SHD) safety net
  baselines.py           CP-PQA + MOQA-v0, CQHA-MEI, NSGA-II, IBEA, SMS-EMOA, MOEA/D
  problems.py            NRP / FSP definitions and feasibility
  quality.py             HV, IGD+, spacing, reference point, union front
experiment_common.py     shared harness (budgets, metrics, aggregation)
benchmark_gen.py         feasibility-aware NRP/FSP instance generator
benchmarks/              the 10 controlled instances used in the paper (n in {20,50,100,200,500})
run_*.py                 per-RQ reproduction scripts (see table below)
analysis_significance.py RQ4 Mann-Whitney U + Vargha-Delaney A12 from raw runs
test_tier1.py            sanity tests
```

## Install
```bash
python -m venv .venv && source .venv/bin/activate      # or: conda create -n cp-pqa python=3.12
pip install -r requirements.txt
python test_tier1.py                                   # 5 tests, should print OK
```
Tested with Python 3.12, `dimod` 0.12, `dwave-neal` 0.6, `pymoo` 0.6.1, `networkx` 3.x, `numpy`, `scipy`.

## Quick start
```bash
python run_tier1_main.py --quick      # 2 seeds x 2 instances x 6 methods (smoke run)
```

## Full reproduction
Each script writes a CSV under `results/tier1/`; the paper's tables are formatted
from those CSVs. Run from the repository root.

| Script | Paper item | Output |
|---|---|---|
| `run_tier1_main.py` | RQ1 coverage, RQ2 HV, RQ3 IGD, RQ7 runtime | `method_summary_tier1.csv`, `raw_runs_tier1.csv` |
| `analysis_significance.py` | RQ4 significance (Mann-Whitney U, Â₁₂) | stdout (reads `raw_runs_tier1.csv`) |
| `run_rq5_ablation_tier1.py` | RQ5 pillar ablation | `rq5_ablation_summary_tier1.csv` |
| `run_curator_capacity_tier1.py` | RQ6 curation effect | `curator_capacity_summary_tier1.csv` |
| `run_budget_sweep.py` | RQ8 sampler-budget sweep (FSP-50) | `*budget*csv` |
| `run_nrp_sweep_v2.py` → `regen_tab_nrp_sat_v2.py` | RQ9 NRP saturation | `nrp_round_sweep_v2.csv` |
| `run_fpe_stress_tier1.py` | RQ10 F1 stress test (dense NRP) | `fpe_stress_tier1.csv` |
| `run_atps_practical_tier1.py` | ATPS vs random-weight practical comparison | `atps_practical_summary_tier1.csv` |

Default is 10 seeds over all 10 instances:
```bash
python run_tier1_main.py
python analysis_significance.py
python run_rq5_ablation_tier1.py
python run_curator_capacity_tier1.py
python run_budget_sweep.py
python run_nrp_sweep_v2.py && python regen_tab_nrp_sat_v2.py
python run_fpe_stress_tier1.py
python run_atps_practical_tier1.py
```
The full table reproduces in a few hours on a single multi-core workstation
(SMS-EMOA and MOEA/D dominate the wall-clock; CP-PQA/MOQA finish in seconds).

### Regenerating the benchmarks
The 10 instances in `benchmarks/` are the exact ones used in the paper. To
regenerate them from scratch with the feasibility-aware generator:
```bash
python benchmark_gen.py
```

## License
MIT — see `LICENSE`.

## Citation
```bibtex
@misc{cppqa,
  title  = {CP-PQA: A Constraint-Preserving Quantum-Ready Pareto Annealing
            Framework for Constraint-Dense Multi-Objective Optimization},
  author = {Anonymous},
  year   = {2026},
  note   = {Code: https://github.com/anonymous-on-submission/cp-pqa}
}
```
