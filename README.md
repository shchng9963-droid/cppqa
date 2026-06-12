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
