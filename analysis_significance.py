"""RQ4 statistics: pairwise CP-PQA vs baseline significance on HV.

Reads results/tier1/raw_runs_tier1.csv (produced by run_tier1_main.py) and
prints, per instance and baseline, the two-sided Mann-Whitney U p-value and the
Vargha-Delaney A12 effect size (CP-PQA over the baseline) on per-seed HV.

A12 > 0.5 favours CP-PQA; the large-effect threshold is A12 >= 0.71.
"""
from __future__ import annotations
import csv
import os
from collections import defaultdict

from scipy.stats import mannwhitneyu

RAW = os.path.join(os.path.dirname(__file__), 'results', 'tier1', 'raw_runs_tier1.csv')
REF = 'CP-PQA'
BASELINES = ['MOQA-v0', 'CQHA-MEI', 'NSGA-II', 'IBEA', 'SMS-EMOA']


def vargha_delaney_a12(a, b):
    """A12 = P(X>Y) + 0.5 P(X=Y), X ~ a (CP-PQA), Y ~ b (baseline)."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return float('nan')
    greater = sum(x > y for x in a for y in b)
    equal = sum(x == y for x in a for y in b)
    return (greater + 0.5 * equal) / (m * n)


def main():
    hv = defaultdict(lambda: defaultdict(list))  # hv[problem][method] = [per-seed HV]
    with open(RAW) as f:
        for row in csv.DictReader(f):
            try:
                hv[row['problem']][row['method']].append(float(row['HV']))
            except (KeyError, ValueError):
                continue

    print(f"{'instance':16s} {'baseline':10s} {'A12':>6s} {'p':>8s}")
    for problem in sorted(hv):
        ours = hv[problem].get(REF, [])
        for base in BASELINES:
            other = hv[problem].get(base, [])
            if len(ours) < 2 or len(other) < 2:
                print(f"{problem:16s} {base:10s} {'--':>6s} {'--':>8s}")
                continue
            a12 = vargha_delaney_a12(ours, other)
            try:
                _, p = mannwhitneyu(ours, other, alternative='two-sided')
            except ValueError:
                p = float('nan')
            print(f"{problem:16s} {base:10s} {a12:6.2f} {p:8.3f}")


if __name__ == '__main__':
    main()
