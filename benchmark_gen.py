"""
Feasibility-aware benchmark generator for NRP and FSP.

The legacy generator (article1_experiments/nrp_fsp_data.py) produces FSP
instances of size >= 50 whose constraint set has *zero* satisfying
assignments (mandatory + require + exclude + alt-group propagation collapse
to a fixed assignment that violates several alt-groups). We replace it
with a constructive generator: first sample a "ground-truth" feasible
assignment x* and then sample constraints that x* satisfies. This gives
non-trivial benchmarks while guaranteeing |Feas| >= 1.

Output schema is *backward compatible* with the legacy datasets so that
NRPMOO / FSPMOO can read them unchanged.
"""
from __future__ import annotations
import json, os, random
import numpy as np


def gen_nrp(n_req, n_stake, density=0.20, seed=0):
    rng = np.random.default_rng(seed)
    revenue = rng.integers(1, 10, size=(n_stake, n_req))
    cost    = rng.integers(1, 20, size=n_req)
    # Random DAG of prereqs: edges only from i to j if j > i (acyclic).
    prereq = np.zeros((n_req, n_req), dtype=int)
    for i in range(n_req):
        for j in range(i + 1, n_req):
            if rng.random() < density:
                prereq[i, j] = 1
    return dict(n_req=int(n_req), n_stake=int(n_stake),
                revenue=revenue.tolist(), cost=cost.tolist(),
                prereq=prereq.tolist())


def gen_fsp_feasible(n_feat, density=0.15, seed=0,
                     mandatory_frac=0.05,
                     n_alt_groups=None,
                     alt_group_size=3):
    rng = np.random.default_rng(seed)
    richness    = rng.random(n_feat)
    reliability = rng.random(n_feat)
    defects     = rng.random(n_feat)
    cost        = rng.random(n_feat)

    # 1. choose alt-groups first (each group contributes exactly one selected feature)
    if n_alt_groups is None:
        n_alt_groups = max(1, n_feat // 8)
    pool = list(range(n_feat))
    rng.shuffle(pool)
    alt_groups = []
    used_in_group = set()
    for g in range(n_alt_groups):
        if len(pool) < alt_group_size:
            break
        grp = [pool.pop() for _ in range(alt_group_size)]
        alt_groups.append(grp)
        used_in_group.update(grp)

    # 2. mandatory: from features NOT in any alt-group
    free = [k for k in range(n_feat) if k not in used_in_group]
    n_mand = max(1, int(np.ceil(mandatory_frac * n_feat)))
    n_mand = min(n_mand, len(free))
    mandatory = sorted(rng.choice(free, size=n_mand, replace=False).tolist())

    # 3. construct ground-truth feasible solution x*
    x = np.zeros(n_feat, dtype=int)
    for k in mandatory:
        x[k] = 1
    # one chosen per alt-group
    for grp in alt_groups:
        chosen = int(rng.choice(grp))
        x[chosen] = 1
    # randomly turn on a few extras from free, with prob 0.3
    free_extras = [k for k in free if k not in mandatory]
    for k in free_extras:
        if rng.random() < 0.3:
            x[k] = 1

    # 4. sample require constraints (i, j) such that x[i] <= x[j]
    require = []
    for _ in range(int(density * n_feat * n_feat)):
        i = int(rng.integers(0, n_feat))
        j = int(rng.integers(0, n_feat))
        if i == j: continue
        if x[i] == 1 and x[j] == 0:
            continue   # would be infeasible
        # avoid spurious self-loop with mandatory
        require.append((i, j))
    require = list(set(require))[:int(n_feat * n_feat * density / 4)]

    # 5. sample exclude constraints (i, j) with NOT (x[i]==1 and x[j]==1)
    exclude = []
    pairs = [(i, j) for i in range(n_feat) for j in range(i+1, n_feat)
             if not (x[i] == 1 and x[j] == 1)]
    rng.shuffle(pairs)
    n_exc = int(density * n_feat / 2)
    exclude = pairs[:n_exc]

    # Final feasibility sanity check
    return dict(n_feat=int(n_feat),
                richness=richness.tolist(),
                reliability=reliability.tolist(),
                defects=defects.tolist(),
                cost=cost.tolist(),
                mandatory=mandatory,
                require=[list(p) for p in require],
                exclude=[list(p) for p in exclude],
                alt_groups=[list(g) for g in alt_groups]), x.tolist()


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmarks')
    os.makedirs(out, exist_ok=True)

    nrp_specs = [
        ('nrp_small_v2',   20,   5,  0.15),
        ('nrp_med_v2',     50,  10,  0.15),
        ('nrp_large_v2',  100,  20,  0.15),
        ('nrp_xlarge_v2', 200,  40,  0.10),
        ('nrp_xxlarge_v2',500, 100,  0.05),
    ]
    fsp_specs = [
        # (name, n, density, mandatory_frac, alt_group_size)
        ('fsp_small_v2',   20, 0.10, 0.05, 3),
        ('fsp_med_v2',     50, 0.10, 0.05, 3),
        ('fsp_large_v2',  100, 0.10, 0.05, 3),
        # SPLOT-calibrated: ~20% mandatory, smaller alt-groups, density 0.05
        ('fsp_xlarge_v2', 200, 0.05, 0.20, 3),
        ('fsp_xxlarge_v2',500, 0.03, 0.20, 3),
    ]

    for name, n, m, dens in nrp_specs:
        d = gen_nrp(n, m, density=dens, seed=hash(name) & 0xffff)
        with open(os.path.join(out, f'{name}.json'), 'w') as f:
            json.dump(d, f)
        print(f'  {name}: n={n}, prereq edges={int(np.sum(np.array(d["prereq"])))}')

    for name, n, dens, mfrac, algrp in fsp_specs:
        d, x_star = gen_fsp_feasible(n, density=dens, seed=hash(name) & 0xffff,
                                     mandatory_frac=mfrac, alt_group_size=algrp)
        # save ground-truth x* alongside, useful for sanity-checking
        d['_x_star'] = x_star
        with open(os.path.join(out, f'{name}.json'), 'w') as f:
            json.dump(d, f)
        print(f'  {name}: n={n}, |mand|={len(d["mandatory"])}, '
              f'|req|={len(d["require"])}, |exc|={len(d["exclude"])}, '
              f'|alt|={len(d["alt_groups"])}, |x*|={sum(x_star)}')


if __name__ == '__main__':
    main()
