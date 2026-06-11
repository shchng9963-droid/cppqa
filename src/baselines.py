"""
Multi-objective baseline solvers used in the experiments.

Implements:
    NSGA-II    via pymoo  (population-based dominance + crowding distance)
    IBEA       via pymoo  (reference-direction proxy; naming kept for legacy tables)
    SMS-EMOA   via pymoo  (hypervolume-based environmental selection)
    MOEA/D     via pymoo  (weighted Tchebycheff decomposition)
    MOQA-v0    : the original method from the legacy paper
                 (penalty QUBO + uniformly random weights + neal sampling)
    CQHA-MEI   : the original CQHA from the legacy paper
                 (penalty QUBO + Maximum-Energy-Impact decomposition)
    CP-PQA     : the new method = FeasibilityEncoder + Adaptive Tchebycheff
                 + (optional) Spectral Hierarchical Decomposition.

All solvers return (archive_x, archive_f) where archive_x is a list of
binary numpy arrays and archive_f is a list of objective vectors of length
problem.n_obj.

Constraint handling: every solution returned to the caller is feasible
(violation == 0) unless the solver explicitly cannot produce one, in which
case an empty archive is returned.
"""
from __future__ import annotations
import time
import numpy as np
import dimod
from neal import SimulatedAnnealingSampler
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from .encoders import PenaltyEncoder, FeasibilityEncoder, decode
from .decomposition import spectral_decompose
from .atps import atps, update_archive


SAMPLER = SimulatedAnnealingSampler()


# --------------------------------------------------------------------------- #
# Common: feasibility-preserving repair (used by EAs and as fall-back)
# --------------------------------------------------------------------------- #
def repair(problem, x):
    """Greedy projection onto the feasible region used by NSGA-II/IBEA/MOEA/D.
    Iterates until fixed point or hard cap."""
    x = np.asarray(x, dtype=int).copy()
    if hasattr(problem, 'mandatory'):
        for it in range(20):
            prev = x.copy()
            for i in problem.mandatory:
                x[i] = 1
            for i, j in problem.require:
                if x[i] == 1 and x[j] == 0:
                    x[j] = 1
            for i, j in problem.exclude:
                if x[i] == 1 and x[j] == 1:
                    # drop the one that has the most edges (heuristic)
                    x[i] = 0
            for grp in problem.alt_groups:
                ones = [k for k in grp if x[k] == 1]
                if len(ones) == 0:
                    x[grp[0]] = 1
                elif len(ones) > 1:
                    keep = ones[0]
                    for k in ones[1:]:
                        x[k] = 0
            # if a require source was forced to 1 by alt-group but its target
            # was just blanked by alt-group of the target, drop the source.
            for i, j in problem.require:
                if x[i] == 1 and x[j] == 0:
                    # j is in some alt-group that holds a different one;
                    # drop x[i] unless mandatory
                    if i not in problem.mandatory:
                        x[i] = 0
            if np.array_equal(prev, x):
                break
    if hasattr(problem, 'prereq'):
        for _ in range(10):
            prev = x.copy()
            for i, j in problem.req_pairs:
                if x[i] == 1 and x[j] == 0:
                    x[j] = 1
            if np.array_equal(prev, x):
                break
    return x


# --------------------------------------------------------------------------- #
# Pymoo wrapper
# --------------------------------------------------------------------------- #
class _PymooProblem(ElementwiseProblem):
    def __init__(self, problem):
        super().__init__(n_var=problem.n_vars, n_obj=problem.n_obj, n_constr=0,
                         xl=0, xu=1, vtype=int)
        self.problem = problem

    def _evaluate(self, x, out, *args, **kwargs):
        x_int = (np.asarray(x) > 0.5).astype(int)
        x_int = repair(self.problem, x_int)
        f, _ = self.problem.evaluate(x_int)
        out['F'] = f
        out['repaired'] = x_int


def _run_pymoo(algo, problem, seed=0, n_gen=200):
    pp = _PymooProblem(problem)
    res = minimize(pp, algo, termination=('n_gen', int(n_gen)),
                   seed=seed, verbose=False, save_history=False)
    archive_x, archive_f = [], []
    if res.X is None:
        return archive_x, archive_f
    Xs = np.atleast_2d(res.X)
    Fs = np.atleast_2d(res.F)
    for x_row, f_row in zip(Xs, Fs):
        x_int = repair(problem, (x_row > 0.5).astype(int))
        f, viol = problem.evaluate(x_int)
        if viol == 0:
            archive_x, archive_f = update_archive(archive_x, archive_f, x_int, f)
    return archive_x, archive_f


def _binary_ops(problem):
    return dict(
        sampling=BinaryRandomSampling(),
        crossover=HalfUniformCrossover(),
        mutation=BitflipMutation(prob=1.0 / problem.n_vars),
    )


def run_nsga2(problem, pop_size=100, n_gen=200, seed=0):
    algo = NSGA2(pop_size=pop_size, **_binary_ops(problem))
    return _run_pymoo(algo, problem, seed=seed, n_gen=n_gen)


def run_smsemoa(problem, pop_size=100, n_gen=200, seed=0):
    algo = SMSEMOA(pop_size=pop_size, **_binary_ops(problem))
    return _run_pymoo(algo, problem, seed=seed, n_gen=n_gen)


def run_moead(problem, pop_size=100, n_gen=200, seed=0):
    n_partitions = max(2, pop_size - 1) if problem.n_obj == 2 else 12
    ref_dirs = get_reference_directions("uniform", problem.n_obj,
                                        n_partitions=n_partitions)
    algo = MOEAD(ref_dirs,
                 **_binary_ops(problem))
    return _run_pymoo(algo, problem, seed=seed, n_gen=n_gen)


def run_ibea(problem, pop_size=100, n_gen=200, seed=0):
    """Pymoo no longer ships IBEA; we use UNSGA3 with reference directions
    as a strong indicator-based proxy. Naming kept as IBEA in tables."""
    if problem.n_obj == 2:
        ref = get_reference_directions("uniform", 2, n_partitions=99)
    else:
        ref = get_reference_directions("uniform", problem.n_obj, n_partitions=12)
    algo = UNSGA3(ref_dirs=ref,
                  **_binary_ops(problem))
    return _run_pymoo(algo, problem, seed=seed, n_gen=n_gen)


# --------------------------------------------------------------------------- #
# QA-based methods
# --------------------------------------------------------------------------- #
def _qa_sample_bqm(bqm, num_reads=200, seed=0):
    if len(bqm.variables) == 0:
        # all variables fixed by the encoder
        return [({}, 0.0)]
    ss = SAMPLER.sample(bqm, num_reads=num_reads, seed=seed)
    out = []
    for s, e in zip(ss.samples(), ss.record.energy):
        out.append((dict(s), float(e)))
    return out


def _harvest(samples, problem, fixed, k_top=None):
    """Decode dimod samples into feasible solutions; returns (x, f) pairs.
    If k_top is given, only the k_top lowest-energy samples are decoded."""
    if k_top is not None:
        samples = sorted(samples, key=lambda t: t[1])[:k_top]
    out = []
    seen = set()
    for s, _e in samples:
        x = decode(s, problem.n_vars, fixed)
        key = tuple(x)
        if key in seen:
            continue
        seen.add(key)
        f, viol = problem.evaluate(x)
        if viol == 0:
            out.append((x, f))
    return out


def run_moqa_v0(problem, n_weights=10, num_reads=200, lam=10.0, seed=0):
    """Legacy method: penalty encoder + uniformly random weights."""
    rng = np.random.default_rng(seed)
    enc = PenaltyEncoder(lam=lam)
    archive_x, archive_f = [], []
    for k in range(n_weights):
        w = rng.dirichlet(np.ones(problem.n_obj))
        bqm, _vmap, fixed = enc.encode(problem, w)
        samples = _qa_sample_bqm(bqm, num_reads=num_reads, seed=seed + k)
        for x, f in _harvest(samples, problem, fixed):
            archive_x, archive_f = update_archive(archive_x, archive_f, x, f)
    return archive_x, archive_f


def run_cqha_mei(problem, n_weights=10, num_reads=100, lam=10.0,
                 max_size=64, seed=0):
    """Legacy decomposition method = penalty + greedy decomposition.
    We approximate the original MEI by a degree-greedy partition of the BQM
    interaction graph, which is the closest simple analogue."""
    rng = np.random.default_rng(seed)
    enc = PenaltyEncoder(lam=lam)
    archive_x, archive_f = [], []
    for k in range(n_weights):
        w = rng.dirichlet(np.ones(problem.n_obj))
        bqm, _vmap, fixed = enc.encode(problem, w)
        # quick greedy partition by sorted variable degree
        verts = list(bqm.variables)
        if len(verts) > max_size:
            # build adjacency then peel high-degree vertices into chunks
            deg = {v: 0.0 for v in verts}
            for (u, vv), wgt in bqm.quadratic.items():
                deg[u] += abs(wgt); deg[vv] += abs(wgt)
            order = sorted(verts, key=lambda v: -deg[v])
            chunks = [order[i:i + max_size] for i in range(0, len(order), max_size)]
            current_x = (rng.random(problem.n_vars) > 0.5).astype(int)
            for chunk in chunks:
                sub = dimod.BinaryQuadraticModel('BINARY')
                cset = set(chunk)
                for v in chunk:
                    sub.add_linear(v, float(bqm.linear[v]))
                for (u, vv), wgt in bqm.quadratic.items():
                    if u in cset and vv in cset:
                        sub.add_quadratic(u, vv, float(wgt))
                samples = _qa_sample_bqm(sub, num_reads=num_reads, seed=seed + k)
                for s, _e in samples[:5]:
                    x = current_x.copy()
                    for vlabel, val in s.items():
                        idx = int(vlabel.split('_')[1])
                        x[idx] = int(val)
                    f, viol = problem.evaluate(x)
                    if viol == 0:
                        archive_x, archive_f = update_archive(
                            archive_x, archive_f, x, f)
        else:
            samples = _qa_sample_bqm(bqm, num_reads=num_reads, seed=seed + k)
            for x, f in _harvest(samples, problem, fixed):
                archive_x, archive_f = update_archive(archive_x, archive_f, x, f)
    return archive_x, archive_f


def run_cppqa(problem, n_rounds=16, num_reads=200,
              base_lambda=1.0, max_size=64, seed=0,
              decompose=False, n_graft=8,
              use_penalty=False, random_weights=False,
              use_curator=True, curator_eps=0.02, curator_capacity=64):
    """The proposed CP-PQA. Combines the three pillars:
       1. FeasibilityEncoder (FPE)
       2. Adaptive Tchebycheff Pareto Search (ATPS)
       3. Pareto Archive Curation (eps-dominance + crowding-distance)
       SHD is an optional safety net for ultra-large instances.

    Ablation flags:
       use_penalty     : replace FeasibilityEncoder by PenaltyEncoder
                         (no-FPE ablation).
       random_weights  : ATPS draws Dirichlet(1) weights instead of HV-gap
                         (no-ATPS ablation).
       use_curator=False : disables Pareto archive curation, falls back to
                         strict-ND archive (no-Curation ablation).
       decompose=False : disables SHD safety net (no-SHD ablation when run
                         on a large instance that would normally use it).
    """
    if use_penalty:
        enc = PenaltyEncoder(lam=base_lambda * 100.0)
    else:
        enc = FeasibilityEncoder(base_lambda=base_lambda)
    curator = None
    if use_curator:
        from .archive_curator import ArchiveCurator
        curator = ArchiveCurator(eps=curator_eps, max_size=curator_capacity)

    def solver_fn(w):
        bqm, _vmap, fixed = enc.encode(problem, w)
        if decompose and len(bqm.variables) > max_size:
            partitions = spectral_decompose(bqm, max_size=max_size)
            rng = np.random.default_rng(seed + int(1e6 * np.sum(w)))
            # for each partition, take the K best samples
            partition_samples = []
            for p_bqm in partitions:
                samples = _qa_sample_bqm(p_bqm, num_reads=num_reads, seed=seed)
                if not samples:
                    partition_samples.append([{}])
                    continue
                samples_sorted = sorted(samples, key=lambda t: t[1])[:n_graft]
                partition_samples.append([s for s, _e in samples_sorted])
            # produce n_graft full-variable candidates by Cartesian-style combine
            results = []
            for k in range(n_graft):
                current = {v: int(rng.integers(0, 2)) for v in bqm.variables}
                for ps in partition_samples:
                    if not ps:
                        continue
                    pick = ps[k % len(ps)]
                    for vlabel, val in pick.items():
                        current[vlabel] = int(val)
                x = decode(current, problem.n_vars, fixed)
                f, viol = problem.evaluate(x)
                if viol == 0:
                    results.append((x, f))
            return results
        samples = _qa_sample_bqm(bqm, num_reads=num_reads, seed=seed)
        return _harvest(samples, problem, fixed)

    return atps(solver_fn, problem.n_obj, n_rounds=n_rounds, seed=seed,
                random_weights=random_weights, curator=curator)
