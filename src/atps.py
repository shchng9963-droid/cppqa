"""
Adaptive Tchebycheff Pareto Search (ATPS).

Replaces the random-weight scalarization of MOQA-v0 with a hypervolume-
gradient driven adaptive scheme that targets the most under-covered region
of the Pareto front at each round. The objective is to give the QA-based
multi-objective workflow a real Pareto-coverage guarantee.

Algorithm sketch (one outer round t):
    1. Maintain external archive A_{t-1} (non-dominated solutions found so far)
       in objective space.
    2. Compute a reference point z* = max(f_k) over A_{t-1}, padded by 1%.
    3. Pick a search direction w_t that points to the largest empty box of
       A_{t-1} (the so-called HV gradient). For two objectives this is just
       the largest gap between consecutive sorted archive points; for k>=3
       we sample candidate weights uniformly from a Dirichlet(1) and pick
       the one that maximises (z* - max_k w_k * f_k(a)) over the archive
       (i.e. the weight whose Tchebycheff floor is *highest*, hence pointing
       at the missing corner).
    4. Build a Tchebycheff-style scalarization:
            min_x  max_k w_t[k] * (f_k(x) - z*[k])
       and approximate it by its smooth upper bound
            sum_k w_t[k] * f_k(x)            (used as the actual QUBO objective)
       for the current round, while w_t is chosen by the Tchebycheff metric.
    5. Sample R candidate solutions from the QA back-end. Push all feasible
       solutions through `update_archive` which removes dominated solutions
       and keeps the archive Pareto-thin.

The function `atps` is back-end agnostic: it takes a `solver_fn(weights) ->
list of (x, f)` callback, so the same routine drives both the simulated
annealer and the D-Wave QPU.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, List, Tuple


def _dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def update_archive(archive_x, archive_f, new_x, new_f):
    """Add (x, f) to archive iff non-dominated; remove newly dominated.
    Duplicate objective vectors are treated as already-archived (no growth)."""
    keep_x, keep_f = [], []
    dominated = False
    for ax, af in zip(archive_x, archive_f):
        if np.allclose(af, new_f, atol=1e-12):
            dominated = True
            keep_x.append(ax); keep_f.append(af)
            continue
        if _dominates(af, new_f):
            dominated = True
            keep_x.append(ax); keep_f.append(af)
        elif _dominates(new_f, af):
            continue
        else:
            keep_x.append(ax); keep_f.append(af)
    if not dominated:
        keep_x.append(new_x); keep_f.append(new_f)
    return keep_x, keep_f


def _gap_weight(archive_f, n_obj, n_candidates=64, rng=None):
    """
    Pick a weight vector that targets the biggest empty corner.
    For 2 objectives we use the largest sorted-gap heuristic. For >2
    objectives we sample n_candidates Dirichlet(1) directions and pick
    the one with the largest Tchebycheff slack.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not archive_f:
        return rng.dirichlet(np.ones(n_obj))
    A = np.asarray(archive_f)
    if n_obj == 2:
        order = np.argsort(A[:, 0])
        sorted_A = A[order]
        gaps = np.linalg.norm(np.diff(sorted_A, axis=0), axis=1)
        if len(gaps) == 0:
            return np.array([0.5, 0.5])
        idx = int(np.argmax(gaps))
        # Direction perpendicular to the largest gap, normalised to the simplex.
        diff = sorted_A[idx + 1] - sorted_A[idx]
        if np.linalg.norm(diff) < 1e-12:
            return rng.dirichlet(np.ones(n_obj))
        # weight on f_0 dimension proportional to gap projection
        w = np.array([abs(diff[1]), abs(diff[0])])
        w = w / max(w.sum(), 1e-12)
        return w
    # k>=3: sample candidate directions and choose by Tchebycheff slack
    z_star = A.min(axis=0) - 1e-3
    candidates = rng.dirichlet(np.ones(n_obj), size=n_candidates)
    # for each candidate compute min_a max_k w_k (f_k(a) - z*_k)
    slacks = []
    for w in candidates:
        tchebs = np.max(w * (A - z_star), axis=1)
        slacks.append(np.min(tchebs))   # high = under-covered
    return candidates[int(np.argmax(slacks))]


def atps(solver_fn: Callable[[np.ndarray], List[Tuple[list, np.ndarray]]],
         n_obj: int,
         n_rounds: int = 16,
         seed: int = 0,
         random_weights: bool = False,
         curator=None):
    """
    Run Adaptive Tchebycheff Pareto Search.

    Parameters
    ----------
    solver_fn : function w (np.ndarray, length n_obj) -> list of (x, f) tuples,
                where x is a binary list and f is a numpy array of length n_obj.
    n_obj     : number of objectives.
    n_rounds  : number of outer rounds (each round = one weighted QA call).
    random_weights : if True, draw weights from a uniform Dirichlet(1) instead
                     of using the HV-gap finder. Used by the no-ATPS ablation.
    curator   : optional ArchiveCurator (Pillar 3). When supplied, all
                feasible candidate samples are offered to the curator
                (eps-dominance + crowding-distance pruning) instead of the
                strict-ND update_archive.

    Returns
    -------
    archive_x, archive_f : non-dominated solutions and their objective values.
    """
    rng = np.random.default_rng(seed)
    archive_x, archive_f = [], []
    for t in range(n_rounds):
        if random_weights:
            w = rng.dirichlet(np.ones(n_obj))
        else:
            w = _gap_weight(archive_f if curator is None else list(curator.F()),
                            n_obj, rng=rng)
        candidates = solver_fn(w)
        for x, f in candidates:
            if curator is not None:
                curator.offer(x, np.asarray(f))
            else:
                archive_x, archive_f = update_archive(archive_x, archive_f, x, f)
    if curator is not None:
        items = curator.items()
        archive_x = [x for x, _ in items]
        archive_f = [f for _, f in items]
    return archive_x, archive_f
