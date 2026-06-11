"""
Quality indicators (HV, IGD, SP, NoP) and feasibility metrics.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist


def _fast_nds(arr):
    arr = np.array(arr)
    if len(arr) == 0:
        return arr
    _, idx = np.unique(np.round(arr, 10), axis=0, return_index=True)
    u = arr[np.sort(idx)]
    m = len(u)
    if m <= 1:
        return u
    is_nd = np.ones(m, dtype=bool)
    chunk = min(500, m)
    for i in range(0, m, chunk):
        end = min(i + chunk, m)
        batch = u[i:end]
        le = u[None, :, :] <= batch[:, None, :]
        lt = u[None, :, :] <  batch[:, None, :]
        dom = np.all(le, axis=2) & np.any(lt, axis=2)
        for k in range(end - i):
            dom[k, i + k] = False
        is_nd[i:end] = ~np.any(dom, axis=1)
    return u[is_nd]


def hypervolume(front, ref_point):
    front = np.array(front)
    ref_point = np.array(ref_point, dtype=float)
    if len(front) == 0:
        return 0.0
    nd = _fast_nds(front)
    if len(nd) == 0:
        return 0.0
    valid = np.all(nd < ref_point, axis=1)
    nd = nd[valid]
    if len(nd) == 0:
        return 0.0
    n_obj = nd.shape[1]
    if n_obj == 2:
        pts = nd[np.argsort(nd[:, 0])]
        hv = 0.0
        prev_y = ref_point[1]
        for i in range(len(pts)):
            if pts[i, 1] < prev_y:
                x_right = ref_point[0] if i == len(pts) - 1 else pts[i + 1, 0]
                hv += (x_right - pts[i, 0]) * (prev_y - pts[i, 1])
                prev_y = pts[i, 1]
        return float(hv)
    rng = np.random.default_rng(0)
    n_samples = 50000
    lb = nd.min(axis=0)
    samples = rng.uniform(lb, ref_point, (n_samples, n_obj))
    dominated = np.zeros(n_samples, dtype=bool)
    for p in nd:
        dominated |= np.all(samples >= p[None, :], axis=1)
    return float(np.prod(ref_point - lb) * np.mean(dominated))


def inverted_generational_distance(front, pareto_front):
    front = np.array(front)
    pf = np.array(pareto_front)
    if len(front) == 0 or len(pf) == 0:
        return float('inf')
    dists = cdist(pf, front)
    return float(np.mean(dists.min(axis=1)))


def spacing(front):
    front = np.array(front)
    if len(front) <= 1:
        return 0.0
    dists = cdist(front, front)
    np.fill_diagonal(dists, np.inf)
    d = dists.min(axis=1)
    mean_d = np.mean(d)
    return float(np.sqrt(np.sum((d - mean_d) ** 2) / max(len(d) - 1, 1)))


def reference_point(all_archives, eps=0.05):
    """Compute a shared reference point from a list of front arrays."""
    arrays = [np.asarray(a) for a in all_archives if len(a) > 0]
    if not arrays:
        return None
    cat = np.concatenate(arrays, axis=0)
    z_max = cat.max(axis=0)
    z_min = cat.min(axis=0)
    pad = (z_max - z_min) * eps + 1e-9
    return z_max + pad


def union_pareto(all_archives):
    arrays = [np.asarray(a) for a in all_archives if len(a) > 0]
    if not arrays:
        return np.zeros((0, 0))
    cat = np.concatenate(arrays, axis=0)
    return _fast_nds(cat)


def feasibility_rate(samples_x, problem):
    if not samples_x:
        return 0.0
    feas = 0
    for x in samples_x:
        _, viol = problem.evaluate(x)
        if viol == 0:
            feas += 1
    return feas / len(samples_x)
