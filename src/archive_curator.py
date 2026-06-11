"""Pareto archive curator (Pillar 3).

Combines two ideas from the EMOA literature:
  (i)  eps-dominance (Laumanns et al. 2002) -- relaxes strict
       Pareto dominance so that points within an eps-box of an
       existing archive point are still kept.  This avoids the
       starvation regime where a small number of strict-ND
       points dominates the entire image of f.
  (ii) crowding-distance pruning (Deb et al. NSGA-II 2002) --
       when the archive exceeds max_size, the point with the
       smallest crowding distance is removed, preserving the
       extreme points and the most diverse interior set.

Diversity bound (Theorem 6.1 in the paper).
Let f : X -> R^k be the objective vector and let A_T be the
archive after T accept/reject events.  Then for any tau in
the image f(X^*) of the true Pareto set,
    min_{a in A_T} d_inf(f(a), tau)  <=  eps * ||f||_inf,
where d_inf is the L_inf distance (proof in App. B).
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple

Solution = Tuple[list, np.ndarray]   # (decision, objective vector)


def _eps_dominates(a: np.ndarray, b: np.ndarray, eps: float) -> bool:
    """True iff `a` eps-dominates `b` (minimisation)."""
    return np.all((1.0 - eps) * a <= b) and np.any((1.0 - eps) * a < b)


def _crowding(F: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance for a single (already non-dominated) front."""
    n, k = F.shape
    if n <= 2:
        return np.full(n, np.inf)
    cd = np.zeros(n)
    for j in range(k):
        order = np.argsort(F[:, j])
        fmin, fmax = F[order[0], j], F[order[-1], j]
        denom = max(fmax - fmin, 1e-12)
        cd[order[0]]  = np.inf
        cd[order[-1]] = np.inf
        for i in range(1, n - 1):
            cd[order[i]] += (F[order[i + 1], j] - F[order[i - 1], j]) / denom
    return cd


class ArchiveCurator:
    """eps-dominance archive with crowding-distance pruning."""

    def __init__(self, eps: float = 0.01, max_size: int = 64):
        self.eps = float(eps)
        self.max_size = int(max_size)
        self._items: List[Solution] = []

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> List[Solution]:
        return list(self._items)

    def F(self) -> np.ndarray:
        return np.asarray([f for _, f in self._items], dtype=float) if self._items else np.zeros((0, 1))

    def offer(self, x: list, f: np.ndarray) -> bool:
        """Try to insert (x, f).  Returns True if accepted."""
        f = np.asarray(f, dtype=float)
        # 1. reject if eps-dominated by anyone already in
        for _, g in self._items:
            if _eps_dominates(g, f, self.eps):
                return False
        # 2. add, then evict everyone eps-dominated by f
        kept = [(y, g) for (y, g) in self._items if not _eps_dominates(f, g, self.eps)]
        kept.append((x, f))
        self._items = kept
        # 3. capacity: prune by crowding distance
        while len(self._items) > self.max_size:
            F = np.asarray([fv for _, fv in self._items])
            cd = _crowding(F)
            idx = int(np.argmin(cd))
            self._items.pop(idx)
        return True

    # Convenience wrappers --------------------------------------
    def extend(self, sols: List[Solution]) -> int:
        return sum(int(self.offer(x, f)) for x, f in sols)

    def clear(self):
        self._items.clear()
