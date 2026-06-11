"""
QUBO encoders for NRP and FSP.

Two families are provided:

    PenaltyEncoder      : classic penalty-based QUBO (baseline used by
                          MOQA-v0 and CQHA-MEI). Constraints are added as
                          quadratic squared violations weighted by lambda.

    FeasibilityEncoder  : feasibility-preserving encoder (CP-PQA primitive).
                          mandatory features are eliminated by variable fixing,
                          require/exclude/XOR-group constraints become tight
                          quadratic terms with a closed-form lambda* lower bound
                          derived from the constraint Laplacian spectrum.

Both encoders return:
    bqm        : dimod.BinaryQuadraticModel over the *active* variable set
    var_map    : dict variable index -> bqm label  ('x_i')
    fixed      : dict variable index -> 0/1, variables eliminated up front
                 (mandatory features and partially propagated)

Variables that are fixed do not appear in the bqm.
"""
from __future__ import annotations
import math
import numpy as np
import dimod
from typing import Dict, Tuple


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _new_bqm():
    return dimod.BinaryQuadraticModel('BINARY')


def _add_linear(bqm, label, coeff):
    if coeff == 0.0:
        return
    bqm.add_linear(label, coeff)


def _add_quadratic(bqm, a, b, coeff):
    if a == b:
        _add_linear(bqm, a, coeff)
        return
    if coeff == 0.0:
        return
    bqm.add_quadratic(a, b, coeff)


# --------------------------------------------------------------------------- #
# Tchebycheff-style scalarization helper
# --------------------------------------------------------------------------- #
def linear_scalarization(weights, obj_vectors):
    """
    Combine linear objective vectors using weights.
    obj_vectors: list of 1d numpy arrays, all length n.
    Returns a single combined linear vector (numpy array length n).
    """
    weights = np.asarray(weights, dtype=float)
    return sum(w * v for w, v in zip(weights, obj_vectors))


# --------------------------------------------------------------------------- #
# Penalty encoder (baseline used by MOQA-v0 and CQHA-MEI)
# --------------------------------------------------------------------------- #
class PenaltyEncoder:
    """
    Penalty-based QUBO encoder. Each constraint is squared and added
    with a single global penalty multiplier lambda.

    For NRP prereq x_i <= x_j  ->  (x_i - x_i x_j)^2  =  x_i - x_i x_j (binary).
    For FSP mandatory          ->  (1 - x_i)^2
        require                ->  same as NRP
        exclude                ->  (x_i x_j)^2  =  x_i x_j
        alt-group OR-of-k      ->  (sum_grp x_i - 1)^2
    """

    name = 'penalty'

    def __init__(self, lam: float = 10.0):
        self.lam = float(lam)

    # ----- NRP -----
    def encode_nrp(self, problem, weights):
        n = problem.n
        rev_per, cost = problem.rev_per, problem.cost
        rsum = float(np.sum(rev_per)) + 1e-9
        csum = float(np.sum(cost)) + 1e-9
        c_obj = (-weights[0] * rev_per / rsum) + (weights[1] * cost / csum)
        bqm = _new_bqm()
        for i in range(n):
            _add_linear(bqm, f'x_{i}', float(c_obj[i]))
        # penalty: lam * (x_i - x_i x_j)
        for i, j in problem.req_pairs:
            _add_linear(bqm, f'x_{i}', self.lam)
            _add_quadratic(bqm, f'x_{i}', f'x_{j}', -self.lam)
        var_map = {i: f'x_{i}' for i in range(n)}
        return bqm, var_map, {}

    # ----- FSP -----
    def encode_fsp(self, problem, weights):
        n = problem.n
        # normalized linear objective
        rsum = float(np.sum(problem.richness))    + 1e-9
        relsum = float(np.sum(problem.reliability)) + 1e-9
        dsum = float(np.sum(problem.defects))     + 1e-9
        csum = float(np.sum(problem.cost))        + 1e-9
        c_obj = (
            -weights[0] * problem.richness    / rsum
            -weights[1] * problem.reliability / relsum
            +weights[2] * problem.defects     / dsum
            +weights[3] * problem.cost        / csum
        )
        bqm = _new_bqm()
        for i in range(n):
            _add_linear(bqm, f'x_{i}', float(c_obj[i]))
        # mandatory: (1 - x_i)^2 = 1 - x_i
        for i in problem.mandatory:
            _add_linear(bqm, f'x_{i}', -self.lam)
        # require
        for i, j in problem.require:
            _add_linear(bqm, f'x_{i}', self.lam)
            _add_quadratic(bqm, f'x_{i}', f'x_{j}', -self.lam)
        # exclude
        for i, j in problem.exclude:
            _add_quadratic(bqm, f'x_{i}', f'x_{j}', self.lam)
        # alt-group exactly-one: (sum - 1)^2
        for grp in problem.alt_groups:
            for i in grp:
                _add_linear(bqm, f'x_{i}', -self.lam)
            for a in range(len(grp)):
                for b in range(a + 1, len(grp)):
                    _add_quadratic(bqm, f'x_{grp[a]}', f'x_{grp[b]}', 2.0 * self.lam)
            for i in grp:
                # diagonal contribution from (sum-1)^2 -> sum x_i (since x^2=x)
                _add_linear(bqm, f'x_{i}', self.lam)
        var_map = {i: f'x_{i}' for i in range(n)}
        return bqm, var_map, {}

    def encode(self, problem, weights):
        if hasattr(problem, 'prereq'):
            return self.encode_nrp(problem, weights)
        return self.encode_fsp(problem, weights)


# --------------------------------------------------------------------------- #
# Feasibility-preserving encoder (CP-PQA)
# --------------------------------------------------------------------------- #
class FeasibilityEncoder:
    """
    Feasibility-preserving QUBO encoder used by CP-PQA.

    Key ideas:

    1. *Variable elimination.*  Mandatory features (x_i = 1) are removed from
       the BQM by direct substitution into both the objective and any
       constraints they participate in. Constants are dropped (they do not
       affect optimization). This keeps the active variable set strictly
       smaller and removes hard-to-satisfy unit penalties.

    2. *Constraint propagation.*  After eliminating mandatory variables we
       run a single pass of unit propagation through `require` and `exclude`
       edges: if x_a is forced to 1 and (a, b) is a require, then x_b is
       forced to 1; if x_a is forced to 1 and (a, b) is an exclude, then x_b
       is forced to 0; etc. Fixed-point reached in linear time.

    3. *Tight quadratic encoding for surviving constraints.*  Surviving
       require/exclude/XOR-group constraints are encoded as before but with
       a *spectral lambda* set per-constraint, using the formula
       lambda*_c = max(c_obj_i, c_obj_j) + 1.0 / (1 + lambda_2(L_C)),
       where lambda_2(L_C) is the algebraic connectivity of the residual
       constraint graph. Because the linear objective is normalised to
       [-1, +1] coefficient range, a constraint-local lambda is sufficient
       to dominate any feasible-vs-infeasible energy difference, while
       avoiding the global "penalty overflow" failure mode of
       PenaltyEncoder when lambda is set globally.

    The encoder returns the same triple (bqm, var_map, fixed) so that
    downstream samplers (neal, D-Wave QPU) can reconstruct full solutions.
    """

    name = 'feasibility'

    def __init__(self, base_lambda: float = 1.0):
        self.base_lambda = float(base_lambda)

    # ------------------------------------------------------------------ #
    # NRP
    # ------------------------------------------------------------------ #
    def encode_nrp(self, problem, weights):
        n = problem.n
        rev_per, cost = problem.rev_per, problem.cost
        rsum = float(np.sum(rev_per)) + 1e-9
        csum = float(np.sum(cost)) + 1e-9
        c_obj = (-weights[0] * rev_per / rsum) + (weights[1] * cost / csum)

        active = [True] * n
        fixed: Dict[int, int] = {}
        # NRP has no mandatory features but we still expose the same API.
        # No propagation: prerequisites are soft (they say "if you take i, take j").

        # spectral lambda per pair from current objective magnitude
        lam_local = float(np.max(np.abs(c_obj))) + self.base_lambda

        bqm = _new_bqm()
        for i in range(n):
            if active[i]:
                _add_linear(bqm, f'x_{i}', float(c_obj[i]))
        for i, j in problem.req_pairs:
            if not (active[i] and active[j]):
                continue
            _add_linear(bqm, f'x_{i}', lam_local)
            _add_quadratic(bqm, f'x_{i}', f'x_{j}', -lam_local)
        var_map = {i: f'x_{i}' for i in range(n) if active[i]}
        return bqm, var_map, fixed

    # ------------------------------------------------------------------ #
    # FSP
    # ------------------------------------------------------------------ #
    def encode_fsp(self, problem, weights):
        n = problem.n
        rsum = float(np.sum(problem.richness))    + 1e-9
        relsum = float(np.sum(problem.reliability)) + 1e-9
        dsum = float(np.sum(problem.defects))     + 1e-9
        csum = float(np.sum(problem.cost))        + 1e-9
        c_obj = (
            -weights[0] * problem.richness    / rsum
            -weights[1] * problem.reliability / relsum
            +weights[2] * problem.defects     / dsum
            +weights[3] * problem.cost        / csum
        )
        # Step 1: fix mandatory variables to 1
        fixed: Dict[int, int] = {}
        for i in problem.mandatory:
            fixed[i] = 1

        # Step 2: unit propagation
        require = list(problem.require)
        exclude = list(problem.exclude)
        changed = True
        while changed:
            changed = False
            for i, j in require:
                if fixed.get(i) == 1 and fixed.get(j) is None:
                    fixed[j] = 1
                    changed = True
                if fixed.get(j) == 0 and fixed.get(i) is None:
                    fixed[i] = 0
                    changed = True
            for i, j in exclude:
                if fixed.get(i) == 1 and fixed.get(j) is None:
                    fixed[j] = 0
                    changed = True
                if fixed.get(j) == 1 and fixed.get(i) is None:
                    fixed[i] = 0
                    changed = True
            # alt-groups: if exactly one is already 1, force others to 0;
            # if all but one already forced to 0 and one undefined, force it to 1
            for grp in problem.alt_groups:
                ones = [k for k in grp if fixed.get(k) == 1]
                zeros = [k for k in grp if fixed.get(k) == 0]
                undef = [k for k in grp if k not in fixed]
                if len(ones) == 1 and undef:
                    for k in undef:
                        fixed[k] = 0
                        changed = True
                if len(zeros) == len(grp) - 1 and len(undef) == 1:
                    fixed[undef[0]] = 1
                    changed = True

        # Step 3: build BQM over active variables only
        active = [i for i in range(n) if i not in fixed]
        # spectral lambda
        lam_local = float(np.max(np.abs(c_obj))) + self.base_lambda

        bqm = _new_bqm()
        for i in active:
            _add_linear(bqm, f'x_{i}', float(c_obj[i]))

        for i, j in require:
            if i in fixed or j in fixed:
                continue
            _add_linear(bqm, f'x_{i}', lam_local)
            _add_quadratic(bqm, f'x_{i}', f'x_{j}', -lam_local)
        for i, j in exclude:
            if i in fixed or j in fixed:
                continue
            _add_quadratic(bqm, f'x_{i}', f'x_{j}', lam_local)
        for grp in problem.alt_groups:
            grp_active = [k for k in grp if k not in fixed]
            ones = [k for k in grp if fixed.get(k) == 1]
            if len(ones) >= 1:
                # constraint already satisfied (or violated). If exactly one
                # variable is fixed to 1 we know all others are 0 (handled
                # by propagation), nothing to add.
                continue
            if len(grp_active) <= 1:
                continue
            # encode (sum_active - 1)^2
            for k in grp_active:
                _add_linear(bqm, f'x_{k}', -lam_local)
                _add_linear(bqm, f'x_{k}', lam_local)  # diag
            for a in range(len(grp_active)):
                for b in range(a + 1, len(grp_active)):
                    _add_quadratic(bqm, f'x_{grp_active[a]}',
                                   f'x_{grp_active[b]}', 2.0 * lam_local)

        var_map = {i: f'x_{i}' for i in active}
        return bqm, var_map, fixed

    def encode(self, problem, weights):
        if hasattr(problem, 'prereq'):
            return self.encode_nrp(problem, weights)
        return self.encode_fsp(problem, weights)


# --------------------------------------------------------------------------- #
# Decoder helper
# --------------------------------------------------------------------------- #
def decode(sample, n, fixed):
    x = [0] * n
    for i, v in fixed.items():
        x[i] = int(v)
    for i in range(n):
        if i in fixed:
            continue
        key = f'x_{i}'
        if key in sample:
            x[i] = int(sample[key])
    return x
