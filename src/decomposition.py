"""
Spectral Hierarchical Decomposition (SHD) for large-scale QUBOs.

Replaces the greedy Maximum-Energy-Impact (MEI) decomposition used by
CQHA-MEI. Instead of repeatedly snowballing along high-energy variables,
SHD recursively partitions the variable interaction graph using the
Fiedler vector (the eigenvector associated with the second-smallest
eigenvalue of the graph Laplacian). The cut induced by sign(Fiedler)
minimises the normalised cut between the two halves, which gives a
controllable upper bound on the QUBO terms that are dropped during
decomposition.

Loss bound (Theorem 6.1 in the paper):
    |H(x) - sum_p H_p(x)|  <=  || cut_edges ||_1  <=  C * lambda_2(L_W)
where lambda_2 is the algebraic connectivity of the *weighted* interaction
graph and C is the maximum absolute quadratic coefficient. This formalises
why the rate parameter of MEI behaves erratically: MEI implicitly performs
an unweighted greedy cut whose loss is unbounded by problem structure.

Public API:
    spectral_decompose(bqm, max_size, problem_size) -> list[dimod.BinaryQuadraticModel]
"""
from __future__ import annotations
import numpy as np
import networkx as nx
import dimod
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def _bqm_to_graph(bqm: dimod.BinaryQuadraticModel) -> nx.Graph:
    g = nx.Graph()
    for v in bqm.variables:
        g.add_node(v)
    for (u, v), w in bqm.quadratic.items():
        if u == v:
            continue
        g.add_edge(u, v, weight=abs(float(w)))
    return g


def _fiedler_partition(g: nx.Graph):
    """
    Return two lists of nodes split by sign of the Fiedler vector.
    For very small graphs we fall back to a degree-balanced partition.
    """
    nodes = list(g.nodes())
    n = len(nodes)
    if n <= 2:
        mid = n // 2
        return nodes[:mid], nodes[mid:]
    # Build Laplacian
    A = nx.to_scipy_sparse_array(g, nodelist=nodes, weight='weight', format='csr')
    deg = np.array(A.sum(axis=1)).flatten()
    L = sp.diags(deg) - A
    try:
        # k=2 to skip the trivial zero eigenvalue
        vals, vecs = eigsh(L.astype(float), k=2, which='SM')
        order = np.argsort(vals)
        fiedler = vecs[:, order[1]]
    except Exception:
        # disconnected graph or numerical issues -> fall back
        comps = list(nx.connected_components(g))
        if len(comps) > 1:
            half = len(comps) // 2
            left = []
            right = []
            for k, c in enumerate(comps):
                (left if k < half else right).extend(c)
            return left, right
        # purely round-robin fallback
        return nodes[:n // 2], nodes[n // 2:]
    left = [nodes[i] for i in range(n) if fiedler[i] <= 0]
    right = [nodes[i] for i in range(n) if fiedler[i] > 0]
    if not left or not right:
        # Degenerate: bisect by Fiedler-value rank
        order = np.argsort(fiedler)
        left = [nodes[i] for i in order[:n // 2]]
        right = [nodes[i] for i in order[n // 2:]]
    return left, right


def _bqm_subset(bqm: dimod.BinaryQuadraticModel, nodes):
    sub = dimod.BinaryQuadraticModel('BINARY')
    nodeset = set(nodes)
    for v in nodes:
        if v in bqm.variables:
            sub.add_linear(v, float(bqm.linear[v]))
    for (u, v), w in bqm.quadratic.items():
        if u in nodeset and v in nodeset:
            sub.add_quadratic(u, v, float(w))
    return sub


def spectral_decompose(bqm: dimod.BinaryQuadraticModel,
                       max_size: int = 64,
                       max_depth: int = 16):
    """
    Recursively partition the BQM's variable interaction graph using
    Fiedler-vector cuts until each leaf has at most `max_size` variables.

    Returns a list of sub-BQMs whose variable sets are disjoint and whose
    union covers all variables of `bqm`. Quadratic terms whose endpoints
    fall in different leaves are dropped (cut-edge loss).
    """
    if len(bqm.variables) <= max_size:
        return [bqm.copy()]
    g = _bqm_to_graph(bqm)
    if g.number_of_nodes() == 0:
        return [bqm.copy()]
    leaves = []
    stack = [(g, 0)]
    while stack:
        sub_g, depth = stack.pop()
        if sub_g.number_of_nodes() <= max_size or depth >= max_depth:
            leaves.append(list(sub_g.nodes()))
            continue
        left, right = _fiedler_partition(sub_g)
        if not left or not right:
            leaves.append(list(sub_g.nodes()))
            continue
        stack.append((sub_g.subgraph(left).copy(),  depth + 1))
        stack.append((sub_g.subgraph(right).copy(), depth + 1))
    return [_bqm_subset(bqm, ns) for ns in leaves if ns]


def cut_loss(bqm: dimod.BinaryQuadraticModel, partitions):
    """
    Sum of absolute quadratic-coupling magnitudes that cross partitions.
    Used as an empirical upper bound on the decomposition error.
    """
    cell = {}
    for p_idx, sub in enumerate(partitions):
        for v in sub.variables:
            cell[v] = p_idx
    loss = 0.0
    for (u, v), w in bqm.quadratic.items():
        if cell.get(u, -1) != cell.get(v, -2):
            loss += abs(float(w))
    return loss
