"""
NRP and FSP problem definitions used by CP-PQA experiments.

Each problem exposes:
    - n_vars: number of binary decision variables
    - n_obj : number of objectives
    - evaluate(x) -> (f, viol) where f is np.ndarray of length n_obj and viol is integer
    - constraint_graph() -> networkx graph with edges between variables that
      co-occur in a constraint. Used by spectral decomposition.
    - constraint_density() -> |E_C| / (n choose 2)
"""
from __future__ import annotations
import json
import numpy as np
import networkx as nx


def load_json(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# NRP
# --------------------------------------------------------------------------- #
class NRP:
    """
    Multi-objective Next Release Problem (2 objectives).

    Decision: x_i in {0,1}, i=1..n_req, indicating whether req i is selected.
    Objectives (all minimization after sign flip):
        f1(x) = -sum_i revenue_i * x_i           (maximize total revenue)
        f2(x) =  sum_i cost_i    * x_i           (minimize total cost)
    Constraints:
        prereq[i,j] = 1  =>  x_i <= x_j   (req i needs req j)
    """

    def __init__(self, data):
        self.n = int(data['n_req'])
        self.m = int(data['n_stake'])
        self.revenue = np.array(data['revenue'])         # (m, n)
        self.cost    = np.array(data['cost']).astype(float)  # (n,)
        self.prereq  = np.array(data['prereq']).astype(int)  # (n, n)
        self.req_pairs = [(i, j) for i in range(self.n)
                          for j in range(self.n) if self.prereq[i, j] == 1]
        self.n_vars = self.n
        self.n_obj  = 2
        # Aggregated revenue per requirement (sum across stakeholders)
        self.rev_per = np.sum(self.revenue, axis=0).astype(float)

    def evaluate(self, x):
        x = np.asarray(x, dtype=int)
        f1 = -float(np.dot(self.rev_per, x))
        f2 =  float(np.dot(self.cost,    x))
        viol = 0
        for i, j in self.req_pairs:
            if x[i] == 1 and x[j] == 0:
                viol += 1
        return np.array([f1, f2], dtype=float), int(viol)

    def constraint_graph(self):
        g = nx.Graph()
        g.add_nodes_from(range(self.n))
        for i, j in self.req_pairs:
            g.add_edge(i, j)
        return g

    def constraint_density(self):
        # number of distinct constraint edges
        seen = set()
        for i, j in self.req_pairs:
            seen.add((min(i, j), max(i, j)))
        denom = self.n * (self.n - 1) / 2.0
        return len(seen) / max(denom, 1.0)


# --------------------------------------------------------------------------- #
# FSP
# --------------------------------------------------------------------------- #
class FSP:
    """
    Multi-objective Feature Selection Problem (4 objectives).

    Decision: x_i in {0,1}, i=1..n_feat, indicating whether feature i is selected.
    Objectives (all minimization):
        f1(x) = -sum_i richness_i    * x_i
        f2(x) = -sum_i reliability_i * x_i
        f3(x) =  sum_i defects_i     * x_i
        f4(x) =  sum_i cost_i        * x_i
    Constraints:
        - mandatory : x_i = 1 for i in mandatory
        - require   : (i, j) means x_i <= x_j
        - exclude   : (i, j) means x_i + x_j <= 1
        - alt_groups: list of groups, exactly one feature in the group selected
    """

    def __init__(self, data):
        self.n = int(data['n_feat'])
        self.richness    = np.array(data['richness']).astype(float)
        self.reliability = np.array(data['reliability']).astype(float)
        self.defects     = np.array(data['defects']).astype(float)
        self.cost        = np.array(data['cost']).astype(float)
        self.mandatory = list(data['mandatory'])
        self.require   = [tuple(p) for p in data['require']]
        self.exclude   = [tuple(p) for p in data['exclude']]
        self.alt_groups = [list(g) for g in data['alt_groups']]
        self.n_vars = self.n
        self.n_obj  = 4

    def evaluate(self, x):
        x = np.asarray(x, dtype=int)
        f1 = -float(np.dot(self.richness,    x))
        f2 = -float(np.dot(self.reliability, x))
        f3 =  float(np.dot(self.defects,     x))
        f4 =  float(np.dot(self.cost,        x))
        viol = 0
        for i in self.mandatory:
            if x[i] == 0:
                viol += 1
        for i, j in self.require:
            if x[i] == 1 and x[j] == 0:
                viol += 1
        for i, j in self.exclude:
            if x[i] == 1 and x[j] == 1:
                viol += 1
        for grp in self.alt_groups:
            if int(np.sum(x[grp])) != 1:
                viol += 1
        return np.array([f1, f2, f3, f4], dtype=float), int(viol)

    def constraint_graph(self):
        g = nx.Graph()
        g.add_nodes_from(range(self.n))
        for i, j in self.require + self.exclude:
            g.add_edge(i, j)
        for grp in self.alt_groups:
            for a in range(len(grp)):
                for b in range(a + 1, len(grp)):
                    g.add_edge(grp[a], grp[b])
        return g

    def constraint_density(self):
        edges = set()
        for i, j in self.require + self.exclude:
            edges.add((min(i, j), max(i, j)))
        for grp in self.alt_groups:
            for a in range(len(grp)):
                for b in range(a + 1, len(grp)):
                    edges.add((min(grp[a], grp[b]), max(grp[a], grp[b])))
        denom = self.n * (self.n - 1) / 2.0
        return len(edges) / max(denom, 1.0)


def load_problem(name, data_dir='datasets'):
    """Load problem by short name (e.g. 'nrp_med', 'fsp_small')."""
    path = f'{data_dir}/{name}.json'
    data = load_json(path)
    if name.startswith('nrp'):
        return NRP(data)
    elif name.startswith('fsp'):
        return FSP(data)
    raise ValueError(f'unknown problem {name}')
