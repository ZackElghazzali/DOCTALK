"""
Demo 2: illustrate the Heterogeneous Screening Threshold intuition.

Low resistance variance => topology spread gap stays compressed.
Higher variance => topology begins to matter.
"""
from __future__ import annotations
import statistics as stats
import networkx as nx


def make_graph(kind: str, resistances: list[float]) -> nx.DiGraph:
    g = nx.DiGraph(name=kind)
    ids = [f"n{i}" for i in range(len(resistances))]
    for nid, r in zip(ids, resistances):
        g.add_node(nid, resistance=r)
    if kind == "chain":
        for i in range(len(ids) - 1):
            g.add_edge(ids[i], ids[i + 1], weight=0.9)
    elif kind == "star":
        for leaf in ids[1:]:
            g.add_edge(leaf, ids[0], weight=0.92)
            g.add_edge(ids[0], leaf, weight=0.95)
    elif kind == "dag":
        g.add_edges_from([
            (ids[0], ids[1], {"weight": 0.75}),
            (ids[0], ids[2], {"weight": 0.80}),
            (ids[1], ids[3], {"weight": 0.70}),
            (ids[2], ids[3], {"weight": 0.70}),
            (ids[3], ids[4], {"weight": 0.88}),
        ])
    return g


def approximate_r0(g: nx.DiGraph, beta: float = 0.27) -> float:
    nodes = list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    mat = [[0.0 for _ in range(n)] for _ in range(n)]
    for src, tgt, data in g.edges(data=True):
        p = (1 - g.nodes[tgt]["resistance"]) * data["weight"]
        mat[idx[tgt]][idx[src]] = beta * p
    # power iteration, because importing numpy for a toy script is optional, not sacred.
    v = [1.0 / n] * n
    for _ in range(40):
        new_v = [sum(mat[i][j] * v[j] for j in range(n)) for i in range(n)]
        norm = max(sum(abs(x) for x in new_v), 1e-9)
        v = [x / norm for x in new_v]
    w = [sum(mat[i][j] * v[j] for j in range(n)) for i in range(n)]
    num = sum(w[i] * v[i] for i in range(n))
    den = sum(v[i] * v[i] for i in range(n))
    return num / den if den else 0.0


def main():
    configs = {
        "low": [0.45, 0.46, 0.44, 0.45, 0.46],
        "medium": [0.25, 0.40, 0.45, 0.60, 0.70],
        "high": [0.15, 0.20, 0.50, 0.75, 0.85],
    }
    topologies = ["chain", "star", "dag"]
    epsilon_dist = 0.10

    for label, resistances in configs.items():
        sigma_sq = stats.pvariance(resistances)
        scores = {t: approximate_r0(make_graph(t, resistances)) for t in topologies}
        spread = max(scores.values()) - min(scores.values())
        exceeded = spread > epsilon_dist
        print(f"{label:6s} sigma^2(r)={sigma_sq:.4f} spread={spread:.4f} threshold_exceeded={exceeded}")
        for topo, r0 in scores.items():
            print(f"  {topo:6s} R0≈{r0:.3f}")
        print()

if __name__ == "__main__":
    main()
