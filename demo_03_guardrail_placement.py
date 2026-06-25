"""
Demo 3: compare simple edge guardrail placement strategies on a tiny rewired MAS.

Goal: protect the path from entry point to code execution / egress under a small budget.
"""
from __future__ import annotations
from dataclasses import dataclass
import networkx as nx

@dataclass
class PlacementResult:
    name: str
    guarded_edges: list[tuple[str, str]]
    residual_risk: float


def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    importance = {
        "advice": 1,
        "orchestrator": 1,
        "diagnosis": 1,
        "stats": 3,
        "db": 2,
        "notifier": 4,
    }
    for node, mu in importance.items():
        g.add_node(node, importance=mu)
    g.add_edges_from([
        ("advice", "orchestrator", {"risk": 0.85}),
        ("orchestrator", "diagnosis", {"risk": 0.30}),
        ("orchestrator", "db", {"risk": 0.42}),
        ("orchestrator", "stats", {"risk": 0.80}),
        ("stats", "notifier", {"risk": 0.90}),
        ("db", "notifier", {"risk": 0.35}),
    ])
    return g


def path_risk(g: nx.DiGraph, guarded: set[tuple[str, str]], miss_rate: float = 0.20) -> float:
    total = 0.0
    for path in nx.all_simple_paths(g, "advice", "notifier"):
        edge_risk = 1.0
        for u, v in zip(path, path[1:]):
            r = g.edges[u, v]["risk"]
            if (u, v) in guarded:
                r *= miss_rate
            edge_risk *= r
        sink_importance = g.nodes[path[-1]]["importance"]
        total += sink_importance * edge_risk
    return total


def degree_centrality_strategy(g: nx.DiGraph, budget: int) -> PlacementResult:
    ranked = sorted(g.edges, key=lambda e: g.in_degree(e[1]), reverse=True)
    chosen = ranked[:budget]
    return PlacementResult("degree_centrality", chosen, path_risk(g, set(chosen)))


def dangerous_path_strategy(g: nx.DiGraph, budget: int) -> PlacementResult:
    ranked = sorted(g.edges, key=lambda e: g.edges[e]["risk"] * g.nodes[e[1]]["importance"], reverse=True)
    chosen = ranked[:budget]
    return PlacementResult("dangerous_path", chosen, path_risk(g, set(chosen)))


def main():
    g = build_graph()
    budget = 2
    baseline = path_risk(g, set())
    results = [
        PlacementResult("guard_nothing", [], baseline),
        degree_centrality_strategy(g, budget),
        dangerous_path_strategy(g, budget),
    ]
    print(f"Baseline blast radius proxy: {baseline:.4f}\n")
    for res in results:
        reduction = 1 - (res.residual_risk / baseline)
        print(f"{res.name:18s} guards={res.guarded_edges} residual={res.residual_risk:.4f} reduction={reduction:.1%}")

if __name__ == "__main__":
    main()
