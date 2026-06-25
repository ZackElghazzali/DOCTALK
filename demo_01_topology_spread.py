"""
Demo 1: compare adversarial spread across small MAS topologies.

This is a lightweight teaching script inspired by PROPAGATOR's SIR-on-graph model.
It does not depend on the actual library, so teammates can run it immediately.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import networkx as nx

@dataclass
class Agent:
    node_id: str
    resistance: float
    importance: float = 1.0

ATTACK_SEVERITY = {
    "assumption_injection": 1.0,
    "context_poisoning": 0.9,
    "direct_override": 0.7,
}


def build_chain() -> nx.DiGraph:
    g = nx.DiGraph(name="chain")
    agents = [
        Agent("entry", 0.25),
        Agent("planner", 0.40),
        Agent("worker", 0.45),
        Agent("egress", 0.55, importance=4.0),
    ]
    for a in agents:
        g.add_node(a.node_id, agent=a, infected=0.0)
    g.add_edges_from([
        ("entry", "planner", {"weight": 0.95}),
        ("planner", "worker", {"weight": 0.90}),
        ("worker", "egress", {"weight": 0.90}),
    ])
    return g


def build_star() -> nx.DiGraph:
    g = nx.DiGraph(name="star")
    agents = [
        Agent("leaf_seed", 0.30),
        Agent("hub", 0.20),
        Agent("leaf_a", 0.55),
        Agent("leaf_b", 0.50),
        Agent("egress", 0.60, importance=4.0),
    ]
    for a in agents:
        g.add_node(a.node_id, agent=a, infected=0.0)
    edges = [
        ("leaf_seed", "hub", {"weight": 0.95}),
        ("hub", "leaf_a", {"weight": 0.95}),
        ("hub", "leaf_b", {"weight": 0.95}),
        ("hub", "egress", {"weight": 0.92}),
    ]
    g.add_edges_from(edges)
    return g


def build_layered_dag() -> nx.DiGraph:
    g = nx.DiGraph(name="layered_dag")
    agents = [
        Agent("entry", 0.25),
        Agent("retriever", 0.55),
        Agent("validator", 0.60),
        Agent("executor", 0.35, importance=3.0),
        Agent("egress", 0.55, importance=4.0),
    ]
    for a in agents:
        g.add_node(a.node_id, agent=a, infected=0.0)
    g.add_edges_from([
        ("entry", "retriever", {"weight": 0.80}),
        ("entry", "validator", {"weight": 0.75}),
        ("retriever", "executor", {"weight": 0.72}),
        ("validator", "executor", {"weight": 0.68}),
        ("executor", "egress", {"weight": 0.88}),
    ])
    return g


def transmission_prob(g: nx.DiGraph, src: str, tgt: str, semantic_similarity: float = 0.95) -> float:
    r_tgt = g.nodes[tgt]["agent"].resistance
    w = g.edges[src, tgt]["weight"]
    return max(0.0, min(1.0, (1.0 - r_tgt) * semantic_similarity * w))


def run_sir(g: nx.DiGraph, seed: str, attack: str = "context_poisoning", beta: float = 0.3, gamma: float = 0.08, rounds: int = 12):
    for n in g.nodes:
        g.nodes[n]["infected"] = 0.0
    g.nodes[seed]["infected"] = 1.0
    beta_eff = beta * ATTACK_SEVERITY[attack]
    history = []

    for _ in range(rounds):
        snapshot = {n: g.nodes[n]["infected"] for n in g.nodes}
        history.append(snapshot)
        next_state = snapshot.copy()
        for node in g.nodes:
            incoming = 0.0
            for pred in g.predecessors(node):
                p = transmission_prob(g, pred, node)
                incoming += p * snapshot[pred]
            current = snapshot[node]
            updated = current + beta_eff * incoming * (1 - current) - gamma * current
            next_state[node] = max(0.0, min(1.0, updated))
        for node, val in next_state.items():
            g.nodes[node]["infected"] = val

    totals = [sum(step.values()) / g.number_of_nodes() for step in history]
    peak_velocity = max(b - a for a, b in zip(totals, totals[1:])) if len(totals) > 1 else 0.0
    mean_final = totals[-1]
    prs = mean_final * ATTACK_SEVERITY[attack] * peak_velocity
    return history, totals, prs


def main():
    topologies = {
        "chain": (build_chain(), "entry"),
        "star": (build_star(), "leaf_seed"),
        "layered_dag": (build_layered_dag(), "entry"),
    }
    print("Topology comparison demo\n")
    for name, (graph, seed) in topologies.items():
        _, totals, prs = run_sir(graph, seed)
        print(f"{name:12s} final_mean_infection={totals[-1]:.3f} peak={max(totals):.3f} prs≈{prs:.4f}")
        print("  curve:", " ".join(f"{x:.2f}" for x in totals))

if __name__ == "__main__":
    main()
