"""
Demo 01 — Topology Comparison: PRS across five canonical layouts.

Topologies covered
------------------
  chain(4)             — linear authority pipeline
  star(5)              — hub + 4 leaf specialists
  binary_tree(depth=2) — 3-layer hierarchy (root → 2 mid → 4 leaf)
  ring(5)              — cyclic pipeline
  layered(2, 3)        — 2-layer DAG (3 wide)

For each topology we:
  1. Run SimulationEngine with CONTEXT_POISONING from agent_0.
  2. Compute PRS via compute_prs().
  3. Report R0, mean infection rate, and PRS in a ranked summary table.
"""

from propagator import (
    TopologyFactory,
    SimulationEngine,
    AttackType,
    AttackConfig,
    compute_prs,
)

BETA   = 0.3
GAMMA  = 0.0   # worst-case: no natural recovery
ROUNDS = 20
TRIALS = 30


def run_topology(name: str, graph, seed_node: str) -> dict:
    engine = SimulationEngine(graph, beta=BETA, gamma=GAMMA, rounds=ROUNDS)
    r0     = engine.compute_r0()

    config = AttackConfig(
        attack_type=AttackType.CONTEXT_POISONING,
        injection_strength=1.0,
        seed_node_id=seed_node,
        trials=TRIALS,
    )
    result = engine.run_experiment(config)
    prs    = compute_prs(result, AttackType.CONTEXT_POISONING, topology_name=name)

    return {
        "topology":       name,
        "n_nodes":        graph.number_of_nodes(),
        "n_edges":        graph.number_of_edges(),
        "R0":             r0,
        "mean_infection": result.mean_final_rate,
        "prs":            prs.prs_value,
    }


def main():
    topologies = {
        "chain_4":        (TopologyFactory.chain(4),              "agent_0"),
        "star_5":         (TopologyFactory.star(5),               "agent_0"),
        "binary_tree_d2": (TopologyFactory.binary_tree(depth=2),  "agent_0"),
        "ring_5":         (TopologyFactory.ring(5),               "agent_0"),
        "layered_2x3":    (TopologyFactory.layered(2, 3),         "agent_0"),
    }

    rows = []
    for name, (g, seed) in topologies.items():
        print(f"  Running {name}...")
        rows.append(run_topology(name, g, seed))

    rows.sort(key=lambda r: r["prs"], reverse=True)

    header = f"\n{'Topology':<20} {'Nodes':>5} {'Edges':>5} {'R0':>7} {'Inf%':>7} {'PRS':>8}"
    print("\nTopology Comparison — CONTEXT_POISONING")
    print(header)
    print("-" * len(header.strip()))
    for r in rows:
        print(
            f"{r['topology']:<20} {r['n_nodes']:>5} {r['n_edges']:>5} "
            f"{r['R0']:>7.3f} {r['mean_infection']:>7.3f} {r['prs']:>8.4f}"
        )

    highest = rows[0]
    lowest  = rows[-1]
    spread  = highest["prs"] - lowest["prs"]
    print(f"\nMost vulnerable:  {highest['topology']}  (PRS={highest['prs']:.4f})")
    print(f"Least vulnerable: {lowest['topology']}  (PRS={lowest['prs']:.4f})")
    print(f"Topology spread:  {spread:.4f}  "
          f"({'topology matters' if spread > 0.10 else 'topology barely matters — check HST'})")


if __name__ == "__main__":
    main()
