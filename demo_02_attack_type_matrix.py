"""
Demo 02 — Attack-type matrix across three contrasting topologies.

Sweeps all five attack types against:
  star(5)              — centralised hub
  chain(5)             — linear, single path
  small_world(6, 2, .2) — Watts-Strogatz shortcuts everywhere

Outputs a PRS heat-table: topology rows × attack-type columns.
"""

from propagator import (
    TopologyFactory,
    SimulationEngine,
    AttackType,
    AttackConfig,
    compute_prs,
)

BETA   = 0.3
ROUNDS = 20
TRIALS = 30

ALL_ATTACKS = list(AttackType)


def prs_for(graph, attack: AttackType, seed: str, topo_name: str) -> float:
    engine = SimulationEngine(graph, beta=BETA, gamma=0.0, rounds=ROUNDS)
    result = engine.run_experiment(AttackConfig(
        attack_type=attack,
        injection_strength=1.0,
        seed_node_id=seed,
        trials=TRIALS,
    ))
    return compute_prs(result, attack, topology_name=topo_name).prs_value


def main():
    topologies = {
        "star_5":         (TopologyFactory.star(5),               "agent_0"),
        "chain_5":        (TopologyFactory.chain(5),              "agent_0"),
        "small_world_6":  (TopologyFactory.small_world(6, 2, 0.2),"agent_0"),
    }

    matrix = {}
    for topo_name, (g, seed) in topologies.items():
        matrix[topo_name] = {}
        for attack in ALL_ATTACKS:
            print(f"  {topo_name} × {attack.value}...")
            matrix[topo_name][attack] = prs_for(g, attack, seed, topo_name)

    col_w = 22
    attack_labels = [a.value for a in ALL_ATTACKS]

    print("\nPRS Matrix — topology × attack_type  (higher = more vulnerable)\n")
    header = f"{'':25}" + "".join(f"{lbl[:col_w]:>{col_w}}" for lbl in attack_labels)
    print(header)
    print("-" * len(header))
    for topo_name, attacks in matrix.items():
        row = f"{topo_name:<25}"
        for attack in ALL_ATTACKS:
            row += f"{attacks[attack]:>{col_w}.4f}"
        print(row)

    # worst (topo, attack) combo
    worst_prs, worst_pair = 0.0, ("", None)
    for topo, attacks in matrix.items():
        for attack, prs in attacks.items():
            if prs > worst_prs:
                worst_prs, worst_pair = prs, (topo, attack)

    print(f"\nWorst combo: {worst_pair[0]} under {worst_pair[1].value}  (PRS={worst_prs:.4f})")
    print("That is the cell your guardrail budget should protect first.")


if __name__ == "__main__":
    main()
