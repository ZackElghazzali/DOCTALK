"""
Demo 03 — HST: when does topology choice actually matter?

The Heterogeneous Screening Threshold (HST) states that topology effects
only emerge when inter-agent resistance variance σ²(r) exceeds a critical τ*.

Reproduces the sweep from PROPAGATOR Documentation §2 / Example 2:
  1. Build three hardening configs: homogeneous, mixed, diverse.
  2. For each, run star vs chain vs random_dag under ASSUMPTION_INJECTION.
  3. Measure topology spread D(σ²) = max R0 − min R0.
  4. Flag where spread exceeds epsilon_dist (= 0.10).

Hardening levels map to resistance as:
  V0 (hl=0): r ≈ 0.30   V1 (hl=1): r ≈ 0.475   V2 (hl=2): r ≈ 0.65
"""

from propagator import TopologyFactory, SimulationEngine

BETA         = 0.3
ROUNDS       = 20
EPSILON_DIST = 0.10

heterogeneity_configs = {
    "low":    [1, 1, 1, 1, 1],    # all V1 — σ²(r) ≈ 0
    "medium": [0, 1, 2, 0, 1],    # mixed
    "high":   [0, 0, 2, 2, 2],    # bimodal: soft entry, hardened internals
}


def build_graphs(hl_list: list[int]) -> dict:
    cfgs = [{"hardening_level": hl} for hl in hl_list]
    return {
        "star_5":  TopologyFactory.star(5,  agent_configs=cfgs),
        "chain_5": TopologyFactory.chain(5, agent_configs=cfgs),
        "dag_5":   TopologyFactory.random_dag(5, 0.6, agent_configs=cfgs),
    }


def compute_r0(g) -> float:
    return SimulationEngine(g, beta=BETA, gamma=0.0, rounds=ROUNDS).compute_r0()


def sigma_sq(hl_list: list[int]) -> float:
    resistances = [0.30 + 0.35 * (hl / 2) for hl in hl_list]
    mean_r = sum(resistances) / len(resistances)
    return sum((r - mean_r) ** 2 for r in resistances) / len(resistances)


def main():
    print("\nHST sweep — does topology matter at each heterogeneity level?\n")
    fmt = f"{'Level':<8} {'σ²(r)':>8}  {'R0 star':>9} {'R0 chain':>10} {'R0 dag':>8}  {'spread':>8}  HST"
    print(fmt)
    print("-" * len(fmt))

    for level, hl_list in heterogeneity_configs.items():
        graphs = build_graphs(hl_list)
        r0s    = {name: compute_r0(g) for name, g in graphs.items()}
        sq     = sigma_sq(hl_list)
        spread = max(r0s.values()) - min(r0s.values())
        flag   = "✓ EXCEEDED" if spread > EPSILON_DIST else "— below"

        print(
            f"{level:<8} {sq:>8.4f}  "
            f"{r0s['star_5']:>9.4f} {r0s['chain_5']:>10.4f} {r0s['dag_5']:>8.4f}  "
            f"{spread:>8.4f}  {flag}"
        )

    print()
    print("Interpretation:")
    print("  σ²(r) < τ* → topology spread compressed: topology choice is incidental.")
    print("  σ²(r) > τ* → structural amplifiers emerge: topology choice is causal.")


if __name__ == "__main__":
    main()
