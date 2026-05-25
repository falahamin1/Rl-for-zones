"""
run_benchmark.py
----------------
Comprehensive benchmark comparing:
  1. Five hand-crafted baselines (random, SEPT, LEPT, MWR, FIFO)
  2. Symbolic RL agent trained with many episodes + epsilon decay
  3. Optimal symbolic policy via exhaustive zone-graph DP
  4. Theoretical expected-makespan lower bound (machine-load bound)

All evaluations use 500 real PTASimulator episodes so results are
directly comparable.

Outputs:
  output/benchmark_summary.csv   — full results table
  output/benchmark_summary.png   — visual table
  output/benchmark_convergence.png — RL learning curves

Usage:
  python run_benchmark.py               # full run (may take 10–15 min)
  python run_benchmark.py --fast        # quick run with fewer episodes
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prob_jobshop.benchmarks.all_instances import get_instance_by_name
from prob_jobshop.simulator import PTASimulator
from prob_jobshop.strategies import (
    random_strategy, sept_strategy, lept_strategy, mwr_strategy, fifo_strategy,
)
from prob_jobshop.symbolic_env import SymbolicJobShopEnv
from prob_jobshop.zone_ops import ZoneContext
from prob_jobshop.symbolic_rl import SymbolicRLAgent
from prob_jobshop.symbolic_optimal import OptimalSymbolicSolver
from prob_jobshop.verification import verify_instance

INSTANCES = [
    "PJS_01", "PJS_02", "PJS_03", "PJS_04", "PJS_05",
    "PJS_06", "PJS_07", "PJS_08", "PJS_09", "PJS_10",
]
SEED      = 42
EVAL_EPS  = 500      # real-simulator evaluation episodes (high for low variance)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Per-instance training budgets (full mode).  Scaled to the state-space sizes
# observed empirically.  The goal is ~100 visits per state on average.
TRAIN_EPS_FULL = {
    "PJS_01":   10_000,   # ~61 states   — converges with  ~10k
    "PJS_02":   15_000,   # ~178 states  — converges with  ~15k
    "PJS_03":  100_000,   # ~2k  states  — good coverage  with 100k
    "PJS_04":  250_000,   # ~10k states  — solid coverage with 250k
    "PJS_05":  250_000,   # 100k+ states — best-effort
    "PJS_06":  500_000,   # 5×5, large state space — best-effort
    "PJS_07":  500_000,   # 5×5, large state space — best-effort
    "PJS_08":  500_000,   # 6×6, very large — best-effort
    "PJS_09":  500_000,   # 8×6, very large — best-effort
    "PJS_10":  250_000,   # 10×10 stress test — best-effort
}
TRAIN_EPS_FAST = {
    "PJS_01":   5_000,
    "PJS_02":   5_000,
    "PJS_03":  20_000,
    "PJS_04":  50_000,
    "PJS_05": 100_000,
    "PJS_06": 100_000,
    "PJS_07": 100_000,
    "PJS_08": 100_000,
    "PJS_09": 100_000,
    "PJS_10":  50_000,
}

# BFS state cap for the exhaustive optimal solver.
# For instances where the optimal BFS would explode (PJS_08+), skip it entirely.
OPTIMAL_STATE_LIMIT = 20_000
SKIP_OPTIMAL = {"PJS_08", "PJS_09", "PJS_10"}  # state spaces too large to be useful


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_baselines(instance, n_episodes: int, seed: int) -> dict:
    sim = PTASimulator(instance, seed=seed)
    results = {}
    for name, factory in [
        ("random", lambda inst: random_strategy(inst, rng=sim.rng)),
        ("sept",   sept_strategy),
        ("lept",   lept_strategy),
        ("mwr",    mwr_strategy),
        ("fifo",   fifo_strategy),
    ]:
        fn = factory(instance)
        r = sim.evaluate_strategy(fn, n_episodes=n_episodes, desc=f"  {name}")
        results[name] = r
    return results


def build_rl_agent(instance, train_eps: int, seed: int) -> SymbolicRLAgent:
    """Train a symbolic RL agent with linear epsilon decay."""
    env  = SymbolicJobShopEnv(instance, seed=seed)
    zctx = ZoneContext(instance)
    M    = instance.worst_case_makespan_upper_bound()

    eps_start, eps_end = 0.40, 0.05
    agent = SymbolicRLAgent(env, zctx, M=M, epsilon=eps_start, seed=seed)

    # Train in blocks, decaying epsilon linearly
    block = max(1, train_eps // 20)
    for i in range(20):
        frac = i / 19
        agent.epsilon = eps_start + frac * (eps_end - eps_start)
        agent.train(episodes=block)

    agent.epsilon = 0.0   # greedy at eval time
    return agent


def build_optimal_solver(instance, seed: int, verbose: bool = False):
    """Zone-graph DP solver.  BFS is capped at OPTIMAL_STATE_LIMIT states."""
    env  = SymbolicJobShopEnv(instance, seed=seed)
    zctx = ZoneContext(instance)
    M    = instance.worst_case_makespan_upper_bound()
    solver = OptimalSymbolicSolver(env, zctx, M=M)
    solver.solve(verbose=verbose, state_limit=OPTIMAL_STATE_LIMIT)
    if solver.is_truncated():
        print(f"    [optimal] BFS truncated at {OPTIMAL_STATE_LIMIT} states "
              f"— policy is partial (best-effort, not provably optimal)")
    return solver


def plot_convergence(episode_costs_by_instance, save_path):
    n = len(episode_costs_by_instance)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (name, costs) in zip(axes, episode_costs_by_instance.items()):
        window = max(1, len(costs) // 40)
        smoothed = np.convolve(costs, np.ones(window) / window, mode="valid")
        ax.plot(costs, alpha=0.2, color="steelblue", linewidth=0.6)
        ax.plot(range(window - 1, len(costs)), smoothed,
                color="steelblue", linewidth=2)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Episode", fontsize=7)
        ax.set_ylabel("min-cost (zone)", fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=7)
    plt.suptitle("Symbolic RL learning curves (zone infimum per episode)", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(fast: bool = False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_eps_map = TRAIN_EPS_FAST if fast else TRAIN_EPS_FULL

    rows = []
    learning_curves = {}

    print(f"{'='*70}")
    mode_str = "FAST" if fast else "FULL"
    print(f"Symbolic RL Benchmark [{mode_str}]  |  eval episodes: {EVAL_EPS}")
    print(f"{'='*70}\n")

    for inst_name in INSTANCES:
        instance = get_instance_by_name(inst_name)

        errors = verify_instance(instance)
        if errors:
            print(f"[FAIL] {inst_name} verification: {errors}")
            sys.exit(1)

        print(f"{'─'*55}")
        print(f" {inst_name}: {instance.description[:55]}")
        print(f"{'─'*55}")

        lb = instance.expected_makespan_lower_bound()
        ub = instance.worst_case_makespan_upper_bound()
        print(f"  LB={lb:.2f}  WC-UB={ub}")

        # ── Baselines ──────────────────────────────────────────────
        t0 = time.time()
        baselines = run_baselines(instance, n_episodes=EVAL_EPS, seed=SEED)
        print(f"  Baselines done in {time.time()-t0:.1f}s")
        for bname, res in baselines.items():
            print(f"    {bname:8s}  mean={res['mean']:7.2f}  std={res['std']:5.2f}")

        # ── Optimal (exhaustive BFS + DP) ──────────────────────────
        if inst_name in SKIP_OPTIMAL:
            print("  Skipping optimal solver (state space too large).")
            solver  = None
            opt_res = None
        else:
            print("  Running optimal solver (exhaustive BFS + DP)…")
            t0 = time.time()
            solver = build_optimal_solver(instance, seed=SEED)
            solve_time = time.time() - t0
            print(f"  Optimal solved in {solve_time:.1f}s  |  states: {solver.n_states()}")
            opt_res = solver.evaluate_real(n_episodes=EVAL_EPS, seed=SEED)
            print(f"  optimal_dp  mean={opt_res['mean']:7.2f}  std={opt_res['std']:5.2f}")

        # ── Symbolic RL ────────────────────────────────────────────
        train_eps = train_eps_map[inst_name]
        print(f"  Training symbolic RL for {train_eps:,} episodes (ε: 0.40→0.05)…")
        t0 = time.time()
        agent = build_rl_agent(instance, train_eps=train_eps, seed=SEED)
        train_time = time.time() - t0
        print(f"  RL trained in {train_time:.1f}s  |  states: {agent.n_states()}")

        rl_res = agent.evaluate_real(n_episodes=EVAL_EPS, seed=SEED)
        print(f"  sym_rl      mean={rl_res['mean']:7.2f}  std={rl_res['std']:5.2f}")

        learning_curves[inst_name] = agent._episode_costs

        # ── Gap to best baseline ───────────────────────────────────
        best_bl = min(res["mean"] for res in baselines.values())
        gap_rl  = rl_res["mean"] - best_bl
        print(f"  Gap vs best baseline — RL: {gap_rl:+.2f}", end="")
        if opt_res is not None:
            gap_opt = opt_res["mean"] - best_bl
            print(f"  |  optimal: {gap_opt:+.2f}", end="")
        print()
        print(f"  Gap vs LB             — RL: {rl_res['mean']-lb:+.2f}", end="")
        if opt_res is not None:
            print(f"  |  optimal: {opt_res['mean']-lb:+.2f}", end="")
        print()

        # ── Build row ──────────────────────────────────────────────
        row = {"Instance": inst_name}
        for bname, res in baselines.items():
            row[f"{bname}_mean"] = round(res["mean"], 2)
            row[f"{bname}_std"]  = round(res["std"],  2)
        row["sym_rl_mean"]     = round(rl_res["mean"],  2)
        row["sym_rl_std"]      = round(rl_res["std"],   2)
        row["optimal_dp_mean"] = round(opt_res["mean"], 2) if opt_res else float("nan")
        row["optimal_dp_std"]  = round(opt_res["std"],  2) if opt_res else float("nan")
        row["exp_lb"]          = round(lb, 2)
        row["rl_states"]       = agent.n_states()
        row["opt_states"]      = solver.n_states() if solver else 0
        row["opt_truncated"]   = solver.is_truncated() if solver else True
        rows.append(row)
        print()

    # ── Save CSV ────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # ── Summary table figure ────────────────────────────────────────
    # Select key columns for the figure
    display_cols = [
        "Instance", "random_mean", "sept_mean", "lept_mean", "mwr_mean", "fifo_mean",
        "sym_rl_mean", "optimal_dp_mean", "exp_lb",
    ]
    col_labels = [
        "Instance", "random", "SEPT", "LEPT", "MWR", "FIFO",
        "Sym-RL", "Optimal-DP", "Exp-LB",
    ]
    disp_df = df[display_cols]

    fig, ax = plt.subplots(figsize=(14, max(3, len(rows) * 0.6 + 1.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=disp_df.values,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    # Highlight best policy per row in green
    best_col_indices = list(range(1, 7))  # baselines + sym_rl + optimal
    for row_idx, row_data in enumerate(disp_df.values):
        vals = [row_data[i] for i in best_col_indices]
        best_val = min(vals)
        for ci in best_col_indices:
            cell = tbl[row_idx + 1, ci]
            if float(row_data[ci]) == best_val:
                cell.set_facecolor("#c8f7c5")

    ax.set_title("Symbolic RL Benchmark — Mean Makespan (lower is better)",
                 fontsize=11, pad=12)
    plt.tight_layout()
    table_path = os.path.join(OUTPUT_DIR, "benchmark_summary.png")
    plt.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Table saved: {table_path}")

    # ── Learning curves ─────────────────────────────────────────────
    curves_path = os.path.join(OUTPUT_DIR, "benchmark_convergence.png")
    plot_convergence(learning_curves, curves_path)
    print(f"Convergence plot saved: {curves_path}")

    # ── Console summary ─────────────────────────────────────────────
    print("\nFull results (mean makespan):")
    summary_cols = ["Instance"] + [f"{b}_mean" for b in
                    ["random","sept","lept","mwr","fifo"]] + \
                   ["sym_rl_mean","optimal_dp_mean","exp_lb"]
    print(df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Fewer training episodes for a quick check")
    args = parser.parse_args()
    main(fast=args.fast)
