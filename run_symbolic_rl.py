"""
run_symbolic_rl.py
------------------
Train a symbolic RL agent on PJS_01 through PJS_05 and compare its performance
against the five baseline strategies.

Outputs:
  output/symbolic_rl_summary.csv        — mean/std makespan table
  output/symbolic_rl_summary.png        — visual summary table
  output/<NAME>_symbolic_learning.png   — per-instance learning curve
"""
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
from prob_jobshop.verification import verify_instance

INSTANCES  = [
    "PJS_01", "PJS_02", "PJS_03", "PJS_04", "PJS_05",
    "PJS_06", "PJS_07", "PJS_08", "PJS_09", "PJS_10",
]
EVAL_EPS   = 100
SEED       = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

EPSILON_START = 0.9
EPSILON_MIN   = 0.05


def compute_train_episodes(instance) -> int:
    """Scale training episodes with instance size: 100 eps/task, capped at [1000, 8000]."""
    return min(8000, max(1000, 100 * instance.num_tasks()))


def run_baselines(instance, n_episodes=EVAL_EPS):
    sim = PTASimulator(instance, seed=SEED)
    results = {}
    for name, factory in [
        ("random", lambda inst: random_strategy(inst, rng=sim.rng)),
        ("sept",   sept_strategy),
        ("lept",   lept_strategy),
        ("mwr",    mwr_strategy),
        ("fifo",   fifo_strategy),
    ]:
        fn = factory(instance)
        r  = sim.evaluate_strategy(fn, n_episodes=n_episodes, desc=f"  {name}")
        results[name] = r
    return results


def plot_learning_curve(episode_costs, instance_name, save_path, checkpoint_evals=None):
    fig, ax = plt.subplots(figsize=(10, 4))

    window = max(1, len(episode_costs) // 20)
    smoothed = np.convolve(episode_costs, np.ones(window) / window, mode="valid")
    ax.plot(episode_costs, alpha=0.2, color="steelblue", linewidth=0.6, label="symbolic cost (training)")
    ax.plot(range(window - 1, len(episode_costs)), smoothed,
            color="steelblue", linewidth=2, label=f"smoothed symbolic (w={window})")

    if checkpoint_evals:
        ck_eps, ck_means = zip(*checkpoint_evals)
        ax.plot(ck_eps, ck_means, color="darkorange", linewidth=2,
                marker="o", markersize=5, label="greedy policy — real sim")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Makespan")
    ax.set_title(f"{instance_name} — Symbolic RL convergence")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []

    print(f"{'='*65}")
    print(f"Symbolic RL Benchmark  |  instances: {INSTANCES}")
    print(f"Eval episodes: {EVAL_EPS}  |  ε: {EPSILON_START} → {EPSILON_MIN}")
    print(f"{'='*65}\n")

    for inst_name in INSTANCES:
        instance = get_instance_by_name(inst_name)

        errors = verify_instance(instance)
        if errors:
            print(f"[FAIL] {inst_name} verification errors: {errors}")
            sys.exit(1)

        print(f"--- {inst_name}: {instance.description[:60]} ---")

        # --- Baselines ---
        t0 = time.time()
        baselines = run_baselines(instance, n_episodes=EVAL_EPS)
        print(f"  Baselines done in {time.time()-t0:.1f}s")
        for bname, res in baselines.items():
            print(f"    {bname:8s}  mean={res['mean']:7.2f}  std={res['std']:5.2f}")

        # --- Symbolic RL ---
        sym_env = SymbolicJobShopEnv(instance, seed=SEED)
        zctx    = ZoneContext(instance)
        M       = instance.worst_case_makespan_upper_bound()
        train_eps = compute_train_episodes(instance)
        agent   = SymbolicRLAgent(
            sym_env, zctx, M=M,
            epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN, seed=SEED,
        )

        eval_interval = max(1, train_eps // 10)
        print(f"  Training symbolic RL for {train_eps} episodes (ε {EPSILON_START}→{EPSILON_MIN},"
              f" greedy eval every {eval_interval} eps) ...")
        t0 = time.time()
        train_info = agent.train(
            episodes=train_eps,
            eval_interval=eval_interval,
            eval_episodes=20,
            eval_seed=SEED,
        )
        train_time = time.time() - t0
        print(f"  Training done in {train_time:.1f}s  |  zone states explored: {agent.n_states()}")

        # Real-episode evaluation using Q-table policy (comparable to baselines)
        real_res = agent.evaluate_real(n_episodes=EVAL_EPS, seed=SEED)
        print(f"  symbolic_rl   mean={real_res['mean']:7.2f}  std={real_res['std']:5.2f}  (real episodes)")

        # Symbolic lower-bound evaluation (zone min_cost; informational only)
        sym_res = agent.evaluate(n_episodes=min(20, EVAL_EPS))
        print(f"  sym zone LB   mean={sym_res['mean']:7.2f}  (zone infimum — theoretical lower bound)")

        # Learning curve (symbolic min_cost per episode)
        plot_learning_curve(
            train_info["episode_costs"],
            inst_name,
            save_path=os.path.join(OUTPUT_DIR, f"{inst_name}_symbolic_learning.png"),
            checkpoint_evals=train_info.get("checkpoint_evals"),
        )

        # LB / UB
        lb = instance.expected_makespan_lower_bound()
        ub = instance.worst_case_makespan_upper_bound()
        print(f"  LB={lb:.1f}  UB={ub}\n")

        # Build summary row (symbolic_rl_mean uses real-episode evaluation)
        row = {"Instance": inst_name}
        for bname, res in baselines.items():
            row[f"{bname}_mean"] = round(res["mean"], 2)
            row[f"{bname}_std"]  = round(res["std"],  2)
        row["symbolic_rl_mean"]    = round(real_res["mean"], 2)
        row["symbolic_rl_std"]     = round(real_res["std"],  2)
        row["sym_zone_lb"]         = round(sym_res["mean"], 2)
        row["zone_states"]         = agent.n_states()
        rows.append(row)

    # CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "symbolic_rl_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved: {csv_path}")

    # Summary table figure
    fig, ax = plt.subplots(figsize=(16, max(3, len(rows) * 0.5 + 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    ax.set_title("Symbolic RL vs Baseline Strategy Comparison", fontsize=11, pad=10)
    plt.tight_layout()
    table_path = os.path.join(OUTPUT_DIR, "symbolic_rl_summary.png")
    plt.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary table saved: {table_path}")

    print("\nFull results:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
