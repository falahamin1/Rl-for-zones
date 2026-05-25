"""
run_comparison.py
-----------------
Train both the Zone-based (Symbolic) RL agent and the Region-based RL agent on
all 10 PJS benchmarks and compare them head-to-head.

For each instance this script produces:
  output/<NAME>_comparison.png   — greedy-policy convergence curves for both
                                   agents overlaid on the same axes, with
                                   baseline means drawn as dashed reference lines

Across all instances:
  output/comparison_summary.csv  — baselines + both agents (mean ± std makespan)
                                   plus state-space sizes
  output/comparison_summary.png  — visual summary table

The per-instance training budget, epsilon schedule, and eval checkpoints are
identical for both agents so the learning curves are directly comparable.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prob_jobshop.benchmarks.all_instances import get_instance_by_name, ALL_INSTANCE_NAMES
from prob_jobshop.simulator import PTASimulator
from prob_jobshop.strategies import (
    random_strategy, sept_strategy, lept_strategy, mwr_strategy, fifo_strategy,
)
from prob_jobshop.symbolic_env import SymbolicJobShopEnv
from prob_jobshop.zone_ops import ZoneContext
from prob_jobshop.symbolic_rl import SymbolicRLAgent
from prob_jobshop.region_env import RegionJobShopEnv
from prob_jobshop.region_rl import RegionRLAgent
from prob_jobshop.verification import verify_instance

INSTANCES     = ALL_INSTANCE_NAMES
EVAL_EPS      = 100
SEED          = 42
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "output")

EPSILON_START = 0.9
EPSILON_MIN   = 0.05

# Baseline colors for the comparison plot reference lines
BASELINE_COLORS = {
    "random": "#aaaaaa",
    "sept":   "#4daf4a",
    "lept":   "#ff7f00",
    "mwr":    "#984ea3",
    "fifo":   "#e41a1c",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def compute_train_episodes(instance) -> int:
    """100 episodes per task, capped in [1 000, 8 000]."""
    return min(8000, max(1000, 100 * instance.num_tasks()))


def run_baselines(instance, n_episodes: int = EVAL_EPS) -> dict:
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
        results[name] = sim.evaluate_strategy(fn, n_episodes=n_episodes, desc=f"  {name}")
    return results


# ---------------------------------------------------------------------------
# Per-instance comparison plot
# ---------------------------------------------------------------------------

def plot_comparison(
    instance_name: str,
    train_eps: int,
    sym_checkpoints,      # list of (episode, mean_makespan)
    region_checkpoints,   # list of (episode, mean_makespan)
    baselines: dict,
    save_path: str,
):
    """Overlay both agents' greedy-eval curves + baseline reference lines."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # --- Baseline reference lines ---
    for bname, res in baselines.items():
        ax.axhline(
            res["mean"],
            color=BASELINE_COLORS.get(bname, "gray"),
            linewidth=1,
            linestyle=":",
            alpha=0.7,
            label=f"{bname} ({res['mean']:.1f})",
        )

    # --- Symbolic RL checkpoints ---
    if sym_checkpoints:
        sx, sy = zip(*sym_checkpoints)
        ax.plot(sx, sy, color="steelblue", linewidth=2.5, marker="o",
                markersize=6, label="Symbolic RL (zone)")

    # --- Region RL checkpoints ---
    if region_checkpoints:
        rx, ry = zip(*region_checkpoints)
        ax.plot(rx, ry, color="darkorange", linewidth=2.5, marker="s",
                markersize=6, label="Region RL (integer clock)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean makespan (greedy policy, real sim)")
    ax.set_title(f"{instance_name} — Symbolic RL vs Region RL  ({train_eps} training episodes)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []

    print(f"{'='*70}")
    print(f"Symbolic RL vs Region RL  |  {len(INSTANCES)} instances")
    print(f"Eval episodes: {EVAL_EPS}  |  ε: {EPSILON_START} → {EPSILON_MIN}")
    print(f"{'='*70}\n")

    for inst_name in INSTANCES:
        instance = get_instance_by_name(inst_name)

        errors = verify_instance(instance)
        if errors:
            print(f"[FAIL] {inst_name}: {errors}")
            sys.exit(1)

        train_eps     = compute_train_episodes(instance)
        eval_interval = max(1, train_eps // 10)

        print(f"=== {inst_name}: {instance.description[:55]} ===")
        print(f"    tasks={instance.num_tasks()}  train_eps={train_eps}  "
              f"eval_interval={eval_interval}")

        # ── Baselines ──────────────────────────────────────────────────
        t0 = time.time()
        baselines = run_baselines(instance)
        print(f"  Baselines ({time.time()-t0:.1f}s):")
        for bname, res in baselines.items():
            print(f"    {bname:8s}  mean={res['mean']:7.2f}  std={res['std']:5.2f}")

        # ── Symbolic RL ────────────────────────────────────────────────
        sym_env  = SymbolicJobShopEnv(instance, seed=SEED)
        zctx     = ZoneContext(instance)
        M        = instance.worst_case_makespan_upper_bound()
        sym_agent = SymbolicRLAgent(
            sym_env, zctx, M=M,
            epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN, seed=SEED,
        )
        print(f"  [Symbolic RL] training {train_eps} episodes ...")
        t0 = time.time()
        sym_info = sym_agent.train(
            episodes=train_eps,
            eval_interval=eval_interval,
            eval_episodes=20,
            eval_seed=SEED,
        )
        print(f"  [Symbolic RL] done in {time.time()-t0:.1f}s  "
              f"zone-states={sym_agent.n_states():,}")
        sym_res = sym_agent.evaluate_real(n_episodes=EVAL_EPS, seed=SEED)
        print(f"  [Symbolic RL] mean={sym_res['mean']:7.2f}  std={sym_res['std']:5.2f}")

        # ── Region RL ──────────────────────────────────────────────────
        reg_env   = RegionJobShopEnv(instance, seed=SEED)
        reg_agent = RegionRLAgent(
            reg_env,
            epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN,
            lr=0.1, gamma=1.0, seed=SEED,
        )
        print(f"  [Region RL]   training {train_eps} episodes ...")
        t0 = time.time()
        reg_info = reg_agent.train(
            episodes=train_eps,
            eval_interval=eval_interval,
            eval_episodes=20,
            eval_seed=SEED,
        )
        print(f"  [Region RL]   done in {time.time()-t0:.1f}s  "
              f"q-states={reg_agent.n_states():,}")
        reg_res = reg_agent.evaluate_real(n_episodes=EVAL_EPS, seed=SEED)
        print(f"  [Region RL]   mean={reg_res['mean']:7.2f}  std={reg_res['std']:5.2f}")

        # Theoretical clock-space upper bound
        clock_space_ub = 1
        for j in instance.jobs:
            clock_space_ub *= (max(t.distribution.max_duration() for t in j.tasks) + 2)

        print(f"  clock-space UB: {clock_space_ub:,}\n")

        # ── Per-instance comparison plot ───────────────────────────────
        plot_comparison(
            inst_name,
            train_eps,
            sym_checkpoints=sym_info.get("checkpoint_evals", []),
            region_checkpoints=reg_info.get("checkpoint_evals", []),
            baselines=baselines,
            save_path=os.path.join(OUTPUT_DIR, f"{inst_name}_comparison.png"),
        )

        # ── CSV row ────────────────────────────────────────────────────
        row = {"Instance": inst_name, "Tasks": instance.num_tasks()}
        for bname, res in baselines.items():
            row[f"{bname}_mean"] = round(res["mean"], 2)
            row[f"{bname}_std"]  = round(res["std"],  2)
        row["symbolic_rl_mean"]  = round(sym_res["mean"], 2)
        row["symbolic_rl_std"]   = round(sym_res["std"],  2)
        row["symbolic_states"]   = sym_agent.n_states()
        row["region_rl_mean"]    = round(reg_res["mean"], 2)
        row["region_rl_std"]     = round(reg_res["std"],  2)
        row["region_states"]     = reg_agent.n_states()
        row["clock_space_ub"]    = clock_space_ub
        rows.append(row)

    # ── Summary CSV ────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved: {csv_path}")

    # ── Summary table figure ───────────────────────────────────────────
    # Highlight columns for the two agents and the best baseline
    fig, ax = plt.subplots(figsize=(20, max(3, len(rows) * 0.6 + 1.5)))
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

    # Colour the header cells for the two agent columns
    col_names = list(df.columns)
    for col_label, colour in [
        ("symbolic_rl_mean", "#cce5ff"),
        ("symbolic_states",  "#cce5ff"),
        ("region_rl_mean",   "#ffe5cc"),
        ("region_states",    "#ffe5cc"),
        ("clock_space_ub",   "#ffe5cc"),
    ]:
        if col_label in col_names:
            ci = col_names.index(col_label)
            tbl[0, ci].set_facecolor(colour)

    ax.set_title(
        "Symbolic RL (zone) vs Region RL (integer clock) — Performance & State-Space Comparison",
        fontsize=10, pad=12,
    )
    plt.tight_layout()
    table_path = os.path.join(OUTPUT_DIR, "comparison_summary.png")
    plt.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary table saved: {table_path}")

    print("\nFull results:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
