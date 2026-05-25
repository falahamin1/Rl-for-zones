"""
run_comparison_all.py
---------------------
Symbolic RL (zone-based) vs Region RL on all 20 benchmark instances:

  PJS_01 … PJS_10   (small-to-medium, 2–10 jobs)
  LJS_01 … LJS_10   (large, 6–14 jobs)

Episode budget: min(30 000, max(5 000, 200 × tasks)).
Eval every max(100, budget // 20) episodes (≈ 20 checkpoints).

Outputs
-------
  output/all/<NAME>_comparison.png   — per-instance convergence curves
  output/all/comparison_summary.csv  — full results table
  output/all/comparison_summary.png  — compact figure for paper
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
from prob_jobshop.benchmarks.all_large_instances import get_large_instance_by_name
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
from prob_jobshop.graph_sizes import count_region_graph, count_zone_graph

INSTANCES = [
    "PJS_01", "PJS_02", "PJS_03", "PJS_04", "PJS_05",
    "PJS_06", "PJS_07", "PJS_08", "PJS_09", "PJS_10",
    "LJS_01", "LJS_02", "LJS_03", "LJS_04", "LJS_05",
    "LJS_06", "LJS_07", "LJS_08", "LJS_09", "LJS_10",
]

EVAL_EPS      = 100
SEED          = 42
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "output", "all")

EPSILON_START = 0.9
EPSILON_MIN   = 0.05

BASELINE_COLORS = {
    "random": "#aaaaaa",
    "sept":   "#4daf4a",
    "lept":   "#ff7f00",
    "mwr":    "#984ea3",
    "fifo":   "#e41a1c",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_instance(name: str):
    if name.startswith("LJS"):
        return get_large_instance_by_name(name)
    return get_instance_by_name(name)


def compute_train_episodes(instance) -> int:
    return min(30_000, max(5_000, 200 * instance.num_tasks()))


def region_theoretical_bound(instance) -> int:
    bound = 1
    for j in instance.jobs:
        bound *= (max(t.distribution.max_duration() for t in j.tasks) + 2)
    return bound


def run_baselines(instance) -> dict:
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
        results[name] = sim.evaluate_strategy(fn, n_episodes=EVAL_EPS, desc=f"  {name}")
    return results


# ---------------------------------------------------------------------------
# Per-instance convergence plot
# ---------------------------------------------------------------------------

def plot_comparison(instance_name, train_eps,
                    sym_checkpoints, region_checkpoints,
                    baselines, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    for bname, res in baselines.items():
        ax.axhline(
            res["mean"],
            color=BASELINE_COLORS.get(bname, "gray"),
            linewidth=1, linestyle=":", alpha=0.7,
            label=f"{bname} ({res['mean']:.1f})",
        )

    if sym_checkpoints:
        sx, sy = zip(*sym_checkpoints)
        ax.plot(sx, sy, color="steelblue", linewidth=2.5, marker="o",
                markersize=5, label="Symbolic RL (zone)")

    if region_checkpoints:
        rx, ry = zip(*region_checkpoints)
        ax.plot(rx, ry, color="darkorange", linewidth=2.5, marker="s",
                markersize=5, label="Region RL (integer clock)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean makespan (greedy policy, real sim)")
    ax.set_title(
        f"{instance_name} — Symbolic RL vs Region RL  ({train_eps} training episodes)"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Summary table figure
# ---------------------------------------------------------------------------

def _fmt_bfs(n: int, truncated: bool) -> str:
    prefix = ">" if truncated else ""
    if n < 1_000_000:
        return f"{prefix}{n:,}"
    return f"{prefix}{n:.2e}"


def plot_summary_table(rows: list, save_path: str):
    headers = [
        "Instance", "Size\n(J×M)", "Tasks",
        "Best Baseline\n(mean ± std)",
        "Symbolic RL\n(mean ± std)", "Zone Graph\n(BFS states)",
        "Region RL\n(mean ± std)",
        "Region Graph\n(BFS states)",
        "Reg. Bound\n∏(cᵢ+2)",
    ]

    cell_rows = []
    for r in rows:
        cell_rows.append([
            r["Instance"],
            r["size"],
            str(r["Tasks"]),
            f"{r['best_baseline_mean']:.2f} ± {r['best_baseline_std']:.2f}\n"
            f"({r['best_baseline_name']})",
            f"{r['symbolic_rl_mean']:.2f} ± {r['symbolic_rl_std']:.2f}",
            _fmt_bfs(r["zone_graph_bfs"], r["zone_graph_truncated"]),
            f"{r['region_rl_mean']:.2f} ± {r['region_rl_std']:.2f}",
            _fmt_bfs(r["region_graph_bfs"], r["region_graph_truncated"]),
            f"{r['region_states_theory']:.2e}",
        ])

    n_rows = len(cell_rows)
    fig_h  = max(4.0, n_rows * 0.65 + 1.8)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_rows, colLabels=headers,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.auto_set_column_width(col=list(range(len(headers))))

    sym_cols = [4, 5]
    reg_cols = [6, 7, 8]
    for ci in range(len(headers)):
        tbl[0, ci].set_facecolor("#e8e8e8")
        tbl[0, ci].set_text_props(weight="bold")
    for ri in range(1, n_rows + 1):
        for ci in sym_cols:
            tbl[ri, ci].set_facecolor("#ddeeff")
        for ci in reg_cols:
            tbl[ri, ci].set_facecolor("#fff0dd")
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_height(0.058)

    ax.set_title(
        "Symbolic RL (zone-based) vs Region RL (integer-clock) — All 20 Benchmarks\n"
        "BFS states = reachable states by exhaustive BFS (> = capped at 200 000).  "
        "Reg. Bound = ∏ᵢ(cᵢ+2) where cᵢ = max task duration in job i.",
        fontsize=8, pad=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary table saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []

    print("=" * 70)
    print(f"Symbolic RL vs Region RL  |  {len(INSTANCES)} instances (all benchmarks)")
    print(f"Eval episodes: {EVAL_EPS}  |  ε: {EPSILON_START} → {EPSILON_MIN}")
    print("=" * 70 + "\n")

    for inst_name in INSTANCES:
        instance = get_instance(inst_name)

        errors = verify_instance(instance)
        if errors:
            print(f"[FAIL] {inst_name}: {errors}")
            sys.exit(1)

        n_jobs     = len(instance.jobs)
        n_machines = len(instance.machines)
        n_tasks    = instance.num_tasks()
        train_eps  = compute_train_episodes(instance)
        eval_interval = max(100, train_eps // 20)
        reg_theory = region_theoretical_bound(instance)

        print(f"=== {inst_name}: {n_jobs}j × {n_machines}m, {n_tasks} tasks, "
              f"{train_eps} episodes ===")

        # ── Baselines ──────────────────────────────────────────────────
        baselines = run_baselines(instance)
        best_name = min(baselines, key=lambda k: baselines[k]["mean"])
        best_res  = baselines[best_name]
        print(f"  Best baseline: {best_name}  mean={best_res['mean']:.2f}")

        # ── Symbolic RL ────────────────────────────────────────────────
        sym_env   = SymbolicJobShopEnv(instance, seed=SEED)
        zctx      = ZoneContext(instance)
        M         = instance.worst_case_makespan_upper_bound()
        sym_agent = SymbolicRLAgent(
            sym_env, zctx, M=M,
            epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN, seed=SEED,
        )
        t0 = time.time()
        sym_info = sym_agent.train(
            episodes=train_eps, eval_interval=eval_interval,
            eval_episodes=30, eval_seed=SEED,
        )
        sym_res = sym_agent.evaluate_real(n_episodes=EVAL_EPS, seed=SEED)
        print(f"  [Symbolic RL] {time.time()-t0:.1f}s  "
              f"mean={sym_res['mean']:.2f}  visited_states={sym_agent.n_states():,}")

        # ── Region RL ──────────────────────────────────────────────────
        reg_env   = RegionJobShopEnv(instance, seed=SEED)
        reg_agent = RegionRLAgent(
            reg_env,
            epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN,
            lr=0.1, gamma=1.0, seed=SEED,
        )
        t0 = time.time()
        reg_info = reg_agent.train(
            episodes=train_eps, eval_interval=eval_interval,
            eval_episodes=30, eval_seed=SEED,
        )
        reg_res = reg_agent.evaluate_real(n_episodes=EVAL_EPS, seed=SEED)
        print(f"  [Region RL]   {time.time()-t0:.1f}s  "
              f"mean={reg_res['mean']:.2f}  visited_states={reg_agent.n_states():,}")

        # ── BFS graph sizes ────────────────────────────────────────────
        t0 = time.time()
        reg_bfs, reg_trunc = count_region_graph(instance, max_states=200_000)
        print(f"  BFS region: {reg_bfs:,}{'+ (cap)' if reg_trunc else ''}  "
              f"({time.time()-t0:.1f}s)")

        t0 = time.time()
        zone_bfs, zone_trunc = count_zone_graph(instance, max_states=200_000)
        print(f"  BFS zone:   {zone_bfs:,}{'+ (cap)' if zone_trunc else ''}  "
              f"({time.time()-t0:.1f}s)\n")

        # ── Per-instance convergence plot ──────────────────────────────
        plot_comparison(
            inst_name, train_eps,
            sym_checkpoints=sym_info.get("checkpoint_evals", []),
            region_checkpoints=reg_info.get("checkpoint_evals", []),
            baselines=baselines,
            save_path=os.path.join(OUTPUT_DIR, f"{inst_name}_comparison.png"),
        )

        rows.append({
            "Instance":               inst_name,
            "size":                   f"{n_jobs}×{n_machines}",
            "Jobs":                   n_jobs,
            "Machines":               n_machines,
            "Tasks":                  n_tasks,
            "best_baseline_name":     best_name,
            "best_baseline_mean":     round(best_res["mean"], 2),
            "best_baseline_std":      round(best_res["std"],  2),
            "symbolic_rl_mean":       round(sym_res["mean"], 2),
            "symbolic_rl_std":        round(sym_res["std"],  2),
            "zone_graph_bfs":         zone_bfs,
            "zone_graph_truncated":   zone_trunc,
            "region_rl_mean":         round(reg_res["mean"], 2),
            "region_rl_std":          round(reg_res["std"],  2),
            "region_graph_bfs":       reg_bfs,
            "region_graph_truncated": reg_trunc,
            "region_states_theory":   reg_theory,
        })

    # ── Summary CSV ────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved: {csv_path}")

    # ── Summary table figure ───────────────────────────────────────────
    plot_summary_table(rows, save_path=os.path.join(OUTPUT_DIR, "comparison_summary.png"))

    # ── Console summary ────────────────────────────────────────────────
    print("\nResults summary:")
    print(f"{'Instance':<10} {'Size':<7} {'Best BL':>9} {'Sym RL':>9} "
          f"{'Zone BFS':>12} {'Reg RL':>9} {'Reg BFS':>12} {'Bound':>12}")
    print("-" * 90)
    for r in rows:
        print(
            f"{r['Instance']:<10} {r['size']:<7} "
            f"{r['best_baseline_mean']:>9.2f} "
            f"{r['symbolic_rl_mean']:>9.2f} "
            f"{_fmt_bfs(r['zone_graph_bfs'], r['zone_graph_truncated']):>12} "
            f"{r['region_rl_mean']:>9.2f} "
            f"{_fmt_bfs(r['region_graph_bfs'], r['region_graph_truncated']):>12} "
            f"{r['region_states_theory']:>12.2e}"
        )


if __name__ == "__main__":
    main()
