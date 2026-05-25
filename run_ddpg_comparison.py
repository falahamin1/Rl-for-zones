"""
run_ddpg_comparison.py
----------------------
Compare three approaches on the selected benchmark instances:

  1. Symbolic RL   — zone-based tabular Q-learning  (existing)
  2. Region RL     — integer-clock tabular Q-learning (existing)
  3. DDPG          — continuous feature + neural actor-critic (new)

All three share the same episode budget and eval protocol so the
convergence curves are directly comparable.

Outputs
-------
  output/ddpg/<NAME>_comparison.png   — three-way convergence curves
  output/ddpg/comparison_summary.csv  — full results
  output/ddpg/comparison_summary.png  — compact summary table
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
from prob_jobshop.pta_env import PTADirectEnv
from prob_jobshop.ddpg_agent import DDPGAgent
from prob_jobshop.verification import verify_instance

INSTANCES = [
    "PJS_01", "PJS_02", "PJS_03",
    "PJS_05", "PJS_06", "PJS_07", "PJS_08", "PJS_09",
    "LJS_08", "LJS_09", "LJS_10",
]

EVAL_EPS      = 100
SEED          = 42
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "output", "ddpg")

EPSILON_START = 0.9
EPSILON_MIN   = 0.05


def get_instance(name):
    if name.startswith("LJS"):
        return get_large_instance_by_name(name)
    return get_instance_by_name(name)


def compute_train_episodes(instance) -> int:
    return min(30_000, max(5_000, 200 * instance.num_tasks()))


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
# Per-instance three-way plot
# ---------------------------------------------------------------------------

COLORS = {
    "symbolic": "steelblue",
    "region":   "darkorange",
    "ddpg":     "mediumseagreen",
}
BASELINE_COLORS = {
    "random": "#aaaaaa", "sept": "#4daf4a",
    "lept": "#ff7f00", "mwr": "#984ea3", "fifo": "#e41a1c",
}


def plot_comparison(inst_name, train_eps, sym_ckpts, reg_ckpts, ddpg_ckpts,
                    baselines, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    for bname, res in baselines.items():
        ax.axhline(res["mean"], color=BASELINE_COLORS[bname], linewidth=1,
                   linestyle=":", alpha=0.7, label=f"{bname} ({res['mean']:.1f})")

    if sym_ckpts:
        sx, sy = zip(*sym_ckpts)
        ax.plot(sx, sy, color=COLORS["symbolic"], linewidth=2.5, marker="o",
                markersize=5, label="Symbolic RL (zone)")
    if reg_ckpts:
        rx, ry = zip(*reg_ckpts)
        ax.plot(rx, ry, color=COLORS["region"], linewidth=2.5, marker="s",
                markersize=5, label="Region RL")
    if ddpg_ckpts:
        dx, dy = zip(*ddpg_ckpts)
        ax.plot(dx, dy, color=COLORS["ddpg"], linewidth=2.5, marker="^",
                markersize=5, label="DDPG (continuous)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean makespan (greedy policy, real sim)")
    ax.set_title(f"{inst_name} — Symbolic RL vs Region RL vs DDPG  ({train_eps} episodes)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _fmt(n) -> str:
    if isinstance(n, str):
        return n
    if n < 1_000_000:
        return f"{n:,}"
    return f"{n:.2e}"


def plot_summary_table(rows, save_path):
    headers = [
        "Instance", "Size\n(J×M)", "Tasks",
        "Best Baseline\n(mean ± std)",
        "Symbolic RL\n(mean ± std)",
        "Region RL\n(mean ± std)",
        "DDPG\n(mean ± std)",
        "DDPG params",
    ]
    cell_rows = []
    for r in rows:
        cell_rows.append([
            r["Instance"], r["size"], str(r["Tasks"]),
            f"{r['best_baseline_mean']:.2f} ± {r['best_baseline_std']:.2f}\n({r['best_baseline_name']})",
            f"{r['symbolic_rl_mean']:.2f} ± {r['symbolic_rl_std']:.2f}",
            f"{r['region_rl_mean']:.2f} ± {r['region_rl_std']:.2f}",
            f"{r['ddpg_mean']:.2f} ± {r['ddpg_std']:.2f}",
            r["ddpg_params"],
        ])

    n_rows = len(cell_rows)
    fig_h  = max(3.5, n_rows * 0.65 + 1.8)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=cell_rows, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.auto_set_column_width(col=list(range(len(headers))))

    for ci in range(len(headers)):
        tbl[0, ci].set_facecolor("#e8e8e8")
        tbl[0, ci].set_text_props(weight="bold")
    for ri in range(1, n_rows + 1):
        tbl[ri, 4].set_facecolor("#ddeeff")
        tbl[ri, 5].set_facecolor("#fff0dd")
        tbl[ri, 6].set_facecolor("#ddffd8")
        tbl[ri, 7].set_facecolor("#ddffd8")
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_height(0.072)

    ax.set_title(
        "Symbolic RL (zone) vs Region RL vs DDPG — Performance Comparison",
        fontsize=9, pad=10,
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
    print(f"Symbolic RL vs Region RL vs DDPG  |  {len(INSTANCES)} instances")
    print(f"Eval episodes: {EVAL_EPS}  |  ε: {EPSILON_START}→{EPSILON_MIN}")
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
        eval_int   = max(100, train_eps // 20)

        print(f"=== {inst_name}: {n_jobs}j × {n_machines}m, {n_tasks} tasks, "
              f"{train_eps} episodes ===")

        # ── Baselines ──────────────────────────────────────────────────
        baselines = run_baselines(instance)
        best_name = min(baselines, key=lambda k: baselines[k]["mean"])
        best_res  = baselines[best_name]

        # ── Symbolic RL ────────────────────────────────────────────────
        sym_env   = SymbolicJobShopEnv(instance, seed=SEED)
        zctx      = ZoneContext(instance)
        M         = instance.worst_case_makespan_upper_bound()
        sym_agent = SymbolicRLAgent(
            sym_env, zctx, M=M,
            epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN, seed=SEED,
        )
        t0 = time.time()
        sym_info = sym_agent.train(episodes=train_eps, eval_interval=eval_int,
                                   eval_episodes=30, eval_seed=SEED)
        sym_res = sym_agent.evaluate_real(n_episodes=EVAL_EPS, seed=SEED)
        print(f"  [Symbolic RL] {time.time()-t0:.1f}s  mean={sym_res['mean']:.2f}")

        # ── Region RL ──────────────────────────────────────────────────
        reg_env   = RegionJobShopEnv(instance, seed=SEED)
        reg_agent = RegionRLAgent(
            reg_env,
            epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN,
            lr=0.1, gamma=1.0, seed=SEED,
        )
        t0 = time.time()
        reg_info = reg_agent.train(episodes=train_eps, eval_interval=eval_int,
                                   eval_episodes=30, eval_seed=SEED)
        reg_res = reg_agent.evaluate_real(n_episodes=EVAL_EPS, seed=SEED)
        print(f"  [Region RL]   {time.time()-t0:.1f}s  mean={reg_res['mean']:.2f}")

        # ── DDPG (PTA-direct) ──────────────────────────────────────────
        ddpg_env   = PTADirectEnv(instance, seed=SEED)
        ddpg_agent = DDPGAgent(
            ddpg_env,
            lr_actor=1e-3, lr_critic=1e-3,
            gamma=1.0, tau=0.005,
            buffer_size=100_000, batch_size=128,
            sigma_start=0.3, sigma_min=0.02,
            seed=SEED,
        )
        t0 = time.time()
        ddpg_info = ddpg_agent.train(episodes=train_eps, eval_interval=eval_int,
                                     eval_episodes=30, eval_seed=SEED)
        ddpg_res = ddpg_agent.evaluate_real(n_episodes=EVAL_EPS, seed=SEED)
        print(f"  [DDPG]        {time.time()-t0:.1f}s  mean={ddpg_res['mean']:.2f}  "
              f"params={ddpg_agent.n_states()}\n")

        # ── Per-instance plot ──────────────────────────────────────────
        plot_comparison(
            inst_name, train_eps,
            sym_ckpts=sym_info.get("checkpoint_evals", []),
            reg_ckpts=reg_info.get("checkpoint_evals", []),
            ddpg_ckpts=ddpg_info.get("checkpoint_evals", []),
            baselines=baselines,
            save_path=os.path.join(OUTPUT_DIR, f"{inst_name}_comparison.png"),
        )

        rows.append({
            "Instance":           inst_name,
            "size":               f"{n_jobs}×{n_machines}",
            "Jobs":               n_jobs,
            "Machines":           n_machines,
            "Tasks":              n_tasks,
            "best_baseline_name": best_name,
            "best_baseline_mean": round(best_res["mean"], 2),
            "best_baseline_std":  round(best_res["std"],  2),
            "symbolic_rl_mean":   round(sym_res["mean"], 2),
            "symbolic_rl_std":    round(sym_res["std"],  2),
            "region_rl_mean":     round(reg_res["mean"], 2),
            "region_rl_std":      round(reg_res["std"],  2),
            "ddpg_mean":          round(ddpg_res["mean"], 2),
            "ddpg_std":           round(ddpg_res["std"],  2),
            "ddpg_params":        ddpg_agent.n_states(),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV: {csv_path}")

    plot_summary_table(rows, os.path.join(OUTPUT_DIR, "comparison_summary.png"))

    print("\nResults:")
    print(f"{'Instance':<10} {'Best BL':>9} {'Sym RL':>9} {'Reg RL':>9} {'DDPG':>9}")
    print("-" * 52)
    for r in rows:
        print(f"{r['Instance']:<10} {r['best_baseline_mean']:>9.2f} "
              f"{r['symbolic_rl_mean']:>9.2f} {r['region_rl_mean']:>9.2f} "
              f"{r['ddpg_mean']:>9.2f}")


if __name__ == "__main__":
    main()
