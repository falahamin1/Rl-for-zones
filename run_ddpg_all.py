"""
run_ddpg_all.py
---------------
Run DDPG (PTADirectEnv) on all 20 benchmark instances (PJS_01-10 + LJS_01-10)
and compare against the five heuristic baselines.

Environment  : PTADirectEnv — raw PTA state, (J+1)-dim continuous action
               (delta timing + per-job priority scores)
Agent        : DDPGAgent — MLP actor-critic with OU noise, replay buffer,
               soft target updates

Episode budget (shared with tabular agents for fair comparison):
  T = min(30_000, max(5_000, 200 × |tasks|))

Outputs
-------
  output/ddpg_all/<NAME>_learning_curve.png   — makespan vs episode curve
  output/ddpg_all/results_summary.csv         — full numeric results
  output/ddpg_all/results_summary.png         — summary table figure
"""
from __future__ import annotations

import os
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prob_jobshop.benchmarks.all_instances import (
    ALL_INSTANCE_NAMES, get_instance_by_name,
)
from prob_jobshop.benchmarks.all_large_instances import (
    ALL_LARGE_INSTANCE_NAMES, get_large_instance_by_name,
)
from prob_jobshop.simulator import PTASimulator
from prob_jobshop.strategies import (
    random_strategy, sept_strategy, lept_strategy, mwr_strategy, fifo_strategy,
)
from prob_jobshop.pta_env import PTADirectEnv
from prob_jobshop.ddpg_agent import DDPGAgent
from prob_jobshop.verification import verify_instance

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALL_NAMES: List[str] = ALL_INSTANCE_NAMES + ALL_LARGE_INSTANCE_NAMES   # 20 total

EVAL_EPS   = 100
SEED       = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "ddpg_all")

SIGMA_START = 0.3
SIGMA_MIN   = 0.02

DDPG_KWARGS = dict(
    lr_actor    = 1e-3,
    lr_critic   = 1e-3,
    gamma       = 1.0,
    tau         = 0.005,
    buffer_size = 100_000,
    batch_size  = 128,
    sigma_start = SIGMA_START,
    sigma_min   = SIGMA_MIN,
)

BASELINE_FACTORIES = [
    ("random", lambda inst, sim: random_strategy(inst, rng=sim.rng)),
    ("sept",   lambda inst, sim: sept_strategy(inst)),
    ("lept",   lambda inst, sim: lept_strategy(inst)),
    ("mwr",    lambda inst, sim: mwr_strategy(inst)),
    ("fifo",   lambda inst, sim: fifo_strategy(inst)),
]

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


def run_baselines(instance) -> dict:
    sim = PTASimulator(instance, seed=SEED)
    results = {}
    for name, factory in BASELINE_FACTORIES:
        fn = factory(instance, sim)
        results[name] = sim.evaluate_strategy(fn, n_episodes=EVAL_EPS, desc=f"  {name}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_learning_curve(
    inst_name: str,
    train_eps: int,
    checkpoints: List[Tuple[int, float]],
    baselines: dict,
    ddpg_final_mean: float,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    # Baseline horizontals
    for bname, res in baselines.items():
        ax.axhline(
            res["mean"],
            color=BASELINE_COLORS[bname],
            linewidth=1.2,
            linestyle=":",
            alpha=0.75,
            label=f"{bname} ({res['mean']:.1f})",
        )

    # DDPG learning curve
    if checkpoints:
        xs, ys = zip(*checkpoints)
        ax.plot(xs, ys, color="mediumseagreen", linewidth=2.5,
                marker="^", markersize=5, label="DDPG (PTADirectEnv)")

    # Final evaluation marker
    ax.axhline(ddpg_final_mean, color="mediumseagreen", linewidth=1,
               linestyle="--", alpha=0.6,
               label=f"DDPG final ({ddpg_final_mean:.1f})")

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean makespan (greedy, 30-ep eval)")
    ax.set_title(f"{inst_name}  —  DDPG vs heuristic baselines  ({train_eps} episodes)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    curve  → {save_path}")


def plot_summary_table(rows: list, save_path: str) -> None:
    headers = [
        "Instance", "Size\n(J×M)", "Tasks",
        "Best baseline\n(mean ± std, method)",
        "DDPG\n(mean ± std)",
        "Δ vs best\nbaseline (%)",
        "Params",
        "Train eps",
        "Wall time (s)",
    ]
    cell_rows = []
    for r in rows:
        delta_pct = 100.0 * (r["ddpg_mean"] - r["best_mean"]) / max(r["best_mean"], 1e-9)
        cell_rows.append([
            r["instance"],
            r["size"],
            str(r["tasks"]),
            f"{r['best_mean']:.2f} ± {r['best_std']:.2f}\n({r['best_name']})",
            f"{r['ddpg_mean']:.2f} ± {r['ddpg_std']:.2f}",
            f"{delta_pct:+.1f}%",
            r["params"],
            str(r["train_eps"]),
            f"{r['wall_s']:.0f}",
        ])

    fig_h = max(4.0, len(cell_rows) * 0.7 + 1.8)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=cell_rows, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.auto_set_column_width(col=list(range(len(headers))))

    for ci in range(len(headers)):
        tbl[0, ci].set_facecolor("#d8e8f8")
        tbl[0, ci].set_text_props(weight="bold")

    for ri in range(1, len(cell_rows) + 1):
        # Colour DDPG column green if within 5% of best baseline, else light red
        delta_pct = float(cell_rows[ri - 1][5].replace("%", "").replace("+", ""))
        colour = "#d8f8d8" if delta_pct <= 5.0 else "#f8d8d8"
        tbl[ri, 4].set_facecolor(colour)
        tbl[ri, 5].set_facecolor(colour)
        # Zebra stripe
        if ri % 2 == 0:
            for ci in [0, 1, 2, 3, 6, 7, 8]:
                tbl[ri, ci].set_facecolor("#f5f5f5")

    for cell in tbl.get_celld().values():
        cell.set_height(0.075)

    ax.set_title("DDPG (PTADirectEnv) vs Heuristic Baselines — All 20 Instances",
                 fontsize=10, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary table → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"DDPG (PTADirectEnv) on all {len(ALL_NAMES)} benchmark instances")
    print(f"Eval: {EVAL_EPS} episodes | seed: {SEED}")
    print(f"OU noise: σ {SIGMA_START} → {SIGMA_MIN}")
    print("=" * 70 + "\n")

    rows = []

    for inst_name in ALL_NAMES:
        instance = get_instance(inst_name)

        # Sanity-check the instance
        errors = verify_instance(instance)
        if errors:
            print(f"[SKIP] {inst_name}: verification errors: {errors}")
            continue

        n_jobs     = len(instance.jobs)
        n_machines = len(instance.machines)
        n_tasks    = instance.num_tasks()
        train_eps  = compute_train_episodes(instance)
        eval_int   = max(100, train_eps // 20)   # ~20 checkpoints

        print(f"─── {inst_name}  ({n_jobs}j × {n_machines}m, {n_tasks} tasks, "
              f"{train_eps} train eps) ───")

        # ── Baselines ──────────────────────────────────────────────────────
        print("  baselines ...")
        baselines = run_baselines(instance)
        best_name = min(baselines, key=lambda k: baselines[k]["mean"])
        best_res  = baselines[best_name]
        print(f"  best baseline: {best_name}  mean={best_res['mean']:.2f}")

        # ── DDPG ───────────────────────────────────────────────────────────
        print(f"  DDPG training ({train_eps} episodes, eval every {eval_int}) ...")
        env   = PTADirectEnv(instance, seed=SEED)
        agent = DDPGAgent(env, seed=SEED, **DDPG_KWARGS)

        t0 = time.time()
        train_info = agent.train(
            episodes     = train_eps,
            eval_interval= eval_int,
            eval_episodes= 30,
            eval_seed    = SEED,
        )
        wall_s = time.time() - t0

        # Final 100-episode evaluation on a fresh env
        final_res = agent.evaluate_real(n_episodes=EVAL_EPS, seed=SEED)
        params    = agent.n_states()

        print(f"  DDPG done: mean={final_res['mean']:.2f} ± {final_res['std']:.2f}  "
              f"({wall_s:.1f}s)  {params}\n")

        # ── Learning curve plot ────────────────────────────────────────────
        plot_learning_curve(
            inst_name      = inst_name,
            train_eps      = train_eps,
            checkpoints    = train_info.get("checkpoint_evals", []),
            baselines      = baselines,
            ddpg_final_mean= final_res["mean"],
            save_path      = os.path.join(OUTPUT_DIR, f"{inst_name}_curve.png"),
        )

        rows.append(dict(
            instance  = inst_name,
            size      = f"{n_jobs}×{n_machines}",
            tasks     = n_tasks,
            best_name = best_name,
            best_mean = round(best_res["mean"], 3),
            best_std  = round(best_res["std"],  3),
            ddpg_mean = round(final_res["mean"], 3),
            ddpg_std  = round(final_res["std"],  3),
            ddpg_min  = round(final_res["min"],  3),
            ddpg_max  = round(final_res["max"],  3),
            params    = params,
            train_eps = train_eps,
            wall_s    = round(wall_s, 1),
            **{f"baseline_{k}_mean": round(v["mean"], 3)
               for k, v in baselines.items()},
            **{f"baseline_{k}_std":  round(v["std"],  3)
               for k, v in baselines.items()},
        ))

    # ── Save CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV → {csv_path}")

    # ── Summary table figure ───────────────────────────────────────────────
    plot_summary_table(rows, os.path.join(OUTPUT_DIR, "results_summary.png"))

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Instance':<10} {'Tasks':>5} {'Best BL':>9} {'DDPG':>9} {'Δ%':>7}  Best method")
    print("-" * 70)
    for r in rows:
        delta_pct = 100.0 * (r["ddpg_mean"] - r["best_mean"]) / max(r["best_mean"], 1e-9)
        print(f"{r['instance']:<10} {r['tasks']:>5} {r['best_mean']:>9.2f} "
              f"{r['ddpg_mean']:>9.2f} {delta_pct:>+7.1f}%  {r['best_name']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
