"""
run_region_rl.py
----------------
Train a region-based tabular Q-learning agent on all 10 PJS benchmarks and
compare against the five baseline strategies.

State space: (simplified_loc, integer_clock_tuple) — exponential in #jobs.
This runner is intentionally parallel in structure to run_symbolic_rl.py so
that learning curves and state-space sizes can be directly compared.

Outputs
-------
  output/region_rl_summary.csv          — mean/std makespan + state-space sizes
  output/region_rl_summary.png          — visual summary table
  output/<NAME>_region_learning.png     — per-instance convergence curve
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
from prob_jobshop.region_env import RegionJobShopEnv
from prob_jobshop.region_rl import RegionRLAgent
from prob_jobshop.verification import verify_instance

INSTANCES     = ALL_INSTANCE_NAMES          # PJS_01 … PJS_10
EVAL_EPS      = 100
SEED          = 42
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "output")

EPSILON_START = 0.9
EPSILON_MIN   = 0.05
LR            = 0.1
GAMMA         = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_train_episodes(instance) -> int:
    """Scale training episodes with instance size: 100 eps/task, capped [1 000, 8 000]."""
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


def plot_learning_curve(episode_costs, instance_name, save_path, checkpoint_evals=None):
    fig, ax = plt.subplots(figsize=(10, 4))

    window = max(1, len(episode_costs) // 20)
    smoothed = np.convolve(episode_costs, np.ones(window) / window, mode="valid")
    ax.plot(episode_costs, alpha=0.2, color="steelblue", linewidth=0.6,
            label="episode makespan (training)")
    ax.plot(range(window - 1, len(episode_costs)), smoothed,
            color="steelblue", linewidth=2, label=f"smoothed (w={window})")

    if checkpoint_evals:
        ck_eps, ck_means = zip(*checkpoint_evals)
        ax.plot(ck_eps, ck_means, color="darkorange", linewidth=2,
                marker="o", markersize=5, label="greedy policy — real sim")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Makespan")
    ax.set_title(f"{instance_name} — Region RL convergence")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []

    print(f"{'='*65}")
    print(f"Region RL Benchmark  |  instances: {INSTANCES}")
    print(f"Eval episodes: {EVAL_EPS}  |  ε: {EPSILON_START} → {EPSILON_MIN}  |  lr: {LR}")
    print(f"{'='*65}\n")

    for inst_name in INSTANCES:
        instance = get_instance_by_name(inst_name)

        errors = verify_instance(instance)
        if errors:
            print(f"[FAIL] {inst_name} verification errors: {errors}")
            sys.exit(1)

        n_jobs     = len(instance.jobs)
        n_machines = len(instance.machines)
        n_tasks    = instance.num_tasks()
        print(f"--- {inst_name}: {instance.description[:60]} ---")
        print(f"    jobs={n_jobs}  machines={n_machines}  tasks={n_tasks}")

        # Theoretical state-space upper bound:
        #   ∏_j (max_duration_j + 2) × number of simplified locations
        # We report the product of per-job max constants as a proxy.
        max_consts = {
            j.job_id: max(t.distribution.max_duration() for t in j.tasks) + 1
            for j in instance.jobs
        }
        clock_space_ub = 1
        for v in max_consts.values():
            clock_space_ub *= (v + 1)
        print(f"    clock-space UB (∏ max_const+1): {clock_space_ub:,}")

        # --- Baselines ---
        t0 = time.time()
        baselines = run_baselines(instance, n_episodes=EVAL_EPS)
        print(f"  Baselines done in {time.time()-t0:.1f}s")
        for bname, res in baselines.items():
            print(f"    {bname:8s}  mean={res['mean']:7.2f}  std={res['std']:5.2f}")

        # --- Region RL ---
        train_eps     = compute_train_episodes(instance)
        eval_interval = max(1, train_eps // 10)
        env   = RegionJobShopEnv(instance, seed=SEED)
        agent = RegionRLAgent(
            env,
            epsilon_start=EPSILON_START,
            epsilon_min=EPSILON_MIN,
            lr=LR,
            gamma=GAMMA,
            seed=SEED,
        )

        print(f"  Training region RL for {train_eps} episodes "
              f"(ε {EPSILON_START}→{EPSILON_MIN}, eval every {eval_interval}) ...")
        t0 = time.time()
        train_info = agent.train(
            episodes=train_eps,
            eval_interval=eval_interval,
            eval_episodes=20,
            eval_seed=SEED,
        )
        train_time = time.time() - t0
        print(f"  Training done in {train_time:.1f}s  |  Q-table states: {agent.n_states():,}")

        # Final greedy evaluation
        real_res = agent.evaluate_real(n_episodes=EVAL_EPS, seed=SEED)
        print(f"  region_rl     mean={real_res['mean']:7.2f}  std={real_res['std']:5.2f}")

        lb = instance.expected_makespan_lower_bound()
        ub = instance.worst_case_makespan_upper_bound()
        print(f"  LB={lb:.1f}  UB={ub}\n")

        # Learning curve
        plot_learning_curve(
            train_info["episode_costs"],
            inst_name,
            save_path=os.path.join(OUTPUT_DIR, f"{inst_name}_region_learning.png"),
            checkpoint_evals=train_info.get("checkpoint_evals"),
        )

        # Build summary row
        row = {"Instance": inst_name, "Tasks": n_tasks, "ClockSpaceUB": clock_space_ub}
        for bname, res in baselines.items():
            row[f"{bname}_mean"] = round(res["mean"], 2)
            row[f"{bname}_std"]  = round(res["std"],  2)
        row["region_rl_mean"]  = round(real_res["mean"], 2)
        row["region_rl_std"]   = round(real_res["std"],  2)
        row["q_table_states"]  = agent.n_states()
        rows.append(row)

    # CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "region_rl_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved: {csv_path}")

    # Summary table figure
    fig, ax = plt.subplots(figsize=(18, max(3, len(rows) * 0.5 + 1)))
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
    ax.set_title("Region RL vs Baseline Strategy Comparison", fontsize=11, pad=10)
    plt.tight_layout()
    table_path = os.path.join(OUTPUT_DIR, "region_rl_summary.png")
    plt.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary table saved: {table_path}")

    print("\nFull results:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
