"""
run_trm_comparison.py
---------------------
Zone-graph vs Region Q-learning on six LB-TRM Frozen Lake instances.

Guard: x >= c_min  AND  y <= D
  x >= c_min : per-phase minimum reaction time (x resets at each goal visit)
  y <= D     : global deadline

Zone graph nodes (4 clock-state pairs, constant for all instances):
  x-node: 0 if x < c_min  else 1
  y-node: 0 if y <= D     else 1

Region clock-state pairs: (c_min+1) × (D+1) — grows linearly with c_min.

Instances fix D=200, d_max=5; c_min varies 2 → 25.

Outputs
-------
  output/trm/<NAME>_convergence.png  — per-instance learning curve
  output/trm/comparison_summary.csv — full results table
  output/trm/comparison_summary.png — summary table figure
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from trm_comparison.trm import INSTANCES, TRMInstance, U_TERMINAL, U1, RM_GOAL_EVENTS
from trm_comparison.grid import (
    GridEnv, ACTIONS, _DELTAS, HOLES, CHECKPOINTS, START, ROWS, COLS,
)
from trm_comparison.zone_agent import TRMZoneAgent
from trm_comparison.region_agent import TRMRegionAgent

# ─── Config ──────────────────────────────────────────────────────────────────

SEED          = 42
SLIP_PROB     = 0.2
EVAL_EPS      = 100
TRAIN_EPS     = 150_000
EVAL_INTERVAL = 5_000
EVAL_INTERIM  = 30
ALPHA         = 0.1
GAMMA         = 0.99
EPS_START     = 0.9
EPS_MIN       = 0.05
BFS_CAP       = 500_000

OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "output", "trm")

INSTANCE_NAMES = ["TRM_01", "TRM_02", "TRM_03", "TRM_04", "TRM_05", "TRM_06"]


# ─── BFS — Region graph ───────────────────────────────────────────────────────

def count_region_graph(trm: TRMInstance, max_states: int = BFS_CAP):
    """BFS over (pos, rm, x_capped, y_capped) states.

    x_capped = min(x, c_min),  y_capped = min(y, D).
    Exact clocks are tracked for guard checking; capped pairs are keys.
    All grid slip outcomes are expanded (conservative reachability).
    """
    initial_key  = (START, U1, 0, 0)
    visited_keys: set = {initial_key}
    queue = deque([(START, U1, 0, 0)])   # (pos, rm, x_exact, y_exact)

    while queue and len(visited_keys) < max_states:
        pos, rm, x_e, y_e = queue.popleft()
        if rm == U_TERMINAL:
            continue

        for delay in range(trm.d_max + 1):
            dt    = delay + 1
            x_new = x_e + dt
            y_new = y_e + dt

            r, c = pos
            for action in ACTIONS:
                for npos in _all_reachable_positions(r, c, action):
                    if npos in HOLES:
                        nkey = (npos, U_TERMINAL, 0, 0)
                        if nkey not in visited_keys:
                            visited_keys.add(nkey)
                        continue

                    event      = CHECKPOINTS.get(npos, "")
                    goal_event = RM_GOAL_EVENTS.get(rm)

                    if event == goal_event:
                        guard_ok = (x_new >= trm.c_min) and (y_new <= trm.D)
                        if guard_ok:
                            next_rm = U_TERMINAL if rm == 3 else rm + 1
                            nx_e    = 0
                        else:
                            next_rm = U_TERMINAL
                            nx_e    = x_new
                        nx_cap = min(nx_e, trm.c_min)
                        ny_cap = min(y_new, trm.D)
                        nkey   = (npos, next_rm, nx_cap, ny_cap)
                        if nkey not in visited_keys:
                            visited_keys.add(nkey)
                            if next_rm != U_TERMINAL:
                                queue.append((npos, next_rm, nx_e, y_new))
                    else:
                        nx_cap = min(x_new, trm.c_min)
                        ny_cap = min(y_new, trm.D)
                        nkey   = (npos, rm, nx_cap, ny_cap)
                        if nkey not in visited_keys:
                            visited_keys.add(nkey)
                            queue.append((npos, rm, x_new, y_new))

    truncated = len(visited_keys) >= max_states
    return len(visited_keys), truncated


def _all_reachable_positions(r: int, c: int, action: int) -> List[Tuple[int, int]]:
    """All grid cells reachable from (r,c) via action (including slip outcomes)."""
    result = set()
    dr, dc = _DELTAS[action]
    nr, nc = max(0, min(ROWS-1, r+dr)), max(0, min(COLS-1, c+dc))
    result.add((nr, nc))
    perp = [2, 3] if action in [0, 1] else [0, 1]
    for pa in perp:
        pdr, pdc = _DELTAS[pa]
        pnr, pnc = max(0, min(ROWS-1, r+pdr)), max(0, min(COLS-1, c+pdc))
        result.add((pnr, pnc))
    return list(result)


# ─── BFS — Zone graph ─────────────────────────────────────────────────────────

def count_zone_graph(trm: TRMInstance, max_states: int = BFS_CAP):
    """BFS over (pos, rm, x_node, y_node) zone-graph states.

    x_node = 0 if x < c_min else 1
    y_node = 0 if y <= D    else 1

    Each zone key is explored from its corner exact values to ensure all
    reachable successor zone keys are discovered.  A single exact (x, y) per
    zone key would miss transitions that only appear from the zone's boundary.
    """
    def zone_key(pos, rm, x, y):
        return (pos, rm, 0 if x < trm.c_min else 1, 0 if y <= trm.D else 1)

    def zone_corners(xn, yn):
        """Exact (x, y) corners that cover all transitions out of zone (xn, yn)."""
        xs = [0, max(0, trm.c_min - 1)] if xn == 0 else [trm.c_min]
        ys = [0, trm.D]                  if yn == 0 else [trm.D + 1]
        return [(x, y) for x in xs for y in ys]

    initial_key = zone_key(START, U1, 0, 0)
    visited_keys: set = {initial_key}
    queue = deque([initial_key])   # queue holds zone keys

    while queue and len(visited_keys) < max_states:
        zk = queue.popleft()
        pos, rm, xn, yn = zk
        if rm == U_TERMINAL:
            continue

        r, c = pos
        for delay in range(trm.d_max + 1):
            for x_c, y_c in zone_corners(xn, yn):
                x_new = x_c + delay + 1
                y_new = y_c + delay + 1

                for action in ACTIONS:
                    for npos in _all_reachable_positions(r, c, action):
                        if npos in HOLES:
                            nkey = zone_key(npos, U_TERMINAL, 0, 0)
                            if nkey not in visited_keys:
                                visited_keys.add(nkey)
                            continue

                        event      = CHECKPOINTS.get(npos, "")
                        goal_event = RM_GOAL_EVENTS.get(rm)

                        if event == goal_event:
                            guard_ok = (x_new >= trm.c_min) and (y_new <= trm.D)
                            if guard_ok:
                                next_rm = U_TERMINAL if rm == 3 else rm + 1
                                nkey = zone_key(npos, next_rm, 0, y_new)  # x resets
                            else:
                                next_rm = U_TERMINAL
                                nkey = zone_key(npos, U_TERMINAL, x_new, y_new)
                            if nkey not in visited_keys:
                                visited_keys.add(nkey)
                                if next_rm != U_TERMINAL:
                                    queue.append(nkey)
                        else:
                            nkey = zone_key(npos, rm, x_new, y_new)
                            if nkey not in visited_keys:
                                visited_keys.add(nkey)
                                queue.append(nkey)

    truncated = len(visited_keys) >= max_states
    return len(visited_keys), truncated


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_convergence(name, train_eps, zone_cps, region_cps, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    if zone_cps:
        zx, zy = zip(*zone_cps)
        ax.plot(zx, zy, color="steelblue", linewidth=2.5, marker="o",
                markersize=5, label="Zone RL (zone-graph)")

    if region_cps:
        rx, ry = zip(*region_cps)
        ax.plot(rx, ry, color="darkorange", linewidth=2.5, marker="s",
                markersize=5, label="Region RL (integer-clock)")

    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Mean total reward (greedy, {EVAL_INTERIM} eval eps)")
    ax.set_title(f"{name} — Zone RL vs Region RL  ({train_eps:,} training episodes)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def _fmt_bfs(n: int, trunc: bool) -> str:
    prefix = ">" if trunc else ""
    return f"{prefix}{n:,}"


def plot_summary_table(rows: list, save_path: str):
    headers = [
        "Instance",
        "c_min / D\n(guards)",
        "Region State\nBound (c_min+1)×(D+1)",
        "Zone RL\nmean ± std",
        "Zone\nVisited States",
        "Region RL\nmean ± std",
        "Region\nVisited States",
        "Zone BFS\n(graph nodes)",
        "Region BFS\n(graph nodes)",
    ]

    cell_rows = []
    for r in rows:
        cell_rows.append([
            r["instance"],
            f"{r['c_min']} / {r['D']}",
            f"{(r['c_min']+1)*(r['D']+1):,}",
            f"{r['zone_mean']:.1f} ± {r['zone_std']:.1f}",
            f"{r['zone_visited']:,}",
            f"{r['region_mean']:.1f} ± {r['region_std']:.1f}",
            f"{r['region_visited']:,}",
            _fmt_bfs(r["zone_bfs"], r["zone_bfs_trunc"]),
            _fmt_bfs(r["region_bfs"], r["region_bfs_trunc"]),
        ])

    n_rows = len(cell_rows)
    fig_h  = max(3.5, n_rows * 0.7 + 1.6)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_rows, colLabels=headers,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(col=list(range(len(headers))))

    for ci in range(len(headers)):
        tbl[0, ci].set_facecolor("#e0e0e0")
        tbl[0, ci].set_text_props(weight="bold")
    for ri in range(1, n_rows + 1):
        for ci in [3, 4, 7]:
            tbl[ri, ci].set_facecolor("#ddeeff")
        for ci in [5, 6, 8]:
            tbl[ri, ci].set_facecolor("#fff0dd")
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_height(0.065)

    ax.set_title(
        f"Zone RL vs Region RL — LB-TRM Frozen Lake (d_max=5, slip=0.2)\n"
        f"Guard: x ≥ c_min AND y ≤ D  |  "
        f"Training: {TRAIN_EPS:,} episodes  |  Final eval: {EVAL_EPS} episodes  |  Seed {SEED}",
        fontsize=9, pad=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary table saved: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []

    print("=" * 70)
    print("Zone RL vs Region RL — LB-TRM Frozen Lake Comparison")
    print(f"Guard: x >= c_min AND y <= D  |  d_max=5  slip={SLIP_PROB}")
    print(f"α={ALPHA}  γ={GAMMA}  ε: {EPS_START}→{EPS_MIN}")
    print(f"training: {TRAIN_EPS:,}  |  eval: {EVAL_EPS}")
    print("=" * 70 + "\n")

    for inst_name in INSTANCE_NAMES:
        trm = INSTANCES[inst_name]
        region_bound = (trm.c_min + 1) * (trm.D + 1)

        print(f"=== {inst_name}: c_min={trm.c_min}, D={trm.D}, d_max={trm.d_max}  "
              f"(region clock pairs: {region_bound:,}) ===")

        # ── Zone RL ───────────────────────────────────────────────────────
        zone_agent = TRMZoneAgent(
            trm,
            alpha=ALPHA, gamma=GAMMA,
            epsilon_start=EPS_START, epsilon_min=EPS_MIN,
            slip_prob=SLIP_PROB, seed=SEED,
        )
        t0 = time.time()
        zone_info = zone_agent.train(
            episodes=TRAIN_EPS,
            eval_interval=EVAL_INTERVAL,
            eval_episodes=EVAL_INTERIM,
            eval_seed=SEED,
        )
        zone_res = zone_agent.evaluate(n_episodes=EVAL_EPS, seed=SEED)
        print(f"  [Zone RL]   {time.time()-t0:.1f}s  "
              f"mean={zone_res['mean']:.1f}  visited={zone_agent.n_states():,}")

        # ── Region RL ─────────────────────────────────────────────────────
        region_agent = TRMRegionAgent(
            trm,
            alpha=ALPHA, gamma=GAMMA,
            epsilon_start=EPS_START, epsilon_min=EPS_MIN,
            slip_prob=SLIP_PROB, seed=SEED,
        )
        t0 = time.time()
        region_info = region_agent.train(
            episodes=TRAIN_EPS,
            eval_interval=EVAL_INTERVAL,
            eval_episodes=EVAL_INTERIM,
            eval_seed=SEED,
        )
        region_res = region_agent.evaluate(n_episodes=EVAL_EPS, seed=SEED)
        print(f"  [Region RL] {time.time()-t0:.1f}s  "
              f"mean={region_res['mean']:.1f}  visited={region_agent.n_states():,}")

        # ── BFS graph sizes ────────────────────────────────────────────────
        t0 = time.time()
        rbfs, rtrunc = count_region_graph(trm, max_states=BFS_CAP)
        print(f"  BFS region: {rbfs:,}{'+ (cap)' if rtrunc else ''}  "
              f"({time.time()-t0:.1f}s)")

        t0 = time.time()
        zbfs, ztrunc = count_zone_graph(trm, max_states=BFS_CAP)
        print(f"  BFS zone:   {zbfs:,}{'+ (cap)' if ztrunc else ''}  "
              f"({time.time()-t0:.1f}s)\n")

        # ── Convergence plot ───────────────────────────────────────────────
        plot_convergence(
            inst_name, TRAIN_EPS,
            zone_cps=zone_info.get("checkpoint_evals", []),
            region_cps=region_info.get("checkpoint_evals", []),
            save_path=os.path.join(OUTPUT_DIR, f"{inst_name}_convergence.png"),
        )

        rows.append({
            "instance":         inst_name,
            "c_min":            trm.c_min,
            "D":                trm.D,
            "d_max":            trm.d_max,
            "region_bound":     region_bound,
            "zone_mean":        round(zone_res["mean"], 2),
            "zone_std":         round(zone_res["std"],  2),
            "zone_min":         round(zone_res["min"],  2),
            "zone_max":         round(zone_res["max"],  2),
            "zone_visited":     zone_agent.n_states(),
            "region_mean":      round(region_res["mean"], 2),
            "region_std":       round(region_res["std"],  2),
            "region_min":       round(region_res["min"],  2),
            "region_max":       round(region_res["max"],  2),
            "region_visited":   region_agent.n_states(),
            "zone_bfs":         zbfs,
            "zone_bfs_trunc":   ztrunc,
            "region_bfs":       rbfs,
            "region_bfs_trunc": rtrunc,
        })

    # ── Summary CSV ────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved: {csv_path}")

    # ── Summary table figure ───────────────────────────────────────────────
    plot_summary_table(rows, save_path=os.path.join(OUTPUT_DIR, "comparison_summary.png"))

    # ── Console summary ────────────────────────────────────────────────────
    print("\nResults:")
    print(f"{'Instance':<10} {'c_min/D':>8}  {'RegBound':>10}  "
          f"{'ZoneRL':>8}  {'ZoneVis':>9}  "
          f"{'RegRL':>8}  {'RegVis':>9}  "
          f"{'ZoneBFS':>10}  {'RegBFS':>10}")
    print("-" * 105)
    for r in rows:
        print(
            f"{r['instance']:<10} "
            f"{r['c_min']}/{r['D']}  "
            f"{r['region_bound']:>10,}  "
            f"{r['zone_mean']:>8.1f}  "
            f"{r['zone_visited']:>9,}  "
            f"{r['region_mean']:>8.1f}  "
            f"{r['region_visited']:>9,}  "
            f"{_fmt_bfs(r['zone_bfs'], r['zone_bfs_trunc']):>10}  "
            f"{_fmt_bfs(r['region_bfs'], r['region_bfs_trunc']):>10}"
        )


if __name__ == "__main__":
    main()
