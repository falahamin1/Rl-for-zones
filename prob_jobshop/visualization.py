from __future__ import annotations
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .instance import ProbJobShopInstance
    from .pta import GlobalState


def _extract_gantt_data(
    state_trace: List["GlobalState"],
    instance: "ProbJobShopInstance",
) -> List[Tuple[str, str, float, float]]:
    """Return list of (task_id, machine, start_time, end_time) from a trace."""
    records = []
    start_times: Dict[str, float] = {}

    for i, state in enumerate(state_trace):
        for tid, ts in state.task_states.items():
            if ts.status == "active" and tid not in start_times:
                start_times[tid] = state.current_time
            if ts.status == "done" and tid in start_times and tid not in {
                r[0] for r in records
            }:
                machine = instance.task_by_id(tid).machine
                records.append((tid, machine, start_times[tid], state.current_time))

    return records


def plot_gantt(
    state_trace: List["GlobalState"],
    instance: "ProbJobShopInstance",
    title: str = "Gantt Chart",
    save_path: Optional[str] = None,
) -> None:
    records = _extract_gantt_data(state_trace, instance)

    machine_order = instance.machines
    machine_idx = {m: i for i, m in enumerate(machine_order)}

    job_ids = [job.job_id for job in instance.jobs]
    cmap = plt.get_cmap("tab10")
    job_colors = {jid: cmap(i % 10) for i, jid in enumerate(job_ids)}

    fig, ax = plt.subplots(figsize=(12, max(4, len(machine_order) * 0.7)))

    for tid, machine, start, end in records:
        job_id = instance.job_of_task(tid).job_id
        color = job_colors[job_id]
        y = machine_idx[machine]
        ax.barh(y, end - start, left=start, height=0.6, color=color, edgecolor="black", linewidth=0.5)
        if end - start > 0.5:
            ax.text(
                start + (end - start) / 2, y, tid.split("_")[1],
                ha="center", va="center", fontsize=6, color="white", fontweight="bold",
            )

    ax.set_yticks(range(len(machine_order)))
    ax.set_yticklabels(machine_order)
    ax.set_xlabel("Time")
    ax.set_title(title)

    legend_patches = [
        mpatches.Patch(color=job_colors[jid], label=jid) for jid in job_ids
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_makespan_histograms(
    results_by_strategy: Dict[str, Dict],
    instance_name: str,
    save_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")

    all_values = []
    for v in results_by_strategy.values():
        all_values.extend(v["raw"])
    lo, hi = min(all_values), max(all_values)
    bins = np.linspace(lo, hi, 31)

    for i, (name, res) in enumerate(results_by_strategy.items()):
        color = cmap(i % 10)
        ax.hist(
            res["raw"], bins=bins, alpha=0.45, color=color,
            label=f"{name} (μ={res['mean']:.1f})",
        )
        ax.axvline(res["mean"], color=color, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Makespan")
    ax.set_ylabel("Count")
    ax.set_title(f"{instance_name} — Makespan Distribution by Strategy")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_summary_table(
    all_results: Dict[str, Dict[str, Dict]],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build and optionally save a summary table as a matplotlib figure.

    all_results: {instance_name: {strategy_name: {mean, std, ...}}}
    Returns the DataFrame for further use.
    """
    rows = []
    for inst_name, strat_results in all_results.items():
        row = {"Instance": inst_name}
        for strat_name, res in strat_results.items():
            row[f"{strat_name}_mean"] = f"{res['mean']:.1f}"
            row[f"{strat_name}_std"] = f"{res['std']:.1f}"
        rows.append(row)

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, max(3, len(rows) * 0.5 + 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    ax.set_title("Strategy Comparison Summary", fontsize=12, pad=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    return df
