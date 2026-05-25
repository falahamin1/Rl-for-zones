"""
run_benchmarks.py
-----------------
Run all 10 PTA job-shop benchmark instances against all 5 baseline strategies.
Produces:
  output/summary.csv           — mean ± std makespan per instance/strategy
  output/summary_table.png     — visual table
  output/<NAME>_histograms.png — makespan histogram per instance
  output/<NAME>_gantt_<strat>.png — Gantt chart for one sample run per strategy
"""
import os
import sys
import time
from typing import Dict

import pandas as pd

from prob_jobshop.benchmarks.all_instances import get_all_instances, ALL_INSTANCE_NAMES
from prob_jobshop.simulator import PTASimulator
from prob_jobshop.strategies import (
    random_strategy,
    sept_strategy,
    lept_strategy,
    mwr_strategy,
    fifo_strategy,
)
from prob_jobshop.verification import verify_instance, verify_simulation
from prob_jobshop.visualization import (
    plot_gantt,
    plot_makespan_histograms,
    plot_summary_table,
)

N_EPISODES = 1000
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    instances = get_all_instances()

    print(f"{'='*70}")
    print(f"Probabilistic Job-Shop Scheduling Benchmark Suite")
    print(f"Instances: {len(instances)}  |  Episodes per strategy: {N_EPISODES}")
    print(f"{'='*70}\n")

    all_results: Dict[str, Dict[str, Dict]] = {}
    csv_rows = []

    for instance in instances:
        print(f"--- {instance.name} : {instance.description[:60]} ---")

        # Verification
        errors = verify_instance(instance)
        if errors:
            print(f"  [FAIL] Verification errors for {instance.name}:")
            for e in errors:
                print(f"    {e}")
            sys.exit(1)

        sim = PTASimulator(instance, seed=SEED)

        strategy_factories = {
            "random": lambda inst, s=sim: random_strategy(inst, rng=s.rng),
            "sept":   sept_strategy,
            "lept":   lept_strategy,
            "mwr":    mwr_strategy,
            "fifo":   fifo_strategy,
        }

        instance_results: Dict[str, Dict] = {}
        gantt_traces = {}

        for strat_name, factory in strategy_factories.items():
            strategy_fn = factory(instance)
            t0 = time.time()
            res = sim.evaluate_strategy(
                strategy_fn,
                n_episodes=N_EPISODES,
                desc=f"  {instance.name} / {strat_name}",
            )
            elapsed = time.time() - t0
            instance_results[strat_name] = res
            print(
                f"  {strat_name:8s}  mean={res['mean']:7.2f}  std={res['std']:6.2f}"
                f"  min={res['min']:6.1f}  max={res['max']:6.1f}"
                f"  p90={res['p90']:6.1f}  [{elapsed:.1f}s]"
            )

            # Record one Gantt trace per strategy (use a fresh sim for reproducibility)
            trace_sim = PTASimulator(instance, seed=SEED + 1)
            _, trace = trace_sim.run_episode(factory(instance), record_trace=True)
            sim_errors = verify_simulation(instance, trace)
            if sim_errors:
                print(f"  [WARN] Simulation trace errors ({strat_name}): {sim_errors}")
            gantt_traces[strat_name] = trace

        all_results[instance.name] = instance_results

        lb = instance.expected_makespan_lower_bound()
        ub = instance.worst_case_makespan_upper_bound()
        print(
            f"  LB={lb:.1f}  UB={ub}  "
            f"#jobs={len(instance.jobs)}  #machines={len(instance.machines)}  "
            f"#tasks={instance.num_tasks()}\n"
        )

        # CSV row
        row = {
            "Instance": instance.name,
            "Jobs": len(instance.jobs),
            "Machines": len(instance.machines),
            "Tasks": instance.num_tasks(),
            "LB_expected": round(lb, 2),
            "UB_worstcase": ub,
        }
        for sn, res in instance_results.items():
            row[f"{sn}_mean"] = round(res["mean"], 2)
            row[f"{sn}_std"] = round(res["std"], 2)
        csv_rows.append(row)

        # Histograms
        plot_makespan_histograms(
            instance_results,
            instance.name,
            save_path=os.path.join(OUTPUT_DIR, f"{instance.name}_histograms.png"),
        )

        # Gantt charts
        for strat_name, trace in gantt_traces.items():
            plot_gantt(
                trace,
                instance,
                title=f"{instance.name} — {strat_name} strategy",
                save_path=os.path.join(
                    OUTPUT_DIR, f"{instance.name}_gantt_{strat_name}.png"
                ),
            )

    # Summary CSV
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to: {csv_path}")

    # Summary table figure
    table_path = os.path.join(OUTPUT_DIR, "summary_table.png")
    plot_summary_table(all_results, save_path=table_path)
    print(f"Summary table saved to: {table_path}")

    print("\nAll benchmarks complete.")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
