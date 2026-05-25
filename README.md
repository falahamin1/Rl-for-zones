# Probabilistic Job-Shop Scheduling Benchmark Suite

A Python package (`prob_jobshop`) providing 10 probabilistic job-shop scheduling benchmark instances, a discrete-event simulator, 5 baseline scheduling strategies, verification utilities, and visualisation tools. Designed as an environment layer for reinforcement learning research.

---

## Table of Contents

1. [Concepts](#concepts)
   - [The Job-Shop Scheduling Problem](#the-job-shop-scheduling-problem)
   - [What Makes It Probabilistic](#what-makes-it-probabilistic)
   - [Probabilistic Timed Automata (PTA)](#probabilistic-timed-automata-pta)
   - [Instance](#instance)
   - [Simulator](#simulator)
   - [Strategy](#strategy)
   - [Makespan](#makespan)
2. [Package Structure](#package-structure)
3. [Benchmark Instances](#benchmark-instances)
   - [PJS_01 — Tiny, near-deterministic](#pjs_01--tiny-near-deterministic)
   - [PJS_02 — Small, moderate uncertainty](#pjs_02--small-moderate-uncertainty)
   - [PJS_03 — Small, high variance](#pjs_03--small-high-variance)
   - [PJS_04 — Probabilistic FT03 equivalent](#pjs_04--probabilistic-ft03-equivalent)
   - [PJS_05 — Bimodal uncertainty](#pjs_05--bimodal-uncertainty)
   - [PJS_06 — Probabilistic LA01 equivalent](#pjs_06--probabilistic-la01-equivalent)
   - [PJS_07 — Asymmetric uncertainty per machine group](#pjs_07--asymmetric-uncertainty-per-machine-group)
   - [PJS_08 — Probabilistic FT06 equivalent](#pjs_08--probabilistic-ft06-equivalent)
   - [PJS_09 — Large, high variance, own routing](#pjs_09--large-high-variance-own-routing)
   - [PJS_10 — Stress test, probabilistic FT10 equivalent](#pjs_10--stress-test-probabilistic-ft10-equivalent)
4. [Difficulty Progression](#difficulty-progression)
5. [Baseline Strategies](#baseline-strategies)
6. [Running the Benchmarks](#running-the-benchmarks)
7. [Output Files](#output-files)
8. [Using the API](#using-the-api)

---

## Concepts

### The Job-Shop Scheduling Problem

In a job-shop scheduling problem (JSSP) you have:

- A set of **jobs**, each consisting of an ordered sequence of **tasks**.
- A set of **machines**. Each task must be processed on a specific machine.
- **Precedence constraints**: a job's tasks must be executed in order — task 2 of a job cannot start until task 1 is finished.
- **Mutual exclusion**: each machine can process at most one task at a time.

The goal is to decide, at every moment, which task to start next on each available machine so that the total completion time (called the **makespan**) is minimised.

Example: Job J1 needs machine m1 then machine m2. Job J2 needs machine m2 then machine m1. If both jobs start at the same time, one task must wait while the other occupies the shared machine. The scheduler must decide who goes first.

### What Makes It Probabilistic

In the classical JSSP every task has a fixed, known duration. In the real world, task durations are uncertain: a machining step might take 3 minutes on a good day or 8 minutes if there is vibration, tool wear, or material variation.

In this benchmark suite, each task has a **discrete probability distribution** over possible durations instead of a single fixed value. For example:

```
TaskDistribution(durations=[3, 4, 5], probabilities=[0.1, 0.8, 0.1])
```

This means the task takes 3 time units with 10% probability, 4 time units with 80% probability, and 5 time units with 10% probability.

The actual duration is **not known in advance** — it is sampled at the moment the task starts. A strategy therefore cannot look up the exact duration; it can only use the distribution (expected value, variance, etc.) to make decisions.

### Probabilistic Timed Automata (PTA)

A **Timed Automaton (TA)** is a formal model for a system that has both discrete states (like "waiting", "running", "done") and continuous clocks that measure elapsed time. A **Probabilistic Timed Automaton (PTA)** extends this by adding discrete probability distributions on transitions — instead of always taking a fixed duration, the system samples a duration from a distribution when a transition fires.

In this package, each task is modelled as a three-state PTA:

```
waiting ──[start]──► active ──[clock = sampled_duration]──► done
```

- **waiting**: the task has not started yet.
- **active**: the task is running; its clock is ticking; the actual duration was sampled at the moment of the `start` transition.
- **done**: the task has completed.

The full system state at any point in time is a `GlobalState` containing:

- The status of every task (`waiting`, `active`, or `done`).
- The current value of each job's clock (one clock per job, reset to zero whenever a new task in that job starts).
- The current simulation time.

`GlobalState` is **immutable by convention**: every operation (`start_task`, `advance_time`) returns a brand-new `GlobalState` object. This makes it safe to pass states to multiple strategies or replay logic without mutation side effects.

### Instance

An **instance** (`ProbJobShopInstance`) is a complete problem description:

- `name` — identifier string (e.g., `"PJS_01"`).
- `description` — human-readable summary.
- `machines` — ordered list of machine IDs (e.g., `["m1", "m2", "m3"]`).
- `jobs` — list of `Job` objects. Each `Job` holds an ordered list of `Task` objects.

Each `Task` specifies:
- `task_id` — unique ID in the format `"J1_T2"` (job 1, second task).
- `machine` — which machine it needs (e.g., `"m1"`).
- `distribution` — a `TaskDistribution` giving the discrete probability distribution over integer durations.

An instance also exposes two theoretical bounds:

- **Expected makespan lower bound**: the maximum, over all machines, of the sum of expected durations of tasks assigned to that machine. A schedule cannot complete faster than this even with perfect routing.
- **Worst-case makespan upper bound**: the sum of the maximum possible duration of every task across all jobs. The actual makespan can never exceed this.

### Simulator

`PTASimulator` is a discrete-event simulation engine. It drives the PTA model forward in time by alternating two phases:

1. **Dispatch phase**: ask the strategy which task to start next. Start it (sample its duration from the distribution), then ask again. Repeat until the strategy returns `None` (meaning no more tasks can or should be started right now).

2. **Advance phase**: find the active task that will finish soonest. Jump the simulation clock forward to that moment and mark that task (and any others finishing at the same instant) as done.

These two phases alternate until all tasks are done. The total elapsed time is the **makespan** for that episode.

The simulator exposes a seeded `numpy` random number generator (`self.rng`). Passing the same seed always produces the same sequence of sampled durations, making results reproducible.

### Strategy

A **strategy** is a function that takes the current `GlobalState` and returns either a `task_id` to start next, or `None` if it does not want to start anything right now.

Strategies are created by **factory functions** that capture the `instance` in a closure:

```python
strategy_fn = sept_strategy(instance)      # create a closure
task_id = strategy_fn(current_state)       # call it at each decision point
```

At each dispatch step the simulator calls the strategy repeatedly until it returns `None`. This means a strategy can start multiple tasks in a single dispatch round (e.g., start one task on m1 and another on m3 before any time passes).

A task is **enabled** (eligible to be started) if and only if:
1. Its status is `"waiting"`.
2. Its predecessor task (the previous task in the same job) is `"done"`, or it has no predecessor.
3. Its required machine is not currently occupied by another active task.

---

## Package Structure

```
scheduling/
├── requirements.txt
├── run_benchmarks.py           ← main runner script
└── prob_jobshop/
    ├── __init__.py
    ├── instance.py             ← TaskDistribution, Task, Job, ProbJobShopInstance
    ├── pta.py                  ← TaskState, GlobalState, build_task_automaton
    ├── simulator.py            ← PTASimulator
    ├── strategies.py           ← random, sept, lept, mwr, fifo
    ├── verification.py         ← verify_instance, verify_simulation
    ├── visualization.py        ← plot_gantt, plot_makespan_histograms, plot_summary_table
    └── benchmarks/
        ├── __init__.py
        ├── _utils.py           ← merge_distribution, distribution formula helpers
        ├── all_instances.py    ← get_all_instances(), get_instance_by_name()
        ├── pjs_01.py
        ├── pjs_02.py
        ├── ...
        └── pjs_10.py
```

---

## Benchmark Instances

All instances use integer-valued time units. Machine IDs are `m1`, `m2`, ... (1-indexed). Task IDs follow the pattern `J{job_index}_T{task_position}`.

---

### PJS_01 — Tiny, near-deterministic

**Size**: 2 jobs, 2 machines, 4 tasks total  
**Purpose**: Sanity check and warm-up. Almost no uncertainty.

| Task   | Machine | Durations   | Probabilities       | E[d] |
|--------|---------|-------------|---------------------|------|
| J1_T1  | m1      | 3, 4, 5     | 0.10, 0.80, 0.10    | 4.0  |
| J1_T2  | m2      | 2, 3, 4     | 0.10, 0.80, 0.10    | 3.0  |
| J2_T1  | m2      | 2, 3, 4     | 0.20, 0.60, 0.20    | 3.0  |
| J2_T2  | m1      | 1, 2, 3     | 0.10, 0.80, 0.10    | 2.0  |

**Routing**: J1 → m1 → m2; J2 → m2 → m1.  
The 80% probability mass on the central value makes variance very low. All five strategies produce similar makespans (~7 time units).

---

### PJS_02 — Small, moderate uncertainty

**Size**: 2 jobs, 3 machines, 5 tasks total  
**Purpose**: Introduces a third machine and moderate symmetric uncertainty (30–40–30 distribution).

| Task   | Machine | Durations   | Probabilities       | E[d] |
|--------|---------|-------------|---------------------|------|
| J1_T1  | m1      | 2, 4, 6     | 0.30, 0.40, 0.30    | 4.0  |
| J1_T2  | m2      | 3, 5, 7     | 0.30, 0.40, 0.30    | 5.0  |
| J2_T1  | m2      | 1, 3, 5     | 0.30, 0.40, 0.30    | 3.0  |
| J2_T2  | m3      | 2, 4, 6     | 0.30, 0.40, 0.30    | 4.0  |
| J2_T3  | m1      | 2, 3, 4     | 0.30, 0.40, 0.30    | 3.0  |

**Routing**: J1 → m1 → m2; J2 → m2 → m3 → m1.  
The 30–40–30 split is the signature pattern used by PJS_02, PJS_04, and PJS_06: equal weight on low and high outcomes, slightly heavier centre.

---

### PJS_03 — Small, high variance

**Size**: 3 jobs, 2 machines, 6 tasks total  
**Purpose**: Stress-tests strategies under high uncertainty with a small instance where optimal decisions are still traceable.

| Task   | Machine | Durations   | Probabilities       | E[d] |
|--------|---------|-------------|---------------------|------|
| J1_T1  | m1      | 1, 4, 8     | 0.40, 0.30, 0.30    | 3.7  |
| J1_T2  | m2      | 2, 5, 8     | 0.30, 0.40, 0.30    | 5.0  |
| J2_T1  | m2      | 1, 3, 6     | 0.30, 0.40, 0.30    | 3.3  |
| J2_T2  | m1      | 2, 5, 8     | 0.30, 0.40, 0.30    | 5.0  |
| J3_T1  | m1      | 2, 4, 7     | 0.40, 0.30, 0.30    | 3.9  |
| J3_T2  | m2      | 1, 4, 7     | 0.30, 0.40, 0.30    | 4.3  |

**Routing**: J1 → m1 → m2; J2 → m2 → m1; J3 → m1 → m2.  
The wide spread (e.g., duration 1 vs 8) means a bad draw from a single task can dominate the makespan. The 40% weight on the low outcome in some tasks skews the distribution left.

---

### PJS_04 — Probabilistic FT03 equivalent

**Size**: 3 jobs, 3 machines, 9 tasks total  
**Purpose**: A probabilistic version of the classic Fisher-Thompson FT03 benchmark. Every task uses the symmetric 30–40–30 distribution.

| Task   | Machine | Durations | E[d] |
|--------|---------|-----------|------|
| J1_T1  | m1      | 1, 2, 3   | 2.0  |
| J1_T2  | m2      | 3, 4, 5   | 4.0  |
| J1_T3  | m3      | 2, 3, 4   | 3.0  |
| J2_T1  | m2      | 2, 3, 4   | 3.0  |
| J2_T2  | m3      | 1, 2, 3   | 2.0  |
| J2_T3  | m1      | 2, 4, 6   | 4.0  |
| J3_T1  | m3      | 2, 4, 6   | 4.0  |
| J3_T2  | m1      | 2, 3, 4   | 3.0  |
| J3_T3  | m2      | 1, 3, 5   | 3.0  |

All probabilities are `[0.30, 0.40, 0.30]`. The FT03 routing structure means jobs interleave across all three machines.

---

### PJS_05 — Bimodal uncertainty

**Size**: 4 jobs, 3 machines, 12 tasks total  
**Purpose**: Introduces a genuine bimodal distribution — each task is either "fast" (small duration, ~50% of the time) or "slow" (large duration, ~40% of the time), with a thin centre.

Representative distribution: `TaskDistribution([2, 3, 7], [0.5, 0.1, 0.4])`

This means a task takes 2 time units 50% of the time and 7 time units 40% of the time. The expected value (~4.4) is not a likely outcome — most runs see either the fast or the slow branch.

**Routing summary** (J1: m1→m2→m3; J2: m2→m1→m3; J3: m3→m2→m1; J4: m1→m3→m2).  
The bimodal structure makes makespan distributions heavy-tailed: strategies that happen to avoid triggering slow branches win by a wide margin. This instance is good for measuring strategy robustness under luck asymmetry.

---

### PJS_06 — Probabilistic LA01 equivalent

**Size**: 5 jobs, 5 machines, 25 tasks total  
**Purpose**: A probabilistic version of the classical Lawrence LA01 benchmark. All tasks use the symmetric 30–40–30 distribution, with moderate uncertainty.

Every task distribution has the pattern `[low, mid, high]` with probabilities `[0.3, 0.4, 0.3]`. The mid value is the modal duration.

**Routing** (first machine in each job's sequence):

| Job | Route              |
|-----|--------------------|
| J1  | m2→m3→m1→m4→m5    |
| J2  | m3→m1→m5→m2→m4    |
| J3  | m5→m4→m2→m1→m3    |
| J4  | m4→m2→m3→m5→m1    |
| J5  | m1→m5→m4→m3→m2    |

This instance serves as the baseline for PJS_07, which uses the same routing but replaces the distributions with machine-group-specific patterns.

---

### PJS_07 — Asymmetric uncertainty per machine group

**Size**: 5 jobs, 5 machines, 25 tasks total  
**Purpose**: Same routing as PJS_06, but uncertainty is deliberately different for each machine group. This tests whether strategies can exploit the fact that some machines are predictable and others are not.

The **nominal duration** `d` for each task is taken as the modal duration (the middle value, index 1 with probability 0.4) from the corresponding PJS_06 task. The new distribution is then derived by the following machine-group rule:

| Machine group | Distribution formula | Interpretation |
|---------------|----------------------|----------------|
| m1, m2        | `(round(0.9d), 0.1), (d, 0.8), (round(1.1d), 0.1)` | Near-deterministic: 80% on nominal, ±10% tails |
| m3, m4        | `(round(0.5d), 0.4), (d, 0.2), (round(1.5d), 0.4)` | High uncertainty: equal weight on half and 1.5× nominal |
| m5            | `(round(0.6d), 0.5), (d, 0.1), (round(1.8d), 0.4)` | Bimodal: half the time fast (60% of nominal), 40% of the time very slow (1.8× nominal) |

When rounding produces duplicate durations, their probabilities are merged (summed) so the distribution remains valid.

Example for a task on m5 with nominal `d = 5`:
- fast outcome: `round(0.6 × 5) = 3` with probability 0.50
- nominal: `5` with probability 0.10
- slow outcome: `round(1.8 × 5) = 9` with probability 0.40

---

### PJS_08 — Probabilistic FT06 equivalent

**Size**: 6 jobs, 6 machines, 36 tasks total  
**Purpose**: A probabilistic version of the classical Fisher-Thompson FT06 benchmark. Nominal durations are taken directly from the published FT06 problem.

**Distribution formula** applied to every nominal duration `d`:

```
(floor(0.7 × d), 0.25),  (d, 0.50),  (ceil(1.4 × d), 0.25)
```

This is a symmetric three-point distribution: 25% chance of a run 30% faster than nominal, 50% chance of the nominal, 25% chance of a run 40% slower.

**FT06 nominal durations** (machine, duration) in job-processing order:

| Job | Sequence |
|-----|----------|
| J1  | m3:1, m1:3, m2:6, m4:7, m6:3, m5:6 |
| J2  | m2:8, m3:5, m5:10, m6:10, m1:10, m4:4 |
| J3  | m3:5, m4:4, m6:8, m1:9, m2:1, m5:7 |
| J4  | m2:5, m1:5, m3:5, m4:3, m5:8, m6:9 |
| J5  | m3:9, m2:3, m5:5, m6:4, m1:3, m4:1 |
| J6  | m2:3, m4:3, m6:9, m1:10, m5:4, m3:1 |

The deterministic FT06 optimal makespan is 55. The probabilistic version will naturally produce higher expected makespans due to uncertainty.

---

### PJS_09 — Large, high variance, own routing

**Size**: 8 jobs, 6 machines, 48 tasks total  
**Purpose**: A large, independently designed instance with very high variance. Not derived from any classical benchmark.

**Distribution formula** applied to every nominal duration `d`:

```
(max(1, d − 3), 0.35),  (d, 0.30),  (d + 4, 0.35)
```

For small nominal durations this produces a high ratio between slow and fast outcomes. For example, with `d = 3`: fast = `max(1, 0) = 1`, slow = `7`. The slow outcome is 7× the fast outcome.

**Routing** (machine, nominal duration) in job-processing order:

| Job | Sequence |
|-----|----------|
| J1  | m1:3, m2:5, m3:4, m4:7, m5:3, m6:6 |
| J2  | m2:6, m3:4, m4:5, m5:8, m6:3, m1:7 |
| J3  | m3:5, m4:6, m5:3, m6:4, m1:8, m2:5 |
| J4  | m4:4, m5:5, m6:7, m1:3, m2:6, m3:4 |
| J5  | m5:7, m6:3, m1:5, m2:4, m3:6, m4:5 |
| J6  | m6:3, m1:6, m2:4, m5:5, m4:3, m3:7 |
| J7  | m1:5, m3:4, m5:6, m2:3, m6:7, m4:4 |
| J8  | m2:4, m4:3, m6:5, m1:6, m3:4, m5:7 |

The cyclic-offset structure in the routing ensures good machine utilisation while keeping each job's path distinct. The wide duration spread (d-3 to d+4) makes this instance the hardest among the medium-sized ones.

---

### PJS_10 — Stress test, probabilistic FT10 equivalent

**Size**: 10 jobs, 10 machines, 100 tasks total  
**Purpose**: The largest instance in the suite, based on the classical Muth-Thompson FT10 benchmark. The deterministic optimal makespan for FT10 is 930, making it a well-known reference point.

**Distribution formula** (same as PJS_08):

```
(max(1, floor(0.7 × d)), 0.25),  (d, 0.50),  (ceil(1.4 × d), 0.25)
```

**FT10 nominal durations** (machine, duration) in job-processing order:

| Job | Sequence |
|-----|----------|
| J1  | m1:29, m2:78, m3:9,  m4:36, m5:49,  m6:11, m7:62, m8:56, m9:44,  m10:21 |
| J2  | m1:43, m3:90, m2:75, m5:11, m4:69,  m7:28, m8:46, m6:46, m10:72, m9:30  |
| J3  | m2:91, m1:85, m4:39, m6:74, m3:90,  m8:10, m7:12, m9:89, m5:45,  m10:33 |
| J4  | m2:81, m3:95, m1:71, m5:99, m7:9,   m9:52, m8:85, m6:98, m4:22,  m10:43 |
| J5  | m3:14, m1:6,  m2:22, m4:61, m5:26,  m8:69, m10:21,m9:49, m7:72,  m6:53  |
| J6  | m3:84, m2:2,  m1:52, m4:95, m8:48,  m7:72, m9:47, m6:65, m10:6,  m5:25  |
| J7  | m2:46, m1:37, m4:61, m3:13, m7:32,  m6:21, m8:32, m9:89, m10:30, m5:55  |
| J8  | m3:31, m1:86, m2:46, m5:74, m4:32,  m7:88, m6:19, m9:48, m10:36, m8:79  |
| J9  | m1:76, m2:69, m3:76, m4:51, m7:85,  m5:11, m10:40,m8:89, m9:26,  m6:74  |
| J10 | m2:85, m1:13, m3:61, m7:7,  m5:64,  m4:76, m9:47, m10:52,m8:90,  m6:45  |

The large and varied nominal durations (ranging from 2 to 99) mean the expected makespan under probabilistic scheduling is substantially above the deterministic 930.

---

## Difficulty Progression

| Instance | Jobs | Machines | Tasks | Uncertainty Profile | Relative Difficulty |
|----------|------|----------|-------|---------------------|---------------------|
| PJS_01   | 2    | 2        | 4     | Near-deterministic (80% on modal) | Trivial |
| PJS_02   | 2    | 3        | 5     | Moderate symmetric (30–40–30)     | Very easy |
| PJS_03   | 3    | 2        | 6     | High variance, wide spread        | Easy |
| PJS_04   | 3    | 3        | 9     | Moderate symmetric (30–40–30)     | Easy |
| PJS_05   | 4    | 3        | 12    | Bimodal (fast or slow)            | Medium |
| PJS_06   | 5    | 5        | 25    | Moderate symmetric (30–40–30)     | Medium |
| PJS_07   | 5    | 5        | 25    | Asymmetric per machine group      | Medium-hard |
| PJS_08   | 6    | 6        | 36    | Symmetric ±30%/+40% (FT06-based)  | Hard |
| PJS_09   | 8    | 6        | 48    | Very high variance (d±3/+4)       | Hard |
| PJS_10   | 10   | 10       | 100   | Symmetric ±30%/+40% (FT10-based)  | Stress test |

---

## Baseline Strategies

All strategies are **factory functions** that accept an instance and return a strategy function (closure). The closure is called once per dispatch step during simulation.

### Random

```python
strategy_fn = random_strategy(instance, rng=simulator.rng)
```

At each decision point, uniformly picks one enabled task at random. The `rng` parameter accepts a `numpy` random generator so that randomness is tied to the simulator's seed. When `rng=None`, a new independent generator is created.

**Behaviour**: No domain knowledge. Provides a baseline that any intelligent strategy should beat.

### SEPT — Smallest Expected Processing Time

```python
strategy_fn = sept_strategy(instance)
```

Among all currently enabled tasks, starts the one with the smallest expected duration `E[d] = Σ p_i × d_i`.

**Intuition**: Finish short tasks first to free up machines quickly. Well-known to minimise mean flow time in single-machine settings. Generally performs well on instances where task durations are correlated with machine utilisation.

### LEPT — Largest Expected Processing Time

```python
strategy_fn = lept_strategy(instance)
```

The opposite of SEPT: always starts the task with the largest expected duration.

**Intuition**: Start long tasks early so they are not blocking machines late in the schedule. Sometimes competitive with SEPT when jobs have long critical paths. Usually performs worse on high-variance instances.

### MWR — Most Work Remaining

```python
strategy_fn = mwr_strategy(instance)
```

Among enabled tasks, starts the one belonging to the job with the most remaining expected work (sum of expected durations of all unfinished tasks in that job, including active ones).

**Intuition**: Prioritise jobs that are furthest from completion. This is a critical-path heuristic: keeping the busiest job moving reduces the risk of it becoming a bottleneck at the end. Tends to be the most competitive baseline on larger instances.

### FIFO — First In, First Out

```python
strategy_fn = fifo_strategy(instance)
```

Uses a fixed priority order based on job index: J1 is always preferred over J2, J2 over J3, and so on. Among enabled tasks, starts the one belonging to the earliest-indexed job.

**Intuition**: No dynamic reasoning — a static priority rule. Serves as a reference for "how well does ignoring the current state do?" Performance depends entirely on whether the instance's job ordering happens to align with a good schedule.

---

## Running the Benchmarks

### Requirements

```bash
pip install -r requirements.txt
```

Requirements: `numpy>=1.24`, `matplotlib>=3.7`, `pandas>=2.0`, `tqdm>=4.65`.

### Full benchmark run

```bash
python run_benchmarks.py
```

Runs all 10 instances × 5 strategies × 1000 episodes. On a typical laptop this takes a few minutes. Progress bars are shown via `tqdm`.

### Quick single-instance test

```python
from prob_jobshop.benchmarks.all_instances import get_instance_by_name
from prob_jobshop.simulator import PTASimulator
from prob_jobshop.strategies import mwr_strategy

instance = get_instance_by_name("PJS_05")
sim = PTASimulator(instance, seed=42)
results = sim.evaluate_strategy(mwr_strategy(instance), n_episodes=100)
print(results["mean"], results["std"])
```

---

## Output Files

After `python run_benchmarks.py` all output is written to `./output/`:

| File | Contents |
|------|----------|
| `summary.csv` | One row per instance: jobs, machines, tasks, LB, UB, mean ± std per strategy |
| `summary_table.png` | Visual table of means and standard deviations |
| `{NAME}_histograms.png` | Overlapping makespan histograms for all 5 strategies on one instance |
| `{NAME}_gantt_{strategy}.png` | Gantt chart of a single recorded episode per strategy |

---

## Using the API

### Load all instances

```python
from prob_jobshop.benchmarks.all_instances import get_all_instances, get_instance_by_name

instances = get_all_instances()           # list of all 10
inst = get_instance_by_name("PJS_08")    # single instance by name
```

### Inspect an instance

```python
print(inst.name, inst.description)
print("Jobs:", len(inst.jobs), "  Machines:", len(inst.machines))
print("LB:", inst.expected_makespan_lower_bound())
print("UB:", inst.worst_case_makespan_upper_bound())

for job in inst.jobs:
    for task in job.tasks:
        d = task.distribution
        print(task.task_id, task.machine, d.durations, d.probabilities, "E=", d.expected_duration())
```

### Run a single episode and record the trace

```python
from prob_jobshop.simulator import PTASimulator
from prob_jobshop.strategies import sept_strategy

sim = PTASimulator(inst, seed=0)
makespan, trace = sim.run_episode(sept_strategy(inst), record_trace=True)
print("Makespan:", makespan, "  Trace length:", len(trace))
```

### Evaluate a strategy over many episodes

```python
results = sim.evaluate_strategy(sept_strategy(inst), n_episodes=1000)
# results keys: "mean", "std", "min", "max", "p10", "p90", "p95", "raw"
print(f"mean={results['mean']:.2f}  std={results['std']:.2f}")
```

### Implement a custom strategy

A strategy is any callable `(GlobalState) -> Optional[str]`. It must return a task ID to start, or `None`.

```python
def my_strategy_factory(instance):
    def _strategy(state):
        enabled = state.enabled_tasks(instance)
        if not enabled:
            return None
        # Pick the task on the machine that currently has the least active load.
        # (illustrative — not optimal)
        return enabled[0]
    return _strategy

results = sim.evaluate_strategy(my_strategy_factory(inst), n_episodes=500)
```

### Verify an instance and a simulation trace

```python
from prob_jobshop.verification import verify_instance, verify_simulation

errors = verify_instance(inst)
assert errors == [], errors

makespan, trace = sim.run_episode(sept_strategy(inst), record_trace=True)
sim_errors = verify_simulation(inst, trace)
assert sim_errors == [], sim_errors
```



Strategy	Rule
random	Pick any enabled task uniformly at random
SEPT	Shortest Expected Processing Time — dispatch the task with the smallest E[duration]
LEPT	Longest Expected Processing Time — the opposite; tries to get long tasks out of the way early
MWR	Most Work Remaining — dispatch the task whose job has the most expected work left; tries to keep the critical-path job moving
FIFO	Fixed priority by job index — always prefer Job 1 over Job 2 over Job 3, etc.