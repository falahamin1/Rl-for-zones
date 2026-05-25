# Symbolic RL for Probabilistic Job-Shop Scheduling

## What this is

Standard RL samples one concrete episode at a time — one set of task durations is drawn, the agent dispatches tasks, and the final makespan is observed. The agent must run thousands of episodes to average out the randomness.

**Symbolic RL** takes a different approach: instead of sampling durations at training time, the agent works with *zones* — sets of clock valuations stored as Difference Bound Matrices (DBMs). Each dispatch action expands into **K branches** (one per possible task duration), and the agent's Q-value is a **probability-weighted expected makespan**:

```
Q[s][dispatch T] = Σ_d P(T=d) × min_cost(Z_d)
```

where `Z_d` is the zone when task T takes exactly d time units. With exact zones, `min_cost(Z_d)` equals the exact makespan contribution for that duration outcome, so `Q[s][a] = E[makespan contribution]` — the agent implicitly learns the probability distribution.

---

## How to run

```bash
# From the scheduling/ directory
python run_symbolic_rl.py
```

Outputs written to `output/`:
- `symbolic_rl_summary.csv` — mean/std makespan table for all instances
- `symbolic_rl_summary.png` — visual summary table
- `PJS_XX_symbolic_learning.png` — learning curve per instance

**Configuration** (top of `run_symbolic_rl.py`):
```python
INSTANCES  = ["PJS_01", "PJS_02", "PJS_03", "PJS_04", "PJS_05"]
TRAIN_EPS  = 2000   # symbolic episodes for training
EVAL_EPS   = 100    # real simulator episodes for evaluation
SEED       = 42
```

---

## Background: clocks and zones

The job-shop is modelled as a **Uniform-rate Probabilistic Timed Automaton (UPTA)**:

- There is one **clock per job** (`c_J1`, `c_J2`, …). Each clock is reset to 0 when a task starts on that job, and counts up at rate 1.
- There is one **cost clock** `delta`. It is never reset and counts up at rate 1 everywhere. Its value at the end of the schedule equals the makespan.
- A **zone** is a convex set of clock valuations, stored as a DBM: it records all pairwise difference constraints between clocks, e.g. `c_J1 - c_J2 ≤ 3`.
- **`min_cost(Z)`** = infimum of `delta` in zone `Z` = the earliest possible makespan over all clock valuations in `Z`.

Zone operations used:

| Operation | Effect |
|-----------|--------|
| `init_zone` | Start at the zero zone (all clocks = 0), intersect with invariant |
| `apply_delay` | Let time pass: `Z.up()` (all clocks grow), then intersect with invariant |
| `reset_clocks` | Set a job clock to 0 (task dispatch) |
| `apply_guard` | Intersect with an upper-bound constraint `c ≤ k` |
| `Z & (c >= k)` | Intersect with a lower-bound constraint (task completion guard) |
| `normalize` | k-normalise by the max constant `M` — ensures the zone graph is finite |
| `dominates(A, B)` | True if `B ⊆ A` (A is at least as good; no need to re-explore B) |

---

## Key insight: distinct states per duration

Each task's duration is sampled from a discrete probability distribution (e.g., d ∈ {3, 4, 5} with probabilities {0.1, 0.8, 0.1}). The PTA model has **distinct active states per duration**:

```
"active_d3" — task is running with sampled duration 3
"active_d4" — task is running with sampled duration 4
"active_d5" — task is running with sampled duration 5
```

This is critical: with a single merged `"active"` state, the zone uses `c_J ≤ max_dur` as the invariant and `c_J ≥ min_dur` as the completion guard — losing all probability information. With distinct states, the invariant becomes `c_J ≤ d` (exact), so the completion guard `c_J ≥ d` gives **`c_J = d` exactly** in the zone.

The zone for duration d forces `delta_inf = exact makespan contribution for that execution`. The probability-weighted Q-value therefore equals `E[makespan]` — the agent minimizes expected makespan.

---

## Pseudocode

### Episode loop

```
INPUTS:
  env    — symbolic environment (location tuples + branched actions)
  zctx   — zone context (pyudbm clocks)
  M      — normalization constant (worst-case makespan)
  Q      — Q-table: (location, zone_str) → {action → value}
  V_rem  — remaining-cost table: (location, zone_str) → value

─────────────────────────────────────────────────────────────────────────
loc, actions = env.reset()   # location = all "waiting"

Z = zero_zone ∩ invariant(loc)
Z = delay(Z, invariant(loc))          # up() then intersect invariant
Z = normalize(Z, M)

LOOP:
    key = (loc, str(Z))               # symbolic state identity

    FOR each enabled action a (dispatch task T):
        branches = []
        FOR each (d, p) in T.distribution:
            Z_branch = Z  (after guard + reset c_J = 0)
            Z_branch = delay(Z_branch, inv=[c_J ≤ d])      # exact upper bound
            FOR each completion step in path[d]:
                Z_branch = Z_branch ∩ {c_J_completing ≥ exact_d}  # → c_J = exact_d
                Z_branch = delay(Z_branch, inv_at_next_loc)
            Z_branch = normalize(Z_branch, M)
            branches.append((d, p, Z_branch, next_key))

        Q[key][a.id] = Σ p × (min_cost(Z_branch) + V_rem.get(next_key, 0))

    ── ε-greedy action selection ──
    chosen = action with smallest Q[key][a.id]  (or random with prob ε)

    ── Execute: sample one concrete duration ──
    loc_new, actions_new, done, info = env.step(chosen.id)
    sampled_d = info["sampled_d"]
    Z_nx, next_key = branch for sampled_d

    ── Dominance + Bellman ──
    Passed[next_key] = best(Passed[next_key], Z_nx)
    V_rem[key] = min(V_rem[key], 0 + V_rem.get(next_key, 0))

    IF done:
        V_rem[next_key] = 0
        RETURN min_cost(Z_nx)
    Z, loc = Z_nx, loc_new
─────────────────────────────────────────────────────────────────────────
```

### What `action["branches"]` is

Each action has one branch per possible task duration:

```python
action = {
  "action_id": "J1_T1",
  "resets":    ["c_J1"],
  "branches": [
    {"d": 3, "prob": 0.1, "next_loc": (...,"active_d3",...), "path": [...], "decision_loc": ...},
    {"d": 4, "prob": 0.8, "next_loc": (...,"active_d4",...), "path": [...], "decision_loc": ...},
    {"d": 5, "prob": 0.1, "next_loc": (...,"active_d5",...), "path": [...], "decision_loc": ...},
  ]
}
```

The `path` records which tasks complete (with exact guards) between the dispatch and the next decision point, for that specific duration d.

---

## Evaluation: two modes

### 1. Symbolic evaluation (`agent.evaluate()`)
Runs the greedy policy. The environment samples concrete durations, and the zone tracks the exact makespan for those durations. Returns `min_cost(Z)` per episode — the **actual makespan** for that sampled trajectory. With exact zones this is no longer just a lower bound.

### 2. Real-episode evaluation (`agent.evaluate_real()`)
Uses the Q-table to build a dispatch policy, normalizing `"active_d{d}"` → `"active"` in location keys to match the simulator's state representation. Falls back to SEPT for unseen locations.

This gives a mean makespan directly comparable to the baselines.

---

## Results on PJS_01–PJS_05

| Instance | random | sept | lept | mwr | fifo | **sym_rl** | Zone States |
|----------|--------|------|------|-----|------|------------|-------------|
| PJS_01 | 7.03 | 7.01 | 7.05 | 7.18 | 7.10 | **7.00** | 58 |
| PJS_02 | 10.71 | 11.07 | 10.96 | 10.92 | 10.67 | 10.79 | 174 |
| PJS_03 | 16.60 | 15.78 | 17.96 | 15.64 | 16.75 | **15.55** | 1009 |
| PJS_04 | 12.44 | 12.35 | 12.34 | 12.29 | 12.28 | 12.50 | 2857 |
| PJS_05 | 24.09 | 22.86 | 25.20 | 22.15 | 24.50 | 23.84 | 10189 |

Symbolic RL wins on PJS_01 and PJS_03 (beating all baselines). PJS_04 and PJS_05 have large state spaces that benefit from more training episodes.

**Zone States** = number of (location, zone) pairs explored during training. With exact per-duration states, counts are higher than the old merged design (was 8–242).

---

## File map

```
prob_jobshop/
  symbolic_env.py   — UPTA interface: "active_d{d}" locations + branched actions
  zone_ops.py       — Zone operations wrapping pyudbm (init, delay, guard, normalize, …)
  symbolic_rl.py    — SymbolicRLAgent: probability-weighted Q over (location, zone) pairs
run_symbolic_rl.py  — Runner: trains on PJS_01–05, evaluates vs baselines, saves plots
SYMBOLIC_RL.md      — This file
```

---

## Dependencies

```
pyudbm    — Python bindings for UPPAAL UDBM (zone/DBM library)
gymnasium — Standard RL environment interface
```

Install: `pip install pyudbm gymnasium`
