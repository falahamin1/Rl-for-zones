# TRM Frozen Lake — Benchmark Suite

Zone-graph vs region-based Q-learning on a stochastic grid with timed reward
machines. The benchmark demonstrates that precomputing the **zone graph** from a
known TRM yields a **constant** clock-state footprint regardless of the guard
constants, while the region representation grows linearly.

---

## Environment — 6×6 Frozen Lake

```
S  .  .  .  .  .
.  .  .  .  .  .
.  H  .  .  H  .
.  .  .  A  .  .
.  .  H  .  .  .
B  H  .  .  .  C
```

| Symbol | Position | Meaning |
|--------|----------|---------|
| S | (0,0) | Start |
| A | (3,3) | Checkpoint A |
| B | (5,0) | Checkpoint B |
| C | (5,5) | Final goal C |
| H | (2,1),(2,4),(4,2),(5,1) | Holes — episode ends with penalty |
| . | — | Safe frozen tile |

The agent must visit **A → B → C in order**. Each action moves one cell
(up/down/left/right); with slip probability 0.2 the agent slides 90° instead.
Boundaries are reflective (agent stays in place if it would leave the grid).

---

## Timed Reward Machine

### States

| State | Role |
|-------|------|
| u1 | Seeking checkpoint A |
| u2 | Seeking checkpoint B |
| u3 | Seeking checkpoint C |
| u0 | Terminal (success or failure) |

### Clocks

| Clock | Behaviour | Purpose |
|-------|-----------|---------|
| x | Resets to 0 at each successful goal visit | Per-phase reaction timer |
| y | Never resets | Global elapsed timer |

### Guard

Every checkpoint transition (A, B, C) checks the same guard:

```
x >= c_min   AND   y <= D
```

**`x >= c_min`** — each phase must accumulate at least `c_min` time steps before
the goal can be visited. Clock `x` resets to 0 on each successful visit, so the
agent must wait at least `c_min` steps after the previous checkpoint.

**`y <= D`** — total elapsed time must not exceed the global deadline `D`.

### Rewards

| Event | Reward | Condition |
|-------|--------|-----------|
| Step with no progress | −1 | always |
| Reach A or B | +100 | guard satisfied |
| Reach C (final) | +500 | guard satisfied |
| Reach any checkpoint early/late | −200 | guard failed |
| Fall in hole | −500 | always |

### Transitions

```
u1 --[event=a, x>=c_min & y<=D]--> u2   reward +100  (x resets)
u1 --[event=a, guard fails    ]--> u0   reward −200
u1 --[event=h                 ]--> u0   reward −500
u1 --[else                    ]--> u1   reward −1

u2 --[event=b, x>=c_min & y<=D]--> u3   reward +100  (x resets)
u2 --[event=b, guard fails    ]--> u0   reward −200
u2 --[event=h                 ]--> u0   reward −500
u2 --[else                    ]--> u2   reward −1

u3 --[event=c, x>=c_min & y<=D]--> u0   reward +500  (terminal success)
u3 --[event=c, guard fails    ]--> u0   reward −200
u3 --[event=h                 ]--> u0   reward −500
u3 --[else                    ]--> u3   reward −1
```

---

## Benchmark Instances

All instances fix `D=200` and `d_max=5`. The minimum reaction time `c_min` grows
across instances, which is the variable that drives the zone vs region comparison.

| Instance | c_min | D | d_max | Region clock pairs (c_min+1)×(D+1) |
|----------|-------|---|-------|--------------------------------------|
| TRM_01   | 2     | 200 | 5 | 3 × 201 = 603                        |
| TRM_02   | 5     | 200 | 5 | 6 × 201 = 1,206                      |
| TRM_03   | 10    | 200 | 5 | 11 × 201 = 2,211                     |
| TRM_04   | 15    | 200 | 5 | 16 × 201 = 3,216                     |
| TRM_05   | 20    | 200 | 5 | 21 × 201 = 4,221                     |
| TRM_06   | 25    | 200 | 5 | 26 × 201 = 5,226                     |

Zone clock-state nodes: exactly **4** for every instance.

---

## State Abstractions

### Zone (precomputed zone graph)

Since the TRM is fully known before training, the zone graph is precomputed
offline and its nodes are used directly as Q-learning states.

For guard `x >= c_min AND y <= D`, the zone graph has exactly **4 clock-state nodes**,
independent of the values of `c_min` and `D`:

```
x_node = 0  if x < c_min   (pre-guard: reaction time not yet accumulated)
          1  if x >= c_min  (guard-satisfiable: can visit next checkpoint)

y_node = 0  if y <= D       (global deadline not exceeded)
          1  if y > D        (deadline exceeded — guard can never be satisfied)
```

State key: `(grid_pos, rm_state, x_node, y_node)`

The zone graph structure depends only on the **number of guard thresholds** (2 here),
not on the threshold values. Any `c_min` and `D` produce the same 4-node zone graph.

### Region (integer-clock abstraction)

```
state key = (grid_pos, rm_state, min(x, c_min), min(y, D))
```

Caps each clock at its guard threshold independently. One distinct class per
integer x in `[0, c_min]` and per integer y in `[0, D]`. State count grows as
`O(c_min × D)` — linearly with `c_min` when `D` is fixed.

### Why zone is structurally smaller

From zone node `{x < c_min}`, the agent learns **one** Q-value table entry for the
entire pre-reaction phase. With d_max=5, any action that increments x by enough
(delay ≥ c_min − x − 1) exits this zone. The optimal policy from this zone is
uniform: take maximum delay to accumulate reaction time as fast as possible.

Region must distinguish x = 0, 1, …, c_min−1 as separate states and learn a
separate Q-entry for each. As c_min grows, region accumulates c_min extra states
per (grid_pos, rm_state) pair while zone always has exactly one.

---

## BFS Graph Sizes

Empirical BFS counts from the benchmark (reachable state-graph nodes):

| Instance | c_min | Zone BFS nodes | Region BFS nodes |
|----------|-------|---------------|-----------------|
| TRM_01   | 2     | 385           | 19,439          |
| TRM_02   | 5     | 385           | 33,845          |
| TRM_03   | 10    | 385           | 81,077          |
| TRM_04   | 15    | 385           | 131,152         |
| TRM_05   | 20    | 385           | 174,827         |
| TRM_06   | 25    | 385           | 212,102         |

Zone BFS: **constant 385** across all instances. Region BFS grows 11× from
TRM_01 to TRM_06.

The 385 zone nodes are `(grid_pos, rm_state, x_node, y_node)` combinations —
36 grid positions × 4 RM states × 4 clock nodes, minus unreachable combinations
(e.g., terminal RM state with non-trivial clock nodes).

---

## Q-learning Setup

| Parameter | Value |
|-----------|-------|
| α (learning rate) | 0.1 |
| γ (discount) | 0.99 |
| ε (exploration) | 0.9 → 0.05 (linear decay) |
| Training episodes | 150,000 |
| Max steps per episode | 400 |
| d_max (delay) | 5 (dt = delay+1 per action) |
| slip_prob | 0.2 |
| Seed | 42 |

Action space: `(delay, grid_action)` where `delay ∈ {0,...,5}` and
`grid_action ∈ {up, down, left, right}` → 24 actions per state.

Q-update:

```
Q[s][a] += α × (r + γ^(delay+1) × max_{a'} Q[s'][a'] − Q[s][a])
```

The `γ^(delay+1)` discount accounts for variable action duration (higher delay
means the action spans more time units, warranting heavier discounting).

Updates only during training (ε-greedy exploration); evaluation runs are
fully greedy with no Q-table modifications.

---

## Output

Running `python run_trm_comparison.py` from the `scheduling/` directory produces:

```
output/trm/TRM_0X_convergence.png   — per-instance learning curve
output/trm/comparison_summary.csv   — full results table
output/trm/comparison_summary.png   — summary table figure
```
