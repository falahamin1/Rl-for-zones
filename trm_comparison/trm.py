"""Timed Reward Machine (TRM) for the 3-goal Frozen Lake task.

RM structure
------------
States: u0 (terminal), u1 (seek A), u2 (seek B), u3 (seek C)

Guard: x >= c_min  AND  y <= D
  x >= c_min : each phase must accumulate at least c_min time before the goal
               can be visited  (x resets to 0 at each successful goal visit)
  y <= D     : total elapsed time must not exceed the global deadline D

Transitions (u_src, event, guard, resets, reward, u_dst):
  u1 --[a, guard ok ]--> u2   +100  (x resets)
  u1 --[a, guard bad]--> u0   -200
  u1 --[h           ]--> u0   -500
  u1 --[else        ]--> u1   -1

  u2 --[b, guard ok ]--> u3   +100  (x resets)
  u2 --[b, guard bad]--> u0   -200
  ... (same pattern for h / else)

  u3 --[c, guard ok ]--> u0   +500  (terminal success)
  u3 --[c, guard bad]--> u0   -200
  ...

Clocks
------
  x   resets at each successful goal visit  — per-phase timer
  y   never resets                          — global timer

Zone graph vs Region graph
--------------------------
Because the TRM is fully known before training, we build its zone graph offline
and use zone graph nodes as Q-learning states.

For guard  x >= c_min AND y <= D  the zone graph has exactly 4 clock-state nodes:
  x-node: 0 if x < c_min  (pre-guard phase)
           1 if x >= c_min (guard satisfiable)
  y-node: 0 if y <= D     (budget ok)
           1 if y > D      (budget exceeded)

These 4 nodes are INDEPENDENT of the constants c_min, D, d_max — the zone graph
structure depends only on the number of guard thresholds.

Region, tracking exact integer clocks, has (c_min+1)×(D+1) clock-state pairs:
one class per integer x in [0, c_min] and one per integer y in [0, D].  As c_min
grows, region accumulates more pre-guard x-states while zone always has exactly
one ({x < c_min}).

Benchmark instances — fix D=200, d_max=5; vary c_min
----------------------------------------------------
  TRM_01  c_min= 2  region clock pairs:  3×201=   603  zone: 4
  TRM_02  c_min= 5  region clock pairs:  6×201= 1,206  zone: 4
  TRM_03  c_min=10  region clock pairs: 11×201= 2,211  zone: 4
  TRM_04  c_min=15  region clock pairs: 16×201= 3,216  zone: 4
  TRM_05  c_min=20  region clock pairs: 21×201= 4,221  zone: 4
  TRM_06  c_min=25  region clock pairs: 26×201= 5,226  zone: 4
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple


# ─── RM state constants ─────────────────────────────────────────────────────
U_TERMINAL = 0
U1         = 1   # seeking A
U2         = 2   # seeking B
U3         = 3   # seeking C

NONTERMINAL_STATES = (U1, U2, U3)
RM_GOAL_EVENTS = {U1: "a", U2: "b", U3: "c"}

# Rewards
R_STEP  = -1
R_GOAL  = 100
R_FINAL = 500
R_LATE  = -200
R_HOLE  = -500


@dataclass
class TRMInstance:
    """A single benchmark instance of the 3-goal TRM."""
    name:  str
    c_min: int   # lower-bound guard on x   (x >= c_min to advance RM)
    D:     int   # upper-bound guard on y   (y <= D to advance RM)
    d_max: int   # maximum action delay     (action space: {0,...,d_max} × 4)

    @property
    def cy(self) -> int:
        """Episode-time cap (= global deadline D for both agents)."""
        return self.D

    @property
    def n_rm_states(self) -> int:
        return 4

    @property
    def clock_names(self):
        return ["x", "y"]

    def guard_satisfied(self, clocks: Dict[str, int]) -> bool:
        return clocks["x"] >= self.c_min and clocks["y"] <= self.D


# Standard benchmark suite — D=200, d_max=5 fixed; c_min varies
INSTANCES: Dict[str, TRMInstance] = {
    "TRM_01": TRMInstance(name="TRM_01", c_min=2,  D=200, d_max=5),
    "TRM_02": TRMInstance(name="TRM_02", c_min=5,  D=200, d_max=5),
    "TRM_03": TRMInstance(name="TRM_03", c_min=10, D=200, d_max=5),
    "TRM_04": TRMInstance(name="TRM_04", c_min=15, D=200, d_max=5),
    "TRM_05": TRMInstance(name="TRM_05", c_min=20, D=200, d_max=5),
    "TRM_06": TRMInstance(name="TRM_06", c_min=25, D=200, d_max=5),
}


# ─── TRM step function ───────────────────────────────────────────────────────

class TRMState:
    """Mutable TRM execution state (rm_state + clock values)."""

    def __init__(self):
        self.u: int = U1
        self.clocks: Dict[str, int] = {"x": 0, "y": 0}

    def reset(self):
        self.u = U1
        self.clocks = {"x": 0, "y": 0}

    def advance_clocks(self, dt: int = 1):
        self.clocks["x"] += dt
        self.clocks["y"] += dt

    def step(self, event: str, trm: TRMInstance) -> Tuple[int, bool]:
        """Process one TRM step; return (reward, done).

        event : "" | "a" | "b" | "c" | "h"
        """
        u = self.u
        assert u != U_TERMINAL

        if event == "h":
            self.u = U_TERMINAL
            return R_HOLE, True

        goal_event = RM_GOAL_EVENTS.get(u)

        if event == goal_event:
            if trm.guard_satisfied(self.clocks):
                self.clocks["x"] = 0
                if u == U3:
                    self.u = U_TERMINAL
                    return R_FINAL, True
                else:
                    self.u = u + 1
                    return R_GOAL, False
            else:
                self.u = U_TERMINAL
                return R_LATE, True

        return R_STEP, False
