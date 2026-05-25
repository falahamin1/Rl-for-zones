"""6×6 Frozen Lake grid with three checkpoints A, B, C.

Layout (row 0 = top, col 0 = left):

    S  .  .  .  .  .
    .  .  .  .  .  .
    .  H  .  .  H  .
    .  .  .  A  .  .
    .  .  H  .  .  .
    B  H  .  .  .  C

S = start (0,0)
A = checkpoint A (3,3)
B = checkpoint B (5,0)
C = checkpoint C (5,5)  ← final goal
H = hole (episode fails)
. = frozen (safe)

The agent starts at S and must visit A then B then C in that order
to maximise reward. Falls into a hole → episode ends with penalty.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np

# Grid dimensions
ROWS: int = 6
COLS: int = 6

# Special positions
START: Tuple[int, int] = (0, 0)
GOAL_A: Tuple[int, int] = (3, 3)
GOAL_B: Tuple[int, int] = (5, 0)
GOAL_C: Tuple[int, int] = (5, 5)

HOLES = frozenset({(2, 1), (2, 4), (4, 2), (5, 1)})
CHECKPOINTS = {GOAL_A: "a", GOAL_B: "b", GOAL_C: "c"}

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


class GridEnv:
    """6×6 stochastic frozen lake with checkpoints A, B, C.

    Parameters
    ----------
    slip_prob   Probability of sliding 90° (perpendicular to intended direction).
                0 = fully deterministic.
    seed        RNG seed.
    """

    def __init__(self, slip_prob: float = 0.2, seed: Optional[int] = None):
        self.slip_prob = slip_prob
        self._rng = np.random.default_rng(seed)
        self._pos: Tuple[int, int] = START

    # ------------------------------------------------------------------ #

    def reset(self) -> Tuple[int, int]:
        self._pos = START
        return self._pos

    def step(self, action: int) -> Tuple[Tuple[int, int], str, bool]:
        """Execute action; return (new_pos, event, done).

        event : "a" | "b" | "c" | "h" | ""
        done  : True if hole or final checkpoint C reached.
        """
        actual = self._sample_action(action)
        dr, dc = _DELTAS[actual]
        r, c = self._pos
        nr, nc = max(0, min(ROWS - 1, r + dr)), max(0, min(COLS - 1, c + dc))
        self._pos = (nr, nc)

        if (nr, nc) in HOLES:
            return self._pos, "h", True
        event = CHECKPOINTS.get((nr, nc), "")
        done = event == "c"
        return self._pos, event, done

    def pos(self) -> Tuple[int, int]:
        return self._pos

    def action_space(self) -> List[int]:
        return ACTIONS

    # ------------------------------------------------------------------ #

    def _sample_action(self, intended: int) -> int:
        if self.slip_prob <= 0.0 or self._rng.random() >= self.slip_prob:
            return intended
        perp = [LEFT, RIGHT] if intended in (UP, DOWN) else [UP, DOWN]
        return int(self._rng.choice(perp))

    def render(self) -> str:
        sym = {(r, c): "." for r in range(ROWS) for c in range(COLS)}
        sym[START] = "S"
        sym[GOAL_A] = "A"
        sym[GOAL_B] = "B"
        sym[GOAL_C] = "C"
        for h in HOLES:
            sym[h] = "H"
        sym[self._pos] = "@"
        rows = []
        for r in range(ROWS):
            rows.append("  ".join(sym[(r, c)] for c in range(COLS)))
        return "\n".join(rows)
