"""FrozenLakeEnv — base gridworld without timed-automaton spec.

State  : (row, col)  — position on the 6×6 grid
Actions: 0=UP  1=DOWN  2=LEFT  3=RIGHT

Transitions are deterministic (no slip) by default; a slip probability
can be supplied to make the ice stochastic.  Timed-automaton constraints
will be layered on top in a subclass once the specification is decided.

Terminal conditions
    • step onto a HOLE  → done, reward -1
    • step onto GOAL    → done, reward +1
    • all other steps   → reward 0
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .grid import Cell, COLS, GOAL, HOLES, ROWS, START, cell_at, is_valid

# Action constants
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


class FrozenLakeEnv:
    """Base deterministic (or slip-stochastic) 6×6 Frozen Lake.

    Parameters
    ----------
    slip_prob   probability of sliding 90° from the intended direction
                (0.0 = fully deterministic)
    seed        RNG seed for stochastic slip
    """

    def __init__(self, slip_prob: float = 0.0, seed: Optional[int] = None):
        self.slip_prob = slip_prob
        self._rng = np.random.default_rng(seed)
        self._pos: Tuple[int, int] = START

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[int, int]:
        self._pos = START
        return self._pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Execute action; return (new_state, reward, done)."""
        actual = self._sample_action(action)
        dr, dc = _DELTAS[actual]
        r, c = self._pos
        nr, nc = r + dr, c + dc

        if not is_valid(nr, nc):
            nr, nc = r, c          # bump into wall — stay

        self._pos = (nr, nc)
        cell = cell_at(nr, nc)

        if cell == Cell.HOLE:
            return self._pos, -1.0, True
        if cell == Cell.GOAL:
            return self._pos, +1.0, True
        return self._pos, 0.0, False

    def state(self) -> Tuple[int, int]:
        return self._pos

    def action_space(self) -> List[int]:
        return ACTIONS

    # ------------------------------------------------------------------

    def _sample_action(self, intended: int) -> int:
        if self.slip_prob == 0.0 or self._rng.random() >= self.slip_prob:
            return intended
        # Slip 90° either way
        perp = [LEFT, RIGHT] if intended in (UP, DOWN) else [UP, DOWN]
        return self._rng.choice(perp)
