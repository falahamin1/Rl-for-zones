"""6×6 Frozen Lake grid definition.

Layout (row 0 = top, col 0 = left):

    S  .  .  .  .  .
    .  .  H  .  .  .
    .  .  .  .  H  .
    .  H  .  .  .  .
    .  .  .  H  .  .
    .  .  .  .  .  G

S = start (0,0), G = goal (5,5), H = hole, . = frozen (safe)
"""
from enum import Enum
from typing import FrozenSet, Tuple


class Cell(Enum):
    FROZEN = "F"
    HOLE   = "H"
    START  = "S"
    GOAL   = "G"


ROWS: int = 6
COLS: int = 6

START: Tuple[int, int] = (0, 0)
GOAL:  Tuple[int, int] = (5, 5)

HOLES: FrozenSet[Tuple[int, int]] = frozenset({
    (1, 2),
    (2, 4),
    (3, 1),
    (4, 3),
})

# Full grid for display / lookup
GRID = [
    [Cell.START,  Cell.FROZEN, Cell.FROZEN, Cell.FROZEN, Cell.FROZEN, Cell.FROZEN],
    [Cell.FROZEN, Cell.FROZEN, Cell.HOLE,   Cell.FROZEN, Cell.FROZEN, Cell.FROZEN],
    [Cell.FROZEN, Cell.FROZEN, Cell.FROZEN, Cell.FROZEN, Cell.HOLE,   Cell.FROZEN],
    [Cell.FROZEN, Cell.HOLE,   Cell.FROZEN, Cell.FROZEN, Cell.FROZEN, Cell.FROZEN],
    [Cell.FROZEN, Cell.FROZEN, Cell.FROZEN, Cell.HOLE,   Cell.FROZEN, Cell.FROZEN],
    [Cell.FROZEN, Cell.FROZEN, Cell.FROZEN, Cell.FROZEN, Cell.FROZEN, Cell.GOAL  ],
]


def cell_at(row: int, col: int) -> Cell:
    return GRID[row][col]


def is_valid(row: int, col: int) -> bool:
    return 0 <= row < ROWS and 0 <= col < COLS


def render() -> str:
    symbols = {Cell.FROZEN: ".", Cell.HOLE: "H", Cell.START: "S", Cell.GOAL: "G"}
    lines = []
    for row in GRID:
        lines.append("  ".join(symbols[c] for c in row))
    return "\n".join(lines)
