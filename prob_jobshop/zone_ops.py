"""Zone operations for symbolic RL using pyudbm.

Clocks: one per job (c_J1, c_J2, ...) + cost clock delta.
Reference clock (index 0) is implicit in pyudbm.

Invariant never constrains delta, so:
- apply_dagger is a no-op (delta stays unbounded above after up())
- increment(Z, 0) is a no-op (all job-shop dispatch prices = 0)

min_cost(Z) = infimum of delta in Z = minimum reachable makespan.
"""
from __future__ import annotations
from typing import List, Optional, Tuple, TYPE_CHECKING

from pyudbm import Context

if TYPE_CHECKING:
    from pyudbm.binding.udbm import Federation
    from .instance import ProbJobShopInstance

# UDBM's internal infinity sentinel (2^30 - 1)
_UDBM_INF = 1073741823


class ZoneContext:
    """Wraps pyudbm Context for a given job-shop instance."""

    def __init__(self, instance: "ProbJobShopInstance"):
        self._clock_names = [f"c_{j.job_id}" for j in instance.jobs] + ["delta"]
        self.ctx = Context(self._clock_names)
        # Map name → Clock object
        self.clocks = {name: getattr(self.ctx, name) for name in self._clock_names}
        self.delta = self.clocks["delta"]
        # Index of delta in UDBM (ref=0, user clocks start at 1)
        self._delta_idx = len(self._clock_names)  # last user clock

    @property
    def clock_names(self) -> List[str]:
        return self._clock_names


# ---------------------------------------------------------------------------
# Zone operations — all return a new Federation, never mutate in place
# ---------------------------------------------------------------------------

def init_zone(zctx: ZoneContext, inv_pairs: List[Tuple[str, int]]) -> "Federation":
    """Z†_0: zero zone intersected with initial invariant."""
    Z = zctx.ctx.get_zero_federation()
    return apply_invariant(Z, zctx, inv_pairs)


def apply_guard(
    Z: "Federation",
    zctx: ZoneContext,
    guards: List[Tuple[str, int]],
) -> Optional["Federation"]:
    """Intersect zone with guard constraints.

    guards: list of (clock_name, upper_bound) representing c <= bound.
    Returns None if the intersection is empty (guard unsatisfiable).
    """
    Z_new = Z
    for cname, bound in guards:
        Z_new = Z_new & (zctx.clocks[cname] <= bound)
    if Z_new.is_empty():
        return None
    return Z_new


def reset_clocks(
    Z: "Federation",
    zctx: ZoneContext,
    clock_names: List[str],
) -> "Federation":
    """Reset named clocks to 0.  delta must never be in clock_names."""
    assert "delta" not in clock_names, "delta must never be reset"
    Z_new = Z
    for cname in clock_names:
        Z_new = Z_new.reset_value(zctx.clocks[cname])
    return Z_new


def apply_delay(
    Z: "Federation",
    zctx: ZoneContext,
    inv_pairs: List[Tuple[str, int]],
    rate: int,
) -> "Federation":
    """Delay operator: up() followed by invariant intersection.

    rate is unused for job-shop (always 1, cost tracked by delta growing
    with up()), but included for generality.
    Invariant never mentions delta, so delta remains unbounded above (†
    property holds automatically).
    """
    Z_new = Z.up()
    return apply_invariant(Z_new, zctx, inv_pairs)


def apply_invariant(
    Z: "Federation",
    zctx: ZoneContext,
    inv_pairs: List[Tuple[str, int]],
) -> "Federation":
    """Intersect zone with invariant upper bounds."""
    Z_new = Z
    for cname, bound in inv_pairs:
        Z_new = Z_new & (zctx.clocks[cname] <= bound)
    return Z_new


def normalize(Z: "Federation", zctx: ZoneContext, M: int) -> "Federation":
    """k-normalization by max constant M.

    Regular job clocks are bounded by M; delta is excluded (bound = INF).
    This ensures the zone graph is finite.
    """
    bounds = {zctx.clocks[cn]: M for cn in zctx.clock_names if cn != "delta"}
    bounds[zctx.delta] = _UDBM_INF
    return Z.extrapolate_max_bounds(bounds)


def dominates(Z_old: "Federation", Z_new: "Federation") -> bool:
    """Z_old dominates Z_new iff Z_new ⊆ Z_old.

    If Z_old dominates Z_new, the old zone is at least as good (represents
    at least all the futures Z_new does, with no higher cost), so Z_new
    need not be explored further.
    From Lemma 4 of the UPTA theory: Z_old ⊑ Z_new  iff  Z_new ⊆ Z_old.
    """
    return Z_new <= Z_old


def min_cost(Z: "Federation", zctx: ZoneContext) -> float:
    """Infimum of the delta clock in zone Z = minimum reachable makespan.

    In UDBM, M[0][delta] encodes  x_0 - delta <= b,  i.e.  delta >= -b.
    So  infimum(delta) = -M[0][delta_idx].
    For a federation (union of DBMs), take the minimum over all DBMs.
    """
    if Z.is_empty():
        return float("inf")
    dbms = Z.to_dbm_list()
    min_val = float("inf")
    for dbm in dbms:
        b = dbm.bound(0, zctx._delta_idx)
        if b is not None and b != _UDBM_INF:
            min_val = min(min_val, -b)
    return min_val
