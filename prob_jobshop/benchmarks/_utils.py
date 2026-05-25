from __future__ import annotations
import math
from typing import Dict, List, Tuple

from ..instance import TaskDistribution


def merge_distribution(raw_pairs: List[Tuple[int, float]]) -> TaskDistribution:
    """Build a TaskDistribution, merging any entries that share the same duration."""
    merged: Dict[int, float] = {}
    for dur, prob in raw_pairs:
        merged[dur] = merged.get(dur, 0.0) + prob
    durations = sorted(merged.keys())
    probs = [merged[d] for d in durations]
    return TaskDistribution(durations=durations, probabilities=probs)


def apply_formula_070_050_140(d: int) -> TaskDistribution:
    """Probabilistic version: {(floor(0.7d), 0.25), (d, 0.50), (ceil(1.4d), 0.25)}."""
    return merge_distribution([
        (max(1, math.floor(d * 0.7)), 0.25),
        (d, 0.50),
        (math.ceil(d * 1.4), 0.25),
    ])


def apply_formula_spread4(d: int) -> TaskDistribution:
    """High-variance formula: {(max(1,d-3), 0.35), (d, 0.30), (d+4, 0.35)}."""
    return merge_distribution([
        (max(1, d - 3), 0.35),
        (d, 0.30),
        (d + 4, 0.35),
    ])
