"""
Pure-Python fallback implementations of axera._core (Rust extension).

Imported automatically when the compiled extension is not available
(e.g. source installs without a Rust toolchain, or CI environments).
Performance will be lower, but all behaviour is identical.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence


def combinations_all(n: int) -> list[list[int]]:
    """All non-empty subsets of range(n), ordered by size then lexicographic."""
    result: list[list[int]] = []
    for k in range(1, n + 1):
        for combo in combinations(range(n), k):
            result.append(list(combo))
    return result


def combinations_k(n: int, k: int) -> list[list[int]]:
    """All k-combinations of range(n)."""
    return [list(c) for c in combinations(range(n), k)]


def grid_compare(cost: float, upper_bounds: Sequence[float]) -> int:
    """First index j such that cost < upper_bounds[j]."""
    for j, ub in enumerate(upper_bounds):
        if cost < ub:
            return j
    return len(upper_bounds)


def grid_find_index(
    cost: Sequence[float],
    upper_bounds: Sequence[Sequence[float]],
    n_grid: int,
) -> int:
    """Flat integer grid index for a particle cost vector."""
    base = n_grid + 2
    idx = 0
    for c, ubs in zip(cost, upper_bounds):
        idx = idx * base + grid_compare(c, ubs)
    return idx


def pareto_dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """True if a weakly dominates b in all objectives and strictly in at least one."""
    all_le = all(ai <= bi for ai, bi in zip(a, b))
    any_lt = any(ai < bi for ai, bi in zip(a, b))
    return all_le and any_lt


def pareto_fast(costs: list[list[float]]) -> list[bool]:
    """Non-dominated front: True = particle is non-dominated."""
    n = len(costs)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i != j and pareto_dominates(costs[j], costs[i]):
                dominated[i] = True
                break
    return [not d for d in dominated]
