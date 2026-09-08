"""Shared result model and QUDO/QUBO utilities.

Adapted from SergioITCL/QUDO-tensor-network-solver, branch
``notebook_to_script``. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def qudo_value(
    x: list[int],
    q_matrix: list[list[float]],
    q_row: list[float],
) -> float:
    """Evaluate the compact banded QUDO/QUBO objective."""
    if len(x) != len(q_matrix) or len(x) != len(q_row):
        raise ValueError("x, q_matrix, and q_row must have the same length")

    total = 0.0
    for i, row in enumerate(q_matrix):
        j_start = i - len(row) + 1
        for offset, coefficient in enumerate(row):
            j = j_start + offset
            total += coefficient * x[i] * x[j]
        total += q_row[i] * x[i]

    return float(total)


def estimate_tau_max(
    n_variables: int,
    dits: int,
    n_neighbors: int,
) -> float:
    """Estimate a numerically stable imaginary-time parameter for SMVC."""
    if n_variables <= 0:
        raise ValueError("n_variables must be positive")
    if dits < 2:
        raise ValueError("dits must be at least 2")
    if n_neighbors < 0:
        raise ValueError("n_neighbors must be non-negative")

    effective_neighbors = min(n_neighbors, n_variables - 1)
    max_terms_per_row = effective_neighbors + 1
    n_quadratic_coefficients = sum(
        min(row + 1, max_terms_per_row)
        for row in range(n_variables)
    )
    n_coefficients = n_quadratic_coefficients + n_variables
    estimated_coefficient_abs = np.sqrt(3.0 / n_coefficients)

    max_dit = dits - 1
    max_local_energy = estimated_coefficient_abs * (
        max_terms_per_row * max_dit**2 + max_dit
    )

    return float(300.0 / max_local_energy)


@dataclass(frozen=True)
class Solution:
    """Common result returned by both solvers."""

    solution_list: list[int]
    dits: int
    cost: float
    execution_time: float

    @classmethod
    def from_solution_list(
        cls,
        q_matrix: list[list[float]],
        q_row: list[float],
        solution_list: list[int],
        dits: int,
        execution_time: float,
    ) -> "Solution":
        solution = [int(value) for value in solution_list]
        return cls(
            solution_list=solution,
            dits=int(dits),
            cost=qudo_value(solution, q_matrix, q_row),
            execution_time=float(execution_time),
        )
