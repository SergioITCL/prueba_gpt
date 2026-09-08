"""Exact vectorized dynamic-programming solver for banded QUDO/QUBO.

Adapted from SergioITCL/QUDO-tensor-network-solver, branch
``notebook_to_script``. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .qubo_common import Solution


def _validate_inputs(
    q_matrix: list[list[float]],
    q_row: list[float],
    dits: int,
    n_neighbors: int,
) -> None:
    if not q_matrix:
        raise ValueError("q_matrix must contain at least one variable")
    if len(q_row) != len(q_matrix):
        raise ValueError("q_matrix and q_row must have the same length")
    if dits < 2:
        raise ValueError("dits must be at least 2")
    if n_neighbors < 0:
        raise ValueError("n_neighbors must be non-negative")

    for position, row in enumerate(q_matrix):
        if not row:
            raise ValueError(f"q_matrix[{position}] must not be empty")

        maximum_row_length = min(position, n_neighbors) + 1
        if len(row) > maximum_row_length:
            raise ValueError(
                f"q_matrix[{position}] has {len(row) - 1} previous-variable "
                f"interactions, but at most {maximum_row_length - 1} are "
                "representable with n_neighbors"
            )


def _parent_dtype(dits: int) -> np.dtype:
    maximum_value = dits - 1
    if maximum_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if maximum_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if maximum_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def solver_vectorized_dynamic_programming(
    q_matrix: list[list[float]],
    q_row: list[float],
    dits: int,
    n_neighbors: int,
    require_nonzero: bool = False,
) -> Solution:
    """Solve a banded QUDO exactly with a NumPy boundary-state DP.

    Complexity is O(n * d**(k+1)) time and O(d**k) working frontier memory,
    plus backpointers required to reconstruct the optimum.
    """
    initial_time = perf_counter()
    _validate_inputs(q_matrix, q_row, dits, n_neighbors)

    if require_nonzero:
        raise NotImplementedError(
            "The paper benchmark path uses require_nonzero=False."
        )

    n_variables = len(q_matrix)
    k = n_neighbors
    values = np.arange(dits, dtype=np.float64)
    values_squared = values * values

    if k == 0:
        solution = [
            int(
                np.argmin(
                    row[-1] * values_squared + linear * values
                )
            )
            for row, linear in zip(q_matrix, q_row)
        ]
        return Solution.from_solution_list(
            q_matrix,
            q_row,
            solution,
            dits,
            perf_counter() - initial_time,
        )

    current_costs = np.asarray(0.0, dtype=np.float64)
    parent_choices: list[np.ndarray | None] = [None] * n_variables
    parent_dtype = _parent_dtype(dits)

    for position, (row, linear_coefficient) in enumerate(
        zip(q_matrix, q_row)
    ):
        boundary_width = min(position, k)
        interaction_count = len(row) - 1

        if interaction_count:
            interaction_field = np.zeros(
                current_costs.shape,
                dtype=np.float64,
            )
            first_relevant_axis = boundary_width - interaction_count

            for offset, coefficient in enumerate(row[:-1]):
                axis = first_relevant_axis + offset
                broadcast_shape = [1] * boundary_width
                broadcast_shape[axis] = dits
                interaction_field += (
                    float(coefficient)
                    * values.reshape(broadcast_shape)
                )
        else:
            interaction_field = 0.0

        diagonal_coefficient = float(row[-1])
        linear_coefficient = float(linear_coefficient)

        if boundary_width < k:
            next_costs = np.empty(
                current_costs.shape + (dits,),
                dtype=np.float64,
            )

            for value in range(dits):
                local_cost = (
                    diagonal_coefficient * values_squared[value]
                    + linear_coefficient * values[value]
                    + values[value] * interaction_field
                )
                next_costs[..., value] = current_costs + local_cost
        else:
            next_costs = np.empty_like(current_costs)
            step_parents = np.empty(
                current_costs.shape,
                dtype=parent_dtype,
            )

            for value in range(dits):
                local_cost = (
                    diagonal_coefficient * values_squared[value]
                    + linear_coefficient * values[value]
                    + values[value] * interaction_field
                )
                candidate_costs = current_costs + local_cost
                next_costs[..., value] = np.min(
                    candidate_costs,
                    axis=0,
                )
                step_parents[..., value] = np.argmin(
                    candidate_costs,
                    axis=0,
                ).astype(parent_dtype, copy=False)

            parent_choices[position] = step_parents

        current_costs = next_costs

    best_flat_index = int(np.argmin(current_costs))
    best_state = tuple(
        int(value)
        for value in np.unravel_index(
            best_flat_index,
            current_costs.shape,
        )
    )

    if n_variables <= k:
        solution = list(best_state)
    else:
        solution = [0] * n_variables
        solution[n_variables - k :] = best_state
        next_state = best_state

        for position in range(n_variables - 1, k - 1, -1):
            step_parents = parent_choices[position]
            if step_parents is None:
                raise RuntimeError(
                    f"Missing backpointer array at DP position {position}"
                )

            eliminated_value = int(step_parents[next_state])
            solution[position - k] = eliminated_value
            next_state = (eliminated_value,) + next_state[:-1]

    return Solution.from_solution_list(
        q_matrix,
        q_row,
        solution,
        dits,
        perf_counter() - initial_time,
    )


solver_dynamic_programming_vectorized = solver_vectorized_dynamic_programming
vectorized_programming_solver = solver_vectorized_dynamic_programming

__all__ = [
    "solver_vectorized_dynamic_programming",
    "solver_dynamic_programming_vectorized",
    "vectorized_programming_solver",
]
