"""Sparse Matrix Vector Contraction (SMVC) solver.

Adapted from SergioITCL/QUDO-tensor-network-solver, branch
``notebook_to_script``. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .qubo_common import Solution, estimate_tau_max, qudo_value
from .smvc_nodes import (
    last_tensor,
    new_initial_tensor,
    node_0,
    node_grow,
    node_intermediate,
)


def _normalize_problem(
    q_matrix: list[list[float]],
    q_row: list[float],
) -> tuple[list[list[float]], list[float]]:
    if len(q_matrix) != len(q_row):
        raise ValueError("q_matrix and q_row must have the same length")

    all_values = [value for row in q_matrix for value in row]
    all_values.extend(q_row)
    norm = float(np.linalg.norm(all_values))

    if norm == 0:
        raise ValueError("A zero-norm problem cannot be normalized")

    return (
        [[float(value / norm) for value in row] for row in q_matrix],
        [float(value / norm) for value in q_row],
    )


def tensor_network_generator(
    q_matrix: list[list[float]],
    q_row: list[float],
    dits: int,
    n_neighbors: int,
    tau: float,
) -> list:
    n_variables = len(q_matrix)
    tensor_network = [
        node_0(q_matrix[0][0], q_row[0], dits, tau)
    ]

    for variable in range(1, n_variables - 1):
        if variable < n_neighbors:
            tensor = node_grow(
                q_matrix[variable],
                q_row[variable],
                dits,
                variable,
                tau,
            )
        else:
            tensor = node_intermediate(
                q_matrix[variable],
                q_row[variable],
                dits,
                n_neighbors,
                tau,
            )
        tensor_network.append(tensor)

    tensor_network.append(
        last_tensor(q_matrix[-1], q_row[-1], dits, tau)
    )
    return tensor_network


def tensor_network_contraction(tensor_list: list) -> tuple[np.ndarray, list]:
    tensor = tensor_list[-1]
    intermediate_tensors = [tensor]

    for current_tensor in reversed(tensor_list[:-1]):
        tensor = current_tensor @ tensor
        norm = float(np.linalg.norm(tensor))
        if norm == 0 or not np.isfinite(norm):
            raise FloatingPointError(
                "SMVC became numerically unstable; try a smaller tau"
            )
        tensor = tensor / norm
        intermediate_tensors.append(tensor)

    intermediate_tensors.reverse()
    return np.asarray(tensor), intermediate_tensors


def solver_smvc(
    q_list: list[list[float]],
    q_row: list[float],
    dits: int,
    n_neighbors: int,
    tau: float | None = None,
) -> Solution:
    """Solve a local QUDO/QUBO instance with the SMVC heuristic."""
    initial_time = perf_counter()

    if not q_list:
        raise ValueError("q_list must contain at least one variable")
    if len(q_list) != len(q_row):
        raise ValueError("q_list and q_row must have the same length")
    if dits < 2:
        raise ValueError("dits must be at least 2")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1 for SMVC")

    q_matrix, normalized_q_row = _normalize_problem(q_list, q_row)
    n_variables = len(q_matrix)
    solution = np.zeros(n_variables, dtype=int)

    if n_variables == 1:
        solution[0] = min(
            range(dits),
            key=lambda value: (
                q_matrix[0][0] * value**2
                + normalized_q_row[0] * value
            ),
        )
        return Solution.from_solution_list(
            q_list,
            q_row,
            list(solution),
            dits,
            perf_counter() - initial_time,
        )

    if tau is None:
        tau = estimate_tau_max(
            n_variables=n_variables,
            dits=dits,
            n_neighbors=n_neighbors,
        )

    network = tensor_network_generator(
        q_matrix,
        normalized_q_row,
        dits,
        n_neighbors,
        tau,
    )
    result, intermediate_tensors = tensor_network_contraction(network)
    solution[0] = int(np.argmax(np.abs(result)))

    for node in range(1, n_variables - 1):
        if node < n_neighbors:
            sol_aux = solution[
                max(0, node - n_neighbors - 1) : node
            ]
        else:
            sol_aux = solution[
                node - n_neighbors + 1 : node
            ]

        new_tensor = new_initial_tensor(
            q_matrix[node],
            normalized_q_row[node],
            dits,
            intermediate_tensors[2].shape[0],
            sol_aux,
            n_neighbors,
            tau,
            int(solution[node - n_neighbors]),
        )
        solution[node] = int(
            np.argmax(np.abs(new_tensor @ intermediate_tensors[2]))
        )
        intermediate_tensors.pop(0)

    normalized_cost = qudo_value(
        list(solution),
        q_matrix,
        normalized_q_row,
    )
    candidate = solution.copy()

    for dit in range(1, dits):
        candidate[-1] = dit
        candidate_cost = qudo_value(
            list(candidate),
            q_matrix,
            normalized_q_row,
        )
        if candidate_cost < normalized_cost:
            solution[-1] = dit
            normalized_cost = candidate_cost

    return Solution.from_solution_list(
        q_list,
        q_row,
        list(solution),
        dits,
        perf_counter() - initial_time,
    )


__all__ = ["solver_smvc"]
