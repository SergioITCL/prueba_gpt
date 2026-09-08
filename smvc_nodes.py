"""Sparse-matrix building blocks used by SMVC.

Adapted from SergioITCL/QUDO-tensor-network-solver, branch
``notebook_to_script``. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.sparse import coo_matrix, csr_array, dia_matrix, diags


def node_0(
    q_matrix_0: float,
    q_row_0: float,
    dits: int,
    tau: float,
) -> dia_matrix:
    values = np.arange(dits)
    diagonal_values = np.exp(
        -tau * (q_matrix_0 * values**2 + q_row_0 * values)
    )
    return diags(diagonal_values, offsets=0, format="csr")


def node_grow(
    q_matrix_row: list[float],
    q_row_row: float,
    dits: int,
    n_neight: int,
    tau: float,
) -> np.ndarray:
    tensor = np.zeros((dits**n_neight, dits ** (n_neight + 1)))

    for element in product(range(dits), repeat=n_neight):
        index_up = sum(
            dits**aux * element[aux]
            for aux in range(n_neight)
        )

        for index_last in range(dits):
            index_down = index_up + dits**n_neight * index_last
            full_element = list(element) + [index_last]
            value = np.exp(-tau * q_row_row * full_element[-1])

            for aux in range(len(full_element)):
                value *= np.exp(
                    -tau
                    * q_matrix_row[aux]
                    * full_element[-1]
                    * full_element[aux]
                )

            tensor[index_up, index_down] = value

    return tensor


def node_intermediate(
    q_matrix_row: list[float],
    q_row_row: float,
    dits: int,
    n_neigh: int,
    tau: float,
) -> csr_array:
    size = dits**n_neigh
    n_blocks = dits ** (n_neigh - 1)
    digits = np.array(
        np.unravel_index(
            np.arange(size),
            (dits,) * n_neigh,
        )
    ).T

    q_lin = np.asarray(q_matrix_row[:n_neigh][::-1])
    interaction_field = digits @ q_lin
    values = np.arange(dits)
    q_last = q_matrix_row[-1]

    exponents = (
        interaction_field[:, None] * values[None, :]
        + (q_last * values**2)[None, :]
        + q_row_row * values[None, :]
    )
    data_matrix = np.exp(-tau * exponents)

    rows = np.repeat(np.arange(size), dits)
    base = np.arange(size) // dits
    cols = (
        base[:, None]
        + values[None, :] * n_blocks
    ).ravel(order="C")

    return coo_matrix(
        (
            data_matrix.ravel(order="C"),
            (rows, cols),
        ),
        shape=(size, size),
    ).tocsr()


def last_tensor(
    q_matrix_row: list[float],
    q_row_row: float,
    dits: int,
    tau: float,
) -> np.ndarray:
    n_neighbors = len(q_matrix_row) - 1
    tensor = np.zeros(dits**n_neighbors)

    for element in product(range(dits), repeat=n_neighbors):
        index_up = sum(
            dits**aux * element[aux]
            for aux in range(n_neighbors)
        )

        for index_last in range(dits):
            full_element = list(element) + [index_last]
            tensor_aux = np.exp(
                -tau * q_row_row * full_element[-1]
            )

            for index in range(len(full_element)):
                tensor_aux *= np.exp(
                    -tau
                    * q_matrix_row[index]
                    * full_element[index]
                    * full_element[-1]
                )

            tensor[index_up] += tensor_aux

    return tensor


def new_initial_tensor(
    q_matrix_row: list[float],
    q_row_row: float,
    dits: int,
    size_2: int,
    solution: np.ndarray | list[int],
    n_neigh: int,
    tau: float,
    last_solution: int,
) -> np.ndarray:
    tensor = np.zeros((dits, size_2))
    solution_tuple = tuple(int(value) for value in solution)
    n = len(solution_tuple) + 1
    index_down = sum(
        dits**aux * solution_tuple[aux]
        for aux in range(len(solution_tuple))
    )

    for element in product(range(dits), repeat=n):
        if element[:-1] != solution_tuple:
            continue

        index_down_aux = (
            index_down + dits ** (n - 1) * element[-1]
        )

        if len(q_matrix_row) == 2 + len(solution_tuple):
            tensor[element[-1], index_down_aux] = np.exp(
                -tau
                * (
                    q_matrix_row[0]
                    * last_solution
                    * element[-1]
                    + q_row_row * element[-1]
                )
            )

            for index in range(len(element)):
                if index + 1 < len(q_matrix_row):
                    tensor[element[-1], index_down_aux] *= np.exp(
                        -tau
                        * q_matrix_row[index + 1]
                        * element[index]
                        * element[-1]
                    )
        else:
            tensor[element[-1], index_down_aux] = np.exp(
                -tau * q_row_row * element[-1]
            )

            for index in range(len(element)):
                if index < len(q_matrix_row):
                    tensor[element[-1], index_down_aux] *= np.exp(
                        -tau
                        * q_matrix_row[index]
                        * element[index]
                        * element[-1]
                    )

    return tensor
