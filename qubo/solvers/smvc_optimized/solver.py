"""Aggressively optimized Sparse Matrix Vector Contraction (SMVC).

This implementation preserves the SMVC decision rule while avoiding the main
sources of overhead in the reference implementation:

- no dense ``new_initial_tensor`` matrices during decoding;
- no explicit Python ``itertools.product`` loops in the hot path;
- cached base-d state tables and transition indices;
- direct vector contractions instead of materializing most sparse operators;
- vectorized local-energy evaluation with NumPy;
- numerically stable exponentials via max-shifting.

The reference implementation remains available at ``qubo/smvc.py``.
"""

from __future__ import annotations

from functools import lru_cache
from math import sqrt
from time import perf_counter

import numpy as np

from qubo.qubo_common import Solution, estimate_tau_max


def _state_dtype(dits: int) -> np.dtype:
    if dits <= 256:
        return np.dtype(np.uint8)
    if dits <= 65_536:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


@lru_cache(maxsize=None)
def _state_digits(dits: int, width: int) -> np.ndarray:
    """Return all width-digit base-d states in little-endian digit order."""
    if width == 0:
        states = np.zeros((1, 0), dtype=_state_dtype(dits))
        states.setflags(write=False)
        return states

    size = dits**width
    indices = np.arange(size, dtype=np.int64)[:, None]
    powers = np.power(
        dits,
        np.arange(width, dtype=np.int64),
        dtype=np.int64,
    )[None, :]
    states = ((indices // powers) % dits).astype(
        _state_dtype(dits),
        copy=False,
    )
    states.setflags(write=False)
    return states


@lru_cache(maxsize=None)
def _shift_indices(dits: int, width: int) -> np.ndarray:
    """Map each boundary state and appended value to the shifted next state."""
    if width < 1:
        raise ValueError("width must be at least 1")

    size = dits**width
    block = dits ** (width - 1)
    base = np.arange(size, dtype=np.int64) // dits
    values = np.arange(dits, dtype=np.int64)
    indices = base[:, None] + values[None, :] * block
    indices.setflags(write=False)
    return indices


def _normalize_problem(
    q_matrix: list[list[float]],
    q_row: list[float],
) -> tuple[list[np.ndarray], np.ndarray]:
    if len(q_matrix) != len(q_row):
        raise ValueError("q_matrix and q_row must have the same length")

    q_row_array = np.asarray(q_row, dtype=np.float64)
    norm_squared = float(np.dot(q_row_array, q_row_array))

    q_arrays: list[np.ndarray] = []
    for row in q_matrix:
        row_array = np.asarray(row, dtype=np.float64)
        q_arrays.append(row_array)
        norm_squared += float(np.dot(row_array, row_array))

    norm = sqrt(norm_squared)
    if norm == 0:
        raise ValueError("A zero-norm problem cannot be normalized")

    return [row / norm for row in q_arrays], q_row_array / norm


def _stable_exp(energy: np.ndarray, tau: float) -> np.ndarray:
    """Exponentiate Boltzmann weights without avoidable under/overflow."""
    exponent = -tau * energy
    exponent -= np.max(exponent)
    return np.exp(exponent)


def _last_suffix_vector(
    row: np.ndarray,
    linear: float,
    dits: int,
    tau: float,
    values: np.ndarray,
    values_squared: np.ndarray,
) -> np.ndarray:
    width = len(row) - 1

    if width:
        states = _state_digits(dits, width)
        interaction = states @ row[:-1]
    else:
        interaction = np.zeros(1, dtype=np.float64)

    energy = (
        interaction[:, None] * values[None, :]
        + row[-1] * values_squared[None, :]
        + linear * values[None, :]
    )
    return np.sum(_stable_exp(energy, tau), axis=1)


def _contract_one_node(
    row: np.ndarray,
    linear: float,
    position: int,
    n_neighbors: int,
    dits: int,
    tau: float,
    suffix: np.ndarray,
    values: np.ndarray,
    values_squared: np.ndarray,
) -> np.ndarray:
    """Contract one SMVC layer directly against the right suffix vector."""
    if position == 0:
        energy = row[-1] * values_squared + linear * values
        result = _stable_exp(energy, tau) * suffix
    else:
        width = min(position, n_neighbors)
        states = _state_digits(dits, width)

        if len(row) > 1:
            interaction = states @ row[:-1]
        else:
            interaction = np.zeros(dits**width, dtype=np.float64)

        energy = (
            interaction[:, None] * values[None, :]
            + row[-1] * values_squared[None, :]
            + linear * values[None, :]
        )
        weights = _stable_exp(energy, tau)

        if position < n_neighbors:
            size = dits**width
            suffix_candidates = suffix.reshape(dits, size).T
        else:
            suffix_candidates = suffix[_shift_indices(dits, n_neighbors)]

        # Faster than allocating weights * suffix_candidates then reducing.
        result = np.einsum(
            "ij,ij->i",
            weights,
            suffix_candidates,
            optimize=False,
        )

    norm = float(np.linalg.norm(result))
    if norm == 0 or not np.isfinite(norm):
        raise FloatingPointError(
            "Optimized SMVC became numerically unstable; try a smaller tau"
        )

    return result / norm


def _select_value(
    position: int,
    solution: list[int],
    row: np.ndarray,
    linear: float,
    n_neighbors: int,
    dits: int,
    tau: float,
    suffix: np.ndarray,
    values: np.ndarray,
    values_squared: np.ndarray,
) -> int:
    """Evaluate all possible values without building new_initial_tensor."""
    interaction_count = len(row) - 1

    if interaction_count:
        history = np.asarray(
            solution[position - interaction_count : position],
            dtype=np.float64,
        )
        field = float(np.dot(row[:-1], history))
    else:
        field = 0.0

    local_energy = (
        field * values
        + row[-1] * values_squared
        + linear * values
    )

    prefix = solution[max(0, position - n_neighbors + 1) : position]
    base_index = sum(
        (dits**index) * int(value)
        for index, value in enumerate(prefix)
    )
    candidate_indices = (
        base_index
        + dits ** len(prefix) * values.astype(np.int64, copy=False)
    )
    suffix_values = suffix[candidate_indices]

    # argmax(exp(-tau E) * suffix) in log-space.
    log_suffix = np.full(dits, -np.inf, dtype=np.float64)
    positive = suffix_values > 0
    log_suffix[positive] = np.log(suffix_values[positive])
    scores = log_suffix - tau * local_energy
    return int(np.argmax(scores))


def solver_smvc_optimized(
    q_list: list[list[float]],
    q_row: list[float],
    dits: int,
    n_neighbors: int,
    tau: float | None = None,
) -> Solution:
    """Solve a local QUDO/QUBO instance using an optimized SMVC implementation."""
    initial_time = perf_counter()

    if not q_list:
        raise ValueError("q_list must contain at least one variable")
    if len(q_list) != len(q_row):
        raise ValueError("q_list and q_row must have the same length")
    if dits < 2:
        raise ValueError("dits must be at least 2")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1 for SMVC")

    for position, row in enumerate(q_list):
        if not row:
            raise ValueError(f"q_list[{position}] must not be empty")
        max_row_length = min(position, n_neighbors) + 1
        if len(row) > max_row_length:
            raise ValueError(
                f"q_list[{position}] contains too many interactions for "
                f"n_neighbors={n_neighbors}"
            )

    q_matrix, normalized_q_row = _normalize_problem(q_list, q_row)
    n_variables = len(q_matrix)
    values = np.arange(dits, dtype=np.float64)
    values_squared = values * values

    if n_variables == 1:
        solution = [
            int(
                np.argmin(
                    q_matrix[0][-1] * values_squared
                    + normalized_q_row[0] * values
                )
            )
        ]
        return Solution.from_solution_list(
            q_list,
            q_row,
            solution,
            dits,
            perf_counter() - initial_time,
        )

    if tau is None:
        tau = estimate_tau_max(
            n_variables=n_variables,
            dits=dits,
            n_neighbors=n_neighbors,
        )

    # suffixes[i] contains the already-contracted right-hand side beginning at i.
    suffixes: list[np.ndarray | None] = [None] * n_variables
    suffixes[-1] = _last_suffix_vector(
        q_matrix[-1],
        float(normalized_q_row[-1]),
        dits,
        tau,
        values,
        values_squared,
    )

    for position in range(n_variables - 2, -1, -1):
        next_suffix = suffixes[position + 1]
        if next_suffix is None:
            raise RuntimeError("Missing SMVC suffix during contraction")

        suffixes[position] = _contract_one_node(
            q_matrix[position],
            float(normalized_q_row[position]),
            position,
            n_neighbors,
            dits,
            tau,
            next_suffix,
            values,
            values_squared,
        )

    first_suffix = suffixes[0]
    if first_suffix is None:
        raise RuntimeError("Missing first SMVC contraction result")

    solution = [0] * n_variables
    solution[0] = int(np.argmax(np.abs(first_suffix)))

    for position in range(1, n_variables - 1):
        suffix = suffixes[position + 1]
        if suffix is None:
            raise RuntimeError("Missing SMVC suffix during decoding")

        solution[position] = _select_value(
            position,
            solution,
            q_matrix[position],
            float(normalized_q_row[position]),
            n_neighbors,
            dits,
            tau,
            suffix,
            values,
            values_squared,
        )

    last_row = q_matrix[-1]
    interaction_count = len(last_row) - 1
    if interaction_count:
        history = np.asarray(
            solution[n_variables - 1 - interaction_count : n_variables - 1],
            dtype=np.float64,
        )
        field = float(np.dot(last_row[:-1], history))
    else:
        field = 0.0

    last_energy = (
        field * values
        + last_row[-1] * values_squared
        + float(normalized_q_row[-1]) * values
    )
    solution[-1] = int(np.argmin(last_energy))

    return Solution.from_solution_list(
        q_list,
        q_row,
        solution,
        dits,
        perf_counter() - initial_time,
    )


__all__ = ["solver_smvc_optimized"]
