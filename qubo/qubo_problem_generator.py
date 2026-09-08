"""Problem generator used by the QUDO/QUBO paper experiments.

Adapted from SergioITCL/QUDO-tensor-network-solver, branch
``notebook_to_script``. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import random
from typing import Literal, TypedDict


class QudoInstance(TypedDict):
    instance_type: Literal["random", "fixed"]
    seed: int
    q_matrix: list[list[float]]
    q_row: list[float]


def generate_k_random_qudo(
    n_variables: int,
    k_neighbor: int,
    seed: int | None = None,
) -> list[list[float]]:
    if n_variables < 1:
        raise ValueError("n_variables must be greater than 0")
    if k_neighbor < 1:
        raise ValueError("k_neighbor must be greater than 0")

    rng = random.Random(seed)
    q_list = [[] for _ in range(n_variables)]
    n_elements_per_row = 1

    for row_index in range(n_variables):
        for _ in range(n_elements_per_row):
            q_list[row_index].append(rng.uniform(-10, 10))

        if n_elements_per_row <= k_neighbor:
            n_elements_per_row += 1

    return q_list


def generate_random_q_row(
    n_variables: int,
    seed: int | None = None,
) -> list[float]:
    if n_variables < 1:
        raise ValueError("n_variables must be greater than 0")

    rng = random.Random(
        f"q-row-random-{seed}" if seed is not None else None
    )
    return [rng.uniform(-10, 10) for _ in range(n_variables)]


def generate_fixed_interactions_qudo(
    n_variables: int,
    k_neighbors: int,
    seed: int | None = None,
) -> list[list[float]]:
    if n_variables < 1:
        raise ValueError("n_variables must be greater than 0")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be greater than 0")

    rng = random.Random(seed)
    neighbor_values = [rng.uniform(-5, 5) for _ in range(k_neighbors)]
    diagonal_value = rng.uniform(-5, 5)

    q_list: list[list[float]] = []
    for i in range(n_variables):
        max_distance = min(i, k_neighbors)
        row = [
            neighbor_values[distance - 1]
            for distance in range(max_distance, 0, -1)
        ]
        row.append(diagonal_value)
        q_list.append(row)

    return q_list


def generate_fixed_q_row(
    n_variables: int,
    seed: int | None = None,
) -> list[float]:
    if n_variables < 1:
        raise ValueError("n_variables must be greater than 0")

    rng = random.Random(
        f"q-row-fixed-{seed}" if seed is not None else None
    )
    linear_value = rng.uniform(-5, 5)
    return [linear_value] * n_variables


def generate_qubo_problem(
    n_variables: int,
    n_neighbors: int,
    seed: int = 0,
    instance_type: Literal["random", "fixed"] = "random",
) -> QudoInstance:
    """Generate one reproducible compact-band QUBO instance."""
    if instance_type == "random":
        return {
            "instance_type": "random",
            "seed": seed,
            "q_matrix": generate_k_random_qudo(
                n_variables, n_neighbors, seed
            ),
            "q_row": generate_random_q_row(n_variables, seed),
        }

    if instance_type == "fixed":
        return {
            "instance_type": "fixed",
            "seed": seed,
            "q_matrix": generate_fixed_interactions_qudo(
                n_variables, n_neighbors, seed
            ),
            "q_row": generate_fixed_q_row(n_variables, seed),
        }

    raise ValueError("instance_type must be 'random' or 'fixed'")


def qudo_problem_generation(
    n_variables: int,
    n_neighbors: int,
    n_random_instances: int,
    n_fixed_instances: int,
    random_seeds: list[int] | None = None,
) -> list[QudoInstance]:
    """Generate the mixed instance collection used by the paper scripts."""
    if n_random_instances < 0 or n_fixed_instances < 0:
        raise ValueError("The number of instances cannot be negative")
    if n_random_instances + n_fixed_instances == 0:
        raise ValueError("At least one instance must be generated")
    if random_seeds is not None and len(random_seeds) != n_random_instances:
        raise ValueError("random_seeds must match n_random_instances")

    seeds = (
        random_seeds
        if random_seeds is not None
        else list(range(n_random_instances))
    )

    instances = [
        generate_qubo_problem(
            n_variables,
            n_neighbors,
            seed=seed,
            instance_type="random",
        )
        for seed in seeds
    ]
    instances.extend(
        generate_qubo_problem(
            n_variables,
            n_neighbors,
            seed=seed,
            instance_type="fixed",
        )
        for seed in range(n_fixed_instances)
    )
    return instances
