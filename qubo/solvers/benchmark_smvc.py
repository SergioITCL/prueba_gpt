"""Benchmark reference SMVC against the optimized implementation."""

from __future__ import annotations

import argparse
from statistics import mean

from qubo.qubo_problem_generator import generate_qubo_problem
from qubo.smvc import solver_smvc
from qubo.solvers.smvc_optimized import solver_smvc_optimized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reference and optimized SMVC on identical QUBO instances."
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--dits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--instance-type",
        choices=("random", "fixed"),
        default="random",
    )
    parser.add_argument("--tau", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n < 2:
        raise SystemExit("--n must be at least 2")
    if args.k < 1:
        raise SystemExit("--k must be at least 1")
    if args.dits < 2:
        raise SystemExit("--dits must be at least 2")
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")

    instance = generate_qubo_problem(
        n_variables=args.n,
        n_neighbors=args.k,
        seed=args.seed,
        instance_type=args.instance_type,
    )

    reference_times: list[float] = []
    optimized_times: list[float] = []
    reference_result = None
    optimized_result = None

    # Warm the optimized implementation's cached state tables before timing.
    solver_smvc_optimized(
        instance["q_matrix"],
        instance["q_row"],
        dits=args.dits,
        n_neighbors=args.k,
        tau=args.tau,
    )

    for _ in range(args.repetitions):
        reference_result = solver_smvc(
            instance["q_matrix"],
            instance["q_row"],
            dits=args.dits,
            n_neighbors=args.k,
            tau=args.tau,
        )
        reference_times.append(reference_result.execution_time)

        optimized_result = solver_smvc_optimized(
            instance["q_matrix"],
            instance["q_row"],
            dits=args.dits,
            n_neighbors=args.k,
            tau=args.tau,
        )
        optimized_times.append(optimized_result.execution_time)

    assert reference_result is not None
    assert optimized_result is not None

    reference_mean = mean(reference_times)
    optimized_mean = mean(optimized_times)
    speedup = reference_mean / optimized_mean

    print("SMVC benchmark")
    print("=" * 72)
    print(
        f"instance={args.instance_type} seed={args.seed} "
        f"n={args.n} k={args.k} dits={args.dits} repetitions={args.repetitions}"
    )
    print()
    print(f"reference mean: {reference_mean:.6f} s")
    print(f"optimized mean: {optimized_mean:.6f} s")
    print(f"speedup:        {speedup:.2f}x")
    print()
    print(f"same solution:  {reference_result.solution_list == optimized_result.solution_list}")
    print(f"reference cost: {reference_result.cost:.12g}")
    print(f"optimized cost: {optimized_result.cost:.12g}")

    if reference_result.solution_list != optimized_result.solution_list:
        raise SystemExit(
            "WARNING: optimized SMVC produced a different solution for this instance"
        )


if __name__ == "__main__":
    main()
