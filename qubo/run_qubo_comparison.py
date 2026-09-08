"""Generate one paper-style QUBO instance and run both requested solvers."""

from __future__ import annotations

import argparse

from .qubo_problem_generator import generate_qubo_problem
from .smvc import solver_smvc
from .vectorized_programming_solver import (
    solver_vectorized_dynamic_programming,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact vectorized dynamic programming against SMVC "
            "on the same reproducible banded QUBO instance."
        )
    )
    parser.add_argument("--n", type=int, default=20, help="Number of variables")
    parser.add_argument("--k", type=int, default=2, help="Neighbor range")
    parser.add_argument("--dits", type=int, default=2, help="Local dimension")
    parser.add_argument("--seed", type=int, default=7, help="Problem seed")
    parser.add_argument(
        "--instance-type",
        choices=("random", "fixed"),
        default="random",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Optional SMVC tau. Default uses the repository estimator.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n < 1:
        raise SystemExit("--n must be at least 1")
    if args.k < 1:
        raise SystemExit("--k must be at least 1")
    if args.dits < 2:
        raise SystemExit("--dits must be at least 2")

    instance = generate_qubo_problem(
        n_variables=args.n,
        n_neighbors=args.k,
        seed=args.seed,
        instance_type=args.instance_type,
    )

    exact = solver_vectorized_dynamic_programming(
        instance["q_matrix"],
        instance["q_row"],
        dits=args.dits,
        n_neighbors=args.k,
    )
    smvc = solver_smvc(
        instance["q_matrix"],
        instance["q_row"],
        dits=args.dits,
        n_neighbors=args.k,
        tau=args.tau,
    )

    absolute_gap = smvc.cost - exact.cost
    relative_gap = (
        absolute_gap / abs(exact.cost)
        if exact.cost != 0
        else float("nan")
    )

    print("QUBO comparison")
    print("=" * 72)
    print(
        f"instance={args.instance_type} seed={args.seed} "
        f"n={args.n} k={args.k} dits={args.dits}"
    )
    print()
    print("Vectorized dynamic programming (exact)")
    print(f"  cost:     {exact.cost:.12g}")
    print(f"  time:     {exact.execution_time:.6f} s")
    print(f"  solution: {exact.solution_list}")
    print()
    print("SMVC")
    print(f"  cost:     {smvc.cost:.12g}")
    print(f"  time:     {smvc.execution_time:.6f} s")
    print(f"  solution: {smvc.solution_list}")
    print()
    print("Comparison")
    print(f"  absolute gap (SMVC - optimum): {absolute_gap:.12g}")
    if exact.cost != 0:
        print(f"  relative gap:                  {relative_gap:.6%}")
    else:
        print("  relative gap:                  undefined (optimum is zero)")
    print(f"  same solution:                 {smvc.solution_list == exact.solution_list}")


if __name__ == "__main__":
    main()
