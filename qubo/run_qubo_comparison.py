"""Generate one paper-style QUBO instance and compare all three solvers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__:
    from .qubo_problem_generator import generate_qubo_problem
    from .smvc import solver_smvc
    from .solvers.smvc_optimized import solver_smvc_optimized
    from .vectorized_programming_solver import solver_vectorized_dynamic_programming
else:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from qubo.qubo_problem_generator import generate_qubo_problem
    from qubo.smvc import solver_smvc
    from qubo.solvers.smvc_optimized import solver_smvc_optimized
    from qubo.vectorized_programming_solver import solver_vectorized_dynamic_programming


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact vectorized dynamic programming, reference SMVC, "
            "and optimized SMVC on the same reproducible banded QUBO instance."
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


def _gap(cost: float, optimum: float) -> tuple[float, float | None]:
    absolute = cost - optimum
    relative = absolute / abs(optimum) if optimum != 0 else None
    return absolute, relative


def _print_result(name: str, result) -> None:
    print(name)
    print(f"  cost:     {result.cost:.12g}")
    print(f"  time:     {result.execution_time:.6f} s")
    print(f"  solution: {result.solution_list}")
    print()


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
    optimized = solver_smvc_optimized(
        instance["q_matrix"],
        instance["q_row"],
        dits=args.dits,
        n_neighbors=args.k,
        tau=args.tau,
    )

    smvc_gap, smvc_relative_gap = _gap(smvc.cost, exact.cost)
    optimized_gap, optimized_relative_gap = _gap(optimized.cost, exact.cost)
    speedup = (
        smvc.execution_time / optimized.execution_time
        if optimized.execution_time > 0
        else float("inf")
    )

    print("QUBO comparison")
    print("=" * 72)
    print(
        f"instance={args.instance_type} seed={args.seed} "
        f"n={args.n} k={args.k} dits={args.dits}"
    )
    print()

    _print_result("Vectorized dynamic programming (exact)", exact)
    _print_result("SMVC (reference)", smvc)
    _print_result("SMVC optimized", optimized)

    print("Comparison against exact optimum")
    print(f"  SMVC absolute gap:             {smvc_gap:.12g}")
    print(f"  SMVC optimized absolute gap:   {optimized_gap:.12g}")
    if smvc_relative_gap is None:
        print("  SMVC relative gap:             undefined (optimum is zero)")
        print("  Optimized relative gap:        undefined (optimum is zero)")
    else:
        print(f"  SMVC relative gap:             {smvc_relative_gap:.6%}")
        print(f"  Optimized relative gap:        {optimized_relative_gap:.6%}")

    print()
    print("SMVC reference vs optimized")
    print(f"  speedup:                       {speedup:.2f}x")
    print(f"  same solution:                 {smvc.solution_list == optimized.solution_list}")
    print(f"  same cost:                     {abs(smvc.cost - optimized.cost) <= 1e-9}")
    print(f"  reference matches optimum:     {abs(smvc.cost - exact.cost) <= 1e-9}")
    print(f"  optimized matches optimum:     {abs(optimized.cost - exact.cost) <= 1e-9}")


if __name__ == "__main__":
    main()
