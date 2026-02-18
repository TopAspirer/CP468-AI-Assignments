"""
fifteen_puzzle.py
-----------------
15-puzzle solver derived from the 8-puzzle (eight_puzzle.py).

FifteenPuzzle inherits ALL logic from the Puzzle base class:
  - A* search
  - h1 (Misplaced Tiles)
  - h2 (Manhattan Distance)
  - h3 (Linear Conflict)
  - Random state generation
  - Benchmarking

The ONLY changes from the 8-puzzle are:
  - SIZE = 4  (4x4 grid instead of 3x3)
  - GOAL    (15 tiles instead of 8)

This directly demonstrates how the 15-puzzle is a modification of the 8-puzzle,
as required by the assignment.

Usage:
    python fifteen_puzzle.py
    (runs 100 random puzzles, prints average stats for h1, h2, h3)
"""

from eight_puzzle import Puzzle


# =============================================================================
# FifteenPuzzle — extends EightPuzzle by overriding size and goal state only
# =============================================================================

class FifteenPuzzle(Puzzle):
    """
    15-puzzle (4x4 sliding tile puzzle).

    Inherits all A* logic, heuristics, and benchmarking from Puzzle (eight_puzzle.py).
    Only overrides SIZE and GOAL to adapt the base class to a 4x4 grid.

    Goal state layout:
         0  1  2  3
         4  5  6  7
         8  9 10 11
        12 13 14 15
    where 0 is the blank tile.
    """

    # --- Override: grid is now 4x4 with 15 tiles ---
    SIZE = 4
    GOAL = ( 0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,
            12, 13, 14, 15)

    # No other changes needed — all methods (astar, h1, h2, h3, generate_random_state,
    # run_benchmark) are inherited from Puzzle and work generically for any SIZE.


# =============================================================================
# Entry point — run 15-puzzle benchmark
# =============================================================================

if __name__ == '__main__':
    print("=" * 55)
    print("15-Puzzle Benchmark (A* with h1, h2, h3)")
    print("=" * 55)
    print("Note: 15-puzzle is significantly harder than 8-puzzle.")
    print("This may take several minutes.\n")

    puzzle = FifteenPuzzle()

    # Use fewer shuffle moves for 15-puzzle to keep runtimes reasonable
    # while still generating non-trivial puzzles.
    # num_moves=40 generates puzzles requiring roughly 20-45 moves to solve.
    results = puzzle.run_benchmark(num_puzzles=100, num_moves=40, seed=42)

    print("\nResults:")
    print(f"{'Heuristic':<12} {'Avg Steps':>12} {'Avg Nodes Expanded':>20}")
    print("-" * 46)
    for h_name, data in results.items():
        print(f"{h_name:<12} {data['avg_steps']:>12.2f} {data['avg_nodes']:>20.2f}")
