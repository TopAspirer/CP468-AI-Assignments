"""
results.py
----------
Runs the full benchmark for both the 8-puzzle and 15-puzzle and prints
the combined results table required by the assignment. Should output a fully formatted table with average steps and nodes expanded for each puzzle type and heuristic combination (h1, h2, h3).
"""

import time
from eight_puzzle import Puzzle as EightPuzzle
from fifteen_puzzle import FifteenPuzzle


def run_and_collect(puzzle_instance, puzzle_label, num_puzzles=100, num_moves=None, seed=42):
    """
    Run the benchmark for a given puzzle instance and return labelled results.

    Parameters
    ----------
    puzzle_instance : Puzzle or FifteenPuzzle
    puzzle_label    : str; e.g. "8-puzzle" or "15-puzzle"
    num_puzzles     : int; number of random puzzles to solve
    num_moves       : int; shuffle depth (defaults per puzzle type if None)
    seed            : int; random seed

    Returns
    -------
    rows : list of dicts with keys: puzzle, heuristic, avg_steps, avg_nodes
    """
    # Default shuffle depths:
    #   8-puzzle:  50 moves  → moderate difficulty, solutions typically 15–25 moves
    #   15-puzzle: 40 moves  → moderate difficulty, solutions typically 20–45 moves
    if num_moves is None:
        num_moves = 50 if puzzle_instance.SIZE == 3 else 40

    print(f"\nRunning {puzzle_label} benchmark...")
    print(f"  Puzzles: {num_puzzles} | Shuffle moves: {num_moves} | Seed: {seed}")
    print(f"{'='*55}")

    start_time = time.time()
    averages = puzzle_instance.run_benchmark(
        num_puzzles=num_puzzles,
        num_moves=num_moves,
        seed=seed
    )
    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed:.1f}s")

    # Build row data for the results table
    rows = []
    for h_name in ['h1', 'h2', 'h3']:
        rows.append({
            'puzzle':     puzzle_label,
            'heuristic':  h_name,
            'avg_steps':  averages[h_name]['avg_steps'],
            'avg_nodes':  averages[h_name]['avg_nodes'],
            'solved':     averages[h_name]['solved'],
        })

    return rows


def print_table(all_rows):
    """
    Print results in an organized
    """
    print("\n")
    print("=" * 80)
    print("  RESULTS TABLE; A* Performance on 8-puzzle and 15-puzzle")
    print("=" * 80)
    header = (f"{'Puzzle Type':<14} {'Heuristic':<12} {'Avg Steps':>12} "
              f"{'Avg Nodes Expanded':>22} {'Solved/100':>12}")
    print(header)
    print("-" * 80)

    current_puzzle = None
    for row in all_rows:
        if current_puzzle is not None and row['puzzle'] != current_puzzle:
            print()
        current_puzzle = row['puzzle']

        print(
            f"{row['puzzle']:<14} "
            f"{row['heuristic']:<12} "
            f"{row['avg_steps']:>12.2f} "
            f"{row['avg_nodes']:>22.2f} "
            f"{row['solved']:>12.2f}"
        )

    print("=" * 80)


def print_analysis(all_rows):
    """
    Print a brief analysis of heuristic performance based on results.
    """
    print("\n")
    print("=" * 72)
    print("  HEURISTIC PERFORMANCE ANALYSIS")
    print("=" * 72)

    for puzzle_label in ['8-puzzle', '15-puzzle']:
        rows = [r for r in all_rows if r['puzzle'] == puzzle_label]
        print(f"\n{puzzle_label}:")
        print("-" * 40)

        # Find best heuristic by fewest nodes expanded
        best = min(rows, key=lambda r: r['avg_nodes'])
        worst = max(rows, key=lambda r: r['avg_nodes'])

        for row in rows:
            tag = " ← best" if row['heuristic'] == best['heuristic'] else ""
            print(f"  {row['heuristic']}: {row['avg_nodes']:.1f} avg nodes expanded{tag}")
        
        if puzzle_label == '15-puzzle':
            by_heuristic = {r['heuristic']: r for r in rows}
            h1_row = by_heuristic['h1']
            h2_row = by_heuristic['h2']
            h3_row = by_heuristic['h3']
            print(f"\n  *** KEY FINDING (15-puzzle) ***")
            print(f"  h1 solved {h1_row['solved']}/100 puzzles within the node limit.")
            print(f"  h2 solved {h2_row['solved']}/100 — a {h2_row['solved'] - h1_row['solved']}-puzzle improvement over h1.")
            print(f"  h3 solved {h3_row['solved']}/100 with {h3_row['avg_nodes']:.0f} avg nodes vs")
            print(f"  h2's {h2_row['avg_nodes']:.0f} — confirming linear conflict's superiority.")

    print("\n  Dominance relationship: h3 >= h2 >= h1 (all admissible)")
    print("  More informed heuristic → fewer nodes → faster solve time")
    print("=" * 72)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("\nCP468 Assignment #1 — Part II: A* Puzzle Solver")
    print("Heuristics: h1=Misplaced Tiles, h2=Manhattan Distance, h3=Linear Conflict")

    all_rows = []

    # 8-puzzle
    eight = EightPuzzle()
    all_rows += run_and_collect(eight, '8-puzzle', num_puzzles=100, seed=42)

    # 15-puzzle
    fifteen = FifteenPuzzle()
    all_rows += run_and_collect(fifteen, '15-puzzle', num_puzzles=100, seed=42)

    # Print combined table
    print_table(all_rows)

    # Print analysis
    print_analysis(all_rows)
