# CP468 Assignment #1 — Part II: A* Puzzle Solver

## Group 02 Members
Ali Khairreddin,<br>
Adam Narciso,  
Daniel Gonzalez,  
Lloyd Nsambu,  
Maor Ethan Chernitzky,<br>
Roneet Topiwala,<br>
Suleman Ali,<br>
Zoya Adnan 

---

## Overview

This project implements the A* search algorithm to solve the **8-puzzle** and **15-puzzle**,
comparing three heuristics completed by group 02:

| Heuristic | Description | Source |
|-----------|-------------|--------|
| **h1** | Misplaced Tiles — count of tiles not in their goal position | Russell & Norvig, 3rd ed., p. 102 |
| **h2** | Manhattan Distance — sum of tile distances from goal positions | Russell & Norvig, 3rd ed., p. 102 |
| **h3** | Linear Conflict — Manhattan + 2× conflicts in rows/cols | Hansson, Mayer & Yung (1992) |

---

## File Structure

```
puzzle_solver/
├── eight_puzzle.py     # Base Puzzle class: A*, h1, h2, h3, benchmarking
├── fifteen_puzzle.py   # FifteenPuzzle extends Puzzle (SIZE=4, new GOAL only)
├── results.py          # Runs both benchmarks, prints results table + analysis
└── README.md           # This file
```

**Design note:** `FifteenPuzzle` inherits *all* logic from the `Puzzle` base class in
`eight_puzzle.py`. The only overrides are `SIZE = 4` and a new `GOAL` tuple.
All heuristics and A* are dimension-agnostic.

---

## Requirements

- Python 3.7 or higher
- No external libraries required (uses only `heapq`, `random`, `math`, `collections` from the standard library)

---

## How to Run

### Option 1 — Run everything
Runs both 8-puzzle and 15-puzzle benchmarks and prints the combined results table:
```bash
python results.py
```

### Option 2 — Run 8-puzzle only
```bash
python eight_puzzle.py
```

### Option 3 — Run 15-puzzle only
```bash
python fifteen_puzzle.py
```

---

## Expected Runtime

| Task | Approximate Time |
|------|-----------------|
| 8-puzzle (100 puzzles × 3 heuristics) | 1–3 minutes |
| 15-puzzle (100 puzzles × 3 heuristics) | 5–20 minutes |

Times vary by hardware. The 15-puzzle uses a moderate shuffle depth (40 random moves
from goal) to keep runtimes tractable while still generating meaningful puzzles.

---

## Output Format

```
================================================================
  RESULTS TABLE — A* Performance on 8-puzzle and 15-puzzle
================================================================
Puzzle Type    Heuristic      Avg Steps to Solution   Avg Nodes Expanded
------------------------------------------------------------------------
8-puzzle       h1                             XX.XX                XX.XX
8-puzzle       h2                             XX.XX                XX.XX
8-puzzle       h3                             XX.XX                XX.XX

15-puzzle      h1                             XX.XX                XX.XX
15-puzzle      h2                             XX.XX                XX.XX
15-puzzle      h3                             XX.XX                XX.XX
================================================================
```

---

## Heuristic Details

### h1 — Misplaced Tiles
Counts tiles not in their goal position excluding blanks.
Simple but weak — only a loose lower bound on actual moves needed.

**Important limitation on the 15-puzzle:** h1 is too weak to reliably guide A*
through the much larger 15-puzzle search space. Because h1 underestimates the true
cost so severely, A* must explore an enormous number of nodes before finding the
solution. In our benchmark, many 15-puzzle instances hit the 500,000 node expansion
limit before a solution was found when using h1, and those puzzles were excluded from
the averages. This is a direct consequence of h1 providing almost no useful
information to A* — it cannot distinguish between a state that is 5 moves from the
goal and one that is 40 moves away if both have the same number of misplaced tiles.
This makes h1 impractical for the 15-puzzle in real use.

### h2 — Manhattan Distance
Sum of |row_current - row_goal| + |col_current - col_goal| for every tile.
Dominates h1: h2(n) ≥ h1(n) for all states, so A* expands fewer nodes.
Performs well on both the 8-puzzle and 15-puzzle, solving all benchmark instances
within the node limit.

### h3 — Linear Conflict *(from literature)*
Extends Manhattan distance by detecting when two tiles in the same row (or column)
are both destined for that row (or column) but are in reversed order relative to
their goals. Each such conflict requires at least 2 extra moves to resolve.

**Formula:** h3(n) = h2(n) + 2 × (number of linear conflicts)

**Admissibility:** Proven admissible by Hansson et al. (1992).
**Dominance:** h3(n) ≥ h2(n) ≥ h1(n) for all states n.
Consistently expands the fewest nodes across both puzzle sizes.

**Reference:** Hansson, O., Mayer, A., & Yung, M. (1992).
*Generating admissible heuristics by criticizing solutions to relaxed models.*

---

## AI Tool Disclosure

This assignment was completed by a group of 8 members. In accordance with the course
guidelines, we disclose the following use of AI tools during the completion of this assignment.

**Tool used:** Claude (Anthropic)

**How it was used:**

- **Code debugging:** Claude was used to help identify and resolve issues encountered
  during development, including a performance problem where certain 15-puzzle states
  caused A* to hang indefinitely. Claude suggested implementing a node expansion limit
  and a controlled puzzle generation strategy to bound difficulty.

- **Conceptual explanations:** Claude was used to help clarify concepts related to
  heuristic admissibility, dominance relationships between heuristics, and why h1
  is insufficient for the 15-puzzle search space.

- **Writing/documentation:** Claude assisted in drafting inline code comments, the
  README, and portions of the heuristic analysis section.

**What AI was NOT used for:**

- The core algorithm design and logic (A* implementation, heuristic formulas) were
  based directly on Russell & Norvig (3rd ed.) and our own understanding of the material.
- All final code was reviewed, understood, and verified by group members before submission.

**Note:** All AI-assisted content was reviewed for correctness and accuracy by the group.
