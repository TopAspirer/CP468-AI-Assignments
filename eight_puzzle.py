"""
eight_puzzle.py
---------------
Base implementation of the 8-puzzle solver using A* search.
Heuristics implemented:
  h1 - Misplaced Tiles 
  h2 - Manhattan Distance
  h3 - Linear Conflict

"""

import heapq
import random
import math


class Puzzle:
    """
    Base class for n^2 - 1 sliding tile puzzles (8-puzzle, 15-puzzle, etc.).
    Subclasses override SIZE and GOAL to adapt to different puzzle dimensions.
    """

    SIZE = 3          # grid is SIZE x SIZE
    GOAL = (0, 1, 2,  # 0 represents the blank tile
            3, 4, 5,
            6, 7, 8)

    def __init__(self):
        self.n = self.SIZE
        self.goal = self.GOAL
        # Precompute goal positions for O(1) heuristic lookup

        # goal_pos[tile] = (row, col)
        self.goal_pos = {}
        for idx, tile in enumerate(self.goal):
            self.goal_pos[tile] = (idx // self.n, idx % self.n)

    
    # Move generation

    def get_neighbours(self, state):
        """
        Return a list of states reachable from the current state
        by sliding a tile into the blank position.
        """
        neighbours = []
        blank = state.index(0)
        row, col = blank // self.n, blank % self.n

        # (delta_row, delta_col) for Up, Down, Left, Right moves
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in moves:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.n and 0 <= nc < self.n:
                neighbour = list(state)
                swap_idx = nr * self.n + nc
                # Slide the tile at (nr, nc) into the blank
                neighbour[blank], neighbour[swap_idx] = neighbour[swap_idx], neighbour[blank]
                neighbours.append(tuple(neighbour))

        return neighbours

    # -------------------------------------------------------------------------
    # Heuristics
    # -------------------------------------------------------------------------

    def h1(self, state):
        """
        Misplaced Tiles heuristic.
        Counts the number of tiles not in their goal position (blank excluded).
        """
        count = 0
        for idx, tile in enumerate(state):
            if tile != 0 and tile != self.goal[idx]:
                count += 1
        return count

    def h2(self, state):
        """
        Manhattan Distance heuristic.
        Sum of horizontal + vertical distances of each tile from its goal position.
        Admissible: each tile needs at least as many moves as its Manhattan distance.
        """
        total = 0
        for idx, tile in enumerate(state):
            if tile != 0:
                cur_row, cur_col = idx // self.n, idx % self.n
                goal_row, goal_col = self.goal_pos[tile]
                total += abs(cur_row - goal_row) + abs(cur_col - goal_col)
        return total

    def h3(self, state):
        """
        Linear Conflict heuristic (Hansson, Mayer & Yung, 1992).
        Manhattan distance + 2 * (number of linear conflicts).

        A linear conflict occurs when two tiles tj and tk are in their goal row
        (or column), but tj is to the right of tk while tj's goal is to the
        left of tk's goal; they must pass each other, costing at least 2 extra moves.

        Admissible: adds only admissible extra cost on top of Manhattan distance.
        Dominates h2.
        """
        manhattan = self.h2(state)
        conflicts = 0

        # check for row conflicts
        for row in range(self.n):
            # Collect tiles in this row that also belong in this row
            row_tiles = []
            for col in range(self.n):
                tile = state[row * self.n + col]
                if tile != 0 and self.goal_pos[tile][0] == row:
                    row_tiles.append((tile, col))  # (tile_value, current_col)

            # Count pairs in conflict
            for i in range(len(row_tiles)):
                for j in range(i + 1, len(row_tiles)):
                    ti, ci = row_tiles[i]
                    tj, cj = row_tiles[j]
                    # Goal columns
                    gi = self.goal_pos[ti][1]
                    gj = self.goal_pos[tj][1]
                    # Conflict: relative order is reversed between current and goal
                    if (ci < cj and gi > gj) or (ci > cj and gi < gj):
                        conflicts += 1

        # check for column conflicts
        for col in range(self.n):
            col_tiles = []
            for row in range(self.n):
                tile = state[row * self.n + col]
                if tile != 0 and self.goal_pos[tile][1] == col:
                    col_tiles.append((tile, row))  # (tile_value, current_row)

            for i in range(len(col_tiles)):
                for j in range(i + 1, len(col_tiles)):
                    ti, ri = col_tiles[i]
                    tj, rj = col_tiles[j]
                    gi = self.goal_pos[ti][0]
                    gj = self.goal_pos[tj][0]
                    if (ri < rj and gi > gj) or (ri > rj and gi < gj):
                        conflicts += 1

        return manhattan + 2 * conflicts
        
    
    
    # A * Search

    def astar(self, start, heuristic, node_limit=None):
        """
        A* search algorithm.

        Parameters:
        - start      
        - heuristic  
        - node_limit 
        
        Returns:
        - steps: int; number of moves
        - nodes_expanded : int; number of nodes popped from the open list
        """
        # Use Priority queue entries (heaps): (f, tie_breaker, g, state)
        counter = 0
        h_start = heuristic(start)
        open_list = [(h_start, counter, 0, start)]  # (f, counter, g, state)
        heapq.heapify(open_list)

        # Best known g-cost to reach each state
        g_cost = {start: 0}

        nodes_expanded = 0

        while open_list:
            f, _, g, state = heapq.heappop(open_list)

            # Skip stale entries (we may have updated a state's g-cost)
            if g > g_cost.get(state, math.inf):
                continue

            nodes_expanded += 1

            # Abort if we've hit the node expansion limit
            if node_limit is not None and nodes_expanded >= node_limit:
                return -1, nodes_expanded

            # Goal check
            if state == self.goal:
                return g, nodes_expanded

            # Expand neighbours
            for neighbour in self.get_neighbours(state):
                new_g = g + 1  # each move costs 1
                if new_g < g_cost.get(neighbour, math.inf):
                    g_cost[neighbour] = new_g
                    f_new = new_g + heuristic(neighbour)
                    counter += 1
                    heapq.heappush(open_list, (f_new, counter, new_g, neighbour))

        # No solution found
        return -1, nodes_expanded

    # Random Reachable State Generations
    def generate_random_state(self, num_moves=100):
        """
        Generate a random reachable state by performing num_moves random
        moves from the goal state. 
        """

        state = list(self.goal)
        prev_state = None

        for _ in range(num_moves):
            neighbours = self.get_neighbours(tuple(state))
            # Avoid immediately reversing the last move
            if prev_state is not None and tuple(prev_state) in neighbours:
                neighbours = [n for n in neighbours if n != tuple(prev_state)]
            chosen = random.choice(neighbours)
            prev_state = state
            state = list(chosen)

        return tuple(state)

    def generate_controlled_state(self, min_h=5, max_h=25):
        """
        Generate a random reachable state whose h3 value falls within
        [min_h, max_h]. 

        """
        while True:
            # Shuffle depth: enough to randomize, not so much it drifts too far
            num_moves = random.randint(20, 60)
            state = self.generate_random_state(num_moves)
            h = self.h3(state)
            if min_h <= h <= max_h:
                return state

   
    # Benchmarking
    def run_benchmark(self, num_puzzles=100, num_moves=50, seed=42):
        """Solve num_puzzles with h1, h2, h3. Returns dict of avg_steps, avg_nodes, solved per heuristic."""
        random.seed(seed)
        node_limit = 500_000 if self.SIZE == 4 else None
        min_h, max_h = (8, 20) if self.SIZE == 4 else (5, 20)
        heuristics = {'h1': self.h1, 'h2': self.h2, 'h3': self.h3}
        results = {name: {'steps': [], 'nodes': []} for name in heuristics}

        puzzles = [self.generate_controlled_state(min_h=min_h, max_h=max_h) for _ in range(num_puzzles)]

        for name, h in heuristics.items():
            for puzzle in puzzles:
                steps, nodes = self.astar(puzzle, h, node_limit=node_limit)
                if steps != -1:
                    results[name]['steps'].append(steps)
                    results[name]['nodes'].append(nodes)

        averages = {}
        for name in heuristics:
            s, n = results[name]['steps'], results[name]['nodes']
            count = len(s)
            averages[name] = {
                'avg_steps': sum(s) / count if count else 0,
                'avg_nodes': sum(n) / count if count else 0,
                'solved': count,
            }
        return averages


# =============================================================================
# Entry point — run 8-puzzle benchmark
# =============================================================================

if __name__ == '__main__':
    print("=" * 55)
    print("8-Puzzle Benchmark (A* with h1, h2, h3)")
    print("=" * 55)

    puzzle = Puzzle()
    results = puzzle.run_benchmark(num_puzzles=100, num_moves=50, seed=42)

    print("\nResults:")
    print(f"{'Heuristic':<12} {'Avg Steps':>12} {'Avg Nodes Expanded':>20}")
    print("-" * 46)
    for h_name, data in results.items():
        print(f"{h_name:<12} {data['avg_steps']:>12.2f} {data['avg_nodes']:>20.2f}")
