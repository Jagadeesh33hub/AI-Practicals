"""
Practical 3: A* Search for 8-Puzzle & Heuristic Evaluation
Aim: Solve the 8-puzzle problem using A* search and compare search efficiency
     between h1 (Misplaced Tiles) and h2 (Manhattan Distance).
"""

import heapq
from typing import Callable

GOAL_STATE = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 0),  # 0 represents the blank tile
)

GOAL_POSITIONS = {
    GOAL_STATE[r][c]: (r, c)
    for r in range(3)
    for c in range(3)
}


def h_misplaced_tiles(state: tuple[tuple[int, ...], ...]) -> int:
    """h1: Count of tiles not in their goal position (excluding blank 0)."""
    count = 0
    for r in range(3):
        for c in range(3):
            val = state[r][c]
            if val != 0 and val != GOAL_STATE[r][c]:
                count += 1
    return count


def h_manhattan_distance(state: tuple[tuple[int, ...], ...]) -> int:
    """h2: Sum of Manhattan distances of all tiles from their goal positions."""
    distance = 0
    for r in range(3):
        for c in range(3):
            val = state[r][c]
            if val != 0:
                gr, gc = GOAL_POSITIONS[val]
                distance += abs(r - gr) + abs(c - gc)
    return distance


def get_neighbors(state: tuple[tuple[int, ...], ...]) -> list[tuple[str, tuple[tuple[int, ...], ...]]]:
    """Generate valid successor states by sliding the blank tile."""
    blank_r, blank_c = next(
        (r, c) for r in range(3) for c in range(3) if state[r][c] == 0
    )
    moves = [("Up", -1, 0), ("Down", 1, 0), ("Left", 0, -1), ("Right", 0, 1)]
    neighbors = []

    for action, dr, dc in moves:
        nr, nc = blank_r + dr, blank_c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            grid = [list(row) for row in state]
            grid[blank_r][blank_c], grid[nr][nc] = grid[nr][nc], grid[blank_r][blank_c]
            neighbors.append((action, tuple(tuple(row) for row in grid)))
    return neighbors


def a_star_search(
    initial_state: tuple[tuple[int, ...], ...],
    heuristic: Callable[[tuple[tuple[int, ...], ...]], int],
) -> dict:
    """Run A* search with f(n) = g(n) + h(n)."""
    start_h = heuristic(initial_state)
    open_heap = [(start_h, 0, initial_state, [])]
    g_scores = {initial_state: 0}
    nodes_expanded = 0
    max_frontier = 1

    while open_heap:
        max_frontier = max(max_frontier, len(open_heap))
        f, g, current, path = heapq.heappop(open_heap)

        if g > g_scores.get(current, float("inf")):
            continue

        nodes_expanded += 1
        if current == GOAL_STATE:
            return {
                "moves": path,
                "solution_depth": g,
                "nodes_expanded": nodes_expanded,
                "max_frontier": max_frontier,
            }

        for action, neighbor in get_neighbors(current):
            tentative_g = g + 1
            if tentative_g < g_scores.get(neighbor, float("inf")):
                g_scores[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor, path + [(action, neighbor)]))

    return {"moves": [], "solution_depth": -1, "nodes_expanded": nodes_expanded, "max_frontier": max_frontier}


def format_board(state: tuple[tuple[int, ...], ...]) -> str:
    return "\n".join("  " + " ".join(str(x) if x != 0 else "_" for x in row) for row in state)


if __name__ == "__main__":
    INITIAL_STATE = (
        (1, 3, 6),
        (5, 0, 2),
        (4, 7, 8),
    )

    print("=" * 72)
    print("A* SEARCH FOR 8-PUZZLE: HEURISTIC COMPARISON")
    print("=" * 72)
    print("Initial State:\n" + format_board(INITIAL_STATE))
    print("\nGoal State:\n" + format_board(GOAL_STATE))

    res_h1 = a_star_search(INITIAL_STATE, h_misplaced_tiles)
    res_h2 = a_star_search(INITIAL_STATE, h_manhattan_distance)

    print("\n" + "-" * 72)
    print(f"{'Heuristic':<24} | {'Optimal Moves':<14} | {'Nodes Expanded':<15} | {'Max Frontier'}")
    print("-" * 72)
    print(f"{'h1: Misplaced Tiles':<24} | {res_h1['solution_depth']:<14} | {res_h1['nodes_expanded']:<15} | {res_h1['max_frontier']}")
    print(f"{'h2: Manhattan Distance':<24} | {res_h2['solution_depth']:<14} | {res_h2['nodes_expanded']:<15} | {res_h2['max_frontier']}")
    print("-" * 72)

    print(f"\nOptimal Action Sequence ({res_h2['solution_depth']} steps):")
    print(" -> ".join(action for action, _ in res_h2["moves"]))
    print("\nConclusion: Manhattan Distance (h2) dominates Misplaced Tiles (h1),")
    print("expanding fewer nodes while guaranteeing the exact same optimal path.")
