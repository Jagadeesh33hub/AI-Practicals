"""
Practical 2: Uninformed Search Algorithms (BFS, DFS, and Uniform-Cost Search)
Aim: Implement BFS, DFS, and UCS on a weighted city navigation graph and compare
     nodes expanded (search cost), path length, and total path cost (path quality).
"""

from collections import deque
import heapq


CITY_GRAPH: dict[str, list[tuple[str, int]]] = {
    "Arad": [("Zerind", 75), ("Sibiu", 140), ("Timisoara", 118)],
    "Zerind": [("Arad", 75), ("Oradea", 71)],
    "Oradea": [("Zerind", 71), ("Sibiu", 151)],
    "Sibiu": [("Arad", 140), ("Oradea", 151), ("Fagaras", 99), ("Rimnicu", 80)],
    "Timisoara": [("Arad", 118), ("Lugoj", 111)],
    "Lugoj": [("Timisoara", 111), ("Mehadia", 70)],
    "Mehadia": [("Lugoj", 70), ("Drobeta", 75)],
    "Drobeta": [("Mehadia", 75), ("Craiova", 120)],
    "Craiova": [("Drobeta", 120), ("Rimnicu", 146), ("Pitesti", 138)],
    "Rimnicu": [("Sibiu", 80), ("Craiova", 146), ("Pitesti", 97)],
    "Fagaras": [("Sibiu", 99), ("Bucharest", 211)],
    "Pitesti": [("Rimnicu", 97), ("Craiova", 138), ("Bucharest", 101)],
    "Bucharest": [("Fagaras", 211), ("Pitesti", 101)],
}


def bfs(graph: dict, start: str, goal: str) -> dict:
    """Breadth-First Search (minimizes number of edges, ignores edge weights)."""
    queue = deque([(start, [start], 0)])
    visited = {start}
    nodes_expanded = 0

    while queue:
        current, path, cost = queue.popleft()
        nodes_expanded += 1
        if current == goal:
            return {"algorithm": "BFS", "path": path, "cost": cost, "expanded": nodes_expanded}

        for neighbor, weight in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor], cost + weight))

    return {"algorithm": "BFS", "path": None, "cost": float("inf"), "expanded": nodes_expanded}


def dfs(graph: dict, start: str, goal: str) -> dict:
    """Depth-First Search (explores deep branches first; neither optimal nor shortest)."""
    stack = [(start, [start], 0)]
    visited = set()
    nodes_expanded = 0

    while stack:
        current, path, cost = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            return {"algorithm": "DFS", "path": path, "cost": cost, "expanded": nodes_expanded}

        for neighbor, weight in reversed(graph.get(current, [])):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor], cost + weight))

    return {"algorithm": "DFS", "path": None, "cost": float("inf"), "expanded": nodes_expanded}


def ucs(graph: dict, start: str, goal: str) -> dict:
    """Uniform-Cost Search (expands lowest cumulative path cost first; optimal)."""
    pq = [(0, start, [start])]
    best_cost = {start: 0}
    nodes_expanded = 0

    while pq:
        cost, current, path = heapq.heappop(pq)
        if cost > best_cost.get(current, float("inf")):
            continue
        nodes_expanded += 1

        if current == goal:
            return {"algorithm": "UCS", "path": path, "cost": cost, "expanded": nodes_expanded}

        for neighbor, weight in graph.get(current, []):
            new_cost = cost + weight
            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))

    return {"algorithm": "UCS", "path": None, "cost": float("inf"), "expanded": nodes_expanded}


if __name__ == "__main__":
    START_NODE = "Arad"
    GOAL_NODE = "Bucharest"

    print("=" * 78)
    print(f"UNINFORMED SEARCH COMPARISON: {START_NODE} -> {GOAL_NODE}")
    print("=" * 78)

    results = [
        bfs(CITY_GRAPH, START_NODE, GOAL_NODE),
        dfs(CITY_GRAPH, START_NODE, GOAL_NODE),
        ucs(CITY_GRAPH, START_NODE, GOAL_NODE),
    ]

    for res in results:
        print(f"\n[{res['algorithm']}]")
        print(f"  Path Found     : {' -> '.join(res['path'])}")
        print(f"  Path Hops      : {len(res['path']) - 1}")
        print(f"  Total Path Cost: {res['cost']} km")
        print(f"  Nodes Expanded : {res['expanded']}")

    print("\n" + "=" * 78)
    print(f"{'Algorithm':<12} | {'Nodes Expanded':<15} | {'Hops':<6} | {'Total Cost (km)':<16} | {'Optimal?'}")
    print("-" * 78)
    min_cost = min(r["cost"] for r in results)
    for r in results:
        is_opt = "Yes" if r["cost"] == min_cost else "No"
        print(f"{r['algorithm']:<12} | {r['expanded']:<15} | {len(r['path'])-1:<6} | {r['cost']:<16} | {is_opt}")
    print("=" * 78)
