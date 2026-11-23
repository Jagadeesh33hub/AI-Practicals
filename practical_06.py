"""
Practical 6: Constraint Satisfaction Problems (CSP)
Aim: Solve Australia Map Coloring and 8-Queens using Backtracking search with
     Minimum Remaining Values (MRV) heuristic and Forward Checking consistency.
"""

REGIONS = ["WA", "NT", "SA", "Q", "NSW", "V", "T"]
NEIGHBORS = {
    "WA": ["NT", "SA"],
    "NT": ["WA", "SA", "Q"],
    "SA": ["WA", "NT", "Q", "NSW", "V"],
    "Q": ["NT", "SA", "NSW"],
    "NSW": ["Q", "SA", "V"],
    "V": ["SA", "NSW"],
    "T": [],
}
COLORS = ["Red", "Green", "Blue"]


def select_unassigned_mrv(assignment: dict[str, str], domains: dict[str, list[str]]) -> str:
    unassigned = [v for v in REGIONS if v not in assignment]
    return min(unassigned, key=lambda var: len(domains[var]))


def forward_check(
    var: str, value: str, assignment: dict[str, str], domains: dict[str, list[str]]
) -> dict[str, list[str]] | None:
    new_domains = {k: list(v) for k, v in domains.items()}
    new_domains[var] = [value]
    for neighbor in NEIGHBORS[var]:
        if neighbor not in assignment and value in new_domains[neighbor]:
            new_domains[neighbor].remove(value)
            if not new_domains[neighbor]:
                return None
    return new_domains


def solve_map_coloring(
    assignment: dict[str, str], domains: dict[str, list[str]], stats: dict[str, int]
) -> dict[str, str] | None:
    if len(assignment) == len(REGIONS):
        return assignment

    var = select_unassigned_mrv(assignment, domains)
    for color in domains[var]:
        stats["assignments_tried"] += 1
        if all(assignment.get(nb) != color for nb in NEIGHBORS[var]):
            pruned_domains = forward_check(var, color, assignment, domains)
            if pruned_domains is not None:
                assignment[var] = color
                result = solve_map_coloring(assignment, pruned_domains, stats)
                if result is not None:
                    return result
                del assignment[var]
    return None


def solve_n_queens(n: int) -> list[int] | None:
    cols = set()
    pos_diag = set()
    neg_diag = set()
    board = [-1] * n

    def backtrack(row: int) -> bool:
        if row == n:
            return True
        for col in range(n):
            if col in cols or (row + col) in pos_diag or (row - col) in neg_diag:
                continue
            board[row] = col
            cols.add(col)
            pos_diag.add(row + col)
            neg_diag.add(row - col)

            if backtrack(row + 1):
                return True

            board[row] = -1
            cols.remove(col)
            pos_diag.remove(row + col)
            neg_diag.remove(row - col)
        return False

    return board if backtrack(0) else None


if __name__ == "__main__":
    print("=" * 70)
    print("CSP PART 1: AUSTRALIA MAP COLORING (MRV + FORWARD CHECKING)")
    print("=" * 70)
    init_domains = {r: list(COLORS) for r in REGIONS}
    stats = {"assignments_tried": 0}
    coloring = solve_map_coloring({}, init_domains, stats)

    for region in REGIONS:
        print(f"  {region:<4} -> {coloring[region]}")
    print(f"Total Assignments Tried (with Forward Checking): {stats['assignments_tried']}")

    print("\n" + "=" * 70)
    print("CSP PART 2: 8-QUEENS PROBLEM (BACKTRACKING + DIAGONAL CONSISTENCY)")
    print("=" * 70)
    n = 8
    queens = solve_n_queens(n)
    print(f"Queen Column Positions (Row 0 to {n-1}): {queens}\n")
    for r in range(n):
        row_str = " ".join("Q" if queens[r] == c else "." for c in range(n))
        print(f"  {row_str}")
