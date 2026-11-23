"""
Practical 10: Sequential Decision Making - Value Iteration in Grid-World (MDP)
Aim: Compute optimal state utilities V*(s) and extract the optimal policy pi*(s)
     for Russell & Norvig's stochastic 4x3 Grid-World using Bellman updates.
"""

ROWS, COLS = 3, 4
WALL = (1, 1)
TERMINAL_REWARDS = {
    (0, 3): +1.0,
    (1, 3): -1.0,
}
LIVING_REWARD = -0.04
GAMMA = 0.99
THETA = 1e-6

ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

PERPENDICULAR = {
    "UP": ["LEFT", "RIGHT"],
    "DOWN": ["LEFT", "RIGHT"],
    "LEFT": ["UP", "DOWN"],
    "RIGHT": ["UP", "DOWN"],
}

ARROWS = {"UP": "^", "DOWN": "v", "LEFT": "<", "RIGHT": ">"}


def step_state(state: tuple[int, int], action: str) -> tuple[int, int]:
    dr, dc = ACTIONS[action]
    nr, nc = state[0] + dr, state[1] + dc
    if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) != WALL:
        return (nr, nc)
    return state


def expected_utility(state: tuple[int, int], action: str, V: dict[tuple[int, int], float]) -> float:
    """Stochastic transition model: 0.8 intended direction, 0.1 each perpendicular."""
    transitions = [(0.8, step_state(state, action))]
    for perp in PERPENDICULAR[action]:
        transitions.append((0.1, step_state(state, perp)))

    total = 0.0
    for prob, next_s in transitions:
        reward = TERMINAL_REWARDS.get(next_s, LIVING_REWARD)
        total += prob * (reward + GAMMA * V[next_s])
    return total


def value_iteration() -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], str], int]:
    states = [
        (r, c)
        for r in range(ROWS)
        for c in range(COLS)
        if (r, c) != WALL
    ]
    V = {s: 0.0 for s in states}
    iterations = 0

    while True:
        delta = 0.0
        V_new = dict(V)
        iterations += 1

        for s in states:
            if s in TERMINAL_REWARDS:
                continue
            V_new[s] = max(expected_utility(s, a, V) for a in ACTIONS)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if delta < THETA:
            break

    policy = {}
    for s in states:
        if s in TERMINAL_REWARDS:
            policy[s] = f"{TERMINAL_REWARDS[s]:+.0f}"
        else:
            best_a = max(ACTIONS, key=lambda a: expected_utility(s, a, V))
            policy[s] = ARROWS[best_a]

    return V, policy, iterations


if __name__ == "__main__":
    V, policy, iters = value_iteration()

    print("=" * 68)
    print(f"GRID-WORLD MDP VALUE ITERATION (Converged in {iters} iterations)")
    print("=" * 68)
    print(f"Parameters: Gamma={GAMMA}, Living Reward={LIVING_REWARD}, Slip Prob=0.2\n")

    print("Optimal State Values V*(s):")
    for r in range(ROWS):
        row_cells = []
        for c in range(COLS):
            if (r, c) == WALL:
                row_cells.append("   WALL  ")
            elif (r, c) in TERMINAL_REWARDS:
                row_cells.append(f" {TERMINAL_REWARDS[(r, c)]:+6.2f}T ")
            else:
                row_cells.append(f" {V[(r, c)]:+7.4f} ")
        print("  " + " | ".join(row_cells))

    print("\nOptimal Policy pi*(s):")
    for r in range(ROWS):
        row_cells = []
        for c in range(COLS):
            if (r, c) == WALL:
                row_cells.append(" WALL ")
            else:
                row_cells.append(f"  {policy[(r, c)]:^2}  ")
        print("  " + " | ".join(row_cells))
