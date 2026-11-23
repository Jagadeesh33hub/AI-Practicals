"""
Practical 4: Adversarial Search - Minimax & Alpha-Beta Pruning for Tic-Tac-Toe
Aim: Implement optimal game-playing agents using standard Minimax and Alpha-Beta
     pruning, and measure the reduction in evaluated game-tree states.
"""

import math

AI_PLAYER = "X"
HUMAN_PLAYER = "O"
EMPTY = " "


def print_board(board: list[str]) -> None:
    for r in range(3):
        row = " | ".join(board[r * 3 : (r + 1) * 3])
        print(f"  {row}")
        if r < 2:
            print("  ---------")


def check_winner(board: list[str]) -> str | None:
    win_lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]
    for a, b, c in win_lines:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    if EMPTY not in board:
        return "Draw"
    return None


def minimax(board: list[str], depth: int, is_maximizing: bool, counter: list[int]) -> int:
    counter[0] += 1
    result = check_winner(board)
    if result == AI_PLAYER:
        return 10 - depth
    if result == HUMAN_PLAYER:
        return depth - 10
    if result == "Draw":
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = AI_PLAYER
                score = minimax(board, depth + 1, False, counter)
                board[i] = EMPTY
                best_score = max(best_score, score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = HUMAN_PLAYER
                score = minimax(board, depth + 1, True, counter)
                board[i] = EMPTY
                best_score = min(best_score, score)
        return best_score


def alphabeta(
    board: list[str], depth: int, alpha: float, beta: float, is_maximizing: bool, counter: list[int]
) -> int:
    counter[0] += 1
    result = check_winner(board)
    if result == AI_PLAYER:
        return 10 - depth
    if result == HUMAN_PLAYER:
        return depth - 10
    if result == "Draw":
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = AI_PLAYER
                score = alphabeta(board, depth + 1, alpha, beta, False, counter)
                board[i] = EMPTY
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = HUMAN_PLAYER
                score = alphabeta(board, depth + 1, alpha, beta, True, counter)
                board[i] = EMPTY
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
        return best_score


def find_best_move(board: list[str], use_alpha_beta: bool = True) -> tuple[int, int, int]:
    best_val = -math.inf
    best_move = -1
    nodes_evaluated = [0]

    for i in range(9):
        if board[i] == EMPTY:
            board[i] = AI_PLAYER
            if use_alpha_beta:
                move_val = alphabeta(board, 0, -math.inf, math.inf, False, nodes_evaluated)
            else:
                move_val = minimax(board, 0, False, nodes_evaluated)
            board[i] = EMPTY

            if move_val > best_val:
                best_val = move_val
                best_move = i

    return best_move, best_val, nodes_evaluated[0]


if __name__ == "__main__":
    sample_board = [
        "X", "O", "X",
        " ", "O", " ",
        " ", " ", " ",
    ]

    print("=" * 68)
    print("TIC-TAC-TOE AI: MINIMAX VS. ALPHA-BETA PRUNING")
    print("=" * 68)
    print("Current Board (AI = X, Opponent = O):")
    print_board(sample_board)

    move_mm, score_mm, nodes_mm = find_best_move(sample_board.copy(), use_alpha_beta=False)
    move_ab, score_ab, nodes_ab = find_best_move(sample_board.copy(), use_alpha_beta=True)

    reduction = ((nodes_mm - nodes_ab) / nodes_mm) * 100

    print("\nComparison on Mid-Game State:")
    print(f"  Standard Minimax     -> Best Cell: {move_mm} | Score: {score_mm} | States Evaluated: {nodes_mm}")
    print(f"  Alpha-Beta Pruning   -> Best Cell: {move_ab} | Score: {score_ab} | States Evaluated: {nodes_ab}")
    print(f"  Search Tree Pruning  -> {reduction:.2f}% fewer nodes evaluated!")

    print("\n" + "-" * 68)
    print("FULL OPTIMAL SELF-PLAY SIMULATION (X vs O)")
    print("-" * 68)
    board = [EMPTY] * 9
    turn = AI_PLAYER
    step = 1
    while check_winner(board) is None:
        if turn == AI_PLAYER:
            idx, _, _ = find_best_move(board, use_alpha_beta=True)
            board[idx] = AI_PLAYER
            print(f"Move {step}: X plays cell {idx}")
            turn = HUMAN_PLAYER
        else:
            best_o_val = math.inf
            best_o_move = -1
            dummy = [0]
            for i in range(9):
                if board[i] == EMPTY:
                    board[i] = HUMAN_PLAYER
                    val = alphabeta(board, 0, -math.inf, math.inf, True, dummy)
                    board[i] = EMPTY
                    if val < best_o_val:
                        best_o_val = val
                        best_o_move = i
            board[best_o_move] = HUMAN_PLAYER
            print(f"Move {step}: O plays cell {best_o_move}")
            turn = AI_PLAYER
        step += 1

    print("\nFinal Board:")
    print_board(board)
    print(f"Game Outcome: {check_winner(board)} (Optimal play always yields a Draw)")
