import sys

def is_winner(board, player):
    """Checks if the given player has won."""
    # Check rows
    for i in range(0, 9, 3):
        if all(board[i + j] == player for j in range(3)):
            return True
    # Check columns
    for i in range(3):
        if all(board[i + j] == player for j in range(0, 9, 3)):
            return True
    # Check diagonals
    if all(board[i] == player for i in [0, 4, 8]):
        return True
    if all(board[i] == player for i in [2, 4, 6]):
        return True
    return False

def is_full(board):
    """Checks if the board is full."""
    return ' ' not in board

def get_possible_moves(board):
    """Returns a list of indices of empty cells."""
    return [i for i, cell in enumerate(board) if cell == ' ']

def get_opponent(player):
    """Returns the opponent's mark."""
    return 'O' if player == 'X' else 'X'

def minimax(board, player):
    """
    Calculates the best move for the given player using the minimax algorithm.
    Returns a dictionary: {'score': score, 'move': move_index}
    """
    opponent = get_opponent(player)
    possible_moves = get_possible_moves(board)

    if is_winner(board, opponent):
        return {'score': -1 if player == 'X' else 1, 'move': None}
    if is_full(board):
        return {'score': 0, 'move': None}

    best_move = None
    if player == 'X':
        best_score = -2  # Initialize with negative infinity
        for move in possible_moves:
            new_board = list(board)
            new_board[move] = player
            score = minimax(tuple(new_board), opponent)['score']
            if score > best_score:
                best_score = score
                best_move = move
    else:  # player == 'O'
        best_score = 2   # Initialize with positive infinity
        for move in possible_moves:
            new_board = list(board)
            new_board[move] = player
            score = minimax(tuple(new_board), opponent)['score']
            if score < best_score:
                best_score = score
                best_move = move

    return {'score': best_score, 'move': best_move}

def generate_all_boards():
    """Generates all possible tic-tac-toe boards and their best next moves."""
    all_boards_with_moves = {}

    def generate(board):
        board_tuple = tuple(board)
        if board_tuple in all_boards_with_moves:
            return

        x_count = board.count('X')
        o_count = board.count('O')

        if is_winner(board, 'X') or is_winner(board, 'O') or is_full(board):
            all_boards_with_moves[board_tuple] = "Game Over"
            return

        if x_count == o_count:
            # X's turn
            best_move_info = minimax(board_tuple, 'X')
            best_move = best_move_info['move']
            all_boards_with_moves[board_tuple] = best_move

            for i in get_possible_moves(board):
                new_board = list(board)
                new_board[i] = 'X'
                generate(new_board)
        elif x_count > o_count:
            # O's turn
            best_move_info = minimax(board_tuple, 'O')
            best_move = best_move_info['move']
            all_boards_with_moves[board_tuple] = best_move

            for i in get_possible_moves(board):
                new_board = list(board)
                new_board[i] = 'O'
                generate(new_board)

    generate([' '] * 9)
    return all_boards_with_moves

filename = "tic_tac_toe_boards_with_moves.txt"

if __name__ == "__main__":
    print("Generating all possible Tic-Tac-Toe boards and best moves...")
    boards_and_moves = generate_all_boards()
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    with open(filename, "w") as f:
        f.write(f"Total number of boards: {len(boards_and_moves)}\n\n")
        for board_tuple, best_move in boards_and_moves.items():
            board_str = "".join(board_tuple)
            f.write(f"Board: {board_str}, Best_Next_Move: {best_move}\n")

    print("All boards and best moves have been saved to tic_tac_toe_boards_with_moves.txt")
