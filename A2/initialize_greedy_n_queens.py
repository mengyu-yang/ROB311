import numpy as np


### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def initialize_greedy_n_queens(N: int) -> list:
    """
    This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
    assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
    produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
    algorithm's performance!

    Example:
    Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

     _ _ _ _
    |Q|_|_|_|
    |_|_|Q|_|
    |_|_|_|Q|
    |_|Q|_|_|

    which has one diagonal conflict between its two rightmost columns.

    You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
    as any additional import statements will be removed by the autograder.

    :param N: integer representing the size of the NxN chessboard
    :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
    conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
    """

    # Given position of a queen, updates the board by incrementing all grids reachable by the queen by 1
    def get_moves(N, position, board):
        row, col = position

        for i in range(1, N):
            # Up_right
            if 0 <= row - i < N and 0 <= col + i < N:
                board[row - i][col + i] += 1
            # Up_left
            if 0 <= row - i < N and 0 <= col - i < N:
                board[row - i][col - i] += 1
            # Down_right
            if 0 <= row + i < N and 0 <= col + i < N:
                board[row + i][col + i] += 1
            # Down_left
            if 0 <= row + i < N and 0 <= col - i < N:
                board[row + i][col - i] += 1
            # Right
            if 0 <= row < N and 0 <= col + i < N:
                board[row][col + i] += 1
            # Left
            if 0 <= row < N and 0 <= col - i < N:
                board[row][col - i] += 1

        return board

    greedy_init = np.zeros(N, dtype=int)
    # First queen goes in a random spot
    greedy_init[0] = np.random.randint(0, N)

    # Initialize board and first queen
    board = np.zeros([N, N])
    board[greedy_init[0]][0] = np.inf
    board = get_moves(N, (greedy_init[0], 0), board)

    # Using a greedy method, iterate through each column of the board and assign a queen to the row with least conflicts
    for col in range(1, N):
        # Randomly choose among squares with minimum conflict
        greedy_init[col] = np.random.choice(np.argwhere(board[:, col] == np.amin(board[:, col])).flatten(), 1)

        # Add a queen there
        board[greedy_init[col]][col] = np.inf

        # Update board
        board = get_moves(N, (greedy_init[col], col), board)

    return greedy_init


if __name__ == '__main__':
    # You can test your code here
    print(initialize_greedy_n_queens(5))
