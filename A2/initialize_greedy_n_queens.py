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

    # Given position index of a queen, return a list of the indices of all valid moves for that queen
    def get_moves(N, position):
        row, col = position

        up_right = [(row - i, col + i) for i in range(1, N) if 0 <= row - i < N and 0 <= col + i < N]
        up_left = [(row - i, col - i) for i in range(1, N) if 0 <= row - i < N and 0 <= col - i < N]
        down_right = [(row + i, col + i) for i in range(1, N) if 0 <= row + i < N and 0 <= col + i < N]
        down_left = [(row + i, col - i) for i in range(1, N) if 0 <= row + i < N and 0 <= col - i < N]

        up = [(row - i, col) for i in range(1, N) if 0 <= row - i < N and 0 <= col < N]
        down = [(row + i, col) for i in range(1, N) if 0 <= row + i < N and 0 <= col < N]
        right = [(row, col + i) for i in range(1, N) if 0 <= row < N and 0 <= col + i < N]
        left = [(row, col - i) for i in range(1, N) if 0 <= row < N and 0 <= col - i < N]

        return up_right + up_left + down_right + down_left + up + down + right + left

    greedy_init = np.zeros(N)
    # First queen goes in a random spot
    greedy_init[0] = np.random.randint(0, N)

    # Using a greedy method, iterate through each column of the board and assign a queen to the row with least conflicts
    for col in range(1, N):

        # Define list where each element corresponds to a row in the column and the value is the number of conflicts a
        # queen in that row will have
        row_list = np.zeros(N)

        # Iterate through each row within the column
        for row in range(1, N):
            # Get list of valid moves for a queen at this position
            moves = get_moves(N, (row, col))
            # Check if any queen positions are in the moves list. If so, this means there is a conflict. This loop
            # calculates the total number of conflicts for a queen in this position.
            for q in range(0, col):
                if (greedy_init[q], q) in moves:
                    row_list[row] += 1

        # Record the indices, which correspond to rows, with the least conflicts
        indices = [i for i, v in enumerate(row_list) if v == min(row_list)]
        indices = np.asarray(indices)

        # Randomly select a row which minimizes conflicts
        greedy_init[col] = np.random.choice(indices, 1)

    return greedy_init


if __name__ == '__main__':
    # You can test your code here
    print(initialize_greedy_n_queens(3))
