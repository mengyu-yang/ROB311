import numpy as np


### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def min_conflicts_n_queens(initialization: list) -> (list, int):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """

    def get_moves(N, position, board, subtract=False):
        row, col = position

        if subtract:
            factor = -1
        else:
            factor = 1

        for i in range(1, N):
            # Up_right
            if 0 <= row - i < N and 0 <= col + i < N:
                board[row - i][col + i] += factor
            # Up_left
            if 0 <= row - i < N and 0 <= col - i < N:
                board[row - i][col - i] += factor
            # Down_right
            if 0 <= row + i < N and 0 <= col + i < N:
                board[row + i][col + i] += factor
            # Down_left
            if 0 <= row + i < N and 0 <= col - i < N:
                board[row + i][col - i] += factor
            # Right
            if 0 <= row < N and 0 <= col + i < N:
                board[row][col + i] += factor
            # Left
            if 0 <= row < N and 0 <= col - i < N:
                board[row][col - i] += factor
        return board

    N = len(initialization)
    solution = initialization.copy()
    num_steps = 0
    max_steps = 1000

    # Initialize board
    board = np.zeros([N, N])
    for col, row in enumerate(solution):
        board = get_moves(N, (row, col), board, subtract=False)

    for idx in range(max_steps):
        # Check solution
        flag = True
        for i in range(0, N):
            if not np.isin(0, board[:, i]):
                flag = False
                break
        if flag:
            return solution, num_steps

        # Choose a random column containing conflicts
        mask = np.array([])
        for col, row in enumerate(solution):
            if board[row][col] > 0:
                mask = np.append(mask, col)
        rand_col = np.int(np.random.choice(mask, 1))

        # Store current position of the queen to be used later when we need to take away conflicts
        prev = solution[rand_col]

        # Choose random square with minimum conflicts in column to move to
        indices = np.argwhere(board[:, rand_col] == np.amin(board[:, rand_col])).flatten()
        solution[rand_col] = np.int(np.random.choice(indices, 1))

        # Update board by first removing conflicts associated with previous queen position
        board = get_moves(N, (prev, rand_col), board, subtract=True)
        # Update board by adding conflicts created by new queen position
        board = get_moves(N, (solution[rand_col], rand_col), board, subtract=False)

        # Update step count
        num_steps += 1

    return solution, num_steps


if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    N = 10
    # Use this after implementing initialize_greedy_n_queens.py
    assignment_initial = initialize_greedy_n_queens(N)
    # Plot the initial greedy assignment

    plot_n_queens_solution(assignment_initial)

    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)
    # Plot the solution produced by your algorithm
    plot_n_queens_solution(assignment_solved)
