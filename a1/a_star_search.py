import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem


def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    # Initialize return values
    num_nodes_expanded = 0
    max_frontier_size = 0

    # Initialize priority queue as frontier and frontier_list, which keeps track of states stored in frontier for
    # easy iteration
    frontier = queue.PriorityQueue()
    frontier_list = []

    # We insert a tuple into the priority queue where the first element is the a-star cost and the second element is
    # the Node object. We also update frontier_list each time we update frontier.
    frontier.put((problem.heuristic(problem.init_state), Node(None, problem.init_state, None, problem.heuristic(problem.init_state))))
    max_frontier_size = max(max_frontier_size, len(frontier_list))

    # Initialize explored set
    explored = set()

    while True:
        if frontier.empty():
            return [], num_nodes_expanded, max_frontier_size

        # Pop lowest-cost node in frontier and update frontier_list
        u = frontier.get()[1]

        # Iterate over all possible actions from current node
        for action in problem.get_actions(u.state):

            # Initialize Node object for action
            child = problem.get_child_node(u, action)
            num_nodes_expanded += 1

            # If child is goal state, then return
            if problem.goal_test(child.state):
                return problem.trace_path(child), num_nodes_expanded, max_frontier_size

            # If the adjacent state has not been explored or in the frontier, then update frontier with it. We use
            # the heuristic of the sum of the parent's path_cost and the child's cost
            if child.state not in explored:
                frontier.put((u.path_cost + problem.heuristic(child.state), child))
                max_frontier_size = max(max_frontier_size, len(frontier_list))
                explored.add(child.state)


def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    ####
    #   REPLACE THESE VALUES
    ####
    transition_start_probability = 0.35
    transition_end_probability = 0.45
    peak_nodes_expanded_probability = 0.35
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 500
    N = 500
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)

    # Experiment and compare with BFS
