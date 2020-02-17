from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem
import time
import numpy

def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of SimpleSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """

    # Initialize values
    num_nodes_expanded = 0
    max_frontier_size = 0

    # To handle case when init_state is passed in as list, like the facebook test case
    if type(problem.init_state) == list:
        problem.init_state = problem.init_state[0]

    # Test if init_state is goal_state. If so, return
    if problem.init_state in problem.goal_states:
        return [problem.init_state], 1, 0

    # Initialize frontier queue with init_state Node
    frontier = deque([Node(None, problem.init_state, None, None)])

    # Update max_frontier_size each time we update frontier queue
    max_frontier_size = max(max_frontier_size, len(frontier))

    # Initialize explored set
    explored = set()

    while len(frontier) != 0:
        # Pop the shallowest node in frontier and add to explored
        u = frontier.pop()
        explored.add(u.state)

        # Iterate over the adjacent nodes of u
        for adj in problem.neighbours[u.state]:

            # Initialize a Node class for the adjacent node
            child = Node(u, adj, None, None)
            num_nodes_expanded += 1

            if child.state not in explored:
                # Return if we have reached the goal state
                if problem.goal_test(child.state):
                    return problem.trace_path(child), max_frontier_size, num_nodes_expanded

                # If goal state has not been reached, add the adjacent node to the queue
                frontier.appendleft(child)
                max_frontier_size = max(max_frontier_size, len(frontier))

    return [], max_frontier_size, num_nodes_expanded

if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [2, 5],
                  [1, 7],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = [0]
    problem = GraphSearchProblem(goal_states, init_state, V, E)

    E_twitter = numpy.load('twitter_edges_project_01.npy')
    V_twitter = numpy.unique(E_twitter)
    twitter_problem = GraphSearchProblem([59999], 0, V_twitter, E_twitter)

    start = time.time()
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(twitter_problem)
    end = time.time()
    print(end - start)
    correct = twitter_problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)