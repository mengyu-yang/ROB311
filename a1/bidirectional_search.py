from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem
import time
import numpy


def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

        :param problem: instance of SimpleSearchProblem
        :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
                 num_nodes_expanded: number of nodes expanded by your search
                 max_frontier_size: maximum frontier size during search
        """

    # This is the breadth-first search function which gets called each time we want to search in either direction.
    # The direction of the search is specified by the frontier parameter, where we either pass in frontier when
    # searching in the forwards direction or goaltier when searching in the backwards direction. Explored_curr is
    # the dynamic explored set which we add to when searching while explored_const is the explored set in the opposite
    # direction which we do not alter and only use to check for collisions. This BFS implementation only searches to
    # nodes of the same depth, which is ensured by setting orig_length to the length of the original queue and then
    # decrementing it with a while loop.

    def BFS(frontier, explored_curr, explored_const, num_nodes_expanded):

        # Calculate number of nodes within same depth in queue
        orig_length = len(frontier.copy())

        size = len(frontier.copy())

        # The found flag informs whether or not we have found a collision
        found = False

        while orig_length > 0:
            size = max(size, len(frontier))
            u = frontier.pop()
            orig_length -= 1

            for action in problem.get_actions(u.state):
                child = problem.get_child_node(u, action)
                num_nodes_expanded += 1

                if child.state in explored_const:
                    found = True
                    return child, found, explored_curr, frontier, size, num_nodes_expanded

                elif child.state not in explored_curr:
                    explored_curr.add(child.state)
                    frontier.appendleft(child)

        return None, found, explored_curr, frontier, size, num_nodes_expanded

    # This function, given the collision node, traces the path from itself to both the goal state and init state.
    # Depending on the direction parameter, the path works for both directions.
    def find_path(node, frontier, direction):
        if direction == 'f':
            forward_path = problem.trace_path(node)
            meeting_node = None
            while True:
                meeting_node = frontier.pop()
                if meeting_node.state == node.state:
                    break
            back_path = problem.trace_path(meeting_node, problem.goal_states[0])
            return forward_path[:-1] + back_path[::-1]

        if direction == 'b':
            back_path = problem.trace_path(node, problem.goal_states[0])
            meeting_node = None
            while True:
                meeting_node = frontier.pop()
                if meeting_node.state == node.state:
                    break
            forward_path = problem.trace_path(meeting_node)
            return forward_path[:-1] + back_path[::-1]

    max_frontier_size = 0
    num_nodes_expanded = 0

    # To handle case when init_state is passed in as list, like the facebook test case
    if type(problem.init_state) == list:
        problem.init_state = problem.init_state[0]

    # Test if init_state is goal_state. If so, return
    if problem.init_state in problem.goal_states:
        return [problem.init_state], 1, 1

    # Initialize frontier queue (used in forward search) with init_state Node and goaltier queue (used in backward search)
    # with goal_state Node
    frontier = deque([Node(None, problem.init_state, None, 0)])
    goaltier = deque([Node(None, problem.goal_states[0], None, 0)])
    frontier_size = 1
    goaltier_size = 1

    # Initialize explored set for both forwards and backwards directions
    explored_f = set()
    explored_b = set()
    explored_f.add(problem.init_state)
    explored_b.add(problem.goal_states[0])

    while len(frontier) != 0 and len(goaltier) != 0:
        # Update max_frontier_size each time we update frontier queue
        max_frontier_size = max(max_frontier_size, frontier_size + goaltier_size)

        # Run BFS in the forward direction
        intersect, found, explored_f, frontier, frontier_size, num_nodes_expanded = BFS(frontier, explored_f,
                                                                                        explored_b, num_nodes_expanded)
        # Find and return if we have found a path
        if found:
            path = find_path(intersect, goaltier, 'f')
            return path, num_nodes_expanded, max_frontier_size

        # Run BFS in the backward direction
        intersect, found, explored_b, goaltier, goaltier_size, num_nodes_expanded = BFS(goaltier, explored_b,
                                                                                        explored_f, num_nodes_expanded)
        # Find and return if have found a path
        if found:
            path = find_path(intersect, frontier, 'b')
            return path, num_nodes_expanded, max_frontier_size

    return [], num_nodes_expanded, max_frontier_size


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
                  [1, 7],
                  [2, 5],
                  [9, 4]])

    # problem = GraphSearchProblem(goal_states, init_state, V, E)
    # path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    # correct = problem.check_graph_solution(path)
    # print("Solution is correct: {:}".format(correct))

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
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(twitter_problem)
    end = time.time()
    print(end - start)
    correct = twitter_problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Be sure to compare with breadth_first_search!
