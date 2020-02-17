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


    max_frontier_size = 0
    num_nodes_expanded = 0
    depth_limit = 1

    # To handle case when init_state is passed in as list, like the facebook test case
    if type(problem.init_state) == list:
        problem.init_state = problem.init_state[0]

    # Test if init_state is goal_state. If so, return
    if problem.init_state in problem.goal_states:
        return [problem.init_state], 1, 0

    # Initialize frontier queue with init_state Node
    frontier = deque([Node(None, problem.init_state, None, None)])
    goaltier = deque([Node(None, problem.goal_states[0], None, None)])

    # Update max_frontier_size each time we update frontier queue
    max_frontier_size = max(max_frontier_size, len(frontier))

    # Initialize explored set
    explored = set()

    while len(frontier) != 0 and len(goaltier) != 0:
        # Pop the shallowest node in frontier and add to explored
        u = frontier.pop()
        explored.add(u.state)

        # Iterate over the adjacent nodes of u
        for adj in problem.neighbours[u.state]:
            # Initialize a Node class for the adjacent node
            child = Node(u, adj, None, None)

            if child.state not in explored:
                frontier.appendleft(child)
                max_frontier_size = max(max_frontier_size, len(frontier))


        # Pop shallowest node in goaltier and add to explored
        v = goaltier.pop()
        explored.add(v.state)

        # Iterate over the adjacent nodes of v
        for adj in problem.neighbours[v.state]:
            # Initialize a Node class for the adjacent node
            child = Node(v, adj, None, None)

            if child.state in [i.state for i in frontier]:
                backpath = problem.trace_path(child, problem.goal_states[0])
                meeting_state = backpath[-1]
                meeting_node = None
                while True:
                    meeting_node = frontier.pop()
                    if meeting_node.state == meeting_state:
                        break
                forward_path = problem.trace_path(meeting_node)
                print(forward_path, backpath)

                return forward_path[:-1] + backpath[::-1], None, None

            if child.state not in explored:
                goaltier.appendleft(child)
                max_frontier_size = max(max_frontier_size, len(frontier))



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
    print(end-start)
    correct = twitter_problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Be sure to compare with breadth_first_search!
