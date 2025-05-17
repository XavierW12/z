# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
      # Use the provided Stack for our fringe
    fringe_stack = util.Stack()
    start_tile = problem.getStartState()
    # Each fringe entry: (current_state, path_taken_to_reach_it)
    fringe_stack.push((start_tile, []))
    
    visited_tiles = set()
    
    while not fringe_stack.isEmpty():
        current_tile, path_to_tile = fringe_stack.pop()
        
        # Goal test
        if problem.isGoalState(current_tile):
            return path_to_tile
        
        # If we haven’t expanded this state yet:
        if current_tile not in visited_tiles:
            visited_tiles.add(current_tile)
            
            # Expand successors in the order returned by getSuccessors
            for succ_tile, move_dir, step_cost in problem.getSuccessors(current_tile):
                if succ_tile not in visited_tiles:
                    # Push successor onto fringe with updated path
                    fringe_stack.push((succ_tile, path_to_tile + [move_dir]))
    
    # No path found (shouldn’t happen on a solvable maze)
    return []
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
        # Use the provided Queue for our fringe
    queue_frontier = util.Queue()
    start_node = problem.getStartState()
    # Each entry: (current_state, path_to_state)
    queue_frontier.push((start_node, []))
    
    seen_states = set([start_node])
    
    while not queue_frontier.isEmpty():
        current_state, path_to_state = queue_frontier.pop()
        
        # Goal test
        if problem.isGoalState(current_state):
            return path_to_state
        
        # Expand successors
        for successor, action, step_cost in problem.getSuccessors(current_state):
            if successor not in seen_states:
                seen_states.add(successor)
                queue_frontier.push((successor, path_to_state + [action]))
    
    # No solution found
    return []
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    # Fringe is a priority queue ordered by cumulative path cost
    fringe_heap = util.PriorityQueue()
    start_node = problem.getStartState()
    # Each entry: (state, path_to_state, cost_so_far)
    fringe_heap.push((start_node, [], 0), 0)
    
    # Track the best cost we've found so far for each state
    explored_costs = { start_node: 0 }
    
    while not fringe_heap.isEmpty():
        current_state, path_to_state, current_cost = fringe_heap.pop()
        
        # If this is outdated (we've found a better way), skip it
        if current_cost > explored_costs.get(current_state, float('inf')):
            continue
        
        # Goal test
        if problem.isGoalState(current_state):
            return path_to_state
        
        # Expand successors
        for successor, action, step_cost in problem.getSuccessors(current_state):
            new_cost = current_cost + step_cost
            # If this path to 'successor' is better, record it and push onto fringe
            if successor not in explored_costs or new_cost < explored_costs[successor]:
                explored_costs[successor] = new_cost
                fringe_heap.push((successor, path_to_state + [action], new_cost), new_cost)
    
    # No solution found
    return []
    util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    # Fringe is a priority queue ordered by f(n) = g(n) + h(n)
    fringe_heap = util.PriorityQueue()
    start_node = problem.getStartState()
    start_h = heuristic(start_node, problem)
    # Each entry: (state, path_to_state, g_cost)
    fringe_heap.push((start_node, [], 0), start_h)
    
    # Track the best g(n) we've found so far for each state
    best_g = { start_node: 0 }
    
    while not fringe_heap.isEmpty():
        current_state, path_to_state, g_cost = fringe_heap.pop()
        
        # If this is outdated (we've found a better g-cost), skip it
        if g_cost > best_g.get(current_state, float('inf')):
            continue
        
        # Goal test
        if problem.isGoalState(current_state):
            return path_to_state
        
        # Expand successors
        for successor, action, step_cost in problem.getSuccessors(current_state):
            new_g = g_cost + step_cost
            # If this path to 'successor' is better, record it and push with new f = g + h
            if successor not in best_g or new_g < best_g[successor]:
                best_g[successor] = new_g
                f_cost = new_g + heuristic(successor, problem)
                fringe_heap.push((successor, path_to_state + [action], new_g), f_cost)
    
    # No solution found
    return []
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
