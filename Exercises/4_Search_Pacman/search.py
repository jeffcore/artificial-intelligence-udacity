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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from game import Directions
    visited = {}
   
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST 

    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
        
    start_state = (problem.getStartState(), None, 0)  
    path = []  
    
    result = dfs_helper(problem, path, visited, start_state)    
   
    # print(path)
    return path

def dfs_helper(problem, path, visited, state_node):    
    
    visited[state_node[0]] = True
    done = False   
        
    if problem.isGoalState(state_node[0]):              
        # print('added winner ' , state_node)  
        return True
    
    successors = problem.getSuccessors(state_node[0])

    for block in successors:    
        if block[0] not in visited:  
            # print('not visited ', block) 
                                        
            done = dfs_helper(problem, path, visited, block)
            
        if done:
            path.insert(0, block[1])    
            break
        
    return done

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontier = util.Queue()
    explored = set()

    start_node = problem.getStartState()    
    
    if problem.isGoalState(start_node):              
        return []
    
    frontier.push((start_node, []))

    while not frontier.isEmpty():
        node, actions = frontier.pop()
        if node not in explored:
            explored.add(node)

            if problem.isGoalState(node):              
                return actions

            successors = problem.getSuccessors(node)
            for child_node in successors:
                new_action = actions + [child_node[1]]
                frontier.push((child_node[0], new_action))
    
    
    # startingNode = problem.getStartState()
    # if problem.isGoalState(startingNode):        
    #     return []

    # myQueue = util.Queue()
    # visitedNodes = []
    # # (node,actions)
    # myQueue.push((startingNode, []))

    # while not myQueue.isEmpty():
    #     currentNode, actions = myQueue.pop()
    #     if currentNode not in visitedNodes:
    #         visitedNodes.append(currentNode)

    #         if problem.isGoalState(currentNode):
    #             return actions

    #         for nextNode, action, cost in problem.getSuccessors(currentNode):
    #             newAction = actions + [action]
    #             myQueue.push((nextNode, newAction))

    # queue = util.Queue()
    # visited = set()
    # start_state = (problem.getStartState(), None, 0)  
    # path = []  
    # queue.push((start_state, []))
    # result = bfs_helper(problem, path, visited, queue)    
   
    # print(path)
    # return path
    util.raiseNotDefined()

def bfs_helper(problem, path, visited, queue):    
    
    if queue.isEmpty():
        return False
    state_node, actions = queue.pop()
    visited.ass(state_node[0])
    done = False   
        
    if problem.isGoalState(state_node[0]):              
        # print('added winner ' , state_node)  
        return True, path
    
    successors = problem.getSuccessors(state_node[0])

    for block in successors:    
        if block[0] not in visited:  
            queue.push(block)
    
    new_path = path.copy()
    new_path 

    done = bfs_helper(problem, path, visited, queue)

    if done and state_node[1] != None:
        path.insert(0, state_node[1])    
        
    return done 
    

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    frontier = util.PriorityQueue()
    explored = set()

    start_node = problem.getStartState()
    if problem.isGoalState(start_node):              
        # print('added winner ' , node)  
        return actions
    
    frontier.push((start_node, [], 0), 0)

    while not frontier.isEmpty():
        node, actions, priority = frontier.pop()
       
        if node not in explored:
            explored.add(node)
            if problem.isGoalState(node):              
                # print('added winner ' , node)  
                return actions

            successors = problem.getSuccessors(node)
            # print('successors ', successors)
            for child_node in successors:
                # print('child_node ', child_node)
                new_action = actions + [child_node[1]]
                new_priority = child_node[2] + priority
                frontier.push((child_node[0], new_action, new_priority), new_priority)
    
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    explored = set()

    start_node = problem.getStartState()
    if problem.isGoalState(start_node):              
        # print('added winner ' , node)  
        return actions
    
    frontier.push((start_node, [], 0), 0)

    while not frontier.isEmpty():
        node, actions, priority = frontier.pop()
       
        if node not in explored:
            explored.add(node)
            if problem.isGoalState(node):              
                # print('added winner ' , node)  
                return actions

            successors = problem.getSuccessors(node)
            # print('successors ', successors)
            for child_node in successors:
                # print('child_node ', child_node)
                new_action = actions + [child_node[1]]
                g = heuristic(child_node[0], problem)
                new_cost = child_node[2] + priority
                h_cost = new_cost + g
                
                frontier.push((child_node[0], new_action, new_cost), h_cost)
    
    util.raiseNotDefined()
    
    
    
    # frontier = util.PriorityQueue()
    # explored = set()
    # start_node = (problem.getStartState(), None, 0)      
    # frontier.push((start_node, [], 0), 0)
    # while True:
    #     # print('frontier ', frontier)
    #     # print('explored ', explored)
    #     if frontier.isEmpty():
    #         return None
    #     state = frontier.pop()
    #     node, actions, priority = state[0], state[1], state[2]
    #     # print('pop state' , state,' node ' , node , 'actions ', actions , ' priority ' , priority)
    #     if node not in explored:
    #         explored.add(node)
    #         if problem.isGoalState(node[0]):              
    #             # print('added winner ' , node)  
    #             return actions

    #         successors = problem.getSuccessors(node[0])
    #         # print('successors ', successors)
    #         for child_node in successors:
    #             # print('child_node ', child_node)
    #             newAction = actions + [child_node[1]]
    #             g = heuristic(child_node[0], problem)
    #             # print('g ', g)
    #             frontier.push((child_node, newAction, child_node[2]+priority+g), child_node[2]+priority+g)



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
