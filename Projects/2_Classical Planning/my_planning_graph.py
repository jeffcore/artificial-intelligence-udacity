
from itertools import chain, combinations
from aimacode.planning import Action
from aimacode.utils import expr

from layers import BaseActionLayer, BaseLiteralLayer, makeNoOp, make_node


class ActionLayer(BaseActionLayer):

    def _inconsistent_effects(self, actionA, actionB):
        """ Return True if an effect of one action negates an effect of the other

        Hints:
            (1) `~Literal` can be used to logically negate a literal
            (2) `self.children` contains a map from actions to effects

        See Also
        --------
        layers.ActionNode
        """
        # TODO: implement this function       

        # look for negations of actions in a in actions in b
        for action_effect_a in actionA.effects:
            if ~action_effect_a in actionB.effects:
                return True

        return False


    def _interference(self, actionA, actionB):
        """ Return True if the effects of either action negate the preconditions of the other 

        Hints:
            (1) `~Literal` can be used to logically negate a literal
            (2) `self.parents` contains a map from actions to preconditions
        
        See Also
        --------
        layers.ActionNode
        """
        # TODO: implement this function
        
        # loop through preconditions of actionB -
        #   search for negation of actionB in actionA effects
        for action_precondition_b in actionB.preconditions:
            if ~action_precondition_b in actionA.effects:
                return True

        return False



    def _competing_needs(self, actionA, actionB):
        """ Return True if any preconditions of the two actions are pairwise mutex in the parent layer

        Hints:
            (1) `self.parent_layer` contains a reference to the previous literal layer
            (2) `self.parents` contains a map from actions to preconditions
        
        See Also
        --------
        layers.ActionNode
        layers.BaseLayer.parent_layer
        """
        # TODO: implement this function

        # look through all preconditions of actionA and actionB
        #    check to see if they are mutex of parent layer
        for actionA_precondition in actionA.preconditions:
            for actionB_precondition in actionB.preconditions: 
                if self.parent_layer.is_mutex(actionA_precondition,actionB_precondition):
                    return True
              
        return False

class LiteralLayer(BaseLiteralLayer):

    def _inconsistent_support(self, literalA, literalB):
        """ Return True if all ways to achieve both literals are pairwise mutex in the parent layer

        Hints:
            (1) `self.parent_layer` contains a reference to the previous action layer
            (2) `self.parents` contains a map from literals to actions in the parent layer

        See Also
        --------
        layers.BaseLayer.parent_layer
        """
        # TODO: implement this function     
        
        # loop though parent actions of literal A and B
        #   see if any of them are not mutexs
        for literalA_actions in self.parents[literalA]:
            for literalB_actions in self.parents[literalB]:      
                if not self.parent_layer.is_mutex(literalA_actions,literalB_actions):
                    return False
                            
        return True


    def _negation(self, literalA, literalB):
        """ Return True if two literals are negations of each other """
        # TODO: implement this function
        return literalA == ~literalB



class PlanningGraph:
    def __init__(self, problem, state, serialize=True, ignore_mutexes=False):
        """
        Parameters
        ----------
        problem : PlanningProblem
            An instance of the PlanningProblem class

        state : tuple(bool)
            An ordered sequence of True/False values indicating the literal value
            of the corresponding fluent in problem.state_map

        serialize : bool
            Flag indicating whether to serialize non-persistence actions. Actions
            should NOT be serialized for regression search (e.g., GraphPlan), and
            _should_ be serialized if the planning graph is being used to estimate
            a heuristic
        """
        self._serialize = serialize
        self._is_leveled = False
        self._ignore_mutexes = ignore_mutexes
        self.goal = set(problem.goal)

        # make no-op actions that persist every literal to the next layer
        no_ops = [make_node(n, no_op=True) for n in chain(*(makeNoOp(s) for s in problem.state_map))]
        self._actionNodes = no_ops + [make_node(a) for a in problem.actions_list]
        
        # initialize the planning graph by finding the literals that are in the
        # first layer and finding the actions they they should be connected to
        literals = [s if f else ~s for f, s in zip(state, problem.state_map)]
        layer = LiteralLayer(literals, ActionLayer(), self._ignore_mutexes)
        layer.update_mutexes()
        self.literal_layers = [layer]
        self.action_layers = []

    def level_cost(self):
        """
        improving the efficiency to analyze graph, open each new layer and analyze
        returns each goal and their layer cost  in a dict      
        """
        # track cost and if goal is complete
        goal_cost = {}
        goal_complete = {}

        for g in self.goal:
            goal_cost[g] = 0
            goal_complete[g] = False
        
        level_count = 0
        # is_leveled means literal layer is repeating itself
        while not self._is_leveled:
            # get most current layer
            layer_extended = self.literal_layers[-1]

            # loop though literals of layer   
            for l_literal in layer_extended:
                # if literal is a goal
                if l_literal in self.goal:
                    # set completed and level count if not already done                    
                    if not goal_complete[l_literal]:
                            goal_complete[l_literal] = True
                            goal_cost[l_literal] = level_count

            # check for all goals satisfied
            all_goals = goal_complete.values()
            if all(all_goals):
                break

            level_count += 1

            # add another layer to graph
            self._extend()

        return goal_cost


    def h_levelsum(self):
        """ Calculate the level sum heuristic for the planning graph

        The level sum is the sum of the level costs of all the goal literals
        combined. The "level cost" to achieve any single goal literal is the
        level at which the literal first appears in the planning graph. Note
        that the level cost is **NOT** the minimum number of actions to
        achieve a single goal literal.
        
        For example, if Goal_1 first appears in level 0 of the graph (i.e.,
        it is satisfied at the root of the planning graph) and Goal_2 first
        appears in level 3, then the levelsum is 0 + 3 = 3.

        Hints
        -----
          (1) See the pseudocode folder for help on a simple implementation
          (2) You can implement this function more efficiently than the
              sample pseudocode if you expand the graph one level at a time
              and accumulate the level cost of each goal rather than filling
              the whole graph at the start.

        See Also
        --------
        Russell-Norvig 10.3.1 (3rd Edition)
        """
        # TODO: implement this function
        level_cost = self.level_cost()

        level_sum = 0

        for i, v in level_cost.items():
            level_sum += v

        return level_sum

    def h_maxlevel(self):
        """ Calculate the max level heuristic for the planning graph

        The max level is the largest level cost of any single goal fluent.
        The "level cost" to achieve any single goal literal is the level at
        which the literal first appears in the planning graph. Note that
        the level cost is **NOT** the minimum number of actions to achieve
        a single goal literal.

        For example, if Goal1 first appears in level 1 of the graph and
        Goal2 first appears in level 3, then the levelsum is max(1, 3) = 3.

        Hints
        -----
          (1) See the pseudocode folder for help on a simple implementation
          (2) You can implement this function more efficiently if you expand
              the graph one level at a time until the last goal is met rather
              than filling the whole graph at the start.

        See Also
        --------
        Russell-Norvig 10.3.1 (3rd Edition)

        Notes
        -----
        WARNING: you should expect long runtimes using this heuristic with A*
        """
        # TODO: implement maxlevel heuristic
        level_cost = self.level_cost()
        level_cost_list = []

        for i,v in level_cost.items():
            level_cost_list.append(v)

        return max(level_cost_list)

    def h_setlevel(self):
        """ Calculate the set level heuristic for the planning graph

        The set level of a planning graph is the first level where all goals
        appear such that no pair of goal literals are mutex in the last
        layer of the planning graph.

        Hints
        -----
          (1) See the pseudocode folder for help on a simple implementation
          (2) You can implement this function more efficiently if you expand
              the graph one level at a time until you find the set level rather
              than filling the whole graph at the start.

        See Also
        --------
        Russell-Norvig 10.3.1 (3rd Edition)

        Notes
        -----
        WARNING: you should expect long runtimes using this heuristic on complex problems
        """
        # TODO: implement setlevel heuristic
        level_count = 0

        # followed pseudocode and added in efficiency graph building
        while not self._is_leveled:
            all_goals_met = True
            layer_extended = self.literal_layers[-1]
            
            for g in self.goal:
                if g not in layer_extended:
                    all_goals_met = False
            
            if not all_goals_met:
                level_count += 1
                self._extend()
                continue

            goals_are_mutex = False

            for goal_1 in self.goal:
                for goal_2 in self.goal:
                    if layer_extended.is_mutex(goal_1, goal_2):
                        goals_are_mutex = True

            if not goals_are_mutex:
                return level_count

            level_count += 1
            self._extend()
        
        return level_count
        


    ##############################################################################
    #                     DO NOT MODIFY CODE BELOW THIS LINE                     #
    ##############################################################################

    def fill(self, maxlevels=-1):
        """ Extend the planning graph until it is leveled, or until a specified number of
        levels have been added

        Parameters
        ----------
        maxlevels : int
            The maximum number of levels to extend before breaking the loop. (Starting with
            a negative value will never interrupt the loop.)

        Notes
        -----
        YOU SHOULD NOT THIS FUNCTION TO COMPLETE THE PROJECT, BUT IT MAY BE USEFUL FOR TESTING
        """
        while not self._is_leveled:
            if maxlevels == 0: break
            self._extend()
            maxlevels -= 1
        return self

    def _extend(self):
        """ Extend the planning graph by adding both a new action layer and a new literal layer

        The new action layer contains all actions that could be taken given the positive AND
        negative literals in the leaf nodes of the parent literal level.

        The new literal layer contains all literals that could result from taking each possible
        action in the NEW action layer. 
        """
        if self._is_leveled: return

        parent_literals = self.literal_layers[-1]
        parent_actions = parent_literals.parent_layer
        action_layer = ActionLayer(parent_actions, parent_literals, self._serialize, self._ignore_mutexes)
        literal_layer = LiteralLayer(parent_literals, action_layer, self._ignore_mutexes)

        for action in self._actionNodes:
            # actions in the parent layer are skipped because are added monotonically to planning graphs,
            # which is performed automatically in the ActionLayer and LiteralLayer constructors
            if action not in parent_actions and action.preconditions <= parent_literals:
                action_layer.add(action)
                literal_layer |= action.effects

                # add two-way edges in the graph connecting the parent layer with the new action
                parent_literals.add_outbound_edges(action, action.preconditions)
                action_layer.add_inbound_edges(action, action.preconditions)

                # # add two-way edges in the graph connecting the new literaly layer with the new action
                action_layer.add_outbound_edges(action, action.effects)
                literal_layer.add_inbound_edges(action, action.effects)

        action_layer.update_mutexes()
        literal_layer.update_mutexes()
        self.action_layers.append(action_layer)
        self.literal_layers.append(literal_layer)
        self._is_leveled = literal_layer == action_layer.parent_layer
