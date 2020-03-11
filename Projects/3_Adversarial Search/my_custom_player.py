
from sample_players import DataPlayer


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def __init__(self, player_id):
        super().__init__(player_id)

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        import random
        
        max_depth = 1
        # Opening book
        book = self.data
        
        while True:
            if state.ply_count < 4:
                if state in book:     
                    self.queue.put(book[state])                     
                    break         
                else:
                    if state.ply_count < 2:
                        self.queue.put(random.choice(state.actions()))
                        break
                    else:
                        self.queue.put(self.alpha_beta_search(state, max_depth))                                   
            else:                              
                if len(state.actions()) == 1:
                    self.queue.put(state.actions()[0])
                else:
                    # handle for losing terminal states
                    move = self.alpha_beta_search(state, max_depth)
                    if move:                        
                        self.queue.put(move)      
                    else:      
                        # terminal losing state                  
                        self.queue.put(random.choice(state.actions()))                        
                        break
            max_depth += 1
                   

    def alpha_beta_search(self, gameState, depth):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.
        
        You can ignore the special case of calling this function
        from a terminal state.
        """
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for action in gameState.actions():                        
            value = self.min_value(gameState.result(action), alpha, beta, depth - 1)
            alpha = max(alpha, value)
            if value > best_score:                
                best_score = value
                best_move = action
        # print('action ' , action , 'value ' , value , 'best score' , best_score, 'bsest move ' , best_move)    
        return best_move

    
    def min_value(self, gameState, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if gameState.terminal_test():
            # print('termainal test' , gameState.utility(self.player_id))
            return gameState.utility(self.player_id)
        
        if depth <= 0:            
            return self.score(gameState)
            
        value = float("inf")
        for action in gameState.actions():            
            # TODO: modify the call to max_value()
            value = min(value, self.max_value(gameState.result(action), alpha, beta, depth - 1))
            
            # TODO: update the value bound
            if value <= alpha:
                return value
            beta = min(beta, value)           
        return value

    
    def max_value(self, gameState, alpha, beta, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if gameState.terminal_test():
            return gameState.utility(self.player_id)
        
        if depth <= 0:
            return self.score(gameState)

        value = float("-inf")
        for action in gameState.actions():
            # TODO: modify the call to min_value()
            value = max(value, self.min_value(gameState.result(action), alpha, beta, depth - 1))
            # TODO: update the value bound
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def score_base(self, state):
        """Calculate the heuristic value of a game state from the point of view
            of the given player. Similiar to custom_score_3:

            1. only avoids center if player 1 has more moves than player 2
        """
        own_loc = state.locs[self.player_id]       
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
     
        return  len(own_liberties) -  len(opp_liberties)

    def score(self, state):
        """Calculate the heuristic value of a game state from the point of view
            of the given player. Similiar to custom_score_3:

            1. only avoids center if player 1 has more moves than player 2
        """
        AVG_PLY_GAME_LENGTH = 70
        own_loc = state.locs[self.player_id]       
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
       
        ratio = state.ply_count / AVG_PLY_GAME_LENGTH
       
        if ratio <= .5:
            # defensive strategy
            h_score = (len(own_liberties) * 2) - len(opp_liberties)                         
        else:
            # offensive strategy
            h_score = len(own_liberties)  - (2 * len(opp_liberties))           
            
        return h_score
