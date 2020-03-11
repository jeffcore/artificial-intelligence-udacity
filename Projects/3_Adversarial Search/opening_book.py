import random
from collections import defaultdict, Counter
from isolation import Isolation
import pickle

def main():
    state = Isolation()
    opening_book = OpeningBook(state, 5000000, 4)
    book = opening_book.build_table()
    # print(book)
    opening_book.save_opening_book(book)

class OpeningBook:

    def __init__(self, state, num_rounds, depth):
        self.state = state
        self.num_rounds = num_rounds
        self.depth = depth

    def build_table(self):
        # Builds a table that maps from game state -> action
        # by choosing the action that accumulates the most
        # wins for the active player. (Note that this uses
        # raw win counts, which are a poor statistic to
        # estimate the value of an action; better statistics
        # exist.)
        from collections import defaultdict, Counter
        book = defaultdict(Counter)
        for n in range(self.num_rounds):
            if n%10000 == 0:
                print(n)
            state =  Isolation()     
            self.build_tree(state, book, self.depth)
        return {k: max(v, key=v.get) for k, v in book.items()}


    def build_tree(self, state, book, depth):
        if depth <= 0 or state.terminal_test():
            return -self.simulate(state)
        # print(state.actions() , " actions")
        action = random.choice(state.actions())
        
        reward = self.build_tree(state.result(action), book, depth - 1)
        book[state][action] += reward
        return -reward


    def simulate(self, state):
        player_id = state.player()
        while not state.terminal_test():
            state = state.result(random.choice(state.actions()))
            # result = self.alpha_beta_search(state, 4)
            # if result: 
            #     state = state.result(result)
            # else:
            #     state = state.result(random.choice(state.actions()))
        return -1 if state.utility(player_id) < 0 else 1  

    def save_opening_book(self, book):
        with open('data.pickle', 'wb') as f:
            pickle.dump(book, f)
   
    
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
        for a in gameState.actions():                        
            v = self.min_value(gameState.result(a), alpha, beta,  depth - 1)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move

    def min_value(self, gameState, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if gameState.terminal_test():
            return gameState.utility(gameState.player())
        
        if depth <= 0:            
            return self.score(gameState)
            
        v = float("inf")
        for a in gameState.actions():            
            # TODO: modify the call to max_value()
            v = min(v, self.max_value(gameState.result(a), alpha, beta, depth - 1))
            
            # TODO: update the value bound
            if v <= alpha:
                return v
            beta = min(beta, v)           
        return v

    
    def max_value(self, gameState, alpha, beta, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if gameState.terminal_test():
            return gameState.utility(gameState.player())
        
        if depth <= 0:
            return self.score(gameState)

        v = float("-inf")
        for a in gameState.actions():
            # TODO: modify the call to min_value()
            v = max(v, self.min_value(gameState.result(a), alpha, beta, depth - 1))
            # TODO: update the value bound
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def score(self, state):
        """Calculate the heuristic value of a game state from the point of view
            of the given player. Similiar to custom_score_3:

            1. only avoids center if player 1 has more moves than player 2
        """
        AVG_PLY_GAME_LENGTH = 70
        own_loc = state.locs[state.player()]       
        opp_loc = state.locs[1 - state.player()]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
       
        ratio = state.ply_count / AVG_PLY_GAME_LENGTH
        if ratio >= .5:
            # offensive strategy
            h_score = (len(own_liberties) * 2) - len(opp_liberties)
        else:
            # defensive strategy
            h_score = len(own_liberties)  - (2 * len(opp_liberties))
            
        return h_score


if __name__ == "__main__":
    main()