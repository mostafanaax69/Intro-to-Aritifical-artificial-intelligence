"""Search Algos: MiniMax, AlphaBeta
"""
from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT


# TODO: you can import more modules, if needed


class SearchAlgos:
    def __init__(self, utility, succ, perform_move, goal=None):
        """The constructor for all the search algos.
        You can code these functions as you like to,
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move
        self.goal = goal

    def search(self, state, depth, maximizing_player):
        pass


class MiniMax(SearchAlgos):

    def search(self, state, depth, maximizing_player):
        """Start the MiniMax algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        checkgoal = self.goal(state)
        if (depth == 0 or checkgoal):
            return (self.utility(state, checkgoal), state[0], depth)

        children = self.succ(state, maximizing_player)

        if (maximizing_player):
            curMax = (float('-inf'), None, depth)
            for child in children:
                val = child[2][child[1][0]][child[1][1]]
                if val > 2 and child[4] is not None:
                    child[4].remove(((child[1][0], child[1][1]), val))
                child[2][child[1][0]][child[1][1]] = 1
                child[2][state[1][0]][state[1][1]] = -1
                v = self.search(child, depth - 1, False)
                if val > 2 and child[4] is not None:
                    child[4].append(((child[1][0], child[1][1]), val))
                child[2][child[1][0]][child[1][1]] = val
                child[2][state[1][0]][state[1][1]] = 1

                if v[0] > curMax[0] or (v[0] == curMax[0] and curMax[2] > v[2]):
                    assert child[0] is not None
                    curMax = (v[0], child[0], depth)

            return curMax
        else:
            curMin = (float('inf'), None, depth)
            for child in children:
                val = child[2][child[3][0]][child[3][1]]
                if val > 2:
                    child[4].remove(((child[3][0], child[3][1]), val))
                child[2][child[3][0]][child[3][1]] = 2
                child[2][state[3][0]][state[3][1]] = -1
                v = self.search(child, depth - 1, True)
                if val > 2:
                    child[4].append(((child[3][0], child[3][1]), val))
                child[2][child[3][0]][child[3][1]] = val
                child[2][state[3][0]][state[3][1]] = 2
                if v[0] < curMin[0]:
                    assert child[0] is not None
                    curMin = (v[0], child[0], depth)
            return curMin


class AlphaBeta(SearchAlgos):

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        checkgoal = self.goal(state)
        if (depth == 0 or checkgoal):
            return (self.utility(state, checkgoal), state[0], depth)

        children = self.succ(state, maximizing_player)

        if maximizing_player:
            curMax = (float('-inf'), None, depth)
            for child in children:
                val = child[2][child[1][0]][child[1][1]]
                if val > 2 and child[4] is not None:
                    child[4].remove(((child[1][0], child[1][1]), val))
                child[2][child[1][0]][child[1][1]] = 1
                child[2][state[1][0]][state[1][1]] = -1
                v = self.search(child, depth - 1, False, alpha, beta)
                if val > 2 and child[4] is not None:
                    child[4].append(((child[1][0], child[1][1]), val))
                child[2][child[1][0]][child[1][1]] = val
                child[2][state[1][0]][state[1][1]] = 1

                if v[0] > curMax[0] :
                    assert child[0] is not None
                    curMax = (v[0], child[0], depth)
                alpha = max(curMax[0], alpha)
                if curMax[0] >= beta:
                    return float('inf'), child[0], depth

            return curMax
        else:
            curMin = (float('inf'), None, depth)
            for child in children:
                val = child[2][child[3][0]][child[3][1]]
                if val > 2:
                    child[4].remove(((child[3][0], child[3][1]), val))
                child[2][child[3][0]][child[3][1]] = 2
                child[2][state[3][0]][state[3][1]] = -1
                v = self.search(child, depth - 1, True, alpha, beta)
                if val > 2:
                    child[4].append(((child[3][0], child[3][1]), val))
                child[2][child[3][0]][child[3][1]] = val
                child[2][state[3][0]][state[3][1]] = 2
                if v[0] < curMin[0]  :
                    assert child[0] is not None
                    curMin = (v[0], child[0], depth)
                beta = min(curMin[0], beta)
                if curMin[0] <= alpha:
                    return float('-inf'), child[0], depth
            return curMin
