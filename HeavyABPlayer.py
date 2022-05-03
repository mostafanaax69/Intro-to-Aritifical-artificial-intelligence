"""
MiniMax Player with AlphaBeta pruning
"""
from players.AbstractPlayer import AbstractPlayer
import copy
import time
import SearchAlgos
import networkx as nx


# TODO: you can import more modules, if needed


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time,
                                penalty_score)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        # TODO: initialize more fields, if needed, and the Minimax algorithm from SearchAlgos.py
        self.loc = None
        self.board = None
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.rival_loc = None
        self.fruits = []
        self.rival_score = 0
        self.score = 0

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        self.board = board
        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == 1:
                    self.loc = (i, j)
                if val == 2:
                    self.rival_loc = (i, j)
                if val > 2:
                    self.fruits.append(((i, j), val))

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        depth = 3
        alpha_beta_alg = SearchAlgos.AlphaBeta(utility=self.utility, succ=self.succ, perform_move=None, goal=self.goal)
        fruit_list = copy.deepcopy(self.fruits)
        (val, move,_d) = alpha_beta_alg.search((None, self.loc, self.board, self.rival_loc, fruit_list), depth, True)
        if move is None:
            move = self.getBestMove((move, self.loc, self.board, self.rival_loc, fruit_list))
        self.perform_move(move, players_score)
        return move

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        self.board[self.rival_loc] = -1
        self.board[pos] = 2
        self.rival_loc = pos

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        for i, row in enumerate(self.board):
            for j, val in enumerate(row):
                if val > 2:
                    self.board[i][j] = 0

        self.fruits.clear()
        for pos, val in fruits_on_board_dict.items():
            self.board[pos] = val
            self.fruits.append(((pos[0], pos[1]), val))

    ########## helper functions in class ##########
    # TODO: add here helper functions in class, if needed

    def buildgraph(self, currboard):
        g = nx.Graph()
        for i in range(len(currboard)):
            for j in range(len(currboard[i])):
                if currboard[i][j] == 0 or currboard[i][j] >= 3:
                    if i < (len(currboard) - 1) and currboard[i + 1][j] != -1:
                        g.add_edge((i, j), (i + 1, j))
                    if j < (len(currboard[i]) - 1) and currboard[i][j + 1] != -1:
                        g.add_edge((i, j), (i, j + 1))
                if currboard[i][j] == 1 or currboard[i][j] == 2:
                    if i < len(currboard) - 1 and (currboard[i + 1][j] == 0 or currboard[i + 1][j] >= 3):
                        g.add_edge((i, j), (i + 1, j))
                    if j < (len(currboard[i]) - 1) and (currboard[i][j + 1] == 0 or currboard[i][j + 1] >= 3):
                        g.add_edge((i, j), (i, j + 1))

        return g.to_undirected()

    def builtIn_heuristic(self, state):
        fruit_val = 0
        fruitBoard = copy.deepcopy(state[2])
        agent_loc=state[1]
        mindist_fruit_agent = -1
        is_reachable = 0
        for fruit in state[4]:
            cur_graph_fruit = self.buildgraph(fruitBoard)
            if (fruit[0][0], fruit[0][1]) not in cur_graph_fruit or (agent_loc[0], agent_loc[1]) not in cur_graph_fruit:
                continue
            if nx.has_path(cur_graph_fruit, agent_loc, fruit[0]):
                bfsRes = len(nx.shortest_path(cur_graph_fruit, agent_loc, fruit[0]))
                if bfsRes < mindist_fruit_agent:
                    mindist_fruit_agent = bfsRes
                    fruit_val = fruit[1]
                    is_reachable = 1
        score = self.longest_path(state[1],5) / (self.longest_path(state[3], 5) + 1)

        if is_reachable == 1:
            for fruit in self.fruits:
                if fruit[0][0] == agent_loc[0] and fruit[0][1] == agent_loc[1]:
                    return score / (len(self.board) * len(self.board[0]))
            return (score / (len(self.board) * len(self.board[0]))) - mindist_fruit_agent

        else:
            return score / (len(self.board) * len(self.board[0]))



    ########## helper functions for MiniMax algorithm ##########
    # TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm

    def utility(self, state, checkgoal):
        if checkgoal:
            agentMove, RivalMove = self.checkLegal(state)
            if agentMove is False and RivalMove:
                return -1
            if agentMove and RivalMove is False:
                return float('inf')
            if agentMove is False and RivalMove is False:
                return 0
        else:
            return self.builtIn_heuristic(state)

    def succ(self, state, playerMinorMax):
        children=[]
        if playerMinorMax:
            for d in self.directions:
                i = state[1][0] + d[0]
                j = state[1][1] + d[1]
                if len(self.board) > i >= 0 and len(self.board[0]) > j >= 0 and (
                        state[2][i][j] > 2 or state[2][i][j] == 0):
                    children.append((d, (i, j), state[2], state[3], state[4]))
        else:
            for d in self.directions:
                i = state[3][0] + d[0]
                j = state[3][1] + d[1]
                if len(self.board) > i >= 0 and len(self.board[0]) > j >= 0 and (
                        state[2][i][j] > 2 or state[2][i][j] == 0):
                    children.append((d, state[1], state[2], (i, j), state[4]))
        return children

    def goal(self, state):
        agentMove, RivalMove = self.checkLegal(state)
        if agentMove is False or RivalMove is False:
            return True
        return False

    def checkLegal(self, state):

        agent_loc = state[1]
        agent_move = False
        rival_loc = state[3]
        rival_move = False
        for d in self.directions:
            agent_i = agent_loc[0] + d[0]
            agent_j = agent_loc[1] + d[1]
            rival_i = rival_loc[0] + d[0]
            rival_j = rival_loc[1] + d[1]
            if 0 <= agent_i < len(self.board) and 0 <= agent_j < len(self.board[0]) and (
                    self.board[agent_i][agent_j] == 0 or self.board[agent_i][agent_j] > 2):
                agent_move = True
            if 0 <= rival_i < len(self.board) and 0 <= rival_j < len(self.board[0]) and (
                    self.board[rival_i][rival_j] == 0 or self.board[rival_i][rival_j] > 2):
                rival_move = True

        return agent_move, rival_move

    def perform_move(self, move, players_score):
        prev_loc = self.loc
        self.board[prev_loc] = -1
        self.loc = (self.loc[0] + move[0], self.loc[1] + move[1])
        if self.board[
            self.loc[0], self.loc[1]] > 2:  # check if it is between min and max and how to get min and max fruit
            players_score[0] += self.board[self.loc[0], self.loc[1]]
        self.board[self.loc] = 1
        self.score = players_score[0]
        self.rival_score = players_score[1]

    def getBestMove(self, state):
        best_move = (0, 0)
        for d in self.directions:
            agent_i = state[1][0] + d[0]
            agent_j = state[1][1] + d[1]
            if 0 <= agent_i < len(self.board) and 0 <= agent_j < len(self.board[0]) and (
                    self.board[agent_i][agent_j] == 0 or self.board[agent_i][agent_j] > 2):
                if self.board[agent_i][agent_j] > 2:
                    return (d[0], d[1])
                else:
                    best_move = (d[0], d[1])
        return best_move



    def getEmpty(self,agent_loc):
        children = [(agent_loc[0] + d[0], agent_loc[1] + d[1]) for d in self.directions
                    if (len(self.board) > (agent_loc[0] + d[0]) >= 0)
                    and (0 <= (agent_loc[1] + d[1]) < len(self.board[0]))
                    and (self.board[agent_loc[0] + d[0]][agent_loc[1] + d[1]] == 0 or self.board[agent_loc[0] + d[0]][
                agent_loc[1] + d[1]] > 2)]

        return children


    def state_score(self, board, pos):
        num_steps_available = 0
        for d in self.directions:
            i = pos[0] + d[0]
            j = pos[1] + d[1]

            if 0 <= i < len(board) and 0 <= j < len(board[0]) and (board[i][j] not in [-1, 1, 2]):
                num_steps_available += 1

        if num_steps_available == 0:
            return -1
        else:
            return 4 - num_steps_available

    def longest_path(self, agent_loc, depth):
        children = []
        longest = 0
        children = self.getEmpty(agent_loc)
        if len(children) == 0 or depth == 0:
            return 0


        for (i, j) in children:
            new_loc = (i, j)
            self.board[i][j] = -1
            empty_space = self.longest_path(new_loc, depth - 1) + 1
            self.board[i][j] = 0
            longest = max(empty_space, longest)

        return longest

