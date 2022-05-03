"""
MiniMax Player
"""
from players.AbstractPlayer import AbstractPlayer
import copy
from collections import deque
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
        depth = 1
        max_neighbours = 4
        start_time = time.time()
        minmax_alg = SearchAlgos.MiniMax(utility=self.utility, succ=self.succ, perform_move=None, goal=self.goal)
        fruit_list = copy.deepcopy(self.fruits)
        (val, move,_d) = minmax_alg.search((None, self.loc, self.board, self.rival_loc, fruit_list), depth, True)
        last_iteration = time.time() - start_time
        next_iteration_max_time = max_neighbours * last_iteration
        passedTime = time.time() - start_time
        while passedTime + next_iteration_max_time < time_limit:
            depth += 1
            iteration_start_time = time.time()
            (val, move,_d) = minmax_alg.search((move, self.loc, self.board, self.rival_loc, fruit_list), depth, True)
            last_iteration = time.time() - iteration_start_time
            next_iteration_max_time = max_neighbours * last_iteration
            passedTime = time.time() - start_time
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
        agent_loc = state[1]
        rival_loc = state[3]
        rivalBoard = copy.deepcopy(state[2])
        agentBoard = copy.deepcopy(state[2])
        fruitBoard = copy.deepcopy(state[2])
        RivalAgentBoard = copy.deepcopy(state[2])
        flag_fruitIsReachable = 0
        minDist_agent_rival = 0
        free_rivalAttainable_spots = self.freeRivalaux(rivalBoard, rival_loc)  # freeRivalNeighbours in dry
        free_agentAttainable_spotsSum = self.freeNeighborsaux(agentBoard, agent_loc)  # sumofavillable in dry
        cur_graph = self.buildgraph(RivalAgentBoard)

        if (rival_loc[0], rival_loc[1]) in cur_graph and (agent_loc[0], agent_loc[1]) in cur_graph:
            if nx.has_path(cur_graph, agent_loc, rival_loc):
                minDist_agent_rival = len(nx.shortest_path(cur_graph, agent_loc, rival_loc))
        fruit_val = 0
        mindist_fruit_agent = float('inf')
        for fruit in state[4]:
            cur_graph_fruit = self.buildgraph(fruitBoard)
            if (fruit[0][0], fruit[0][1]) not in cur_graph_fruit or (agent_loc[0], agent_loc[1]) not in cur_graph_fruit:
                continue
            if nx.has_path(cur_graph_fruit, agent_loc, fruit[0]):
                bfsRes = len(nx.shortest_path(cur_graph_fruit, agent_loc, fruit[0]))
                if bfsRes < mindist_fruit_agent:
                    flag_fruitIsReachable = 1
                    mindist_fruit_agent = bfsRes
                    fruit_val = fruit[1]


        if minDist_agent_rival == 0 and flag_fruitIsReachable:
            res = 0.80 * free_agentAttainable_spotsSum + (0.20 * (1/mindist_fruit_agent) * fruit_val)
            return res

        if minDist_agent_rival == 0 and flag_fruitIsReachable == 0:
            return free_agentAttainable_spotsSum

        if minDist_agent_rival > 0:
            if free_rivalAttainable_spots > 0 and flag_fruitIsReachable:
                res = (free_agentAttainable_spotsSum / max(minDist_agent_rival, free_rivalAttainable_spots)) + 1/mindist_fruit_agent
                return res

            if free_rivalAttainable_spots == 0 and flag_fruitIsReachable:
                return 1/mindist_fruit_agent

        return free_agentAttainable_spotsSum / max(minDist_agent_rival,free_rivalAttainable_spots)

    def freeRivalaux(self, board, location):
        freeRivalNeighbours = 0
        for d in self.directions:
            rival_i = location[0] + d[0]
            rival_j = location[1] + d[1]
            if 0 <= rival_i < len(self.board) and 0 <= rival_j < len(self.board[0]) and (
                    board[rival_i][rival_j] == 0 or board[rival_i][rival_j] > 2):
                new_location = (rival_i, rival_j)
                board[rival_i][rival_j] = -1
                freeRivalNeighbours = 1 + self.freeRivalaux(board, new_location)
        return freeRivalNeighbours

    def freeNeighborsaux(self, board, location):
        freeNeighbours = 0
        leg = 0
        for d in self.directions:
            i = location[0] + d[0]
            j = location[1] + d[1]
            if 0 <= i < len(self.board) and 0 <= j < len(self.board[0]) and (
                    board[i][j] == 0 or board[i][j] > 2):
                new_location = (i, j)
                board[i][j] = -1
                freeNeighbours = 1 + self.freeRivalaux(board, new_location)
                leg+=1

        return freeNeighbours

    ########## helper functions for MiniMax algorithm ##########
    # TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm

    def utility(self, state, checkgoal):
        if checkgoal:
            agentMove, RivalMove = self.checkLegal(state)
            if agentMove is False and RivalMove:
                return -1000
            if agentMove and RivalMove is False:
                return float('inf')
            if agentMove is False and RivalMove is False:
                return 0
        else:
            return self.builtIn_heuristic(state)

    def succ(self, state, playerMinorMax):
        children = []
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


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class queueNode:
    def __init__(self, pt: Point, dist: int):
        self.pt = pt
        self.dist = dist


def isValid(row, col, mat):
    return (row >= 0) and (row < len(mat)) and (col >= 0) and (col < len(mat[0]))


def BFS(mat, src: Point, dest: Point, val):
    if mat[src.x][src.y] != 1 or mat[dest.x][dest.y] != val:
        return -1
    rowNum = [-1, 0, 0, 1]
    colNum = [0, -1, 1, 0]
    visited = [[False for i in range(len(mat))] for j in range(len(mat[0]))]
    visited[src.x][src.y] = True
    q = deque()
    s = queueNode(src, 0)
    q.append(s)
    while q:
        curr = q.popleft()  # Dequeue the front cell
        pt = curr.pt

        if pt.x == dest.x and pt.y == dest.y:
            return curr.dist

        for i in range(4):
            row = pt.x + rowNum[i]
            col = pt.y + colNum[i]

            if (isValid(row, col, mat) and
                    (mat[row][col] == 0 or mat[row][col] == 2) and not visited[row][col]):
                visited[row][col] = True
                Adjcell = queueNode(Point(row, col), curr.dist + 1)
                q.append(Adjcell)

    return -1
