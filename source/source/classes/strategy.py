import copy
import math
import random
from math import log, sqrt, inf
from random import randrange
import numpy as np
from rich.table import Table
from rich.progress import track
from rich.console import Console
from rich.progress import Progress
import time
import classes.logic as logic
import classes.graphes as graphes

# When implementing a new strategy add it to the `str2strat`
# dictionary at the end of the file


class PlayerStrat:
    def __init__(self, _board_state, player):
        self.root_state = _board_state
        self.player = player
        
        if (self.player == logic.BLACK_PLAYER):
            self.otherPlayer = logic.WHITE_PLAYER
        else:
            self.otherPlayer = logic.BLACK_PLAYER

    def start(self):
        """
        This function selects a tile from the board.

        @returns (x, y): A tuple of integers corresponding to a valid
                and free tile on the board.
        """
        raise NotImplementedError


class Node:
    """
    This class implements the main object that you will manipulate: nodes.
    Nodes include the state of the game (i.e., the 2D board), children (i.e., other children nodes), a list of
    untried moves, etc...
    """
    def __init__(self, board, move=(None, None), wins=0, visits=0, children=None, player=2):
        # Save the #wins:#visited ratio
        self.state = board
        self.move = move
        self.wins = wins
        self.visits = visits
        self.children = children or []
        self.parent = None
        self.untried_moves = logic.get_possible_moves(board)

        self.player = player

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def add_children(self):
        possible_moves = logic.get_possible_moves(self.state)

        child_player = logic.BLACK_PLAYER if self.player == logic.WHITE_PLAYER else logic.WHITE_PLAYER

        for element in possible_moves:
            copy_state = copy.deepcopy(self.state)
            copy_state[element[0]][element[1]] = self.player

            child = Node(copy_state, element, player=child_player)
            self.add_child(child)


class Random(PlayerStrat):
    def __init__(self, _board_state, player):
        super().__init__(_board_state, player)

    def start(self):
        """
        This function selects a random valid and free tile on the board.

        @returns (x, y): A tuple of integers corresponding to a valid
                and free tile on the board.
        """
        possible_moves = logic.get_possible_moves(self.root_state)
        return random.choice(possible_moves)

class MiniMax(PlayerStrat):
    def start(self):
        node = Node(self.root_state, player=self.player)

        if len(node.untried_moves) > 14:
            possible_moves = logic.get_possible_moves(self.root_state)
            move = random.choice(possible_moves)
            return move

        start_time = time.time()
        _, move = self.minimax(node, float('-inf'), float('inf'))
        self.display_time(start_time)

        return move

    def minimax(self, node, alpha, beta):
        _, move = self.max_value(node, alpha, beta)
        return move

    def max_value(self, node, alpha, beta):
        if node.is_terminal():
            return node.utility(), node.move

        v = float('-inf')
        best_move = None

        for child in node.generate_children():
            child_value, _ = self.min_value(child, alpha, beta)
            if child_value > v:
                v = child_value
                best_move = child.move

            alpha = max(alpha, v)
            if v >= beta:
                break

        return v, best_move

    def min_value(self, node, alpha, beta):
        if node.is_terminal():
            return node.utility(), node.move

        v = float('inf')
        best_move = None

        for child in node.generate_children():
            child_value, _ = self.max_value(child, alpha, beta)
            if child_value < v:
                v = child_value
                best_move = child.move

            beta = min(beta, v)
            if v <= alpha:
                break

        return v, best_move

    def display_time(self, start_time):
        time_elapsed = time.time() - start_time
        hours, rem = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        time_string = ""
        if hours > 0:
            time_string += "{:0>2}h".format(int(hours)) + " "
        if minutes > 0:
            time_string += "{:0>2}m".format(int(minutes)) + " "
        if seconds > 0:
            time_string += "{:05.2f}s".format(seconds)

        print(time_string)

        
class Evaluate(PlayerStrat):
    def start(self):
        node = Node(self.root_state, player=self.player)
        
        matrix_length = len(self.root_state)
        middle = matrix_length // 2

        if matrix_length % 2 != 0 and self.root_state[middle][middle] == 0:
            return middle, middle

        depth = 0
        if len(node.untried_moves) > 10:
            self.maxDepth = 4
        else:
            self.maxDepth = 7
            
        start_time = time.time()
        node.move = self.minimax(node, depth)
        self.display_time(start_time)
        
        return node.move
        
    def minimax(self, node, depth):  
        value, move = self.max_value(node, -inf, inf, depth)
        return move
    
    def max_value(self, node, alpha, beta, depth):
        winning, value, move = self.winning(node)
        if winning:
            return value, move
            
        if depth >= self.maxDepth:
            for row in range(len(self.root_state)):
                for col in range(len(self.root_state)):
                    if self.root_state[row][col] == 0:
                        win, value, move = self.win(Node(self.root_state, player=self.player), row, col)
                        if win:
                            return value, move
            return self.evaluate()
        
        v = -inf
        a1 = (-1, -1)
        
        node.add_children()
        
        for child in node.children:
            v2, a2 = self.min_value(child, alpha, beta, depth+1)
            if v2 >= v:
                v, a1 = v2, a2  
                alpha = max(alpha, v)
            if v >= beta:
                return v, a1
        return v, a1
        
    def min_value(self, node, alpha, beta, depth):
        winning, value, move = self.winning(node)
        if winning:
            return value, move
            
        if depth >= self.maxDepth:
            return self.evaluate()
        
        v = +inf
        a1 = (-1, -1)
                
        node.add_children()
            
        for child in node.children:
            v2, a2 = self.max_value(child, alpha, beta, depth+1)
            if v2 < v:
                v, a1 = v2, a2
                beta = min(beta, v)
                if v <= alpha:
                    return v, a1

        return v, a1
    
    def winning(self, node):            
        if logic.is_game_over(self.player, node.state) == self.player:
            return True, 200, node.move        
        elif logic.is_game_over(self.otherPlayer, node.state) == self.otherPlayer:
            return True, 200, node.move   
        else:
            return False, 0, node.move   
             
    def win(self, node, x, y):
        original_state_xy = node.state[x][y]
        
        node.state[x][y] = self.player
        if logic.is_game_over(self.player, node.state) == self.player and original_state_xy != self.otherPlayer:
            node.state[x][y] = original_state_xy
            return True, 200, (x, y)  
        
        node.state[x][y] = self.otherPlayer      
        if logic.is_game_over(self.otherPlayer, node.state) == self.otherPlayer and original_state_xy != self.player:
            node.state[x][y] = original_state_xy
            return True, 200, (x, y)   
        
        node.state[x][y] = original_state_xy
        return False, 0, node.move
        
    
    def evaluate(self):      
        matrix_length = len(self.root_state)
        middle = matrix_length // 2            

        value_matrix = [[0] * matrix_length for _ in range(matrix_length)]
        best_value = -inf
        best_move = (-1, -1)
        
        if self.player == logic.BLACK_PLAYER: 
            for row in range(matrix_length):
                other_on_same_row = 0
                for col in  range(matrix_length):
                    win, value, move = self.win(Node(self.root_state, player=self.player), row, col)
                    if win:
                        return value, move
                    
                    if self.root_state[row][col] == 0:
                        value_matrix[row][col] += (matrix_length - abs(middle - row) - abs(middle - col)) / 2
                        value_matrix[row][col] += (matrix_length - abs(middle - row))
                        distance_diagonal = abs((row + col) - (matrix_length - 1))
                        value_matrix[row][col] += (matrix_length - distance_diagonal) / 4
                            
                    elif self.root_state[row][col] == self.otherPlayer:
                        other_on_same_row += 1
                        
                for col in range(matrix_length):
                    if other_on_same_row > 0:
                        value_matrix[row][col] = max(0, value_matrix[row][col] - other_on_same_row)
                    if value_matrix[row][col] > best_value:
                        best_value = value_matrix[row][col]
                        best_move = (row, col)
                       
        else:
            for col in range(matrix_length):
                other_on_same_col = 0
                for row in  range(matrix_length):
                    win, value, move = self.win(Node(self.root_state, player=self.player), row, col)
                    if win:
                        return value, move
                    
                    if self.root_state[row][col] == 0:
                        value_matrix[row][col] += (matrix_length - abs(middle - row) - abs(middle - col)) / 2
                        value_matrix[row][col] += (matrix_length - abs(middle - col))
                        distance_diagonal = abs((row + col) - (matrix_length - 1))
                        value_matrix[row][col] += (matrix_length - distance_diagonal) / 4       
                            
                    elif self.root_state[row][col] == self.otherPlayer:
                        other_on_same_col += 1
                        
                for row in range(matrix_length):
                    if other_on_same_col > 0:
                        value_matrix[row][col] = max(0, value_matrix[row][col] - other_on_same_col)
                    if value_matrix[row][col] > best_value:
                        best_value = value_matrix[row][col]
                        best_move = (row, col)                
             
        return best_value, best_move   
    
    def display_time(self, start_time):
        time_elapsed = time.time() - start_time
        hours, rem = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        
        time_string = ""
        if hours > 0:
            time_string += "{:0>2}h".format(int(hours)) + " "
        if minutes > 0:
            time_string += "{:0>2}m".format(int(minutes)) + " "
        if seconds > 0:
            time_string += "{:05.2f}s".format(seconds)

        print(time_string)
import random

class ShortPath(PlayerStrat):    
    def start(self):
        node = Node(self.root_state, player=self.player)
        
        depth = 0
        if len(node.untried_moves) > 10:
            self.maxDepth = 4
        else:
            self.maxDepth = 7
            
        startTime = time.time()
        node.move = self.minimax(node, depth)
        self.displayTime(startTime)
        
        return node.move
        
    def minimax(self, node, depth):  
        value, move = self.max_value(node, -inf, inf, depth)
        return move
    
    def max_value(self, node, alpha, beta, depth):
        winning, value, move = self.Winning(node)
        if winning:
            return value, move
            
        if depth == self.maxDepth:
            return self.evaluate(node, self.player, self.otherPlayer)
        
        v = -inf
        a1 = (-1, -1)
        
        node.add_children()
        
        for child in node.children:
            v2, a2 = self.min_value(child, alpha, beta, depth+1)
            if v2 > v:
                v, a1 = v2, a2  
                alpha = max(alpha, v)
            if v >= beta:
                return v, a1
        return v, a1
        
    def min_value(self, node, alpha, beta, depth):
        winning, value, move = self.Winning(node)
        if winning:
            return value, move
            
        if depth == self.maxDepth:
            evaluation, move = self.evaluate(node, self.otherPlayer, self.player)
            if evaluation < 0:
                return evaluation, move
            else:
                return evaluation/2, move
        
        v = +inf
        a1 = (-1, -1)
                
        node.add_children()
            
        for child in node.children:
            v2, a2 = self.max_value(child, alpha, beta, depth+1)
            if v2 < v:
                v, a1 = v2, a2
                beta = min(beta, v)
                if v <= alpha:
                    return v, a1

        return v, a1
    
    def Winning(self, node):            
        if logic.is_game_over(self.player, node.state) == self.player:
            return True, 200, node.move        
        elif logic.is_game_over(self.otherPlayer, node.state) == self.otherPlayer:
            return True, 200, node.move   
        else:
            return False, 0, node.move
             
    def evaluate(self, node, player, otherPlayer):
        costMin = +inf
        shortestPath = []
        
        g = graphes.Graph(self.root_state, player, otherPlayer)

        if player == logic.BLACK_PLAYER:
            leftEdge = [(x, y) for x, y in logic.get_possible_moves(self.root_state) if y == 0]
            rightEdge = [(x, y) for x, y in logic.get_possible_moves(self.root_state) if y == len(self.root_state)-1]
            
            ownOnSameRows = [sum(1 for col in range(len(self.root_state)) if self.root_state[row][col] == player) for row in range(len(self.root_state))]
            sorted_indices = sorted(range(len(ownOnSameRows)), key=lambda i: ownOnSameRows[i], reverse=True)[:4]
            
            for start in sorted_indices:
                if (start, 0) not in leftEdge:
                    continue
                for end in sorted_indices:
                    if (end, len(self.root_state)-1) not in rightEdge:
                        continue
                
                    start_node = graphes.Node(start, 0, 0)
                    end_node = graphes.Node(end, len(self.root_state)-1, 0)
                    path = graphes.astar(g, start_node, end_node)
                    if path is None:
                        continue
                    cost = graphes.path_cost(path, g)
                    if cost < costMin:
                        costMin = cost 
                        shortestPath = path
            
            if not shortestPath:
                start = random.choice(leftEdge)
                end = random.choice(rightEdge)
                start_node = graphes.Node(start[0], 0, 0)
                end_node = graphes.Node(end[0], len(self.root_state)-1, 0)
                path = graphes.astar(g, start_node, end_node)
                if path is not None:
                    cost = graphes.path_cost(path, g)
                    if cost < costMin:
                        costMin = cost 
                        shortestPath = path
                    
        else:
            upEdge = [(x, y) for x, y in logic.get_possible_moves(self.root_state) if x == 0]
            bottomEdge = [(x, y) for x, y in logic.get_possible_moves(self.root_state) if x == len(self.root_state)-1]
            
            ownOnSameCols = [sum(1 for row in range(len(self.root_state)) if self.root_state[row][col] == player) for col in range(len(self.root_state))]
            sorted_indices = sorted(range(len(ownOnSameCols)), key=lambda i: ownOnSameCols[i], reverse=True)[:4]
            
            for start in sorted_indices:
                if (0, start) not in upEdge:
                    continue
                for end in sorted_indices:
                    if (len(self.root_state)-1, end) not in bottomEdge:
                        continue
                
                    start_node = graphes.Node(0, start, 0)
                    end_node = graphes.Node(len(self.root_state)-1, end, 0)
                    path = graphes.astar(g, start_node, end_node)
                    if path is None:
                        continue
                    cost = graphes.path_cost(path, g)
                    if cost < costMin:
                        costMin = cost 
                        shortestPath = path
                        
            if not shortestPath:
                start = random.choice(upEdge)
                end = random.choice(bottomEdge)
                start_node = graphes.Node(0, start[1], 0)
                end_node = graphes.Node(len(self.root_state)-1, end[1], 0)
                path = graphes.astar(g, start_node, end_node)
                if path is not None:
                    cost = graphes.path_cost(path, g)
                    if cost < costMin:
                        costMin = cost 
                        shortestPath = path

        move = (-1, -1)
        if not shortestPath:
            return -10, move  
        elif len(shortestPath) == 1:
            move = shortestPath[0] if logic.is_node_free(shortestPath[0], self.root_state) else move
        else:              
            move = shortestPath[len(shortestPath)//2]
            
        busyNode = not logic.is_node_free(move, self.root_state)

        while busyNode:
            shortestPath.pop(len(shortestPath)//2)
            if len(shortestPath) == 1:
                move = shortestPath[0]
            else:
                move = shortestPath[len(shortestPath)//2]
            busyNode = not logic.is_node_free(move, self.root_state)
        
        if not logic.is_node_free(move, self.root_state): 
            return -10, move
        
        return abs(50 - costMin), move
    
    def displayTime(self, startTime):
        timeElapsed = time.time() - startTime
        hours, rem = divmod(timeElapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        
        timeString = ""
        if hours > 0:
            timeString += "{:0>2}h".format(int(hours)) + " "
        if minutes > 0:
            timeString += "{:0>2}m".format(int(minutes)) + " "
        if seconds > 0:
            timeString += "{:05.2f}s".format(seconds)

        print(timeString)



class MonteCarlo(PlayerStrat):
    def start(self):
        node = Node(self.root_state, player=self.player)
        
        matrix_length = len(self.root_state)
        middle = matrix_length // 2

        if matrix_length % 2 != 0 and self.root_state[middle][middle] == 0:
            return middle, middle
        
        depth = 0
        if len(node.untried_moves) > 10:
            self.max_depth = 4
        else:
            self.max_depth = 7
            
        start_time = time.time()
        node.move = self.monte_carlo(node, depth)
        self.display_time(start_time)

        return node.move

    def monte_carlo(self, node, depth):
        simulations = 100

        legal_moves = node.untried_moves + [child.move for child in node.children]
        scores = {move: 0 for move in legal_moves}

        for _ in range(simulations):
            move = random.choice(legal_moves)
            child = Node(node.state, move, player=self.player)
            score = self.simulate(child)
            scores[move] += score

        best_move = max(scores, key=scores.get)
        return best_move

    def simulate(self, node):
        player = node.player
        while True:
            if logic.is_game_over(player, node.state) == player:
                return 1  
            elif logic.is_game_over(logic.other_player(player), node.state) == logic.other_player(player):
                return 0 

            legal_moves = node.untried_moves + [child.move for child in node.children]
            if not legal_moves:
                return 0.5  

            move = random.choice(legal_moves)
            node = Node(node.state, move, player=player)
            player = logic.other_player(player)

    def display_time(self, start_time):
        time_elapsed = time.time() - start_time
        hours, rem = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        time_string = ""
        if hours > 0:
            time_string += "{:0>2}h".format(int(hours)) + " "
        if minutes > 0:
            time_string += "{:0>2}m".format(int(minutes)) + " "
        if seconds > 0:
            time_string += "{:05.2f}s".format(seconds)

        print(time_string)

str2strat = {
    "human": None,
    "random": Random,
    "minimax": MiniMax,
    "evaluate": Evaluate,
    "short_path": ShortPath,
    "monte_carlo": MonteCarlo,

}

