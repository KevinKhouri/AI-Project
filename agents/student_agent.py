# Student agent: Add your own agent here
import numpy as np
from copy import deepcopy
from queue import Empty
from agents.agent import Agent
from store import register_agent
import sys


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # dummy return
        return my_pos, self.dir_map["u"]

#Represents a node in the SearchTree that MCTS will construct.
class Node:
    #Moves (Up, Right, Down, Left). Represents moving a position on the board.
    #Example, to move up one position, you must subtract 1 from row and 0 from the column,
    #hence (-1, 0)
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    #Represents opposite directions. Example: Opposite of Up (0), is Down (2)
    #Directions and their numbers are shown in constants.py. They are Up (0), Right (1),
    #Down (2), Left (3)
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}

    #my_pos = my position, adv_pos = advesaries position. Position is tuple (row, column).
    #my_turn is boolean representing who's turn it is (our agent or theirs).
    def __init__(self, chess_board, my_pos, adv_pos, my_turn, parentNode):
        #defines a state
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.my_turn = my_turn
        #defines MCTS statistics and tree parameters
        self.wins = 0
        self.totalPlays = 0
        self.parentNode = parentNode
        self.childNodes = None

    #This function generates all child nodes. Uses BFS up to depth max_step to find all legal states
    #we can transition to from this state, creates a node for each state, stores them in an array, 
    #and sets this node's childNodes array to equal the array created.
    def createChildrenNodes(self, max_step):
        #Supposedly arrays are faster than lists, but is there an arraylist like thing in python?
        childrenNodes = []

        start_pos = self.my_pos if self.my_turn else self.adv_pos

        #The second element in the tuples is the count of steps needed to get to that position. Performs BFS.
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}

        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            row, column = cur_pos
            #Find where the player could place a barrier in the current position, and create a node for 
            #each successful spot.
            #dir = direction. A value in chess_board at row,column,dir is true if that spot has a barrier.
            for dir in range(0,4):
                if not self.chess_board[row, column, dir]:
                    chess_board_copy = deepcopy(self.chess_board)
                    Node.set_barrier(chess_board_copy, row, column, dir)
                    if self.my_turn:
                        childrenNodes.append(Node(chess_board_copy, cur_pos, self.adv_pos, False, self))
                    else:
                        childrenNodes.append(Node(chess_board_copy, self.my_pos, cur_pos, True, self))
            
            #If the cur_step == max_step, the player can't move anywhere else from here, so don't add
            #anything to the queue and skip.
            #Else, find each nonvisted neighboring position the player could visit (not blocked by barrier
            #or by opposing agent), add its position and corresponding number of steps needed to get there 
            #to the queue.
            if cur_step != max_step:
                for dir in range(0,4):
                    new_pos = cur_pos + Node.moves[dir]
                    if (not self.chess_board[row, column, dir] and tuple(new_pos) not in visited 
                            and not np.array_equal(new_pos, self.my_pos) and not np.array_equal(new_pos, self.adv_pos)):
                        state_queue.append((new_pos, cur_step + 1))

        self.childNodes = childrenNodes
        return

    #I'm assuming python passes chess_board by reference...?
    @staticmethod
    def set_barrier(chess_board, r, c, dir):
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = Node.moves[dir]
        chess_board[r + move[0], c + move[1], Node.opposites[dir]] = True