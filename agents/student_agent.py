# Student agent: Add your own agent here
from array import array
from logging import root
import random
from re import I
import time
import numpy as np
from copy import deepcopy
from agents.agent import Agent
from store import register_agent
import math
import operator


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
        self.first_move = True
        self.autoplay = True
        self.searchTree = None #The search tree with the nodes.

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
        #If this is the first move, we call Monte_Carlo_Tree_Search with a time limit of 30 seconds.
        #Otherwise we call it with a time limit of 2 seconds. This would be the location where we could use
        #threads to enforce the time limit.
        if self.first_move:
            self.first_move = False
            #Since this is our first move, we must initialize the tree.
            self.searchTree = SearchTree(Node(chess_board, my_pos, adv_pos, True, None, None))
            bestMove = self.Monte_Carlo_Tree_Search(max_step, True)
            #Once we've received the result of the best move, we must change the SearchTree's root to the
            #node that we will transition to.
            self.searchTree.root = bestMove
            bestMove.parentNode = None #Here is where you'll use the garbage collector to see if it works.
            #Unpack and return our move.
            pos = bestMove.my_pos
            dir = bestMove.wall_direction
            return pos, dir
        else:
            #Since this is not our first move, we must update our root node to the state that the advesary put us in.
            foundChild = False
            if self.searchTree.root.childNodes != None:
                for childNode in self.searchTree.root.childNodes:
                    if my_pos == childNode.my_pos and adv_pos == childNode.adv_pos and np.array_equiv(chess_board, childNode.chess_board):
                        self.searchTree.root = childNode
                        self.searchTree.root.parentNode = None
                        foundChild = True
                        break
                #If we did not find a Node representing the state we trantitioned to, we must create a new node for this state.
            if (not foundChild):
                self.searchTree.root = Node(chess_board, my_pos, adv_pos, True, None, None) #Should check garbage collection here too.
            #Run MCTS:
            bestMove = self.Monte_Carlo_Tree_Search(max_step, False)
            pos = bestMove.my_pos
            dir = bestMove.wall_direction
            return pos, dir

    def Monte_Carlo_Tree_Search(self, max_step, first_move):
        start_time = time.time()
        totalTime = 29.99 if first_move else 1.99 #Number of seconds we can think.
        numOfSimulations = 0 #REMOVE THIS
        while (time.time() - start_time < totalTime):
            leaf = self.searchTree.select()
            child = self.searchTree.expand(leaf, max_step)
            result = self.searchTree.simulate(child, max_step)
            self.searchTree.backPropagate(result, child)
            numOfSimulations += 1 #REMOVE THIS
        #Time is up, so we must chose the child with the highest number of visits as the next move.
        print("Total time this turn: " + str((time.time() - start_time)) + " Simulations: " + str(numOfSimulations)) #REMOVE THIS
        return max(self.searchTree.root.childNodes, key=lambda x: StudentAgent.nodeValue(x)) #We assume we'll always have a child node to choose. To be safe we could put an if statement here.
        
    @staticmethod
    def nodeValue(child):
        value = child.UCB1()
        if value == math.inf: #When selecting the final action to take, we should ignore nodes that haven't been explored if we can.
            return -math.inf
        elif child.my_win == 1:
            return math.inf
        else:
            return value

#Represents the SerachTree that consists of nodes from class Node, that MCTS will construct and traverse.
#Each tree is initialized with a root node, then expanded/travsersed according to the UCT value of the children of the root.
#Methods that belong to SearchTree are select(), expand(), simulate() and backpropagate().
class SearchTree:
    def __init__(self, root):
        self.root = root

    #Returns the leaf node selected based on the UCB1 value.
    def select(self):
        return SearchTree.selectHelper(self.root)

   
   #Given the current node, selectHelper will return the child node with the highest UCB1 value
   #the Node returned by this method will be the node with which the expand() method will be called.
    @staticmethod
    def selectHelper(node):
        if (node.unusedMoveSet == None or not node.unusedMoveSet.isEmpty()): #This is equivalent to isFullyExpanded == False. There is no boolean isFullyExpanded, its implied by this statement.
            return node
        else:
            return SearchTree.selectHelper(max(node.childNodes, key=lambda x: x.UCB1()))

    #Expands the selected leaf node. If its a terminal node or hasn't been played yet, it should be
    #passed directly to simulation. Otherwise if its not a terminal node, and has been played before, this
    #leaf should have its children generated.
    @staticmethod
    def expand(leaf, max_step):
        terminal, result = leaf.isTerminal()
        if (terminal or leaf.totalPlays == 0):
            return leaf
        else:
            child = leaf.createChildNode(max_step)  #Generates a child node based on a random move within the max_step limits.
            return child

            
    #Simulates a full game starting from the state represented by 'node'.
    #Returns the result 1 for win, 0 for loss. If node is already a terminal state, simply returns
    #the result.
    @staticmethod
    def simulate(node, max_step):
        #If the node we want to simulate from is already representing a terminal state, 
        #we simply return the result immediately.
        gameOver, result = node.isTerminal()
        if (gameOver): 
            return result
        #Node was not a terminal state, so we must simulate a game from here:
        chess_board_copy = deepcopy(node.chess_board)
        starting_player_pos = deepcopy(node.my_pos) if node.my_turn else deepcopy(node.adv_pos)
        secondary_player_pos =  deepcopy(node.adv_pos) if node.my_turn else deepcopy(node.my_pos)
        strt_player_score = 0
        secd_player_score = 0
        #Until we determine the simulated game is over, keep performing random walks
        #for the start and secondary player until it's over.
        while (not gameOver):
            #Starting Player's Turn:
            starting_player_pos, dir = (SearchTree.random_walk(chess_board_copy, 
            starting_player_pos, secondary_player_pos, max_step))
            r, c = starting_player_pos
            Node.set_barrier(chess_board_copy, r, c, dir)
            gameOver, strt_player_score, secd_player_score = SearchTree.isTerminalState(chess_board_copy, starting_player_pos, secondary_player_pos)
            if (gameOver): break
            #Secondary Player's Turn
            secondary_player_pos, dir = (SearchTree.random_walk(chess_board_copy, 
            secondary_player_pos, starting_player_pos, max_step))
            r, c = secondary_player_pos
            Node.set_barrier(chess_board_copy, r, c, dir)
            gameOver, secd_player_score, strt_player_score = SearchTree.isTerminalState(chess_board_copy, starting_player_pos, secondary_player_pos)

        #Map the score to the corresponding agent.
        my_score = strt_player_score if node.my_turn else secd_player_score
        adv_score = strt_player_score if not node.my_turn else secd_player_score

        #Return the result of the simulation (win/loss)
        #We consider a tie a loss.
        if (my_score > adv_score):
                return 1
        else:
                return 0

    #Back propagated the result of a simulation starting from node, all the way 
    #up the tree.
    @staticmethod
    def backPropagate(result, node):
        temp = node
    
        while (temp.parentNode != None):
            temp.totalPlays += 1
            temp.wins += result
            temp = temp.parentNode

        temp.totalPlays += 1
        temp.wins += result
        return

    #Checks if the chess_board is in a terminal state, and returns the results of
    #player at my_pos, and adv_pos respectively.
    @staticmethod
    def isTerminalState(chess_board, my_pos, adv_pos):
        board_size = len(chess_board)

         # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    Node.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        return True, p0_score, p1_score
            
    #Performs a random walk for my_pos on the chess_board.
    @staticmethod
    def random_walk(chess_board, my_pos, adv_pos, max_step):
        """
        Randomly walk to the next position in the board.

        Parameters
        ----------
        my_pos : tuple
            The position of the agent.
        adv_pos : tuple
            The position of the adversary.
        """
        ori_pos = deepcopy(my_pos)
        steps = np.random.randint(0, max_step + 1)
        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = Node.moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = Node.moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir


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

    board_size = None
    #my_pos = my position, adv_pos = advesaries position. Position is tuple (row, column).
    #my_turn is boolean representing who's turn it is (our agent or theirs).
    #Wall direction is the direction the wall is placed at my_pos if this node is the result of an action.
    #Otherwise it's none.
    def __init__(self, chess_board, my_pos, adv_pos, my_turn, parentNode, wall_direction):
        #defines a state
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.my_turn = my_turn
        self.terminal = None #Boolean
        self.my_win = None #Either 1 or 0
        self.wall_direction = wall_direction
        #defines MCTS statistics and tree parameters
        self.wins = 0
        self.totalPlays = 0
        self.parentNode = parentNode
        self.childNodes = [] #This...
        self.unusedMoveSet = None #And this could possibly be replaced by a numpy array.

    #This function generates all available moves if they haven't already been created. Uses BFS up to depth max_step to find all legal states
    #we can transition to from this state, stores the legal move (pos, dir) in an array. 
    #Will then generate a child by randomly selected an unusedLegalMove, and return that child.
    def createChildNode(self, max_step):
        #Supposedly arrays are faster than lists, but is there an arraylist like thing in python?
        if (self.unusedMoveSet == None):
            self.unusedMoveSet = MoveSet(self.getBoardSize())

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
                        self.unusedMoveSet.append((cur_pos, dir))
                #If the cur_step == max_step, the player can't move anywhere else from here, so don't add
                #anything to the queue and skip.
                #Else, find each nonvisted neighboring position the player could visit (not blocked by barrier
                #or by opposing agent), add its position and corresponding number of steps needed to get there 
                #to the queue.
                if cur_step != max_step:
                    for dir in range(0,4):
                        new_pos = tuple(map(operator.add, cur_pos, Node.moves[dir]))
                        if (not self.chess_board[row, column, dir] and tuple(new_pos) not in visited 
                                and not np.array_equal(new_pos, self.my_pos) and not np.array_equal(new_pos, self.adv_pos)):
                            visited.add(tuple(new_pos))    
                            state_queue.append((new_pos, cur_step + 1))
        #The unused available moves has either just been created or has already been created.
        #So now we select one at random, generate a child for it, and return that child.
        newpos, newdir = self.unusedMoveSet.popRandomMove()
        chess_board_copy = deepcopy(self.chess_board)
        r, c = newpos
        Node.set_barrier(chess_board_copy, r, c, newdir)
        if self.my_turn:
            newnode = Node(chess_board_copy, newpos, self.adv_pos, False, self, newdir)
            self.childNodes.append(newnode)
            return newnode
        else:
            newnode = Node(chess_board_copy, self.my_pos, newpos, True, self, None)
            self.childNodes.append(newnode)
            return newnode

    #Returns a boolean,int  representing if this node is a Terminal node (true/false) and if so, then if it's a win
    #for our agent or a loss (1/0)
    def isTerminal(self):
        if (self.terminal != None): #If we already computed if this node is a terminal
            return self.terminal, self.my_win
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        board_size = self.getBoardSize()

         # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    Node.moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(self.my_pos))
        p1_r = find(tuple(self.adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            self.terminal = False
            return False, 0
        self.terminal = True
        if (p0_score > p1_score):
            self.my_win = 1
            return True, 1
        else:
            self.my_win = 0
            return True, 0

    #Should only be applied to nodes with parents. Does not make sense to compute
    #UCB1 value of a root node.
    def UCB1(self):
        if(self.totalPlays == 0):
            return math.inf
        else:
            return (self.wins/self.totalPlays +
             1.4142 * math.sqrt(math.log2(self.parentNode.totalPlays)/self.totalPlays))



#Node helper methods:

    #I'm assuming python passes chess_board by reference...?
    @staticmethod
    def set_barrier(chess_board, r, c, dir):
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = Node.moves[dir]
        chess_board[r + move[0], c + move[1], Node.opposites[dir]] = True

    def getBoardSize(self):
        if (Node.board_size == None):
            Node.board_size = len(self.chess_board)
        return Node.board_size

#ToDo: Implement a class to represent a set of moves using a single python array of integers as the underlying data structure.
#Every three elements in the array represents a move.
#Functions needed to be implemented must behave the same as their use above with lists such as in createChildNode.
#isEmpty() to replace node.unusedMoveSet != []
#constructor to replace line 326
#create function to replace self.unusedMoveSet.append((cur_pos, dir)) with same parameters.
#replace newpos, newdir = self.unusedMoveSet.pop(random.randrange(len(self.unusedMoveSet))) with .size() and .pop() function with same parameters and return type.
class MoveSet: 
    def __init__(self, board_size): 
        #Version 1
        #self.tuples = array("b")
        #self.elementsAdded = 0
        #self.elementsRemoved = 0

        #Version 2
        #self.tuples = set()

        #Version 3
        #self.tuples = []

        #Version 4 (Numpy Attempt)
        if (board_size == 4):
            self.tuples = np.empty(108, dtype=np.int8)
        elif (board_size == 5):
            self.tuples = np.empty(216, dtype=np.int8)
        elif (board_size == 6):
            self.tuples = np.empty(252, dtype=np.int8)
        elif (board_size == 7):
            self.tuples = np.empty(408, dtype=np.int8)
        elif (board_size == 8):
            self.tuples = np.empty(444, dtype=np.int8)
        elif (board_size == 9):
            self.tuples = np.empty(648, dtype=np.int8)
        elif (board_size == 10):
            self.tuples = np.empty(684, dtype=np.int8)
        elif (board_size == 11):
            self.tuples = np.empty(936, dtype=np.int8)
        elif (board_size == 12):
            self.tuples = np.empty(972, dtype=np.int8)
        else:
            raise Exception("The Board Size is larger than 12! This isn't allowed!")

        self.elementsAdded = 0
        self.elementsRemoved = 0

    def append(self, tuple): #((r, c), d)
        #Version 1
        #pos, d = tuple
        #r , c = pos
        #self.tuples.append(r)
        #self.tuples.append(c)
        #self.tuples.append(d)
        #self.elementsAdded += 1

        #Version 2
        #self.tuples.add(tuple)

        #Version 3
        #self.tuples.append(tuple)

        #Version 4
        pos, d = tuple
        r , c = pos
        startIndex = self.elementsAdded*3
        self.tuples[startIndex] = r
        self.tuples[startIndex+1] = c
        self.tuples[startIndex+2] = d
        self.elementsAdded += 1


    def popRandomMove(self):
        #Version 1
        #maxIndex = self.elementsAdded*3 - 1
        #randomIndex = random.randrange(self.elementsAdded)*3
        #while (self.tuples[randomIndex] == -1):
        #    if (randomIndex == maxIndex):
        #        randomIndex = 0
        #    else:
        #        randomIndex += 1
        #r = self.tuples[randomIndex]
        #self.tuples[randomIndex] = -1
        #c = self.tuples[randomIndex + 1]
        #self.tuples[randomIndex + 1] = -1
        #d = self.tuples[randomIndex + 2]
        #self.tuples[randomIndex + 2] = -1
        #self.elementsRemoved += 1
        #return (r, c), d 

        #Version 2
        #return self.tuples.pop()

        #Version 3
        #return self.tuples.pop(random.randrange(len(self.tuples)))

        #Version 4
        maxIndex = self.elementsAdded*3 - 1
        randomIndex = random.randrange(self.elementsAdded)*3
        while (self.tuples[randomIndex] == -1):
            randomIndex += 3
            if (randomIndex >= maxIndex):
                randomIndex = 0
            
        r = self.tuples[randomIndex]
        self.tuples[randomIndex] = -1
        c = self.tuples[randomIndex + 1]
        self.tuples[randomIndex + 1] = -1
        d = self.tuples[randomIndex + 2]
        self.tuples[randomIndex + 2] = -1
        self.elementsRemoved += 1
        return (r, c), d 

    def isEmpty(self):
        i = self.elementsAdded - self.elementsRemoved
        return (i == 0)
