import copy

# Tic Tac Toe game for 7x7 board

# Initialize the board with all empty cells
board = [[' ' for j in range(7)] for i in range(7)]

# Define the function to print the board
def print_board(state):
    for row in state:
        print(' | '.join(row))

# Define the function to check if there's a winner
def check_winner(player, state):
    # Check rows for a win
    for i in range(7):
        for j in range(4):
            if all([state[i][j+k] == player for k in range(4)]):
                return True
    # Check columns for a win
    for i in range(4):
        for j in range(7):
            if all([state[i+k][j] == player for k in range(4)]):
                return True
    # Check east-south diagonal for a win
    for i in range(4):
        for j in range(4):
            if all([state[i+k][j+k] == player for k in range(4)]):
                return True
    # Check west-south diagonal for a win
    for i in range(4):
        for j in range(3, 7):
            if all([state[i+k][j-k] == player for k in range(4)]):
                return True
    return False

def evaluate_board(state):

    # evaluate board function is misleading as the AI opponent just makes the move sequentially (selects the next empty/available cell)

    # # Evaluate the board by counting the number of 3-in-a-row for each player and get a score
    # x_3_in_a_row = o_3_in_a_row = 0
    # for i in range(7): # row wise
    #     for j in range(4):

    #         if 'X' in [state[i][j+k] for k in range(3)] and ' ' in [state[i][j+k] for k in range(4)]:
    #             x_3_in_a_row += 1
    #         elif 'O' in [state[i][j+k] for k in range(4)] and ' ' in [state[i][j+k] for k in range(4)]:
    #             o_3_in_a_row += 1

    # for i in range(4): # col wise
    #     for j in range(7):

    #         if 'X' in [state[i+k][j] for k in range(4)] and ' ' in [state[i+k][j] for k in range(4)]:
    #             x_3_in_a_row += 1
    #         elif 'O' in [state[i+k][j] for k in range(4)] and ' ' in [state[i+k][j] for k in range(4)]:
    #             o_3_in_a_row += 1

    # for i in range(4): # east-south diagonal wise
    #     for j in range(4):

    #         if 'X' in [state[i+k][j+k] for k in range(4)] and ' ' in [state[i+k][j+k] for k in range(4)]:
    #             x_3_in_a_row += 1
    #         elif 'O' in [state[i+k][j+k] for k in range(4)] and ' ' in [state[i+k][j+k] for k in range(4)]:
    #             o_3_in_a_row += 1

    # for i in range(4): # west-south diagonal wise
    #     for j in range(3, 7):

    #         if 'X' in [state[i+k][j-k] for k in range(4)] and ' ' in [state[i+k][j-k] for k in range(4)]:
    #             x_3_in_a_row += 1
    #         elif 'O' in [state[i+k][j-k] for k in range(4)] and ' ' in [state[i+k][j-k] for k in range(4)]:
    #             o_3_in_a_row += 1

    # if x_3_in_a_row > o_3_in_a_row:
    #     return 1
    # elif o_3_in_a_row > x_3_in_a_row:
    #     return -1
    # else: return 0

    # although there can be a much more optimal approach but the following method (a very naive approach) serves as one which is slightly better than that above:
    
    # directly checking the winner:
    if (check_winner('X', state)):
        return 1
    elif (check_winner('O', state)):
        return -1
    
    return 0

# Define the function to play the game
def play_game():
    # Initialize player and turn count
    player = 'X'
    turn_count = 0
    # Loop until a winner is found or the board is full
    while turn_count < 49:
        # Print the board
        print_board(board)
        if player == 'X':
            # Use minimax with alpha-beta pruning to find the best move for player X
            score, row, col = minimax_alpha_beta(player, float('-inf'), float('inf'), 4)
            print("Player X's move: ({}, {})".format(row, col))
            print(f"Best Score: {score}")
        else:
            # Ask player O for their move
            move = input("Player O, enter your move (row, column): ")
            row, col = map(int, move.split(','))
        # Check if the move is valid and update the board
        if board[row][col] == ' ':
            board[row][col] = player
            # Check if the player has won
            if check_winner(player, board):
                print_board(board)
                print("Player " + player + " wins!")
                return
            # Switch to the other player
            player = 'O' if player == 'X' else 'X'
            turn_count += 1
        else:
            print("Invalid move. Try again.")
    # If the loop finishes without a winner, it's a tie
    print_board(board)
    print("It's a tie!")

# Define the minimax function with alpha-beta pruning
def minimax_alpha_beta(player, alpha, beta, depth):
    """
    Implements the minimax algorithm with alpha-beta pruning to determine the best move for the computer player.

    Parameters:
        - player (string): 'X' if the current player is the computer player (maximizing player),
            'O' if the current player is the human player (minimizing player)
        - depth (int): the maximum depth of the game tree to explore
        - alpha (float): the alpha value for alpha-beta pruning
        - beta (float): the beta value for alpha-beta pruning

    Returns:
        - score, row, col
            - best_score, best_row, best_col to play for player 'X'
    """

    state = copy.deepcopy(board)

    alpha_beta_search = max_value(state, alpha, beta, depth)
    score = alpha_beta_search[0]
    best_action = alpha_beta_search[1]

    return score, best_action[0], best_action[1] 


def actions(state):
    """
    defines the actions that can be taken from a given state (in the form of move in [row,column])
    """

    actions = []
    
    k = 0
    for i in range(7):
        for j in range(7):
            if (state[i][j] == ' '):
                actions.append([i, j])

    return actions


def max_value(state, alpha, beta, depth):
    """
    max-value func as the part of the entire minimax func
    """

    # terminal-state test
    if (depth == 0 | terminal_test(state)):
        return evaluate_board(state), None
    
    v = float("-inf")
    best_action = None

    for a in actions(state):
        u = min_value(result('X', state, a), alpha, beta, depth - 1)[0]  # depth is decremented at each func call
        if (u > v):
            v = u
            best_action = a
        # v = max(v, min_value(result(state, a), alpha, beta, depth - 1))
        # for testing:
        # if (depth == 0):
        #     print_board(result('X', state, a))
        if (v >= beta):
            return v, best_action
        alpha = max(alpha, v)

    return v, best_action


def min_value(state, alpha, beta, depth):
    """
    min-value func as the part of the entire minimax func
    """

    # terminal-state test
    if (depth == 0 | terminal_test(state)):
        return evaluate_board(state), None
    
    v = float("inf")    # min-value
    best_action = None

    for a in actions(state):
        u = max_value(result('O', state, a), alpha, beta, depth - 1)[0]  # depth is decremented at each func call
        if (u < v):
            v = u
            best_action = a
        # v = min(v, max_value(result(state, a), alpha, beta, depth - 1))
        # for testing:
        # if (depth == 1):
        #     print_board(result('O', state, a))
        if (v <= alpha):
            return v, best_action
        beta = min(beta, v)

    return v, best_action

def result(player, state, action):
    """
    returns the result of an action on a given state by any player
    """

    result = copy.deepcopy(state)
    result[action[0]][action[1]] = player
    return result

def terminal_test(state):
    """
    returns true if terminal state (end of tree) has been reached, false otherwise
    """
    
    # moves left
    for i in range(7):
        for j in range(7):
            if (state[i][j] == ' '):
                return False
    
    return True

# Start the game
play_game()
