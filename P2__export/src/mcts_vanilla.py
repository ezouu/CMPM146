
from __future__ import annotations

from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log
from timeit import default_timer as time

num_nodes = 1000
explore_faction = 2.0
DEBUG = False
TIME_BUDGET_S: float | None = None
LAST_ITERATIONS = 0

def traverse_nodes(node: MCTSNode, board: Board, state, bot_identity: int):
    """ Traverses the tree until the end criterion are met.
    e.g. find the best expandable node (node with untried action) if it exist,
    or else a terminal node

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 1 or 2

    Returns:
        node: A node from which the next stage of the search can proceed.
        state: The state associated with that node

    """
    while True:
        if board.is_ended(state):
            return node, state

        # If there are actions we haven't explored from this node, stop here.
        if node.untried_actions:
            return node, state

        if not node.child_nodes:
            return node, state

        actor = board.current_player(state)
        is_opponent = actor != bot_identity
        best_action, best_child = max(
            node.child_nodes.items(),
            key=lambda kv: ucb(kv[1], is_opponent),
        )
        node = best_child
        state = board.next_state(state, best_action)

def expand_leaf(node: MCTSNode, board: Board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node (if it is non-terminal).

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:
        node: The added child node
        state: The state associated with that node

    """
    if board.is_ended(state):
        return node, state

    if not node.untried_actions:
        return node, state

    # Pick an untried action uniformly at random (vanilla expansion).
    action = choice(node.untried_actions)
    node.untried_actions.remove(action)

    next_state = board.next_state(state, action)
    child = MCTSNode(
        parent=node,
        parent_action=action,
        action_list=board.legal_actions(next_state),
    )
    node.child_nodes[action] = child
    return child, next_state


def rollout(board: Board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.
    
    Returns:
        state: The terminal game state

    """
    while not board.is_ended(state):
        actions = board.legal_actions(state)
        state = board.next_state(state, choice(actions))
    return state


def backpropagate(node: MCTSNode | None, result: float):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        result: Terminal result from the root bot's perspective.
                Typically 1.0 for win, 0.0 for loss, 0.5 for draw.

    """
    while node is not None:
        node.visits += 1
        node.wins += result
        node = node.parent

def ucb(node: MCTSNode, is_opponent: bool):
    """ Calcualtes the UCB value for the given node from the perspective of the bot

    Args:
        node:   A node.
        is_opponent: A boolean indicating whether or not the last action was performed by the MCTS bot
    Returns:
        The value of the UCB function for the given node
    """

    if node.visits == 0:
        return float("inf")
    exploitation = node.wins / node.visits
    if is_opponent:
        exploitation = 1.0 - exploitation

    # Exploration term.
    parent_visits = node.parent.visits if node.parent is not None else 0
    if parent_visits <= 0:
        return exploitation
    exploration = explore_faction * sqrt(log(parent_visits) / node.visits)
    return exploitation + exploration

def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node
    
    """
    if not root_node.child_nodes:
        # Fallback: if we never expanded, pick any legal move.
        return choice(root_node.untried_actions)

    # Common robust choice: highest visit count.
    def score(item):
        action, child = item
        win_rate = (child.wins / child.visits) if child.visits else 0.0
        return (child.visits, win_rate)

    best_action, _ = max(root_node.child_nodes.items(), key=score)
    return best_action

def is_win(board: Board, state, identity_of_bot: int):
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1

def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        current_state:  The current state of the game.

    Returns:    The action to be taken from the current state

    """
    bot_identity = board.current_player(current_state) # 1 or 2
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(current_state))

    global LAST_ITERATIONS
    LAST_ITERATIONS = 0

    start = time()
    def keep_going():
        if TIME_BUDGET_S is None:
            return LAST_ITERATIONS < num_nodes
        return (time() - start) < TIME_BUDGET_S

    while keep_going():
        state = current_state
        node = root_node

        # 1) Selection
        node, state = traverse_nodes(node, board, state, bot_identity)

        # 2) Expansion
        if not board.is_ended(state):
            node, state = expand_leaf(node, board, state)

        # 3) Simulation
        terminal_state = rollout(board, state)

        # 4) Backpropagation
        terminal_score = board.points_values(terminal_state)
        assert terminal_score is not None
        v = terminal_score[bot_identity]
        result = 1.0 if v == 1 else (0.5 if v == 0 else 0.0)
        backpropagate(node, result)
        LAST_ITERATIONS += 1
    best_action = get_best_action(root_node)
    
    if DEBUG:
        print(f"Action chosen: {best_action}")
    return best_action
