
from __future__ import annotations

from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice, random
from math import sqrt, log
from timeit import default_timer as time

num_nodes = 1000
explore_faction = 1.6
DEBUG = False
TIME_BUDGET_S: float | None = None
LAST_ITERATIONS = 0
ROLLOUT_EPSILON = 0.10  # chance to take a random action during rollout

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


def _two_in_row_count(board: Board, state, R: int, C: int, player: int) -> int:
    """Counts how many 'two-in-a-row with one empty' lines exist for player on a local board."""
    board_index = 2 * (3 * R + C)
    p_mask = state[board_index + (player - 1)]
    o_mask = state[board_index + (2 - (player - 1))]  # other player index
    occ = p_mask | o_mask

    count = 0
    for w in board.wins:
        if w & o_mask:
            continue
        bits = p_mask & w
        if bits and (bits & (bits - 1)) and ((w & ~occ) != 0):
            # at least 2 bits set for player in this winning line and 1 empty
            count += 1
    return count


def _opponent_has_immediate_big_win(board: Board, state, opponent: int) -> bool:
    """True if opponent can win the whole game in one move from state."""
    for a in board.legal_actions(state):
        s2 = board.next_state(state, a)
        if board.is_ended(s2):
            pts = board.points_values(s2)
            if pts is not None and pts[opponent] == 1:
                return True
    return False


def _heuristic_rollout_action(board: Board, state):
    """Pick a rollout move using a cheap heuristic + tie-breaking randomness."""
    actions = board.legal_actions(state)
    if len(actions) == 1:
        return actions[0]

    me = board.current_player(state)
    opp = 3 - me

    if random() < ROLLOUT_EPSILON:
        return choice(actions)

    best_score = float("-inf")
    best_actions = []

    me_big_before = state[17 + me]

    for a in actions:
        R, C, r, c = a
        s1 = board.next_state(state, a)

        if board.is_ended(s1):
            pts = board.points_values(s1)
            if pts is not None and pts[me] == 1:
                return a  # immediate game-winning move
            # draw or loss are still considered, but not auto-chosen

        score = 0.0

        # Big-board progress: did this move win a local board for us?
        me_big_after = s1[17 + me]
        if me_big_after != me_big_before:
            score += 120.0

        # Avoid giving opponent an immediate win of the whole game.
        if _opponent_has_immediate_big_win(board, s1, opponent=opp):
            score -= 250.0

        # Local tactical potential on the board we just played in.
        score += 8.0 * _two_in_row_count(board, s1, R, C, me)
        score -= 6.0 * _two_in_row_count(board, s1, R, C, opp)

        # Positional nudges (small, but help rollouts).
        if (R, C) == (1, 1):
            score += 3.0  # center big-board
        if (r, c) == (1, 1):
            score += 2.0  # center of local board

        if s1[20] is None:
            score -= 2.0

        if score > best_score:
            best_score = score
            best_actions = [a]
        elif score == best_score:
            best_actions.append(a)

    return choice(best_actions) if best_actions else choice(actions)


def rollout(board: Board, state):
    """Heuristic rollout: mostly-greedy playout using cheap tactical signals.

    Args:
        board:  The game setup.
        state:  The state of the game.
    
    Returns:
        state: The terminal game state

    """
    while not board.is_ended(state):
        a = _heuristic_rollout_action(board, state)
        state = board.next_state(state, a)
    return state


def backpropagate(node: MCTSNode | None, result: float):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        result: Terminal result from the root bot's perspective.

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

    parent_visits = node.parent.visits if node.parent is not None else 0
    if parent_visits <= 0:
        return exploitation
    return exploitation + explore_faction * sqrt(log(parent_visits) / node.visits)

def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node
    
    """
    if not root_node.child_nodes:
        return choice(root_node.untried_actions)

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
