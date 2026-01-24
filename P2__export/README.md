#Edward Zou P2

#implemented selection (`traverse_nodes`), expansion (`expand_leaf`), simulation (`rollout`), and backpropagation (`backpropagate`) so the bot runs end-to-end like the vanilla version.
#replaced the fully-random rollout with `_heuristic_rollout_action`, which prioritizes immediate wins, sub-board captures, local two-in-a-row threats, and basic positional preferences (with small random tie-breaking).
#added a penalty for moves that allow the opponent an immediate game-winning reply (`_opponent_has_immediate_big_win`), to reduce catastrophic rollout lines.
#introduced `ROLLOUT_EPSILON` so rollouts are mostly heuristic-guided but still occasionally take random actions to avoid over-biasing and to keep the search stochastic.
#added `TIME_BUDGET_S` (switch from fixed `num_nodes` to “search until time runs out”) and `LAST_ITERATIONS` (how many rollouts were completed in the last `think()` call) for the time-constraint experiment.

