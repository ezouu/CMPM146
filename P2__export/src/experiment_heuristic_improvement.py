"""
Experiment 2 – Heuristic Improvement

Play modified MCTS vs vanilla MCTS with equal tree sizes.
Runs N games per size and alternates which bot goes first each game to reduce first-move bias.

Example:
  python3 src/experiment_heuristic_improvement.py --rounds 100 --sizes 250 500 1000 --seed 1
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import p2_t3


ThinkFn = Callable[[p2_t3.Board, tuple], tuple]


def _load_module(path: Path, unique_name: str, num_nodes: int):
    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]

    module.num_nodes = int(num_nodes)
    if hasattr(module, "DEBUG"):
        module.DEBUG = False
    return module


def make_bot(module_filename: str, num_nodes: int) -> ThinkFn:
    here = Path(__file__).resolve().parent
    path = here / module_filename
    mod = _load_module(path, f"{module_filename.replace('.py','')}_{num_nodes}_{random.randint(0, 1_000_000)}", num_nodes)
    return mod.think


@dataclass
class Result:
    modified_wins: int = 0
    vanilla_wins: int = 0
    draws: int = 0


def play_game(board: p2_t3.Board, p1: ThinkFn, p2: ThinkFn, seed: int) -> int | str:
    random.seed(seed)
    state = board.starting_state()
    current = p1
    while not board.is_ended(state):
        a = current(board, state)
        state = board.next_state(state, a)
        current = p1 if current is p2 else p2

    pts = board.points_values(state)
    assert pts is not None
    if pts[1] == 1:
        return 1
    if pts[2] == 1:
        return 2
    return "draw"


def _fmt_secs(s: float) -> str:
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    r = s - 60 * m
    if m < 60:
        return f"{m}m {r:.0f}s"
    h = int(m // 60)
    mm = m - 60 * h
    return f"{h}h {mm}m"


def run_equal_size(rounds: int, size: int, seed: int, *, progress_every: int = 5) -> Result:
    board = p2_t3.Board()
    modified = make_bot("mcts_modified.py", size)
    vanilla = make_bot("mcts_vanilla.py", size)

    res = Result()
    t0 = perf_counter()
    for i in range(rounds):
        game_t0 = perf_counter()
        # Alternate who is Player 1 to reduce first-move bias.
        if i % 2 == 0:
            p1, p2 = modified, vanilla
            winner = play_game(board, p1, p2, seed=seed + i)
            if winner == 1:
                res.modified_wins += 1
            elif winner == 2:
                res.vanilla_wins += 1
            else:
                res.draws += 1
        else:
            p1, p2 = vanilla, modified
            winner = play_game(board, p1, p2, seed=seed + i)
            if winner == 1:
                res.vanilla_wins += 1
            elif winner == 2:
                res.modified_wins += 1
            else:
                res.draws += 1

        # Progress output
        done = i + 1
        if progress_every > 0 and (done % progress_every == 0 or done == rounds):
            elapsed = perf_counter() - t0
            avg = elapsed / done
            eta = avg * (rounds - done)
            game_elapsed = perf_counter() - game_t0
            print(
                f"  [{done:>3}/{rounds}] "
                f"last_game={_fmt_secs(game_elapsed)} "
                f"elapsed={_fmt_secs(elapsed)} "
                f"eta={_fmt_secs(eta)} | "
                f"mod={res.modified_wins} van={res.vanilla_wins} draw={res.draws}"
            )
    return res


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=100, help="Games per tree size (100 recommended)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[250, 500, 1000], help="Tree sizes to test")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("--progress_every", type=int, default=5, help="Print progress every N games (0 disables)")
    args = parser.parse_args(argv)

    results: Dict[int, Result] = {}
    for s in args.sizes:
        print(f"Running size={s}: modified vs vanilla for {args.rounds} games (alternating first) ...")
        r = run_equal_size(args.rounds, s, args.seed, progress_every=args.progress_every)
        results[s] = r
        print(f"  results: modified wins={r.modified_wins}, vanilla wins={r.vanilla_wins}, draws={r.draws}")

    print("\nSummary (tree size -> wins):")
    for s in sorted(results.keys()):
        r = results[s]
        print(f"  {s:>5}: modified wins={r.modified_wins} | vanilla wins={r.vanilla_wins} | draws={r.draws}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

