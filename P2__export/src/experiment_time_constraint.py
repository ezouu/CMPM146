"""
Experiment 3 – Time as a Constraint (Extra Credit)

Run vanilla and modified MCTS under a fixed time budget per move and report:
- Average MCTS iterations (rollouts) per move for each bot (a proxy for "tree size")
- Optional: win counts if you pit them against each other

Examples:
  python3 src/experiment_time_constraint.py --time 1.0 --games 10 --seed 1
  python3 src/experiment_time_constraint.py --times 0.25 0.5 1.0 2.0 --games 5 --seed 1
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import p2_t3


ThinkFn = Callable[[p2_t3.Board, tuple], tuple]


def _load_bot(module_filename: str, time_budget_s: float):
    here = Path(__file__).resolve().parent
    path = here / module_filename

    unique = f"{module_filename.replace('.py','')}_t{time_budget_s}_{random.randint(0, 1_000_000)}"
    spec = importlib.util.spec_from_file_location(unique, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique] = mod
    spec.loader.exec_module(mod)  # type: ignore[assignment]

    # Enable time-budget mode and disable debug prints.
    mod.TIME_BUDGET_S = float(time_budget_s)
    if hasattr(mod, "DEBUG"):
        mod.DEBUG = False
    return mod


@dataclass
class Stats:
    moves: int = 0
    total_iters: int = 0

    def add(self, iters: int):
        self.moves += 1
        self.total_iters += int(iters)

    @property
    def avg_iters(self) -> float:
        return (self.total_iters / self.moves) if self.moves else 0.0


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


def play_game(board: p2_t3.Board, p1_mod, p2_mod, seed: int, s1: Stats, s2: Stats) -> int | str:
    random.seed(seed)
    state = board.starting_state()
    current = 1

    while not board.is_ended(state):
        if current == 1:
            a = p1_mod.think(board, state)
            s1.add(getattr(p1_mod, "LAST_ITERATIONS", 0))
        else:
            a = p2_mod.think(board, state)
            s2.add(getattr(p2_mod, "LAST_ITERATIONS", 0))
        state = board.next_state(state, a)
        current = 2 if current == 1 else 1

    pts = board.points_values(state)
    assert pts is not None
    if pts[1] == 1:
        return 1
    if pts[2] == 1:
        return 2
    return "draw"


def run(time_budget_s: float, games: int, seed: int, *, progress_every: int = 1):
    board = p2_t3.Board()
    vanilla = _load_bot("mcts_vanilla.py", time_budget_s)
    modified = _load_bot("mcts_modified.py", time_budget_s)

    # Alternate who goes first each game.
    mod_stats = Stats()
    van_stats = Stats()
    wins = {"modified": 0, "vanilla": 0, "draw": 0}

    t0 = perf_counter()
    for i in range(games):
        game_t0 = perf_counter()
        if i % 2 == 0:
            # P1 modified, P2 vanilla
            winner = play_game(board, modified, vanilla, seed + i, mod_stats, van_stats)
            if winner == 1:
                wins["modified"] += 1
            elif winner == 2:
                wins["vanilla"] += 1
            else:
                wins["draw"] += 1
        else:
            # P1 vanilla, P2 modified
            winner = play_game(board, vanilla, modified, seed + i, van_stats, mod_stats)
            if winner == 1:
                wins["vanilla"] += 1
            elif winner == 2:
                wins["modified"] += 1
            else:
                wins["draw"] += 1

        # Progress output
        done = i + 1
        if progress_every > 0 and (done % progress_every == 0 or done == games):
            elapsed = perf_counter() - t0
            avg = elapsed / done
            eta = avg * (games - done)
            last_game = perf_counter() - game_t0
            print(
                f"  [{done:>3}/{games}] last_game={_fmt_secs(last_game)} "
                f"elapsed={_fmt_secs(elapsed)} eta={_fmt_secs(eta)} | "
                f"avg_iters: van={van_stats.avg_iters:.1f} mod={mod_stats.avg_iters:.1f} | "
                f"wins: {wins}"
            )

    print(f"\nTime budget: {time_budget_s:.2f}s per move")
    print(f"Avg iterations per move: vanilla={van_stats.avg_iters:.1f} | modified={mod_stats.avg_iters:.1f}")
    print(f"Win counts (alternating first): {wins}")


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--time", type=float, default=None, help="Single time budget (seconds) per move")
    p.add_argument("--times", type=float, nargs="*", default=[], help="Multiple time budgets to test")
    p.add_argument("--games", type=int, default=10, help="Games per time budget")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=1, help="Print progress every N games (0 disables)")
    args = p.parse_args(argv)

    budgets: List[float] = []
    if args.time is not None:
        budgets.append(args.time)
    budgets.extend(args.times)
    if not budgets:
        budgets = [1.0]

    for t in budgets:
        print(f"Running time budget {float(t):.2f}s for {args.games} games ...")
        run(float(t), args.games, args.seed, progress_every=args.progress_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

