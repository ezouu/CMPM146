"""
Experiment 1 – Tree Size

Pit two vanilla MCTS bots against each other:
- Player 1 fixed at 100 nodes/tree
- Player 2 varies across several tree sizes

This script runs N games per size and saves a plot image.

Example:
  python src/experiment_tree_size.py --rounds 100 --sizes 50 100 250 500 1000
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import p2_t3


ThinkFn = Callable[[p2_t3.Board, tuple], tuple]


def _load_mcts_vanilla_module(unique_name: str, num_nodes: int):
    """Load a fresh copy of mcts_vanilla.py under a unique module name."""
    here = Path(__file__).resolve().parent
    mcts_path = here / "mcts_vanilla.py"

    spec = importlib.util.spec_from_file_location(unique_name, str(mcts_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load mcts_vanilla.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]

    # Configure this instance.
    module.num_nodes = int(num_nodes)
    if hasattr(module, "DEBUG"):
        module.DEBUG = False

    return module


def make_vanilla_bot(num_nodes: int) -> ThinkFn:
    mod = _load_mcts_vanilla_module(f"mcts_vanilla_{num_nodes}", num_nodes)
    return mod.think


@dataclass
class MatchResult:
    p1_wins: int = 0
    p2_wins: int = 0
    draws: int = 0


def play_game(board: p2_t3.Board, p1: ThinkFn, p2: ThinkFn, seed: int) -> int | str:
    """Returns winner: 1, 2, or 'draw'."""
    random.seed(seed)
    state = board.starting_state()
    current_player = p1

    # Play until terminal
    while not board.is_ended(state):
        action = current_player(board, state)
        state = board.next_state(state, action)
        current_player = p1 if current_player is p2 else p2

    final_score = board.points_values(state)
    assert final_score is not None
    if final_score[1] == 1:
        return 1
    if final_score[2] == 1:
        return 2
    return "draw"


def run_match(rounds: int, p1_nodes: int, p2_nodes: int, seed: int) -> MatchResult:
    board = p2_t3.Board()
    p1 = make_vanilla_bot(p1_nodes)
    p2 = make_vanilla_bot(p2_nodes)

    res = MatchResult()
    for i in range(rounds):
        winner = play_game(board, p1, p2, seed=seed + i)
        if winner == 1:
            res.p1_wins += 1
        elif winner == 2:
            res.p2_wins += 1
        else:
            res.draws += 1
    return res


def _save_svg_plot(results: Dict[int, MatchResult], out_path: Path, p1_nodes: int, rounds: int):
    """Dependency-free bar plot written as an SVG."""
    sizes = sorted(results.keys())
    series = [
        ("P1 wins", "#4c78a8", [results[s].p1_wins for s in sizes]),
        ("P2 wins", "#f58518", [results[s].p2_wins for s in sizes]),
        ("Draws", "#54a24b", [results[s].draws for s in sizes]),
    ]
    max_y = max(1, max(v for _, _, vals in series for v in vals))

    # Canvas + layout
    W, H = 1000, 520
    margin = dict(l=80, r=30, t=60, b=90)
    plot_w = W - margin["l"] - margin["r"]
    plot_h = H - margin["t"] - margin["b"]

    def x0(i: int) -> float:
        return margin["l"] + (i + 0.5) * (plot_w / max(1, len(sizes)))

    def y(v: float) -> float:
        # SVG y grows downward
        return margin["t"] + plot_h * (1.0 - (v / max_y))

    # Bar geometry
    group_w = plot_w / max(1, len(sizes))
    bar_w = min(40.0, group_w / 4.0)
    offsets = [-bar_w * 1.2, 0.0, bar_w * 1.2]

    # Build SVG
    title = f"Experiment 1: Vanilla MCTS win counts vs tree size (P1 fixed at {p1_nodes})"
    xlabel = "Player 2 tree size (num_nodes)"
    ylabel = "Count (over N games)"

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="18">{title}</text>')

    # Axes
    x_axis_y = margin["t"] + plot_h
    parts.append(f'<line x1="{margin["l"]}" y1="{x_axis_y}" x2="{W-margin["r"]}" y2="{x_axis_y}" stroke="#333" stroke-width="1"/>')
    parts.append(f'<line x1="{margin["l"]}" y1="{margin["t"]}" x2="{margin["l"]}" y2="{x_axis_y}" stroke="#333" stroke-width="1"/>')

    # Y ticks (0, 25, 50, 75, 100%)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        v = max_y * frac
        yy = y(v)
        parts.append(f'<line x1="{margin["l"]-5}" y1="{yy}" x2="{W-margin["r"]}" y2="{yy}" stroke="#eee" stroke-width="1"/>')
        parts.append(f'<text x="{margin["l"]-10}" y="{yy+4}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#333">{int(round(v))}</text>')

    # Bars
    for i, s in enumerate(sizes):
        cx = x0(i)
        for (label, color, vals), off in zip(series, offsets):
            v = vals[i]
            x = cx + off - bar_w / 2.0
            y_top = y(v)
            h = x_axis_y - y_top
            parts.append(f'<rect x="{x:.2f}" y="{y_top:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}"/>')

        # X tick label
        parts.append(f'<text x="{cx}" y="{x_axis_y+25}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#333">{s}</text>')

    # Axis labels
    parts.append(f'<text x="{W/2}" y="{H-30}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#333">{xlabel}</text>')
    # Rotated y-label
    parts.append(f'<text x="25" y="{H/2}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#333" transform="rotate(-90 25 {H/2})">{ylabel}</text>')

    # Legend
    legend_x = W - margin["r"] - 260
    legend_y = margin["t"] + 10
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="250" height="70" fill="white" stroke="#ddd"/>')
    for j, (label, color, _) in enumerate(series):
        ly = legend_y + 20 + j * 20
        parts.append(f'<rect x="{legend_x+10}" y="{ly-12}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x+30}" y="{ly-2}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#333">{label}</text>')

    # Footer info
    parts.append(
        f'<text x="{margin["l"]}" y="{H-10}" text-anchor="start" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#666">'
        f"Rounds per size: {rounds} | P1 tree size: {p1_nodes}"
        "</text>"
    )

    parts.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved plot to: {out_path}")


def try_plot(results: Dict[int, MatchResult], out_path: Path, *, p1_nodes: int, rounds: int):
    """Save a bar plot. Uses matplotlib if available; otherwise writes an SVG plot."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print("matplotlib not available; writing SVG fallback instead:", repr(e))
        svg_path = out_path
        if svg_path.suffix.lower() != ".svg":
            svg_path = svg_path.with_suffix(".svg")
        _save_svg_plot(results, svg_path, p1_nodes=p1_nodes, rounds=rounds)
        return

    sizes = sorted(results.keys())
    p1_w = [results[s].p1_wins for s in sizes]
    p2_w = [results[s].p2_wins for s in sizes]
    drw = [results[s].draws for s in sizes]

    x = list(range(len(sizes)))
    width = 0.28

    plt.figure(figsize=(10, 5))
    plt.bar([i - width for i in x], p1_w, width=width, label="P1 wins (100 nodes)")
    plt.bar(x, p2_w, width=width, label="P2 wins (varying nodes)")
    plt.bar([i + width for i in x], drw, width=width, label="Draws")

    plt.xticks(x, [str(s) for s in sizes])
    plt.xlabel("Player 2 tree size (num_nodes)")
    plt.ylabel("Count (over N games)")
    plt.title("Experiment 1: Vanilla MCTS win counts vs tree size (P1 fixed at 100)")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to: {out_path}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=100, help="Games per tree size (>=100 recommended)")
    parser.add_argument("--p1_nodes", type=int, default=100, help="Fixed tree size for Player 1")
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 250, 500, 1000], help="Player 2 tree sizes")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument(
        "--out",
        type=str,
        default=str((Path(__file__).resolve().parent.parent / "tree_size_experiment.png")),
        help="Output image path",
    )
    args = parser.parse_args(argv)

    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")

    results: Dict[int, MatchResult] = {}
    for s in args.sizes:
        print(f"Running: P1={args.p1_nodes} vs P2={s} for {args.rounds} games ...")
        results[s] = run_match(rounds=args.rounds, p1_nodes=args.p1_nodes, p2_nodes=s, seed=args.seed)
        r = results[s]
        print(f"  results: P1 wins={r.p1_wins}, P2 wins={r.p2_wins}, draws={r.draws}")

    # Attach labels for SVG fallback.
    out_path = Path(args.out)
    try:
        try_plot(results, out_path, p1_nodes=args.p1_nodes, rounds=args.rounds)
    except Exception:
        # If something unexpected happened during matplotlib plotting, still produce SVG.
        svg_path = out_path if out_path.suffix.lower() == ".svg" else out_path.with_suffix(".svg")
        _save_svg_plot(results, svg_path, p1_nodes=args.p1_nodes, rounds=args.rounds)

    print("\nSummary (P2 tree size -> wins):")
    for s in sorted(results.keys()):
        r = results[s]
        print(f"  {s:>5}: P2 wins={r.p2_wins} | P1 wins={r.p1_wins} | draws={r.draws}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

