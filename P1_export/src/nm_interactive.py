from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure matplotlib uses a writable config/cache directory (important on some locked-down setups).
if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = Path(__file__).resolve().parent / ".mplconfig"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib.pyplot as plt

try:
    # Running as a script from within `src/`
    from nm_pathfinder import find_path
except ModuleNotFoundError:  # pragma: no cover
    # Running from repo root as `python -m src.nm_interactive` or imported as a module
    from src.nm_pathfinder import find_path

Point = Tuple[float, float]  # (row, col) in original image coordinates


def _load_image(path: str):
    # matplotlib returns floats in [0,1] for PNG, and possibly RGBA.
    return plt.imread(path)


def _subsample_image(img, k: int):
    if k <= 1:
        return img
    if img.ndim == 2:
        return img[::k, ::k]
    return img[::k, ::k, :]


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


@dataclass
class DrawState:
    subsample: int
    src_original: Optional[Point] = None
    dst_original: Optional[Point] = None
    src_artist = None
    dst_artist = None
    path_artist = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive navmesh pathfinding viewer (click to set source then destination)."
    )
    parser.add_argument("image_png", help="PNG image to display")
    parser.add_argument("mesh_pickle", help="Pickled mesh file (.mesh.pickle)")
    parser.add_argument("subsample", type=int, help="Integer subsampling factor for display (e.g., 2)")
    args = parser.parse_args()

    img = _load_image(args.image_png)
    disp = _subsample_image(img, args.subsample)

    with open(args.mesh_pickle, "rb") as f:
        mesh = pickle.load(f)

    state = DrawState(subsample=max(1, int(args.subsample)))

    fig, ax = plt.subplots()
    ax.set_title("Click: source then destination. Right-click: reset.")
    ax.imshow(disp, origin="upper")
    ax.set_axis_off()

    disp_h, disp_w = disp.shape[0], disp.shape[1]

    def clear_artists():
        for a in [state.src_artist, state.dst_artist, state.path_artist]:
            if a is not None:
                try:
                    a.remove()
                except Exception:
                    pass
        state.src_artist = None
        state.dst_artist = None
        state.path_artist = None

    def reset():
        state.src_original = None
        state.dst_original = None
        clear_artists()
        fig.canvas.draw_idle()

    def draw_point(display_row: float, display_col: float):
        # matplotlib plotting uses (x=col, y=row)
        return ax.plot(display_col, display_row, "ro", markersize=6)[0]

    def draw_path(path_original: List[Point]):
        if not path_original:
            return None
        xs = [p[1] / state.subsample for p in path_original]  # cols
        ys = [p[0] / state.subsample for p in path_original]  # rows
        return ax.plot(xs, ys, color="cyan", linewidth=2)[0]

    def on_click(event):
        # Only handle clicks inside the image axes.
        if event.inaxes != ax:
            return

        # Right-click: reset.
        if event.button == 3:
            reset()
            return

        # Left-click only.
        if event.button != 1:
            return

        if event.xdata is None or event.ydata is None:
            return

        # Convert click to display pixel coordinates.
        disp_col = int(event.xdata + 0.5)
        disp_row = int(event.ydata + 0.5)
        disp_col = _clamp_int(disp_col, 0, disp_w - 1)
        disp_row = _clamp_int(disp_row, 0, disp_h - 1)

        # Map display pixel to original image coordinates.
        orig_row = float(disp_row * state.subsample)
        orig_col = float(disp_col * state.subsample)

        # Third click starts over (new source).
        if state.src_original is not None and state.dst_original is not None:
            reset()

        if state.src_original is None:
            state.src_original = (orig_row, orig_col)
            if state.src_artist is not None:
                try:
                    state.src_artist.remove()
                except Exception:
                    pass
            state.src_artist = draw_point(disp_row, disp_col)
            fig.canvas.draw_idle()
            return

        # Set destination, compute/draw path.
        state.dst_original = (orig_row, orig_col)
        if state.dst_artist is not None:
            try:
                state.dst_artist.remove()
            except Exception:
                pass
        state.dst_artist = draw_point(disp_row, disp_col)

        path, _explored = find_path(state.src_original, state.dst_original, mesh)
        if state.path_artist is not None:
            try:
                state.path_artist.remove()
            except Exception:
                pass
        state.path_artist = draw_path(path)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


if __name__ == "__main__":
    main()

