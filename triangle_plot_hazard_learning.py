"""Plot analytical hazard-learning grids in an ``XX /  X`` layout."""

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm

from matplotlib import ticker


def load_grid_module():
    script_path = Path("__file__").with_name("hazard-learning_grids.py")
    spec = importlib.util.spec_from_file_location(
        "hazard_learning_grids_for_plot", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GRID_MODULE = load_grid_module()
parameter_values = GRID_MODULE.parameter_values
build_filename = GRID_MODULE.build_filename
fiducial_values = GRID_MODULE.fiducial_values

COLOR_MAP = matplotlib.colormaps['PuRd']# "cool"#"tab20b"#"viridis"


PAIR_POSITIONS = {
    ("PRACTICE_COST", "SEVERITY_DECAY"): (0, 0),
    ("SEVERITY_DECAY", "HAZARD_RATE"): (0, 1),
    ("PRACTICE_COST", "HAZARD_RATE"): (1, 1),
}


def load_grid(path: Path) -> dict:
    """Load and validate one grid saved by hazard-learning_grids.py."""
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    if not isinstance(payload, dict):
        raise ValueError(f"Grid file does not contain a dictionary: {path}")

    required = {
        "parameter_x",
        "parameter_y",
        "x_values",
        "y_values",
        "practice_use_fraction",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Grid file is missing {sorted(missing)}: {path}")
    return payload


def expected_path(
    data_dir: Path, parameter_x: str, parameter_y: str, points: int
) -> Path:
    """Build the filename using the grid generator's exact sampling."""
    fixed = fiducial_values()
    x_values = parameter_values(parameter_x, points)
    y_values = parameter_values(parameter_y, points)
    return data_dir / build_filename(
        parameter_x, parameter_y, x_values, y_values, fixed
    )


def format_ticks(values: np.ndarray) -> list[str]:
    return [f"{value:.2e}" if abs(value) < 0.01 else f"{value:.3g}" for value in values]


def display_ticks(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return one uniform-grid tick and label for every parameter value."""
    indices = np.arange(values.size)
    positions = (indices + 0.5) / values.size
    return positions, format_ticks(values[indices])


def plot_triangle(data_dir: Path, output_path: Path, points: int) -> Path:
    figure = plt.figure(figsize=(11, 8))
    axes = {}
    meshes = []
    x_axes = {}
    y_axes = {}

    for (parameter_x, parameter_y), (row, column) in PAIR_POSITIONS.items():
        axis = figure.add_subplot(
            2,
            2,
            row * 2 + column + 1,
            sharex=x_axes.get(parameter_x),
            sharey=y_axes.get(parameter_y),
        )
        axes[(parameter_x, parameter_y)] = axis
        x_axes.setdefault(parameter_x, axis)
        y_axes.setdefault(parameter_y, axis)

        path = expected_path(data_dir, parameter_x, parameter_y, points)
        payload = load_grid(path)
        x_values = np.asarray(payload["x_values"], dtype=float)
        y_values = np.asarray(payload["y_values"], dtype=float)
        if parameter_x == "SEVERITY_DECAY" and parameter_y == "HAZARD_RATE":
            matrix = np.asarray(payload["practice_use_fraction"], dtype=float).T
        else:
            matrix = np.asarray(payload["practice_use_fraction"], dtype=float)
        expected_shape = (x_values.size, y_values.size)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Grid shape {matrix.shape} does not match axes "
                f"{expected_shape}: {path}"
            )

        # Use normalized index coordinates so every cell has width 1/N.
        # The labels below still show the actual, potentially logarithmic,
        # parameter values represented by each cell.
        x_edges = np.linspace(0.0, 1.0, x_values.size + 1)
        y_edges = np.linspace(0.0, 1.0, y_values.size + 1)
        mesh = axis.pcolormesh(
            x_edges,
            y_edges,
            matrix.T,
            shading="auto",
            cmap=COLOR_MAP,
            vmin=0.0,
            vmax=1.0,
        )
        meshes.append(mesh)

        # Tick positions and labels come from the same arrays used by pcolormesh.
        x_tick_values, x_tick_labels = display_ticks(x_values)
        y_tick_values, y_tick_labels = display_ticks(y_values)
        axis.set_xticks(x_tick_values)
        axis.set_yticks(y_tick_values)
        axis.set_xticklabels(x_tick_labels, rotation=90, ha="left", fontsize=8)
        axis.set_yticklabels(y_tick_labels, fontsize=8)
        axis.xaxis.set_label_position("top")
        axis.xaxis.tick_top()
        axis.yaxis.set_label_position("right")
        axis.yaxis.tick_right()
        axis.set_xlabel(parameter_x)
        axis.set_ylabel(parameter_y, rotation=270, labelpad=15)

    # Match the reference convention: top-row x labels and right-edge y labels
    # for each occupied row. Shared axes retain the same numerical scale.
    for (parameter_x, parameter_y), axis in axes.items():
        row, column = PAIR_POSITIONS[(parameter_x, parameter_y)]
        right_edge = column == max(
            panel_column
            for panel_row, panel_column in PAIR_POSITIONS.values()
            if panel_row == row
        )
        axis.tick_params(
            axis="x",
            labeltop=row == 0,
            labelbottom=False,
        )
        axis.tick_params(
            axis="y",
            labelright=right_edge,
            labelleft=False,
        )

        #axis.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        #axis.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}'))
        if row != 0:
            axis.set_xlabel("")
        if not right_edge:
            axis.set_ylabel("")

    colorbar_axis = figure.add_axes([0.93, 0.15, 0.018, 0.7])
    cbar = figure.colorbar(meshes[0],cax=colorbar_axis)
    cbar.set_label("Final practice-use fraction", rotation=-90, labelpad=22, fontsize=12)
    figure.subplots_adjust(wspace=0, hspace=0, right=0.85)
    

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved triangular plot to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data/hazard_learning_grids")
    parser.add_argument("--points", type=int, default=15)
    parser.add_argument(
        "--output",
        default="./plots/triangle_hazard_learning_practice_use.png",
    )
    args = parser.parse_args()
    if args.points < 2:
        parser.error("--points must be at least 2")
    plot_triangle(Path(args.data_dir), Path(args.output), args.points)


if __name__ == "__main__":
    main()
