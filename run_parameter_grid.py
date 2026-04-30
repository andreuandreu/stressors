#!/usr/bin/env python3
import argparse
import copy
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np

import generate_cells  # Required for pickle loading of Cell instances
from run_replicate_stats import compute_replicate_stats
from stressor_dynamics import StressorDynamics
import stressor_dynamics as sd_module
from config import (
    MEAN_E,
    STD_DEV,
    MEDIA,
    PRC_SAVING,
    MEAN_E_RANGE,
    STD_DEV_RANGE,
    MEDIA_RANGE,
    PRC_SAVING_RANGE,
    DURATION_YEARS,
    RADIUS,
)


class CellUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "Cell":
            return generate_cells.Cell
        return super().find_class(module, name)


def load_cells(cell_file: Path):
    if not cell_file.exists():
        raise FileNotFoundError(
            f"Cell file not found: {cell_file}. Run generate_cells.py first."
        )
    with cell_file.open("rb") as f:
        cells = CellUnpickler(f).load()
    return cells


def get_pair_ranges():
    return {
        "MEAN_E": np.array(MEAN_E_RANGE, dtype=float),
        "STD_DEV": np.array(STD_DEV_RANGE, dtype=float),
        "MEDIA": np.array(MEDIA_RANGE, dtype=float),
        "PRC_SAVING": np.array(PRC_SAVING_RANGE, dtype=float),
    }


def get_fiducial_values():
    return {
        "MEAN_E": MEAN_E,
        "STD_DEV": STD_DEV,
        "MEDIA": MEDIA,
        "PRC_SAVING": PRC_SAVING,
    }


def build_filename(param_x: str, param_y: str, fixed: dict) -> str:
    fixed_parts = [f"{key.lower()}{fixed[key]}" for key in sorted(fixed)]
    fixed_label = "_".join(fixed_parts)
    return f"active_fraction_{param_x.lower()}_vs_{param_y.lower()}_{fixed_label}.pkl"


def run_grid(cell_file: Path, output_dir: Path, seed: int = 123):
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = load_cells(cell_file)
    param_ranges = get_pair_ranges()
    fiducial = get_fiducial_values()
    variable_names = list(param_ranges.keys())

    for param_x, param_y in combinations(variable_names, 2):
        x_values = param_ranges[param_x]
        y_values = param_ranges[param_y]
        fixed_params = {
            key: fiducial[key]
            for key in variable_names
            if key not in {param_x, param_y}
        }

        avg_matrix = np.zeros((x_values.size, y_values.size), dtype=float)
        std_matrix = np.zeros((x_values.size, y_values.size), dtype=float)
        print(f"Running grid for {param_x} vs {param_y} with fixed {fixed_params}")

        for ix, x_val in enumerate(x_values):
            for iy, y_val in enumerate(y_values):
                params = fiducial.copy()
                params[param_x] = float(x_val)
                params[param_y] = float(y_val)

                values, mean_val, std_val = compute_replicate_stats(
                    cells,
                    runs=11,
                    seed=seed + ix * len(y_values) + iy,
                    mean_e=params["MEAN_E"],
                    std_dev=params["STD_DEV"],
                    media=params["MEDIA"],
                    prc_saving=params["PRC_SAVING"],
                )

                avg_matrix[ix, iy] = mean_val
                std_matrix[ix, iy] = std_val

                print(
                    f"  {param_x}={x_val}, {param_y}={y_val} -> avg={mean_val:.4f}, std={std_val:.4f}"
                )

        output_path = output_dir / build_filename(param_x, param_y, fixed_params)
        with output_path.open("wb") as f:
            pickle.dump(
                {
                    "param_x": param_x,
                    "param_y": param_y,
                    "x_values": x_values.tolist(),
                    "y_values": y_values.tolist(),
                    "fixed_params": fixed_params,
                    "duration_years": DURATION_YEARS,
                    "replicate_runs": 11,
                    "average_active_fraction": avg_matrix,
                    "std_active_fraction": std_matrix,
                },
                f,
            )

        print(f"Saved grid results to {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Run stressor dynamics for parameter grids.")
    parser.add_argument(
        "--cells",
        default="./data/cells_state.pkl",
        help="Path to the cell state pickle file.",
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Directory to save grid result files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed base for reproducible grid runs.",
    )
    args = parser.parse_args()

    run_grid(Path(args.cells), Path(args.output_dir), seed=args.seed)


if __name__ == "__main__":
    main()
