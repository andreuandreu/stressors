"""Run pairwise parameter grids for the analytical hazard-learning model.

Each output NumPy file contains the final practice-use fraction for one pair
of varied parameters. All other model parameters remain at their fiducial
values from ``config_LearnHaz.py``.
"""

import argparse
import importlib.util
import itertools
from pathlib import Path
from typing import Mapping

import numpy as np
import time
import sys

import config_LearnHaz as config

'''
GRID_PARAMETERS = (
    "PRACTICE_DECAY",
    "SEVERITY_DECAY",
    "HAZARD_RATE",
    "PRACTICE_COST",
)

GRID_PARAMETERS = (
    "PRACTICE_COST",
    "SEVERITY_DECAY",
    "LEARNING_TIMES",
    "HAZARD_RATE",
)

PARAMETER_RANGES = {
    "PRACTICE_COST": (0.001, 0.55),
    "SEVERITY_DECAY": (0.0001, 0.05),
    "LEARNING_TIMES": (2, 9),
    "HAZARD_RATE": (0.005, 0.015),
}

PARAMETER_RANGES = {
    "PRACTICE_DECAY": (0.0001, 0.5),
    "SEVERITY_DECAY": (0.0001, 0.05),
    "HAZARD_RATE": (0.005, 0.015),
    "PRACTICE_COST": (0.001, 0.55),
}
'''

GRID_PARAMETERS = (
    "PRACTICE_COST",
    "SEVERITY_DECAY",
    "HAZARD_RATE",
)


PARAMETER_RANGES = {
    "PRACTICE_COST": (0.001, 0.55),
    "SEVERITY_DECAY": (0.0001, 0.05),
    "HAZARD_RATE": (0.001, 0.013),
}


def load_analytical_module():
    script_path = Path(__file__).with_name("hazard-learning_analytical.py")
    spec = importlib.util.spec_from_file_location(
        "hazard_learning_analytical_for_grid", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fiducial_values() -> dict[str, float]:
    """Return the configured values for all parameters varied by this script."""
    return {name: float(getattr(config, name)) for name in GRID_PARAMETERS}


def parameter_values(parameter: str, points: int) -> np.ndarray:
    minimum, maximum = PARAMETER_RANGES[parameter]

    if parameter == "LEARNING_TIMES":
        return np.linspace(minimum, maximum, points, dtype=int)
    
    if parameter == "SEVERITY_DECAY":
        return np.logspace(np.log10(minimum), np.log10(maximum), num=points, dtype=float)
    
    return np.linspace(minimum, maximum, points, dtype=float)


def format_value(value: float) -> str:
    return f"{value:.6g}"


def build_filename(
    parameter_x: str,
    parameter_y: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    fixed: Mapping[str, float],
) -> str:
    varied = (
        f"{parameter_x}-{format_value(x_values[0])}-{format_value(x_values[-1])}"
        f"_n{x_values.size}"
        f"_{parameter_y}-{format_value(y_values[0])}-{format_value(y_values[-1])}"
        f"_n{y_values.size}"
    )
    fixed_label = "_".join(
        f"{name}-{format_value(fixed[name])}"
        for name in GRID_PARAMETERS
        if name not in {parameter_x, parameter_y}
    )
    context = (
        f"agents-{config.N_AGENTS}_months-{config.DURATION_MONTHS}"
        f"_network-{config.NETWORK_TYPE}"
    )
    return f"practice_use_{varied}_fixed-{fixed_label}_{context}.npy"


def run_pair_grid(
    analytical_module,
    parameter_x: str,
    parameter_y: str,
    points: int,
    output_dir: Path,
    seed: int,
) -> Path:
    x_values = parameter_values(parameter_x, points)
    y_values = parameter_values(parameter_y, points)
    fixed = fiducial_values()
    matrix = np.zeros((x_values.size, y_values.size), dtype=float)

    start_time = time.time()
    for ix, x_value in enumerate(x_values):
        for iy, y_value in enumerate(y_values):
            overrides = {
                name: value
                for name, value in fixed.items()
                if name not in {parameter_x, parameter_y}
            }
            overrides[parameter_x] = float(x_value)
            overrides[parameter_y] = float(y_value)
            model = analytical_module.HazardLearningAnalytical(
                n_agents=config.N_AGENTS,
                duration_months=config.DURATION_MONTHS,
                network_type=config.NETWORK_TYPE,
                seed=seed,
                parameter_overrides=overrides,
            )
            results = model.run_simulation()
            matrix[ix, iy] = results["practice_use_fraction_history"][-1]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / build_filename(
        parameter_x, parameter_y, x_values, y_values, fixed
    )
    payload = {
        "parameter_x": parameter_x,
        "parameter_y": parameter_y,
        "x_values": x_values,
        "y_values": y_values,
        "practice_use_fraction": matrix,
        "fixed_parameters": {
            name: value
            for name, value in fixed.items()
            if name not in {parameter_x, parameter_y}
        },
        "fiducial_parameters": fixed,
        "n_agents": config.N_AGENTS,
        "duration_months": config.DURATION_MONTHS,
        "network_type": config.NETWORK_TYPE,
        "seed": seed,
    }
    np.save(output_path, payload, allow_pickle=True)
    
    print(f"time to run: {time.time() - start_time:.2f} seconds")
    print(f"Saved {parameter_x} vs {parameter_y} to {output_path} \n")
    print(x_values)
    print(y_values)
    print(payload['practice_use_fraction'], '\n')
    
    return output_path


def run_all_grids(points: int, output_dir: Path, seed: int) -> list[Path]:
    analytical_module = load_analytical_module()
    paths = []

    for parameter_x, parameter_y in itertools.combinations(GRID_PARAMETERS, 2):
        print(f"Running grid for {parameter_x} vs {parameter_y}")
        paths.append(
            run_pair_grid(
                analytical_module,
                parameter_x,
                parameter_y,
                points,
                output_dir,
                seed,
            )
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=15)
    parser.add_argument("--output-dir", default="./data/hazard_learning_grids")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()
    if args.points < 2:
        parser.error("--points must be at least 2")
    run_all_grids(args.points, Path(args.output_dir), args.seed)


if __name__ == "__main__":
    main()
