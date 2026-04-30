#!/usr/bin/env python3
import argparse
import copy
import pickle
from pathlib import Path

import numpy as np

import generate_cells  # Required for pickle loading of Cell instances
import stressor_dynamics as sd_module
from stressor_dynamics import StressorDynamics
from config import DURATION_YEARS, RADIUS


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
        return CellUnpickler(f).load()


def compute_last_three_year_fraction(results):
    ntime = len(results["time_history"])
    if ntime < 3:
        raise ValueError("Simulation must contain at least 3 years to compute the last-3-year average.")

    active_counts = np.array(
        [
            sum(state.active_history[t] for state in results["cell_states"])
            for t in range(ntime)
        ],
        dtype=float,
    )
    last_3 = active_counts[-3:]
    n_cells = len(results["cell_states"])
    return float(np.mean(last_3) / n_cells)


def compute_replicate_stats(
    cells,
    runs: int = 11,
    seed: int = 0,
    mean_e=None,
    std_dev=None,
    media=None,
    prc_saving=None,
):
    values = []

    for i in range(runs):
        run_seed = seed + i
        np.random.seed(run_seed)

        if media is not None:
            sd_module.MEDIA = media
        if prc_saving is not None:
            sd_module.PRC_SAVING = prc_saving

        run_cells = copy.deepcopy(cells)
        for cell in run_cells:
            cell.active = False

        simulation = StressorDynamics(run_cells, duration_years=DURATION_YEARS, radius=RADIUS)
        if mean_e is not None:
            simulation.MEAN_E = mean_e
        if std_dev is not None:
            simulation.STD_DEV = std_dev

        results = simulation.run_simulation()
        active_fraction = compute_last_three_year_fraction(results)
        values.append(active_fraction)

        #print(f"Run {i + 1}/{runs}: seed={run_seed}, avg active last 3 yrs={active_fraction:.4f}")

    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if runs > 1 else 0.0
    return values, mean_val, std_val


def run_replicates(cell_file: Path, output_file: Path, runs: int = 11, seed: int = 0):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cells = load_cells(cell_file)

    values, mean_val, std_val = compute_replicate_stats(cells, runs=runs, seed=seed)
    metadata = {
        "cell_file": str(cell_file),
        "runs": runs,
        "seed": seed,
        "duration_years": DURATION_YEARS,
        "radius": RADIUS,
    }

    with output_file.open("wb") as f:
        pickle.dump({"metadata": metadata, "values": values}, f)

    print(f"\nSaved replicate results to {output_file}")
    print(f"Average active fraction over {runs} runs: {mean_val:.4f}")
    print(f"Standard deviation: {std_val:.4f}")
    return values, mean_val, std_val


def summarize_file(input_file: Path):
    with input_file.open("rb") as f:
        data = pickle.load(f)
    values = np.array(data["values"], dtype=float)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    print(f"Loaded {len(values)} replicate values from {input_file}")
    print(f"Average active fraction: {mean_val:.4f}")
    print(f"Standard deviation: {std_val:.4f}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Run replicate stressor dynamics simulations and summarize active cell statistics.")
    parser.add_argument(
        "--cells",
        default="./data/cells_state.pkl",
        help="Path to the cell state pickle file.",
    )
    parser.add_argument(
        "--output",
        default="./data/replicate_active_fraction_last3yrs.pkl",
        help="Path to the output pickle file for replicate results.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=11,
        help="Number of replicate simulations to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed base for the replicate runs.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Only summarize an existing results file without rerunning simulations.",
    )
    args = parser.parse_args()

    if args.summarize:
        summarize_file(Path(args.output))
    else:
        run_replicates(Path(args.cells), Path(args.output), runs=args.runs, seed=args.seed)


if __name__ == "__main__":
    main()
