"""Compare analytical and dynamics sweeps in a 2x5 grid of paired plots."""

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "1parameter_sweep"
DEFAULT_OUTPUT = PROJECT_DIR / "plots" / "parameter_sweep_pairs_3x2.png"
THRESHOLD = 0.95

MODEL_INFO = {
    "HL_analy": ("Analytical", "tab:blue"),
    "HL_dynam": ("Dynamics", "tab:orange"),
}

PARAMETER_LABELS = {
    "PRACTICE_COST": r"$C_p$",
    "PRACTICE_DECAY": r"$d_p$",
    "SEVERITY_DECAY": r"$d_s$",
    "HAZARD_RATE": r"$h_r$",
    "LEARNING_TIMES": r"$L_t$",
}


def model_info(path: Path) -> tuple[str, str, str]:
    for prefix, (label, color) in MODEL_INFO.items():
        if path.name.startswith(prefix):
            return prefix, label, color
    raise ValueError(f"Unrecognized model filename: {path.name}")


def pair_key(path: Path) -> str:
    """Normalize model-specific filename parts to identify matching files."""
    stem = path.stem
    for prefix in MODEL_INFO:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break

    # The dynamics filename may include this dynamics-only component.
    return re.sub(r"_Nneigh[^_]+", "", stem)


def get_knowledge_history(
    results: dict, path: Path, index: int
) -> list[float]:
    for key in (
        "mean_knowledge_history",
    ):
        history = results.get(key)
        if history is not None:
            return [float(value) for value in history]

    states = results.get("agent_states")
    if states:
        histories = [
            state.get("knowledge_history")
            for state in states
            if isinstance(state, dict) and state.get("knowledge_history") is not None
        ]
        if histories:
            length = min(len(history) for history in histories)
            return np.mean(
                np.asarray([history[:length] for history in histories], dtype=float),
                axis=0,
            ).tolist()

    raise ValueError(f"Entry {index} in {path} has no knowledge history")


def load_sweep(
    path: Path,
) -> tuple[str, list[tuple[float, float, float]]]:
    """Return parameter and sorted (value, final use, threshold time) records."""
    with path.open("rb") as result_file:
        dataset = pickle.load(result_file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError(f"{path} is not a non-empty sweep dataset")

    records = []
    parameter_names = set()

    for index, entry in enumerate(dataset):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {index} in {path} is not a dictionary")

        parameter = entry.get("parameter")
        value = entry.get("value")
        results = entry.get("results")
        if parameter is None or value is None or not isinstance(results, dict):
            raise ValueError(f"Entry {index} in {path} is missing sweep data")

        use_history = results.get("practice_use_fraction_history")
        time_history = results.get("time_history")
        if not use_history or not time_history:
            raise ValueError(f"Entry {index} in {path} has empty use/time history")

        knowledge = get_knowledge_history(results, path, index)
        length = min(len(time_history), len(knowledge))
        crossing_time = next(
            (
                float(time_history[i])
                for i in range(length)
                if knowledge[i] > THRESHOLD
            ),
            np.nan,
        )

        parameter_names.add(str(parameter))
        records.append(
            (float(value), float(use_history[-1]), crossing_time)
        )

    if len(parameter_names) != 1:
        raise ValueError(
            f"{path} contains multiple swept parameters: {sorted(parameter_names)}"
        )

    records.sort(key=lambda record: record[0])
    return parameter_names.pop(), records


def find_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """Find analytical/dynamics files with matching sweep filename components."""
    grouped: dict[str, dict[str, Path]] = {}

    for path in data_dir.glob("HL_*.pkl"):
        prefix = next(
            (name for name in MODEL_INFO if path.name.startswith(name)),
            None,
        )
        if prefix is None:
            continue

        models = grouped.setdefault(pair_key(path), {})
        if prefix in models:
            raise ValueError(f"Multiple {prefix} files match {path.name}")
        models[prefix] = path

    pairs = []
    for key, models in sorted(grouped.items()):
        if "HL_analy" in models and "HL_dynam" in models:
            pairs.append((models["HL_analy"], models["HL_dynam"]))
        else:
            missing = "HL_analy" if "HL_analy" not in models else "HL_dynam"
            print(f"Skipping unmatched sweep {key!r}; missing {missing}")

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pairs = find_pairs(args.data_dir)
    if not pairs:
        raise SystemExit(f"No matching analytical/dynamics pairs found in {args.data_dir}")
    if len(pairs) > 6:
        raise SystemExit(
            f"Found {len(pairs)} pairs; the 3x2 layout supports at most 6"
        )

    figure = plt.figure(figsize=(10, 15))
    grid = figure.add_gridspec(
        6, 2,
        left=0.09, right=0.98, bottom=0.0, top=1.0,
        hspace=0.25, wspace=0.0,
    )

    top_axes = []
    bottom_axes = []

    for index, (analytical_path, dynamics_path) in enumerate(pairs):
        pair_row, column = divmod(index, 2)
        top_row = pair_row * 2
        bottom_row = top_row + 1

        pair_grid = grid[pair_row, column].subgridspec(2, 1, hspace=0)

        top_axis = figure.add_subplot(pair_grid[0])
        bottom_axis = figure.add_subplot(pair_grid[1], sharex=top_axis)
  
        if top_axes:
            top_axis.sharey(top_axes[0])

        top_axes.append(top_axis)
        bottom_axes.append(bottom_axis)

        pair_parameter = None
        for path in (analytical_path, dynamics_path):
            _, model, color = model_info(path)
            parameter, records = load_sweep(path)

            if pair_parameter is not None and parameter != pair_parameter:
                raise ValueError(
                    f"Pair has inconsistent swept parameters: "
                    f"{pair_parameter} and {parameter}"
                )
            pair_parameter = parameter

            values = [record[0] for record in records]
            final_use = [record[1] for record in records]
            crossing_times = [record[2] for record in records]

            top_axis.plot(
                values, final_use, marker="o", color=color, label=model + " " + PARAMETER_LABELS.get(parameter, parameter)
            )
            bottom_axis.plot(
                values, crossing_times, marker="o", color=color, label=model
            )

        symbol = PARAMETER_LABELS.get(pair_parameter, pair_parameter)
        #top_axis.set_title(f"{pair_parameter} ({symbol})", fontsize="medium")
        top_axis.set_ylim(0.0, 1.0)
        bottom_axis.set_xlabel(f"{symbol} ({pair_parameter})")

        # Show each shared y-axis label only in the first column of its row.
        if column == 0:
            top_axis.set_ylabel(r"Final $U_f$")
            bottom_axis.set_ylabel(
                r"Time to $K_f$ > 0.95"
            )

        if column == 1:
            top_axis.tick_params(axis="y", labelleft=False)
            bottom_axis.tick_params(axis="y", labelleft=False)
        
        top_axis.tick_params(axis="x", labelbottom=False)
        top_axis.grid(False)
        bottom_axis.grid(False)
        top_axis.legend(frameon=False, fontsize="small", loc="best")

    # Hide unused pair cells while retaining the 3x2 layout.
    for index in range(len(pairs), 6):
        pair_row, column = divmod(index, 2)
        figure.add_subplot(grid[pair_row * 2, column]).set_visible(False)
        figure.add_subplot(grid[pair_row * 2 + 1, column]).set_visible(False)

    #figure.suptitle("Analytical and dynamics parameter-sweep comparisons")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved 3x2 parameter-sweep figure to {args.output}")


if __name__ == "__main__":
    main()