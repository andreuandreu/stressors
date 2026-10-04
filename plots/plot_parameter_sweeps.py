"""Plot final practice-use fraction against one parameter at a time."""

import argparse
import re
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Filename tags expected in the grid result files.
PARAMETER_TAGS = {
    "HAZARD_RATE": "HAZARD_RATE",
    "SEVERITY_DECAY": "SEVERITY_DECAY",
    "PRACTICE_COST": "PRACTICE_COST",
    "PRACTICE_DECAY": "PRACTICE_DECAY",
    "N_AGENTS": "N_AGENTS",
    "NETWORK_TYPE": "NETWORK_TYPE",
    "LEARNING_TIMES": "LEARNING_TIMES",
    "NETWORK_NEIGHBORS": "NETWORK_NEIGHBORS",
}

TAG_PATTERN = "|".join(
    re.escape(tag) for tag in sorted(PARAMETER_TAGS, key=len, reverse=True)
)
PARAMETER_PATTERN = re.compile(
    rf"(?:^|_)({TAG_PATTERN})(.*?)(?=_(?:{TAG_PATTERN})|$)"
)

SWEEP_PARAMETERS = (
    ("HAZARD_RATE", "HAZARD_RATE"),
    ("SEVERITY_DECAY", "SEVERITY_DECAY"),
    ("PRACTICE_COST", "PRACTICE_COST"),
)


def parse_parameters(path: Path) -> dict[str, str]:
    """Extract parameter tags and values from a result filename."""
    return {
        match.group(1): match.group(2)
        for match in PARAMETER_PATTERN.finditer(path.stem)
    }


def load_final_use(path: Path) -> float:
    with path.open("rb") as result_file:
        results = np.load(result_file, allow_pickle=True)

    # np.save(..., dictionary) stores it as a zero-dimensional object array.
    if isinstance(results, np.ndarray) and results.dtype == object:
        results = results.item()

    if not isinstance(results, dict):
        raise ValueError(f"{path} does not contain a saved results dictionary")

    history = results.get("practice_use_fraction_history")
    if history is None or len(history) == 0:
        raise ValueError(
            f"{path} has no non-empty 'practice_use_fraction_history'"
        )

    return float(np.asarray(history, dtype=float)[-1])


def group_label(parameters: dict[str, str], varied_tag: str) -> str:
    """Label a sweep by the parameters held constant."""
    fixed = [
        f"{PARAMETER_TAGS[tag]}={value}"
        for tag, value in sorted(parameters.items())
        if tag != varied_tag
    ]
    return ", ".join(fixed) if fixed else "other parameters"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Grid-result .npy files (shell wildcards are supported).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./plots/hazard_learning_grid_sweeps.png"),
    )
    args = parser.parse_args()

    records = []
    for path in args.files:
        parameters = parse_parameters(path)
        for tag, parameter_name in SWEEP_PARAMETERS:
            print(tag, "aaaaaaa", parameters)
            if tag not in parameters:
                raise ValueError(
                    f"Could not find the filename tag '{tag}' "
                    f"({parameter_name}) in {path.name}"
                )
        records.append((path, parameters, load_final_use(path)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for axis, (varied_tag, parameter_name) in zip(axes, SWEEP_PARAMETERS):
        groups = defaultdict(list)

        for path, parameters, final_use in records:
            fixed_parameters = tuple(
                sorted(
                    (tag, value)
                    for tag, value in parameters.items()
                    if tag != varied_tag
                )
            )
            groups[fixed_parameters].append(
                (float(parameters[varied_tag]), final_use, path.name)
            )

        plotted = False
        for fixed_parameters, points in groups.items():
            # Only draw actual sweeps: at least two distinct x values.
            if len({x for x, _, _ in points}) < 2:
                continue

            points.sort(key=lambda point: point[0])
            x_values = [point[0] for point in points]
            final_use_values = [point[1] for point in points]
            label = group_label(dict(fixed_parameters), varied_tag)

            axis.plot(
                x_values,
                final_use_values,
                marker="o",
                linewidth=1.5,
                label=label,
            )
            plotted = True

        axis.set_title(parameter_name.replace("_", " ").title())
        axis.set_xlabel(parameter_name)
        axis.grid(True, alpha=0.3)

        if plotted:
            axis.legend(title="Fixed parameters", fontsize="small")
        else:
            axis.text(
                0.5,
                0.5,
                "No sweep found\n(two or more values with other parameters fixed)",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    axes[0].set_ylabel(r"Final use fraction, $U_f$")
    fig.suptitle("Final practice use across hazard-learning parameter sweeps")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()