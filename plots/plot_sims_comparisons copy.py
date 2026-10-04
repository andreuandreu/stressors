"""Overlay saved hazard-learning results, labeling runs by differing parameters."""

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PARAMETER_LABELS = {
    "Ltimes": r"$L_t$",
    "pdecay": r"$d_p$",
    "sdecay": r"$d_s$",
    "network": r"$\mathcal{N}$",
    "agents": r"$N$",
    "Nneigh": r"$E$",
    "cost": r"$C_p$",
    "hrate": r"$r_h$",
}

PARAMETER_NAMES = "|".join(
    sorted(PARAMETER_LABELS, key=len, reverse=True)
)
PARAMETER_PATTERN = re.compile(
    rf"(?:^|_)({PARAMETER_NAMES})(.*?)(?=_(?:{PARAMETER_NAMES})|$)"
)


def load_results(path: Path) -> dict:
    with path.open("rb") as result_file:
        results = pickle.load(result_file)
    if not isinstance(results, dict):
        raise ValueError(f"{path} does not contain a results dictionary")
    return results


def parse_filename(path: Path) -> tuple[dict[str, str], str]:
    """Return parameter values and the filename remainder without parameters."""
    parameters = {
        match.group(1): match.group(2)
        for match in PARAMETER_PATTERN.finditer(path.stem)
    }
    remainder = PARAMETER_PATTERN.sub("", path.stem)
    remainder = re.sub(r"_+", " ", remainder).strip()
    return parameters, remainder


def format_parameter(key: str, value: str) -> str:
    """Format a parameter/value pair for a Matplotlib legend."""
    label = PARAMETER_LABELS.get(key, key)

    if key == "network":
        return f"{label} = {value}"

    math_label = label.strip("$")
    return f"${math_label} = {value}$"


def run_label(
    parameters: dict[str, str],
    differing_keys: list[str],
    fallback: str,
) -> str:
    if not differing_keys:
        return fallback
    return ", ".join(
        format_parameter(key, parameters.get(key, "not set"))
        for key in differing_keys
    )


def extract_metrics(results: dict, name_file: str) -> dict:
    """Extract comparable time series from a dynamics or analytical result."""
    time = np.asarray(results["time_history"], dtype=float)

    if 'dyna' in  name_file:
        events_key = "total_events_history"
        if "agent_states" in results:
            states = results["agent_states"]
            knowledge = np.mean(
                np.asarray(
                    [state["knowledge_history"] for state in states],
                    dtype=float,
                ),
                axis=0,
            )
        else:
            knowledge = np.asarray(results["mean_knowledge_history"], dtype=float)
    else:
        events_key = "expected_events_history"
        knowledge = np.asarray(results["mean_knowledge_history"], dtype=float)

    return {
        "time": time,
        "events": np.asarray(results[events_key], dtype=float),
        "memory": np.asarray(results["mean_memory_history"], dtype=float),
        "knowledge": knowledge,
        "use": np.asarray(results["practice_use_fraction_history"], dtype=float),
    }


def plot_comparison(runs: list[tuple[str, dict]], title: str):
    """Overlay each run on the reference script's four-panel plot."""
    metric_labels = {
        "events": "Total hazard events",
        "memory": "Mean severity memory",
        "knowledge": "Mean practice knowledge",
        "use": "Practice use fraction",
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    colors = plt.colormaps["tab10"].colors

    for axis, metric in zip(axes.flat, metric_labels):
        for index, (label, run) in enumerate(runs):
            time = run["time"]
            values = run[metric]
            length = min(len(time), len(values))

            if metric == "memory":
                axis.plot(
                    time[:length],
                    values[:length]/run["events"][:length],
                    color=colors[index % len(colors)],
                    linewidth=1.5,
                    alpha=0.85,
                    label=label,
                )
            else:
                axis.plot(
                    time[:length],
                    values[:length],
                    color=colors[index % len(colors)],
                    linewidth=1.5,
                    alpha=0.85,
                    label=label,
                )

        if axis is axes.flat[0]:
            axis.legend(frameon=False, fontsize="small")

        if metric == "events":
            axis.tick_params(labelbottom=False)
            axis.set_ylabel("Events per month")
            axis.set_yscale("log")
        elif metric == "memory":
            axis.set_ylabel("Severity memory/events")
            axis.yaxis.set_label_position("right")
            axis.yaxis.tick_right()
            axis.tick_params(labelbottom=False)
            axis.set_yscale("log")
        elif metric == "knowledge":
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel("Time (months)")
            axis.set_ylabel("Knowledge fraction")
        elif metric == "use":
            axis.set_ylabel("Use fraction")
            axis.yaxis.set_label_position("right")
            axis.yaxis.tick_right()
            axis.set_yscale("log")
            axis.set_xlabel("Time (months)")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dynamics",
        nargs="*",
        type=Path,
        default=[],
        help="Saved dynamics result files to plot.",
    )
    parser.add_argument(
        "--analytical",
        nargs="*",
        type=Path,
        default=[],
        help="Saved analytical result files to plot.",
    )

    parser.add_argument(
        "--both", 
        nargs="*", 
        type=Path, 
        default=[],
        help="Saved analytical result files to plot.",
    )

    parser.add_argument(
        "--mode",
        choices=("both", "dynamics", "analytical"),
        default="both",
        help="Choose which result type(s) to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./plots/sims_comparison.png"),
    )
    args = parser.parse_args()

    files = []
    if args.mode in ("both", "dynamics"):
        files.extend(("dynamics", path) for path in args.dynamics)
    if args.mode in ("both", "analytical"):
        files.extend(("analytical", path) for path in args.analytical)
    if args.mode in ("both", "both"):
        files.extend(("both", path) for path in args.both)

    if len(files) < 2:
        parser.error("Provide at least two result files for the selected mode")

    parsed = [(kind, path, *parse_filename(path)) for kind, path in files]
    all_parameters = [item[2] for item in parsed]
    all_keys = {key for parameters in all_parameters for key in parameters}
    differing_keys = sorted(
        key
        for key in all_keys
        if len({parameters.get(key) for parameters in all_parameters}) > 1
    )

    remainders = [item[3] for item in parsed]
    shared_remainder = remainders[0] if len(set(remainders)) == 1 else ""

    shared_parameters = []
    for key in sorted(all_keys):
        values = [parameters.get(key) for parameters in all_parameters]
        if values[0] is not None and len(set(values)) == 1 and key not in differing_keys:
            shared_parameters.append(format_parameter(key, values[0]))

    title_parts = [part for part in (shared_remainder, *shared_parameters) if part]
    title = " — ".join(title_parts) if title_parts else "Hazard-learning simulations"

    runs = []
    for kind, path, parameters, _remainder in parsed:
        results = load_results(path)
        label = run_label(parameters, differing_keys, path.stem)
        runs.append((label, extract_metrics(results, path.stem)))

    figure = plot_comparison(runs, title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to {args.output}")


if __name__ == "__main__":
    main()