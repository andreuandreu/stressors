"""Run and compare the current stochastic and analytical hazard-learning models."""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def maximize_practice_uses(
    hazard_response,
    practice_cost: float,
    cost_sensitivity: float,
    max_uses: int = 30,
) -> np.ndarray:
    """Choose utility-maximizing monthly integer use counts per agent."""
    response = np.clip(np.asarray(hazard_response, dtype=float), 0.0, 1.0)
    counts = np.arange(max_uses + 1, dtype=float)

    benefits = 1.0 - np.power(1.0 - response[:, None], counts[None, :])
    utilities = benefits - cost_sensitivity * practice_cost * counts[None, :]

    best_counts = np.argmax(utilities, axis=1)
    best_utilities = utilities[np.arange(response.size), best_counts]
    return np.where(best_utilities > 0.0, best_counts, 0).astype(int)


def load_model(filename: str, module_name: str):
    """Load a model script and provide compatibility for the pasted versions."""
    path = PROJECT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model from {path}")

    module = importlib.util.module_from_spec(spec)

    # The analytical script uses this name as an optimizer default argument,
    # but does not currently import it from config_LearnHaz.
    if filename == "hazard-learning_analytical.py":
        module.MAX_PRACTICE_USES_PER_MONTH = 30

    # Dataclasses need their module registered while the script is executing.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # The dynamics script calls this helper but does not currently define it.
    if filename == "hazard-learning_dynamics.py":
        module.maximize_practice_uses = maximize_practice_uses

    return module


def _mean_uses_from_agent_histories(results: dict) -> np.ndarray:
    """Calculate mean uses per month from per-agent histories."""
    histories = [
        state["mean_practice_uses_history"]
        for state in results.get("agent_states", [])
        if "mean_practice_uses_history" in state
    ]
    if not histories:
        raise KeyError(
            "No mean_practice_uses_history or per-agent "
            "practice_uses_history found in model results"
        )
    return np.mean(np.asarray(histories, dtype=float), axis=0)


def simulation_metrics(results: dict) -> dict[str, np.ndarray]:
    """Extract metrics from the stochastic per-agent simulation results."""
    states = results["agent_states"]
    knowledge_histories = [state["knowledge_history"] for state in states]
    mean_uses = results.get("mean_practice_uses_history")
    if mean_uses is None:
        mean_uses = _mean_uses_from_agent_histories(results)

    return {
        "time": np.asarray(results["time_history"], dtype=float),
        "events": np.asarray(results["total_events_history"], dtype=float),
        "memory": np.asarray(results["mean_memory_history"], dtype=float),
        "knowledge": np.mean(np.asarray(knowledge_histories, dtype=float), axis=0),
        "use": np.asarray(results["mean_practice_use_fraction_history"], dtype=float),
        "mean_uses": np.asarray(mean_uses, dtype=float),
    }


def analytical_metrics(results: dict) -> dict[str, np.ndarray]:
    """Extract aggregate metrics from the analytical results."""
    mean_uses = results.get("mean_practice_uses_history")
    if mean_uses is None:
        raise KeyError(
            "Analytical results do not contain mean_practice_uses_history. "
            "Add that history to the analytical model's returned results."
        )

    return {
        "time": np.asarray(results["time_history"], dtype=float),
        "events": np.asarray(results["expected_events_history"], dtype=float),
        "memory": np.asarray(results["mean_memory_history"], dtype=float),
        "knowledge": np.asarray(results["mean_knowledge_history"], dtype=float),
        "use": np.asarray(results["mean_practice_use_fraction_history"], dtype=float),
        "mean_uses": np.asarray(mean_uses, dtype=float),
    }


def plot_comparison(simulation: dict, analytical: dict):
    labels = {
        "events": "Total hazard events",
        "memory": "Mean severity memory",
        "knowledge": "Mean practice knowledge",
        "use": "Practice-use fraction",
        "mean_uses": "Mean practice uses per agent per month",
    }

    figure, axes = plt.subplots(3, 2, figsize=(9, 12), sharex=True)

    for axis, metric in zip(axes.flat, labels):
        axis.plot(
            simulation["time"],
            simulation[metric],
            color="#1769aa",
            linewidth=1.5,
            alpha=0.85,
            label="Dynamics simulation",
        )
        axis.plot(
            analytical["time"],
            analytical[metric],
            color="#d95f02",
            linewidth=2.0,
            label="Analytical expected value",
        )
        axis.set_ylabel(labels[metric])
        axis.grid(False)

        if metric == "use" :#or metric == "mean_uses"
            axis.set_yscale("log")

    # There are five metrics, so hide the unused sixth subplot.
    axes.flat[-1].set_visible(False)
    axes[0, 0].legend(frameon=False)

    for axis in axes[1, :]:
        if axis.get_visible():
            axis.set_xlabel("Time (months)")
        

    figure.suptitle("Hazard-learning: dynamics versus analytical model")
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--months", type=int, default=None)
    parser.add_argument(
        "--network",
        choices=["small_world", "erdos_renyi", "complete"],
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "plots" / "hazard_learning_comparison.png",
    )
    args = parser.parse_args()

    from config_LearnHaz import DURATION_MONTHS, N_AGENTS, NETWORK_TYPE, RANDOM_SEED

    agents = args.agents if args.agents is not None else N_AGENTS
    months = args.months if args.months is not None else DURATION_MONTHS
    network = args.network if args.network is not None else NETWORK_TYPE
    seed = args.seed if args.seed is not None else RANDOM_SEED

    dynamics_module = load_model(
        "hazard-learning_dynamics.py",
        "hazard_learning_dynamics_for_comparison",
    )
    analytical_module = load_model(
        "hazard-learning_analytical.py",
        "hazard_learning_analytical_for_comparison",
    )

    dynamics_results = dynamics_module.HazardLearningDynamics(
        n_agents=agents,
        duration_months=months,
        network_type=network,
        seed=seed,
    ).run_simulation()

    analytical_results = analytical_module.HazardLearningAnalytical(
        n_agents=agents,
        duration_months=months,
        network_type=network,
        seed=seed,
    ).run_simulation()

    figure = plot_comparison(
        simulation_metrics(dynamics_results),
        analytical_metrics(analytical_results),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved comparison plot to {args.output}")


if __name__ == "__main__":
    main()