"""Run and plot comparable stochastic and analytical hazard-learning models."""

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

'''
python plot_hazard_learning_comparison.py \
  --agents 250 \
  --months 240 \
  --network small_world \
  --seed 123


'''


def load_module(script_name: str, module_name: str):
    script_path = Path(__file__).with_name(script_name)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simulation_metrics(results: dict) -> dict:
    """Extract mean knowledge from the portable simulation result."""
    states = results["agent_states"]
    mean_knowledge = np.mean(
        np.asarray([state["knowledge_history"] for state in states], dtype=float),
        axis=0,
    )
    return {
        "time": np.asarray(results["time_history"]),
        "events": np.asarray(results["total_events_history"], dtype=float),
        "memory": np.asarray(results["mean_memory_history"], dtype=float),
        "knowledge": mean_knowledge,
        "use": np.asarray(results["practice_use_fraction_history"], dtype=float),
    }


def analytical_metrics(results: dict) -> dict:
    return {
        "time": np.asarray(results["time_history"]),
        "events": np.asarray(results["expected_events_history"], dtype=float),
        "memory": np.asarray(results["mean_memory_history"], dtype=float),
        "knowledge": np.asarray(results["mean_knowledge_history"], dtype=float),
        "use": np.asarray(results["practice_use_fraction_history"], dtype=float),
    }


def plot_comparison(simulation: dict, analytical: dict):
    """Plot equivalent aggregate observables for both model forms."""
    labels = {
        "events": "Total hazard events",
        "memory": "Mean severity memory",
        "knowledge": "Mean practice knowledge",
        "use": "Practice use fraction",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), gridspec_kw = {'wspace':0, 'hspace':0})
    
    for axis, metric in zip(axes.flat, labels):
        axis.plot(
            simulation["time"],
            simulation[metric],
            color="#1769aa",
            linewidth=1.5,
            alpha=0.85,
            label="Agent simulation",
        )
        axis.plot(
            analytical["time"],
            analytical[metric],
            color="#d95f02",
            linewidth=2.0,
            label="Analytical expected value",
        )
        
        if metric in {"events"}:
            axis.legend(frameon=False)
            axis.set_xticklabels([])
            axis.set_ylabel("Events per month")
        if metric in {"memory"}:
            axis.set_ylabel("Severity memory")
            axis.yaxis.set_label_position("right")
            axis.yaxis.tick_right()
            axis.set_xticklabels([])
        if metric in {"knowledge"}:
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel("Time (months)")
            axis.set_ylabel("Knowledge fraction")
        if metric in {"use"}:
            axis.set_ylabel("Use fraction")
            axis.yaxis.set_label_position("right")
            axis.yaxis.tick_right()
            axis.set_yscale("log")
            axis.set_xlabel("Time (months)")
    
    fig.suptitle("Hazard-learning dynamics: simulation versus analytical approximation")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--months", type=int, default=None)
    parser.add_argument("--network", choices=["small_world", "erdos_renyi", "complete"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default="./plots/hazard_learning_comparison.png")
    args = parser.parse_args()

    simulation_module = load_module(
        "hazard-learning_dynamics.py", "hazard_learning_dynamics_for_comparison"
    )
    analytical_module = load_module(
        "hazard-learning_analytical.py", "hazard_learning_analytical_for_comparison"
    )

    from config_LearnHaz import DURATION_MONTHS, N_AGENTS, NETWORK_TYPE, RANDOM_SEED

    agents = args.agents if args.agents is not None else N_AGENTS
    months = args.months if args.months is not None else DURATION_MONTHS
    network = args.network if args.network is not None else NETWORK_TYPE
    seed = args.seed if args.seed is not None else RANDOM_SEED

    simulation = simulation_module.HazardLearningDynamics(
        n_agents=agents, duration_months=months, network_type=network, seed=seed
    ).run_simulation()
    analytical = analytical_module.HazardLearningAnalytical(
        n_agents=agents, duration_months=months, network_type=network, seed=seed
    ).run_simulation()

    figure = plot_comparison(
        simulation_metrics(simulation), analytical_metrics(analytical)
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to {output_path}")


if __name__ == "__main__":
    main()