"""Deterministic analytical approximation of ``hazard-learning_dynamics.py``.

The analytical model keeps the same social graph and initial known agents as
the stochastic simulation for a given seed. Individual Bernoulli outcomes are
replaced by expected probabilities, and random hazard impacts are replaced by
their expected value. The result is a network mean-field approximation that
can be compared directly with one simulation realization.
"""

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

from config_LearnHaz import (
    COST_SENSITIVITY,
    DURATION_MONTHS,
    HAZARD_INITIAL_FREQUENCY,
    HAZARD_RATE,
    KNOWN_FRACTION,
    LEARNING_TIMES,
    MEMORY_RESPONSE_SCALE,
    NEV_BETA,
    NEV_MU,
    NETWORK_TYPE,
    N_AGENTS,
    PRACTICE_COST,
    PRACTICE_DECAY,
    RANDOM_SEED,
    SEVERITY_DECAY,
)

def _load_network_builder():
    """Reuse the simulation's network construction exactly."""
    script_path = Path(__file__).with_name("hazard-learning_dynamics.py")
    spec = importlib.util.spec_from_file_location("hazard_learning_dynamics", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load network builder from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_social_network

def expected_absolute_gumbel(mu: float, beta: float) -> float:
    """Numerically integrate E[abs(X)] for X distributed as Gumbel(mu, beta).
    The numerical integration is performed with NumPy's trapezoidal rule."""
    if beta <= 0:
        raise ValueError("Gumbel beta must be positive")
    values = np.linspace(mu - 12.0 * beta, mu + 12.0 * beta, 20001)
    standardized = (values - mu) / beta
    density = np.exp(-(standardized + np.exp(-standardized))) / beta
    return float(np.trapezoid(np.abs(values) * density, values))

class HazardLearningAnalytical:
    """Expected-value dynamics on the simulation's fixed social network."""
    def __init__(
        self,
        n_agents: int = N_AGENTS,
        duration_months: int = DURATION_MONTHS,
        network_type: str = NETWORK_TYPE,
        seed: int | None = RANDOM_SEED,
        parameter_overrides: Mapping[str, float] | None = None,
    ):
        self.n_agents = n_agents
        self.duration_months = duration_months
        parameters = parameter_overrides or {}
        self.hazard_initial_frequency = parameters.get(
            "HAZARD_INITIAL_FREQUENCY", HAZARD_INITIAL_FREQUENCY
        )
        self.hazard_rate = parameters.get("HAZARD_RATE", HAZARD_RATE)
        self.severity_decay = parameters.get("SEVERITY_DECAY", SEVERITY_DECAY)
        self.practice_decay = parameters.get("PRACTICE_DECAY", PRACTICE_DECAY)
        self.practice_cost = parameters.get("PRACTICE_COST", PRACTICE_COST)
        self.cost_sensitivity = parameters.get(
            "COST_SENSITIVITY", COST_SENSITIVITY
        )
        self.rng = np.random.default_rng(seed)
        build_social_network = _load_network_builder()
        self.network = build_social_network(
            n_agents=n_agents, network_type=network_type, rng=self.rng
        )

        self.learning_progress = np.zeros(n_agents, dtype=float)
        #initial_known = int(round(KNOWN_FRACTION * n_agents))
        #initial_agents = self.rng.choice(n_agents, size=initial_known, replace=False)
        #self.learning_progress[initial_agents] = LEARNING_TIMES
        self.severity_memory = np.zeros(n_agents, dtype=float)
        self.practice_use_probability = np.zeros(n_agents, dtype=float)
        self.expected_event_severity = expected_absolute_gumbel(NEV_MU, NEV_BETA)

        self.time_history = []
        self.hazard_frequency_history = []
        self.expected_events_history = []
        self.mean_memory_history = []
        self.mean_knowledge_history = []
        self.known_fraction_history = []
        self.practice_use_fraction_history = []

    def hazardFrec(self, time_months: int) -> float:
        """Return the hazard event frequency per agent at time in months."""
        return self.hazard_initial_frequency * np.exp(self.hazard_rate * time_months)

    def _neighbor_exposure_probability(self) -> np.ndarray:
        """Probability that each agent sees at least one practicing neighbor."""
        exposure = np.zeros(self.n_agents, dtype=float)
        for agent, neighbors in self.network.items():
            if neighbors:
                neighbor_probabilities = self.practice_use_probability[list(neighbors)]
                exposure[agent] = 1.0 - np.prod(1.0 - neighbor_probabilities)
        return exposure

    def run_simulation(self) -> dict:
        """Run the deterministic expected-value dynamics."""
        memory_factor = np.exp(-self.severity_decay)
        cost_factor = np.exp(-self.practice_cost / max(self.cost_sensitivity, 1e-12))
        n_always_know = int(round((KNOWN_FRACTION*N_AGENTS)))
        ind_always_know = self.rng.choice(N_AGENTS, size=n_always_know, replace=False)
        
        for time_months in range(self.duration_months + 1):
            frequency = self.hazardFrec(time_months)
            expected_impact = frequency * self.expected_event_severity
            self.severity_memory = memory_factor * self.severity_memory + expected_impact

            exposure_probability = self._neighbor_exposure_probability()
            learned_progress = np.minimum(LEARNING_TIMES, self.learning_progress + 1.0)
            forgotten_progress =  (1-self.practice_decay) * self.learning_progress#forgetting_factor
            self.learning_progress = (
                exposure_probability * learned_progress
                + (1.0 - exposure_probability) * forgotten_progress
            )
        
            knowledge = np.clip(
                self.learning_progress / max(LEARNING_TIMES, 1.0), 0, 1.0
            )
            knowledge[ind_always_know] = 1

            if time_months == 0:
                hazard_response = 1.0 - np.exp(-self.severity_memory / max(MEMORY_RESPONSE_SCALE , 1e-12))
            else:
                hazard_response = 1.0 - np.exp(-self.severity_memory / max(MEMORY_RESPONSE_SCALE * (1-self.practice_use_fraction_history[-1]), 1e-12))
            
            self.practice_use_probability = np.clip(
                (knowledge > (LEARNING_TIMES-1)/LEARNING_TIMES).astype(int) * hazard_response * cost_factor, 0.0, 1.0
            )

            self.time_history.append(time_months)
            self.hazard_frequency_history.append(float(frequency))
            self.expected_events_history.append(float(self.n_agents * frequency))
            self.mean_memory_history.append(float(np.mean(self.severity_memory)))
            self.mean_knowledge_history.append(float(np.mean(knowledge)))
            self.known_fraction_history.append(
                float(np.mean(self.learning_progress >= LEARNING_TIMES))
            )
            self.practice_use_fraction_history.append(
                float(np.mean(self.practice_use_probability))
            )

        return {
            "time_history": self.time_history,
            "hazard_frequency_history": self.hazard_frequency_history,
            "expected_events_history": self.expected_events_history,
            "mean_memory_history": self.mean_memory_history,
            "mean_knowledge_history": self.mean_knowledge_history,
            "known_fraction_history": self.known_fraction_history,
            "practice_use_fraction_history": self.practice_use_fraction_history,
            "expected_event_severity": self.expected_event_severity,
            "network": self.network,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=N_AGENTS)
    parser.add_argument("--months", type=int, default=DURATION_MONTHS)
    parser.add_argument("--network", choices=["small_world", "erdos_renyi", "complete"], default=NETWORK_TYPE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", default="./data/hazard_learning_analytical_results.pkl")
    args = parser.parse_args()

    model = HazardLearningAnalytical(
        n_agents=args.agents,
        duration_months=args.months,
        network_type=args.network,
        seed=args.seed,
    )
    results = model.run_simulation()
    #output_path = Path(args.output)
    
    output_path = (
        #Path(args.output)
        #if args.output
        Path("./data") / (
            f"HL_analy"
            f"_cost{model.practice_cost:g}"
            f"_pdecay{model.practice_decay:g}"
            f"_sdecay{model.severity_decay:g}"
            f"_hrate{model.hazard_rate:g}"
            f"_Ltimes{LEARNING_TIMES}"
            f"_agents{args.agents}"
            f"_network{args.network}.pkl"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("wb") as output_file:
        pickle.dump(results, output_file)
    print(
        f"Saved {len(results['time_history'])} analytical months for "
        f"{args.agents} agents to {output_path}"
    )


if __name__ == "__main__":
    main()