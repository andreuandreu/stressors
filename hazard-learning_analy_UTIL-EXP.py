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
    MAX_PRACTICE_USES_PER_MONTH,
    MEMORY_RESPONSE_SCALE,
    NEV_BETA,
    NEV_MU,
    NETWORK_TYPE,
    NETWORK_NEIGHBORS,
    N_AGENTS,
    PRACTICE_COST,
    PRACTICE_DECAY,
    RANDOM_SEED,
    SEVERITY_DECAY,
    TAG,
    USE_COUNT_VARIATION,
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

def practice_use_probabilities(
    hazard_response,
    practice_cost: float,
    cost_sensitivity: float,
    max_uses: int,
    use_count_variation: float,
) -> np.ndarray:
    """Return soft-choice probabilities for use counts 0..max_uses."""
    if use_count_variation <= 0:
        raise ValueError("use_count_variation must be positive")

    response = np.clip(np.asarray(hazard_response, dtype=float), 0.0, 1.0)
    counts = np.arange(max_uses + 1, dtype=float)

    benefit = 1.0 - np.power(1.0 - response[:, None], counts[None, :])
    utility = benefit - cost_sensitivity * practice_cost * counts[None, :]

    # Numerically stable softmax, independently for each agent.
    scaled = utility / use_count_variation
    scaled -= np.max(scaled, axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)

    # Keep the requested zero-memory behavior: no perceived benefit means no use.
    zero_response = response <= 0.0
    probabilities[zero_response] = 0.0
    probabilities[zero_response, 0] = 1.0
    return probabilities


# Replace this helper in both hazard-learning_dynamics.py and hazard-learning_analytical.py.
def marginal_use_probabilities(
    hazard_response,
    practice_cost: float,
    cost_sensitivity: float,
    max_uses: int,
    practice_knowledge: float,
) -> np.ndarray:
    """Conditional probabilities for successive uses with increasing marginal cost."""

    #print("knowledge fraction per agent must be positive and non-zero", practice_knowledge)
    #if practice_knowledge <= 0:
    #    practice_knowledge = KNOWN_FRACTION

    response = np.clip(np.asarray(hazard_response, dtype=float), 0.0, 1.0)

    # Zero-based exponent gives the first use benefit h, second h*(1-h), etc.
    use_exponent = np.arange(max_uses, dtype=float)
    marginal_benefit = response[:, None] * np.power(
        1.0 - response[:, None], use_exponent[None, :]
    )

    # One-based cost index: the kth use has marginal cost k * base_cost.
    use_number = np.arange(1, max_uses + 1, dtype=float)
    marginal_cost = (
        cost_sensitivity * practice_cost * use_number[None, :]
    )

    baseline_x = np.clip(-marginal_cost / max(practice_knowledge, KNOWN_FRACTION), -700.0, 700.0)
    choice_x = np.clip(
        (marginal_benefit - marginal_cost) / max(practice_knowledge, KNOWN_FRACTION),
        -700.0,
        700.0,
    )

    baseline_probability = 1.0 / (1.0 + np.exp(-baseline_x))
    choice_probability = 1.0 / (1.0 + np.exp(-choice_x))

    probabilities = (
        choice_probability - baseline_probability
    ) / np.maximum(1.0 - baseline_probability, 1e-12)

    # No benefit means zero chance of that use; positive benefit stays smooth.
    probabilities = np.where(
        marginal_benefit > 0.0,
        np.clip(probabilities, 0.0, 1.0),
        0.0,
    )
    return np.clip((practice_knowledge * (marginal_benefit - marginal_cost)), 0.0, 1.0)

class HazardLearningAnalytical:
    """Expected-value dynamics on the simulation's fixed social network."""
    def __init__(
        self,
        n_agents=N_AGENTS,
        duration_months=DURATION_MONTHS,
        network_type=NETWORK_TYPE,
        seed=RANDOM_SEED,
        parameter_overrides=None,
    ):
        self.n_agents = n_agents
        self.duration_months = duration_months
        parameters = parameter_overrides or {}
        self.hazard_initial_frequency = parameters.get(
            "HAZARD_INITIAL_FREQUENCY", HAZARD_INITIAL_FREQUENCY
        )
        self.learning_times = parameters.get("LEARNING_TIMES", LEARNING_TIMES)
        self.hazard_rate = parameters.get("HAZARD_RATE", HAZARD_RATE)
        self.severity_decay = parameters.get("SEVERITY_DECAY", SEVERITY_DECAY)
        self.practice_decay = parameters.get("PRACTICE_DECAY", PRACTICE_DECAY)
        self.practice_cost = parameters.get("PRACTICE_COST", PRACTICE_COST)
        self.cost_sensitivity = parameters.get("COST_SENSITIVITY", COST_SENSITIVITY)
        self.rng = np.random.default_rng(seed)
        build_social_network = _load_network_builder()
        self.network = build_social_network(
            n_agents=n_agents, network_type=network_type, rng=self.rng
        )

        self.use_count_variation = parameters.get(
            "USE_COUNT_VARIATION", USE_COUNT_VARIATION
        )
        self.practice_uses = np.zeros(n_agents, dtype=float)
        self.practice_use_probability = np.zeros(n_agents, dtype=float)
        self.mean_practice_uses_history = []

        self.learning_progress = np.zeros(n_agents, dtype=float)
        #initial_known = int(round(KNOWN_FRACTION * n_agents))
        #initial_agents = self.rng.choice(n_agents, size=initial_known, replace=False)
        #self.learning_progress[initial_agents] = LEARNING_TIMES
        self.severity_memory = np.zeros(n_agents, dtype=float)
        self.expected_event_severity = expected_absolute_gumbel(NEV_MU, NEV_BETA)

        self.time_history = []
        self.hazard_frequency_history = []
        self.expected_events_history = []
        self.mean_memory_history = []
        self.mean_knowledge_history = []
        self.known_fraction_history = []
        self.practice_use_fraction_history = []
        self.practice_uses = np.zeros(n_agents, dtype=float)

        self.reminder_rng = np.random.default_rng(seed)

    def hazardFrec(self, time_months: int) -> float:
        """Return the hazard event frequency per agent at time in months."""
        return self.hazard_initial_frequency * np.exp(self.hazard_rate * time_months)

    def _neighbor_expected_uses(self) -> np.ndarray:
        """Expected total practice uses received from neighbors."""
        return np.fromiter(
            (
                np.sum(self.practice_uses[list(neighbors)])
                if neighbors
                else 0.0
                for neighbors in self.network.values()
            ),
            dtype=float,
            count=self.n_agents,
        )

    def _top_up_known_fraction(self) -> None:
        """Randomly restore the minimum fully knowledgeable fraction."""
        target_known = min(
            self.n_agents,
            int(np.ceil(KNOWN_FRACTION * self.n_agents)),
        )
        known = self.learning_progress >= self.learning_times
        deficit = target_known - int(np.count_nonzero(known))

        if deficit > 0:
            candidates = np.flatnonzero(~known)
            selected = self.reminder_rng.choice(
                candidates, size=deficit, replace=False
            )
            self.learning_progress[selected] = self.learning_times

    def run_simulation(self) -> dict:
        """Run the deterministic expected-value dynamics."""
        memory_factor = np.exp(-self.severity_decay)
        #cost_factor = np.exp(-self.practice_cost / max(self.cost_sensitivity, 1e-12))

        #n_always_know = int(round(KNOWN_FRACTION * self.n_agents))
        #ind_always_know = self.rng.choice(
        #    self.n_agents, size=n_always_know, replace=False
        #)

        for time_months in range(self.duration_months + 1):
            frequency = self.hazardFrec(time_months)
            expected_impact = frequency * self.expected_event_severity
            self.severity_memory = memory_factor * self.severity_memory + expected_impact
        
            neighbor_uses = self._neighbor_expected_uses()
            previous_any_use = np.clip(
                self.practice_use_probability, 0.0, 1.0
            )

            # Approximate independent agent decisions: probability that neither
            # this agent nor any neighbor used the practice last month.
            no_activity_probability = np.empty(self.n_agents, dtype=float)
            for agent, neighbors in self.network.items():
                neighbor_ids = list(neighbors)
                no_neighbor_use = (
                    float(np.prod(1.0 - previous_any_use[neighbor_ids]))
                    if neighbor_ids
                    else 1.0
                )
                no_activity_probability[agent] = (
                    (1.0 - previous_any_use[agent]) * no_neighbor_use
                )

            forgetting_factor = np.exp(-self.practice_decay)
            expected_retention = 1.0 - (
                1.0 - forgetting_factor
            ) * no_activity_probability

            self.learning_progress = np.minimum(
                self.learning_times,
                expected_retention * self.learning_progress + neighbor_uses,
            )

            knowledge = np.clip(
                self.learning_progress / max(self.learning_times, 1.0),
                0.0,
                1.0,
            )

            hazard_response = 1.0 - np.exp(
                -self.severity_memory / max(MEMORY_RESPONSE_SCALE, 1e-12)
            )

            #print('kaKOKOKOKO', knowledge, len(knowledge))
            conditional_use_probabilities = marginal_use_probabilities(
                hazard_response,
                self.practice_cost,
                self.cost_sensitivity,
                MAX_PRACTICE_USES_PER_MONTH,
                np.mean(knowledge),
            )

            #knows_practice = self.learning_progress >= self.learning_times
            #conditional_use_probabilities[~knows_practice, :] = 0.0

            # P(agent reaches use k) is the product of accepting uses 1..k.
            reach_probabilities = np.cumprod(
                conditional_use_probabilities, axis=1
            )
            self.practice_uses = np.sum(reach_probabilities, axis=1)
            self.practice_use_probability = conditional_use_probabilities[:, 0]

            self.practice_use_fraction_history.append(
                float(np.mean(self.practice_use_probability))
            )
            self.mean_practice_uses_history.append(
                float(np.mean(self.practice_uses))
            )
            

            self.time_history.append(time_months)
            self.hazard_frequency_history.append(float(frequency))
            self.expected_events_history.append(float(self.n_agents * frequency))
            self.mean_memory_history.append(float(np.mean(self.severity_memory)))
            self.mean_knowledge_history.append(float(np.mean(knowledge)))
            self.known_fraction_history.append(
                float(np.mean(knowledge >= 1.0))
            )

            # End-of-month top-up affects eligibility starting next month.
            self._top_up_known_fraction()

        return {
            "time_history": self.time_history,
            "hazard_frequency_history": self.hazard_frequency_history,
            "expected_events_history": self.expected_events_history,
            "mean_memory_history": self.mean_memory_history,
            "mean_knowledge_history": self.mean_knowledge_history,
            "known_fraction_history": self.known_fraction_history,
            "practice_use_fraction_history": self.practice_use_fraction_history,
            "expected_event_severity": self.expected_event_severity,
            "mean_practice_uses_history":self.mean_practice_uses_history,
            "network": self.network,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=N_AGENTS)
    parser.add_argument("--months", type=int, default=DURATION_MONTHS)
    parser.add_argument("--network", choices=["small_world", "erdos_renyi", "complete"], default=NETWORK_TYPE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", default="./data/hazard_learning_analytical_results.pkl")
    parser.add_argument("--practice-cost", type=float, default=PRACTICE_COST)
    parser.add_argument("--practice-decay", type=float, default=PRACTICE_DECAY)
    parser.add_argument("--severity-decay", type=float, default=SEVERITY_DECAY)
    parser.add_argument("--hazard-rate", type=float, default=HAZARD_RATE)

    args = parser.parse_args()

    model = HazardLearningAnalytical(
        n_agents=args.agents,
        duration_months=args.months,
        network_type=args.network,
        seed=args.seed,
        parameter_overrides={
            "PRACTICE_COST": args.practice_cost,
            "PRACTICE_DECAY": args.practice_decay,
            "SEVERITY_DECAY": args.severity_decay,
            "HAZARD_RATE": args.hazard_rate,
        },
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
            f"_Nneigh{NETWORK_NEIGHBORS}"
            f"_network{args.network}"
            f"_CoUt{TAG}.pkl"
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