"""Theoretical hazard-learning dynamics for a population of individuals.

Unlike ``stressor_dynamics.py``, this model has no spatial cells. Individuals
are nodes in a configurable social network. Time is measured in months.

For agent i, the main processes are:

    f(t) = f_0 exp(hazardRate * t)
    M_i(t+1) = exp(-severityDecay) M_i(t) + impact_i(t)
    K_i(t+1) = min(1, K_i(t) + 1/learningTimes)  [neighbor uses practice]
    K_i(t+1) = K_i(t) exp(-practiceDecay)         [no neighbor uses practice]

Practice use is probabilistic. It increases with remembered hazard impact and
decreases with practice cost, so social learning, hazard memory, and cost all
modulate the temporal adoption dynamics.
"""

import argparse
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Mapping

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
    NETWORK_EDGE_PROBABILITY,
    NETWORK_NEIGHBORS,
    NETWORK_REWIRE_PROBABILITY,
    NETWORK_TYPE,
    N_AGENTS,
    PRACTICE_COST,
    PRACTICE_DECAY,
    MAX_PRACTICE_USES_PER_MONTH,
    USE_COUNT_VARIATION,
    RANDOM_SEED,
    SEVERITY_DECAY,
    TAG,
)


def build_social_network(
    n_agents: int,
    network_type: str = NETWORK_TYPE,
    neighbors: int = NETWORK_NEIGHBORS,
    rewire_probability: float = NETWORK_REWIRE_PROBABILITY,
    edge_probability: float = NETWORK_EDGE_PROBABILITY,
    rng: np.random.Generator | None = None,
) -> Dict[int, Set[int]]:
    """Build an undirected social network without requiring external packages."""
    if n_agents < 2:
        raise ValueError("n_agents must be at least 2")
    if rng is None:
        rng = np.random.default_rng()

    network = {agent: set() for agent in range(n_agents)}
    if network_type == "complete":
        for agent in range(n_agents):
            network[agent].update(other for other in range(n_agents) if other != agent)
        return network

    if network_type == "erdos_renyi":
        for agent in range(n_agents):
            for other in range(agent + 1, n_agents):
                if rng.random() < edge_probability:
                    network[agent].add(other)
                    network[other].add(agent)
        return network

    if network_type != "small_world":
        raise ValueError(f"Unsupported network_type: {network_type}")
    if neighbors < 2 or neighbors >= n_agents or neighbors % 2:
        raise ValueError("small-world neighbors must be an even number below n_agents")

    half_neighbors = neighbors // 2
    for agent in range(n_agents):
        for offset in range(1, half_neighbors + 1):
            other = (agent + offset) % n_agents
            network[agent].add(other)
            network[other].add(agent)

    # Rewire only the clockwise half of each ring edge, as in Watts-Strogatz.
    for agent in range(n_agents):
        for offset in range(1, half_neighbors + 1):
            other = (agent + offset) % n_agents
            if rng.random() >= rewire_probability:
                continue
            candidates = [
                candidate
                for candidate in range(n_agents)
                if candidate != agent and candidate not in network[agent]
            ]
            if not candidates:
                continue
            network[agent].discard(other)
            network[other].discard(agent)
            replacement = int(rng.choice(candidates))
            network[agent].add(replacement)
            network[replacement].add(agent)

    return network


@dataclass
class AgentState:
    """Dynamic state and histories for one individual agent."""

    agent_id: int
    severity_memory: float = 0.0
    current_impact: float = 0.0
    practice_knowledge: float = 0.0
    learning_progress: float = 0.0
    practice_used: bool = False
    practice_uses: int = 0
    severity_history: List[float] = field(default_factory=list)
    memory_history: List[float] = field(default_factory=list)
    knowledge_history: List[float] = field(default_factory=list)
    practice_history: List[bool] = field(default_factory=list)
    practice_uses_history: List[int] = field(default_factory=list)


class HazardLearningDynamics:
    """Simulate hazard exposure, social learning, forgetting, and practice use."""

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
        self.learning_times = parameters.get("LEARNING_TIMES", LEARNING_TIMES)
        self.hazard_rate = parameters.get("HAZARD_RATE", HAZARD_RATE)
        self.severity_decay = parameters.get("SEVERITY_DECAY", SEVERITY_DECAY)
        self.practice_decay = parameters.get("PRACTICE_DECAY", PRACTICE_DECAY)
        self.practice_cost = parameters.get("PRACTICE_COST", PRACTICE_COST)
        self.cost_sensitivity = parameters.get(
            "COST_SENSITIVITY", COST_SENSITIVITY
        )
        self.max_practice_uses = parameters.get(
            "MAX_PRACTICE_USES_PER_MONTH",
            MAX_PRACTICE_USES_PER_MONTH,
        )
        self.use_count_variation = parameters.get(
            "USE_COUNT_VARIATION", USE_COUNT_VARIATION
        )

        self.rng = np.random.default_rng(seed)
        self.reminder_rng = np.random.default_rng(seed)
        self.network = build_social_network(
            n_agents=n_agents, network_type=network_type, rng=self.rng
        )
        self.agent_states = [AgentState(agent_id=agent) for agent in range(n_agents)]
        #initial_known = int(round(KNOWN_FRACTION * self.n_agents))
        #self.always_know_agents = set(
        #    self.rng.choice(self.n_agents, size=initial_known, replace=False)
        #)
        #for agent in self.always_know_agents:
        #    state = self.agent_states[agent]
        #    state.learning_progress = float(self.learning_times)
        #    state.practice_knowledge = 1.0

        self.time_history: List[int] = []
        self.hazard_frequency_history: List[float] = []
        self.total_events_history: List[int] = []
        self.mean_memory_history: List[float] = []
        self.known_fraction_history: List[float] = []
        self.mean_knowledge_history: List[float] = []
        self.practice_use_fraction_history: List[float] = []
        self.mean_practice_uses_history: List[float] = []

    def hazardFrec(self, time_months: int) -> float:
        """Return the hazard event frequency per agent at time in months."""
        return HAZARD_INITIAL_FREQUENCY * np.exp(self.hazard_rate * time_months)

    def hazard_freq(self, time_months: int) -> float:
        """Readable alias for :meth:`hazardFrec`."""
        return self.hazardFrec(time_months)

    def _update_hazard_memory(self, frequency: float) -> int:
        total_events = 0
        decay = np.exp(-self.severity_decay)
        for state in self.agent_states:
            event_count = int(self.rng.poisson(frequency))
            total_events += event_count
            if event_count:
                severities = np.abs(
                    self.rng.gumbel(NEV_MU, NEV_BETA, size=event_count)
                )
                state.current_impact = float(np.sum(severities))
            else:
                state.current_impact = 0.0
            state.severity_memory = decay * state.severity_memory + state.current_impact
            state.severity_history.append(state.current_impact)
            state.memory_history.append(state.severity_memory)
        return total_events

    def _update_practice_dynamics(self) -> None:
        previous_uses = [state.practice_uses for state in self.agent_states]
        forgetting_factor = np.exp(-self.practice_decay)

        for state in self.agent_states:
            neighbor_uses = sum(
                previous_uses[neighbor]
                for neighbor in self.network[state.agent_id]
            )
            had_activity = neighbor_uses > 0 or state.practice_used
            retention = 1.0 if had_activity else forgetting_factor

            state.learning_progress = min(
                self.learning_times,
                retention * state.learning_progress + neighbor_uses,
            )
            state.practice_knowledge = min(
                1.0,
                state.learning_progress / max(self.learning_times, 1.0),
            )

        hazard_response = 1.0 - np.exp(
            -np.fromiter(
                (state.severity_memory for state in self.agent_states),
                dtype=float,
                count=self.n_agents,
            ) / max(MEMORY_RESPONSE_SCALE, 1e-12)
        )
        conditional_use_probabilities = marginal_use_probabilities(
            hazard_response,
            self.practice_cost,
            self.cost_sensitivity,
            self.max_practice_uses,
            state.practice_knowledge,
        )

        '''
        knows_practice = np.fromiter(
            (
                state.learning_progress >= self.learning_times
                for state in self.agent_states
            ),
            dtype=bool,
            count=self.n_agents,
        )
        conditional_use_probabilities[~knows_practice, :] = 0.0
        '''

        use_counts = np.zeros(self.n_agents, dtype=int)
        active = np.ones(self.n_agents, dtype=bool)

        for use_index in range(self.max_practice_uses):
            accepted = active & (
                self.rng.random(self.n_agents)
                < conditional_use_probabilities[:, use_index]
            )
            use_counts += accepted
            active = accepted  # Stop after the first rejected marginal use.

        for state, uses in zip(self.agent_states, use_counts):
            state.practice_uses = int(uses)
            state.practice_used = state.practice_uses > 0
            state.knowledge_history.append(state.practice_knowledge)
            state.practice_history.append(state.practice_used)
            state.practice_uses_history.append(state.practice_uses)

    def _top_up_known_fraction(self) -> None:
        """At month end, randomly restore the minimum fully-known fraction."""
        target_known = min(
            self.n_agents,
            int(np.ceil(KNOWN_FRACTION * self.n_agents)),
        )
        known = np.fromiter(
            (
                state.learning_progress >= self.learning_times
                for state in self.agent_states
            ),
            dtype=bool,
            count=self.n_agents,
        )
        deficit = target_known - int(np.count_nonzero(known))

        #if deficit > 0:
        candidates = np.flatnonzero(~known)
        selected = self.reminder_rng.choice(
                ~known, size=target_known, replace=True
        )
        selected = self.rng.choice(N_AGENTS, size=target_known, replace=False)
        for agent in selected:
                state = self.agent_states[int(agent)]
                state.learning_progress = float(self.learning_times)
                #state.practice_knowledge = 1.0

    def run_simulation(self) -> dict:
        """Run the model and return aggregate and per-agent histories."""
        #n_always_know = int(round((KNOWN_FRACTION*N_AGENTS)))
        for time_months in range(self.duration_months + 1):
            frequency = self.hazardFrec(time_months)
            total_events = self._update_hazard_memory(frequency)
            self._update_practice_dynamics()
            #self.rng.choice(N_AGENTS, size=n_always_know, replace=False)
            #for agent in self.rng.choice(N_AGENTS, size=n_always_know, replace=False):
                #self.agent_states[int(agent)].learning_progress = LEARNING_TIMES
                #self.agent_states[int(agent)].practice_knowledge 
                #self.knowledge_history[int(agent)][-1] = 1.0#append(state.practice_knowledge)

            self.time_history.append(time_months)
            self.hazard_frequency_history.append(float(frequency))
            self.total_events_history.append(total_events)
            self.mean_memory_history.append(
                float(np.mean([state.severity_memory for state in self.agent_states]))
            )
            
            self.known_fraction_history.append(
                float(np.mean([state.practice_knowledge >= 1.0 for state in self.agent_states]))
            )
            self.practice_use_fraction_history.append(
                float(np.mean([state.practice_used for state in self.agent_states]))
            )
            self.mean_knowledge_history.append(
                float(np.mean([state.practice_knowledge for state in self.agent_states]))
            )
            self.mean_practice_uses_history.append(
                float(np.mean([state.practice_uses for state in self.agent_states]))
            )

            # Month-end reminder: affects knowledge and use eligibility next month.
            self._top_up_known_fraction()

        #print('LEEEENENNE', self.learning_times, '\n', self.mean_knowledge_history)

        return self._results()

    def _results(self) -> dict:
        return {
            "time_history": self.time_history,
            "hazard_frequency_history": self.hazard_frequency_history,
            "total_events_history": self.total_events_history,
            "mean_memory_history": self.mean_memory_history,
            #"known_fraction_history": self.known_fraction_history,
            "mean_knowledge_history": self.mean_knowledge_history,
            "practice_use_fraction_history": self.practice_use_fraction_history,
            "mean_practice_uses_history": self.mean_practice_uses_history,
            "agent_states": [asdict(state) for state in self.agent_states],
            "network": self.network,
        }

    def save_results(self, filename: str = "./data/hazard_learning_dynamics_results.pkl") -> None:
        """Save the simulation results as a pickle file."""
        if not self.time_history:
            self.run_simulation()
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as output_file:
            pickle.dump(self._results(), output_file)


#
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

    #print('dadadada', len(marginal_cost), len(marginal_benefit), np.max(marginal_benefit - marginal_cost))

    return practice_knowledge * (marginal_benefit - marginal_cost)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=N_AGENTS)
    parser.add_argument("--months", type=int, default=DURATION_MONTHS)
    parser.add_argument("--network", choices=["small_world", "erdos_renyi", "complete"], default=NETWORK_TYPE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", default="./data/hazard_learning_dynamics_results.pkl")
    args = parser.parse_args()

    simulation = HazardLearningDynamics(
        n_agents=args.agents,
        duration_months=args.months,
        network_type=args.network,
        seed=args.seed,
    )
    results = simulation.run_simulation()
    #output_path = Path(args.output)
    output_path = (
        #Path(args.output)
        #if args.output
        Path("./data") / (
            f"HL_dynam"
            f"_cost{PRACTICE_COST:g}"
            f"_pdecay{PRACTICE_DECAY:g}"
            f"_sdecay{SEVERITY_DECAY:g}"
            f"_hrate{HAZARD_RATE:g}"
            f"_Ltimes{LEARNING_TIMES}"
            f"_agents{args.agents}"
            f"_Nneigh{NETWORK_NEIGHBORS:g}"
            f"_network{args.network}"
            f"_CoUt{TAG}.pkl"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        pickle.dump(results, output_file)
    print(
        f"Saved {len(results['time_history'])} months for {args.agents} agents "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()