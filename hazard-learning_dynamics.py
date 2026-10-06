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
    RANDOM_SEED,
    SEVERITY_DECAY,
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
    severity_history: List[float] = field(default_factory=list)
    memory_history: List[float] = field(default_factory=list)
    knowledge_history: List[float] = field(default_factory=list)
    practice_history: List[bool] = field(default_factory=list)


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
        self.rng = np.random.default_rng(seed)
        self.network = build_social_network(
            n_agents=n_agents, network_type=network_type, rng=self.rng
        )
        self.agent_states = [AgentState(agent_id=agent) for agent in range(n_agents)]
        initial_known = int(round(KNOWN_FRACTION * n_agents))
        #print('INITIAL KNOWN', initial_known)
        #for agent in self.rng.choice(n_agents, size=initial_known, replace=False):
        #    self.agent_states[int(agent)].learning_progress = LEARNING_TIMES
        self.time_history: List[int] = []
        self.hazard_frequency_history: List[float] = []
        self.total_events_history: List[int] = []
        self.mean_memory_history: List[float] = []
        self.known_fraction_history: List[float] = []
        self.mean_knowledge_history: List[float] = []
        self.practice_use_fraction_history: List[float] = []

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
        previous_use = [state.practice_used for state in self.agent_states]
        for state in self.agent_states:
            neighbors = self.network[state.agent_id]
            exposed = any(previous_use[neighbor] for neighbor in neighbors)
            if exposed:
                state.learning_progress = min(
                    self.learning_times, state.learning_progress + 1.0
                )
            else:
                state.learning_progress *= np.exp(-self.practice_decay)
            state.practice_knowledge = min(
                1.0, state.learning_progress / max(self.learning_times, 1.0)
            )
            if np.random.rand() < KNOWN_FRACTION:
                state.practice_knowledge = 1.0

            if len(self.practice_use_fraction_history) == 0:
                hazard_response = 1.0 - np.exp(-state.severity_memory / max(MEMORY_RESPONSE_SCALE , 1e-12))
            else:
                hazard_response = 1.0 - np.exp(
                    -state.severity_memory / max(MEMORY_RESPONSE_SCALE * (1-self.practice_use_fraction_history[-1]), 1e-12)
                )
            cost_factor = np.exp(-self.practice_cost / max(COST_SENSITIVITY, 1e-12))
            use_probability = np.clip(
                state.practice_knowledge * hazard_response * cost_factor, 0.0, 1.0
            )
            state.practice_used = bool(self.rng.random() < use_probability)
            state.knowledge_history.append(state.practice_knowledge)
            state.practice_history.append(state.practice_used)

    def run_simulation(self) -> dict:
        """Run the model and return aggregate and per-agent histories."""
        n_always_know = int(round((KNOWN_FRACTION*N_AGENTS)))
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
            #print("Month", time_months, "Frequency", frequency, "Total events", total_events)


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
            f"_network{args.network}.pkl"
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