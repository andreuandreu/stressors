import pickle
import numpy as np
from typing import List
from dataclasses import dataclass, field
from generate_cells import Cell
from config import (
    NEV_MU, NEV_BETA, SEV_THRESHOLD, DECAY,
    NSTUBBORN, STUBORNESS, NEEDED_COST, MAX_CAP, RADIUS, DURATION_YEARS, YRS_THRES
)


@dataclass
class CellDynamicsState:
    """Extended state for each cell incorporating forcing and effort dynamics."""
    cell: Cell
    
    # Forcing dynamics
    num_events: float = 0.0  # Number of events this cell experiences in the year
    severity: float = 0.0  # Severity of the current stressor event
    memory: float = 0.0  # Time (years) since last severe event (severity >= sebThres)
    
    # Effort dynamics
    stubborn: bool = False  # If True, cell will never activate (probability Nstubborn = 0.1)
    exposure: float = 0.0  # Random [0, 1], stable across simulation
    capacity: float = 0.0  # Random * cellSize, computed at initialization
    willing_cost: float = 0.0  # Computed each tick based on dynamics
    consecutive_positive_delta_cost_years: int = 0  # Years with DeltaCost > 0 (triggers deactivation)
    
    # History tracking
    severity_history: List[float] = field(default_factory=list)
    memory_history: List[float] = field(default_factory=list)
    num_events_history: List[float] = field(default_factory=list)
    active_history: List[bool] = field(default_factory=list)
    delta_cost_history: List[float] = field(default_factory=list)
    willing_cost_history: List[float] = field(default_factory=list)


class StressorDynamics:
    """Implements coupled forcing and effort dynamics on a cell network."""
    
    # Load parameters from config file
    NEV_MU = NEV_MU
    NEV_BETA = NEV_BETA
    SEV_THRESHOLD = SEV_THRESHOLD
    DECAY = DECAY
    NSTUBBORN = NSTUBBORN
    STUBORNESS = STUBORNESS
    NEEDED_COST = NEEDED_COST
    MAX_CAP = MAX_CAP
    
    def __init__(self, cells: List[Cell], duration_years: int = None, radius: float = None):
        """
        Initialize the stressor dynamics simulation.
        
        Args:
            cells: List of Cell objects from generate_cells.py
            duration_years: Number of years to simulate (default from config)
            radius: Radius for neighbor consideration (default from config)
        """
        self.cells = cells
        self.duration_years = duration_years if duration_years is not None else DURATION_YEARS
        self.radius = radius if radius is not None else RADIUS
        self.ncells = len(cells)
        self.time = YRS_THRES + 1  # Start at YRS_THRES to allow for initial memory accumulation
        
        # Initialize extended state for each cell
        self.cell_states: List[CellDynamicsState] = []
        self._initialize_cell_states()
        
        # Global history
        self.nevents_history: List[float] = []
        self.time_history: List[int] = []
        
    def _initialize_cell_states(self):
        """Initialize all cell states with random attributes."""
        for cell in self.cells:
            state = CellDynamicsState(cell=cell)
            
            # Forcing dynamics initialization
            state.severity = np.random.uniform(0, 1)  # Initial severity at t=0
            state.memory = np.random.uniform(0, YRS_THRES*2)  # Random initial memory to avoid all starting at 0
            
            # Effort dynamics initialization
            state.stubborn = np.random.random() < self.NSTUBBORN
            state.exposure = np.random.uniform(0, 1)
            
            # Capacity depends on cell area (as proxy for cellSize)
            cell_size = cell.area if cell.area > 0 else 1
            state.capacity = np.random.uniform(
                self.NEEDED_COST, 
                self.MAX_CAP * self.NEEDED_COST
            ) * cell_size
            
            self.cell_states.append(state)
    
    def get_nevents(self, t: int) -> float:
        """
        Generate overall stressor meanfield for year t.
        
        Args:
            t: Year (0 to duration_years)
            
        Returns:
            Total number of events Nevents(t) = Ncells/10.0 * t
        """
        return (self.ncells / 10.0) * t + 10  # Added +10 to ensure some events at t=0
    
    def distribute_events(self, nevents: float):
        """
        Distribute randomly the total number of events across cells,
        ensuring sum equals nevents and each cell gets [0, nevents].
        """
        if nevents == 0:
            for state in self.cell_states:
                state.num_events = 0.0
            return
        
        # Use Dirichlet distribution to randomly distribute events
        # This ensures: sum = nevents, each cell gets portion of total
        portions = np.random.dirichlet(np.ones(self.ncells))
        for state, portion in zip(self.cell_states, portions):
            state.num_events = portion * nevents
    
    def update_forcing_dynamics(self, t: int):
        """
        Update forcing dynamics for all cells at year t.
        
        Steps:
        1. Generate Nevents(t) = Ncells/10.0 * t
        2. Distribute events across cells
        3. Update severity for each cell (Gumbel distribution)
        4. Update memory (time since last severe event)
        """
        # Step 1: Calculate total events
        nevents = self.get_nevents(t)
        self.nevents_history.append(nevents)
        
        # Step 2: Distribute events
        self.distribute_events(nevents)
        
        # Step 3-4: Update severity and memory for each cell
        for state in self.cell_states:
            # Generate severity if there's an event this tick
            if state.num_events > 0:
                # Severity is sampled from Gumbel distribution
                severity_sample = np.abs(np.random.gumbel(self.NEV_MU, self.NEV_BETA))
                # Update only if severity exceeds threshold
                if severity_sample > self.SEV_THRESHOLD:
                    state.severity = severity_sample
                    state.memory = 0.0  # Reset memory (just had severe event)
                # else: keep previous severity
            
            # Increment memory (years since last severe event)
            state.memory += 1.0
            
            # Store history
            state.severity_history.append(state.severity)
            state.memory_history.append(state.memory)
            state.num_events_history.append(state.num_events)
    
    def update_effort_dynamics(self, t: int):
        """
        Update effort dynamics for all cells at year t.
        
        For each cell:
        - If stubborn, cell remains inactive
        - Otherwise, compute willing_cost and check if will activate
        - Activation depends on social influence and cost-benefit analysis
        """
        nevents = self.nevents_history[-1] if self.nevents_history else 0
        
        for state in self.cell_states:
            delta_cost = 0.0
            
            # Skip if cell is stubborn and currently inactive
            if state.stubborn and not state.cell.active:
                state.active_history.append(state.cell.active)
                state.delta_cost_history.append(delta_cost)
                state.willing_cost_history.append(state.willing_cost)
                state.consecutive_positive_delta_cost_years = 0  # Reset for inactive stubborn cells
                continue
            
            # For susceptible cells, compute willing cost and decision
            if  not state.stubborn or state.cell.active:
                # Calculate smoothed severity over last YRS_THRES years
                # smoothSeb(t) = sum(severity from t-YRS_THRES to t) / sum(events in same window)
                severity_window_start = max(0, t - YRS_THRES)
                smoothed_severity = 0.0
                total_events_in_window = 0.0
                
                for time_idx in range(severity_window_start, t + 1):
                    if time_idx < len(state.severity_history):
                        smoothed_severity += state.severity_history[time_idx]
                        total_events_in_window += state.num_events_history[time_idx]
                
                # Use smoothed severity, normalized by total events in window
                if total_events_in_window > 0:
                    smoothed_severity = smoothed_severity / total_events_in_window
                else:
                    smoothed_severity = state.severity
                
                # Compute willing cost
                # willingCost(t) = exposure*capacity*smoothedSeverity/log(decay*memory*Nevents)
                # Avoid log of zero or negative
                memory_term = max(state.memory, 0.1)  # Prevent zero
                nevents_term = max(nevents, 1.0)  # Prevent zero
                denominator = np.log( memory_term * nevents_term/self.DECAY)
                
                if denominator > 0:
                    state.willing_cost = (
                        state.exposure * state.capacity * smoothed_severity / denominator
                    )
                else:
                    state.willing_cost = 0.0
                
                # Decision: activate if DeltaCost < 0
                # DeltaCost = neededCost - willingCost
                delta_cost = self.NEEDED_COST - state.willing_cost
                
                # Check social influence from neighbors
                active_radius_neighbors = sum(
                    1 for n in state.cell.radius_neighbours 
                    if n.active
                )
                total_radius_neighbors = len(state.cell.radius_neighbours)
                
                if total_radius_neighbors > 0:
                    neighbor_fraction = active_radius_neighbors / total_radius_neighbors
                else:
                    neighbor_fraction = 0
                
                # Adoption probability with social influence
                # p_adoption = stuborness * rand(0,1) > neighbor_fraction
                rand_val = np.random.random()
                adoption_prob = self.STUBORNESS * rand_val
                
                # If delta_cost < 0 AND adoption condition met, activate
                if delta_cost < 0 and adoption_prob > neighbor_fraction:
                    state.cell.active = True
                    state.consecutive_positive_delta_cost_years = 0  # Reset counter on activation
                
                # Track consecutive years of positive delta cost
                if delta_cost > 0:
                    # Increment counter for positive delta cost
                    if state.cell.active:
                        state.consecutive_positive_delta_cost_years += 1
                        # Deactivate if threshold reached
                        if state.consecutive_positive_delta_cost_years >= YRS_THRES:
                            state.cell.active = False
                else:
                    # Reset counter if delta cost becomes non-positive
                    state.consecutive_positive_delta_cost_years = 0
            
            state.active_history.append(state.cell.active)
            state.delta_cost_history.append(delta_cost)
            state.willing_cost_history.append(state.willing_cost)
    
    def run_simulation(self) -> dict:
        """
        Run the full simulation for duration_years.
        
        Returns:
            Dictionary containing simulation results and histories
        """
        print(f"Starting stressor dynamics simulation for {self.duration_years} years")
        print(f"Number of cells: {self.ncells}\n")
        
        for t in range(self.duration_years + 1):
            self.time = t
            self.time_history.append(t)
            
            # Update forcing dynamics
            self.update_forcing_dynamics(t)
            
            # Update effort dynamics
            self.update_effort_dynamics(t)
            
            # Progress report every 10 years
            if t % 10 == 0:
                nevents = self.nevents_history[-1]
                active_count = sum(1 for c in self.cells if c.active)
                print(f"Year {t}: Nevents={nevents:.2f}, Active cells={active_count}/{self.ncells}")
        
        print(f"\nSimulation complete!")
        
        results = {
            'time_history': self.time_history,
            'nevents_history': self.nevents_history,
            'cell_states': self.cell_states,
            'cells': self.cells,
        }
        
        return results
    
    def save_results(self, filename: str = './data/stressor_dynamics_results.pkl'):
        """Save simulation results to file."""
        results = {
            'time_history': self.time_history,
            'nevents_history': self.nevents_history,
            'cell_states': self.cell_states,
            'cells': self.cells,
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {filename}")


def main():
    """Main entry point: load cells and run stressor dynamics simulation."""
    # Load cells from file
    print("Loading cells from cells_state.pkl...")
    with open('./data/cells_state.pkl', 'rb') as f:
        cells = pickle.load(f)
    
    print(f"Loaded {len(cells)} cells\n")
    
    # Create and run simulation
    dynamics = StressorDynamics(cells, duration_years=DURATION_YEARS, radius=RADIUS)
    results = dynamics.run_simulation()
    
    # Save results
    dynamics.save_results('./data/stressor_dynamics_results.pkl')
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Final Nevents: {dynamics.nevents_history[-1]:.2f}")
    print(f"Final active cells: {sum(1 for c in cells if c.active)}/{len(cells)}")
    
    # Print per-cell statistics
    print("\n=== PER-CELL STATISTICS ===")
    for idx, state in enumerate(dynamics.cell_states[:10], start=1):  # Print first 10
        print(
            f"Cell {idx}: "
            f"final_severity={state.severity:.4f} "
            f"final_memory={state.memory:.1f} "
            f"active={state.cell.active} "
            f"max_severity={max(state.severity_history) if state.severity_history else 0:.4f}"
        )


if __name__ == "__main__":
    main()
