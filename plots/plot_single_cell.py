import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./')
from stressor_dynamics import CellDynamicsState
from config import SEV_THRESHOLD, NEEDED_COST, YRS_THRES


def load_results(filename: str = './data/stressor_dynamics_results.pkl'):
    """Load simulation results from pickle file."""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results


def extract_single_cell_metrics(results, cell_idx: int = 12):  # 0-indexed, so 12 = cell 13
    """Extract time series data for a single cell."""
    time_history = results['time_history']
    cell_states = results['cell_states']
    
    if cell_idx >= len(cell_states):
        raise ValueError(f"Cell index {cell_idx} out of range. Total cells: {len(cell_states)}")
    
    cell_state = cell_states[cell_idx]
    cell = cell_state.cell
    ntime = len(time_history)
    
    # Extract basic histories
    memory_history = cell_state.memory_history
    severity_history = cell_state.severity_history
    willing_cost_history = cell_state.willing_cost_history
    
    # Calculate number of active radius neighbors over time
    active_radius_neighbors = []
    for t in range(ntime):
        # Count how many radius neighbors are active at time t
        active_count = 0
        for neighbor in cell.radius_neighbours:
            # Find the state object for this neighbor
            neighbor_idx = None
            for idx, state in enumerate(cell_states):
                if state.cell is neighbor:
                    neighbor_idx = idx
                    break
            
            if neighbor_idx is not None and cell_states[neighbor_idx].active_history[t]:
                active_count += 1
        
        active_radius_neighbors.append(active_count)
    
    return {
        'time': time_history,
        'memory': memory_history,
        'severity': severity_history,
        'active_neighbors': active_radius_neighbors,
        'willing_cost': willing_cost_history,
        'cell_active': cell_state.active_history,
        'total_radius_neighbors': len(cell.radius_neighbours),
    }


def plot_single_cell(metrics, cell_id: int = 18):
    """Create 2x2 subplot figure for single cell dynamics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Cell {cell_id} Dynamics Over Time', fontsize=16, fontweight='bold')
    
    time = metrics['time']
    memory = metrics['memory']
    severity = metrics['severity']
    active_neighbors = metrics['active_neighbors']
    willing_cost = metrics['willing_cost']
    cell_active = metrics['cell_active']
    total_neighbors = metrics['total_radius_neighbors']
    
    # Color background based on cell active state
    for ax in axes.flat:
        for i, t in enumerate(time[:-1]):
            if cell_active[i]:
                ax.axvspan(t, time[i+1], alpha=0.1, color='green', zorder=0)
            else:
                ax.axvspan(t, time[i+1], alpha=0.05, color='red', zorder=0)
    
    # Plot 1: Memory
    ax = axes[0, 0]
    ax.plot(time[YRS_THRES*2:], memory[YRS_THRES*2:], 'b-', linewidth=2, marker='o', markersize=3)
    ax.fill_between(time[YRS_THRES*2:], memory[YRS_THRES*2:], alpha=0.3, color='blue')
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Memory (years)', fontsize=11)
    ax.set_title('Memory - Time Since Last Severe Event', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Severity
    ax = axes[0, 1]
    ax.plot(time[YRS_THRES*2:] , severity[YRS_THRES*2:], 'r-', linewidth=2, marker='o', markersize=3)
    ax.fill_between(time[YRS_THRES*2:], severity[YRS_THRES*2:], alpha=0.3, color='red')
    ax.axhline(y=SEV_THRESHOLD, color='k', linestyle='--', linewidth=1, alpha=0.5, label=f'Threshold={SEV_THRESHOLD}')
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Severity', fontsize=11)
    ax.set_title('Severity of Stressor Events', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Plot 3: Active neighbors in radius
    ax = axes[1, 0]
    ax.bar(time[YRS_THRES*2:], active_neighbors[YRS_THRES*2:], width=0.8, color='purple', alpha=0.7, edgecolor='black')
    ax.axhline(y=total_neighbors/2, color='k', linestyle='--', linewidth=1, alpha=0.5, 
               label=f'Half of total neighbors ({total_neighbors/2:.0f})')
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Active Neighbors in Radius R (Total: {total_neighbors})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, total_neighbors + 1])
    ax.legend(fontsize=10)
    
    # Plot 4: Willing cost
    ax = axes[1, 1]
    ax.plot(time[YRS_THRES*2:], willing_cost[YRS_THRES*2:], 'orange', linewidth=2, marker='o', markersize=3)
    ax.fill_between(time[YRS_THRES*2:], willing_cost[YRS_THRES*2:], alpha=0.3, color='orange')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Breakeven (WC=0)')
    ax.axhline(y=NEEDED_COST, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f'Needed Cost (${NEEDED_COST})')
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Willing Cost ($)', fontsize=11)
    ax.set_title('Willing Cost (Capacity to Adopt)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add legend for background shading
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.1, edgecolor='black', label='Cell Active'),
        Patch(facecolor='red', alpha=0.05, edgecolor='black', label='Cell Inactive'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10, frameon=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    return fig


def main():
    """Load results and plot single cell dynamics."""
    print("Loading results from stressor_dynamics_results.pkl...")
    results = load_results()
    

    cell_id = int(sys.argv[1]) if len(sys.argv) > 1 else 13  # Default to cell 13 (0-indexed 12)
    cell_idx = cell_id - 1  # Convert to 0-indexed
    
    print(f"Extracting metrics for cell {cell_id} (index {cell_idx})...")
    metrics = extract_single_cell_metrics(results, cell_idx)
    
    print("Creating plot...")
    fig = plot_single_cell(metrics, cell_id)
    
    # Save figure
    fig.savefig(f'./plots/cell_{cell_id}_dynamics.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to cell_{cell_id}_dynamics.png")
    
    # Print summary statistics
    print("\n=== CELL SUMMARY STATISTICS ===")
    print(f"Cell ID: {cell_id}")
    print(f"Total radius neighbors: {metrics['total_radius_neighbors']}")
    print(f"Final memory: {metrics['memory'][-1]:.2f} years")
    print(f"Final severity: {metrics['severity'][-1]:.4f}")
    print(f"Final active neighbors: {metrics['active_neighbors'][-1]}/{metrics['total_radius_neighbors']}")
    print(f"Final willing cost: ${metrics['willing_cost'][-1]:.2f}")
    print(f"Cell active at end: {metrics['cell_active'][-1]}")
    print(f"\nMax memory: {max(metrics['memory']):.2f} years")
    print(f"Max severity: {max(metrics['severity']):.4f}")
    print(f"Max active neighbors: {max(metrics['active_neighbors'])}/{metrics['total_radius_neighbors']}")
    print(f"Max willing cost: ${max(metrics['willing_cost']):.2f}")
    print(f"Min willing cost: ${min(metrics['willing_cost']):.2f}")
    
    # Count activation events
    activation_count = sum(1 for i in range(1, len(metrics['cell_active'])) 
                          if metrics['cell_active'][i] and not metrics['cell_active'][i-1])
    print(f"Activation events: {activation_count}")
    plt.show()


if __name__ == "__main__":
    main()
