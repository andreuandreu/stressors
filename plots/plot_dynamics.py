import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./')
from stressor_dynamics import CellDynamicsState
from config import NEEDED_COST


def load_results(filename: str = './data/stressor_dynamics_results.pkl'):
    """Load simulation results from pickle file."""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results


def extract_metrics(results):
    """Extract time series data for plotting."""
    time_history = results['time_history']
    nevents_history = results['nevents_history']
    cell_states = results['cell_states']
    
    ncells = len(cell_states)
    ntime = len(time_history)
    
    # Calculate percentage active cells over time
    active_percentage = []
    for t in range(ntime):
        active_count = sum(1 for state in cell_states if state.active_history[t])
        active_percentage.append(100.0 * active_count / ncells)
    
    # Calculate mean severity over time
    mean_severity = []
    for t in range(ntime):
        severities = [state.severity_history[t] for state in cell_states]
        mean_severity.append(np.mean(severities))
    
    # Calculate mean delta cost over time
    mean_delta_cost = []
    for t in range(ntime):
        delta_costs = [state.delta_cost_history[t] for state in cell_states]
        mean_delta_cost.append(np.mean(delta_costs))
    
    return {
        'time': time_history,
        'nevents': nevents_history,
        'active_percentage': active_percentage,
        'mean_severity': mean_severity,
        'mean_delta_cost': mean_delta_cost,
    }


def plot_dynamics(metrics):
    """Create 2x2 subplot figure with all metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stressor Dynamics Simulation Results', fontsize=16, fontweight='bold')
    
    time = metrics['time']
    nevents = metrics['nevents']
    active_pct = metrics['active_percentage']
    mean_sev = metrics['mean_severity']
    mean_dc = metrics['mean_delta_cost']
    
    # Plot 1: Percentage of active cells
    ax = axes[0, 0]
    ax.plot(time, active_pct, 'b-', linewidth=2, label='Active cells %')
    ax.fill_between(time, active_pct, alpha=0.3)
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Percentage Active (%)', fontsize=11)
    ax.set_title('Percentage of Active Cells Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Plot 2: Number of events (Nevents)
    ax = axes[0, 1]
    ax.plot(time, nevents, 'g-', linewidth=2, label='Number of events')
    ax.fill_between(time, nevents, alpha=0.3, color='green')
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Number of Events', fontsize=11)
    ax.set_title('Number of Events Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Mean severity
    ax = axes[1, 0]
    ax.plot(time, mean_sev, 'r-', linewidth=2, label='Mean severity')
    ax.fill_between(time, mean_sev, alpha=0.3, color='red')
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Mean Severity', fontsize=11)
    ax.set_title('Mean Severity Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Mean delta cost
    ax = axes[1, 1]
    ax.plot(time, mean_dc, 'purple', linewidth=2, label='Mean delta cost')
    ax.fill_between(time, mean_dc, alpha=0.3, color='purple')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Mean Delta Cost ($)', fontsize=11)
    ax.set_title('Mean Delta Cost Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Load results, extract metrics, and create plots."""
    results = load_results()
    
    print("Extracting metrics...")
    metrics = extract_metrics(results)
    
    print("Creating plots...")
    fig = plot_dynamics(metrics)
    
    # Save figure
    fig.savefig('./plots/stressor_dynamics_plots.png', dpi=300, bbox_inches='tight')
    print("Plot saved to stressor_dynamics_plots.png")
    
    # Show summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Final % active cells: {metrics['active_percentage'][-1]:.1f}%")
    print(f"Final number of events: {metrics['nevents'][-1]:.1f}")
    print(f"Mean severity (final): {metrics['mean_severity'][-1]:.4f}")
    print(f"Mean delta cost (final): {metrics['mean_delta_cost'][-1]:.2f}")
    print(f"Max mean severity: {max(metrics['mean_severity']):.4f}")
    print(f"Min mean severity: {min(metrics['mean_severity']):.4f}")
    print(f"Max mean delta cost: {max(metrics['mean_delta_cost']):.2f}")
    print(f"Min mean delta cost: {min(metrics['mean_delta_cost']):.2f}")
    
    # Uncomment to display plot in interactive mode
    plt.show()


if __name__ == "__main__":
    main()
