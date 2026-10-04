# To run this script you must use the Python interpreter from
# the virtual environment that has matplotlib installed.
# e.g.:
#   source ~/dev/venvs/stressors/bin/activate
#   python plot_cells.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import stressor_dynamics
from generate_cells import Cell  # Import Cell class so pickle can find it

def load_cells():
    """Load cells from simulation results if available, otherwise fall back to the static cell pickle."""
    results_path = os.path.join('data', 'stressor_dynamics_results.pkl')
    if os.path.exists(results_path):
        import __main__
        __main__.CellDynamicsState = stressor_dynamics.CellDynamicsState
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        cells = results['cells']
        cell_states = results['cell_states']

        state_map = {}
        for state in cell_states:
            state_map[id(state.cell)] = state

        mean_events_per_cell = float(np.mean([state.num_events for state in cell_states])) if cell_states else 0.0
        return cells, state_map, mean_events_per_cell

    with open('cells_state_2.pkl', 'rb') as f:
        cells = pickle.load(f)
    # ensure neighbours are populated
    try:
        from generate_cells import compute_neighbours
        compute_neighbours(cells)
    except ImportError:
        pass
    return cells, None, None

def plot_cells(cells, state_map=None, mean_events_per_cell=None):
    """Plot cells as rectilinear polygons with green for active=True and red for active=False.

    If simulation results are available, annotate each cell with the latest annual case count
    plus the mean events-per-cell baseline.
    """
    # determine bounding box of all cells to size canvas dynamically
    all_x = [x for c in cells for (x, y) in c.vertices]
    all_y = [y for c in cells for (x, y) in c.vertices]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width = max_x - min_x
    height = max_y - min_y
    # choose a figure size that keeps axes square, with 0.05 margin
    fig_size = max(width, height) / 20
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Count polygon sides distribution
    sides_count = {}
    for c in cells:
        s = c.num_sides
        sides_count[s] = sides_count.get(s, 0) + 1

    for c in cells:
        color = "green" if c.active else "red"
        # Create polygon patch
        polygon = patches.Polygon(
            c.vertices,
            linewidth=1.5,
            edgecolor="black",
            facecolor=color,
            alpha=0.75
        )
        ax.add_patch(polygon)

        # Annotate cell-level event metrics when available.
        cx, cy = c.center
        direct = len(c.neighbours)
        radius = len(c.radius_neighbours)
        state = None
        if state_map is not None:
            state = state_map.get(id(c))

        if state is not None:
            cases = float(state.num_events)
            deviation = cases - mean_events_per_cell
            sign = '+' if deviation >= 0 else '-'
            label = (
                f"cases={cases:.2f}\n"
                f"avg={mean_events_per_cell:.2f}\n"
                f"dev={sign}{abs(deviation):.2f}"
            )
            ax.text(cx, cy, label, ha='center', va='center', fontsize=6,
                    color='white', weight='bold')
        else:
            ax.text(cx, cy, f"{direct}\n{radius}", ha='center', va='center',
                    fontsize=6, color='white', weight='bold')

    # set limits using bounding box with small padding
    pad = 1
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # Create title with polygon distribution info
    sides_range = f"{min(sides_count.keys())}-{max(sides_count.keys())} sides"
    dist_str = ", ".join([f"{s}:{sides_count[s]}" for s in sorted(sides_count.keys())])
    title = f'Cell Distribution ({len(cells)} cells, {sides_range})\nRed=Inactive, Green=Active\nDistribution: {dist_str}'
    if mean_events_per_cell is not None:
        title += f'\nMean events/cell: {mean_events_per_cell:.2f}'
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cells, state_map, mean_events_per_cell = load_cells()
    source = 'data/stressor_dynamics_results.pkl' if state_map is not None else 'cells_state_2.pkl'
    print(f"Loaded {len(cells)} cells from {source}")
    plot_cells(cells, state_map=state_map, mean_events_per_cell=mean_events_per_cell)
