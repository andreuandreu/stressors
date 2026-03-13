# To run this script you must use the Python interpreter from
# the virtual environment that has matplotlib installed.
# e.g.:
#   source ~/dev/venvs/stressors/bin/activate
#   python plot_cells.py

import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from generate_cells import Cell  # Import Cell class so pickle can find it

def load_cells():
    """Load cells from pickle file and compute neighbours if needed."""
    with open('cells_state_2.pkl', 'rb') as f:
        cells = pickle.load(f)
    # ensure neighbours are populated
    try:
        from generate_cells import compute_neighbours
        compute_neighbours(cells)
    except ImportError:
        pass
    return cells

def plot_cells(cells):
    """Plot cells as rectilinear polygons with green for active=True and red for active=False."""
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
        # annotate counts at the centroid: direct neighbours on top, radius neighbours below
        cx, cy = c.center
        direct = len(c.neighbours)
        radius = len(c.radius_neighbours)
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
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cells = load_cells()
    print(f"Loaded {len(cells)} cells from cells_state.pkl")
    plot_cells(cells)
