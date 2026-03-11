import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import tempfile
import os
from pathlib import Path
from stressor_dynamics import CellDynamicsState
from config import DURATION_YEARS

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not available. Will create frame PNGs only.")


def load_results(filename: str = 'stressor_dynamics_results.pkl'):
    """Load simulation results from pickle file."""
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results


def polygon_to_patch(vertices):
    """Convert polygon vertices to matplotlib polygon patch."""
    polygon = patches.Polygon(vertices, closed=True)
    return polygon


def create_frame(cells, cell_states, time_step, nevents, output_path):
    """Create a single frame showing active/inactive cells at a specific time step."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create patches for each cell colored by active state
    patches_list = []
    colors = []
    
    for cell, state in zip(cells, cell_states):
        patch = polygon_to_patch(cell.vertices)
        patches_list.append(patch)
        
        # Color by active state
        if state.active_history[time_step]:
            colors.append('green')
        else:
            colors.append('lightcoral')
    
    # Create patch collection
    pc = PatchCollection(patches_list, facecolors=colors, edgecolors='black', linewidths=0.5)
    ax.add_collection(pc)
    
    # Set axis limits based on cell positions
    all_vertices = []
    for cell in cells:
        all_vertices.extend(cell.vertices)
    
    if all_vertices:
        xs = [v[0] for v in all_vertices]
        ys = [v[1] for v in all_vertices]
        ax.set_xlim(min(xs) - 5, max(xs) + 5)
        ax.set_ylim(min(ys) - 5, max(ys) + 5)
    
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis to match standard image coordinates
    
    # Add title with simulation info
    active_count = sum(1 for state in cell_states if state.active_history[time_step])
    title = f'Year {time_step}: {active_count}/{len(cells)} cells active | Nevents={nevents[time_step]:.1f}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('X (pixels)', fontsize=11)
    ax.set_ylabel('Y (pixels)', fontsize=11)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Active'),
        Patch(facecolor='lightcoral', edgecolor='black', label='Inactive'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def create_movie(results, output_video: str = 'stressor_dynamics_movie.mp4', 
                 tframe: float = 0.5):
    """
    Create a movie from simulation results using imageio.
    
    Args:
        results: Dictionary from stressor_dynamics_results.pkl
        output_video: Output video filename
        tframe: Time per frame in seconds (default 0.5)
    """
    if not HAS_IMAGEIO:
        create_frames_only(results, tframe)
        return
    
    cells = results['cells']
    cell_states = results['cell_states']
    time_history = results['time_history']
    nevents_history = results['nevents_history']
    
    nframes = len(time_history)
    fps = 1.0 / tframe  # Calculate fps from frame duration
    
    print(f"Creating {nframes} frames for movie...")
    print(f"Frame duration: {tframe} seconds (fps={fps:.1f})")
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Generate frames
        frames = []
        for t in range(nframes):
            if t % 10 == 0:
                print(f"  Generating frame {t}/{nframes}...")
            
            frame_path = os.path.join(temp_dir, f'frame_{t:04d}.png')
            create_frame(cells, cell_states, t, nevents_history, frame_path)
            
            # Read frame and add to list
            frame = imageio.imread(frame_path)
            frames.append(frame)
        
        print(f"All {nframes} frames generated.")
        
        # Write video
        print(f"Writing video: {output_video}...")
        imageio.mimsave(output_video, frames, fps=fps)
        print(f"✓ Video saved to {output_video}")
        
    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir)
        print(f"Temporary frames directory cleaned up.")


def create_frames_only(results, tframe: float = 0.5):
    """Create frame PNG files only (when imageio not available)."""
    cells = results['cells']
    cell_states = results['cell_states']
    time_history = results['time_history']
    nevents_history = results['nevents_history']
    
    nframes = len(time_history)
    
    print(f"Creating {nframes} frame PNG files...")
    print(f"Frame duration: {tframe} seconds (fps={1.0/tframe:.1f})")
    
    frames_dir = 'stressor_dynamics_frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    # Generate frames
    for t in range(nframes):
        if t % 10 == 0:
            print(f"  Generating frame {t}/{nframes}...")
        
        frame_path = os.path.join(frames_dir, f'frame_{t:04d}.png')
        create_frame(cells, cell_states, t, nevents_history, frame_path)
    
    print(f"✓ All {nframes} frames saved to {frames_dir}/")
    print(f"Total movie duration: {nframes * tframe:.1f} seconds")
    print(f"\nTo create a video from these frames, use:")
    print(f"  ffmpeg -framerate {1.0/tframe:.1f} -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p stressor_dynamics_movie.mp4")


def main():
    """Load results and create movie."""
    print("Loading results from stressor_dynamics_results.pkl...")
    results = load_results()
    
    print(f"Loaded {len(results['cells'])} cells and {len(results['time_history'])} time steps\n")
    
    # Create movie with 0.5 seconds per frame
    create_movie(results, output_video='stressor_dynamics_movie.mp4', tframe=0.5)
    
    # Print info
    print("\n=== MOVIE CREATION SUMMARY ===")
    print(f"Total cells: {len(results['cells'])}")
    print(f"Total time steps: {len(results['time_history'])}")
    print(f"Frame duration: 0.5 seconds (2 fps)")
    print(f"Total movie duration: {len(results['time_history']) * 0.5:.1f} seconds")
    if HAS_IMAGEIO:
        print(f"Output file: stressor_dynamics_movie.mp4")
    else:
        print(f"Output frames directory: stressor_dynamics_frames/")


if __name__ == "__main__":
    main()
