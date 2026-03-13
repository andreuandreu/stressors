# Stressor Dynamics Movie

## Summary

A movie visualization has been created showing the spatial-temporal evolution of cell activation states throughout the 50-year simulation.

## Files Generated

### Video
- **stressor_dynamics_movie.mp4** (103 KB)
  - 51 frames (years 0-50)
  - Frame rate: 2.0 fps (0.5 seconds per frame)
  - Total duration: 25.5 seconds
  - Resolution: High-quality visualization with labeled cells

### Frame Sequence
- **stressor_dynamics_frames/** (1.6 MB)
  - 51 individual PNG frames (frame_0000.png through frame_0050.png)
  - Each frame shows the complete simulation state at one time step
  - 100 dpi resolution for clarity

## Visualization Details

Each frame displays:
- **Green cells**: Currently active (have adopted the stressor response)
- **Red/salmon cells**: Currently inactive (have not adopted)
- **Black borders**: Cell boundaries
- **Title**: Year number, count of active cells, and total number of events (Nevents)
- **Legend**: Shows color coding for active/inactive states

## How to View

The movie can be viewed with any standard video player:
```bash
open stressor_dynamics_movie.mp4     # On macOS
```

Or use ffmpeg to get information:
```bash
ffmpeg -i stressor_dynamics_movie.mp4
```

## What the Movie Shows

The movie captures the coupled dynamics between:

1. **Forcing Dynamics**: Increasing stress over time (Nevents grows from 0 to ~355)
2. **Effort Dynamics**: Adaptive responses showing adoption patterns influenced by:
   - Cost-benefit analysis (willing cost vs. needed cost)
   - Social influence from neighboring cells
   - Memory of past severe events
   - Deactivation threshold (3 consecutive years of positive delta cost)

## Key Observations

You can observe in the movie:
- Initial propagation of adoption from scattered seed cells
- Spatial clustering of active cells due to social influence effects
- Activation waves spreading as stress increases
- Deactivation events when costs exceed willingness to pay for 3+ consecutive years
- Dynamic spatial patterns of cooperation/adoption

## Recreation

To regenerate the movie after changing simulation parameters:

```bash
# 1. Generate cells
python3 generate_cells.py

# 2. Run dynamics simulation
python3 stressor_dynamics.py

# 3. Create movie
python3 ./plots/create_movie.py
```

All scripts automatically use parameters from `config.py`, so you can modify the simulation by editing that file.
