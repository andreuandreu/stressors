"""
Configuration file for stressor dynamics simulation.
Centralized parameters for cell generation and dynamics simulation.
"""

# ============================================================================
# Cell Generation Parameters
# ============================================================================
RADIUS = 33
SIZE_PIX = 10
NX_CELLS = 12
NY_CELLS = NX_CELLS
NMERGE = 22

# ============================================================================
# Forcing Dynamics Parameters
# ============================================================================
# Gumbel distribution parameters for severity sampling
NEV_MU = 0.0
NEV_BETA = 0.1

# Threshold for severe event classification
SEV_THRESHOLD = 0.22

# Decay parameter for willing cost calculation
DECAY = 3.3

# ============================================================================
# Effort Dynamics Parameters
# ============================================================================
# Fraction of cells that are stubborn (never adopt)
NSTUBBORN = 0.1

# Base stubbornness parameter for adoption probability
STUBORNESS = 0.9

# Fixed cost of adoption
NEEDED_COST = 500.0

# Scaling factor for capacity distribution
MAX_CAP = 0.1

# Consecutive years threshold for deactivation (if DeltaCost > 0 for this many years, deactivate)
YRS_THRES = 3

# ============================================================================
# Simulation Parameters
# ============================================================================
DURATION_YEARS = 50 + YRS_THRES  # Total simulation duration in years (including threshold period)
