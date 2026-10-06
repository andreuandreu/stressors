"""Configuration for the individual-agent hazard-learning model."""

# Population and time
N_AGENTS = 250
DURATION_MONTHS = 300
TIME_STEP_MONTHS = 1

# Social network. SMALL_WORLD is the default Watts-Strogatz-style graph.
NETWORK_TYPE = "small_world"  # small_world, erdos_renyi, or complete
NETWORK_NEIGHBORS = 24  # Even number of initial ring neighbors
NETWORK_REWIRE_PROBABILITY = 0.05
NETWORK_EDGE_PROBABILITY = 0.04  # Used by erdos_renyi

# Hazard forcing. Frequency is events per agent per month.
HAZARD_INITIAL_FREQUENCY = 0.01
HAZARD_RATE = 0.012  # Exponential growth rate per month
NEV_MU = 0.0
NEV_BETA = 0.1

# Memory and event response
SEVERITY_DECAY = 0.025  # Exponential decay rate per month

# Practice learning and use
KNOWN_FRACTION = 0.05 # minimum number of agents that always knows the practice
LEARNING_TIMES = 4  # Neighbor-use exposures needed to learn the practice
PRACTICE_DECAY = 0.04 # Exponential forgetting rate without neighbor exposure
PRACTICE_COST = 0.3 # Cost of practice adoption relative to hazard impact
COST_SENSITIVITY = 0.5 # Sensitivity of practice adoption to cost relative to hazard impact
MEMORY_RESPONSE_SCALE = 1.0 # Scale factor for the response to hazard memory in practice use probability

# Reproducible default for the command-line example
RANDOM_SEED = 123
