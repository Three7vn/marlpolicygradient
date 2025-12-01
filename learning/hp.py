# ============================================================================
# Core Training & Memory Configuration
# ============================================================================

SEED = 2                        # Random seed for reproducibility across runs
BUFFER_SIZE = int(1e6)          # Maximum capacity of experience replay buffer
BATCH_SIZE = 1024               # Sample size for each training iteration
START_SINCE = 1024              # Minimum experiences before training begins
GAMMA = 0.95                    # Temporal discount factor for future rewards
TAU = 0.2                       # Polyak averaging coefficient for target network updates
INIT_EPS = 0.                   # Initial noise scale for Ornstein-Uhlenbeck process
ACTOR_LR = 1e-3                 # Policy network optimization learning rate
CRITIC_LR = 1e-3                # Value network optimization learning rate
WEIGHT_DECAY = 0                # L2 regularization strength for critic optimizer
UPDATE_EVERY = 1                # Training frequency (steps between updates)
N_UPDATES = 1                   # Number of gradient steps per training cycle
A = 0.                          # Priority exponent alpha for weighted sampling
INIT_BETA = 0.                  # Initial importance sampling correction factor
P_EPS = 1e-3                    # Small constant added to priorities for numerical stability
N_STEPS = 1                     # Lookahead steps for n-step return calculation
V_MIN = -0.1                    # Lower bound of value distribution support
V_MAX = 0.1                     # Upper bound of value distribution support
CLIP = None                     # Gradient norm clipping threshold (None disables)

# ============================================================================
# Neural Network Architecture Parameters
# ============================================================================

N_ATOMS = 51                    # Discrete support size for distributional Q-learning
INIT_SIGMA = 0.500              # Initial standard deviation for noisy layer parameters
LINEAR = 'noisy'                # Layer type selection: 'linear' (standard) or 'noisy' (exploratory)
FACTORIZED = True               # Use factorized Gaussian noise (more efficient than independent)
DISTRIBUTIONAL = True           # Enable distributional critic (C51-style value distribution)


# ============================================================================
# Parameter Validation
# ============================================================================
# Validate core training parameters
assert isinstance(SEED, int), "SEED must be an integer"
assert isinstance(BUFFER_SIZE, int) and BUFFER_SIZE > 0, "BUFFER_SIZE must be positive integer"
assert isinstance(BATCH_SIZE, int) and BATCH_SIZE > 0, "BATCH_SIZE must be positive integer"
assert isinstance(START_SINCE, int) and START_SINCE >= BATCH_SIZE, "START_SINCE must be >= BATCH_SIZE"
assert isinstance(GAMMA, (int, float)) and 0 <= GAMMA <= 1, "GAMMA must be in [0, 1]"
assert isinstance(TAU, (int, float)) and 0 <= TAU <= 1, "TAU must be in [0, 1]"
assert isinstance(INIT_EPS, (int, float)) and 0 <= INIT_EPS <= 1, "INIT_EPS must be in [0, 1]"
assert isinstance(ACTOR_LR, (int, float)) and ACTOR_LR >= 0, "ACTOR_LR must be non-negative"
assert isinstance(CRITIC_LR, (int, float)) and CRITIC_LR >= 0, "CRITIC_LR must be non-negative"
assert isinstance(WEIGHT_DECAY, (int, float)) and WEIGHT_DECAY >= 0, "WEIGHT_DECAY must be non-negative"
assert isinstance(UPDATE_EVERY, int) and UPDATE_EVERY > 0, "UPDATE_EVERY must be positive integer"
assert isinstance(N_UPDATES, int) and N_UPDATES > 0, "N_UPDATES must be positive integer"
assert isinstance(A, (int, float)) and 0 <= A <= 1, "A (priority exponent) must be in [0, 1]"
assert isinstance(INIT_BETA, (int, float)) and 0 <= INIT_BETA <= 1, "INIT_BETA must be in [0, 1]"
assert isinstance(P_EPS, (int, float)) and P_EPS >= 0, "P_EPS must be non-negative"
assert isinstance(N_STEPS, int) and N_STEPS > 0, "N_STEPS must be positive integer"
assert isinstance(V_MIN, (int, float)) and isinstance(V_MAX, (int, float)) and V_MIN < V_MAX, "V_MIN must be < V_MAX"
if CLIP: assert isinstance(CLIP, (int, float)) and CLIP >= 0, "CLIP must be non-negative if enabled"

# Validate network architecture parameters
assert isinstance(N_ATOMS, int) and N_ATOMS > 0, "N_ATOMS must be positive integer"
assert isinstance(INIT_SIGMA, (int, float)), "INIT_SIGMA must be numeric"
assert isinstance(LINEAR, str) and LINEAR.lower() in ('linear', 'noisy'), "LINEAR must be 'linear' or 'noisy'"
assert isinstance(FACTORIZED, bool), "FACTORIZED must be boolean"
assert isinstance(DISTRIBUTIONAL, bool), "DISTRIBUTIONAL must be boolean"