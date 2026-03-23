import os

# Reproducibility
RANDOM_SEED = 42

# Dataset
IMAGE_SHAPE = (64, 64)          # Olivetti faces are 64 × 64 pixels
N_PIXELS    = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]   # 4096
TEST_SIZE   = 0.25              # fraction held out for testing

# PCA / Eigenfaces
# Maximum number of components retained when fitting PCA.
# None -> keep all components (min(n_samples, n_features)).
MAX_COMPONENTS = None

# Component counts used for reconstruction & recognition sweeps.
COMPONENT_COUNTS = [1, 2, 5, 10, 20, 30, 50, 80, 100, 150, 200, 250, 300]

# Number of eigenfaces to display in the grid visualisation.
N_EIGENFACES_DISPLAY = 16

# Visualisation / plotting
FIGURE_DPI       = 200
CMAP             = "gray"
FONT_SIZE_TITLE  = 14
FONT_SIZE_LABEL  = 12
FONT_SIZE_TICK   = 10

# Colour palette for projection scatter plots (up to 40 subjects).
PROJECTION_PALETTE = "tab20"

# Output paths
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "output")
FIGURES_DIR   = os.path.join(OUTPUT_DIR, "figures")

def ensure_output_dirs():
    """Create output directories if they do not already exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
