import torch.nn.functional as F

# Seed
SEED = 1

# Dataset

CLASSES = (
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
)

SHUFFLE = True
DATA_DIR = "../data"
NUM_WORKERS = 4
PIN_MEMORY = True

# Training Hyperparameters
CRITERION = F.cross_entropy
INPUT_SIZE = (3, 32, 32)
NUM_CLASSES = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
NUM_EPOCHS = 24
DROPOUT_PERCENTAGE = 0.05
LAYER_NORM = "bn"  # Batch Normalization

# OPTIMIZER & SCHEDULER

LRFINDER_END_LR = 0.1
LRFINDER_NUM_ITERATIONS = 50
LRFINDER_STEP_MODE = "exp"

OCLR_DIV_FACTOR = 100
OCLR_FINAL_DIV_FACTOR = 100
OCLR_THREE_PHASE = False
OCLR_ANNEAL_STRATEGY = "linear"

# Compute Related

ACCELERATOR = "cpu"
PRECISION = 32

# Store

TRAINING_STAT_STORE = "Store/training_stats.csv"
MODEL_SAVE_PATH = "Store/model.pth"
PRED_STORE_PATH = "Store/pred_store.pth"
EXAMPLE_IMG_PATH = "Store/examples/"

# Visualization

NORM_CONF_MAT = True
