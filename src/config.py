import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COMP_IDS = [72, 53, 53, 43, 55, 55]
SEAS_IDS = [107, 315, 106, 106, 282, 43]
FORCE_REDOWNLOAD = False

# Hyperparameters
LR = 0.001
LR_3D = 0.0001
LAMBDA_GOAL = 0.5 # parameter to weight two different objectives
GRID_HEIGHT = 12
GRID_WIDTH = 8
NUM_EVENT_CLASSES = 3
LOSS_TYPE = 'Focal' # Or CE **doesnt work yet**


# Model Batch Sizes
BATCH_SIZE = 32
BASELINE_BATCH_SIZE = 64
CONTEXT_BATCH_SIZE = 64
KINETIC_BATCH_SIZE = 64
VOXEL_BATCH_SIZE = 64

# Model Epochs
NUM_EPOCHS = 1
BASELINE_NUM_EPOCHS = 5
CONTEXT_NUM_EPOCHS = 5
KINETIC_NUM_EPOCHS = 5
VOXEL_NUM_EPOCHS = 5

VOXEL_WEIGHTS = [1, 1, 1]
#VOXEL_WEIGHTS = [0.69, 1.15, 6.7]

competition_ids = [72, 53, 53, 43, 55, 55]
season_ids = [107, 315, 106, 106, 282, 43]