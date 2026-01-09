import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
LR = 0.001
LR_3D = 0.0001
LAMBDA_GOAL = 0.5
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
BASELINE_NUM_EPOCHS = 10
CONTEXT_NUM_EPOCHS = 10
KINETIC_NUM_EPOCHS = 10
VOXEL_NUM_EPOCHS = 10
