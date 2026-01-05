import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 32
LR = 0.001
NUM_EPOCHS = 15
LAMBDA_GOAL = 1.5
GRID_HEIGHT = 12
GRID_WIDTH = 8
NUM_EVENT_CLASSES = 3
LOSS_TYPE = 'Focal' # Or CE doesnt work yet