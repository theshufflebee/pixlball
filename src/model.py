import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


#---------------------------------------------------------------
# NNs without temporal Context
#-----------------------------------------------------------------------

# The Baseline model
# -------------------------------------------------------
class TinyCNN_MultiTask_Threat(nn.Module):
    def __init__(self, grid_height=12, grid_width=8, num_event_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2), # Standardized pooling
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )
        # 12x8 -> 6x4 -> 3x2
        self.flatten_size = 32 * 3 * 2 
        
        self.fc_shared = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        self.fc_event = nn.Linear(128, num_event_classes)
        self.fc_goal = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_shared(x)
        return self.fc_event(x), self.fc_goal(x).squeeze(-1)
    
    
# The Context Model
#----------------------------------------
class TinyCNN_MultiTask_Context_Threat(nn.Module):
    def __init__(self, grid_height=12, grid_width=8, num_event_classes=3, num_context_features=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )
        self.cnn_flat_size = 32 * 3 * 2
        self.fc_context = nn.Sequential(
            nn.Linear(num_context_features, 32),
            nn.LeakyReLU(0.1)
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(self.cnn_flat_size + 32, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)
        )
        self.fc_event = nn.Linear(128, num_event_classes)
        self.fc_goal = nn.Linear(128, 1)

    def forward(self, x_grid, x_context):
        x = self.features(x_grid)
        x = x.view(x.size(0), -1)
        c = self.fc_context(x_context)
        combined = torch.cat((x, c), dim=1)
        shared = self.fc_shared(combined)
        return self.fc_event(shared), self.fc_goal(shared).squeeze(-1)
    
    
    
#---------------------------------------------------------------
# NNs with temporal Context
#-----------------------------------------------------------------------

# The Kinetic Model
# ---------------------------------------------------------------------
class TinyCNN_MultiTask_Context_Ball_Vector(nn.Module):
    def __init__(self, grid_height, grid_width, num_event_classes, num_context_features):
        super(TinyCNN_MultiTask_Context, self).__init__()
        
        # --- 1. Spatial Branch (The CNN) ---
        # Input: [Batch, 3, 12, 8] (Ball, Teammates, Opponents)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        # Calculate the flattened size after two 2x2 poolings
        # For 12x8: 12->6->3 and 8->4->2
        self.cnn_flat_size = 32 * (grid_height // 2 // 2) * (grid_width // 2 // 2)
        
        # --- 2. Context Branch (The Ball History) ---
        # Input: [Batch, num_context_features] (e.g., 8 features for 4 positions)
        self.context_fc = nn.Sequential(
            nn.Linear(num_context_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # --- 3. Fusion & Multi-Task Heads ---
        self.fc_combined = nn.Linear(self.cnn_flat_size + 32, 64)
        
        # Task 1: Event Classification (Keep, Lose, Shot)
        self.event_head = nn.Linear(64, num_event_classes)
        
        # Task 2: Possession Threat (xG / Goal Probability)
        self.threat_head = nn.Linear(64, 1)

    def forward(self, x_grid, x_context):
        # x_grid: [Batch, 3, 12, 8]
        # x_context: [Batch, 8]
        
        # Spatial processing
        x = F.relu(self.conv1(x_grid))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x_spatial = x.view(-1, self.cnn_flat_size)
        
        # Contextual processing (Ball Trajectory)
        x_val = self.context_fc(x_context)
        
        # Late Fusion
        combined = torch.cat((x_spatial, x_val), dim=1)
        combined = F.relu(self.fc_combined(combined))
        
        # Multi-task outputs
        event_logits = self.event_head(combined)
        threat_score = torch.sigmoid(self.threat_head(combined)) # Binary Goal/No-Goal
        
        return event_logits, threat_score

    
# The Voxel Model
# ----------------------------------------------------
class Tiny3DCNN_MultiTask(nn.Module):
    def __init__(self, num_event_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        self.flatten_size = 32 * 2 * 3 * 2 
        self.fc_shared = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4)
        )
        self.event_head = nn.Linear(128, num_event_classes)
        self.goal_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_shared(x)
        return self.event_head(x), self.goal_head(x).squeeze(-1)