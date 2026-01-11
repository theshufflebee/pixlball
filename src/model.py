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
            nn.MaxPool2d(2),
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
        
        # Task 1: Event Classification (Keep, Lose, Shot)
        self.fc_event = nn.Linear(128, num_event_classes)
        
        # Task 2: Possession Threat (xG / Goal Probability)
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
    def __init__(self, grid_height=12, grid_width=8, num_event_classes=3, num_context_features=8):
        super().__init__()
        # 1. Spatial Branch - EXACT MATCH to Baseline
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
        self.cnn_flat_size = 32 * 3 * 2 # Standardized for 12x8
        
        # 2. Kinetic Context Branch
        self.fc_context = nn.Sequential(
            nn.Linear(num_context_features, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        # 3. Shared Fusion
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
        # Return Logits for both (Standard for BCEWithLogitsLoss)
        return self.fc_event(shared), self.fc_goal(shared).squeeze(-1)

    
# The Voxel Model
# ----------------------------------------------------
class Tiny3DCNN_MultiTask(nn.Module):
    def __init__(self, num_event_classes=3):
        super().__init__()
        
        # 1. Spatiotemporal Feature Extractor
        self.features = nn.Sequential(
            # Layer 1: Capture early motion
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.1),
            # KEY CHANGE: Temporal Compression (2, 2, 2)
            # This reduces Time (4->2) and Space (12x8 -> 6x4) early
            nn.MaxPool3d(kernel_size=(2, 2, 2)), 
            
            # Layer 2: Deeper tactical patterns
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1),
            # KEY CHANGE: Final compression to a 1D temporal state
            # Time (2->1), Space (6x4 -> 3x2)
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        
        # Flattened size: 32 channels * 1 (Time) * 3 (Height) * 2 (Width) = 192
        self.flatten_size = 32 * 1 * 3 * 2 
        
        self.fc_shared = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3) 
        )
        
        self.event_head = nn.Linear(128, num_event_classes)
        self.goal_head = nn.Linear(128, 1)

        # 2. KEY CHANGE: Kaiming (He) Initialization
        # This prevents "Dying Neurons" before the first epoch even finishes
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                # Kaiming Normal is optimal for LeakyReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x input: [Batch, 3, 4, 12, 8]
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.fc_shared(x)
        return self.event_head(x), self.goal_head(x).squeeze(-1)