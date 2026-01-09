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
        
        # --- 1. Spatial Feature Path (CNN) ---
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        
        cnn_output_dim = 16 * grid_height * grid_width
        
        # Intermediate FC layer for CNN output
        self.fc_cnn = nn.Linear(cnn_output_dim, 32)
        
        # --- 2. Shared Layer ---
        self.fc_shared = nn.Linear(32, 32)
        
        # --- 3. Output Heads ---
        self.fc_event = nn.Linear(32, num_event_classes)
        self.fc_goal = nn.Linear(32, 1)

    def forward(self, x):
        # 1. Spatial Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) 
        shared_features = F.relu(self.fc_cnn(x))
        
        # 2. Final Shared Layer
        shared_features = F.relu(self.fc_shared(shared_features))
        
        # 3. Output Predictions
        event_logits = self.fc_event(shared_features)
        goal_logits = self.fc_goal(shared_features).squeeze(-1)
        
        return event_logits, goal_logits
    
# The Context CNN
#----------------------------------------
class TinyCNN_MultiTask_Context_Threat(nn.Module):
    def __init__(self, grid_height=12, grid_width=8, num_event_classes=3, num_context_features=3):
        super().__init__()
        
        # --- 1. Spatial Feature Path (CNN) ---
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        
        cnn_output_dim = 16 * grid_height * grid_width
        
        # Intermediate FC layer for CNN output
        self.fc_cnn = nn.Linear(cnn_output_dim, 32)
        
        # --- 2. Contextual Feature Path (MLP) ---
        self.fc_context = nn.Linear(num_context_features, 16) 
        
        # --- 3. Combined Shared Layer ---
        # Input size = CNN output (32) + Context output (16) = 48
        self.fc_shared_combined = nn.Linear(32 + 16, 32)
        
        # --- 4. Output Heads ---
        self.fc_event = nn.Linear(32, num_event_classes)
        self.fc_goal = nn.Linear(32, 1)

    def forward(self, x, context_data):
        # 1. Spatial Feature Extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) 
        cnn_features = F.relu(self.fc_cnn(x)) # (B, 32)
        
        # 2. Contextual Feature Processing
        context_features = F.relu(self.fc_context(context_data)) # (B, 16)
        
        # 3. Concatenate Features 
        combined_features = torch.cat((cnn_features, context_features), dim=1) # (B, 48)
        

        # 4. Final Shared Layer
        shared_features = F.relu(self.fc_shared_combined(combined_features)) # (B, 32)
        
        # 5. Output Predictions
        event_logits = self.fc_event(shared_features)
        goal_logits = self.fc_goal(shared_features).squeeze(-1)
        
        return event_logits, goal_logits

#---------------------------------------------------------------
# NNs with temporal Context
#-----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    
class Tiny3DCNN_MultiTask(nn.Module):
    def __init__(self, num_event_classes=3):
        super(Tiny3DCNN_MultiTask, self).__init__()
        
        # Input shape: (Batch, 3, 4, 12, 8)
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),  # CRITICAL: Stabilizes the voxel activations
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(1, 2, 2)), 
            
            # Block 2
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),  # CRITICAL: Prevents gradient collapse
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        
        self.flatten_size = 32 * 2 * 3 * 2 # 384
        
        self.fc_shared = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.BatchNorm1d(128), # Stabilize the dense layer
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4)      # Increased slightly to prevent "lazy" learning
        )
        
        self.event_head = nn.Linear(128, num_event_classes)
        
        self.goal_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1) # Removed Sigmoid here if using BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.fc_shared(x)
        
        event_logits = self.event_head(x)
        goal_logits = self.goal_head(x) # Outputting raw logits is better for torch loss
        
        return event_logits, goal_logits.squeeze(-1)