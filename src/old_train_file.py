import torch
from torch.utils.data import Dataset
import numpy as np

class PitchDatasetMultiTask(Dataset):
    def __init__(self, nn_layers_df, event_targets, goal_flags):
        
        # 1. Prepare data as a list of stacked NumPy arrays
        layers_list = [
            np.stack([row['ball_layer'], row['teammates_layer'], row['opponents_layer']], axis=0)
            for _, row in nn_layers_df.iterrows()
        ]
        
        # 2. CRITICAL FIX: Convert the list to a single NumPy array (fast operation)
        # This converts the list of (3, 12, 8) arrays into one large (N, 3, 12, 8) array.
        layers_array = np.array(layers_list, dtype=np.float32)
        
        # 3. Convert the single NumPy array to a PyTorch tensor (fast operation)
        self.layers = torch.tensor(layers_array)
        
        # Targets (Ensure goal_flags is float for BCEWithLogitsLoss)
        self.event_targets = torch.tensor(event_targets, dtype=torch.long)
        self.goal_flags = torch.tensor(goal_flags, dtype=torch.float32)

    def __len__(self):
        return len(self.event_targets)

    def __getitem__(self, idx):
        return self.layers[idx], self.event_targets[idx], self.goal_flags[idx]


class TemporalPitchDataset(Dataset):
    def __init__(self, windows, event_labels, goal_flags):
        """
        windows: [num_events, T, 4, 12, 8]  (4 = ball, teammates, opponents, mask)
        """
        
        # 1. CRITICAL FIX: Convert the list of 5D arrays into a single, contiguous NumPy array.
        windows_array = np.array(windows, dtype=np.float32)
        
        # 2. Convert the single NumPy array to a PyTorch tensor (fast operation)
        self.windows = torch.tensor(windows_array)
        
        # Ensure goal_flags is float for BCEWithLogitsLoss
        self.event_labels = torch.tensor(event_labels, dtype=torch.long)
        self.goal_flags = torch.tensor(goal_flags, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Returns [T, 4, 12, 8], event_label (int), goal_flag (float)
        return self.windows[idx], self.event_labels[idx], self.goal_flags[idx]
    
    
class ContextPitchDatasetMultiTask(Dataset):
    # CRITICAL: Added contextual_features_df to the signature
    def __init__(self, nn_layers_df, event_targets, goal_flags, contextual_features_df):
        
        # --- 1. Spatial Layers (Same as before) ---
        layers_list = [
            np.stack([row['ball_layer'], row['teammates_layer'], row['opponents_layer']], axis=0)
            for _, row in nn_layers_df.iterrows()
        ]
        layers_array = np.array(layers_list, dtype=np.float32)
        self.layers = torch.tensor(layers_array)
        
        # --- 2. Contextual Features (NEW) ---
        # Convert the context DataFrame values directly to a PyTorch tensor
        context_array = np.array(contextual_features_df.values, dtype=np.float32)
        self.context_features = torch.tensor(context_array)
        
        # --- 3. Targets (Same as before) ---
        self.event_targets = torch.tensor(event_targets, dtype=torch.long)
        self.goal_flags = torch.tensor(goal_flags, dtype=torch.float32)

    def __len__(self):
        return len(self.event_targets)

    def __getitem__(self, idx):
        # CRITICAL: Return four items now
        return self.layers[idx], self.context_features[idx], self.event_targets[idx], self.goal_flags[idx]
    

# --- New Dataset for LSTM Fused Model ---
class FusionPitchDataset(Dataset):
    def __init__(self, windows, contextual_features, event_labels, goal_flags):
        """
        windows: [num_events, T, 4, 12, 8] (Spatial Sequence)
        contextual_features: [num_events, T, num_context_features] (Context Sequence)
        """
        
        # 1. Spatial Windows 
        windows_array = np.array(windows, dtype=np.float32)
        self.windows = torch.tensor(windows_array)
        
        # 2. Contextual Features Sequence
        context_array = np.array(contextual_features, dtype=np.float32)
        self.context_features = torch.tensor(context_array)
        
        # 3. Targets
        self.event_labels = torch.tensor(event_labels, dtype=torch.long)
        self.goal_flags = torch.tensor(goal_flags, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # CRITICAL: Returns 4 items: Spatial Seq, Context Seq, Event Label, Goal Flag
        return (self.windows[idx], self.context_features[idx], 
                self.event_labels[idx], self.goal_flags[idx])