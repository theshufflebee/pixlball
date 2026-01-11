import torch
from torch.utils.data import Dataset
import numpy as np

class PitchDatasetMultiTask(Dataset):
    def __init__(self, nn_layers_df, event_targets, goal_flags):
        
        # Prepare data as a list of stacked NumPy arrays
        layers_list = [
            np.stack([row['ball_layer'], row['teammates_layer'], row['opponents_layer']], axis=0)
            for _, row in nn_layers_df.iterrows()
        ]
        
        # This converts the list of (3, 12, 8) arrays into one large (N, 3, 12, 8) array.
        layers_array = np.array(layers_list, dtype=np.float32)
        
        # Convert the single NumPy array to a PyTorch tensor (fast operation)
        self.layers = torch.tensor(layers_array)
        
        # Targets
        self.event_targets = torch.tensor(event_targets, dtype=torch.long)
        self.goal_flags = torch.tensor(goal_flags, dtype=torch.float32)

    def __len__(self):
        return len(self.event_targets)

    def __getitem__(self, idx):
        return self.layers[idx], self.event_targets[idx], self.goal_flags[idx]
    
    
class ContextPitchDatasetMultiTask(Dataset):
    # Added contextual features
    def __init__(self, nn_layers_df, event_targets, goal_flags, contextual_features_df):
        
        # Spatial Layers
        layers_list = [
            np.stack([row['ball_layer'], row['teammates_layer'], row['opponents_layer']], axis=0)
            for _, row in nn_layers_df.iterrows()
        ]
        layers_array = np.array(layers_list, dtype=np.float32)
        self.layers = torch.tensor(layers_array)
        
        # Convert the context DataFrame values directly to a PyTorch tensor
        context_array = np.array(contextual_features_df.values, dtype=np.float32)
        self.context_features = torch.tensor(context_array)
        
        # Targets
        self.event_targets = torch.tensor(event_targets, dtype=torch.long)
        self.goal_flags = torch.tensor(goal_flags, dtype=torch.float32)

    def __len__(self):
        return len(self.event_targets)

    def __getitem__(self, idx):
        return self.layers[idx], self.context_features[idx], self.event_targets[idx], self.goal_flags[idx]
    

class ContextBallVectorPitchDatasetMultiTask(Dataset):
    def __init__(self, nn_layers_df, event_targets, goal_flags, contextual_features_df):
        # Spatial Grids
        self.ball_layers = np.stack(nn_layers_df['ball_layer'].values)
        self.team_layers = np.stack(nn_layers_df['teammates_layer'].values)
        self.opp_layers = np.stack(nn_layers_df['opponents_layer'].values)
        
        # Context Vector
        self.context_data = contextual_features_df.values.astype(np.float32)
        
        # Targets
        self.event_targets = torch.tensor(event_targets, dtype=torch.long)
        self.goal_targets = torch.tensor(goal_flags, dtype=torch.float32)

    def __len__(self):
        return len(self.event_targets)

    def __getitem__(self, idx):
        grid = np.stack([
            self.ball_layers[idx],
            self.team_layers[idx],
            self.opp_layers[idx]
        ], axis=0)
        
        return (
            torch.tensor(grid, dtype=torch.float32), 
            torch.tensor(self.context_data[idx], dtype=torch.float32),
            self.event_targets[idx],
            self.goal_targets[idx]
        )

    
class VoxelPitchDataset(Dataset):
    def __init__(self, df, voxel_col='temporal_voxel'):
        self.df = df
        
        # Memory: Store as uint8
        # Stack the voxels
        self.voxels = np.stack(df[voxel_col].values).astype(np.uint8)
        
        # Convert targets to tensors
        self.event_targets = torch.tensor(df['nn_target_int'].values, dtype=torch.long)
        self.goal_targets = torch.tensor(df['goal_flag'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.voxels)

    def __getitem__(self, idx):
        # Cast to float32 only for the specific batch item being loaded
        voxel_tensor = torch.from_numpy(self.voxels[idx]).float()
        
        return (
            voxel_tensor, 
            self.event_targets[idx], 
            self.goal_targets[idx]
        )