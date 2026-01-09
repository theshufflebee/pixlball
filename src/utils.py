from collections import Counter
import numpy as np
import torch
import random
import os

def enforce_replicability(seed=42):
    """
    This function sets all the needed seeds to ensure replicability over the whole project
    """
    # Basic Python and OS
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # Hardware Determinism (CRITICAL)
    torch.backends.cudnn.deterministic = True  # Forces deterministic algorithms
    torch.backends.cudnn.benchmark = False      # Disables auto-tuner for consistency
    
    # Force deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)

def get_sequence_lengths(input_tensor):
    """
    Calculates the true length of the sequence based on the mask channel (4th channel).
    
    Args:
        input_tensor (torch.Tensor): Shape (B, T, 4, H, W).
        
    Returns:
        torch.Tensor: Tensor of shape (B) containing the true sequence length for each sample.
    """
    # The mask channel is the last (4th) channel. 
    # We take the mean across H and W for the mask channel.
    # The result is (B, T, 1) or (B, T) where 1 means real data, 0 means padding.
    mask = input_tensor[:, :, 3, :, :].mean(dim=(-2, -1)) # shape (B, T)
    
    # Sum across the Time dimension (T) to get the true length
    lengths = mask.sum(dim=1).long()
    
    # Lengths cannot be zero since the current event (T_0) is always present (length >= 1)
    return lengths


def get_multitask_loss_weights(nn_dataset, device):
    """
    Calculates balanced, smoothed weights for Event and Goal loss heads.
    Uses Square-Root scaling to prevent majority class suppression.
    """
    event_targets = nn_dataset['nn_target_int'].values   # 0=keep, 1=lose, 2=shot
    
    # Event Head Weights (Smoothed Inverse Frequency)
    event_counts = Counter(event_targets)
    total_events = len(event_targets)
    num_classes = len(event_counts)
    
    # Calculate sqrt weights to dampen extreme imbalances
    raw_weights = [
        np.sqrt(total_events / event_counts.get(c, 1)) 
        for c in range(num_classes)
    ]
    
    # Normalize: Ensure the average weight is 1.0 to keep gradients stable
    normalization_factor = num_classes / sum(raw_weights)
    event_weights = [w * normalization_factor for w in raw_weights]
    
    event_weights_tensor = torch.tensor(event_weights, dtype=torch.float32).to(device)
    
    # Goal Head Weight 
    # Moving higher helps the model focus more on the rare 'Goal' outcome
    STABLE_GOAL_POS_WEIGHT = 5.0
    goal_pos_weight_tensor = torch.tensor([STABLE_GOAL_POS_WEIGHT], dtype=torch.float32).to(device)
    
    # Print weights so you can see the change in your logs
    print(f"Calculated Event Weights: {event_weights}")
    print(f"Goal Pos Weight: {STABLE_GOAL_POS_WEIGHT}")
    
    return event_weights_tensor, goal_pos_weight_tensor


def perform_replicable_split(df, train_ratio=0.8, seed=42):
    
    """
    This function does the train, test split for the Data. It splits on Matches, not on events to avoid
    temporal leakage.
    """
    
    # Get unique Match IDs
    unique_matches = df['match_id'].unique()
    
    # Sort them first to ensure order is deterministic before shuffling
    unique_matches.sort() 
    
    # Shuffle using your fixed seed
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_matches)
    
    # Split the match IDs
    split_idx = int(len(unique_matches) * train_ratio)
    train_match_ids = unique_matches[:split_idx]
    test_match_ids = unique_matches[split_idx:]
    
    # Create the dataframes
    train_df = df[df['match_id'].isin(train_match_ids)].copy()
    test_df = df[df['match_id'].isin(test_match_ids)].copy()
    
    print(f"Replicable Split: {len(train_match_ids)} Train Matches, {len(test_match_ids)} Test Matches")
    return train_df, test_df