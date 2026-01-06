from collections import Counter
import numpy as np
import torch

def get_sequence_lengths(input_tensor):
    """Calculates the true length of the sequence based on the mask channel (4th channel).
    
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
    Calculates weights for both Event and Goal loss heads.
    
    Args:
        nn_dataset (pandas.DataFrame): Dataset containing 'nn_target_int' and 'goal_flag' columns
        device: torch.device (e.g., 'cuda' or 'cpu')
        
    Returns:
        tuple: (event_weights_tensor, goal_pos_weight_tensor)
    """
    # Ensure labels are in the correct format
    event_targets = nn_dataset['nn_target_int'].values   # 0=keep, 1=lose, 2=shot (int)
    # CRITICAL: Goal flags must be float for BCEWithLogitsLoss
    goal_flags = nn_dataset['goal_flag'].values.astype(np.float32) 
        
    # 1. Event Head Weights (Inverse Frequency)
    # total / (num_classes * class_count)
    event_counts = Counter(event_targets)
    total_events = len(event_targets)
    num_classes = len(event_counts)
    
    event_weights = [
        total_events / (num_classes * event_counts.get(c, 1)) 
        for c in range(num_classes)
    ]
    event_weights_tensor = torch.tensor(event_weights, dtype=torch.float32).to(device)
    
    # 2. Goal Head Weight (Manual Stable Weight)
    # Using the 5.0 constant you specified as stable for your model
    STABLE_GOAL_POS_WEIGHT = 5.0
    goal_pos_weight_tensor = torch.tensor([STABLE_GOAL_POS_WEIGHT], dtype=torch.float32).to(device)
    
    return event_weights_tensor, goal_pos_weight_tensor