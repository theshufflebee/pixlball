
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