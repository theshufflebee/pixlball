import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------
# Custom Focal Loss (Class definition remains the same)
# -----------------

class FocalLossThreat(nn.Module):
    """
    Implements Focal Loss for multi-class classification.
    
    Focal Loss down-weights the loss assigned to well-classified examples 
    and focuses on hard, misclassified examples (like 'Lose Possession' or 'Shot').
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Parameters:
        - alpha (torch.Tensor, optional): Class weights (e.g., inverse frequency). 
                                          Shape: (num_classes,).
        - gamma (float): Focusing parameter. Higher gamma increases focusing power.
        - reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLossThreat, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # This is the class weight tensor
        self.reduction = reduction

    def forward(self, input, target):
        """
        Input: Raw logits (B, C).
        Target: Class labels (B).
        """
        
        # Calculate Standard Cross-Entropy Loss (unreduced)
        # Use the standard PyTorch function for numerical stability (LogSumExp)
        # Note: F.cross_entropy handles the weight (alpha) internally if provided.
        ce_loss = F.cross_entropy(input, target, weight=self.alpha, reduction='none')
        
        # Calculate p_t (Probability of the correct class)
        # p_t = exp(-ce_loss) is the numerically stable way to calculate p_t from ce_loss
        pt = torch.exp(-ce_loss) 

        # Calculate the Focal Term
        focal_term = (1 - pt) ** self.gamma
        
        # Final Focal Loss
        loss = focal_term * ce_loss

        # Apply Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# -----------------------------------------------
# Loss Initialization Function
# -----------------------------------------------
def get_model_criteria(event_class_weights, goal_pos_weight, loss_type='CE'):
    """
    Returns the loss criteria objects for the multi-task model.

    Args:
        event_class_weights (torch.Tensor): Weights for event CrossEntropyLoss.
        goal_pos_weight (torch.Tensor): Positive weight for Goal BCEWithLogitsLoss.
        loss_type (str): 'Focal' or 'CE' for the event loss.

    Returns:
        tuple: (criterion_event, criterion_goal)
    """
    
    # Event Loss (L1)
    if loss_type == 'Focal':
        # Use your custom FocalLoss for the event classification (recommended for imbalance)
        criterion_event = FocalLossThreat(alpha=event_class_weights, gamma=2)
        #criterion_event = FocalLoss(alpha=event_class_weights, gamma=2)
    else:
        # Use standard CrossEntropyLoss with weights
        criterion_event = nn.CrossEntropyLoss(weight=event_class_weights)

    # Goal Loss (L2)
    # BCEWithLogitsLoss is the standard for binary classification (Goal/No Goal)
    criterion_goal = nn.BCEWithLogitsLoss(pos_weight=goal_pos_weight)
    
    return criterion_event, criterion_goal
