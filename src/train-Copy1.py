import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import src.config as config
import torch.nn.utils.rnn as rnn_utils
from src.model import (
    TinyCNN_MultiTask_Threat, 
    TinyCNN_MultiTask_Context_Threat, 
)
from src.losses import get_model_criteria, FocalLoss, FocalLossThreat
from src.utils import get_sequence_lengths
import torch.nn as nn

# ----------------------------------------------------------------------
# Helper: Initializes Model and Criteria based on type
# ----------------------------------------------------------------------


def get_threat_model_and_criteria(model_type, event_class_weights, goal_pos_weight, num_context_features=0, loss_type='Focal'):
    """
    Initializes a new _threat model and the corresponding loss criteria.
    
    Parameters:
    - model_type (str): 'base_threat', 'context_threat', 'lstm_threat', or 'fused_threat'.
    - loss_type (str): 'CE' for CrossEntropyLoss or 'Focal' for FocalLoss.
    """
    
    # 1. Initialize Loss Criteria
    if loss_type == 'Focal':
        # Loss for event classification (Multi-class)
        criterion_event = FocalLossThreat(alpha=event_class_weights, gamma=2.0)
    else:
        # Standard Cross-Entropy Loss (default)
        criterion_event = nn.CrossEntropyLoss(weight=event_class_weights)
        
    # Loss for goal prediction (Binary)
    criterion_goal = nn.BCEWithLogitsLoss(pos_weight=goal_pos_weight)

    # 2. Initialize Model Architecture
    
    if model_type == 'base_threat':
        # Static CNN Baseline
        model = TinyCNN_MultiTask_Threat(
            grid_height=config.GRID_HEIGHT,
            grid_width=config.GRID_WIDTH,
            num_event_classes=config.NUM_EVENT_CLASSES
        ).to(config.DEVICE)

    elif model_type == 'context_threat':
        # Static CNN with Context
        if num_context_features == 0:
            raise ValueError("Model type 'context_threat' requires num_context_features > 0.")
            
        model = TinyCNN_MultiTask_Context_Threat(
            grid_height=config.GRID_HEIGHT,
            grid_width=config.GRID_WIDTH,
            num_event_classes=config.NUM_EVENT_CLASSES,
            num_context_features=num_context_features
        ).to(config.DEVICE)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model, criterion_event, criterion_goal


#---------------------------------------------------
# The Updated Threat Models
# ----------------------------------------------

# The Baseline Model
# In src/train.py: Add the new training function

def train_model_base_threat(dataset, event_class_weights, goal_pos_weight, loss_type='CE'):
    """Trains the TinyCNN_MultiTask_Threat model (Static Baseline)."""
    
    model, criterion_event, criterion_goal = get_threat_model_and_criteria(
        'base_threat', event_class_weights, goal_pos_weight, loss_type=loss_type
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(train_loader, desc=f"Base CNN Threat Epoch {epoch+1}")
        for X, event_labels, goal_flags in loop:
            
            X, event_labels, goal_flags = X.to(config.DEVICE), event_labels.to(config.DEVICE), goal_flags.to(config.DEVICE)
            optimizer.zero_grad()
            
            event_logits, goal_logits = model(X)
            
            # Loss Calculation (Multi-task)
            loss_event = criterion_event(event_logits, event_labels)
            
            shot_mask = (event_labels == 2)
            if shot_mask.any():
                loss_shot = criterion_goal(goal_logits[shot_mask], goal_flags[shot_mask])
            else:
                loss_shot = torch.tensor(0.0, device=config.DEVICE)
            
            loss = loss_event + config.LAMBDA_GOAL * loss_shot
            
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item(), event_loss=loss_event.item())
            
    return model



# The Context Model

def train_model_context_threat(dataset, event_class_weights, goal_pos_weight, num_context_features=3):
    """
    Trains the TinyCNN_MultiTask_Context model, handling 4 inputs (X, context_data, labels, flags).
    """
    
    # 1. Initialize Model and Criteria 
    model, criterion_event, criterion_goal = get_threat_model_and_criteria(
        'context_threat', event_class_weights, goal_pos_weight, 
        num_context_features=num_context_features
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(train_loader, desc=f"Context CNN Epoch {epoch+1}")
        # CRITICAL: Unpack 4 items
        for X, context_data, event_labels, goal_flags in loop:
            
            X, context_data, event_labels, goal_flags = \
                X.to(config.DEVICE), context_data.to(config.DEVICE), \
                event_labels.to(config.DEVICE), goal_flags.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass requires both spatial (X) and contextual (context_data) inputs
            event_logits, goal_logits = model(X, context_data)
            
            # --- Loss Calculation (Same as base model) ---
            loss_event = criterion_event(event_logits, event_labels)
            
            shot_mask = (event_labels == 2)
            if shot_mask.any():
                loss_shot = criterion_goal(goal_logits[shot_mask], goal_flags[shot_mask])
            else:
                loss_shot = torch.tensor(0.0, device=config.DEVICE)
            
            loss = loss_event + config.LAMBDA_GOAL * loss_shot
            
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(
                loss=loss.item(), 
                event_loss=loss_event.item(), 
                shot_loss=loss_shot.item() if torch.is_tensor(loss_shot) else loss_shot
            )
            
    return model

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import src.config as config

def train_model_context_threat_ball_vector(dataset, event_class_weights, goal_pos_weight, num_context_features=8):
    """
    Trains the TinyCNN_MultiTask_Context_Ball_Vector model using both spatial grids and 
    the temporal ball trajectory context vector.
    
    Parameters:
        dataset: PitchDataset yielding (X, context_data, event_labels, goal_flags)
        event_class_weights: Weights for Focal/CE loss (imbalance handling)
        goal_pos_weight: Weight for the positive goal class
        num_context_features: Default 8 (current ball x,y + 3 past steps x,y)
    """
    
    # 1. Initialize Model and Criteria 
    # This factory function should return the model instance, 
    # the event criterion (e.g. FocalLoss), and goal criterion (e.g. BCEWithLogits)
    model, criterion_event, criterion_goal = get_threat_model_and_criteria(
        'context_threat', 
        event_class_weights, 
        goal_pos_weight, 
        num_context_features=num_context_features
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    model.to(config.DEVICE)
    model.train()

    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(train_loader, desc=f"Context CNN Epoch {epoch+1}")
        
        for X, context_data, event_labels, goal_flags in loop:
            # Move data to device
            X = X.to(config.DEVICE)                      # [Batch, Channels, H, W]
            context_data = context_data.to(config.DEVICE) # [Batch, num_context_features]
            event_labels = event_labels.to(config.DEVICE) # [Batch]
            goal_flags = goal_flags.to(config.DEVICE)     # [Batch]
            
            optimizer.zero_grad()
            
            # 2. Forward pass: Model accepts spatial and contextual inputs
            event_logits, goal_logits = model(X, context_data)
            
            # 3. Multi-Task Loss Calculation
            # Task A: Event Outcome Classification
            loss_event = criterion_event(event_logits, event_labels)
            
            # Task B: Possession Threat (Only calculated on actual shots)
            # We use a mask to ignore non-shot events for the xG head
            shot_mask = (event_labels == 2)
            if shot_mask.any():
                # Ensure logits and flags are both [N, 1] or [N] for BCE
                # Using .view(-1) to ensure they are the same shape
                loss_shot = criterion_goal(
                    goal_logits[shot_mask].view(-1), 
                    goal_flags[shot_mask].view(-1)
                )
            else:
                loss_shot = torch.tensor(0.0, device=config.DEVICE)
            
            # Total Loss = Classification + Weighted Threat
            loss = loss_event + config.LAMBDA_GOAL * loss_shot
            
            # 4. Backward pass
            loss.backward()
            optimizer.step()
            
            # Update Progress Bar
            loop.set_postfix(
                loss=f"{loss.item():.4f}", 
                ev_loss=f"{loss_event.item():.4f}", 
                sh_loss=f"{loss_shot.item() if torch.is_tensor(loss_shot) else loss_shot:.4f}"
            )
            
    return model