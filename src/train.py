import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

import src.config as config
from src.model import (
    TinyCNN_MultiTask_Threat, 
    TinyCNN_MultiTask_Context_Threat,
    Tiny3DCNN_MultiTask,
    TinyCNN_MultiTask_Context_Ball_Vector 
)
from src.losses import FocalLossThreat

# ----------------------------------------------------------------------
# Helper: Initializes Model and Criteria based on type
# ----------------------------------------------------------------------

def get_threat_model_and_criteria(model_type, event_class_weights, goal_pos_weight, num_context_features=0, loss_type='Focal'):
    """
    Initializes a new _threat model and the corresponding loss criteria.
    """
    # Initialize Loss Criteria
    if loss_type == 'Focal':
        criterion_event = FocalLossThreat(alpha=event_class_weights, gamma=2.0)
    else:
        criterion_event = nn.CrossEntropyLoss(weight=event_class_weights)
        
    # Ensure goal_pos_weight is a tensor for BCEWithLogitsLoss
    if not isinstance(goal_pos_weight, torch.Tensor):
        goal_pos_weight = torch.tensor([goal_pos_weight], device=config.DEVICE)
        
    criterion_goal = nn.BCEWithLogitsLoss(pos_weight=goal_pos_weight)

    # Initialize Model Architecture
    if model_type == 'base_threat':
        model = TinyCNN_MultiTask_Threat(
            grid_height=config.GRID_HEIGHT,
            grid_width=config.GRID_WIDTH,
            num_event_classes=config.NUM_EVENT_CLASSES
        ).to(config.DEVICE)

    elif model_type == 'context_threat':
        if num_context_features == 0:
            raise ValueError("Model type 'context_threat' requires num_context_features > 0.")
            
        model = TinyCNN_MultiTask_Context_Threat(
            grid_height=config.GRID_HEIGHT,
            grid_width=config.GRID_WIDTH,
            num_event_classes=config.NUM_EVENT_CLASSES,
            num_context_features=num_context_features
        ).to(config.DEVICE)
        
    elif model_type == '3d_threat':
        model = Tiny3DCNN_MultiTask(
            num_event_classes=config.NUM_EVENT_CLASSES
        ).to(config.DEVICE)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model, criterion_event, criterion_goal



import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import src.config as config

def train_multi_task_model(
    model, 
    train_loader, 
    criterion_event, 
    criterion_goal, 
    epochs = config.NUM_EPOCHS,
    model_name="Model",
    lr = config.LR
):
    """
    A single unified training function for:
    - Base CNN (X)
    - Context CNN (X, context)
    - 3D CNN (voxels)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(config.DEVICE)
    model.train()

    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}")
        
        for batch in loop:
            # Dynamic Unpacking
            # Handles (X, ev, gl) OR (X, ctx, ev, gl)
            if len(batch) == 3:
                inputs, event_labels, goal_flags = batch
                inputs = inputs.to(config.DEVICE)
                forward_args = [inputs]
            else:
                inputs, context, event_labels, goal_flags = batch
                inputs = inputs.to(config.DEVICE)
                context = context.to(config.DEVICE)
                forward_args = [inputs, context]

            event_labels = event_labels.to(config.DEVICE)
            goal_flags = goal_flags.to(config.DEVICE)

            # Training Step
            optimizer.zero_grad()
            
            # Forward pass using unpacked arguments
            event_logits, goal_logits = model(*forward_args)
            
            # Event Loss
            loss_event = criterion_event(event_logits, event_labels)
            
            # Masked Goal Loss (Only for Shots/Class 2)
            # Model only learns from true goal observations
            shot_mask = (event_labels == 2)
            if shot_mask.any():
                loss_goal = criterion_goal(
                    goal_logits[shot_mask].view(-1), 
                    goal_flags[shot_mask].view(-1)
                )
            else:
                loss_goal = torch.tensor(0.0, device=config.DEVICE)
            
            # Combined Loss
            loss = loss_event + config.LAMBDA_GOAL * loss_goal
            
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=f"{loss.item():.4f}", ev_loss=f"{loss_event.item():.4f}")

    return model


# ---------------------------------------------------------
# Training Functions
# ---------------------------------------------------------


def train_3d_model(
    dataset, 
    event_class_weights, 
    goal_pos_weight, 
    epochs=config.NUM_EPOCHS, 
    batch_size=config.BATCH_SIZE, 
    lr=config.LR,
    loss_type='Focal'
):
    """
    Fixed 3D training function. 
    Uses the standardized helper to avoid instantiation type errors.
    """
    # Instantiate manually
    model = Tiny3DCNN_MultiTask(num_event_classes=config.NUM_EVENT_CLASSES).to(config.DEVICE)
    
    if loss_type == 'Focal':
        criterion_event = FocalLossThreat(alpha=event_class_weights.to(config.DEVICE), gamma=2.0)
    else:
        criterion_event = nn.CrossEntropyLoss(weight=event_class_weights.to(config.DEVICE))
    
    criterion_goal = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([goal_pos_weight]).to(config.DEVICE))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"3D CNN Epoch {epoch+1}")
        for voxels, event_targets, goal_targets in loop:
            voxels, event_targets, goal_targets = (
                voxels.to(config.DEVICE), 
                event_targets.to(config.DEVICE), 
                goal_targets.to(config.DEVICE)
            )
            
            optimizer.zero_grad()
            event_logits, goal_logits = model(voxels)
            
            # --- Standardized Loss Handling ---
            loss_event = criterion_event(event_logits, event_targets)
            
            # Masked goal loss (only on class 2)
            shot_mask = (event_targets == 2)
            if shot_mask.any():
                loss_goal = criterion_goal(
                    goal_logits[shot_mask].view(-1), 
                    goal_targets[shot_mask].view(-1)
                )
            else:
                loss_goal = torch.tensor(0.0, device=config.DEVICE)
            
            loss = loss_event + config.LAMBDA_GOAL * loss_goal
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=f"{loss.item():.4f}", ev_loss=f"{loss_event.item():.4f}")
            
    return model