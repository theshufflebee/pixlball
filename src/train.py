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
    # 1. Initialize Loss Criteria
    if loss_type == 'Focal':
        criterion_event = FocalLossThreat(alpha=event_class_weights, gamma=2.0)
    else:
        criterion_event = nn.CrossEntropyLoss(weight=event_class_weights)
        
    # Ensure goal_pos_weight is a tensor for BCEWithLogitsLoss
    if not isinstance(goal_pos_weight, torch.Tensor):
        goal_pos_weight = torch.tensor([goal_pos_weight], device=config.DEVICE)
        
    criterion_goal = nn.BCEWithLogitsLoss(pos_weight=goal_pos_weight)

    # 2. Initialize Model Architecture
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
            # Step 1: Dynamic Unpacking
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

            # Step 2: Training Step
            optimizer.zero_grad()
            
            # Forward pass using unpacked arguments
            event_logits, goal_logits = model(*forward_args)
            
            # Event Loss
            loss_event = criterion_event(event_logits, event_labels)
            
            # Masked Goal Loss (Only for Shots/Class 2)
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

def train_model_base_threat(dataset, event_class_weights, goal_pos_weight, loss_type='Focal'):
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
            
            loss_event = criterion_event(event_logits, event_labels)
            
            shot_mask = (event_labels == 2)
            if shot_mask.any():
                loss_shot = criterion_goal(goal_logits[shot_mask].view(-1), goal_flags[shot_mask].view(-1))
            else:
                loss_shot = torch.tensor(0.0, device=config.DEVICE)
            
            loss = loss_event + config.LAMBDA_GOAL * loss_shot
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=f"{loss.item():.4f}", ev_loss=f"{loss_event.item():.4f}")
            
    return model


def train_model_context_threat(dataset, event_class_weights, goal_pos_weight, num_context_features=8, loss_type='Focal'):
    """
    Unified training function for Contextual CNNs. 
    Handles any size of context_data (e.g., the 8-feature ball trajectory vector).
    """
    model, criterion_event, criterion_goal = get_threat_model_and_criteria(
        'context_threat', 
        event_class_weights, 
        goal_pos_weight, 
        num_context_features=num_context_features,
        loss_type=loss_type
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(train_loader, desc=f"Context CNN Epoch {epoch+1}")
        
        for X, context_data, event_labels, goal_flags in loop:
            X = X.to(config.DEVICE)
            context_data = context_data.to(config.DEVICE)
            event_labels = event_labels.to(config.DEVICE)
            goal_flags = goal_flags.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass: uses spatial (X) and temporal context (context_data)
            event_logits, goal_logits = model(X, context_data)
            
            loss_event = criterion_event(event_logits, event_labels)
            
            # Threat loss only on shots (Class 2)
            shot_mask = (event_labels == 2)
            if shot_mask.any():
                loss_shot = criterion_goal(
                    goal_logits[shot_mask].view(-1), 
                    goal_flags[shot_mask].view(-1)
                )
            else:
                loss_shot = torch.tensor(0.0, device=config.DEVICE)
            
            loss = loss_event + config.LAMBDA_GOAL * loss_shot
            
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(
                loss=f"{loss.item():.4f}", 
                ev_loss=f"{loss_event.item():.4f}", 
                sh_loss=f"{loss_shot.item():.4f}"
            )
            
    return model


def train_3d_model_alt(
    dataset, 
    event_class_weights, 
    goal_pos_weight, 
    epochs=config.NUM_EPOCHS, 
    batch_size=config.BATCH_SIZE, 
    lr=config.LR_3D,
    loss_type='Focal'
):
    """
    Trains the Tiny3DCNN_MultiTask model using the same masked loss logic
    as the 2D threat models.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Tiny3DCNN_MultiTask().to(config.DEVICE)
    
    # 1. Standardize Loss Selection (Match the other functions)
    if loss_type == 'Focal':
        criterion_event = FocalLossThreat(alpha=event_class_weights.to(config.DEVICE), gamma=2.0)
    else:
        criterion_event = nn.CrossEntropyLoss(weight=event_class_weights.to(config.DEVICE))
    
    # Ensure goal_pos_weight is a tensor for BCEWithLogitsLoss
    if not isinstance(goal_pos_weight, torch.Tensor):
        goal_pos_weight = torch.tensor([goal_pos_weight], device=config.DEVICE)
        
    criterion_goal = nn.BCEWithLogitsLoss(pos_weight=goal_pos_weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"3D CNN Threat Epoch {epoch+1}")
        total_epoch_loss = 0
        
        for voxels, event_targets, goal_targets in loop:
            voxels = voxels.to(config.DEVICE)
            event_targets = event_targets.to(config.DEVICE)
            goal_targets = goal_targets.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass (Input: Batch, Channels, Time, Height, Width)
            event_logits, goal_logits = model(voxels)
            
            # 2. Event Head Loss
            loss_event = criterion_event(event_logits, event_targets)
            
            # 3. Masked Goal Head Loss (CRITICAL CHANGE)
            # Only calculate xG loss for events that are actually shots (Class 2)
            shot_mask = (event_targets == 2)
            if shot_mask.any():
                loss_goal = criterion_goal(
                    goal_logits[shot_mask].view(-1), 
                    goal_targets[shot_mask].view(-1)
                )
            else:
                # If no shots in this batch, loss is zero
                loss_goal = torch.tensor(0.0, device=config.DEVICE)
            
            # 4. Total Loss with weighting (using config.LAMBDA_GOAL)
            loss = loss_event + config.LAMBDA_GOAL * loss_goal
            
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            
            # 5. Update Progress Bar
            loop.set_postfix(
                loss=f"{loss.item():.4f}", 
                ev_loss=f"{loss_event.item():.4f}", 
                gl_loss=f"{loss_goal.item():.4f}"
            )
            
        avg_loss = total_epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
        
    return model

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
    # 1. Use the helper to get the model and criteria (Add '3d_threat' to your helper if needed)
    # For now, we will instantiate manually but safely:
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