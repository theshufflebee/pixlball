import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

import src.config as config
from src.model import (
    TinyCNN_MultiTask_Threat, 
    TinyCNN_MultiTask_Context_Threat, 
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


# ---------------------------------------------------------
# Training Functions
# ---------------------------------------------------------

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


def train_3d_model(
    dataset, 
    event_class_weights, 
    goal_pos_weight, 
    epochs=15, 
    batch_size=32, 
    lr=0.001
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Tiny3DCNN_MultiTask().to(DEVICE)
    
    # Define Losses
    criterion_event = nn.CrossEntropyLoss(weight=event_class_weights.to(DEVICE))
    criterion_goal = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([goal_pos_weight]).to(DEVICE))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for voxels, event_targets, goal_targets in dataloader:
            voxels = voxels.to(DEVICE)
            event_targets = event_targets.to(DEVICE)
            goal_targets = goal_targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass (3D CNN expects 4D input: C, T, H, W)
            event_logits, goal_prob = model(voxels)
            
            # Calculate Losses
            loss_event = criterion_event(event_logits, event_targets)
            # Use raw logits for BCEWithLogits if your head doesn't have Sigmoid, 
            # OR adjust if your model already has Sigmoid (as defined in previous step)
            loss_goal = nn.functional.binary_cross_entropy(goal_prob, goal_targets)
            
            loss = loss_event + (2.0 * loss_goal) # Weight goal loss higher
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
    return model