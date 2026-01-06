import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
# Assuming get_sequence_lengths is available from src.utils
from src.utils import get_sequence_lengths 
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import numpy as np

# -------------------------------------------------
# New Threat Models
#-------------------------------------------

# Base Function
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    confusion_matrix, 
    classification_report, 
    roc_auc_score # Required for probability metrics
)

def _calculate_metrics_threat(all_event_preds, all_event_labels, all_goal_preds, all_goal_labels, all_event_probs, all_goal_probs):
    """
    Calculates and prints all event, goal metrics, including probability-based metrics (AUC).

    Takes 6 arguments: preds/labels and the two probability arrays (event_probs, goal_probs).
    """

    # --- 1. Goal metrics (Conditional on Shot) ---
    # Check if goal labels are present and not all zero
    is_goal_data_valid = (
        all_goal_labels is not None and 
        len(all_goal_labels) > 0 and 
        np.sum(all_goal_labels) > 0 and 
        np.sum(all_goal_labels) < len(all_goal_labels) # Ensure there's more than one class
    )
    
    if is_goal_data_valid:
        goal_acc = accuracy_score(all_goal_labels, all_goal_preds)
        goal_bal_acc = balanced_accuracy_score(all_goal_labels, all_goal_preds)
        goal_cm = confusion_matrix(all_goal_labels, all_goal_preds)
        goal_cr = classification_report(all_goal_labels, all_goal_preds, digits=2, zero_division=0)
        
        # Calculate AUC (Standard metric for xG models)
        auc_score = roc_auc_score(all_goal_labels, all_goal_probs)
    else:
        # Default/placeholder values if no goal data or only one class is present
        goal_acc, goal_bal_acc, goal_cm, goal_cr, auc_score = 0.0, 0.5, "N/A", "N/A", "N/A"
        if len(all_goal_labels) > 0:
             print("Warning: Goal AUC cannot be calculated (only one class present in goal labels).")


    # --- 2. Event metrics (Classification) ---
    event_acc = accuracy_score(all_event_labels, all_event_preds)
    event_bal_acc = balanced_accuracy_score(all_event_labels, all_event_preds)
    event_cm = confusion_matrix(all_event_labels, all_event_preds)
    event_cr = classification_report(all_event_labels, all_event_preds, digits=2)

    # --- 3. Print Results ---
    print("\n--- Event Outcome Metrics ---")
    print("Event Accuracy:", event_acc)
    print("Event Balanced Accuracy:", event_bal_acc)
    print("Event Confusion Matrix:\n", event_cm)
    print(event_cr)

    print("\n--- Goal Prediction (xG) Metrics ---")
    print("Goal Accuracy:", goal_acc)
    print("Goal Balanced Accuracy:", goal_bal_acc)
    print("Goal AUC-ROC Score:", auc_score)
    print("Goal Confusion Matrix:\n", goal_cm)
    print(goal_cr)

    # --- 4. Return Results (Including probabilities) ---
    return {
        "event_preds": all_event_preds, "event_labels": all_event_labels,
        "event_probs": all_event_probs,
        "goal_preds": all_goal_preds, "goal_labels": all_goal_labels,
        "goal_probs": all_goal_probs,
        "goal_auc": auc_score
    }


# The Baseline Model
# ------------------------------------------------------------------
def evaluate_model_base_threat(model, dataset, batch_size=16):
    """
    Evaluates the Static CNN Baseline model (4D input), returning probabilities.
    """
    
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_event_preds, all_event_labels, all_goal_preds, all_goal_labels = [], [], [], []
    all_event_probs, all_goal_probs = [], []

    with torch.no_grad():
        for X, event_labels, goal_flags in loader:
            # 4D input: (B, C, H, W). No sequence handling required.
            X, event_labels, goal_flags = X.to(device), event_labels.to(device), goal_flags.to(device)
            
            # Simple Forward Pass (Model takes only X)
            event_logits, shot_logits = model(X)
            
            # --- Prediction Aggregation ---
            
            # 1. Event probabilities and predictions (P_outcome)
            event_probs = F.softmax(event_logits, dim=1)
            all_event_probs.append(event_probs.cpu().numpy())
            
            event_pred = torch.argmax(event_logits, dim=1)
            all_event_preds.extend(event_pred.cpu().numpy())
            all_event_labels.extend(event_labels.cpu().numpy())

            # 2. Goal probabilities and predictions (xG)
            shot_mask = (event_labels == 2)
            if shot_mask.any():
                goal_probs = torch.sigmoid(shot_logits[shot_mask])
                all_goal_probs.extend(goal_probs.cpu().numpy())
                
                goal_pred = (goal_probs > 0.5).long()
                all_goal_preds.extend(goal_pred.cpu().numpy())
                all_goal_labels.extend(goal_flags[shot_mask].cpu().numpy())

    # --- FINAL CONCATENATION/CONVERSION (Robust Fix) ---
    # Convert list of (B, 3) arrays into one large (N, 3) array
    if all_event_probs:
        all_event_probs = np.concatenate(all_event_probs, axis=0)
    else:
        all_event_probs = np.array([], dtype=np.float32).reshape(0, 3)

    # Convert list of floats into one large (N_shots) array
    all_goal_probs = np.array(all_goal_probs, dtype=np.float32)

    # Pass all 6 arguments, including the probability arrays, to the metric helper
    return _calculate_metrics_threat(all_event_preds, all_event_labels, all_goal_preds, all_goal_labels, all_event_probs, all_goal_probs)

# The Context Model
# ------------------------------------------------------------------

def evaluate_model_context_threat(model, dataset, batch_size=16):
    """
    Evaluates the Static CNN model with contextual features (4 inputs: X, context_data, labels, flags), 
    returning raw probabilities for both event classification and goal prediction.
    """
    
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    # Lists to collect results
    all_event_preds, all_event_labels, all_goal_preds, all_goal_labels = [], [], [], []
    all_event_probs, all_goal_probs = [], []

    with torch.no_grad():
        for X, context_data, event_labels, goal_flags in loader:
            
            # Send all four tensors to the appropriate device
            X, context_data, event_labels, goal_flags = (
                X.to(device), context_data.to(device), 
                event_labels.to(device), goal_flags.to(device)
            )
            
            # Forward Pass: Requires both spatial (X) and contextual (context_data) inputs
            event_logits, shot_logits = model(X, context_data)
            
            # --- Prediction Aggregation ---
            
            # 1. Event probabilities and predictions (P_outcome)
            # Apply Softmax to convert logits to [P(Keep), P(Lose), P(Shot)]
            event_probs = F.softmax(event_logits, dim=1)
            all_event_probs.append(event_probs.cpu().numpy())
            
            event_pred = torch.argmax(event_logits, dim=1)
            all_event_preds.extend(event_pred.cpu().numpy())
            all_event_labels.extend(event_labels.cpu().numpy())

            # 2. Goal probabilities and predictions (xG)
            shot_mask = (event_labels == 2)
            if shot_mask.any():
                # Apply Sigmoid to convert logit to P(Goal)
                goal_probs = torch.sigmoid(shot_logits[shot_mask])
                all_goal_probs.extend(goal_probs.cpu().numpy()) # List of floats
                
                goal_pred = (goal_probs > 0.5).long()
                all_goal_preds.extend(goal_pred.cpu().numpy())
                all_goal_labels.extend(goal_flags[shot_mask].cpu().numpy())

    # --- FINAL CONCATENATION/CONVERSION ---
    
    # 1. Event probabilities (list of (B, 3) arrays)
    if all_event_probs:
        all_event_probs = np.concatenate(all_event_probs, axis=0)
    else:
        all_event_probs = np.array([], dtype=np.float32).reshape(0, 3)

    # 2. Goal probabilities (list of floats)
    # CRITICAL FIX: Convert the list of floats directly to a single 1D array.
    all_goal_probs = np.array(all_goal_probs, dtype=np.float32)

    return _calculate_metrics_threat(all_event_preds, all_event_labels, all_goal_preds, all_goal_labels, all_event_probs, all_goal_probs)

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, 
    classification_report, roc_auc_score
)


def evaluate_model_context_threat_ball_vector(model, dataset, num_context_features=8):
    """
    Evaluates the Contextual CNN on both Event Classification and Goal Prediction.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    all_event_preds = []
    all_event_labels = []
    all_goal_preds = []
    all_goal_labels = []

    with torch.no_grad():
        for X, context_data, event_labels, goal_flags in loader:
            X = X.to(config.DEVICE)
            context_data = context_data.to(config.DEVICE)
            
            # Forward pass
            event_logits, goal_logits = model(X, context_data)
            
            # 1. Process Event Classification (Keep/Lose/Shot)
            event_preds = torch.argmax(event_logits, dim=1)
            all_event_preds.extend(event_preds.cpu().numpy())
            all_event_labels.extend(event_labels.numpy())
            
            # 2. Process Goal Prediction (only for events that are actually shots)
            # We filter by the ground truth 'Shot' label (Class 2)
            shot_mask = (event_labels == 2)
            if shot_mask.any():
                # Apply sigmoid if the model doesn't include it in the head
                # Use .view(-1) for consistent shapes
                probs = torch.sigmoid(goal_logits[shot_mask]).view(-1)
                all_goal_preds.extend(probs.cpu().numpy())
                all_goal_labels.extend(goal_flags[shot_mask].numpy())

    # --- Compute Event Metrics ---
    event_acc = accuracy_score(all_event_labels, all_event_preds)
    event_bal_acc = balanced_accuracy_score(all_event_labels, all_event_preds)
    event_cm = confusion_matrix(all_event_labels, all_event_preds)
    
    print("\n--- Event Outcome Metrics ---")
    print(f"Event Accuracy: {event_acc}")
    print(f"Event Balanced Accuracy: {event_bal_acc}")
    print(f"Event Confusion Matrix:\n {event_cm}")
    print(classification_report(all_event_labels, all_event_preds))

    # --- Compute Goal Prediction (xG) Metrics ---
    if len(all_goal_labels) > 0:
        goal_preds_binary = [1 if p > 0.5 else 0 for p in all_goal_preds]
        goal_acc = accuracy_score(all_goal_labels, goal_preds_binary)
        goal_bal_acc = balanced_accuracy_score(all_goal_labels, goal_preds_binary)
        
        # AUC-ROC requires probability scores
        try:
            goal_auc = roc_auc_score(all_goal_labels, all_goal_preds)
        except ValueError:
            goal_auc = 0.0 # Handle cases with only one class in batch
            
        goal_cm = confusion_matrix(all_goal_labels, goal_preds_binary)

        print("\n--- Goal Prediction (xG) Metrics ---")
        print(f"Goal Accuracy: {goal_acc}")
        print(f"Goal Balanced Accuracy: {goal_bal_acc}")
        print(f"Goal AUC-ROC Score: {goal_auc}")
        print(f"Goal Confusion Matrix:\n {goal_cm}")
        print(classification_report(all_goal_labels, goal_preds_binary))
    else:
        print("\nNo shots found in evaluation set.")

    return {
        "event_bal_acc": event_bal_acc,
        "goal_auc": goal_auc if len(all_goal_labels) > 0 else 0
    }


def evaluate_3d_model(model, dataset):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    
    all_event_preds = []
    all_event_targets = []
    all_goal_probs = []
    all_goal_targets = []
    
    with torch.no_grad():
        for voxels, event_targets, goal_targets in dataloader:
            voxels = voxels.to(DEVICE)
            
            event_logits, goal_probs = model(voxels)
            
            preds = torch.argmax(event_logits, dim=1)
            
            all_event_preds.extend(preds.cpu().numpy())
            all_event_targets.extend(event_targets.numpy())
            all_goal_probs.extend(goal_probs.cpu().numpy())
            all_goal_targets.extend(goal_targets.numpy())
            
    # Calculate Metrics (Accuracy, AUC, etc.)
    # ... [Same logic as your previous evaluation functions] ...
    return all_event_preds, all_goal_probs


def get_3d_predictions(model, dataset, device):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_event_preds = []
    all_event_targets = []
    all_goal_probs = []
    all_goal_targets = []
    
    with torch.no_grad():
        for voxels, event_targets, goal_targets in dataloader:
            voxels = voxels.to(device)
            
            # Forward pass
            event_logits, goal_probs = model(voxels)
            
            # For multi-class (Events), take the argmax
            event_preds = torch.argmax(event_logits, dim=1)
            
            all_event_preds.extend(event_preds.cpu().numpy())
            all_event_targets.extend(event_targets.numpy())
            all_goal_probs.extend(goal_probs.cpu().numpy())
            all_goal_targets.extend(goal_targets.numpy())
            
    return np.array(all_event_targets), np.array(all_event_preds), \
           np.array(all_goal_targets), np.array(all_goal_probs)
