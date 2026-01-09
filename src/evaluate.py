import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, roc_auc_score
# Assuming get_sequence_lengths is available from src.utils
from src.utils import get_sequence_lengths
import src.config as config
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import numpy as np




#############################################################################
# New Evaluation Function
#############################################################################

@torch.no_grad()
def evaluate_paper_metrics(model, validation_dataset, model_name="Model"):
    """
    Evaluates the model and returns overall accuracy, balanced accuracy, 
    and recall for every event class.
    """
    model.eval()
    val_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    all_event_preds = []
    all_event_targets = []
    all_goal_probs = []
    all_goal_targets = []

    for batch in val_loader:
        if len(batch) == 3:
            inputs, event_labels, goal_flags = batch
            inputs = inputs.to(config.DEVICE)
            forward_args = [inputs]
        else:
            inputs, context, event_labels, goal_flags = batch
            inputs = inputs.to(config.DEVICE)
            context = context.to(config.DEVICE)
            forward_args = [inputs, context]

        event_logits, goal_logits = model(*forward_args)

        # Event Head
        preds = torch.argmax(event_logits, dim=1).cpu().numpy()
        all_event_preds.extend(preds)
        all_event_targets.extend(event_labels.numpy())

        # Goal Head (Masked for Class 2)
        shot_mask = (event_labels == 2)
        if shot_mask.any():
            probs = torch.sigmoid(goal_logits[shot_mask]).view(-1).cpu().numpy()
            all_goal_probs.extend(probs)
            all_goal_targets.extend(goal_flags[shot_mask].numpy())

    # --- Metric Calculations ---
    acc = accuracy_score(all_event_targets, all_event_preds)
    bal_acc = balanced_accuracy_score(all_event_targets, all_event_preds)
    
    # Detailed Recall Metrics using classification_report
    # zero_division=0 ensures we don't crash if a class isn't predicted
    report = classification_report(all_event_targets, all_event_preds, output_dict=True, zero_division=0)
    
    # Extract specific recalls (handling cases where class indices might be strings or ints)
    recall_0 = report.get('0', report.get(0, {})).get('recall', 0.0)
    recall_1 = report.get('1', report.get(1, {})).get('recall', 0.0)
    recall_2 = report.get('2', report.get(2, {})).get('recall', 0.0)

    goal_auc = roc_auc_score(all_goal_targets, all_goal_probs) if len(all_goal_targets) > 0 else 0.5

    print(f"\n--- {model_name} Summary ---")
    print(f"Overall Acc: {acc:.4f} | Bal. Acc: {bal_acc:.4f} | Goal AUC: {goal_auc:.4f}")
    print(f"Recall -> Keep: {recall_0:.4f} | Loss: {recall_1:.4f} | Shot: {recall_2:.4f}")
    
    return {
        "Accuracy": acc,
        "Balanced Acc": bal_acc,
        "Goal AUC": goal_auc,
        "Recall_Keep": recall_0,
        "Recall_Loss": recall_1,
        "Recall_Shot": recall_2
    }

@torch.no_grad()
def get_predictions(model, validation_dataset):
    """Helper to get raw predictions and targets for both heads."""
    model.eval()
    loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    results = {
        "ev_targets": [], "ev_preds": [],
        "gl_targets": [], "gl_preds": []
    }

    for batch in loader:
        # Unpack 3 items (Base/3D) or 4 items (Context)
        if len(batch) == 3:
            inputs, ev_labels, gl_flags = batch
            logits_ev, logits_gl = model(inputs.to(config.DEVICE))
        else:
            inputs, ctx, ev_labels, gl_flags = batch
            logits_ev, logits_gl = model(inputs.to(config.DEVICE), ctx.to(config.DEVICE))

        # Event Head (Argmax)
        results["ev_preds"].extend(torch.argmax(logits_ev, dim=1).cpu().numpy())
        results["ev_targets"].extend(ev_labels.numpy())

        # Goal Head (Only where actual event is a Shot: Class 2)
        shot_mask = (ev_labels == 2)
        if shot_mask.any():
            # Threshold at 0.5 for binary confusion matrix
            gl_p = (torch.sigmoid(logits_gl[shot_mask]) > 0.5).int().view(-1).cpu().numpy()
            results["gl_preds"].extend(gl_p)
            results["gl_targets"].extend(gl_flags[shot_mask].numpy())
            
    return results
