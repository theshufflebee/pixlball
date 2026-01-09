import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from mplsoccer import Pitch
import os
from src import config


def plot_event_confusion_matrix(targets, preds, model_name="Model", save_path=None):
    """
    Displays and optionally saves the 3x3 Event Matrix.
    """
    
    cm = confusion_matrix(targets, preds)
    labels = ['Keep', 'Loss', 'Shot']
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar=False)
    
    plt.title(f"Event Confusion Matrix\n{model_name}", fontsize=12)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matrix saved to {save_path}")
    
    plt.show()
    
    return cm

def plot_goal_confusion_matrix(targets, preds, model_name="Model", save_path=None):
    
    """
    Displays the 2x2 Goal Matrix (Goal vs No Goal).
    """
    
    if len(targets) == 0:
        print("No shots found in dataset to create Goal Matrix.")
        return None
        
    cm = confusion_matrix(targets, preds)
    labels = ['No Goal', 'Goal']
    
    # Use a reasonable figure size for 2x2
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=labels, yticklabels=labels, cbar=False)
    
    plt.title(f"Goal Confusion Matrix (xG)\n{model_name}", fontsize=12)
    plt.ylabel('Actual Result')
    plt.xlabel('Predicted Result')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matrix saved to {save_path}")

    plt.show()
    
    return cm

def plot_shot_freeze_frame(shot_event):
   
    """
    Plots a StatsBomb 360 shot freeze-frame.
    """
    
    if 'shot_freeze_frame' not in shot_event or not shot_event['shot_freeze_frame']:
        print("No freeze-frame data available for this shot.")
        return

    pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=(12, 8))

    shooter_loc = shot_event['location']
    shooter_name = shot_event['player']
    
    # Plot Players
    for player in shot_event['shot_freeze_frame']:
        x, y = player['location']
        if x < 60: continue # Only attacking half

        name = player['player']['name']
        teammate = player['teammate']
        color = 'blue' if teammate else 'red'

        ax.scatter(x, y, c=color, s=100, edgecolors='black', zorder=3)
        ax.text(x+0.5, y+0.5, f"{name}", fontsize=7, zorder=4,
                color='darkblue' if color=='blue' else 'darkred')

    # Plot shooter
    ax.scatter(shooter_loc[0], shooter_loc[1], c='yellow', s=300, marker='*', edgecolors='black', zorder=5)
    ax.text(shooter_loc[0]+0.5, shooter_loc[1]+0.5, f"{shooter_name} (Shooter)", 
            fontsize=8, color='darkblue', zorder=6)

    ax.set_title(f"Freeze Frame: {shooter_name}", fontsize=14)
    plt.show()
    
from torch.utils.data import DataLoader

    
def plot_2d_channels_separated(train_dataset, save_path="figures/input_channels_split.png"):
    """
    Plots the 12x8 grid for each channel separately: Ball, Teammates, and Opponents.
    Assumes tensor shape (C, 12, 8).
    Used for figure 1 in the report
    """

    # Create a non-shuffling loader specifically for visualization (always returns the same frame)
    viz_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Extract the first batch (this will now be the same every time)
    inputs, targets_ev, targets_gl = next(iter(viz_loader))

    # Select a fixed Frame (determined by me)
    sample_2d = inputs[28].cpu().numpy()
    
    # Channel mapping based on your 360-layer logic
    titles = ['Channel 0: Ball', 'Channel 1: Teammates', 'Channel 2: Opponents']
    cmaps = ['YlOrRd', 'Blues', 'Reds'] # Red for ball, Blue for team, Dark Red for opps
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    for i in range(3):
        sns.heatmap(sample_2d[i], ax=axes[i], cmap=cmaps[i], cbar=True,
                    xticklabels=False, yticklabels=False, linewidths=.5)
        axes[i].set_title(titles[i], fontsize=12)
        axes[i].set_xlabel("Width (8)")
        if i == 0:
            axes[i].set_ylabel("Length (12)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()