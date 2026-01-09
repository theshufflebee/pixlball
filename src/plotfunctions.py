import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from mplsoccer import Pitch


def plot_event_confusion_matrix(targets, preds, model_name="Model", save_path=None):
    """Displays and optionally saves the 3x3 Event Matrix."""
    cm = confusion_matrix(targets, preds)
    labels = ['Keep', 'Loss', 'Shot']
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar=False)
    
    plt.title(f"Event Confusion Matrix\n{model_name}", fontsize=12)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    # Save if path given
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matrix saved to {save_path}")
    
    plt.show()
    
    return cm

def plot_goal_confusion_matrix(targets, preds, model_name="Model", save_path=None):
    """Displays the 2x2 Goal Matrix (Goal vs No Goal)."""
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
    
    # Save if path given
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matrix saved to {save_path}")

    plt.show()
    
    return cm

def plot_shot_freeze_frame(shot_event):
    """Plots a StatsBomb 360 shot freeze-frame."""
    if 'shot_freeze_frame' not in shot_event or not shot_event['shot_freeze_frame']:
        print("No freeze-frame data available for this shot.")
        return

    pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=(12, 8))

    shooter_loc = shot_event['location']
    shooter_name = shot_event['player']

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