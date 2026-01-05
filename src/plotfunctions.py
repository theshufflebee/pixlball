import matplotlib.pyplot as plt
from mplsoccer import Pitch

def plot_shot_freeze_frame(shot_event):
    """
    Plots a StatsBomb 360 shot freeze-frame on the attacking half of the pitch,
    Input is a shot event containing freeze-frame data, in the statsbomb event level data

    Parameters
    ----------
    shot_event : dict or pandas Series
        Shot event containing:
        - 'location': [x, y] of shot
        - 'player': shooter name
        - 'shot_freeze_frame': list of player dicts
            - 'location': [x, y]
            - 'player': {'id', 'name'}
            - 'teammate': bool
    """
    if 'shot_freeze_frame' not in shot_event or not shot_event['shot_freeze_frame']:
        print("No freeze-frame data available for this shot.")
        return

    pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=(12, 8))

    shooter_loc = shot_event['location']
    shooter_name = shot_event['player']  # shooter from event-level data

    # Plot all other players
    for player in shot_event['shot_freeze_frame']:
        x, y = player['location']

        # Only plot players on attacking half
        if x < 60:
            continue

        name = player['player']['name']
        teammate = player['teammate']
        color = 'blue' if teammate else 'red'

        ax.scatter(x, y, c=color, s=100, edgecolors='black', zorder=3)
        ax.text(x+0.5, y+0.5, f"{name}", fontsize=7, zorder=4,
                color='darkblue' if color=='blue' else 'darkred')

    # Plot shooter explicitly
    ax.scatter(shooter_loc[0], shooter_loc[1], c='yellow', s=300, marker='*', edgecolors='black', zorder=5)
    ax.text(shooter_loc[0]+0.5, shooter_loc[1]+0.5, f"{shooter_name} (Shooter)", fontsize=8, color='darkblue', zorder=6)

    # Zoom to attacking half
    ax.set_xlim(60, 120)
    ax.set_ylim(0, 80)
    ax.set_title(f"Freeze Frame: {shooter_name} for {shot_event['team']}", fontsize=14)
    plt.show()
