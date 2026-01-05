# xt_benchmark.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch

def get_end_location(row):
    if row['type'] == 'Pass':
        return row['pass_end_location']
    elif row['type'] == 'Carry':
        return row['carry_end_location']
    elif row['type'] == 'Shot':
        return row['location']
    else:
        return np.nan


def prepare_events(all_events_df):
    """Filter and prepare events for xT computation."""
    xt_events = ['Pass', 'Carry', 'Shot']
    columns_needed = [
        'match_id', 'period', 'team_id', 'player_id', 'type',
        'location', 'pass_end_location', 'carry_end_location',
        'shot_statsbomb_xg', 'pass_outcome', 'shot_type'
    ]
    events_xt = all_events_df[all_events_df['type'].isin(xt_events)].copy()
    events_xt = events_xt[events_xt['shot_type'] != 'Penalty'].copy()

    events_xt = events_xt[columns_needed]

    events_xt['end_location'] = events_xt.apply(get_end_location, axis=1)
    events_xt = events_xt.drop(columns=['pass_end_location', 'carry_end_location'])
    return events_xt

def clip_locations(events_xt, pitch_length=120, pitch_height=80, epsilon=1e-6):
    """Clip locations to stay inside pitch boundaries."""
    events = events_xt.copy()
    safe_x = lambda x: min(x, pitch_length - epsilon)
    safe_y = lambda y: min(y, pitch_height - epsilon)

    events['location'] = events['location'].apply(lambda loc: [safe_x(loc[0]), safe_y(loc[1])])
    events['end_location'] = events['end_location'].apply(lambda loc: [safe_x(loc[0]), safe_y(loc[1])])
    return events

def map_to_grid(events, num_x_cells=12, num_y_cells=8, pitch_length=120, pitch_height=80):
    """Map events to discrete grid cells."""
    cell_width = pitch_length / num_x_cells
    cell_height = pitch_height / num_y_cells

    events_grid = events.copy()
    events_grid['start_cell_x'] = events_grid['location'].apply(lambda x: int(x[0] / cell_width))
    events_grid['start_cell_y'] = events_grid['location'].apply(lambda x: int(x[1] / cell_height))
    events_grid['end_cell_x'] = events_grid['end_location'].apply(lambda x: int(x[0] / cell_width))
    events_grid['end_cell_y'] = events_grid['end_location'].apply(lambda x: int(x[1] / cell_height))
    return events_grid

def compute_transition_matrices(events_grid, num_x_cells=12, num_y_cells=8):
    """Compute transition counts, shot counts, xG sums, and dead-state counts."""
    dead_state = num_x_cells * num_y_cells
    transition_counts = np.zeros((num_x_cells, num_y_cells, num_x_cells, num_y_cells))
    shot_counts = np.zeros((num_x_cells, num_y_cells))
    xG_sums = np.zeros((num_x_cells, num_y_cells))
    dead_counts = np.zeros((num_x_cells, num_y_cells))

    for _, row in events_grid.iterrows():
        sx, sy = row['start_cell_x'], row['start_cell_y']

        if row['type'] == 'Shot':
            shot_counts[sx, sy] += 1
            xG_sums[sx, sy] += row['shot_statsbomb_xg']
        else:
            ex, ey = row['end_cell_x'], row['end_cell_y']
            if row['type'] == 'Pass' and pd.notna(row['pass_outcome']):
                dead_counts[sx, sy] += 1
            else:
                transition_counts[sx, sy, ex, ey] += 1

    total_actions = transition_counts.sum(axis=(2,3)) + shot_counts + dead_counts
    total_actions_4d = total_actions[:, :, None, None]

    p_shot = np.divide(shot_counts, total_actions, out=np.zeros_like(shot_counts), where=total_actions!=0)
    p_transition = np.divide(transition_counts, total_actions_4d, out=np.zeros_like(transition_counts), where=total_actions_4d!=0)
    p_dead = np.divide(dead_counts, total_actions, out=np.zeros_like(dead_counts), where=total_actions!=0)
    xG_cell = np.divide(xG_sums, shot_counts, out=np.zeros_like(xG_sums), where=shot_counts!=0)

    return p_shot, p_transition, p_dead, xG_cell

def compute_xT(p_shot, p_transition, xG_cell, num_iterations=20, tolerance=1e-5):
    """Iteratively compute the xT grid."""
    num_x_cells, num_y_cells = p_shot.shape
    xT = np.zeros((num_x_cells, num_y_cells))

    for _ in range(num_iterations):
        xT_new = np.zeros_like(xT)
        for sx in range(num_x_cells):
            for sy in range(num_y_cells):
                xT_new[sx, sy] = p_shot[sx, sy] * xG_cell[sx, sy]
                xT_new[sx, sy] += np.sum(p_transition[sx, sy, :, :] * xT)
        if np.max(np.abs(xT_new - xT)) < tolerance:
            xT = xT_new
            break
        xT = xT_new
    return xT

def plot_xT_grid(xT, num_x_cells=12, num_y_cells=8, pitch_length=120, pitch_height=80, cmap='viridis'):
    """Plot xT heatmap on a soccer pitch."""
    pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
    fig, ax = pitch.draw(figsize=(12, 8))

    im = ax.imshow(
        xT.T,
        origin='lower',
        cmap=cmap,
        extent=[0, pitch_length, 0, pitch_height],
        aspect='auto',
        alpha=0.8
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('xT', fontsize=12)

    x_edges = np.linspace(0, pitch_length, num_x_cells + 1)
    y_edges = np.linspace(0, pitch_height, num_y_cells + 1)
    for xe in x_edges:
        ax.axvline(x=xe, color='white', linewidth=1, alpha=0.5)
    for ye in y_edges:
        ax.axhline(y=ye, color='white', linewidth=1, alpha=0.5)

    ax.set_title(f"Expected Threat (xT) grid ({num_x_cells} x {num_y_cells} cells)", fontsize=14)
    plt.show()

import pandas as pd
import numpy as np

def compute_player_cumulative_xT(xt_final, all_events_df, xT, num_x_cells=12, num_y_cells=8, pitch_length=120, pitch_height=80):
        
    """
    Compute cumulative xT for each player.
    
    Parameters
    ----------
    xt_final : pd.DataFrame
        Event-level dataframe with start/end cells, shot/xG info.
    all_events_df : pd.DataFrame
        Original StatsBomb event dataframe (to get player names).
    xT : np.ndarray
        Computed xT grid (num_x_cells x num_y_cells)
    
    Returns
    -------
    xt_final : pd.DataFrame
        Event-level dataframe with 'bellman_target' and 'player_name'.
    player_xt : pd.DataFrame
        Aggregated cumulative xT per player with player names.
    """

    cell_width = pitch_length / num_x_cells
    cell_height = pitch_height / num_y_cells

    worker_df = xt_final.copy()
    worker_df['start_cell_x'] = worker_df['location'].apply(lambda x: int(x[0] / cell_width))
    worker_df['start_cell_y'] = worker_df['location'].apply(lambda x: int(x[1] / cell_height))
    worker_df['end_cell_x'] = worker_df['end_location'].apply(lambda x: int(x[0] / cell_width))
    worker_df['end_cell_y'] = worker_df['end_location'].apply(lambda x: int(x[1] / cell_height))

    # 1️⃣ Compute Bellman targets
    bellman_targets = []
    for _, row in worker_df.iterrows():
        sx, sy = row['start_cell_x'], row['start_cell_y']
        if row['type'] == 'Shot':
            target = row['shot_statsbomb_xg']  # immediate reward
        else:
            ex, ey = row['end_cell_x'], row['end_cell_y']
            target = xT[ex, ey]  # value of next state
        bellman_targets.append(target)

    worker_df = worker_df.copy()
    worker_df['bellman_target'] = bellman_targets

    # 2️⃣ Aggregate xT by player
    agg_cols = {'bellman_target': 'sum'}
    player_xt = worker_df.groupby('player_id').agg(agg_cols).rename(columns={'bellman_target': 'cum_xT'})

    # 3️⃣ Merge player names
    df_players = all_events_df[['player_id', 'player']].drop_duplicates()
    
    # Add names to event-level dataframe
    xt_final = xt_final.merge(df_players, on='player_id', how='left').rename(columns={'player': 'player_name'})
    
    # Add names to aggregated player xT
    player_xt = player_xt.reset_index().merge(df_players, on='player_id', how='left').set_index('player_id')
    player_xt = player_xt.rename(columns={'player': 'player_name'})
    
    # Sort descending by cumulative xT
    player_xt = player_xt.sort_values('cum_xT', ascending=False)
    
    return xt_final, player_xt
