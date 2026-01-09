import numpy as np
import pandas as pd
from torch.utils.data import Dataset

#--------------------------------------------------------
# Event Data Cleaning
#-----------------------------------------------------------

def event_data_loader(data_events):
    """
    Main preprocessing pipeline for cleaning raw event data.
    Loads Data, cleans its and assigns NN targets
    """
    df = data_events.copy()
    
    # 1. Define Administrative/Non-Game events to remove
    admin_events = [
        'Starting XI', 'Half Start', 'Half End', 'Player On', 'Player Off',
        'Substitution', 'Tactical Shift', 'Referee Ball-Drop', 'Injury Stoppage',
        'Bad Behaviour', 'Shield', 'Goal Keeper', 'Pressure', 'Duel'
    ]

    # 2. Filter rows
    df = drop_events(df, rows_to_drop=admin_events)

    # 3. Define the specific StatsBomb metadata/detail columns to drop
    columns_to_drop = [
        'clearance_body_part', 'clearance_head', 'clearance_left_foot',
        'clearance_other', 'clearance_right_foot', 'shot_technique',
        'substitution_replacement_id', 'substitution_replacement',
        'substitution_outcome', 'shot_saved_off_target', 'pass_miscommunication',
        'goalkeeper_shot_saved_off_target', 'goalkeeper_punched_out',
        'shot_first_time', 'shot_body_part', 'related_events',
        'pass_shot_assist', 'pass_straight', 'pass_switch', 'pass_technique', 
        'pass_through_ball', 'goalkeeper_body_part', 'goalkeeper_end_location', 
        'goalkeeper_outcome', 'goalkeeper_position', 'goalkeeper_technique', 
        'goalkeeper_type', 'goalkeeper_penalty_saved_to_post', 
        'goalkeeper_shot_saved_to_post', 'goalkeeper_lost_out', 
        'goalkeeper_Clear', 'goalkeeper_In Play Safe', 'shot_key_pass_id',
        'shot_one_on_one', 'shot_end_location', 'shot_type', 'pass_angle',
        'pass_body_part', 'pass_type', 'pass_length', 'pass_outswinging',
        'pass_inswinging', 'pass_cross', 'pass_cut_back', 'pass_deflected', 
        'pass_goal_assist', 'pass_recipient', 'pass_recipient_id', 
        'pass_assisted_shot_id', 'pass_no_touch', 'pass_end_location', 
        'pass_aerial_won', 'pass_height', 'substitution_outcome_id',
        'tactics', 'block_deflection', 'dribble_no_touch', 'shot_open_goal', 
        'shot_saved_to_post', 'shot_redirect', 'shot_follows_dribble',
        'period', 'injury_stoppage_in_chanin', 'block_save_block',
        'ball recovery_offensive'
    ]

    # 4. Drop the columns
    df = drop_columns(df, columns_to_drop)

    # 5. Assign outcomes (Target variable generation)
    df = assign_lookahead_outcomes(df, lookahead=6)

    return df




def drop_events(events_df, rows_to_drop=None):
    """
    Sorts events by match_id and index, and drops non-play administrative events.
    
    Parameters:
        events_df (pd.DataFrame): Events dataframe with columns ['match_id', 'index', 'type', ...]
        
    Usage:
        Statsbomb has the Column 'type' that contains event types. This function drops all rows that are in the supplied events to drop
    
    Returns:
        pd.DataFrame: Cleaned and sorted dataframe
    """
    # Define non-play administrative events to drop
    
    
    # Drop administrative events
    df_cleaned = events_df[~events_df['type'].isin(rows_to_drop)].copy()
    
    # Sort by match_id and index to maintain chronological order
    df_cleaned.sort_values(by=['match_id', 'index'], inplace=True)
    
    # Reset index
    df_cleaned.reset_index(drop=True, inplace=True)
    print(f"{len(events_df) - len(df_cleaned)} events.")
    
    return df_cleaned


def drop_columns(df, columns_to_drop):
    """
    Drops columns from a DataFrame if they exist.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_drop (list of str): List of column names to remove.
    
    Returns:
        pd.DataFrame: DataFrame with specified columns removed.
    """
    cols_to_remove = [col for col in columns_to_drop if col in df.columns]
    cleaned_df = df.drop(columns=cols_to_remove)
    return cleaned_df


# ------------------------------------------------------
# Target and Context Variables
# -------------------------------------------------------

def assign_lookahead_outcomes(events_df, lookback=6, lookahead=6):
    """
    Assign outcomes to each event with:
      - nn_target: 'Keep Possession', 'Lose Possession', 'Shot'
      - goal_flag: 1 if a goal occurs in lookahead
    Also backpropagates 'Lose Possession' up to 'lookback' events without overwriting shots.
    """
    df = events_df.copy()
    df.sort_values(by=['match_id', 'possession', 'index'], inplace=True)

    df['nn_target'] = 'Keep Possession'
    df['goal_flag'] = 0

    for (match_id, possession), group in df.groupby(['match_id', 'possession'], sort=False):
        idxs = group.index.tolist()
        n = len(idxs)

        # Step 1: Propagate shots/goals forward
        shot_events = group[group['type'].str.startswith('Shot')].index.tolist()
        for shot_idx in shot_events:
            start_idx = max(0, idxs.index(shot_idx) - lookahead + 1)
            shot_window = idxs[start_idx:idxs.index(shot_idx)+1]
            df.loc[shot_window, 'nn_target'] = 'Shot'

            shot_row = df.loc[shot_idx]
            if shot_row.get('shot_outcome', '') == 'Goal':
                df.loc[shot_window, 'goal_flag'] = 1

        # Step 2: Propagate Lose Possession backward from last event
        last_idx = idxs[-1]
        if df.loc[last_idx, 'nn_target'] == 'Keep Possession':
            # Last event is not a shot: mark as Lose Possession
            df.loc[last_idx, 'nn_target'] = 'Lose Possession'

        # Backpropagate Lose Possession up to 'lookback', stop at shots
        for i in range(1, lookback):
            if n - 1 - i < 0:
                break
            current_idx = idxs[-1 - i]
            if df.loc[current_idx, 'nn_target'] == 'Keep Possession':
                df.loc[current_idx, 'nn_target'] = 'Lose Possession'
            else:
                # Stop if we hit a Shot
                break
    print(f"counts of each outcome {df['nn_target'].value_counts()}")

    return df

def add_context_cols(nn_df):
    """
    Adds static context columns to the NN dataset if they exist.
    """
    context_features = ['under_pressure', 'counterpress', 'dribble_nutmeg']
    for feat in context_features:
        if feat in nn_df.columns:
            nn_df[feat] = nn_df[feat].fillna(0).astype(np.float32)

    return nn_df

def add_target_as_int(nn_df):
    """
    Maps string target labels to integer indices for NN training.
    """
    target_map = {"Keep Possession": 0, "Lose Possession": 1, "Shot": 2}

    # Apply mapping
    nn_df['nn_target_int'] = nn_df['nn_target'].map(target_map)

    return nn_df



#--------------------------------------------------
# Spatial Logic and 360 Data Processing
#----------------------------------------------------


def assign_grid_cells(df_360, grid_height=12, grid_width=8, pitch_length=120, pitch_width=80):
    """
    Vectorized computation of grid cells for StatsBomb 360 data.

    Parameters:
        df_360 (pd.DataFrame): 360 data with a 'location' column [x, y]
        grid_height (int): number of rows in the pitch grid
        grid_width (int): number of columns in the pitch grid
        pitch_length (float): pitch length in meters
        pitch_width (float): pitch width in meters

    Returns:
        pd.DataFrame: same df with added 'cell_x' and 'cell_y' columns
    """
    locations = np.array(df_360['location'].tolist())
    x = locations[:, 0]
    y = locations[:, 1]

    cell_x = np.clip((x / pitch_length * grid_width).astype(int), 0, grid_width - 1)
    cell_y = np.clip((y / pitch_width * grid_height).astype(int), 0, grid_height - 1)

    df_360 = df_360.copy()
    df_360['cell_x'] = cell_x
    df_360['cell_y'] = cell_y

    return df_360


def aggregate_nn_layers_vectorized(df_360, grid_height=12, grid_width=8):
    """
    Vectorized creation of neural network input layers for all events.

    Parameters:
        df_360 (pd.DataFrame): 360 data with 'id', 'cell_x', 'cell_y', 'actor', 'teammate'
        grid_height (int): number of rows in the pitch grid
        grid_width (int): number of columns in the pitch grid

    Returns:
        pd.DataFrame: each row corresponds to an event id with columns:
            'id', 'ball_layer', 'teammates_layer', 'opponents_layer'
    """
    all_event_ids = df_360['id'].unique()
    event_index = {eid: i for i, eid in enumerate(all_event_ids)}
    num_events = len(all_event_ids)

    ball_layer_array = np.zeros((num_events, grid_height, grid_width), dtype=np.float32)
    teammates_layer_array = np.zeros((num_events, grid_height, grid_width), dtype=np.float32)
    opponents_layer_array = np.zeros((num_events, grid_height, grid_width), dtype=np.float32)

    event_idx = df_360['id'].map(event_index)

    # Ball layer
    ball_mask = df_360['actor'] == True
    yx_ball = df_360.loc[ball_mask, ['cell_y', 'cell_x']].values
    events_ball = event_idx[ball_mask].values
    ball_layer_array[events_ball, yx_ball[:, 0], yx_ball[:, 1]] = 1

    # Teammates layer
    teammate_mask = (df_360['teammate'] == True) & (~df_360['actor'])
    yx_team = df_360.loc[teammate_mask, ['cell_y', 'cell_x']].values
    events_team = event_idx[teammate_mask].values
    for e, (y, x) in zip(events_team, yx_team):
        teammates_layer_array[e, y, x] += 1

    # Opponents layer
    opponent_mask = df_360['teammate'] == False
    yx_opp = df_360.loc[opponent_mask, ['cell_y', 'cell_x']].values
    events_opp = event_idx[opponent_mask].values
    for e, (y, x) in zip(events_opp, yx_opp):
        opponents_layer_array[e, y, x] += 1

    nn_layers_df = pd.DataFrame({
        'id': all_event_ids,
        'ball_layer': list(ball_layer_array),
        'teammates_layer': list(teammates_layer_array),
        'opponents_layer': list(opponents_layer_array)
    })

    return nn_layers_df


#-------------------------------------
# Final Preparation of the NN Dataset
#----------------------------------------


def prepare_nn_dataset(
        events_df,
        nn_layers_df,
        target_cols=['nn_target'],
        id_col='id',
        context_cols=False,
        temporal_context=True, 
        keep_context_ids=False
    ):

    """
    Prepare dataset for NN training by keeping ID + target columns and merging 
    with the NN input layers (360° grids). Optionally keeps match and possession IDs.
    
    Parameters:
        events_df (pd.DataFrame): Events with targets assigned
        nn_layers_df (pd.DataFrame): NN layers DataFrame with same 'id'
        target_cols (list of str): List of columns to use as NN targets
        id_col (str): Column name of the unique event ID
        keep_context_ids (bool): If True, also keep match_id and possession_id
        
    Returns:
        pd.DataFrame: Merged dataset ready for NN training
    """

    # Base columns to keep
    cols_to_keep = [id_col] + target_cols

    # 1. Add Temporal Context (Ball Trajectory Vector)
    if temporal_context and 'ball_trajectory_vector' in events_df.columns:
        cols_to_keep.append('ball_trajectory_vector')

    # 2. If keeping match & possession IDs
    if keep_context_ids:
        for col in ['match_id', 'possession']:
            if col in events_df.columns:
                cols_to_keep.append(col)
        
    # 3. If keeping static context cols
    if context_cols:
        static_features = ['under_pressure', 'counterpress', 'dribble_nutmeg']
        for feat in static_features:
            if feat in events_df.columns:
                cols_to_keep.append(feat)

    events_trimmed = events_df[cols_to_keep].drop_duplicates(subset=id_col)
    return nn_layers_df.merge(events_trimmed, on=id_col, how='inner')



#------------------------
# Build Temporal Layers (Past observations)
#----------------------

def add_ball_trajectory_features(events_df, steps_back=3):
    df = events_df.copy()
    
    # 1. Get current coordinates
    df['ball_x'] = df['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else np.nan)
    df['ball_y'] = df['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else np.nan)

    # 2. Fill current NaNs first 
    # (Use ffill then bfill so every row has a 'current' ball position)
    df['ball_x'] = df.groupby('match_id')['ball_x'].ffill().bfill()
    df['ball_y'] = df.groupby('match_id')['ball_y'].ffill().bfill()

    # 3. Create historical shifts
    all_traj_cols = ['ball_x', 'ball_y']
    for i in range(1, steps_back + 1):
        x_col = f'ball_x_t-{i}'
        y_col = f'ball_y_t-{i}'
        
        # Shift globally within the match
        df[x_col] = df.groupby('match_id')['ball_x'].shift(i)
        df[y_col] = df.groupby('match_id')['ball_y'].shift(i)
        
        # KEY: Apply your logic. If the past is unknown, 
        # assume the ball was at the CURRENT position (bfill)
        df[x_col] = df[x_col].fillna(df['ball_x'])
        df[y_col] = df[y_col].fillna(df['ball_y'])
        
        all_traj_cols.extend([x_col, y_col])

    # 4. Final Vector
    df['ball_trajectory_vector'] = df[all_traj_cols].values.tolist()
    df['ball_trajectory_vector'] = df['ball_trajectory_vector'].apply(lambda x: np.array(x, dtype=np.float32))

    return df.drop(columns=[c for c in all_traj_cols if c != 'ball_trajectory_vector'])


def build_temporal_windows_with_mask(nn_df, past_steps=3):
    """
    Create temporal windows with a mask channel.
    Each window: [T, 4, 12, 8] where 4 = 3 layers + 1 mask channel.
    
    Parameters:
        nn_df: DataFrame with columns ['id', 'possession', 'ball_layer', 'teammates_layer', 'opponents_layer']
        past_steps: number of past steps to include (T = past_steps + 1)
        
    Returns:
        List of tensors [T, 4, 12, 8]
    """
    windows = []

    grouped = nn_df.groupby("possession")
    for _, group in grouped:
        group = group.sort_values("id")  # ensure chronological order
        layers = group[["ball_layer", "teammates_layer", "opponents_layer"]].values
        layers = [np.stack(row, axis=0) for row in layers]  # shape [3, 12, 8]

        for i in range(len(layers)):
            start = max(0, i - past_steps)
            context = layers[start:i+1]

            # Mask channel: 1 for real, 0 for padded
            mask = [np.ones((1, 12, 8), dtype=np.float32) for _ in range(len(context))]

            # Pad if needed
            if len(context) < past_steps + 1:
                pad_count = past_steps + 1 - len(context)
                pad_layers = [np.zeros((3, 12, 8), dtype=np.float32) for _ in range(pad_count)]
                pad_mask = [np.zeros((1, 12, 8), dtype=np.float32) for _ in range(pad_count)]
                context = pad_layers + context
                mask = pad_mask + mask

            # Add mask as 4th channel
            context_with_mask = [np.concatenate([c, m], axis=0) for c, m in zip(context, mask)]
            windows.append(np.stack(context_with_mask))  # shape [T, 4, 12, 8]

    return windows





import numpy as np

def generate_temporal_voxels(df, lookback=3, grid_cols=['ball_layer', 'teammates_layer', 'opponents_layer']):
    """
    Creates a 4D voxel (C, T, H, W) for each event.
    C = Channels (3), T = Time steps (lookback + 1), H, W = Grid size (12, 8).
    """
    voxels = []

    # 2. Group by match_id only
    grouped = df.groupby('match_id')
    
    for _, group in grouped:
        # Pre-stack the layers for the entire match: shape (N_events, 3, 12, 8)
        # uint8 because otherwise nuvolos crashes due to high memory demand
        group_grids = np.stack([
            np.stack(group[col].values).astype(np.uint8) for col in grid_cols], axis=1) 
        
        num_events = len(group)
        window_size = lookback + 1
        
        for i in range(num_events):
            start_idx = i - lookback
            
            if start_idx >= 0:
                # Normal case: we have enough history
                voxel = group_grids[start_idx : i + 1] # shape (T, C, H, W)
            else:
                # Boundary case (start of match): Pad with zeros
                padding_size = abs(start_idx)
                padding = np.zeros((padding_size, 3, 12, 8), dtype=np.uint8)
                available_frames = group_grids[0 : i + 1]
                voxel = np.concatenate([padding, available_frames], axis=0)
            
            # 3. Transpose to PyTorch Conv3D format: (Channels, Time, Height, Width)
            # Input voxel is (T, C, H, W) -> Output is (C, T, H, W)
            voxel = np.transpose(voxel, (1, 0, 2, 3))
            voxels.append(voxel)
            
    return voxels






def add_ball_coordinates(df, column_name='ball_trajectory_vector'):
    """
    Dynamically expands a column of coordinate lists into individual x and y columns.
    
    Args:
        df (pd.DataFrame): The dataset containing the trajectory vectors.
        column_name (str): The name of the column containing the lists.
        
    Returns:
        pd.DataFrame: The dataframe with new x1, y1, x2, y2... columns added.
    """
    if df[column_name].empty:
        return df

    # 1. Determine the length from the first entry in the data
    first_vector = df[column_name].iloc[0]
    vector_length = len(first_vector)
    num_points = vector_length // 2

    # 2. Generate the column names dynamically: x1, y1, x2, y2...
    vector_names = [f"{coord}{i}" for i in range(1, num_points + 1) for coord in ['x', 'y']]

    # 3. Expand and concatenate to the original dataframe
    expanded_coords = pd.DataFrame(
        df[column_name].tolist(), 
        index=df.index, 
        columns=vector_names
    )
    
    df = pd.concat([df, expanded_coords], axis=1)
    
    for col in df.columns:
        if col.startswith('x'):
            df[col] = df[col] / 120.0  # Scale X to [0, 1]
        elif col.startswith('y'):
            df[col] = df[col] / 80.0   # Scale Y to [0, 1]
    
    return df, vector_names