import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# 1. Handle Pathing
# Dynamically locate the repository root to ensure imports work across environments
def find_repo_root(start_path=None, marker_dirs=('src', '.git')):
    p = os.path.abspath(start_path or os.getcwd())
    while True:
        if any(os.path.isdir(os.path.join(p, m)) for m in marker_dirs):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent

repo_root = find_repo_root()

if repo_root is None:
    repo_root = "/files/pixlball" # hard coded fallback

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
print(f"Using repo_root: {repo_root}")

# 2. Import Pixlball Modules
import src.config as config
import src.data as data
import src.model as model
import src.train as train
import src.dataset as dataset
import src.evaluate as evaluate
import src.utils as utils
import src.losses as losses
import src.plotfunctions as plotfunctions



import pandas as pd
from statsbombpy import sb
import os
from tqdm import tqdm



def main():
    print(f"--- Starting Pixlball Workflow (Repo Root: {repo_root}) ---")
    utils.enforce_replicability(42)
    
    data.run_full_data_pipeline(config.COMP_IDS, config.SEAS_IDS, force_redownload=config.FORCE_REDOWNLOAD)

    # ---------------------------------------------------------
    # Step 1: Data Ingestion & Preprocessing
    # ---------------------------------------------------------
    print("Loading raw Parquet data...")
    data_events = pd.read_parquet(os.path.join(repo_root, "data", "events_data.parquet"), engine="fastparquet")
    data_360 = pd.read_parquet(os.path.join(repo_root, "data", "sb360_data.parquet"), engine="fastparquet")

    print("Cleaning events and generating Kinetic features...")
    df_with_targets = data.event_data_loader(data_events)
    df_with_targets = data.add_ball_trajectory_features(df_with_targets)
    
    print("Processing 360 spatial grids...")
    df_360 = data.assign_grid_cells(data_360)
    nn_final_grids = data.aggregate_nn_layers_vectorized(df_360)

    print("Merging datasets and finalizing features...")
    nn_dataset = data.prepare_nn_dataset(
        df_with_targets,
        nn_final_grids, 
        target_cols=['nn_target', 'goal_flag'], 
        context_cols=True,
        keep_context_ids=True
    )
    nn_dataset = data.add_context_cols(nn_dataset)
    nn_dataset = data.add_target_as_int(nn_dataset)
    nn_dataset, ball_vector_columns = data.add_ball_coordinates(nn_dataset)

    # Calculate Weights for imbalanced Focal Loss
    class_weights_event, goal_pos_weight = utils.get_multitask_loss_weights(nn_dataset, config.DEVICE)
    
    # Storage for final comparison table
    final_results = {}

    # ---------------------------------------------------------
    # Step 2: Model Training Loops (Hierarchical)
    # ---------------------------------------------------------
    layer_columns = ["ball_layer", "teammates_layer", "opponents_layer"]
    train_df, val_df = utils.perform_replicable_split(nn_dataset)

    ###################################################################################
    # --- MODEL 1: Baseline CNN ---
    ######################################################################################
    print("\n--- Training Baseline 2D Model ---")
    train_ds = dataset.PitchDatasetMultiTask(train_df[layer_columns], train_df['nn_target_int'].values, train_df['goal_flag'].values)
    val_ds = dataset.PitchDatasetMultiTask(val_df[layer_columns], val_df['nn_target_int'].values, val_df['goal_flag'].values)
    
    base_model = model.TinyCNN_MultiTask_Threat(config.GRID_HEIGHT, config.GRID_WIDTH, config.NUM_EVENT_CLASSES).to(config.DEVICE)
    criterion_ev = losses.FocalLossThreat(alpha=class_weights_event, gamma=2.0)
    criterion_gl = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([goal_pos_weight]).to(config.DEVICE))

    base_model = train.train_multi_task_model(
        base_model, DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True),
        criterion_ev, criterion_gl, config.BASELINE_NUM_EPOCHS, "Baseline"
    )
    
    # Get The Metrics
    # -----------------------------------------------------------------------------------------
    final_results['Baseline'] = evaluate.evaluate_paper_metrics(base_model, val_ds, "Baseline")
    
    # Get predictions for a specific model
    res = evaluate.get_predictions(base_model, val_ds)

    # Extract Event Matrix
    ev_cm = plotfunctions.plot_event_confusion_matrix(res['ev_targets'], res['ev_preds'], "Baseline CNN", "figures/baseline_event_cm.png")

    # Extract Goal Matrix
    gl_cm = plotfunctions.plot_goal_confusion_matrix(res['gl_targets'], res['gl_preds'], "Baseline CNN", "figures/baseline_goal_cm.png")
    
    
    
    ###################################################################################
    # --- MODEL 2: Situational Context CNN ---
    ###################################################################################
    print("\n--- Training Situational Context Model ---")

    # 1. Define the specific static context columns
    static_features = ['under_pressure', 'counterpress', 'dribble_nutmeg']
    
    # 2. Setup Datasets using these specific features
    # Note: We use the same Context Dataset class but feed it the 3 static columns
    train_ds_sit = dataset.ContextBallVectorPitchDatasetMultiTask(
        train_df[layer_columns], 
        train_df['nn_target_int'].values, 
        train_df['goal_flag'].values, 
        train_df[static_features]
    )
    val_ds_sit = dataset.ContextBallVectorPitchDatasetMultiTask(
        val_df[layer_columns], 
        val_df['nn_target_int'].values, 
        val_df['goal_flag'].values, 
        val_df[static_features]
    )

    # 3. Initialize Model with 3 context features
    context_model = model.TinyCNN_MultiTask_Context_Threat(
        config.GRID_HEIGHT, 
        config.GRID_WIDTH, 
        config.NUM_EVENT_CLASSES, 
        num_context_features=len(static_features)
    ).to(config.DEVICE)

    # 4. Criteria
    criterion_ev = losses.FocalLossThreat(alpha=class_weights_event, gamma=2.0)
    criterion_gl = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([goal_pos_weight]).to(config.DEVICE))

    # 5. Train
    context_model = train.train_multi_task_model(
        context_model, 
        DataLoader(train_ds_sit, batch_size=config.BATCH_SIZE, shuffle=True),
        criterion_ev, 
        criterion_gl, 
        config.CONTEXT_NUM_EPOCHS, 
        "Situational-Context"
    )

    # Get Metrics
    # -----------------------------------------------------------------------------
    final_results['Context-Situational'] = evaluate.evaluate_paper_metrics(
        context_model, val_ds_sit, "Context-Situational"
    )

    # 7. Generate Figures
    res_sit = evaluate.get_predictions(context_model, val_ds_sit)

    # Event Confusion Matrix
    plotfunctions.plot_event_confusion_matrix(
        res_sit['ev_targets'], 
        res_sit['ev_preds'], 
        "Situational Context CNN", 
        "figures/context_event_cm.png"
    )

    # Goal Confusion Matrix
    plotfunctions.plot_goal_confusion_matrix(
        res_sit['gl_targets'], 
        res_sit['gl_preds'], 
        "Situational Context CNN", 
        "figures/context_goal_cm.png"
    )
    
    

    #############################################################################################################
    # --- MODEL 3: Kinetic Context CNN (Champion) ---
    #############################################################################################################
    
    print("\n--- Training Kinetic Context Model ---")
    train_ds_kin = dataset.ContextBallVectorPitchDatasetMultiTask(
        train_df[layer_columns], train_df['nn_target_int'].values, train_df['goal_flag'].values, train_df[ball_vector_columns]
    )
    val_ds_kin = dataset.ContextBallVectorPitchDatasetMultiTask(
        val_df[layer_columns], val_df['nn_target_int'].values, val_df['goal_flag'].values, val_df[ball_vector_columns]
    )

    kinetic_model = model.TinyCNN_MultiTask_Context_Threat(
        config.GRID_HEIGHT, config.GRID_WIDTH, config.NUM_EVENT_CLASSES, len(ball_vector_columns)
    ).to(config.DEVICE)

    kinetic_model = train.train_multi_task_model(
        kinetic_model, DataLoader(train_ds_kin, batch_size=config.BATCH_SIZE, shuffle=True),
        criterion_ev, criterion_gl, config.KINETIC_NUM_EPOCHS, "Kinetic"
    )
    
    # Get Metrics
    # -----------------------------------------------------------------------------
    final_results['Context-Kinetic'] = evaluate.evaluate_paper_metrics(kinetic_model, val_ds_kin, "Kinetic")
    
    # Get predictions
    res = evaluate.get_predictions(kinetic_model, val_ds_kin)

    # Extract Event Matrix
    ev_cm = plotfunctions.plot_event_confusion_matrix(res['ev_targets'], res['ev_preds'], "Kinetic CNN", "figures/kinetic_event_cm.png")

    # Extract Goal Matrix
    gl_cm = plotfunctions.plot_goal_confusion_matrix(res['gl_targets'], res['gl_preds'], "Kinetic CNN", "figures/kinetic_goal_cm.png")

    #########################################################################################
    # --- MODEL 4: 3D Voxel CNN ---
    ############################################################################################
    
    print("\n--- Training 3D Voxel Model ---")
    
    # generate voxels and perform a fresh split to ensure alignment
    voxels_list = data.generate_temporal_voxels(nn_dataset, lookback=3)
    nn_dataset['temporal_voxel'] = voxels_list
    train_df_3d, val_df_3d = utils.perform_replicable_split(nn_dataset)
    
    train_ds_3d = dataset.VoxelPitchDataset(train_df_3d)
    val_ds_3d = dataset.VoxelPitchDataset(val_df_3d)
    
    # Calculate Sampler weights using the 3D dataframe
    targets_3d = train_df_3d['nn_target_int'].values
    class_counts = np.bincount(targets_3d)
    class_weights_3d = 1. / class_counts
    sample_weights = torch.from_numpy(class_weights_3d[targets_3d]).double()

    # Create the Sampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Define UNWEIGHTED Loss for 3D
    criterion_ev_3d = losses.FocalLossThreat(alpha=None, gamma=2.0)
    
    # Initialize and Train
    voxel_model = model.Tiny3DCNN_MultiTask(config.NUM_EVENT_CLASSES).to(config.DEVICE)
    voxel_model = train.train_multi_task_model(
        voxel_model, 
        DataLoader(train_ds_3d, batch_size=config.BATCH_SIZE, sampler=sampler),
        criterion_ev_3d,   # Use the unweighted version!
        criterion_gl,      # Keep the goal weight (xG is still imbalanced)
        config.VOXEL_NUM_EPOCHS, 
        "3D-Voxel", 
        lr=config.LR_3D
    )
    
    # Get Metrics
    #------------------------------------------------------------------------------------------------
    
    final_results['3D-Voxel'] = evaluate.evaluate_paper_metrics(voxel_model, val_ds_3d, "3D-Voxel")

    # Get predictions
    res = evaluate.get_predictions(voxel_model, val_ds_3d)

    # Extract Event Matrix
    ev_cm = plotfunctions.plot_event_confusion_matrix(res['ev_targets'], res['ev_preds'], "Voxel CNN", "figures/voxel_event_cm.png")

    # Extract Goal Matrix
    gl_cm = plotfunctions.plot_goal_confusion_matrix(res['gl_targets'], res['gl_preds'], "Voxel CNN", "figures/voxel_goal_cm.png")

    # ---------------------------------------------------------
    # Step 3: Final Comparison & Export
    # ---------------------------------------------------------
    df_results = pd.DataFrame(final_results).T
    cols = ["Accuracy", "Balanced Acc", "Recall_Keep", "Recall_Loss", "Recall_Shot", "Goal AUC"]
    df_results = df_results[cols]
    
    output_path = os.path.join(repo_root, "figures", "model_comparison_table.csv")
    df_results.to_csv(output_path)
    
    print("\n--- FINAL UTILITY COMPARISON TABLE ---")
    print(df_results)
    print(f"\nResults saved to: {output_path}")
    
    plotfunctions.plot_2d_channels_separated(train_ds, save_path="figures/channel_visualization.png")
    
    print(f"\n Scrip End")

if __name__ == "__main__":
    main()