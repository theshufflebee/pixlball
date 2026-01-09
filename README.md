This repository contains a multi-task deep learning framework designed to quantify possession utility in
professional football using StatsBomb data.

pixlball/
├─ src/                     # Core project source code
│  ├─ config.py             # Constants, paths, and hyperparameters
│  ├─ data.py               # Data loading and cleaning pipeline
│  ├─ dataset.py            # PyTorch Dataset classes (Spatial, Kinetic, and 3D)
│  ├─ evaluate.py           # Evaluation suite (Recall, Balanced Acc, Goal AUC)
│  ├─ losses.py             # Custom Focal Loss for class imbalance
│  ├─ model.py              # CNN definitions (Baseline, Context, Kinetic, Voxel)
│  ├─ plotfunctions.py      # Visualizations (Confusion Matrices, xT Maps)
│  ├─ train.py              # Unified training loop with masked loss logic
│  ├─ utils.py              # Path management and grid assignment
│  └─ xt_benchmark_gen.py   # Expected Threat (xT) baseline comparison
├─ data/                    # Local storage (Ignored by Git)
│  ├─ events_data.parquet   # Raw StatsBomb event data
│  └─ sb360_data.parquet    # Raw StatsBomb 360 spatial data
├─ notebooks/               # Experimental Jupyter Notebooks
│  ├─ 00_setup.ipynb        # Initial environment and data check
│  ├─ 01_MASTER.ipynb       # Main execution script and results generator
│  ├─ figures/              # Generated plots and confusion matrices
│  └─ model_comparison_table.csv # Final exported metrics
├─ requirements.txt         # Project dependencies (frozen for replicability)
└─ main.py                  # Script to run the full workflow