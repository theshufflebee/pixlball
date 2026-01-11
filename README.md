# Pixlball: Multi-Task Possession Utility Prediction

This repository contains a multi-task deep learning framework designed to quantify possession utility in professional football using StatsBomb 360 data.

## Project Structure
```text
pixlball/
├─ report/                  # Project documentation
│  └─ ada_pixlball.pdf      # Final research report (SIAM Style)
├─ src/                     # Core source code
│  ├─ config.py             # Hyperparameters (Gamma, Lambda_Goal, Seed)
│  ├─ data.py               # StatsBomb API ingestion pipeline
│  ├─ dataset.py            # PyTorch Datasets (Spatial, Kinetic, 3D)
│  ├─ evaluate.py           # Metrics (Balanced Acc, Shot Recall, Goal AUC)
│  ├─ losses.py             # Focal Loss implementation
│  ├─ model.py              # CNN Architectures
│  ├─ plotfunctions.py      # Confusion Matrix & xT generators
│  ├─ train.py              # Training loop logic
│  ├─ utils.py              # Grid assignment & replicability tools
│  └─ xt_benchmark_gen.py   # Baseline xT comparison
├─ data/                    # Local storage (Ignored by Git)
│  ├─ events_data.parquet   # Processed events
│  └─ sb360_data.parquet    # Processed 360-frames
├─ figures/                 # Exported plots for the report
├─ notebooks/               # Research & Exploration
│  ├─ 00_setup.ipynb        # Environment validation
│  └─ 01_MASTER.ipynb       # Results generation & Table export
├─ requirements.txt         # Frozen dependencies
└─ main.py                  # Entry point for full workflow
```
## Installation
1. Base Dependencies
Install the core machine learning and data processing libraries: 
``pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn tqdm``

2. Specialized Packages
Required for StatsBomb API access, parquet handling, and pitch visualization:
``pip install statsbombpy fastparquet mplsoccer python-snappy``

3. Full Replication
Alternatively, install the requirements via:
``pip install -r requirements.txt``

## Data Management
The dataset is fetched via the StatsBomb API and cached locally as Parquet files to optimize storage and loading speed.

Replication Note: Since the StatsBomb API is live and data can be updated by the provider, it is recommended to use the provided .parquet files for exact replication of report metrics.

Redownloading: To refresh the local cache from the source, set FORCE_REDOWNLOAD = True in config.py.

## Execution
Once the environment is configured, execute the full end-to-end workflow (Data Ingestion -> Training -> Evaluation -> Plotting) by running:

``python main.py``
