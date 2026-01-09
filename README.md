pixlball/
├─ src/                 # This contains everything concerning the project happens
│  ├─ config.py         # constants, paths, hyperparameters
│  ├─ data.py           # data loading & preprocessing
│  ├─ dataset.py        # Dataset classes
│  ├─ evaluate.py       # evaluation functions
│  ├─ losses.py       # loss functions
│  ├─ model.py          # CNN / model definitions
│  ├─ plotfunctions.py       # plot functions
│  ├─ train.py          # training loop function
│  ├─ utils.py          # helper functions, plotting, etc.
│  ├─ xt_benchmark_gen.py          # contains all the functions to recreate the expected threat benchmark
├─ data/                # raw and processed data
│  ├─ events_data.parquet         # Event level Data downloaded via the statsbomb API
│  ├─ sb360_data.parquet           # Statsbomb 360 data downloaded via the statsbomb API
├─ notebooks/           # optional working notebooks
│  ├─          
│  ├─  10_run...
├─ requirements.txt
├─ notes_to_self.txt
├─ README.md # this fule
└─ main.py              # script to run the full workflow (tbd)





Config contains
    all global variables
    data patj
    
data.py contains
    all data processing
    
dataset.py contains
    all dataset classes for py torch
    
    
model.py contains
    all model rchiteture
    
train.py contains
    all training loops and model setups

evaluate_model.py contains
    all metrics and evaluation setups

    
