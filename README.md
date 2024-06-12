###  Next-Activity Prediction for Non-stationary Processes with Unseen Data Variability

This repository contains the code and simulation results presented in the
experiments section of the paper. The commands described bellow must be
run from the root directory of the project.

Python >= 3.8 and PyTorch >= 1.10 (via conda) are required.

```
# Install additional libraries 
pip install numpy toolz sklearn tensorboard pandas pm4py

# For data generation
run data_gen.py with required persona, context, skip

# For model training and inference
run run_model.py with required persona, context, skip - here content and skip are just to access correct trace file
```

