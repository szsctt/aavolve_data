#!/bin/bash

# Python code to check wandb login status
python <<EOF
import wandb

WANDB_NOTEBOOK_NAME = '2024-06-13_mbe-linear.ipynb'

wandb.login()
EOF

# Capture the exit status of the Python script
if [ $? -ne 0 ]; then
    exit 1
fi

# Check if the Python script exists
if [ ! -f scripts/MBE_tunning_TOMAS/run_script.py ]; then
    echo "run_script.py not found!"
    exit 1
fi

# Pass all parameters to the Python script
python3 scripts/MBE_tunning_TOMAS/run_script.py "$@"

# Check the exit status of the Python script
if [ $? -ne 0 ]; then
    echo "Failed to log parameters to wandb."
    exit 1
else
    echo "Parameters logged to wandb successfully."
fi
