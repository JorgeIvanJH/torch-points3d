#!/bin/bash

# Define an array of learning rate schedulers
lr_schedulers=("cosine" "cyclic" "exponential" "multi_step_reg" "multi_step" "plateau" "poly_lr" "step")

# Iterate through each lr_scheduler
for scheduler in "${lr_schedulers[@]}"
do
  echo "Starting training with lr_scheduler: $scheduler"
  
  # Run Hydra with the current lr_scheduler
  python train.py lr_scheduler=$scheduler
  
  # Wait for the training to finish before starting the next one (optional)
  # sleep 10  # Add a delay if needed
  
  echo "Finished training with lr_scheduler: $scheduler"
done
