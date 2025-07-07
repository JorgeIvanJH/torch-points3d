#!/bin/bash

# Define an array of models
models=("segmentation/pointnet" "segmentation/pointnet2" "segmentation/ppnet" "segmentation/kpconv" "segmentation/rsconv")

# Define model names for WandB logging or saving
model_names=("PointNet" "pointnet2_charlesssg" "PPNet" "KPConvPaper" "RSConv_MSN")

# Iterate through each model
for i in "${!models[@]}"
do
  model=${models[$i]}       # Get model path
  model_name=${model_names[$i]}  # Get model name
  
  echo "full command: poetry run python train.py ++model=$model ++model_name=$model_name"
  poetry run python train.py +models=$model ++model_name=$model_name
  
done
