import torch
import os
import sys
BASEDIR = os.path.dirname(os.getcwd())
sys.path.append(BASEDIR)

from omegaconf import OmegaConf
from torch_points3d.datasets.dataset_factory import instantiate_dataset
from torch_points3d.models.model_factory import instantiate_model
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint
import open3d as o3d
import numpy as np

# Step 1: Paths
checkpoint_path = BASEDIR + "/outputs/2025-06-22/17-33-52/KPConvPaper.pt"
config_path = BASEDIR + "/conf/config.yaml"

# Step 2: Load config and model checkpoint
cfg = ModelCheckpoint(
    load_dir=os.path.dirname(checkpoint_path),
    check_name=os.path.basename(checkpoint_path).replace(".pt", ""),
    selection_stage="val",
    run_config=OmegaConf.load(config_path),
    resume=True,
).run_config

model_ckpt = ModelCheckpoint(
    load_dir=os.path.dirname(checkpoint_path),
    check_name=os.path.basename(checkpoint_path).replace(".pt", ""),
    selection_stage="val",  # or "test" or "train"
    run_config=cfg,
    resume=True,
)

# Step 3: Dataset and model
dataset = instantiate_dataset(model_ckpt.data_config)

model = model_ckpt.create_model(dataset, weight_name="train_iou")  # or "latest"
model.eval()
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
if hasattr(dataset, "test_dataset") and dataset.test_dataset:
    dataset.val_dataset = dataset.test_dataset[0]
dataset.create_dataloaders(
    model,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    precompute_multi_scale=False
)


# Step 4: Inference
from torch_geometric.data import Batch

sample = dataset.val_dataloader.dataset[0]
batched_sample = Batch.from_data_list([sample])
model.set_input(batched_sample, device=model.device)

with torch.no_grad():
    model.forward()

# Step 5: Visualization
visuals = model.get_current_visuals()
data_visual = visuals["data_visual"]
points = data_visual.pos.cpu().numpy()
preds = data_visual.pred.cpu().numpy()

colors = np.zeros_like(points)
colors[preds == 0] = [1, 0, 0]
colors[preds == 1] = [0, 1, 0]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Convert to numpy
points = data_visual.pos.cpu().numpy()
preds = data_visual.pred.cpu().numpy()

# Use RGB from features or assign gray if not available
if hasattr(data_visual, "x") and data_visual.x is not None and data_visual.x.shape[1] >= 3:
    rgb = data_visual.x[:, :3].cpu().numpy()
    # Only normalize if values are not already between 0 and 1
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
else:
    print("[WARNING] No valid RGB found in data_visual.x, using gray color.")
    rgb = np.ones_like(points) * 0.5  # fallback


# Predicted segmentation colors
colors = np.zeros_like(points)
colors[preds == 0] = [1, 0, 0]  # red
colors[preds == 1] = [0, 1, 0]  # green

# Plot side by side
fig = plt.figure(figsize=(14, 6))

# Original color
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("Original Colors")
ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=rgb, s=1)
ax1.axis("off")

# Prediction
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("Predicted Segmentation")
ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
ax2.axis("off")

plt.tight_layout()
plt.savefig("segmentation_comparison.png", dpi=300)
print("Saved visualization to segmentation_comparison.png")


