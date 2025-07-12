import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch_geometric.data import Data, Batch
import h5py

def visualize_segmentation(model, i):
    # Extract data for visualization
    points = model.data_visual.pos.detach().cpu().numpy()
    preds = model.data_visual.pred.detach().cpu().numpy()

    # Original input RGB colors (normalize if needed)
    if hasattr(model.data_visual, "x") and model.data_visual.x is not None and model.data_visual.x.shape[1] >= 3:
        rgb = model.data_visual.x[:, :3].detach().cpu().numpy()
        if rgb.max() > 1.0:
            rgb_viz = rgb / 255.0
        else:
            rgb_viz = rgb
    else:
        print("[WARNING] No valid RGB found in model.data_visual.x, using gray color.")
        rgb_viz = np.ones_like(points) * 0.5

    # Predicted segmentation colors: class 0 = red, class 1 = green
    colors = np.zeros_like(points)
    colors[preds == 0] = [1, 0, 0]
    colors[preds == 1] = [0, 1, 0]

    # Plot side by side
    fig = plt.figure(figsize=(14, 6))

    # Original color
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Original Colors")
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=rgb_viz, s=1)
    ax1.axis("off")

    # Prediction
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Predicted Segmentation")
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    ax2.axis("off")

    plt.tight_layout()

    # Save with unique filename per batch
    save_path = f"segmentation_comparison_sample_{i}.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")
