import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch_geometric.data import Data, Batch
import h5py


def rotate_point_cloud(points, axis='z', angle_deg=90):
    """Rotate point cloud around a specified axis (x, y, z) by a given angle in degrees."""
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    
    # Define rotation matrix based on the chosen axis
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(angle_rad), -np.sin(angle_rad)],
                                    [0, np.sin(angle_rad), np.cos(angle_rad)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                                    [0, 1, 0],
                                    [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                    [np.sin(angle_rad), np.cos(angle_rad), 0],
                                    [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")
    
    # Rotate the points
    rotated_points = points.dot(rotation_matrix.T)
    return rotated_points


def visualize_segmentation(model, i):
    # Extract data for visualization
    points = model.data_visual.pos.detach().cpu().numpy()
    points = rotate_point_cloud(points, axis='x', angle_deg=200)
    points = rotate_point_cloud(points, axis='y', angle_deg=45)
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
    colors[preds == 0] = [0, 1, 0] # Green for class 0
    colors[preds == 1] = [1, 0, 0] # Red for class 1

    # Plot side by side
    points = points.transpose(0, 1, 2)[0]
    rgb_viz = rgb_viz.transpose(0, 2, 1)[0]*255
    colors = colors.transpose(0, 1, 2)[0]
    fig = plt.figure(figsize=(14, 6))
    # print("points shape:", points.shape)
    # print("rgb_viz shape:", rgb_viz.shape)
    # print("colors shape:", colors.shape)
    # Original color visualization
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Original Colors")
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=rgb_viz, s=1)  # Scatter with RGB colors
    ax1.axis("off")

    # Predicted segmentation visualization
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Predicted Segmentation")
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)  # Scatter with segmentation colors
    ax2.axis("off")

    plt.tight_layout()

    # Save with a unique filename per batch
    save_path = f"segmentation_comparison_sample_{i}.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")
