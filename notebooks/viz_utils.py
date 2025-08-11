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


def visualize_segmentation(model, i, iou_scores):
    # Extract and rotate point coordinates
    points = model.data_visual.pos.detach().cpu().numpy()
    points = rotate_point_cloud(points, axis='x', angle_deg=200)
    points = rotate_point_cloud(points, axis='y', angle_deg=15)

    # Predicted labels
    preds = model.data_visual.pred.detach().cpu().numpy()
    # Ground truth labels
    gt_labels = model.data_visual.y.detach().cpu().numpy()

    # Input RGB colors
    if hasattr(model.data_visual, "x") and model.data_visual.x is not None and model.data_visual.x.shape[1] >= 3:
        rgb = model.data_visual.x[:, :3].detach().cpu().numpy()
        rgb_viz = rgb / 255.0 if rgb.max() > 1.0 else rgb
    else:
        print("[WARNING] No valid RGB found in model.data_visual.x, using gray color.")
        rgb_viz = np.ones_like(points) * 0.5

    # Visualization color maps
    def get_segmentation_colors(labels):
        colors = np.zeros_like(points)
        colors[labels == 0] = [0, 1, 0]  # Green
        colors[labels == 1] = [1, 0, 0]  # Red
        return colors

    pred_colors = get_segmentation_colors(preds)
    gt_colors = get_segmentation_colors(gt_labels)

    # Plot side by side
    pred_colors = pred_colors.transpose(0, 1, 2)[0] if pred_colors.ndim == 3 else pred_colors
    rgb_viz = rgb_viz.transpose(0, 2, 1)[0]*255 if rgb_viz.ndim == 3 else rgb_viz*255
    gt_colors = gt_colors.transpose(0, 1, 2)[0] if gt_colors.ndim == 3 else gt_colors
    points = points.transpose(0, 1, 2)[0] if points.ndim == 3 else points

    # Clip colors
    rgb_viz = np.clip(rgb_viz, 0, 1)
    pred_colors = np.clip(pred_colors, 0, 1)
    gt_colors = np.clip(gt_colors, 0, 1)

    # Plot
    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Original Colors")
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=rgb_viz, s=1)  # Scatter with RGB colors
    ax1.axis("off")

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title(f"Predicted Segmentation\nIoU class 0: {iou_scores[0]:.2f} | class 1: {iou_scores[1]:.2f}")
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=pred_colors, s=1)
    ax2.axis("off")

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title("Ground Truth Labels")
    ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c=gt_colors, s=1)
    ax3.axis("off")

    plt.tight_layout()
    save_path = f"segmentation_comparison_sample_{i}.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")
