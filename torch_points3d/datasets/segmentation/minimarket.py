# torch_points3d/datasets/segmentation/minimarket.py

import os
import torch
import h5py
import numpy as np
from typing import Iterable, Union, List, Dict, Tuple
from torch_geometric.data import InMemoryDataset, Data
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker


def _as_list(x: Union[float, Iterable[float]]) -> List[float]:
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        return list(x)
    return [float(x)]


def _voxelize_sample(
    pos_np: np.ndarray,
    rgb_np: np.ndarray,
    labels_np: np.ndarray,
    voxel_size: float,
    min_points_per_voxel: int = 32,
    normalize_in_voxel: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Slice a single point cloud into voxels and return a list of voxel point sets.
    Returns a list of tuples: (pos, rgb, labels, voxel_origin)
    """
    # Anchor grid to sample AABB min corner for stable keys
    aabb_min = pos_np.min(axis=0, keepdims=True)
    # Compute integer voxel keys
    keys = np.floor((pos_np - aabb_min) / voxel_size).astype(np.int32)
    # Map voxel key -> indices
    # Use a structured array so we can hash rows efficiently
    keys_view = keys.view([('', keys.dtype)] * keys.shape[1])
    uniq, inverse, counts = np.unique(keys_view, return_inverse=True, return_counts=True, axis=0)

    voxels = []
    for kid, cnt in enumerate(counts):
        if cnt < min_points_per_voxel:
            continue
        idx = np.where(inverse == kid)[0]
        vpos = pos_np[idx]
        vrgb = rgb_np[idx]
        vlbl = labels_np[idx]
        # Recover voxel integer key and origin
        k = uniq[kid].view(keys.dtype).reshape(1, -1).astype(np.int64)[0]
        voxel_origin = (aabb_min[0] + k * voxel_size).astype(np.float32)
        if normalize_in_voxel:
            # Translate to voxel-local coords (keep absolute info in voxel_origin meta)
            vpos = vpos - voxel_origin[None, :]
        voxels.append((vpos, vrgb, vlbl, voxel_origin))
    return voxels


class MiniMarketRawDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        split: str = "train",
        filename: str = "minimarket.h5",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        # --- NEW ARGUMENTS ---
        voxel_sizes: Union[float, Iterable[float]] = None,
        min_points_per_voxel: int = 32,
        normalize_in_voxel: bool = False,
    ):
        """
        If voxel_sizes is provided (float or list), each original sample is split into
        per-voxel sub-samples for every size in voxel_sizes. Each voxel becomes a Data().
        """
        self.split = split
        self.filename = filename
        # defaults
        self.voxel_sizes = _as_list(voxel_sizes) if voxel_sizes is not None else None
        self.min_points_per_voxel = int(min_points_per_voxel)
        self.normalize_in_voxel = bool(normalize_in_voxel)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self._split_index(split)])
        print("filename:", self.filename)

    def _split_index(self, split):
        mapping = {"train": 0, "val": 1, "test": 2}
        return mapping[split]

    @property
    def processed_file_names(self):
        base = os.path.splitext(self.filename)[0]
        return [f"{base}_train.pt", f"{base}_val.pt", f"{base}_test.pt"]

    @property
    def raw_file_names(self):
        return [self.filename]

    def download(self):
        pass  # file should already be in place

    def process(self):
        path = os.path.join(self.raw_dir, self.filename)
        print(f"Processing {self.split} dataset from {path}")

        with h5py.File(path, 'r') as f:
            seg_points = f['seg_points'][:]
            seg_colors = f['seg_colors'][:]
            seg_labels = f['seg_labels'][:]

        # # Subsamplin IN CASE NECESSARY
        # num_points = seg_points.shape[1]
        # num_keep = int(num_points * 0.5)
        # idx = np.sort(np.random.choice(num_points, num_keep, replace=False))
        # seg_points = seg_points[:, idx, :]
        # seg_colors = seg_colors[:, idx, :]
        # seg_labels = seg_labels[:, idx, :]

        data_list = []
        N = seg_points.shape[0]

        for i in range(N):
            pos_np = seg_points[i].astype(np.float32)      # (P, 3)
            rgb_np = (seg_colors[i].astype(np.float32) / 255.0)  # (P, 3)
            labels_np = np.argmax(seg_labels[i], axis=-1).astype(np.int64)  # (P,)

            if self.voxel_sizes is None:
                # ORIGINAL BEHAVIOR: keep whole sample as one Data
                data = Data(
                    pos=torch.from_numpy(pos_np),
                    rgb=torch.from_numpy(rgb_np),
                    y=torch.from_numpy(labels_np),
                    parent_id=torch.tensor([i], dtype=torch.long)
                )
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
            else:
                # NEW: expand into per-voxel samples for each voxel size
                for vs in self.voxel_sizes:
                    voxels = _voxelize_sample(
                        pos_np, rgb_np, labels_np,
                        voxel_size=float(vs),
                        min_points_per_voxel=self.min_points_per_voxel,
                        normalize_in_voxel=self.normalize_in_voxel,
                    )
                    for vpos, vrgb, vlbl, vorig in voxels:
                        data = Data(
                            pos=torch.from_numpy(vpos),
                            rgb=torch.from_numpy(vrgb),
                            y=torch.from_numpy(vlbl),
                            voxel_size=torch.tensor([vs], dtype=torch.float32),
                            voxel_origin=torch.from_numpy(vorig),  # (3,)
                            parent_id=torch.tensor([i], dtype=torch.long)
                        )
                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)
                        data_list.append(data)

        if self.filename == "minimarket_train.h5":
            torch.save(self.collate(data_list), self.processed_paths[0])
        elif self.filename == "minimarket_valid.h5":
            torch.save(self.collate(data_list), self.processed_paths[1])
        elif self.filename == "minimarket_test.h5":
            torch.save(self.collate(data_list), self.processed_paths[2])


class MiniMarketDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        # Pull new options (with sensible defaults)
        voxel_sizes = getattr(dataset_opt, "voxel_sizes", None)  # e.g., 0.02 or [0.02, 0.03]
        min_pts = getattr(dataset_opt, "min_points_per_voxel", 32)
        norm_voxel = getattr(dataset_opt, "normalize_in_voxel", False)

        self.train_dataset = MiniMarketRawDataset(
            self._data_path, split="train", filename="minimarket_train.h5",
            pre_transform=self.pre_transform, transform=self.train_transform,
            voxel_sizes=voxel_sizes, min_points_per_voxel=min_pts, normalize_in_voxel=norm_voxel
        )

        self.val_dataset = MiniMarketRawDataset(
            self._data_path, split="val", filename="minimarket_valid.h5",
            pre_transform=self.pre_transform, transform=self.val_transform,
            voxel_sizes=voxel_sizes, min_points_per_voxel=min_pts, normalize_in_voxel=norm_voxel
        )

        self.test_dataset = MiniMarketRawDataset(
            self._data_path, split="test", filename="minimarket_test.h5",
            pre_transform=self.pre_transform, transform=self.test_transform,
            voxel_sizes=voxel_sizes, min_points_per_voxel=min_pts, normalize_in_voxel=norm_voxel
        )

        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
