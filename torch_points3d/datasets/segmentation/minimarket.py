# torch_points3d/datasets/segmentation/minimarket.py

import os
import torch
import h5py
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch.utils.data import random_split
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker


class MiniMarketRawDataset(InMemoryDataset):
    def __init__(self, root, split="train", filename="minimarket.h5", transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        self.filename = filename
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self._split_index(split)])
        print("filename: ", self.filename)
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
        # No download, the file should already be in place
        pass

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
        for i in range(seg_points.shape[0]):
            pos = torch.tensor(seg_points[i], dtype=torch.float)
            rgb = torch.tensor(seg_colors[i], dtype=torch.float) / 255.0
            labels = torch.tensor(np.argmax(seg_labels[i], axis=-1), dtype=torch.long)
            data = Data(pos=pos, rgb=rgb, y=labels)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        total = len(data_list)

        if self.filename == "minimarket_train.h5":
            # Use **100% for training**
            torch.save(self.collate(data_list), self.processed_paths[0])
        elif self.filename == "minimarket_validtest.h5":
            # Split into **50% validation + 50% testing**
            mid = total // 2
            torch.save(self.collate(data_list[:mid]), self.processed_paths[1])  # validation
            torch.save(self.collate(data_list[mid:]), self.processed_paths[2])  # test



class MiniMarketDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        self.train_dataset = MiniMarketRawDataset(
            self._data_path, split="train", filename="minimarket_train.h5",
            pre_transform=self.pre_transform,
            transform=self.train_transform
        )

        self.val_dataset = MiniMarketRawDataset(
            self._data_path, split="val", filename="minimarket_validtest.h5",
            pre_transform=self.pre_transform,
            transform=self.val_transform
        )

        self.test_dataset = MiniMarketRawDataset(
            self._data_path, split="test", filename="minimarket_validtest.h5",
            pre_transform=self.pre_transform,
            transform=self.test_transform
        )
        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)


    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
