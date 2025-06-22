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
    def __init__(self, root, split="train", transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[self._split_index(split)])
    
    def _split_index(self, split):
        mapping = {"train": 0, "val": 1, "test": 2}
        return mapping[split]

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def raw_file_names(self):
        return ['minimarket.h5']

    def download(self):
        # No download, the file should already be in place
        pass

    def process(self):
        path = os.path.join(self.raw_dir, 'minimarket.h5')
        with h5py.File(path, 'r') as f:
            seg_points = f['seg_points'][:]
            seg_colors = f['seg_colors'][:]
            seg_labels = f['seg_labels'][:]

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
        train_end = int(0.7 * total)
        val_end = int(0.85 * total)
        torch.save(self.collate(data_list[:train_end]), self.processed_paths[0])
        torch.save(self.collate(data_list[train_end:val_end]), self.processed_paths[1])
        torch.save(self.collate(data_list[val_end:]), self.processed_paths[2])


class MiniMarketDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        self.train_dataset = MiniMarketRawDataset(
            self._data_path, split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform
        )

        self.val_dataset = MiniMarketRawDataset(
            self._data_path, split="val",
            pre_transform=self.pre_transform,
            transform=self.val_transform
        )

        self.test_dataset = MiniMarketRawDataset(
            self._data_path, split="test",
            pre_transform=self.pre_transform,
            transform=self.test_transform
        )

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
