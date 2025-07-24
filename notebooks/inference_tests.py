import torch
import os
import sys
BASEDIR = os.path.dirname(os.getcwd())
sys.path.append(BASEDIR)
import h5py
from torch_geometric.data import Data

from torch_geometric.data import Batch

from omegaconf import OmegaConf
from torch_points3d.datasets.dataset_factory import instantiate_dataset
from torch_points3d.models.model_factory import instantiate_model
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset
import open3d as o3d
import numpy as np
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer
import pyvista as pv
from tqdm.auto import tqdm
from viz_utils import visualize_segmentation


# === Paths ===
BASEDIR = os.path.dirname(os.getcwd())
sys.path.append(BASEDIR)


checkpoint_path = "/home/segment1/jorge/torch-points3d/outputs/2025-07-22/23-15-04"
config_path = checkpoint_path+"/.hydra"



@hydra.main(config_path=config_path, config_name="config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.training.cuda > -1 and torch.cuda.is_available():
        device = "cuda"
        torch.cuda.set_device(cfg.training.cuda)
    else:
        raise ValueError("CUDA is not available or not set correctly in the config.")
    cfg.training.batch_size = 1
    cfg.training.checkpoint_dir = checkpoint_path
    checkpoint: ModelCheckpoint = ModelCheckpoint(
            cfg.training.checkpoint_dir,
            cfg.model_name,
            cfg.training.weight_name,
            run_config=cfg,
            resume=False,
        )
    dataset: BaseDataset = instantiate_dataset(checkpoint.data_config)
    model: BaseModel = checkpoint.create_model(dataset, weight_name=cfg.training.weight_name)
    dataset.create_dataloaders(
            model,
            cfg.training.batch_size,
            cfg.training.shuffle,
            cfg.training.num_workers,
            cfg.training.precompute_multi_scale,
        )
    model.verify_data(dataset.train_dataset[0])
    model = model.to(device)
    model.eval()
    test_loader = dataset.test_dataloaders[0]
    with torch.no_grad():
        with tqdm(test_loader) as tq_test_loader:
            for i, data in enumerate(tq_test_loader):
                data.to(device)
                model.set_input(data, device)
                model.forward(data)
                visualize_segmentation(model, i)

            



if __name__ == "__main__":
    main()

# poetry run python inference_tests.py



