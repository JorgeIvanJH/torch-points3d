import os
import copy
import torch
import hydra
import time
import logging

from tqdm.auto import tqdm
import wandb

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset
from torch_points3d.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

# Import from metrics
from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from torch_points3d.utils.colors import COLORS
from torch_points3d.utils.wandb_utils import Wandb
from torch_points3d.visualization import Visualizer

# PyTorch Profiler import
import torch.profiler
import torch.autograd.profiler
from contextlib import nullcontext

log = logging.getLogger(__name__)


class Trainer:
    """
    TorchPoints3d Trainer handles the logic between
        - BaseModel,
        - Dataset and its Tracker
        - A custom ModelCheckpoint
        - A custom Visualizer
    It supports MC dropout - multiple voting_runs for val / test datasets
    """

    def __init__(self, cfg):
        self._cfg = cfg
        self._initialize_trainer()

    def _initialize_trainer(self):
        # Enable CUDNN BACKEND
        torch.backends.cudnn.enabled = self.enable_cudnn

        if not self.has_training:
            self._cfg.training = self._cfg
            resume = bool(self._cfg.checkpoint_dir)
        else:
            resume = bool(self._cfg.training.checkpoint_dir)

        # Get device
        if self._cfg.training.cuda > -1 and torch.cuda.is_available():
            device = "cuda"
            torch.cuda.set_device(self._cfg.training.cuda)
        else:
            device = "cpu"
        self._device = torch.device(device)
        log.info("DEVICE : {}".format(self._device))

        # Profiling
        if self.profiling:
            # Set the num_workers as torch.utils.bottleneck doesn't work well with it
            self._cfg.training.num_workers = 0

        # Start Wandb if public
        if self.wandb_log:
            Wandb.launch(self._cfg, self._cfg.training.wandb.public and self.wandb_log)

        # Checkpoint

        self._checkpoint: ModelCheckpoint = ModelCheckpoint(
            self._cfg.training.checkpoint_dir,
            self._cfg.model_name,
            self._cfg.training.weight_name,
            run_config=self._cfg,
            resume=resume,
        )

        # Create model and datasets
        if not self._checkpoint.is_empty:
            self._dataset: BaseDataset = instantiate_dataset(self._checkpoint.data_config)
            self._model: BaseModel = self._checkpoint.create_model(
                self._dataset, weight_name=self._cfg.training.weight_name
            )
        else:
            self._dataset: BaseDataset = instantiate_dataset(self._cfg.data)
            self._model: BaseModel = instantiate_model(copy.deepcopy(self._cfg), self._dataset)
            self._model.instantiate_optimizers(self._cfg, "cuda" in device)
            self._model.set_pretrained_weights()
            if not self._checkpoint.validate(self._dataset.used_properties):
                log.warning(
                    "The model will not be able to be used from pretrained weights without the corresponding dataset. Current properties are {}".format(
                        self._dataset.used_properties
                    )
                )
        self._checkpoint.dataset_properties = self._dataset.used_properties

        log.info(self._model)

        self._model.log_optimizers()
        log.info("Model size = %i", sum(param.numel() for param in self._model.parameters() if param.requires_grad))

        # Set dataloaders
        self._dataset.create_dataloaders(
            self._model,
            self._cfg.training.batch_size,
            self._cfg.training.shuffle,
            self._cfg.training.num_workers,
            self.precompute_multi_scale,
        )
        log.info(self._dataset)

        # Verify attributes in dataset
        self._model.verify_data(self._dataset.train_dataset[0])

        # Choose selection stage
        selection_stage = getattr(self._cfg, "selection_stage", "")
        self._checkpoint.selection_stage = self._dataset.resolve_saving_stage(selection_stage)
        self._tracker: BaseTracker = self._dataset.get_tracker(self.wandb_log, self.tensorboard_log)

        if self.wandb_log:
            Wandb.launch(self._cfg, not self._cfg.training.wandb.public and self.wandb_log)

        # Run training / evaluation
        self._model = self._model.to(self._device)
        if self.has_visualization:
            self._visualizer = Visualizer(
                self._cfg.visualization, self._dataset.num_batches, self._dataset.batch_size, os.getcwd(), self._tracker
            )

    def train(self):
        self._is_training = True

        for epoch in range(self._checkpoint.start_epoch, self._cfg.training.epochs):
            log.info("EPOCH %i / %i", epoch, self._cfg.training.epochs)

            self._train_epoch(epoch)

            if self.profiling:
                return 0

            if epoch % self.eval_frequency != 0:
                continue

            if self._dataset.has_val_loader:
                self._test_epoch(epoch, "val")

            if self._dataset.has_test_loaders:
                self._test_epoch(epoch, "test")

        # Single test evaluation in resume case
        if self._checkpoint.start_epoch > self._cfg.training.epochs:
            if self._dataset.has_test_loaders:
                self._test_epoch(epoch, "test")

    def eval(self, stage_name=""):
        self._is_training = False

        epoch = self._checkpoint.start_epoch
        if self._dataset.has_val_loader:
            if not stage_name or stage_name == "val":
                self._test_epoch(epoch, "val")

        if self._dataset.has_test_loaders:
            if not stage_name or stage_name == "test":
                self._test_epoch(epoch, "test")

    def _finalize_epoch(self, epoch):
        self._tracker.finalise(**self.tracker_options)
        if self._is_training:
            metrics = self._tracker.publish(epoch)
            self._checkpoint.save_best_models_under_current_metrics(self._model, metrics, self._tracker.metric_func)
            if self.wandb_log and self._cfg.training.wandb.public:
                Wandb.add_file(self._checkpoint.checkpoint_path)
            if self._tracker._stage == "train":
                log.info("Learning rate = %f" % self._model.learning_rate)

    def _train_epoch(self, epoch: int):

        self._model.train()
        self._tracker.reset("train")
        self._visualizer.reset(epoch, "train")
        train_loader = self._dataset.train_dataloader

        with self.profiler_profile(epoch) as prof:
            iter_data_time = time.time()
            with Ctq(train_loader) as tq_train_loader:
                for i, data in enumerate(tq_train_loader):
                    t_data = time.time() - iter_data_time
                    iter_start_time = time.time()

                    with self.profiler_record_function("train_step"):
                        self._model.set_input(data, self._device)
                        self._model.optimize_parameters(epoch, self._dataset.batch_size)

                    with self.profiler_record_function("track/log/visualize"):
                        if i % 10 == 0:
                            with torch.no_grad():
                                self._tracker.track(self._model, data=data, **self.tracker_options)

                        tq_train_loader.set_postfix(
                            **self._tracker.get_metrics(),
                            data_loading=float(t_data),
                            iteration=float(time.time() - iter_start_time),
                            color=COLORS.TRAIN_COLOR
                        )

                        if self._visualizer.is_active:
                            self._visualizer.save_visuals(self._model.get_current_visuals())

                    iter_data_time = time.time()

                    if self.pytorch_profiler_log:
                        pass
                        # prof.step()

                    if self.early_break:
                        break

                    if self.profiling:
                        if i > self.num_batches:
                            return 0

        self._finalize_epoch(epoch)

    def _test_epoch(self, epoch, stage_name: str):
        voting_runs = self._cfg.get("voting_runs", 1)
        if stage_name == "test":
            loaders = self._dataset.test_dataloaders
        else:
            loaders = [self._dataset.val_dataloader]

        self._model.eval()
        if self.enable_dropout:
            self._model.enable_dropout_in_eval()

        for loader in loaders:
            stage_name = loader.dataset.name
            self._tracker.reset(stage_name)
            if self.has_visualization:
                self._visualizer.reset(epoch, stage_name)
            if not self._dataset.has_labels(stage_name) and not self.tracker_options.get(
                "make_submission", False
            ):  # No label, no submission -> do nothing
                log.warning("No forward will be run on dataset %s." % stage_name)
                continue

            with self.profiler_profile(epoch) as prof:
                for i in range(voting_runs):
                    with Ctq(loader) as tq_loader:
                        for data in tq_loader:
                            with torch.no_grad():
                                with self.profiler_record_function("test_step"):
                                    self._model.set_input(data, self._device)
                                    with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
                                        self._model.forward(epoch=epoch)

                                with self.profiler_record_function("track/log/visualize"):
                                    self._tracker.track(self._model, data=data, **self.tracker_options)
                                    tq_loader.set_postfix(**self._tracker.get_metrics(), color=COLORS.TEST_COLOR)

                                    if self.has_visualization and self._visualizer.is_active:
                                        self._visualizer.save_visuals(self._model.get_current_visuals())

                            if self.pytorch_profiler_log:
                                prof.step()

                            if self.early_break:
                                break

                            if self.profiling:
                                if i > self.num_batches:
                                    return 0

            self._finalize_epoch(epoch)
            self._tracker.print_summary()

    @property
    def early_break(self):
        return getattr(self._cfg.debugging, "early_break", False) and self._is_training

    @property
    def profiling(self):
        return getattr(self._cfg.debugging, "profiling", False)

    @property
    def num_batches(self):
        return getattr(self._cfg.debugging, "num_batches", 50)

    @property
    def enable_cudnn(self):
        return getattr(self._cfg.training, "enable_cudnn", True)

    @property
    def enable_dropout(self):
        return getattr(self._cfg, "enable_dropout", True)

    @property
    def has_visualization(self):
        return getattr(self._cfg, "visualization", False)

    @property
    def has_tensorboard(self):
        return getattr(self._cfg.training, "tensorboard", False)

    @property
    def has_training(self):
        return getattr(self._cfg, "training", None)

    @property
    def precompute_multi_scale(self):
        return self._model.conv_type == "PARTIAL_DENSE" and getattr(self._cfg.training, "precompute_multi_scale", False)

    @property
    def wandb_log(self):
        if getattr(self._cfg.training, "wandb", False):
            return getattr(self._cfg.training.wandb, "log", False)
        else:
            return False

    @property
    def tensorboard_log(self):
        if self.has_tensorboard:
            return getattr(self._cfg.training.tensorboard, "log", False)
        else:
            return False

    @property
    def pytorch_profiler_log(self):
        if self.tensorboard_log:
            if getattr(self._cfg.training.tensorboard, "pytorch_profiler", False):
                return getattr(self._cfg.training.tensorboard.pytorch_profiler, "log", False)
        return False

    # pyTorch Profiler
    def profiler_profile(self, epoch):
        if self.pytorch_profiler_log and (
            getattr(self._cfg.training.tensorboard.pytorch_profiler, "nb_epoch", 3) == 0
            or epoch <= getattr(self._cfg.training.tensorboard.pytorch_profiler, "nb_epoch", 3)
        ):
            return torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                if self._cfg.training.cuda > -1
                else [torch.profiler.ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(
                    skip_first=getattr(self._cfg.training.tensorboard.pytorch_profiler, "skip_first", 10),
                    wait=getattr(self._cfg.training.tensorboard.pytorch_profiler, "wait", 5),
                    warmup=getattr(self._cfg.training.tensorboard.pytorch_profiler, "warmup", 3),
                    active=getattr(self._cfg.training.tensorboard.pytorch_profiler, "active", 5),
                    repeat=getattr(self._cfg.training.tensorboard.pytorch_profiler, "repeat", 0),
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self._tracker._tensorboard_dir),
                record_shapes=getattr(self._cfg.training.tensorboard.pytorch_profiler, "record_shapes", True),
                profile_memory=getattr(self._cfg.training.tensorboard.pytorch_profiler, "profile_memory", True),
                with_stack=getattr(self._cfg.training.tensorboard.pytorch_profiler, "with_stack", True),
                with_flops=getattr(self._cfg.training.tensorboard.pytorch_profiler, "with_flops", True),
            )
        else:
            return nullcontext(type("", (), {"step": lambda self: None})())

    def profiler_record_function(self, name: str):
        if self.pytorch_profiler_log:
            return torch.autograd.profiler.record_function(name)
        else:
            return nullcontext()

    @property
    def tracker_options(self):
        return self._cfg.get("tracker_options", {})

    @property
    def eval_frequency(self):
        return self._cfg.get("eval_frequency", 1)
