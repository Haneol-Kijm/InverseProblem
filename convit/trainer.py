import os, sys, importlib

import numpy as np

import torch
from torch import nn
import torch.utils.data as data

from torchmetrics import PeakSignalNoiseRatio

import pytorch_lightning as pl

# from pytorch_lightning.utilities import finite_checks
# from pytorch_lightning.callbacks import LearningRateMonitor, Timer, EarlyStopping, ModelCheckpoint
# from pytorch_lightning.plugins.environments import SLURMEnvironment

import timm

from dataset import PairDataset, RandomTransform, PairCenterCrop

DATASET_PATH = "/home/haneol.kijm/Works/data/ImageNet_tar"
# CHECKPOINT_PATH=os.environ.get("PATH_CHECKPOINT", "saved_models/SimpleViT/")
CHECKPOINT_PATH = "/home/haneol.kijm/Works/template_torch_lightning/saved_models/MLPMixer/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class LitModel(pl.LightningModule):
    def __init__(self, config, data_dir=None) -> None:
        super().__init__()
        # self.save_hyperparameters()
        model_name = config["model_name"]
        print(f"Initializing {model_name} model...")
        print(config)
        sys.path.append("/home/haneol.kijm/Works/git/imaging_MLPs/")
        model_mod = importlib.import_module(f"networks.{model_name}")
        self.model = model_mod.Model(**config)

        self.data_dir = data_dir or os.getcwd()

        self.image_size = config["img_size"]
        self.lr = config["lr"]
        self.wd = config["wd"]
        self.batch_size = config["batch_size"]
        if config["loss_fn"] == "l1":
            self.loss_fn = nn.L1Loss()
        elif config["loss_fn"] == "l2":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("Wrong loss name")
        self.max_epochs = config["max_epochs"]
        self.warmup = config["warmup"]

        # self.model = model

        self.valid_psnr = PeakSignalNoiseRatio()

    def forward(self, input):
        return self.model(input)

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        optimizer = timm.optim.create_optimizer_v2(
            self.model, opt="lookahead_AdamW", lr=self.lr, weight_decay=self.wd
        )
        # self.scheduler, _ = timm.scheduler.create_scheduler_v2(
        #     optimizer,
        #     sched="cosine",
        #     num_epochs=self.max_epochs,
        #     warmup_epochs=self.warmup,
        # )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # if self.scheduler is not None:
        #     self.scheduler.step_update(num_updates=self.global_step)  ## step per iteration

    def _calculate_loss(self, batch, mode="train"):
        clean_imgs, noisy_imgs = batch
        # preds = self.model(images, training= True if mode=="train" else False)
        outputs = self.model(noisy_imgs)
        # finite_checks.detect_nan_parameters(self.model)
        denoised_imgs = torch.clamp(noisy_imgs - outputs, 0, 1)   ## outputs are learning the noise itself
        loss = self.loss_fn(denoised_imgs, clean_imgs)
        self.log("ptl/%s_loss" % mode, loss, batch_size=self.batch_size)
        # self.log(
        #     "%s_lr" % mode,
        #     self.scheduler._get_lr(self.current_epoch)[0],
        #     batch_size=self.batch_size,
        #     prog_bar=True,
        # )
        if mode == "val":
            self.valid_psnr(denoised_imgs, clean_imgs)
            self.log("ptl/val_psnr", self.valid_psnr, batch_size=self.batch_size)

        # finite_checks.detect_nan_parameters(self.model)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def training_epoch_end(self, training_step_outputs):

        if hasattr(self.optimizers, "sync_lookahead"):
            self.optimizers.sync_lookahead()

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    # def validation_epoch_end(self, validation_step_outputs):
        # if self.scheduler is not None:
        #     self.scheduler.step(self.current_epoch + 1)

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    def prepare_data(self):
        self.clean_train_path = self.data_dir + "clean_train"
        self.noisy_train_path = self.data_dir + "noisy_train"
        self.clean_eval_path = self.data_dir + "clean_val"
        self.noisy_eval_path = self.data_dir + "noisy_val"

    def train_dataloader(self):
        train_dataset = PairDataset(
            self.clean_train_path,
            self.noisy_train_path,
            split="train",
            transform=RandomTransform(self.image_size),
        )
        return data.DataLoader(
            train_dataset,
            batch_size=int(self.batch_size),
            shuffle=True,
            num_workers=8,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        val_dataset = PairDataset(
            self.clean_eval_path,
            self.noisy_eval_path,
            split="eval",
            transform=PairCenterCrop(self.image_size),
        )
        return data.DataLoader(val_dataset, batch_size=int(self.batch_size), num_workers=8)
