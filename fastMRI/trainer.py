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

from dataset import fastMRIDataset

# from fastmri.evaluate import ssim, psnr, nmse
from fastmri.losses import SSIMLoss

DATASET_PATH = "/home/haneol.kijm/Works/data/ImageNet_tar"
# CHECKPOINT_PATH=os.environ.get("PATH_CHECKPOINT", "saved_models/SimpleViT/")
CHECKPOINT_PATH = "/home/haneol.kijm/Works/template_torch_lightning/saved_models/MLPMixer/"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class LitModel_fastmri(pl.LightningModule):
    def __init__(self, config, data_dir=None) -> None:
        super().__init__()
        # self.save_hyperparameters()
        model_name = config["model_name"]
        print(f"Initializing {model_name} model...")
        print(config)
        sys.path.append("/home/haneol.kijm/Works/git/imaging_MLPs/")
        model_mod = importlib.import_module(f"networks.{model_name}")
        self.net = model_mod.Model(**config)
        from networks.recon_net import ReconNet
        self.model = ReconNet(self.net, config["patch_size"])

        self.data_dir = data_dir or os.getcwd()
        self.dataset_size = config["dataset_size"]
        self.val_size = config["val_size"]

        self.image_size = config["img_size"]
        self.lr = config["lr"]
        self.wd = config["wd"]
        self.batch_size = config["batch_size"]
        self.loss_fn = SSIMLoss()
        self.max_epochs = config["max_epochs"]
        self.warmup = config["warmup"]

        # self.model = model

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
        inputs, targets, maxval = batch
        # preds = self.model(images, training= True if mode=="train" else False)
        outputs = self.model(inputs)
        # finite_checks.detect_nan_parameters(self.model)
        loss = self.loss_fn(outputs, targets, maxval)
        self.log("ptl/%s_loss" % mode, loss, batch_size=self.batch_size)
        if mode == "val":
            self.log("ptl/val_SSIM", 1-loss, batch_size=self.batch_size, on_step=False, on_epoch=True)

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

    # def prepare_data(self):
    #     self.clean_train_path = self.data_dir + "clean_train"
    #     self.noisy_train_path = self.data_dir + "noisy_train"
    #     self.clean_eval_path = self.data_dir + "clean_val"
    #     self.noisy_eval_path = self.data_dir + "noisy_val"

    def train_dataloader(self):
        dataset = fastMRIDataset(isval=False)
        ntrain = self.dataset_size
        train_dataset, _ = data.random_split(dataset, [ntrain, len(dataset)-ntrain], generator=torch.Generator().manual_seed(1234))
        # print("dataset size: ", len(train_dataset))
        return data.DataLoader(
            train_dataset, 
            batch_size=1, 
            shuffle=True, 
            num_workers=4, 
            generator=torch.Generator().manual_seed(1234))
      
    def val_dataloader(self):
        dataset = fastMRIDataset(isval=True)
        nval = self.val_size if self.val_size!=0 else len(dataset)
        val_dataset, _ = data.random_split(dataset, [nval, len(dataset)-nval], generator=torch.Generator().manual_seed(1234))
        # print("dataset size: ", len(val_dataset))
        
        return data.DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=4)
 