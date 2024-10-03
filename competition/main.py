# %load_ext autoreload
# %autoreload 2

import torch

import torchvision
import torchvision.transforms as transforms
import lightning as L
from dotmap import DotMap
from torch.utils.data import Subset

torch.manual_seed(821)

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from lightning.pytorch.loggers import TensorBoardLogger


# setting up hydra and getting settings from file
with hydra.initialize(version_base=None, config_path="../competition/config"):
    cfg = hydra.compose(config_name="train")
    print(OmegaConf.to_yaml(cfg))


# With lightning, automization of training and validation loop is performed, more functions are possible
class training_model(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = DotMap(cfg)
        self.save_hyperparameters(ignore="model")

    def training_step(self, batch, batch_idx):
        x, _ = batch
        out = self.model(x)
        loss = self.model.loss_fn(**out)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        out = self.model(x)

        if batch_idx == 0:
            # save for visualization
            self.val_samples = out

        loss = self.model.loss_fn(**out)
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        # add code here to visualize generated coordinates on the images and export to tensorboard
        None

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)


# Making model
model = instantiate(cfg.model)
gateModel = training_model(model, OmegaConf.to_container(cfg, resolve=True))

# loading dataset
train_dataset = instantiate(cfg.dataset)
train_set, val_set = torch.utils.data.random_split(train_dataset, [0.85, 0.15])
train_loader = instantiate(cfg.train_loader)(train_set)
val_loader = instantiate(cfg.val_loader)(val_set)

# training
logger = TensorBoardLogger("runs", name="gates", version=1)
trainer = L.Trainer(logger=logger, limit_train_batches=1.0, max_epochs=cfg.max_epochs)
trainer.fit(gateModel, train_dataloaders=train_loader, val_dataloaders=val_loader)
