# %load_ext autoreload
# %autoreload 2

import torch
import os

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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# setting up hydra and getting settings from file
with hydra.initialize(version_base=None, config_path="../competition/config"):
    cfg = hydra.compose(config_name="train")
    print(OmegaConf.to_yaml(cfg))


# With lightning, automization of training and validation loop is performed, more functions are possible
class training_model(L.LightningModule):
    def __init__(self, model: torch.nn.Module, cfg: dict) -> None:
        super().__init__()
        self.model = model
        self.cfg = DotMap(cfg)
        self.save_hyperparameters(ignore="model")

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, gt = batch
        out = self.model(x)
        loss = self.model.loss_fn(out, gt)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, gt = batch
        out = self.model(x)

        if batch_idx == 0:
            # save for visualization
            self.val_samples_x = x[0:5]
            self.val_samples_gt = gt[0:5]
            self.val_samples_res = out[0:5]

        loss = self.model.loss_fn(out, gt)
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        # add code here to visualize generated coordinates on the images and export to tensorboard
        x = self.val_samples_x
        res = self.val_samples_res
        gt = self.val_samples_gt

        # x = self.val_samples_x
        # res = self.val_samples_gt
        # gt = self.val_samples_x

        for i in range(5):
            gatecoor = torch.Tensor([])
            coorgt = torch.Tensor([])
            for j in range(9):
                for k in range(4):
                    if res[i][12 * j + 3 * k + 2] > 0.9:
                        coor = res[i][12 * j + 3 * k : 12 * j + 3 * k + 2]
                        cgt = gt[i][12 * j + 3 * k : 12 * j + 3 * k + 2]

                        if coor[0] < 0:
                            coor[0] = 0
                        else:
                            coor[0] = coor[0] * x[i].size()[2]
                        if coor[1] < 0:
                            coor[1] = 0
                        else:
                            coor[1] = coor[1] * x[i].size()[1]

                        cgt[0] = cgt[0] * x[i].size()[2]
                        cgt[1] = cgt[1] * x[i].size()[1]

                        if gatecoor.size()[0] == 0:
                            gatecoor = coor.expand(1, -1)
                        else:
                            gatecoor = torch.cat(
                                (
                                    gatecoor,
                                    coor.expand(1, -1),
                                )
                            )

                        if coorgt.size()[0] == 0:
                            coorgt = cgt.expand(1, -1)
                        else:
                            coorgt = torch.cat(
                                (
                                    coorgt,
                                    cgt.expand(1, -1),
                                )
                            )

            if gatecoor.size()[0] > 0:
                # connec = []
                # for k in range(gatecoor.size()[0]):
                #     connec.append((k, k+1))
                # print(gatecoor.size())
                x[i] = torchvision.utils.draw_keypoints(
                    x[i], keypoints=gatecoor.expand(1, -1, -1), colors="red"
                )
            # if coorgt.size()[0] > 0:
            #     # connec = []
            #     # for k in range(coorgt.size()[0]):
            #     #     connec.append((k - 1, k))

            #     x[i] = torchvision.utils.draw_keypoints(
            #         x[i],
            #         keypoints=coorgt.expand(1, -1, -1),
            #         # connectivity=connec,
            #         colors="green",
            #     )

        grid = torchvision.utils.make_grid(x[0:5], nrow=5)
        self.logger.experiment.add_image("Recognized gates", grid, self.current_epoch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.lr)


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
