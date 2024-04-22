from models.metrics import PseudoFMeasure
from typing import Any, Callable
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
import torch
from torch.optim.optimizer import Optimizer

import wandb

from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics.classification as metrics
from config import CHANNELS, MapType, get_input_slice
from utils.logging import log_image
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from utils.utils import UNNORMALIZE

archs = [
    'unet',
    'unetplusplus',
    'manet',
    'linknet',
    'fpn',
    'pspnet',
    'deeplabv3',
    'deeplabv3plus',
    'pan'
]

IMAGE_TO_LOG = 5


class UNet(pl.LightningModule):
    def __init__(self, arch: str, in_channels: int, out_channels: int, lr: float, dropout: float = 0.0, enc_str: str = None, use_cps: bool = False, gamma: float = 1, patch_size=512):
        """
        :param in_channels: number of channels in the input image
        :param out_channels: number of channels in the result feature map
        """
        super().__init__()

        self.lr = lr
        self.cps = use_cps
        self.gamma = gamma

        aux_params = {'dropout': dropout, 'classes':1}

        self.model_student = smp.create_model(
            arch, enc_str, None, in_channels, out_channels, aux_params=aux_params)
        if self.cps:
            self.model_teacher = smp.create_model(
                arch, enc_str, None, in_channels, out_channels, aux_params=aux_params)

        self.loss = DiceLoss(mode='binary')

        self.batch_to_log = (None, -1)

        self.metrics = nn.ModuleDict({
            'pf_measure': PseudoFMeasure(),
            'jaccard_index': metrics.BinaryJaccardIndex(),
            'dice': metrics.Dice(),
            'f1': metrics.BinaryF1Score(),
            'precision': metrics.BinaryPrecision(),
            'recall': metrics.BinaryRecall(),
            'auroc': metrics.BinaryAUROC()
        })

        self.patch_size = patch_size
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        y1 = self.propagate(x, self.model_student)
        y2 = None
        if self.cps:
            y2 = self.propagate(x, self.model_teacher)
        return y1, y2

    def propagate(self, x, model):
        x = x.clone()
        ys = []
        y = model(x)
        y = y[0] if type(y) is tuple else y
        ys.append(y)
        return ys

    def training_step(self, batch, batch_idx):
        x, y_gt = batch['labelled']
        y_hat_student, y_hat_teacher = self(x)

        loss = self.calc_loss(y_gt, y_hat_student) + \
            (self.calc_loss(y_gt, y_hat_teacher) if self.cps else 0)

        if 'unlabelled' in batch:
            loss_l_cps = 0
            x, _ = batch['unlabelled']
            y_hat_student, y_hat_teacher = self(x)
            y_target_student = torch.sigmoid(
                y_hat_student[-1]).round().detach()
            y_target_teacher = torch.sigmoid(
                y_hat_teacher[-1]).round().detach()
            loss_u_cps = self.calc_loss(y_target_student, y_hat_teacher) + \
                self.calc_loss(y_target_teacher, y_hat_student)
            y_gt = torch.round(y_target_teacher)

            loss = loss + self.gamma*(loss_l_cps + loss_u_cps)

        if batch_idx == 0:
            self.create_image(
                x[:IMAGE_TO_LOG], y_gt[:IMAGE_TO_LOG], y_hat_student[-1][:IMAGE_TO_LOG], "train")

        self.log('train/loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def calc_loss(self, target, preds):
        return sum([self.loss(pred, target)
                    for pred in preds])/len(preds)

    def log_images(self):
        x, target = self.batch_to_log[0]
        sorting = reversed(torch.argsort(
            target.reshape((target.shape[0], -1)).sum(-1)))[:IMAGE_TO_LOG]
        x, target = x[sorting], target[sorting]

        model_device = next(self.parameters()).device
        if x.device != model_device:
            x = x.to(model_device)

        with torch.no_grad():
            preds = self(x)
            preds = self.aggregate_preds(preds)
            # log only first channel
            self.create_image(x, target, preds, "val")

    def create_image(self, x, target, preds, name):
        x = UNNORMALIZE(x)
        channels = get_input_slice([MapType.NX, MapType.NY, MapType.NZ]
                                   ) if MapType.NX in CHANNELS else get_input_slice([MapType.DEPTH])
        x = x[:, channels].clone().detach()
        preds = preds.clone().detach().squeeze()
        target = target.clone().detach().squeeze()
        preds = torch.round(torch.sigmoid(preds))

        image = log_image(x, target, preds)
        self.logger.experiment.log({f'visualization/{name}': wandb.Image(
            image)})

    def validation_step(self, batch, _, name='val'):
        # run validation once at final iteration
        x, target = batch

        # select batch with most annotations to log
        if self.trainer.current_epoch == 0 and target.sum() > self.batch_to_log[1]:
            self.batch_to_log = (batch, target.sum())
        preds = self(x)
        loss = self.calc_loss(target, preds[0]) + \
            (self.calc_loss(target, preds[1]) if self.cps else 0)

        self.log(f'{name}/loss', loss.item(), on_epoch=True,
                 prog_bar=True, sync_dist=True)

        preds = self.aggregate_preds(preds)

        self.calc_metrics(target, preds, name=name)

    def aggregate_preds(self, preds):
        if self.cps:
            preds = (preds[0][-1]+preds[1][-1])/2
        else:
            preds = preds[0][-1]
        return preds

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, name='test')

    def calc_metrics(self, target, preds, name):
        preds = preds.squeeze()
        target = target.squeeze().round().int()

        for metric_name, metric in self.metrics.items():
            metric.update(preds, target)
            self.log(f"metric/{name}/{metric_name}",
                     metric, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self.log_images()
        return super().on_validation_epoch_end()

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer: Optimizer | LightningOptimizer, optimizer_closure: Callable[[], Any] | None = None) -> None:
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True),
            'monitor': 'metric/val/pf_measure',
        }

        return [optimizer], [scheduler]
