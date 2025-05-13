from typing import Optional

import torch
from lightning import pytorch as pl

from train.architectures import Architecture
from train.callbacks import ModelCheckpoint, PsdPlotter
from train.metrics import  OfflinePsdRatio, OnlinePsdRatio, PsdRatio

Tensor = torch.Tensor


class DeepClean(pl.LightningModule):
    def __init__(
        self,
        arch: Architecture,
        loss: PsdRatio,
        metric: OfflinePsdRatio,
        patience: Optional[int] = None,
        save_top_k_models: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["arch", "loss", "metric"])

        self.model = arch
        self.loss = loss
        self.metric = metric
        self.metric.loss_fn = self.loss
        self._plots = {}

    def forward(self, X: Tensor) -> Tensor:
        return self.model(X)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        X, y_true = batch
        y_pred = self(X)
        loss = self.loss(y_pred, y_true).mean()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def _shared_eval_step(self, X, y_true) -> None:
        """
        Note that the actual computation of the loss function
        happens via the PsdPlotter Callback
        """

        if y_true is not None:
            self.metric.update(y_true[:, 0], "strain")
        if X is not None:
            y_pred = self(X)
            self.metric.update(y_pred, "predictions")

    def validation_step(self, batch, _) -> None:
        return self._shared_eval_step(*batch)

    def test_step(self, batch, _) -> None:
        return self._shared_eval_step(*batch)

    def predict_step(self, batch, _) -> None:
        return self._shared_eval_step(*batch)

    def configure_callbacks(self) -> list[pl.Callback]:
        # first callback actually computes all of our
        # validation metrics and any associated plots
        callbacks = [PsdPlotter(self._plots)]

        # then tack on a checkpointer that uses these
        # metrcis for checkpointing the model
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=self.hparams.save_top_k_models,
            save_last=True,
            auto_insert_metric_name=False,
            mode="min",
            save_on_train_epoch_end=False,
        )
        callbacks.append(checkpoint)

        # if we specified an early-stopping patience
        # interval, add early stopping
        if self.hparams.patience is not None:
            early_stop = pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.hparams.patience,
                mode="min",
                min_delta=0.00,
            )
            callbacks.append(early_stop)
        return callbacks
