from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from configs import TrainConfig


class AndiModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=5)
        self.val_acc = Accuracy(task="multiclass", num_classes=5)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        if y is None:
            return None, preds, None
        loss = self.loss_fn(logits, y)

        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        _, preds, _ = self.model_step(batch)

        # Create the sample submission file
        sample_submission = []
        for i, class_label in enumerate(preds):
            sample_submission.append(f"{i},{class_label.item()}")

        # Write the sample submission to a file
        with open("../sample_submission.csv", "w") as f:
            f.write("index,class\n")
            f.write("\n".join(sample_submission))

    # def configure_optimizers(self):
    #     # optimizer = self.hparams.optimizer(params=self.parameters())
    #     # if self.hparams.scheduler is not None:
    #     #     scheduler = self.hparams.scheduler(optimizer=optimizer)
    #     #     return {
    #     #         "optimizer": optimizer,
    #     #         "lr_scheduler": {
    #     #             "scheduler": scheduler,
    #     #             "monitor": "val/loss",
    #     #             "interval": "epoch",
    #     #             "frequency": 1,
    #     #         },
    #     #     }

    #     # UNSUPPORTED KEYS - TRZEBA ZMIENIC
    #     return {"optimizer": self.optimizer, "scheduler": self.scheduler, "interval":"epoch"}
    def configure_optimizers(self):
        config = TrainConfig()
        optimizer = self.optimizer(self.parameters(), lr=config.lr)
        scheduler = self.scheduler(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    _ = AndiModule(None, None, None)
