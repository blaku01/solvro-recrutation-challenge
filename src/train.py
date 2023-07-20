import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

import wandb
from configs import TrainConfig
from data.datamodule import AndiDataModule
from models.andi_module import AndiModule
from models.nets.BiLSTM_CNN_net import BiLSTM_CNN_net

if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    # Parse command line arguments
    # Load the configuration
    config = TrainConfig()
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="model_checkpoints/checkpoints/",
        filename="bi-lstm-{epoch:02d}-{val_loss:.2f}",
    )
    # Initialize WandB if enabled
    if config.wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            entity=config.entity,
        )

    # Initialize the PyTorch Lightning logger if enabled
    logger = CSVLogger("logs", name="BiLSTM")
    if config.wandb:
        logger = WandbLogger()

    # Create the AndiDataModule
    data_module = AndiDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    net = BiLSTM_CNN_net()
    # Initialize the PyTorch Lightning module
    andi_module = AndiModule(
        net=net,
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.StepLR,
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="gpu" if config.gpu else "cpu",
        devices=config.devices,
        default_root_dir=config.save_model_path,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(andi_module, datamodule=data_module)
