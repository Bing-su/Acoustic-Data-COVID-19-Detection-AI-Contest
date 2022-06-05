import json
import sys

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from app.dataset import AudioDataModule, AudioDataset
from app.module import AudioClassificationModule


def train():
    with open("config.json") as file:
        cfg = json.load(file)

    module = AudioClassificationModule(
        cfg["model_name"], cfg["embedding_dim"], cfg["optimizer_name"]
    )
    train_df = pd.read_csv("data/train_df.csv")
    datamodule = AudioDataModule(train_df)

    logger = WandbLogger(name="dacon-covid19-audio-classification")

    trainer = pl.Trainer(
        accelerator="auto",
        gpus=1,
        logger=logger,
        log_every_n_steps=30,
        callbacks=[RichProgressBar()],
        max_epochs=cfg["epochs"],
        auto_scale_batch_size=True,
    )

    trainer.fit(module, datamodule=datamodule)

    test_df = pd.read_csv("data/test_df.csv")
    test_dataset = AudioDataset(test_df, train=False)
    num_workers = 0 if sys.platform != "linux" else 10
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=num_workers
    )

    pred = trainer.predict(module, dataloaders=test_loader)
    output = torch.cat(pred, dim=-1)

    torch.save(output, "output.pt")

    submission = pd.read_csv("data/sample_submission.csv")
    output = output.cpu().numpy()
    submission["covid19"] = output
    submission.to_csv("submission01.csv", index=False)


if __name__ == "__main__":
    train()
