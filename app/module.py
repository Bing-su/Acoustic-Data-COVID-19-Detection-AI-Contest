import pytorch_lightning as pl
import torch
from timm import create_model
from timm.optim import create_optimizer_v2
from torch import nn
from torchmetrics import F1Score


class AudioClassificationModule(pl.LightningModule):
    def __init__(
        self, model_name: str, embedding_dim: int = 8, optimizer_name: str = "madgradw"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(
            model_name, pretrained=True, in_chans=1, num_classes=128
        )
        self.optimizer_name = optimizer_name

        self.emb1 = nn.Embedding(2, embedding_dim)
        self.emb2 = nn.Embedding(2, embedding_dim)
        self.emb3 = nn.Embedding(2, embedding_dim)

        self.data_emb = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 1, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
        )

        self.head = nn.Linear(128 + 32, 2)

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_f1 = F1Score(num_classes=2)
        self.val_f1 = F1Score(num_classes=2)

    def forward(self, batch):
        img, data, _ = batch
        img_emb = self.model(img)  # (batch_size, 128)

        age, cat1, cat2, cat3 = data
        age = age.unsqueeze(1)
        cat1 = self.emb1(cat1)
        cat2 = self.emb2(cat2)
        cat3 = self.emb3(cat3)

        data_cat = torch.cat([age, cat1, cat2, cat3], dim=1)
        data_emb = self.data_emb(data_cat)  # (batch_size, 32)

        x = torch.cat([img_emb, data_emb], dim=1)
        out = self.head(x)  # (batch_size, 2)
        return out

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        *_, label = batch

        loss = self.loss_fn(pred, label)
        f1 = self.train_f1(pred, label)

        self.log_dict(
            {"train_loss": loss, "train_f1": f1},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_id):
        pred = self(batch)
        *_, label = batch

        loss = self.loss_fn(pred, label)
        f1 = self.val_f1(pred, label)

        self.log_dict({"val_loss": loss, "val_f1": f1}, prog_bar=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        pred = self(batch)
        output = torch.argmax(pred, dim=-1)
        return output

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.parameters(), opt=self.optimizer_name, weight_decay=1e-4
        )
        return optimizer
