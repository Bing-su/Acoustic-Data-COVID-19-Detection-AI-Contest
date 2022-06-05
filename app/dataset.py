from typing import Optional

import librosa
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchvision.transforms import Compose, InterpolationMode, Resize


class AudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = True):
        self.df = df.reset_index(drop=True)
        transform = []
        if train:
            transform += [TimeMasking(10), FrequencyMasking(10)]
        transform.append(Resize((224, 224), InterpolationMode.BICUBIC))
        self.transform = Compose(transform)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.loc[idx, "audio"]
        waveform, _ = librosa.load(audio_path, sr=16000)
        spec = librosa.feature.melspectrogram(
            y=waveform, sr=16000, n_fft=1024, hop_length=512, n_mels=128
        )
        spec = librosa.power_to_db(spec, top_db=80)
        spec = torch.from_numpy(spec).float().unsqueeze(0)
        spec = self.transform(spec)

        age = self.df.loc[idx, "age_scaled"]
        gender = self.df.loc[idx, "gender"]
        respiratory = self.df.loc[idx, "respiratory_condition"]
        pain = self.df.loc[idx, "fever_or_muscle_pain"]

        age = torch.tensor(age, dtype=torch.float32)
        gender = torch.tensor(gender, dtype=torch.int)
        respiratory = torch.tensor(respiratory, dtype=torch.int)
        pain = torch.tensor(pain, dtype=torch.int)

        if "covid19" in self.df.columns:
            label = self.df.loc[idx, "covid19"]
        else:
            label = -1
        label = torch.tensor(label, dtype=torch.int)
        return spec, (age, gender, respiratory, pain), label


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, batch_size: int = 32):
        super().__init__()
        self.df = df
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        train_df, val_df = train_test_split(
            self.df, test_size=0.1, random_state=42, stratify=self.df["covid19"]
        )

        self.train_ds = AudioDataset(train_df, train=True)
        self.val_ds = AudioDataset(val_df, train=False)

    def train_dataloader(self):
        loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
        return loader
