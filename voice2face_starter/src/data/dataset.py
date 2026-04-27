from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


class VoiceFaceDataset(Dataset):
    def __init__(self, csv_path: str, image_size: int = 64, max_frames: int = 128):
        self.df = pd.read_csv(csv_path)
        self.max_frames = max_frames
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def _load_mel(self, path: str) -> torch.Tensor:
        mel = np.load(path)
        if mel.ndim == 2:
            mel = mel[:, : self.max_frames]
            if mel.shape[1] < self.max_frames:
                pad = np.zeros((mel.shape[0], self.max_frames - mel.shape[1]), dtype=np.float32)
                mel = np.concatenate([mel, pad], axis=1)
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        return mel

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.image_transform(image)
        mel = self._load_mel(row["audio_path"])

        return {
            "mel": mel,
            "image": image,
            "identity": row["identity"],
        }


def make_splits(csv_path: str, val_split: float = 0.1, seed: int = 42):
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=val_split, random_state=seed, shuffle=True)

    train_csv = Path(csv_path).with_name("train_pairs.csv")
    val_csv = Path(csv_path).with_name("val_pairs.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    return str(train_csv), str(val_csv)
