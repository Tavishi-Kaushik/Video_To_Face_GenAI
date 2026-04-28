from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
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

        if mel.ndim != 2:
            raise ValueError(f"Expected mel spectrogram with 2 dims, got shape {mel.shape} from {path}")

        mel = mel[:, : self.max_frames]

        if mel.shape[1] < self.max_frames:
            pad = np.zeros(
                (mel.shape[0], self.max_frames - mel.shape[1]),
                dtype=np.float32,
            )
            mel = np.concatenate([mel, pad], axis=1)

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        return mel

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image = self.image_transform(image)

        mel = self._load_mel(row["mel_path"])

        return {
            "mel": mel,
            "image": image,
            "speaker_id": row["speaker_id"],
            "utterance_id": row["utterance_id"],
        }


def get_dataset(csv_path: str, image_size: int = 64, max_frames: int = 128):
    return VoiceFaceDataset(
        csv_path=csv_path,
        image_size=image_size,
        max_frames=max_frames,
    )