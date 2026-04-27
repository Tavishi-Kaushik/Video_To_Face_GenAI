import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data.utils import load_audio, waveform_to_mel
from models.cvae import ConditionalVAE
from utils.io import load_yaml


def preprocess_audio(audio_path: str, n_mels: int = 80, max_frames: int = 128):
    waveform, sr = load_audio(audio_path)
    mel = waveform_to_mel(waveform, sample_rate=sr, n_mels=n_mels)
    mel = mel[:, :max_frames]
    if mel.shape[1] < max_frames:
        import numpy as np
        pad = np.zeros((mel.shape[0], max_frames - mel.shape[1]), dtype=mel.dtype)
        mel = np.concatenate([mel, pad], axis=1)
    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mel


def save_tensor_image(image: torch.Tensor, output_path: str):
    img = image.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(img)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main(args):
    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu")

    model = ConditionalVAE(
        latent_dim=cfg["model"]["latent_dim"],
        audio_embedding_dim=cfg["model"]["audio_embedding_dim"],
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mel = preprocess_audio(
        args.audio,
        n_mels=cfg["data"]["n_mels"],
        max_frames=cfg["data"]["max_frames"],
    ).to(device)

    with torch.no_grad():
        generated = model.generate(mel)

    save_tensor_image(generated, args.output)
    print(f"Saved generated face to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    main(parser.parse_args())
