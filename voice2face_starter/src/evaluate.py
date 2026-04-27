import argparse

import torch
from torch.utils.data import DataLoader

from data.dataset import VoiceFaceDataset
from models.cvae import ConditionalVAE, cvae_loss
from utils.io import load_yaml


def main(args):
    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu")

    dataset = VoiceFaceDataset(
        cfg["data"]["csv_path"],
        image_size=cfg["data"]["image_size"],
        max_frames=cfg["data"]["max_frames"],
    )
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], shuffle=False)

    model = ConditionalVAE(
        latent_dim=cfg["model"]["latent_dim"],
        audio_embedding_dim=cfg["model"]["audio_embedding_dim"],
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            mel = batch["mel"].to(device)
            image = batch["image"].to(device)
            recon, mu, logvar, _ = model(mel, image)
            loss, metrics = cvae_loss(recon, image, mu, logvar, beta=cfg["train"]["beta_kl"])
            total += loss.item()
            count += 1

    print(f"Average loss: {total / max(count, 1):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    main(parser.parse_args())
