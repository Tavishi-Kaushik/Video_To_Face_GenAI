import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import VoiceFaceDataset, make_splits
from models.cvae import ConditionalVAE, cvae_loss
from utils.io import ensure_dir, load_yaml
from utils.seed import set_seed
from utils.visualize import save_image_grid


@torch.no_grad()
def validate(model, loader, device, beta_kl):
    model.eval()
    total_loss = 0.0
    count = 0
    sample_batch = None

    for batch in loader:
        mel = batch["mel"].to(device)
        image = batch["image"].to(device)
        recon, mu, logvar, _ = model(mel, image)
        loss, _ = cvae_loss(recon, image, mu, logvar, beta=beta_kl)
        total_loss += loss.item()
        count += 1
        if sample_batch is None:
            sample_batch = recon[:8]

    return total_loss / max(count, 1), sample_batch


def main(args):
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    csv_path = cfg["data"]["csv_path"]
    train_csv, val_csv = make_splits(csv_path, val_split=cfg["data"]["val_split"], seed=cfg["seed"])

    train_ds = VoiceFaceDataset(train_csv, image_size=cfg["data"]["image_size"], max_frames=cfg["data"]["max_frames"])
    val_ds = VoiceFaceDataset(val_csv, image_size=cfg["data"]["image_size"], max_frames=cfg["data"]["max_frames"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    requested_device = cfg["train"]["device"]
    device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")

    model = ConditionalVAE(
        latent_dim=cfg["model"]["latent_dim"],
        audio_embedding_dim=cfg["model"]["audio_embedding_dim"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
    checkpoint_dir = ensure_dir(cfg["train"]["checkpoint_dir"])
    sample_dir = ensure_dir(cfg["train"]["sample_dir"])

    best_val_loss = float("inf")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['train']['epochs']}")

        for step, batch in enumerate(progress):
            mel = batch["mel"].to(device)
            image = batch["image"].to(device)

            optimizer.zero_grad()
            recon, mu, logvar, _ = model(mel, image)
            loss, metrics = cvae_loss(recon, image, mu, logvar, beta=cfg["train"]["beta_kl"])
            loss.backward()
            optimizer.step()

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{metrics['recon_loss']:.4f}",
                kl=f"{metrics['kl_loss']:.4f}",
            )

        val_loss, sample_batch = validate(model, val_loader, device, cfg["train"]["beta_kl"])
        print(f"Epoch {epoch + 1}: val_loss={val_loss:.4f}")

        if (epoch + 1) % cfg["train"]["sample_every_epoch"] == 0 and sample_batch is not None:
            save_image_grid(sample_batch, str(sample_dir / f"epoch_{epoch + 1:03d}.png"))

        latest_path = Path(checkpoint_dir) / "latest.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            },
            latest_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = Path(checkpoint_dir) / "best.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": cfg,
                },
                best_path,
            )
            print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    main(parser.parse_args())
