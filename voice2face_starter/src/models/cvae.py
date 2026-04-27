import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio_encoder import AudioEncoder
from models.decoder import FaceDecoder
from models.image_encoder import ImageEncoder


class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim: int = 64, audio_embedding_dim: int = 128):
        super().__init__()
        self.audio_encoder = AudioEncoder(embedding_dim=audio_embedding_dim)
        self.image_encoder = ImageEncoder(latent_dim=latent_dim)
        self.decoder = FaceDecoder(latent_dim=latent_dim, audio_embedding_dim=audio_embedding_dim)
        self.latent_dim = latent_dim

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, mel: torch.Tensor, image: torch.Tensor):
        audio_embedding = self.audio_encoder(mel)
        mu, logvar = self.image_encoder(image)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, audio_embedding)
        return recon, mu, logvar, audio_embedding

    def generate(self, mel: torch.Tensor, z: torch.Tensor | None = None):
        audio_embedding = self.audio_encoder(mel)
        if z is None:
            z = torch.randn(mel.size(0), self.latent_dim, device=mel.device)
        return self.decoder(z, audio_embedding)


def cvae_loss(recon, target, mu, logvar, beta: float = 0.001):
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl
    return total, {"recon_loss": recon_loss.item(), "kl_loss": kl.item()}
