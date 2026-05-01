import torch
import torch.nn as nn


class FaceDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64, audio_embedding_dim: int = 128):
        super().__init__()
        in_dim = latent_dim + audio_embedding_dim
        self.fc = nn.Linear(in_dim, 256 * 8 * 8)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, audio_embedding: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, audio_embedding], dim=1)
        h = self.fc(x).view(-1, 256, 8, 8)
        return self.net(h)