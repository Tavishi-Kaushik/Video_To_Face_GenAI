from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def save_image_grid(images: torch.Tensor, save_path: str, nrow: int = 4) -> None:
    grid = make_grid(images.detach().cpu(), nrow=nrow, normalize=True, value_range=(0, 1))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).numpy())
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
