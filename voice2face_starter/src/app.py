from pathlib import Path

import gradio as gr
import numpy as np
import torch

from data.utils import load_audio, waveform_to_mel
from models.cvae import ConditionalVAE
from utils.io import load_yaml


CFG_PATH = "configs/default.yaml"
CKPT_PATH = "data/checkpoints/best.pt"

cfg = load_yaml(CFG_PATH)
device = torch.device(
    "cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu"
)

model = ConditionalVAE(
    latent_dim=cfg["model"]["latent_dim"],
    audio_embedding_dim=cfg["model"]["audio_embedding_dim"],
).to(device)

checkpoint_loaded = False
if Path(CKPT_PATH).exists():
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    checkpoint_loaded = True


def preprocess_audio(audio_path: str):
    waveform, sr = load_audio(audio_path)
    mel = waveform_to_mel(
        waveform,
        sample_rate=sr,
        n_mels=cfg["data"]["n_mels"],
    )

    mel = mel[:, : cfg["data"]["max_frames"]]

    if mel.shape[1] < cfg["data"]["max_frames"]:
        pad = np.zeros(
            (mel.shape[0], cfg["data"]["max_frames"] - mel.shape[1]),
            dtype=mel.dtype,
        )
        mel = np.concatenate([mel, pad], axis=1)

    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return mel


def tensor_to_image(image_tensor: torch.Tensor):
    image = image_tensor.squeeze(0).detach().cpu().clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def generate_faces(audio_path):
    if not checkpoint_loaded:
        raise gr.Error("No checkpoint found. Train the model first.")

    if audio_path is None:
        raise gr.Error("Please upload an audio file.")

    mel = preprocess_audio(audio_path)

    outputs = []
    with torch.no_grad():
        for i in range(3):
            image = model.generate(mel)
            outputs.append((tensor_to_image(image), f"Generated sample {i+1}"))

    return outputs


demo = gr.Interface(
    fn=generate_faces,
    inputs=gr.Audio(type="filepath", label="Upload a speech clip (.wav)"),
    outputs=gr.Gallery(label="Generated Faces", columns=3, rows=1, height="auto"),
    title="Voice2Face Demo",
    description=(
        "Upload a speech clip and the trained CVAE will generate plausible face images "
        "conditioned on the voice. This demo shows 3 sampled outputs for the same audio."
    ),
)


if __name__ == "__main__":
    demo.launch()