import tempfile

import gradio as gr
import torch

from data.utils import load_audio, waveform_to_mel
from models.cvae import ConditionalVAE
from utils.io import load_yaml


CFG_PATH = "configs/default.yaml"
CKPT_PATH = "data/checkpoints/best.pt"


cfg = load_yaml(CFG_PATH)
device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu")
model = ConditionalVAE(
    latent_dim=cfg["model"]["latent_dim"],
    audio_embedding_dim=cfg["model"]["audio_embedding_dim"],
).to(device)

try:
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
except FileNotFoundError:
    checkpoint = None


def preprocess_audio(audio_path: str):
    waveform, sr = load_audio(audio_path)
    mel = waveform_to_mel(waveform, sample_rate=sr, n_mels=cfg["data"]["n_mels"])
    mel = mel[:, : cfg["data"]["max_frames"]]
    if mel.shape[1] < cfg["data"]["max_frames"]:
        import numpy as np
        pad = np.zeros((mel.shape[0], cfg["data"]["max_frames"] - mel.shape[1]), dtype=mel.dtype)
        mel = np.concatenate([mel, pad], axis=1)
    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return mel


def generate_face(audio_path):
    if checkpoint is None:
        raise gr.Error("No checkpoint found yet. Train the model first.")
    mel = preprocess_audio(audio_path)
    with torch.no_grad():
        image = model.generate(mel).squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    return image


iface = gr.Interface(
    fn=generate_face,
    inputs=gr.Audio(type="filepath", label="Upload audio clip"),
    outputs=gr.Image(type="numpy", label="Generated face"),
    title="Voice2Face Demo",
    description="Upload a speech clip and generate a plausible face using the trained CVAE baseline.",
)


if __name__ == "__main__":
    iface.launch()
