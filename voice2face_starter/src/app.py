from pathlib import Path

import gradio as gr
import numpy as np
import torch

from data.utils import load_audio, waveform_to_mel
from models.cvae import ConditionalVAE
from utils.io import load_yaml


MODEL_OPTIONS = {
    "Single-speaker": {
        "config": "configs/single_speaker.yaml",
        "checkpoint": "data/checkpoints_single/best.pt",
    },
    "Multi-speaker": {
        "config": "configs/multi_speaker.yaml",
        "checkpoint": "data/checkpoints_multi/best.pt",
    },
}

_loaded_models = {}


def load_model_bundle(model_choice: str):
    if model_choice in _loaded_models:
        return _loaded_models[model_choice]

    paths = MODEL_OPTIONS[model_choice]
    cfg_path = paths["config"]
    ckpt_path = paths["checkpoint"]

    if not Path(cfg_path).exists():
        raise gr.Error(f"Config not found: {cfg_path}")

    if not Path(ckpt_path).exists():
        raise gr.Error(f"Checkpoint not found: {ckpt_path}. Train this model first.")

    cfg = load_yaml(cfg_path)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu"
    )

    model = ConditionalVAE(
        latent_dim=cfg["model"]["latent_dim"],
        audio_embedding_dim=cfg["model"]["audio_embedding_dim"],
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    bundle = {
        "cfg": cfg,
        "device": device,
        "model": model,
    }
    _loaded_models[model_choice] = bundle
    return bundle


def preprocess_audio(audio_path: str, cfg, device):
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


def generate_faces(audio_path, model_choice):
    if audio_path is None:
        raise gr.Error("Please upload an audio file.")

    bundle = load_model_bundle(model_choice)
    cfg = bundle["cfg"]
    device = bundle["device"]
    model = bundle["model"]

    mel = preprocess_audio(audio_path, cfg, device)

    outputs = []
    with torch.no_grad():
        for i in range(3):
            image = model.generate(mel)
            outputs.append((tensor_to_image(image), f"{model_choice} sample {i + 1}"))

    return outputs


demo = gr.Interface(
    fn=generate_faces,
    inputs=[
        gr.Audio(type="filepath", label="Upload a speech clip (.wav)"),
        gr.Dropdown(
            choices=list(MODEL_OPTIONS.keys()),
            value="Single-speaker",
            label="Model version",
        ),
    ],
    outputs=gr.Gallery(label="Generated Faces", columns=3, rows=1, height="auto"),
    title="Voice2Face Demo",
    description=(
        "Upload a speech clip and choose either the single-speaker or multi-speaker model. "
        "The app generates 3 face samples conditioned on the voice input."
    ),
)


if __name__ == "__main__":
    demo.launch()