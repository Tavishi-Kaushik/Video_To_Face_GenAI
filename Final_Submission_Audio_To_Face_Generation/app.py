from pathlib import Path
import tempfile

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import yaml
import gradio as gr

from PIL import Image, ImageFilter, ImageOps


PROJECT_ROOT = Path.cwd()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio(audio_path: str, sample_rate: int = 16000):
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    return audio, sample_rate


def waveform_to_mel(
    waveform,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
):
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel.astype(np.float32)


def preprocess_audio_for_inference(audio_path: str, cfg):
    waveform, sr = load_audio(audio_path)

    mel = waveform_to_mel(
        waveform,
        sample_rate=sr,
        n_mels=cfg["data"]["n_mels"],
    )

    mel = np.clip(mel, -80, 0)
    mel = (mel + 80) / 80.0

    max_frames = cfg["data"]["max_frames"]
    mel = mel[:, :max_frames]

    if mel.shape[1] < max_frames:
        pad = np.zeros((mel.shape[0], max_frames - mel.shape[1]), dtype=mel.dtype)
        mel = np.concatenate([mel, pad], axis=1)

    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mel


class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.proj(h)


class ImageEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.flatten_dim = 256 * 8 * 8
        self.mu = nn.Linear(self.flatten_dim, latent_dim)
        self.logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.features(x).flatten(1)
        return self.mu(h), self.logvar(h)


class FaceDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, audio_embedding_dim: int = 128):
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


class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim: int = 128, audio_embedding_dim: int = 128):
        super().__init__()

        self.latent_dim = latent_dim

        self.audio_encoder = AudioEncoder(embedding_dim=audio_embedding_dim)
        self.image_encoder = ImageEncoder(latent_dim=latent_dim)

        self.decoder = FaceDecoder(
            latent_dim=latent_dim,
            audio_embedding_dim=audio_embedding_dim,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, mel: torch.Tensor, image: torch.Tensor):
        audio_embedding = self.audio_encoder(mel)

        image_mu, image_logvar = self.image_encoder(image)
        image_z = self.reparameterize(image_mu, image_logvar)

        recon = self.decoder(image_z, audio_embedding)

        return recon, image_mu, image_logvar, audio_embedding

    def generate(self, mel: torch.Tensor) -> torch.Tensor:
        audio_embedding = self.audio_encoder(mel)

        # Since the saved checkpoint does not have audio_mu/audio_logvar,
        # sample z directly from a standard normal latent space.
        z = torch.randn(
            mel.size(0),
            self.latent_dim,
            device=mel.device,
        )

        return self.decoder(z, audio_embedding)


def load_config(mode: str):
    config_name = "single_speaker.yaml" if mode == "Single Speaker" else "multi_speaker.yaml"
    config_path = PROJECT_ROOT / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg, config_path


def load_model(mode: str):
    cfg, config_path = load_config(mode)

    model = ConditionalVAE(
        latent_dim=cfg["model"]["latent_dim"],
        audio_embedding_dim=cfg["model"]["audio_embedding_dim"],
    ).to(DEVICE)

    checkpoint_dir = PROJECT_ROOT / cfg["train"]["checkpoint_dir"]
    checkpoint_path = checkpoint_dir / "best.pt"

    if not checkpoint_path.exists():
        checkpoint_path = checkpoint_dir / "latest.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for {mode}.\n"
            f"Expected best.pt or latest.pt inside: {checkpoint_dir}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    details = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch", "unknown"),
        "val_loss": checkpoint.get("val_loss", "unknown"),
    }

    return model, cfg, details


def tensor_to_pil(image_tensor: torch.Tensor, enhance: bool = True):
    image = image_tensor.squeeze(0).detach().cpu().clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)

    pil_image = Image.fromarray(image)

    if enhance:
        pil_image = pil_image.resize(
            (pil_image.width * 2, pil_image.height * 2),
            Image.Resampling.BICUBIC,
        )
        pil_image = ImageOps.autocontrast(pil_image)
        pil_image = pil_image.filter(
            ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=3)
        )

    return pil_image


def sharpness_score(pil_image: Image.Image):
    arr = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# =========================
# Inference function
# =========================

def generate_faces(audio_file, mode, num_outputs, enhance_images):
    if audio_file is None:
        raise gr.Error("Please upload a .wav audio file first.")

    model, cfg, details = load_model(mode)

    mel = preprocess_audio_for_inference(audio_file, cfg).to(DEVICE)

    outputs = []

    with torch.no_grad():
        for i in range(int(num_outputs)):
            generated = model.generate(mel)
            pil_image = tensor_to_pil(generated, enhance=enhance_images)
            score = sharpness_score(pil_image)
            outputs.append((score, pil_image))

    outputs.sort(key=lambda x: x[0], reverse=True)
    images = [img for _, img in outputs]

    status = (
        f"### Run details\n"
        f"- **Mode:** {mode}\n"
        f"- **Device:** {DEVICE}\n"
        f"- **Checkpoint:** `{details['checkpoint']}`\n"
        f"- **Epoch:** {details['epoch']}\n"
        f"- **Validation loss:** {details['val_loss']}\n\n"
        f"The CVAE samples from a latent distribution, so multiple outputs for the same audio can look slightly different."
    )

    return images, status


# =========================
# Aesthetic UI
# =========================

CUSTOM_CSS = """
.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(167, 139, 250, 0.22), transparent 30%),
        radial-gradient(circle at bottom right, rgba(244, 114, 182, 0.16), transparent 30%),
        linear-gradient(135deg, #faf7ff 0%, #fff7fb 45%, #f8fbff 100%);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#hero {
    text-align: center;
    padding: 34px 28px;
    border-radius: 28px;
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid rgba(180, 160, 255, 0.35);
    box-shadow: 0 20px 60px rgba(110, 80, 180, 0.12);
    backdrop-filter: blur(18px);
    margin-bottom: 20px;
}

#hero h1 {
    font-size: 48px;
    line-height: 1.05;
    margin-bottom: 10px;
    letter-spacing: -1.5px;
    color: #241437;
}

#hero p {
    font-size: 17px;
    color: #6b5b7c;
    margin: 0 auto;
    max-width: 850px;
}

#tag {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    background: linear-gradient(90deg, #ede9fe, #fce7f3);
    color: #6d28d9;
    font-weight: 700;
    font-size: 13px;
    margin-bottom: 14px;
}

.block {
    border-radius: 24px !important;
}

#left-card, #right-card {
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(200, 180, 255, 0.38);
    border-radius: 26px;
    padding: 18px;
    box-shadow: 0 18px 45px rgba(80, 60, 130, 0.10);
}

button.primary {
    border-radius: 16px !important;
    background: linear-gradient(90deg, #7c3aed, #db2777) !important;
    border: none !important;
    color: white !important;
    font-weight: 800 !important;
    box-shadow: 0 14px 30px rgba(124, 58, 237, 0.25) !important;
}

#footer-note {
    text-align: center;
    color: #7c6f8a;
    font-size: 13px;
    margin-top: 18px;
}
"""


with gr.Blocks() as demo:
    gr.HTML(
        """
        <div id="hero">
            <div id="tag">Conditional VAE · GRID Audio-Visual Corpus · Single + Multi Speaker</div>
            <h1>Voice2Face</h1>
            <p>
                Upload a speech clip and generate plausible face images conditioned on voice features.
                Compare the focused single-speaker model with the harder multi-speaker setting.
            </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, elem_id="left-card"):
            gr.Markdown("## Input")
            audio_input = gr.Audio(
                label="Upload speech audio",
                type="filepath",
                sources=["upload", "microphone"],
            )

            mode = gr.Radio(
                choices=["Single Speaker", "Multi Speaker"],
                value="Single Speaker",
                label="Experiment mode",
                info="Single is usually sharper. Multi is harder and can average identities.",
            )

            num_outputs = gr.Slider(
                minimum=1,
                maximum=8,
                value=4,
                step=1,
                label="Number of generated samples",
            )

            enhance_images = gr.Checkbox(
                value=True,
                label="Apply light visual enhancement",
                info="Upscales and sharpens the generated image for display only.",
            )

            generate_button = gr.Button("Generate Face Images", variant="primary")

        with gr.Column(scale=2, elem_id="right-card"):
            gr.Markdown("## Generated Results")
            gallery = gr.Gallery(
                label="Generated face samples",
                columns=4,
                rows=2,
                height=430,
                object_fit="contain",
            )

            status = gr.Markdown(
                "Upload an audio clip, choose single or multi mode, then generate."
            )

    gr.HTML(
        """
        <div id="footer-note">
            Built as a GUI extra-criteria component for the Voice2Face CVAE pipeline.
        </div>
        """
    )

    generate_button.click(
        fn=generate_faces,
        inputs=[audio_input, mode, num_outputs, enhance_images],
        outputs=[gallery, status],
    )


if __name__ == "__main__":
    demo.launch(
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS
)