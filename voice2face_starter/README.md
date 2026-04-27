# Voice2Face2Caricature

Starter codebase for a course project that generates plausible face images from speech using a Conditional Variational Autoencoder (CVAE), with an optional second stage for stylization.

## What is included

- VoxCeleb-style preprocessing pipeline
- Mel-spectrogram extraction from audio
- Face image loading pipeline
- CVAE model scaffold for `voice -> face`
- Training loop with checkpointing and sample generation
- Inference script for generating faces from audio
- Simple Gradio demo
- Config-driven setup

## Planned pipeline

### Stage 1: Voice -> Face
- Input: audio clip
- Preprocess: mel-spectrogram
- Model: Conditional VAE
- Output: plausible face image

### Stage 2: Face -> Caricature
- Optional extension
- Add later as an image-to-image model trained on paired stylization data

---

## Repository layout

```text
voice2face_starter/
├── configs/
│   └── default.yaml
├── scripts/
│   ├── train.sh
│   └── infer.sh
├── src/
│   ├── app.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── train.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocess_voxceleb.py
│   │   └── utils.py
│   ├── models/
│   │   ├── audio_encoder.py
│   │   ├── cvae.py
│   │   ├── decoder.py
│   │   └── image_encoder.py
│   └── utils/
│       ├── io.py
│       ├── seed.py
│       └── visualize.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv
source .venv/bin/activate   # mac/linux
# .venv\Scripts\activate    # windows
pip install -r requirements.txt
```

## Expected data layout

After downloading VoxCeleb-1 / Kaggle assets, organize data like this:

```text
data/
├── raw/
│   ├── videos/
│   └── metadata/
├── processed/
│   ├── faces/
│   ├── mels/
│   └── pairs.csv
└── checkpoints/
```

`pairs.csv` should contain rows like:

```csv
audio_path,image_path,identity
processed/mels/sample_001.npy,processed/faces/sample_001.jpg,id0001
```

---

## Preprocessing

This starter includes a preprocessing script scaffold:

```bash
python src/data/preprocess_voxceleb.py \
  --input_dir data/raw/videos \
  --output_dir data/processed \
  --sample_every_n_frames 10
```

What you still need to do:
1. Put the actual VoxCeleb files in `data/raw/`
2. Run preprocessing
3. Verify that `pairs.csv` is created correctly
4. Inspect saved face crops and mel arrays

---

## Training

```bash
bash scripts/train.sh
```

Or directly:

```bash
python src/train.py --config configs/default.yaml
```

The training script:
- loads config
- builds dataset/dataloaders
- trains the CVAE
- logs reconstruction + KL losses
- saves sample image grids
- writes checkpoints

---

## Inference

```bash
bash scripts/infer.sh path/to/audio.wav outputs/generated.png
```

Or:

```bash
python src/infer.py \
  --config configs/default.yaml \
  --checkpoint data/checkpoints/best.pt \
  --audio path/to/audio.wav \
  --output outputs/generated.png
```

---

## Demo

```bash
python src/app.py
```

This launches a basic Gradio app for audio upload and face generation.

---

## Suggested next milestones before May 7

1. Get preprocessing working on a small subset first
2. Train on tiny data to verify end-to-end pipeline
3. Improve face crops and data pairing quality
4. Add TensorBoard or Weights & Biases logging
5. Add the second-stage caricature model
6. Run interpolation experiments for latent-space analysis
7. Save a gallery of outputs for your final report/demo

---

## Good first commit message

```text
Initial project scaffold for Voice2Face CVAE pipeline
```

## Stronger “big commit” message

```text
Add end-to-end starter pipeline for Voice2Face project

- add VoxCeleb preprocessing scaffold
- add paired dataset loader for audio/image training
- implement conditional VAE baseline
- add train, eval, inference, and demo scripts
- add config, utilities, and project README
```
