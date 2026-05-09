# NOTE: PLEASE REVIEW FINAL SUBMISSION AUDIO TO FACE GENERATION Folder for final submission


# Voice2Face: Generating Faces from Speech

Voice2Face is a course project that explores whether a model can generate a plausible face image from a speech clip alone. The core system uses a **Conditional Variational Autoencoder (CVAE)**: audio is converted into a mel-spectrogram, encoded into a voice representation, and used to generate a face image conditioned on the speech input.

The final project includes:
- preprocessing paired audio/video data
- mel-spectrogram extraction
- face crop extraction from video frames
- CVAE training and checkpointing
- inference from audio alone
- qualitative comparison tools
- an interactive Gradio demo

## Project Goal

**Input:** speech audio clip  
**Output:** generated face image

During training, the model learns from paired audio and face-image examples. During inference, it takes only audio and generates a face image.

## Dataset

The final implementation uses the **GRID audio-visual corpus**, which provides aligned:
- speech audio (`.wav`)
- face video clips (`.mpg`)

From GRID, the preprocessing pipeline extracts:
- mel-spectrograms from audio
- face crops from video frames

Although the original proposal mentioned VoxCeleb-1, the accessible version initially obtained only contained metadata and split files rather than usable media. GRID was therefore used as the practical implementation dataset.

## Implemented Features

- GRID preprocessing pipeline
- paired manifest generation (`train.csv`, `val.csv`)
- mel-spectrogram extraction
- face-detected image cropping
- Conditional VAE baseline for **voice → face**
- training with checkpointing and saved sample outputs
- single-file inference script
- batch sample generation
- side-by-side comparison grid script
- Gradio demo for audio upload and face generation
- single-speaker and multi-speaker experiment modes

## Extra Criteria Pursued

- **ML pipeline / operations:** preprocessing, manifests, checkpoints, training, inference
- **Gallery GUI:** interactive Gradio demo
- **Generative variation:** multiple generated samples from the same audio
- **Qualitative evaluation:** sample grids and target-vs-generated comparisons

## Repository Layout

```text
voice2face_starter/
├── configs/
│   ├── default.yaml
│   ├── single_speaker.yaml
│   └── multi_speaker.yaml
├── scripts/
│   ├── train.sh
│   └── infer.sh
├── src/
│   ├── app.py
│   ├── evaluate.py
│   ├── generate_samples.py
│   ├── infer.py
│   ├── make_comparison_grid.py
│   ├── train.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocess_grid.py
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
├── requirements.txt
└── README.md
Setup
git clone <your-repo-url>
cd Video_To_Face_GenAI/voice2face_starter
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Expected Data Layout
data/
├── raw/
│   └── grid/
│       ├── audio_25k/
│       │   └── s1/ ... sN/
│       └── video/
│           └── s1/ ... sN/
├── processed/
│   ├── grid/
│   │   ├── faces/
│   │   ├── mels/
│   │   └── manifests/
│   │       ├── train.csv
│   │       └── val.csv
│   └── grid_single/
│       ├── faces/
│       ├── mels/
│       └── manifests/
│           ├── train.csv
│           └── val.csv
├── checkpoints/
├── checkpoints_single/
└── checkpoints_multi/
Preprocessing
Multi-speaker preprocessing
python src/data/preprocess_grid.py \
  --raw_root data/raw/grid \
  --processed_root data/processed/grid \
  --max_speakers 4 \
  --max_utterances_per_speaker 100 \
  --image_size 128
Single-speaker preprocessing
python src/data/preprocess_grid.py \
  --raw_root data/raw/grid \
  --processed_root data/processed/grid_single \
  --max_speakers 1 \
  --max_utterances_per_speaker 1000 \
  --image_size 128

This script:

matches .wav and .mpg files by speaker and utterance id
extracts mel-spectrograms
extracts face crops from video frames
writes train/validation manifests
Training
Single-speaker
python src/train.py --config configs/single_speaker.yaml
Multi-speaker
python src/train.py --config configs/multi_speaker.yaml

The training loop:

loads processed manifests
trains the CVAE
logs reconstruction and KL loss
saves checkpoints
saves generated sample outputs
Inference
Single-speaker inference
python src/infer.py \
  --config configs/single_speaker.yaml \
  --checkpoint data/checkpoints_single/best.pt \
  --audio data/raw/grid/audio_25k/s1/bbaf2n.wav \
  --output outputs/generated_single.png
Multi-speaker inference
python src/infer.py \
  --config configs/multi_speaker.yaml \
  --checkpoint data/checkpoints_multi/best.pt \
  --audio data/raw/grid/audio_25k/s2/bgauza.wav \
  --output outputs/generated_multi.png
Batch Sample Generation
python src/generate_samples.py \
  --config configs/single_speaker.yaml \
  --checkpoint data/checkpoints_single/best.pt \
  --csv data/processed/grid_single/manifests/val.csv \
  --output_dir outputs/final_samples_single \
  --num_samples 12
Comparison Grid
python src/make_comparison_grid.py \
  --csv outputs/final_samples_single/generated_samples.csv \
  --output outputs/comparisons/comparison_grid_single.png \
  --num_samples 8
Demo
python src/app.py

The Gradio demo supports both:

Single-speaker model
Multi-speaker model

A user can upload a speech clip, choose a model version, and view multiple generated face outputs.

Results Summary

The model successfully generates face-like images from voice input. In the multi-speaker setting, the outputs tend to average identities and appear blurrier. In the controlled single-speaker setting, the generated faces become more stable, centered, and recognizable. As expected for a CVAE trained on limited data, outputs remain somewhat soft/blurry, but they clearly resemble human faces rather than random noise.

Main Challenges

Some of the main challenges in this project were:

dataset access issues with the originally proposed VoxCeleb setup
Git/GitHub cleanup after accidentally tracking environment files
blurry VAE outputs
identity averaging in multi-speaker training

To improve results, the project iteratively added:

128×128 image generation
tighter face crops
face-detected preprocessing
single-speaker experiments
L1 reconstruction loss
lower-learning-rate refinement


Future Work
face-to-caricature stylization stage
latent interpolation experiments
stronger decoder / perceptual losses
larger multi-speaker training runs
sharper image generation with more advanced generative models