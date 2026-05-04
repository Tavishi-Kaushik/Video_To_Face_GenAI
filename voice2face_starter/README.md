# Voice2Face: Generating Faces from Speech

Voice2Face is a course project that explores whether a model can generate a plausible face image from a speech clip alone. The system uses a **Conditional Variational Autoencoder (CVAE)**: audio is converted into a mel-spectrogram, encoded, and used to generate a face image conditioned on the voice.

The final pipeline is end-to-end and includes:
- preprocessing paired audio/video data
- mel-spectrogram extraction
- face crop extraction
- CVAE training
- inference from audio alone
- qualitative comparison tools
- an interactive Gradio demo

## Task

**Input:** speech audio clip  
**Output:** generated face image

During training, the model learns from paired audio and face-image examples. During inference, it takes only audio and generates a face.

## Dataset

The final implementation uses the **GRID audio-visual corpus**, which provides aligned:
- speech audio (`.wav`)
- face video clips (`.mpg`)

From GRID, the preprocessing pipeline extracts:
- mel-spectrograms from audio
- face crops from video frames

## Implemented Features

- GRID preprocessing pipeline
- paired manifest generation (`train.csv`, `val.csv`)
- mel-spectrogram extraction
- face-detected image cropping
- Conditional VAE baseline for **voice → face**
- training with checkpointing and sample saving
- single-file inference script
- batch sample generation
- side-by-side comparison grid script
- Gradio demo for audio upload and face generation

## Extra Criteria Pursued

- **ML pipeline / operations:** preprocessing, manifests, checkpoints, training, inference
- **Gallery GUI:** interactive Gradio demo
- **Generative variation:** multiple generated samples from the same audio
- **Qualitative evaluation:** sample grids and target-vs-generated comparisons

## Repository Layout

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
│   ├── generate_samples.py
│   ├── infer.py
│   ├── make_comparision_grid.py
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