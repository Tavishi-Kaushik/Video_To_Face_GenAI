from pathlib import Path

import cv2
import numpy as np
import torchaudio


def load_audio(audio_path: str, sample_rate: int = 16000):
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def waveform_to_mel(
    waveform,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel = mel_transform(waveform)
    mel = torchaudio.transforms.AmplitudeToDB()(mel)
    return mel.squeeze(0).numpy().astype(np.float32)


def center_crop_face(frame, size: int = 128):
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = frame[y0:y0 + side, x0:x0 + side]
    crop = cv2.resize(crop, (size, size))
    return crop


def save_npy(array: np.ndarray, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
