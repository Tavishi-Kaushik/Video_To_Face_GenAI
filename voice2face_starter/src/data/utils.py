from pathlib import Path

import cv2
import librosa
import numpy as np


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