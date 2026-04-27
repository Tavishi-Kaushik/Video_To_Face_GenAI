from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import librosa
import numpy as np
from tqdm import tqdm


def extract_middle_frame(video_path: Path, output_image_path: Path, image_size: int = 64) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return False

    middle_idx = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return False

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (image_size, image_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image_path), frame)
    return True


def compute_mel(wav_path: Path, sample_rate: int = 16000, n_mels: int = 80) -> np.ndarray:
    audio, sr = librosa.load(str(wav_path), sr=sample_rate)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def get_matching_pairs(audio_speaker_dir: Path, video_speaker_dir: Path):
    wavs = {p.stem: p for p in audio_speaker_dir.glob("*.wav")}
    vids = {p.stem: p for p in video_speaker_dir.glob("*.mpg")}
    common = sorted(set(wavs.keys()) & set(vids.keys()))
    return [(stem, wavs[stem], vids[stem]) for stem in common]


def write_manifest(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "speaker_id",
                "utterance_id",
                "audio_path",
                "video_path",
                "mel_path",
                "image_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    random.seed(args.seed)

    raw_root = Path(args.raw_root)
    audio_root = raw_root / "audio_25k"
    video_root = raw_root / "video"

    processed_root = Path(args.processed_root)
    mel_root = processed_root / "mels"
    face_root = processed_root / "faces"
    manifest_root = processed_root / "manifests"

    speakers = sorted([p.name for p in video_root.iterdir() if p.is_dir()])
    if args.max_speakers is not None:
        speakers = speakers[: args.max_speakers]

    all_rows = []

    for speaker_id in tqdm(speakers, desc="Speakers"):
        audio_speaker_dir = audio_root / speaker_id
        video_speaker_dir = video_root / speaker_id

        if not audio_speaker_dir.exists() or not video_speaker_dir.exists():
            continue

        pairs = get_matching_pairs(audio_speaker_dir, video_speaker_dir)

        if args.max_utterances_per_speaker is not None:
            pairs = pairs[: args.max_utterances_per_speaker]

        for utterance_id, wav_path, video_path in tqdm(
            pairs, desc=f"{speaker_id}", leave=False
        ):
            mel = compute_mel(wav_path, sample_rate=args.sample_rate, n_mels=args.n_mels)

            mel_out = mel_root / speaker_id / f"{utterance_id}.npy"
            mel_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(mel_out, mel)

            img_out = face_root / speaker_id / f"{utterance_id}.png"
            ok = extract_middle_frame(video_path, img_out, image_size=args.image_size)
            if not ok:
                continue

            all_rows.append(
                {
                    "speaker_id": speaker_id,
                    "utterance_id": utterance_id,
                    "audio_path": str(wav_path),
                    "video_path": str(video_path),
                    "mel_path": str(mel_out),
                    "image_path": str(img_out),
                }
            )

    random.shuffle(all_rows)
    split_idx = int(len(all_rows) * (1.0 - args.val_ratio))
    train_rows = all_rows[:split_idx]
    val_rows = all_rows[split_idx:]

    write_manifest(train_rows, manifest_root / "train.csv")
    write_manifest(val_rows, manifest_root / "val.csv")

    print(f"Total samples: {len(all_rows)}")
    print(f"Train samples: {len(train_rows)}")
    print(f"Val samples: {len(val_rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default="data/raw/grid")
    parser.add_argument("--processed_root", type=str, default="data/processed/grid")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_speakers", type=int, default=4)
    parser.add_argument("--max_utterances_per_speaker", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)