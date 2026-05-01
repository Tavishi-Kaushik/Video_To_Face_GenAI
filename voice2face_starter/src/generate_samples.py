import argparse
from pathlib import Path

import pandas as pd
import torch

from infer import preprocess_audio, save_tensor_image
from models.cvae import ConditionalVAE
from utils.io import load_yaml


def main(args):
    cfg = load_yaml(args.config)
    df = pd.read_csv(args.csv)

    if args.num_samples is not None:
        df = df.head(args.num_samples)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu"
    )

    model = ConditionalVAE(
        latent_dim=cfg["model"]["latent_dim"],
        audio_embedding_dim=cfg["model"]["audio_embedding_dim"],
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_rows = []

    for _, row in df.iterrows():
        speaker_id = row["speaker_id"]
        utterance_id = row["utterance_id"]
        audio_path = row["audio_path"]

        mel = preprocess_audio(
            audio_path,
            n_mels=cfg["data"]["n_mels"],
            max_frames=cfg["data"]["max_frames"],
        ).to(device)

        with torch.no_grad():
            generated = model.generate(mel)

        output_path = output_dir / f"{speaker_id}_{utterance_id}_generated.png"
        save_tensor_image(generated, str(output_path))

        saved_rows.append(
            {
                "speaker_id": speaker_id,
                "utterance_id": utterance_id,
                "audio_path": audio_path,
                "target_image_path": row["image_path"],
                "generated_image_path": str(output_path),
            }
        )

        print(f"Saved {output_path}")

    pd.DataFrame(saved_rows).to_csv(output_dir / "generated_samples.csv", index=False)
    print(f"\nSaved sample manifest to {output_dir / 'generated_samples.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=12)
    main(parser.parse_args())