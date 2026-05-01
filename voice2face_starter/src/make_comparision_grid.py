import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def load_image(path):
    return Image.open(path).convert("RGB")


def main(args):
    df = pd.read_csv(args.csv)

    if args.num_samples is not None:
        df = df.head(args.num_samples)

    rows = len(df)
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(6, 3 * rows))

    if rows == 1:
        axes = [axes]

    for i, (_, row) in enumerate(df.iterrows()):
        target = load_image(row["target_image_path"])
        generated = load_image(row["generated_image_path"])

        axes[i][0].imshow(target)
        axes[i][0].set_title(f"Target: {row['speaker_id']}_{row['utterance_id']}")
        axes[i][0].axis("off")

        axes[i][1].imshow(generated)
        axes[i][1].set_title("Generated")
        axes[i][1].axis("off")

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison grid to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    main(parser.parse_args())