#!/usr/bin/env bash
python src/infer.py --config configs/default.yaml --checkpoint data/checkpoints/best.pt --audio "$1" --output "$2"
