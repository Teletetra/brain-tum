from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print("Evaluation hook ready for Dice / IoU / HD95 reporting.")


if __name__ == "__main__":
    main()
