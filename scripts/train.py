from __future__ import annotations

import argparse
from pathlib import Path
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hybrid brain tumor segmentation model")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    print("Training configuration loaded.")
    print(f"Model: CNN + ViT | H-CSAF={config['model']['h_csaf']} | epochs={config['training']['epochs']}")
    print("Connect this entrypoint to the full dataset trainer after local BraTS preprocessing is configured.")


if __name__ == "__main__":
    main()
