from __future__ import annotations

import argparse
from pathlib import Path
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and prepare BraTS data")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config["data"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    print(f"BraTS data root: {root}")
    print(f"Expected modalities: {', '.join(config['data']['modalities'])}")
    print("Dataset preparation hook ready; add the local BraTS volume conversion here.")


if __name__ == "__main__":
    main()
