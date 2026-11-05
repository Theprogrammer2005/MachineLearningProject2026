"""CSE382 project entry point.

Usage:
    python main.py analyze
"""
import argparse
import random

import numpy as np

from src.data.eda import run_analysis
from src.data.load import load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="City Hotel (H2) ML project")
    p.add_argument("--config", default="config.yaml", help="path to config file")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("analyze", help="run data understanding / EDA")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    if args.command == "analyze":
        run_analysis(cfg)


if __name__ == "__main__":
    main()
