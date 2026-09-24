"""Create train/validation/test folders for YOLO classification."""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocess.dataset_utils import MIN_IMAGE_SIZE, RANDOM_SEED, SPLIT_RATIOS, split_labelled


PROJECT_ROOT = Path(__file__).resolve().parent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a classification dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT /  "datasets" / "leloi",
        help="Folder containing normal/ and sleep/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets"/"split_leloi",
        help="Destination for train/val/test",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--min-size",
        type=int,
        default=MIN_IMAGE_SIZE,
        help="Skip images with width or height smaller than this value.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Do not downsample classes before splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = split_labelled(
        args.input,
        args.output,
        seed=args.seed,
        ratios=SPLIT_RATIOS,
        min_size=args.min_size,
        balance=not args.no_balance,
    )
    for split_name, counts in summary.items():
        if split_name == "_skipped":
            continue
        print(
            f"{split_name:<5}: normal={counts['normal']:,}, "
            f"sleep={counts['sleep']:,}"
        )
    skipped = summary.get("_skipped")
    if skipped:
        print(
            f"skip : normal={skipped['normal']:,}, "
            f"sleep={skipped['sleep']:,} (min_size={args.min_size})"
        )
    balanced_out = summary.get("_balanced_out")
    if balanced_out:
        print(
            f"balance drop: normal={balanced_out['normal']:,}, "
            f"sleep={balanced_out['sleep']:,}"
        )
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
