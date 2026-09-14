from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "split_leloi" / "split_leloi"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "split_leloi" / "merged_leloi"
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_LABELS = ("normal", "sleep")


def iter_images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def copy_unique(src: Path, dst_dir: Path, prefix: str = "") -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_name = f"{prefix}{src.name}" if prefix else src.name
    dst = dst_dir / dst_name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        try:
            shutil.copy2(src, dst)
        except PermissionError:
            locked_dst = dst_dir / f"{dst.stem}_locked_copy{dst.suffix.lower()}"
            shutil.copy2(src, locked_dst)
            return locked_dst
    return dst

    for index in range(1, 100000):
        candidate = dst_dir / f"{Path(dst_name).stem}_{index:05d}{src.suffix.lower()}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
    raise RuntimeError(f"Could not find unique output name for {src}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge train/val/test classification labels into one normal/ and sleep/ dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files when possible, without deleting the output folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input
    output_dir = args.output

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    counts: Counter[tuple[str, str]] = Counter()

    for split in args.splits:
        for label in args.labels:
            source_dir = input_dir / split / label
            if not source_dir.exists():
                print(f"skip missing: {source_dir}")
                continue

            target_dir = output_dir / label
            for src in iter_images(source_dir):
                dst = copy_unique(src, target_dir, prefix=f"{split}_")
                manifest_rows.append({
                    "split": split,
                    "label": label,
                    "source": str(src),
                    "saved_to": str(dst),
                    "case": dst.name,
                })
                counts[(split, label)] += 1

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "label", "source", "saved_to", "case"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Output: {output_dir.resolve()}")
    for split in args.splits:
        line = [f"{split:<5}"]
        for label in args.labels:
            line.append(f"{label}={counts[(split, label)]:,}")
        print("  ".join(line))

    totals = Counter()
    for (_split, label), count in counts.items():
        totals[label] += count
    print("total " + "  ".join(f"{label}={totals[label]:,}" for label in args.labels))
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
