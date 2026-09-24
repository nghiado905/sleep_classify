from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


def load_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"File CSV rỗng: {csv_path}")

    required = {"gt", "label"}
    if not required.issubset(set(rows[0].keys())):
        raise ValueError(f"CSV phải có cột: {required}. Hiện có: {set(rows[0].keys())}")

    # Lọc bỏ dòng lỗi (header bị ghi lại, giá trị rỗng...)
    cleaned = []
    invalid_values = {"gt", "label", ""}
    for r in rows:
        gt = (r.get("gt") or "").strip().lower()
        label = (r.get("label") or "").strip().lower()
        if gt in invalid_values or label in invalid_values:
            continue
        cleaned.append(r)

    if not cleaned:
        raise ValueError("Không còn dòng hợp lệ sau khi lọc CSV.")

    return cleaned


def normalize_label(value: str) -> str:
    return value.strip().lower()


def make_run_name(csv_path: Path) -> str:
    parent_name = csv_path.parent.name.strip()
    if parent_name:
        return f"{parent_name}_{csv_path.stem}"
    return csv_path.stem


def evaluate(
    csv_path: Path,
    positive_class: str = "sleep",
    output_dir: Path | None = None,
) -> None:
    rows = load_csv(csv_path)
    positive_class = positive_class.strip().lower()

    y_true = [normalize_label(r["gt"]) for r in rows]
    y_pred = [normalize_label(r["label"]) for r in rows]
    classes = sorted(set(y_true) | set(y_pred))

    TP = FP = TN = FN = 0
    for gt, pred in zip(y_true, y_pred):
        if gt == positive_class and pred == positive_class:
            TP += 1
        elif gt != positive_class and pred == positive_class:
            FP += 1
        elif gt != positive_class and pred != positive_class:
            TN += 1
        elif gt == positive_class and pred != positive_class:
            FN += 1

    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    specificity = TN / (TN + FP) if (TN + FP) else 0.0

    gt_count = Counter(y_true)
    pred_count = Counter(y_pred)

    if output_dir is None:
        output_dir = csv_path.parent / "eval"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = make_run_name(csv_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{run_name}_{timestamp}"

    csv_out = output_dir / f"predictions_{base_name}.csv"
    shutil.copy2(csv_path, csv_out)

    lines = []
    lines.append("=" * 70)
    lines.append(" EVALUATION REPORT ".center(70, "="))
    lines.append("=" * 70)
    lines.append(f"  Time            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Source CSV      : {csv_path}")
    lines.append(f"  Run name        : {run_name}")
    lines.append(f"  Total samples   : {len(rows):,}")
    lines.append(f"  Positive class  : {positive_class}")
    lines.append(f"  Classes         : {classes}")
    lines.append("-" * 70)
    lines.append(" CONFUSION MATRIX (Binary) ".center(70, "-"))
    lines.append(f"  TP (True Positive)  : {TP:>6}")
    lines.append(f"  FP (False Positive) : {FP:>6}")
    lines.append(f"  TN (True Negative)  : {TN:>6}")
    lines.append(f"  FN (False Negative) : {FN:>6}")
    lines.append("-" * 70)
    lines.append(" METRICS ".center(70, "-"))
    lines.append(f"  Accuracy    : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    lines.append(f"  Precision   : {precision:.4f}  ({precision*100:.2f}%)")
    lines.append(f"  Recall      : {recall:.4f}  ({recall*100:.2f}%)")
    lines.append(f"  F1-Score    : {f1:.4f}  ({f1*100:.2f}%)")
    lines.append(f"  Specificity : {specificity:.4f}  ({specificity*100:.2f}%)")
    lines.append("-" * 70)
    lines.append(" LABEL DISTRIBUTION ".center(70, "-"))
    lines.append(f"  {'Class':<15} {'GT':>8} {'Pred':>8}")
    for cls in classes:
        lines.append(f"  {cls:<15} {gt_count[cls]:>8} {pred_count[cls]:>8}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Ma trận nhầm lẫn (positive = '{positive_class}'):")
    lines.append(f"""
                    Pred {positive_class:^10}   Pred other
  Actual {positive_class:<10}     TP={TP:<6}         FN={FN:<6}
  Actual other          FP={FP:<6}         TN={TN:<6}
""")

    summary_text = "\n".join(lines)
    print(summary_text)

    summary_path = output_dir / f"summary_{base_name}.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    print(f"\n✅ Đã lưu kết quả vào: {output_dir}")
    print(f"   - {summary_path.name}")
    print(f"   - {csv_out.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Danh gia mo hinh tu file predictions.csv")
    parser.add_argument("--csv", type=Path, required=True, help="Duong dan file predictions.csv")
    parser.add_argument("--positive", type=str, default="sleep", help="Ten class duong (mac dinh: sleep)")
    parser.add_argument("--output", type=Path, default=None, help="Thu muc luu ket qua")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.csv, positive_class=args.positive, output_dir=args.output)