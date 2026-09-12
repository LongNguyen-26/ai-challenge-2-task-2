#!/usr/bin/env python3
"""Valid format baseline: label every released test point as normal."""
from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    with Path("test.csv").open("r", encoding="utf-8", newline="") as source, Path("predictions.csv").open("w", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "row_id" not in reader.fieldnames:
            raise ValueError("test.csv must contain row_id")
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(("row_id", "anomaly"))
        for row in reader:
            writer.writerow((row["row_id"], 0))


if __name__ == "__main__":
    main()
