#!/usr/bin/env python3
"""Summarize profiler CSV files without assuming a vendor, schema, or metric name."""
import argparse
import csv
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    files = sorted(args.directory.rglob("*.csv"))
    result = {"directory": str(args.directory), "files": []}
    for path in files:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        columns = {}
        for key in rows[0] if rows else []:
            values = []
            for row in rows:
                try:
                    value = float(row[key])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            if values:
                columns[key] = {"count": len(values), "mean": sum(values) / len(values),
                                "min": min(values), "max": max(values)}
        result["files"].append({"path": str(path), "rows": len(rows), "numeric": columns})
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
