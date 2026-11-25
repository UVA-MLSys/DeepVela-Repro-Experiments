#!/usr/bin/env python3
"""Validate a generic counter-plan JSON against a profiler's available-counter listing."""
import argparse
import json
import re
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path, help="JSON with passes containing counter names")
    parser.add_argument("available", type=Path, help="text emitted by the profiler")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    available_text = args.available.read_text()
    counters = [c for p in plan.get("passes", []) for c in p.get("counters", [])]
    missing = sorted({counter for counter in set(counters)
                      if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(counter)}(?![A-Za-z0-9_])",
                                       available_text)})
    if missing:
        print("missing counters:")
        print("\n".join(f"- {name}" for name in missing))
        return 1
    print(f"OK: {len(set(counters))} requested counters are listed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
