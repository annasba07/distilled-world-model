#!/usr/bin/env python3
"""Deduplicate metadata JSONL files produced by youtube_collector.

The collector appends metadata on every run, which can leave duplicates when a
video surfaces across multiple passes. This helper keeps the most recent record
per ``video_id`` and writes a clean JSONL file.

Usage examples::

    python scripts/dedupe_metadata.py datasets/cc_platformers/pixel_platformers/metadata.jsonl
    python scripts/dedupe_metadata.py input.jsonl --output cleaned.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate collector metadata JSONL")
    parser.add_argument("input", type=Path, help="Path to metadata JSONL")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional output path (defaults to in-place overwrite)")
    return parser.parse_args()


def load_unique_records(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    buffer: Dict[str, str] = {}
    order: Dict[str, int] = {}
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        video_id = record.get("video_id")
        if not video_id:
            continue
        buffer[video_id] = json.dumps(record, ensure_ascii=False)
        order[video_id] = idx
    # Stable sort based on original last-seen order
    sorted_ids = sorted(order.items(), key=lambda item: item[1])
    return [buffer[video_id] for video_id, _ in sorted_ids]


def main() -> None:
    args = parse_args()
    records = load_unique_records(args.input)
    output_path = args.output or args.input
    output_path.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")
    print(f"Wrote {len(records)} unique records to {output_path}")


if __name__ == "__main__":
    main()
