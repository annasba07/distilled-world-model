#!/usr/bin/env python3
"""Batch orchestrator for Creative Commons YouTube gameplay collection.

This utility repeatedly invokes ``youtube_collector.py`` using a YAML
configuration file that defines multiple search queries, API keys, and
collection targets in hours. It keeps looping each query until the unique
runtime captured for that query meets the configured target.

Example usage::

    python scripts/batch_collect.py --config configs/cc_platformers.yaml

The config file ships with environment-variable based API key references so
that long-running jobs can rotate keys and stay within quota limits. Each
query writes into its own subdirectory under ``output_root`` to keep metadata,
video assets, and extracted frames isolated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import yaml
import re

METADATA_FILENAME_FALLBACK = "metadata.jsonl"
COLLECTOR_SCRIPT = Path(__file__).with_name("youtube_collector.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch YouTube gameplay collector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration describing API keys and queries",
    )
    parser.add_argument(
        "--collector",
        type=Path,
        default=COLLECTOR_SCRIPT,
        help="Path to youtube_collector.py (override for custom builds)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if "api_keys" not in data or not data["api_keys"]:
        raise ValueError("Configuration requires at least one API key under 'api_keys'")
    if "queries" not in data or not data["queries"]:
        raise ValueError("Configuration requires at least one entry under 'queries'")
    return data


def resolve_api_keys(entries: Sequence) -> List[str]:
    keys: List[str] = []
    for entry in entries:
        value: Optional[str] = None
        if isinstance(entry, str):
            value = _expand_env_reference(entry)
        elif isinstance(entry, dict):
            if "env" in entry:
                env_name = entry["env"]
                value = os.environ.get(env_name)
                if not value:
                    raise ValueError(f"Environment variable '{env_name}' is not set")
            elif "value" in entry:
                value = str(entry["value"])
        if not value:
            raise ValueError("Every api_keys entry must resolve to a non-empty string")
        keys.append(value)
    return keys


def _expand_env_reference(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("${") and raw.endswith("}"):
        env_name = raw[2:-1]
        value = os.environ.get(env_name)
        if not value:
            raise ValueError(f"Environment variable '{env_name}' is not set")
        return value
    return raw


@dataclass
class QueryProgress:
    video_ids: Dict[str, Dict]
    total_seconds: float


def read_metadata(metadata_path: Path) -> QueryProgress:
    if not metadata_path.exists():
        return QueryProgress(video_ids={}, total_seconds=0.0)
    seen: Dict[str, Dict] = {}
    total_seconds = 0.0
    with metadata_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            video_id = record.get("video_id")
            if not video_id or video_id in seen:
                continue
            seen[video_id] = record
            duration = record.get("duration")
            seconds = iso8601_duration_to_seconds(duration) if duration else 0.0
            total_seconds += seconds
    return QueryProgress(video_ids=seen, total_seconds=total_seconds)


_DURATION_RE = re.compile(
    r"^P(?:(?P<weeks>\d+)W|(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?)$"
)

def iso8601_duration_to_seconds(value: str) -> float:
    if not value:
        return 0.0
    match = _DURATION_RE.match(value)
    if not match:
        return 0.0
    groups = match.groupdict()
    if groups.get("weeks"):
        return float(groups["weeks"]) * 7 * 24 * 3600
    days = float(groups.get("days") or 0.0)
    hours = float(groups.get("hours") or 0.0)
    minutes = float(groups.get("minutes") or 0.0)
    seconds = float(groups.get("seconds") or 0.0)
    return (((days * 24) + hours) * 60 + minutes) * 60 + seconds


def build_command(
    collector: Path,
    api_key: str,
    query_cfg: Dict,
    global_cfg: Dict,
    output_dir: Path,
    metadata_file: str,
) -> List[str]:
    cmd = [
        sys.executable,
        str(collector),
        "--api-key",
        api_key,
        "--query",
        query_cfg["query"],
        "--max-videos",
        str(query_cfg.get("max_videos_per_call", global_cfg.get("max_videos_per_call", 50))),
        "--output-dir",
        str(output_dir),
        "--metadata-file",
        metadata_file,
        "--tag",
        query_cfg["name"],
    ]

    # Flags
    if query_cfg.get("download_videos", global_cfg.get("download_videos", True)):
        cmd.append("--download-videos")
    if query_cfg.get("extract_frames", global_cfg.get("extract_frames", True)):
        cmd.append("--extract-frames")
    if query_cfg.get("skip_existing", global_cfg.get("skip_existing", True)):
        cmd.append("--skip-existing")

    # Scalars / overrides
    field_map = {
        "video_duration": "--video-duration",
        "topic_id": "--topic-id",
        "published_after": "--published-after",
        "published_before": "--published-before",
        "license": "--license",
        "video_quality": "--video-quality",
        "frame_interval": "--frame-interval",
        "max_frames": "--max-frames",
    }
    for key, flag in field_map.items():
        value = query_cfg.get(key, global_cfg.get(key))
        if value is not None:
            cmd.extend([flag, str(value)])

    # Frame size is a 2-tuple/list
    frame_size = query_cfg.get("frame_size", global_cfg.get("frame_size"))
    if frame_size:
        if not isinstance(frame_size, (list, tuple)) or len(frame_size) != 2:
            raise ValueError(f"frame_size must be [width, height], got {frame_size}")
        cmd.extend(["--frame-size", str(frame_size[0]), str(frame_size[1])])

    return cmd


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def human_hours(seconds: float) -> float:
    return seconds / 3600.0


def run_query_collector(
    collector: Path,
    api_keys: Iterable[str],
    query_cfg: Dict,
    global_cfg: Dict,
    output_root: Path,
    dry_run: bool = False,
) -> None:
    metadata_file = query_cfg.get("metadata_file", global_cfg.get("metadata_file", METADATA_FILENAME_FALLBACK))
    output_dir = output_root / query_cfg["name"]
    ensure_directory(output_dir)

    key_cycle = cycle(api_keys)
    target_hours = float(query_cfg["target_hours"])
    sleep_seconds = float(query_cfg.get("sleep_seconds", global_cfg.get("sleep_seconds", 60)))
    max_stall_runs = int(query_cfg.get("max_stall_runs", global_cfg.get("max_stall_runs", 3)))

    stall_runs = 0
    iteration = 0

    while True:
        progress = read_metadata(output_dir / metadata_file)
        collected_hours = human_hours(progress.total_seconds)
        needed = target_hours - collected_hours
        if needed <= 0:
            print(f"✅ Query '{query_cfg['name']}' reached {collected_hours:.1f} hours (target {target_hours}h)")
            break

        iteration += 1
        api_key = next(key_cycle)
        cmd = build_command(collector, api_key, query_cfg, global_cfg, output_dir, metadata_file)
        print(
            f"[{query_cfg['name']}] iteration {iteration}: collected {collected_hours:.2f}h, "
            f"target {target_hours}h → running collector (max {cmd[cmd.index('--max-videos')+1]} videos)"
        )

        if dry_run:
            print("DRY-RUN:", " ".join(cmd))
            if iteration >= 1:
                break
            continue

        before_ids = set(progress.video_ids.keys())
        start = time.time()
        result = subprocess.run(cmd, check=False)
        duration = time.time() - start
        if result.returncode != 0:
            print(f"⚠️ Collector exited with code {result.returncode}; sleeping {sleep_seconds:.0f}s before retry")
            time.sleep(sleep_seconds)
            stall_runs += 1
            if stall_runs >= max_stall_runs:
                print(f"⚠️ Giving up on query '{query_cfg['name']}' after {stall_runs} stalled runs")
                break
            continue

        progress_after = read_metadata(output_dir / metadata_file)
        new_ids = set(progress_after.video_ids.keys()) - before_ids
        added_hours = human_hours(progress_after.total_seconds - progress.total_seconds)
        collected_hours = human_hours(progress_after.total_seconds)

        if new_ids:
            stall_runs = 0
            print(
                f"   ↳ Added {len(new_ids)} new videos (+{added_hours:.2f}h) in {duration:.0f}s."
                f" Total: {collected_hours:.2f}h / {target_hours}h"
            )
        else:
            stall_runs += 1
            print(
                f"   ↳ No new videos detected; run took {duration:.0f}s. "
                f"Stall count: {stall_runs}/{max_stall_runs}"
            )
            if stall_runs >= max_stall_runs:
                print(f"⚠️ Giving up on query '{query_cfg['name']}' after repeated stalls")
                break

        time.sleep(sleep_seconds)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    api_keys = resolve_api_keys(config["api_keys"])
    global_cfg = config.get("global", {})
    queries: List[Dict] = config["queries"]
    output_root = Path(config.get("output_root", "datasets/youtube"))
    ensure_directory(output_root)

    print(f"Batch collector starting with {len(api_keys)} API key(s) and {len(queries)} queries")
    try:
        for query in queries:
            if "name" not in query or "query" not in query or "target_hours" not in query:
                raise ValueError("Each query must define 'name', 'query', and 'target_hours'")
            run_query_collector(
                collector=args.collector,
                api_keys=api_keys,
                query_cfg=query,
                global_cfg=global_cfg,
                output_root=output_root,
                dry_run=args.dry_run,
            )
    except KeyboardInterrupt:
        print("Interrupted by user; exiting gracefully.")


if __name__ == "__main__":
    main()
