#!/usr/bin/env python3
"""Collect Creative Commons YouTube gameplay data for world model training.

This script relies on the official YouTube Data API v3 to discover videos that
match a query, persists rich metadata, and (optionally) downloads the source
video and extracts frame samples. Only videos surfaced through the API with a
Creative Commons license are processed by default so you stay within YouTube's
Terms of Service and the original creators' licensing choices.

Usage example (metadata only):
    python scripts/youtube_collector.py \
        --api-key $YOUTUBE_API_KEY \
        --query "pixel art platformer gameplay" \
        --max-videos 100 \
        --output-dir datasets/youtube

Usage with downloads + frame extraction (requires yt-dlp + ffmpeg codecs):
    python scripts/youtube_collector.py \
        --api-key $YOUTUBE_API_KEY \
        --query "roguelike gameplay" \
        --max-videos 25 \
        --output-dir datasets/youtube \
        --download-videos \
        --extract-frames \
        --frame-interval 1.0 \
        --frame-size 256 256

Note: Respect creator rights. Verify every video you download grants a license
compatible with your downstream use, even when marked Creative Commons.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# OpenCV is optional until frame extraction is requested.
try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover
    raise ImportError("tqdm is required; install tqdm>=4.65.0") from exc

# Lazily imported to keep startup snappy when only writing metadata.
YOUTUBE_DISCOVERY_SERVICE = "youtube"
YOUTUBE_DISCOVERY_VERSION = "v3"


@dataclass
class VideoMetadata:
    """Structured metadata persisted to JSONL."""

    video_id: str
    title: str
    description: str
    channel_title: str
    publish_time: str
    tags: List[str]
    duration: str
    definition: str
    category_id: str
    default_audio_language: Optional[str]
    default_language: Optional[str]
    view_count: Optional[int]
    like_count: Optional[int]
    comment_count: Optional[int]
    license: str
    thumbnails: Dict[str, Dict[str, str]]
    query_tag: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Creative Commons YouTube gameplay data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-key", type=str, default=os.environ.get("YOUTUBE_API_KEY"),
                        help="YouTube Data API key (or set YOUTUBE_API_KEY env var)")
    parser.add_argument("--query", type=str, required=True,
                        help="Search query used to discover videos")
    parser.add_argument("--max-videos", type=int, default=50,
                        help="Maximum number of videos to fetch")
    parser.add_argument("--topic-id", type=str, default=None,
                        help="Optional Freebase topicId to bias search results")
    parser.add_argument("--published-after", type=str, default=None,
                        help="ISO8601 timestamp to restrict older uploads")
    parser.add_argument("--published-before", type=str, default=None,
                        help="ISO8601 timestamp to restrict newer uploads")
    parser.add_argument("--video-duration", choices=["any", "short", "medium", "long"],
                        default="medium", help="Duration bucket as defined by YouTube")
    parser.add_argument("--license", choices=["any", "creativeCommon"], default="creativeCommon",
                        help="License filter; prefer Creative Commons for reuse")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/youtube"),
                        help="Root directory to store metadata, videos, and frames")
    parser.add_argument("--metadata-file", type=str, default="metadata.jsonl",
                        help="Filename (inside output-dir) for metadata JSONL")
    parser.add_argument("--download-videos", action="store_true",
                        help="Download the source MP4 using yt-dlp (must be installed)")
    parser.add_argument("--video-quality", type=str, default="bestvideo[height<=720]+bestaudio/best",
                        help="yt-dlp format selector when downloading videos")
    parser.add_argument("--extract-frames", action="store_true",
                        help="Extract frames with OpenCV after download")
    parser.add_argument("--frame-interval", type=float, default=1.0,
                        help="Seconds between sampled frames")
    parser.add_argument("--max-frames", type=int, default=900,
                        help="Hard cap on frames extracted per video")
    parser.add_argument("--frame-size", type=int, nargs=2, default=(256, 256),
                        metavar=("WIDTH", "HEIGHT"),
                        help="Resize frames to this resolution")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip videos whose assets already exist on disk")
    parser.add_argument("--dry-run", action="store_true",
                        help="List candidate videos without downloading or writing files")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional label stored alongside each metadata record for bookkeeping")

    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key is required (or set YOUTUBE_API_KEY)")
    return args


def build_service(api_key: str):
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-api-python-client is required; run 'pip install google-api-python-client'"
        ) from exc
    return build(YOUTUBE_DISCOVERY_SERVICE, YOUTUBE_DISCOVERY_VERSION, developerKey=api_key)


def batched(iterable: Iterable, batch_size: int) -> Iterable[List]:
    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def search_video_ids(service, args: argparse.Namespace) -> List[str]:
    search_request = service.search().list(
        q=args.query,
        part="id",
        type="video",
        maxResults=min(50, args.max_videos),
        videoDefinition="high",
        videoDuration=args.video_duration,
        order="relevance",
        safeSearch="none",
        publishedAfter=args.published_after,
        publishedBefore=args.published_before,
        topicId=args.topic_id,
        videoLicense=args.license,
    )

    video_ids: List[str] = []
    pbar = tqdm(total=args.max_videos, desc="Fetching video ids", unit="vid")

    while search_request and len(video_ids) < args.max_videos:
        response = search_request.execute()
        for item in response.get("items", []):
            if item["id"]["kind"] == "youtube#video":
                video_ids.append(item["id"]["videoId"])
                pbar.update(1)
                if len(video_ids) >= args.max_videos:
                    break
        search_request = service.search().list_next(search_request, response)
    pbar.close()

    return video_ids


def fetch_metadata(service, video_ids: List[str], tag: Optional[str] = None) -> List[VideoMetadata]:
    metadata: List[VideoMetadata] = []
    for batch in batched(video_ids, 50):
        response = service.videos().list(
            part="id,snippet,contentDetails,statistics,status",
            id=",".join(batch),
        ).execute()
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            details = item.get("contentDetails", {})
            stats = item.get("statistics", {})
            status = item.get("status", {})
            metadata.append(
                VideoMetadata(
                    video_id=item["id"],
                    title=snippet.get("title", ""),
                    description=snippet.get("description", ""),
                    channel_title=snippet.get("channelTitle", ""),
                    publish_time=snippet.get("publishedAt", ""),
                    tags=snippet.get("tags", []),
                    duration=details.get("duration", ""),
                    definition=details.get("definition", ""),
                    category_id=snippet.get("categoryId", ""),
                    default_audio_language=snippet.get("defaultAudioLanguage"),
                    default_language=snippet.get("defaultLanguage"),
                    view_count=int(stats["viewCount"]) if "viewCount" in stats else None,
                    like_count=int(stats["likeCount"]) if "likeCount" in stats else None,
                    comment_count=int(stats["commentCount"]) if "commentCount" in stats else None,
                    license=status.get("license", "unknown"),
                    thumbnails=snippet.get("thumbnails", {}),
                    query_tag=tag,
                )
            )
    return metadata


def save_metadata(metadata: List[VideoMetadata], metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("a", encoding="utf-8") as fh:
        for record in metadata:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def download_video(video_id: str, destination_dir: Path, format_selector: str) -> Optional[Path]:
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yt-dlp is required for downloads; install yt-dlp>=2023.7.6") from exc

    destination_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(destination_dir / f"{video_id}.%(ext)s")
    ydl_opts = {
        "format": format_selector,
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except yt_dlp.utils.DownloadError as exc:
            print(f"⚠️ Failed to download {video_id}: {exc}")
            return None

    # yt-dlp chooses extension based on format; find resulting file.
    for ext in (".mp4", ".mkv", ".webm"):
        candidate = destination_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    interval_seconds: float,
    max_frames: int,
    frame_size: Optional[tuple[int, int]] = None,
) -> int:
    if cv2 is None:  # pragma: no cover
        raise ImportError("OpenCV is required for frame extraction; install opencv-python")
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"⚠️ Unable to open video for frames: {video_path}")
        return 0

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(fps * interval_seconds), 1)

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    saved = 0
    frame_index = 0

    with tqdm(total=min(total // frame_interval + 1, max_frames),
              desc=f"Frames {video_path.stem}", unit="frm") as bar:
        while saved < max_frames:
            ret, frame = capture.read()
            if not ret:
                break
            if frame_index % frame_interval == 0:
                if frame_size:
                    frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
                output_path = frames_dir / f"{video_path.stem}_{saved:05d}.png"
                cv2.imwrite(str(output_path), frame)
                saved += 1
                bar.update(1)
            frame_index += 1

    capture.release()
    return saved


def main() -> None:
    args = parse_args()
    service = build_service(args.api_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / args.metadata_file

    video_ids = search_video_ids(service, args)
    if not video_ids:
        print("No videos found for query – adjust filters and try again.")
        return

    if args.dry_run:
        print("Dry run complete. Candidate video ids:")
        for vid in video_ids:
            print(f"  https://www.youtube.com/watch?v={vid}")
        return

    metadata_records = fetch_metadata(service, video_ids, tag=args.tag)
    save_metadata(metadata_records, metadata_path)
    print(f"Metadata appended to {metadata_path}")

    if not args.download_videos:
        return

    videos_dir = args.output_dir / "videos"
    frames_root = args.output_dir / "frames"

    for record in metadata_records:
        video_id = record.video_id
        frames_dir = frames_root / video_id

        existing_video = None
        for ext in (".mp4", ".mkv", ".webm"):
            candidate = videos_dir / f"{video_id}{ext}"
            if candidate.exists():
                existing_video = candidate
                break

        if args.skip_existing and existing_video and (not args.extract_frames or frames_dir.exists()):
            print(f"Skipping existing assets for {video_id}")
            continue

        downloaded_path = existing_video
        if downloaded_path is None:
            downloaded_path = download_video(video_id, videos_dir, args.video_quality)
            if not downloaded_path:
                continue

        if args.extract_frames:
            target_size = tuple(args.frame_size) if args.frame_size else None
            extracted = extract_frames(
                downloaded_path,
                frames_dir,
                interval_seconds=args.frame_interval,
                max_frames=args.max_frames,
                frame_size=target_size,
            )
            print(f"Extracted {extracted} frames from {video_id}")
        time.sleep(1)  # Friendly pacing to avoid hammering endpoints


if __name__ == "__main__":
    main()
