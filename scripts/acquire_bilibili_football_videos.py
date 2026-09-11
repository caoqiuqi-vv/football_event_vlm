#!/usr/bin/env python
"""Discover and download public Bilibili football videos with yt-dlp.

Search-query labels are weak metadata only. They must not be treated as event
ground truth without clip generation and manual review.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
BV_RE = re.compile(r"\b(BV[0-9A-Za-z]+)\b", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exception:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exception
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    if not config.get("queries"):
        raise ValueError(f"Config has no queries: {path}")
    return config


def resolve_output_root(config: dict[str, Any], override: str) -> Path:
    value = override or str(config.get("output_root", "outputs/bilibili_football"))
    return Path(value).expanduser().resolve()


def resolve_yt_dlp(binary: str, required: bool = True) -> list[str]:
    executable = shutil.which(binary)
    if executable:
        return [executable]
    if binary == "yt-dlp" and importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    if required:
        raise RuntimeError(
            "yt-dlp is not installed. Install it with `python -m pip install -U yt-dlp` "
            "or pass --yt-dlp-bin /path/to/yt-dlp."
        )
    return [binary]


def search_target(query: str, count: int) -> str:
    return f"bilisearch{count}:{query}"


def build_discovery_command(
    yt_dlp: list[str], query: dict[str, Any], discovery: dict[str, Any]
) -> list[str]:
    count = int(query.get("max_results", discovery.get("max_results_per_query", 40)))
    command = [
        *yt_dlp,
        "--ignore-errors",
        "--no-warnings",
        "--skip-download",
        "--dump-json",
        "--playlist-end",
        str(count),
        "--sleep-requests",
        str(float(discovery.get("sleep_requests_sec", 1.5))),
        "--retries",
        str(int(discovery.get("retries", 3))),
    ]
    if str(discovery.get("metadata_mode", "full")) == "flat":
        command.append("--flat-playlist")
    command.append(search_target(str(query["query"]), count))
    return command


def source_id_from_entry(entry: dict[str, Any]) -> str:
    for value in (entry.get("bvid"), entry.get("id"), entry.get("url"), entry.get("webpage_url")):
        match = BV_RE.search(str(value or ""))
        if match:
            return match.group(1)
    extractor = str(entry.get("extractor_key") or entry.get("extractor") or "bilibili")
    raw_id = str(entry.get("id") or entry.get("url") or "").strip()
    return f"{extractor}:{raw_id}" if raw_id else ""


def canonical_url(source_id: str, entry: dict[str, Any]) -> str:
    if source_id.casefold().startswith("bv"):
        return f"https://www.bilibili.com/video/{source_id}"
    url = str(entry.get("webpage_url") or entry.get("url") or "")
    if url.startswith("http://bilibili.com/"):
        return url.replace("http://bilibili.com/", "https://www.bilibili.com/", 1)
    if url.startswith("https://bilibili.com/"):
        return url.replace("https://bilibili.com/", "https://www.bilibili.com/", 1)
    return url


def normalize_entry(entry: dict[str, Any], query: dict[str, Any]) -> dict[str, Any] | None:
    source_id = source_id_from_entry(entry)
    if not source_id:
        return None
    duration = entry.get("duration")
    try:
        duration = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None
    return {
        "source": "bilibili",
        "source_id": source_id,
        "webpage_url": canonical_url(source_id, entry),
        "title": str(entry.get("title") or "").strip(),
        "description": str(entry.get("description") or "").strip(),
        "uploader": str(entry.get("uploader") or entry.get("channel") or "").strip(),
        "uploader_id": str(entry.get("uploader_id") or entry.get("channel_id") or "").strip(),
        "duration_sec": duration,
        "upload_date": entry.get("upload_date"),
        "timestamp": entry.get("timestamp"),
        "view_count": int(entry.get("view_count") or 0),
        "like_count": int(entry.get("like_count") or 0),
        "is_live": bool(entry.get("is_live")),
        "age_limit": int(entry.get("age_limit") or 0),
        "availability": str(entry.get("availability") or "").strip(),
        "license": str(entry.get("license") or "").strip(),
        "weak_event_labels": [str(query["label"])],
        "discovery_queries": [str(query["query"])],
        "query_groups": [str(query.get("group") or query["label"])],
        "label_status": "weak_search_query",
        "rights_status": "unverified",
        "requires_manual_review": True,
        "discovered_at": utc_now(),
    }


def pattern_matches(patterns: Iterable[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def filter_reason(
    record: dict[str, Any], filters: dict[str, Any], query: dict[str, Any] | None = None
) -> str | None:
    title = str(record.get("title") or "")
    searchable = f"{title}\n{record.get('description') or ''}"
    if record.get("is_live"):
        return "live_video"
    availability = str(record.get("availability") or "").casefold()
    if availability not in {"", "public"}:
        return "non_public_availability"
    if int(record.get("age_limit") or 0) > int(filters.get("max_age_limit", 0)):
        return "age_restricted"
    duration = record.get("duration_sec")
    if duration is None:
        if bool(filters.get("reject_unknown_duration", False)):
            return "unknown_duration"
    else:
        if duration < float(filters.get("min_duration_sec", 15)):
            return "duration_too_short"
        if duration > float(filters.get("max_duration_sec", 10800)):
            return "duration_too_long"
    if int(record.get("view_count") or 0) < int(filters.get("min_view_count", 0)):
        return "view_count_too_low"
    exclude_patterns = list(filters.get("exclude_title_patterns", []))
    if query:
        exclude_patterns += list(query.get("exclude_title_patterns", []))
    if pattern_matches(exclude_patterns, title):
        return "excluded_title"
    required_patterns = list(filters.get("require_any_text_patterns", []))
    if required_patterns and not pattern_matches(required_patterns, searchable):
        return "missing_football_signal"
    return None


def merge_record(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key in ("weak_event_labels", "discovery_queries", "query_groups"):
        merged[key] = sorted(set(existing.get(key, [])) | set(incoming.get(key, [])))
    for key, value in incoming.items():
        if key in {"weak_event_labels", "discovery_queries", "query_groups"}:
            continue
        if value not in (None, "", 0, [], {}) and merged.get(key) in (None, "", 0, [], {}):
            merged[key] = value
    merged["view_count"] = max(int(existing.get("view_count") or 0), int(incoming.get("view_count") or 0))
    merged["like_count"] = max(int(existing.get("like_count") or 0), int(incoming.get("like_count") or 0))
    return merged


def apply_uploader_cap(
    records: list[dict[str, Any]], max_per_uploader: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_per_uploader <= 0:
        return records, []
    counts: Counter[str] = Counter()
    accepted = []
    rejected = []
    for record in sorted(records, key=lambda item: (-int(item.get("view_count") or 0), item["source_id"])):
        uploader_key = str(record.get("uploader_id") or record.get("uploader") or "").strip()
        if not uploader_key:
            uploader_key = f"unknown:{record['source_id']}"
        if counts[uploader_key] >= max_per_uploader:
            rejected.append({**record, "rejection_reason": "uploader_cap"})
            continue
        counts[uploader_key] += 1
        accepted.append(record)
    return accepted, rejected


def stream_json_command(command: list[str], error_log: Path) -> tuple[list[dict[str, Any]], int]:
    error_log.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with error_log.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                stderr_handle.write(f"unparsed_stdout={line[:1000]}\n")
        return_code = process.wait()
    return records, return_code


def discover(args: argparse.Namespace, config: dict[str, Any], root: Path) -> None:
    yt_dlp = resolve_yt_dlp(args.yt_dlp_bin)
    discovery = dict(config.get("discovery", {}))
    filters = dict(config.get("filters", {}))
    metadata_dir = root / "metadata"
    logs_dir = root / "logs"
    all_discovered: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    query_stats = []
    for index, query in enumerate(config["queries"], start=1):
        command = build_discovery_command(yt_dlp, query, discovery)
        print(
            f"[{index}/{len(config['queries'])}] discover label={query['label']} "
            f"query={query['query']!r}",
            flush=True,
        )
        entries, return_code = stream_json_command(
            command, logs_dir / f"discover_{index:03d}_{query['label']}.stderr.log"
        )
        accepted_for_query = 0
        for entry in entries:
            record = normalize_entry(entry, query)
            if record is None:
                continue
            all_discovered.append(record)
            reason = filter_reason(record, filters, query)
            if reason:
                rejected.append({**record, "rejection_reason": reason})
            else:
                accepted_for_query += 1
        query_stats.append(
            {
                "label": query["label"],
                "query": query["query"],
                "found": len(entries),
                "accepted_before_dedup": accepted_for_query,
                "return_code": return_code,
            }
        )

    merged: dict[str, dict[str, Any]] = {}
    for record in all_discovered:
        reason = filter_reason(record, filters)
        if reason:
            continue
        key = record["source_id"]
        merged[key] = merge_record(merged[key], record) if key in merged else record

    accepted, capped = apply_uploader_cap(
        list(merged.values()), int(filters.get("max_videos_per_uploader", 12))
    )
    rejected.extend(capped)
    max_total = int(filters.get("max_total_videos", 0))
    if max_total > 0 and len(accepted) > max_total:
        overflow = accepted[max_total:]
        rejected.extend({**record, "rejection_reason": "total_cap"} for record in overflow)
        accepted = accepted[:max_total]
    accepted.sort(key=lambda item: item["source_id"])

    write_jsonl(metadata_dir / "discovered.jsonl", all_discovered)
    write_jsonl(metadata_dir / "accepted.jsonl", accepted)
    write_jsonl(metadata_dir / "rejected.jsonl", rejected)
    summary = {
        "created_at": utc_now(),
        "output_root": str(root),
        "query_stats": query_stats,
        "num_discovered_rows": len(all_discovered),
        "num_unique_accepted": len(accepted),
        "num_rejected_rows": len(rejected),
        "accepted_by_weak_label": dict(
            sorted(Counter(label for item in accepted for label in item["weak_event_labels"]).items())
        ),
        "rejected_by_reason": dict(sorted(Counter(item["rejection_reason"] for item in rejected).items())),
        "label_warning": "weak_event_labels come from search queries and are not event ground truth",
    }
    write_json(metadata_dir / "discovery_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def select_download_records(
    records: list[dict[str, Any]], labels: set[str], limit: int
) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        if labels and not labels.intersection(record.get("weak_event_labels", [])):
            continue
        if not str(record.get("webpage_url") or "").startswith("https://www.bilibili.com/video/"):
            continue
        selected.append(record)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def build_download_command(
    yt_dlp: list[str], root: Path, config: dict[str, Any], batch_file: Path
) -> list[str]:
    settings = dict(config.get("download", {}))
    metadata_dir = root / "metadata"
    command = [
        *yt_dlp,
        "--batch-file",
        str(batch_file),
        "--no-playlist",
        "--continue",
        "--ignore-errors",
        "--download-archive",
        str(metadata_dir / "download_archive.txt"),
        "--write-info-json",
        "--no-write-playlist-metafiles",
        "--output",
        str(root / "videos" / "%(id)s" / "%(id)s.%(ext)s"),
        "--format",
        str(settings.get("format", "bv*[height<=720]+ba/b[height<=720]/b")),
        "--merge-output-format",
        str(settings.get("merge_output_format", "mp4")),
        "--sleep-interval",
        str(float(settings.get("sleep_interval_sec", 5))),
        "--max-sleep-interval",
        str(float(settings.get("max_sleep_interval_sec", 12))),
        "--sleep-requests",
        str(float(settings.get("sleep_requests_sec", 1.5))),
        "--concurrent-fragments",
        str(int(settings.get("concurrent_fragments", 1))),
        "--retries",
        str(int(settings.get("retries", 5))),
        "--fragment-retries",
        str(int(settings.get("fragment_retries", 5))),
    ]
    if bool(settings.get("write_thumbnail", False)):
        command.append("--write-thumbnail")
    return command


def run_logged_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{utc_now()}] command={json.dumps(command, ensure_ascii=False)}\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
        return process.wait()


def build_downloaded_manifest(root: Path, accepted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accepted_by_id = {record["source_id"].casefold(): record for record in accepted}
    downloaded = []
    videos_root = root / "videos"
    if not videos_root.exists():
        return downloaded
    for directory in sorted(path for path in videos_root.iterdir() if path.is_dir()):
        info_paths = sorted(directory.glob("*.info.json"))
        info = {}
        if info_paths:
            try:
                info = json.loads(info_paths[0].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                info = {}
        source_id = source_id_from_entry(info) or directory.name
        video_paths = sorted(
            path.resolve() for path in directory.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES
        )
        if not video_paths:
            continue
        base = accepted_by_id.get(source_id.casefold(), {})
        downloaded.append(
            {
                **base,
                "source_id": source_id,
                "path": str(video_paths[0]),
                "all_video_paths": [str(path) for path in video_paths],
                "file_size_bytes": video_paths[0].stat().st_size,
                "downloaded": True,
                "download_manifest_updated_at": utc_now(),
            }
        )
    write_jsonl(root / "metadata" / "downloaded.jsonl", downloaded)
    return downloaded


def download(args: argparse.Namespace, config: dict[str, Any], root: Path) -> None:
    if not args.rights_acknowledged:
        raise SystemExit(
            "Download blocked: pass --rights-acknowledged only after confirming that the selected "
            "public videos may be downloaded and used for your intended research/training purpose."
        )
    accepted_path = root / "metadata" / "accepted.jsonl"
    accepted = read_jsonl(accepted_path)
    if not accepted:
        raise SystemExit(f"No accepted records found: run discover first ({accepted_path})")
    labels = {item.strip() for item in args.labels.split(",") if item.strip()}
    selected = select_download_records(accepted, labels, args.limit)
    if not selected:
        raise SystemExit(f"No records selected. labels={sorted(labels)} limit={args.limit}")
    metadata_dir = root / "metadata"
    batch_file = metadata_dir / "selected_urls.txt"
    batch_file.parent.mkdir(parents=True, exist_ok=True)
    batch_file.write_text(
        "".join(f"{record['webpage_url']}\n" for record in selected), encoding="utf-8"
    )
    write_jsonl(metadata_dir / "selected_for_download.jsonl", selected)
    command = build_download_command(resolve_yt_dlp(args.yt_dlp_bin), root, config, batch_file)
    print(f"download selected={len(selected)} root={root}", flush=True)
    return_code = run_logged_command(command, root / "logs" / "download.log")
    downloaded = build_downloaded_manifest(root, accepted)
    summary = {
        "updated_at": utc_now(),
        "selected": len(selected),
        "downloaded_total": len(downloaded),
        "return_code": return_code,
        "archive": str(metadata_dir / "download_archive.txt"),
        "manifest": str(metadata_dir / "downloaded.jsonl"),
    }
    write_json(metadata_dir / "download_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if return_code != 0:
        raise SystemExit(return_code)


def print_plan(args: argparse.Namespace, config: dict[str, Any], root: Path) -> None:
    yt_dlp = resolve_yt_dlp(args.yt_dlp_bin, required=False)
    discovery = dict(config.get("discovery", {}))
    plan = {
        "output_root": str(root),
        "num_queries": len(config["queries"]),
        "estimated_search_results_before_dedup": sum(
            int(query.get("max_results", discovery.get("max_results_per_query", 40)))
            for query in config["queries"]
        ),
        "queries": [
            {
                "label": query["label"],
                "group": query.get("group", query["label"]),
                "query": query["query"],
                "command": build_discovery_command(yt_dlp, query, discovery),
            }
            for query in config["queries"]
        ],
        "outputs": {
            "accepted": str(root / "metadata" / "accepted.jsonl"),
            "rejected": str(root / "metadata" / "rejected.jsonl"),
            "downloaded": str(root / "metadata" / "downloaded.jsonl"),
            "videos": str(root / "videos"),
        },
        "warning": "query labels are weak metadata; review clips before supervised training",
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a rate-limited, reviewable public Bilibili football video reserve."
    )
    parser.add_argument(
        "--config",
        default="configs/football/bilibili_football_acquisition.yaml",
    )
    parser.add_argument("--output-root", default="", help="Override config output_root.")
    parser.add_argument("--yt-dlp-bin", default="yt-dlp")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="Validate config and print commands without network access.")
    subparsers.add_parser("discover", help="Search public metadata, filter and deduplicate.")
    download_parser = subparsers.add_parser("download", help="Download accepted public videos.")
    download_parser.add_argument("--labels", default="", help="Comma-separated weak labels.")
    download_parser.add_argument("--limit", type=int, default=0, help="0 downloads all selected rows.")
    download_parser.add_argument("--rights-acknowledged", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    root = resolve_output_root(config, args.output_root)
    if args.command == "plan":
        print_plan(args, config, root)
    elif args.command == "discover":
        discover(args, config, root)
    elif args.command == "download":
        download(args, config, root)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
