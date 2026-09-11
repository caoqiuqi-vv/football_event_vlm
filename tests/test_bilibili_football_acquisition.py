from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "acquire_bilibili_football_videos.py"
SPEC = importlib.util.spec_from_file_location("bilibili_acquisition", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_record(source_id: str, uploader: str, views: int) -> dict:
    return {
        "source_id": source_id,
        "uploader": uploader,
        "uploader_id": uploader,
        "view_count": views,
        "weak_event_labels": ["shot"],
    }


def test_bvid_case_and_canonical_url_are_preserved() -> None:
    entry = {"id": "BV1ab411c7De"}
    source_id = MODULE.source_id_from_entry(entry)
    assert source_id == "BV1ab411c7De"
    assert MODULE.canonical_url(source_id, entry) == (
        "https://www.bilibili.com/video/BV1ab411c7De"
    )


def test_filter_rejects_games_but_keeps_real_matches() -> None:
    filters = {
        "min_duration_sec": 20,
        "max_duration_sec": 10800,
        "max_age_limit": 0,
        "exclude_title_patterns": ["实况足球|FIFA[0-9 ]"],
        "require_any_text_patterns": ["足球|football"],
    }
    game = {
        "title": "实况足球 射门集锦",
        "description": "",
        "duration_sec": 100,
        "view_count": 10,
    }
    match = {
        "title": "青少年足球比赛全场录像",
        "description": "",
        "duration_sec": 3600,
        "view_count": 10,
    }
    assert MODULE.filter_reason(game, filters) == "excluded_title"
    assert MODULE.filter_reason(match, filters) is None


def test_duplicate_queries_merge_weak_labels() -> None:
    first = {
        "source_id": "BV1ab411c7De",
        "weak_event_labels": ["shot"],
        "discovery_queries": ["足球 射门"],
        "query_groups": ["event"],
        "view_count": 10,
        "like_count": 1,
    }
    second = {
        **first,
        "weak_event_labels": ["save"],
        "discovery_queries": ["门将 扑救"],
        "view_count": 20,
    }
    merged = MODULE.merge_record(first, second)
    assert merged["weak_event_labels"] == ["save", "shot"]
    assert merged["view_count"] == 20


def test_uploader_cap_keeps_highest_view_count() -> None:
    records = [
        make_record("BV-low", "same", 10),
        make_record("BV-high", "same", 100),
        make_record("BV-other", "other", 1),
    ]
    accepted, rejected = MODULE.apply_uploader_cap(records, max_per_uploader=1)
    assert {item["source_id"] for item in accepted} == {"BV-high", "BV-other"}
    assert rejected[0]["source_id"] == "BV-low"
    assert rejected[0]["rejection_reason"] == "uploader_cap"


def test_download_command_is_resumable_and_rate_limited() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        command = MODULE.build_download_command(
            ["yt-dlp"],
            root,
            {"download": {"sleep_interval_sec": 5, "concurrent_fragments": 1}},
            root / "urls.txt",
        )
        assert "--download-archive" in command
        assert "--continue" in command
        assert command[command.index("--concurrent-fragments") + 1] == "1"


if __name__ == "__main__":
    test_bvid_case_and_canonical_url_are_preserved()
    test_filter_rejects_games_but_keeps_real_matches()
    test_duplicate_queries_merge_weak_labels()
    test_uploader_cap_keeps_highest_view_count()
    test_download_command_is_resumable_and_rate_limited()
    print("5 acquisition tests passed")
