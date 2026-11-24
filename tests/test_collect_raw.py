"""원본 저장·매니페스트(collect/raw.py) 테스트."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mlbproj.collect import raw

FIXED = datetime(2026, 11, 24, 1, 2, 3, tzinfo=UTC)
REL = Path("statsapi/2026_bat.json")
URL = "https://example.test/a"


class FakeGetter:
    def __init__(self, body: bytes = b'{"ok": 1}') -> None:
        self.body = body
        self.calls: list[str] = []

    def get(self, url: str) -> bytes:
        self.calls.append(url)
        return self.body


def test_collect_raw_save_writes_file_and_manifest(tmp_path: Path) -> None:
    entry = raw.save_raw(tmp_path, REL, URL, b"abc", now=lambda: FIXED)
    assert (tmp_path / REL).read_bytes() == b"abc"
    assert entry == raw.ManifestEntry(
        url=URL,
        fetched_at="2026-11-24T01:02:03Z",
        sha256=hashlib.sha256(b"abc").hexdigest(),
        bytes=3,
    )
    doc = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert doc == {
        "statsapi/2026_bat.json": {
            "url": URL,
            "fetched_at": "2026-11-24T01:02:03Z",
            "sha256": hashlib.sha256(b"abc").hexdigest(),
            "bytes": 3,
        }
    }
    assert not list(tmp_path.rglob("*.tmp"))


def test_collect_raw_manifest_is_sorted_and_stable(tmp_path: Path) -> None:
    raw.save_raw(tmp_path, Path("savant/2026_frv.csv"), URL, b"x", now=lambda: FIXED)
    raw.save_raw(tmp_path, Path("savant/2019_batter.csv"), URL, b"y", now=lambda: FIXED)
    text = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert list(json.loads(text)) == ["savant/2019_batter.csv", "savant/2026_frv.csv"]
    raw.save_raw(tmp_path, Path("savant/2019_batter.csv"), URL, b"y", now=lambda: FIXED)
    assert (tmp_path / "manifest.json").read_text(encoding="utf-8") == text


def test_collect_raw_load_manifest_absent_is_empty(tmp_path: Path) -> None:
    assert raw.load_manifest(tmp_path) == {}


def test_collect_raw_is_cached_rules(tmp_path: Path) -> None:
    assert not raw.is_cached(tmp_path, REL, URL)
    raw.save_raw(tmp_path, REL, URL, b"abc", now=lambda: FIXED)
    assert raw.is_cached(tmp_path, REL, URL)
    assert not raw.is_cached(tmp_path, REL, URL + "?other")  # 요청이 달라지면 다시 받는다
    (tmp_path / REL).write_bytes(b"tampered")
    assert not raw.is_cached(tmp_path, REL, URL)  # 내용이 매니페스트와 다르다


def test_collect_raw_fetch_uses_cache(tmp_path: Path) -> None:
    getter = FakeGetter()
    first = raw.fetch_to_raw(getter, tmp_path, REL, URL, now=lambda: FIXED)
    second = raw.fetch_to_raw(getter, tmp_path, REL, URL, now=lambda: FIXED)
    assert first.fetched and not second.fetched
    assert getter.calls == [URL]
    assert second.path == REL and second.entry == first.entry
    forced = raw.fetch_to_raw(getter, tmp_path, REL, URL, force=True, now=lambda: FIXED)
    assert forced.fetched and getter.calls == [URL, URL]


def test_collect_raw_fetch_check_rejects_before_saving(tmp_path: Path) -> None:
    def check(body: bytes) -> None:
        raise ValueError("형식이 다르다")

    with pytest.raises(ValueError, match="형식"):
        raw.fetch_to_raw(FakeGetter(b"<html>"), tmp_path, REL, URL, check=check)
    assert not (tmp_path / REL).exists()
    assert raw.load_manifest(tmp_path) == {}


def test_collect_raw_remove(tmp_path: Path) -> None:
    raw.save_raw(tmp_path, REL, URL, b"abc", now=lambda: FIXED)
    raw.remove_raw(tmp_path, REL)
    assert not (tmp_path / REL).exists()
    assert raw.load_manifest(tmp_path) == {}
    raw.remove_raw(tmp_path, REL)  # 없어도 오류가 아니다


def test_collect_raw_readers_handle_bom(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "raw"
    df = raw.read_csv(fixtures, Path("savant/2026_batter.csv"))
    assert df.columns[0] == "last_name, first_name"
    doc = raw.read_json(fixtures, Path("statsapi/2026_bat.json"))
    assert "stats" in doc


def test_collect_raw_file_sha(tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"abc")
    assert raw.file_sha256(tmp_path / "f") == hashlib.sha256(b"abc").hexdigest()


def test_collect_raw_read_csv_only_blank_is_missing(tmp_path: Path) -> None:
    """pandas 기본 결측 글자("NA"·"null"·"None"·"n/a")를 결측으로 바꾸지 않는다."""
    (tmp_path / "x.csv").write_text(
        "player_id,name,v\n1,NA,\n2,null,1.5\n3,None,n/a\n", encoding="utf-8"
    )
    df = raw.read_csv(tmp_path, Path("x.csv"))
    assert df["name"].tolist() == ["NA", "null", "None"]
    assert df["v"].isna().tolist() == [True, False, False]
    assert df["v"].tolist()[2] == "n/a"
