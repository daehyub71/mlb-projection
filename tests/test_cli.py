"""명령줄(cli.py) 테스트 — 수집·검증 명령을 가짜 요청으로 끝까지 돌린다."""

import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from mlbproj import cli
from mlbproj import contracts as c
from mlbproj.collect.http import FetchError

FIXTURE_RAW = Path(__file__).parent / "fixtures" / "raw"


class FixtureGetter:
    """계약 URL마다 픽스처 내용을 돌려주는 가짜 요청 객체."""

    def __init__(self, seasons: list[int]) -> None:
        self.calls: list[str] = []
        self.urls: dict[str, bytes] = {}
        for season in seasons:
            for role in c.ROLES:
                self.urls[c.statsapi_stats_url(season, role)] = (
                    FIXTURE_RAW / c.raw_statsapi_path(season, role)
                ).read_bytes()
                kind: c.SavantKind = "batter" if role == "bat" else "pitcher"
                self.urls[c.savant_leaderboard_url(season, role)] = (
                    FIXTURE_RAW / c.raw_savant_path(season, kind)
                ).read_bytes()
            if season in c.FIELDING_YEARS:
                self.urls[c.savant_oaa_url(season)] = (
                    FIXTURE_RAW / c.raw_savant_path(season, "oaa")
                ).read_bytes()
                self.urls[c.savant_frv_url(season)] = (
                    FIXTURE_RAW / c.raw_savant_path(season, "frv")
                ).read_bytes()
        docs = [
            json.loads((FIXTURE_RAW / c.raw_people_path(i)).read_text(encoding="utf-8"))
            for i in (0, 1)
        ]
        self.people = {p["id"]: p for doc in docs for p in doc["people"]}
        self.copyright = docs[0]["copyright"]

    def get(self, url: str) -> bytes:
        self.calls.append(url)
        if url in self.urls:
            return self.urls[url]
        query = parse_qs(urlparse(url).query)
        if "personIds" in query:
            ids = [int(i) for i in query["personIds"][0].split(",")]
            body = {"copyright": self.copyright, "people": [self.people[i] for i in ids]}
            return json.dumps(body).encode("utf-8")
        raise AssertionError(f"예상하지 못한 URL: {url}")


def test_cli_parse_seasons() -> None:
    assert cli.parse_seasons("2019-2026") == list(c.SEASONS)
    assert cli.parse_seasons("2025,2024,2025") == [2024, 2025]
    for bad in ("2018-2020", "abc", "", "2027"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_seasons(bad)


def test_cli_collect_then_cache(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    getter = FixtureGetter([2025, 2026])
    code = cli.main(
        ["collect", "--seasons", "2025-2026", "--raw-dir", str(tmp_path)], getter=getter
    )
    assert code == 0
    for season in (2025, 2026):
        for role in c.ROLES:
            assert (tmp_path / c.raw_statsapi_path(season, role)).exists()
        kinds: tuple[c.SavantKind, ...] = ("batter", "pitcher", "oaa", "frv")
        for kind in kinds:
            assert (tmp_path / c.raw_savant_path(season, kind)).exists()
    assert (tmp_path / c.raw_people_path(0)).exists()
    manifest = json.loads((tmp_path / c.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert len(manifest) == len(getter.calls) == 13  # 시즌 기록 4 + Savant 8 + 선수 정보 1
    assert "새로 받음 13" in capsys.readouterr().out

    again = FixtureGetter([2025, 2026])
    assert (
        cli.main(["collect", "--seasons", "2025-2026", "--raw-dir", str(tmp_path)], getter=again)
        == 0
    )
    assert again.calls == []
    assert "캐시 13" in capsys.readouterr().out


def test_cli_collect_reports_fetch_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class Failing:
        def get(self, url: str) -> bytes:
            raise FetchError(url, 503, 4, "재시도 4회 모두 실패")

    assert (
        cli.main(["collect", "--seasons", "2026", "--raw-dir", str(tmp_path)], getter=Failing())
        == 1
    )
    assert "수집 실패" in capsys.readouterr().err


def test_cli_validate_fixture_fails_league_totals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """픽스처는 선수 14명뿐이라 기본 기준값의 리그 합계(F2-5)에서 실패해야 한다."""
    report = tmp_path / "report.json"
    code = cli.main(
        ["validate", "--seasons", "2026", "--raw-dir", str(FIXTURE_RAW), "--report", str(report)]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert report.exists()
    assert "✗ F2-5 리그 합계" in out
    assert "✓ F2-3 타자 ID 일치" in out
