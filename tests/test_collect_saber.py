"""MLB Stats API 세이버메트릭스(WAR) 수집 테스트 — collect/statsapi.py의 saber_splits·collect_saber (SPEC v0.5 F1 ⑤·E26).

`collect_saber`는 `collect_season_stats`와 같은 규칙(fetch_to_raw · check_stats_body · 캐시 · force)이다.
요청은 가짜 getter가 계약 URL(`statsapi_saber_url`) → 픽스처 바이트로 답한다. 원본 디렉토리는 tmp_path이고,
`tests/fixtures/raw`는 읽기만 한다(고칠 때는 tmp_path로 복사한 뒤 고친다).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from mlbproj.collect import raw, statsapi
from mlbproj.collect.http import Fetcher, FetchError
from mlbproj.contracts import (
    ROLES,
    SEASONS,
    USER_AGENT,
    Role,
    raw_saber_path,
    raw_statsapi_path,
    statsapi_saber_url,
    statsapi_stats_url,
)

FIXTURES = Path(__file__).parent / "fixtures" / "raw"
OHTANI = 660271


class FakeGetter:
    """URL → 응답 바이트(또는 낼 예외) 사전으로 답하는 getter. 모르는 URL이면 테스트를 실패시킨다."""

    def __init__(self, responses: Mapping[str, bytes | BaseException] | None = None) -> None:
        self.responses: dict[str, bytes | BaseException] = dict(responses or {})
        self.calls: list[str] = []

    def get(self, url: str) -> bytes:
        self.calls.append(url)
        if url not in self.responses:
            pytest.fail(f"계약에 없는 URL을 요청했다: {url}")
        response = self.responses[url]
        if isinstance(response, BaseException):
            raise response
        return response


def fixture_bytes(rel: Path) -> bytes:
    return (FIXTURES / rel).read_bytes()


def saber_doc(season: int, role: Role) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(fixture_bytes(raw_saber_path(season, role)))
    return doc


def encode(doc: Any) -> bytes:
    return json.dumps(doc, ensure_ascii=False).encode("utf-8")


def _with_block(doc: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {**doc, "stats": [{**doc["stats"][0], **changes}]}


def _without_block_key(doc: dict[str, Any], key: str) -> dict[str, Any]:
    block = {k: v for k, v in doc["stats"][0].items() if k != key}
    return {**doc, "stats": [block]}


def _truncated(doc: dict[str, Any]) -> dict[str, Any]:
    # limit에 잘린 응답: totalSplits가 받은 행 수보다 크다
    return _with_block(doc, splits=doc["stats"][0]["splits"][:-1])


# ---------------------------------------------------------------------------
# collect_saber — 저장·URL·경로
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_collect_saber_saves_raw_with_contract_url(tmp_path: Path, role: Role) -> None:
    url = statsapi_saber_url(2025, role)
    rel = raw_saber_path(2025, role)
    body = fixture_bytes(rel)
    getter = FakeGetter({url: body})
    outcome = statsapi.collect_saber(getter, tmp_path, 2025, role)
    assert outcome.fetched
    assert outcome.path == rel
    assert getter.calls == [url]  # 시즌 기록 URL이 아니라 세이버메트릭스 URL
    assert (tmp_path / rel).read_bytes() == body  # 받은 바이트 그대로
    entry = raw.load_manifest(tmp_path)[rel.as_posix()]
    assert entry == outcome.entry
    assert entry.url == url
    assert entry.bytes == len(body)
    # 시즌 기록 원본 자리에는 아무것도 쓰지 않는다
    assert not (tmp_path / raw_statsapi_path(2025, role)).exists()


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("season", SEASONS)
def test_collect_saber_accepts_every_fixture_season(
    tmp_path: Path, season: int, role: Role
) -> None:
    # 2019~2026 모든 세이버메트릭스 픽스처가 형식 검사(check_stats_body)를 통과해 저장된다
    url = statsapi_saber_url(season, role)
    rel = raw_saber_path(season, role)
    getter = FakeGetter({url: fixture_bytes(rel)})
    outcome = statsapi.collect_saber(getter, tmp_path, season, role)
    assert outcome.fetched
    assert outcome.path == rel
    assert (tmp_path / rel).read_bytes() == fixture_bytes(rel)


def test_collect_saber_uses_cache_and_force(tmp_path: Path) -> None:
    url = statsapi_saber_url(2026, "bat")
    getter = FakeGetter({url: fixture_bytes(raw_saber_path(2026, "bat"))})
    first = statsapi.collect_saber(getter, tmp_path, 2026, "bat")
    second = statsapi.collect_saber(getter, tmp_path, 2026, "bat")
    assert first.fetched
    assert not second.fetched
    assert getter.calls == [url]  # 캐시가 있으면 요청하지 않는다
    assert second.path == first.path
    assert second.entry == first.entry
    forced = statsapi.collect_saber(getter, tmp_path, 2026, "bat", force=True)
    assert forced.fetched
    assert getter.calls == [url, url]


def test_collect_saber_caches_each_season_and_role(tmp_path: Path) -> None:
    pairs: list[tuple[int, Role]] = [(2024, "bat"), (2024, "pit"), (2025, "pit"), (2020, "bat")]
    getter = FakeGetter(
        {statsapi_saber_url(s, r): fixture_bytes(raw_saber_path(s, r)) for s, r in pairs}
    )
    for season, role in pairs:
        assert statsapi.collect_saber(getter, tmp_path, season, role).fetched
    assert getter.calls == [statsapi_saber_url(s, r) for s, r in pairs]
    for season, role in pairs:
        assert not statsapi.collect_saber(getter, tmp_path, season, role).fetched
    assert len(getter.calls) == len(pairs)
    assert sorted(raw.load_manifest(tmp_path)) == sorted(
        raw_saber_path(s, r).as_posix() for s, r in pairs
    )


def test_collect_saber_does_not_collide_with_season_stats(tmp_path: Path) -> None:
    # 같은 시즌·역할의 시즌 기록과 세이버메트릭스는 다른 파일·다른 캐시다
    stats_url = statsapi_stats_url(2025, "bat")
    saber_url = statsapi_saber_url(2025, "bat")
    getter = FakeGetter(
        {
            stats_url: fixture_bytes(raw_statsapi_path(2025, "bat")),
            saber_url: fixture_bytes(raw_saber_path(2025, "bat")),
        }
    )
    assert statsapi.collect_season_stats(getter, tmp_path, 2025, "bat").fetched
    saber = statsapi.collect_saber(getter, tmp_path, 2025, "bat")
    assert saber.fetched  # 시즌 기록 캐시를 세이버메트릭스 캐시로 쓰지 않는다
    assert getter.calls == [stats_url, saber_url]
    assert (tmp_path / raw_statsapi_path(2025, "bat")).read_bytes() == fixture_bytes(
        raw_statsapi_path(2025, "bat")
    )
    assert (tmp_path / raw_saber_path(2025, "bat")).read_bytes() == fixture_bytes(
        raw_saber_path(2025, "bat")
    )
    manifest = raw.load_manifest(tmp_path)
    assert manifest[raw_statsapi_path(2025, "bat").as_posix()].url == stats_url
    assert manifest[raw_saber_path(2025, "bat").as_posix()].url == saber_url
    assert not statsapi.collect_saber(getter, tmp_path, 2025, "bat").fetched
    assert not statsapi.collect_season_stats(getter, tmp_path, 2025, "bat").fetched
    assert len(getter.calls) == 2


def test_collect_saber_refetches_when_cached_under_other_url(tmp_path: Path) -> None:
    # 같은 경로라도 매니페스트 URL이 세이버메트릭스 URL이 아니면 캐시가 아니다
    rel = raw_saber_path(2025, "pit")
    body = fixture_bytes(rel)
    raw.save_raw(tmp_path, rel, statsapi_stats_url(2025, "pit"), body)
    url = statsapi_saber_url(2025, "pit")
    getter = FakeGetter({url: body})
    outcome = statsapi.collect_saber(getter, tmp_path, 2025, "pit")
    assert outcome.fetched
    assert getter.calls == [url]
    assert raw.load_manifest(tmp_path)[rel.as_posix()].url == url


def test_collect_saber_refetches_damaged_raw(tmp_path: Path) -> None:
    url = statsapi_saber_url(2026, "pit")
    rel = raw_saber_path(2026, "pit")
    body = fixture_bytes(rel)
    getter = FakeGetter({url: body})
    statsapi.collect_saber(getter, tmp_path, 2026, "pit")
    (tmp_path / rel).write_bytes(b"tampered")
    again = statsapi.collect_saber(getter, tmp_path, 2026, "pit")
    assert again.fetched
    assert getter.calls == [url, url]
    assert (tmp_path / rel).read_bytes() == body


def test_collect_saber_refetches_deleted_raw(tmp_path: Path) -> None:
    url = statsapi_saber_url(2023, "bat")
    rel = raw_saber_path(2023, "bat")
    getter = FakeGetter({url: fixture_bytes(rel)})
    statsapi.collect_saber(getter, tmp_path, 2023, "bat")
    (tmp_path / rel).unlink()
    assert statsapi.collect_saber(getter, tmp_path, 2023, "bat").fetched
    assert getter.calls == [url, url]


# ---------------------------------------------------------------------------
# collect_saber — 응답 검사(check_stats_body)와 실패
# ---------------------------------------------------------------------------


Mutation = Callable[[dict[str, Any]], Any]

BAD_SABER: list[Any] = [
    pytest.param(_truncated, id="truncated"),
    pytest.param(
        lambda d: _with_block(d, totalSplits=d["stats"][0]["totalSplits"] + 1),
        id="total-more-than-rows",
    ),
    pytest.param(
        lambda d: _with_block(d, totalSplits=d["stats"][0]["totalSplits"] - 1),
        id="total-less-than-rows",
    ),
    pytest.param(lambda d: _without_block_key(d, "totalSplits"), id="no-total"),
    pytest.param(lambda d: _with_block(d, totalSplits=True), id="total-bool"),
    pytest.param(lambda d: _without_block_key(d, "splits"), id="no-splits"),
    pytest.param(lambda d: _with_block(d, splits={}), id="splits-object"),
    pytest.param(lambda d: {k: v for k, v in d.items() if k != "stats"}, id="no-stats"),
    pytest.param(lambda d: {**d, "stats": []}, id="stats-empty"),
    pytest.param(lambda d: {**d, "stats": d["stats"] * 2}, id="stats-two"),
    pytest.param(lambda d: [d], id="top-level-list"),
]


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("mutate", BAD_SABER)
def test_collect_saber_rejects_bad_structure_without_saving(
    tmp_path: Path, mutate: Mutation, role: Role
) -> None:
    url = statsapi_saber_url(2025, role)
    getter = FakeGetter({url: encode(mutate(saber_doc(2025, role)))})
    with pytest.raises(ValueError):
        statsapi.collect_saber(getter, tmp_path, 2025, role)
    assert not (tmp_path / raw_saber_path(2025, role)).exists()
    assert raw.load_manifest(tmp_path) == {}


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"<html><body>Too Many Requests</body></html>", id="html"),
        pytest.param(b"\xff\xfe\x00\x01", id="not-utf8"),
        pytest.param(fixture_bytes(raw_saber_path(2025, "bat"))[:-40], id="cut-off"),
        pytest.param("﻿".encode() + fixture_bytes(raw_saber_path(2025, "bat")), id="bom"),
    ],
)
def test_collect_saber_rejects_non_json_without_saving(tmp_path: Path, body: bytes) -> None:
    url = statsapi_saber_url(2025, "bat")
    getter = FakeGetter({url: body})
    with pytest.raises(ValueError):
        statsapi.collect_saber(getter, tmp_path, 2025, "bat")
    assert not (tmp_path / raw_saber_path(2025, "bat")).exists()
    assert raw.load_manifest(tmp_path) == {}


def test_collect_saber_accepts_empty_splits(tmp_path: Path) -> None:
    # 행이 0개여도 totalSplits = 0이면 형식은 맞다 (리그 합 검사는 F2 ⑦이 한다)
    doc = _with_block(saber_doc(2025, "pit"), splits=[], totalSplits=0)
    url = statsapi_saber_url(2025, "pit")
    getter = FakeGetter({url: encode(doc)})
    assert statsapi.collect_saber(getter, tmp_path, 2025, "pit").fetched
    assert statsapi.saber_splits(tmp_path, 2025, "pit") == []


def test_collect_saber_force_keeps_old_raw_when_new_body_invalid(tmp_path: Path) -> None:
    url = statsapi_saber_url(2024, "bat")
    rel = raw_saber_path(2024, "bat")
    good = fixture_bytes(rel)
    getter = FakeGetter({url: good})
    first = statsapi.collect_saber(getter, tmp_path, 2024, "bat")
    getter.responses[url] = encode(_truncated(saber_doc(2024, "bat")))
    with pytest.raises(ValueError):
        statsapi.collect_saber(getter, tmp_path, 2024, "bat", force=True)
    assert (tmp_path / rel).read_bytes() == good
    assert raw.load_manifest(tmp_path)[rel.as_posix()] == first.entry


def test_collect_saber_propagates_fetch_error(tmp_path: Path) -> None:
    # 받지 못한 파일은 부르는 쪽이 그 파일만 실패로 기록한다(F1) — 삼키지 않고 그대로 올린다
    url = statsapi_saber_url(2026, "pit")
    getter = FakeGetter({url: FetchError(url, 503, 4, "busy")})
    with pytest.raises(FetchError) as excinfo:
        statsapi.collect_saber(getter, tmp_path, 2026, "pit")
    assert excinfo.value.url == url
    assert getter.calls == [url]
    assert not (tmp_path / raw_saber_path(2026, "pit")).exists()
    assert raw.load_manifest(tmp_path) == {}


class _Timeline:
    """Fetcher에 넣을 가짜 시계·sleep·transport."""

    def __init__(self, responses: Sequence[tuple[int, bytes]]) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self.responses = list(responses)
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now

    def transport(self, url: str, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
        self.urls.append(url)
        self.headers.append(dict(headers))
        return self.responses.pop(0)

    def fetcher(self) -> Fetcher:
        return Fetcher(self.transport, sleep=self.sleep, clock=self.clock)


def test_collect_saber_through_fetcher_retries_then_caches(tmp_path: Path) -> None:
    rel = raw_saber_path(2025, "bat")
    body = fixture_bytes(rel)
    timeline = _Timeline([(503, b"busy"), (200, body)])
    fetcher = timeline.fetcher()
    outcome = statsapi.collect_saber(fetcher, tmp_path, 2025, "bat")
    assert outcome.fetched
    assert (tmp_path / rel).read_bytes() == body  # 503 본문은 저장되지 않는다
    assert timeline.urls == [statsapi_saber_url(2025, "bat")] * 2
    assert [h["User-Agent"] for h in timeline.headers] == [USER_AGENT, USER_AGENT]
    again = statsapi.collect_saber(fetcher, tmp_path, 2025, "bat")
    assert not again.fetched
    assert fetcher.requests_made == 2  # 캐시가 있으면 transport를 부르지 않는다


# ---------------------------------------------------------------------------
# saber_splits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("season", "role"),
    [(2019, "bat"), (2020, "pit"), (2025, "bat"), (2025, "pit"), (2026, "pit")],
)
def test_collect_saber_splits_reads_saved_splits(season: int, role: Role) -> None:
    splits = statsapi.saber_splits(FIXTURES, season, role)
    block = saber_doc(season, role)["stats"][0]
    assert splits == block["splits"]
    assert len(splits) == block["totalSplits"]
    assert all("war" in split["stat"] for split in splits)  # 시즌 기록이 아니라 세이버메트릭스


def test_collect_saber_splits_round_trip_after_collect(tmp_path: Path) -> None:
    url = statsapi_saber_url(2023, "pit")
    getter = FakeGetter({url: fixture_bytes(raw_saber_path(2023, "pit"))})
    statsapi.collect_saber(getter, tmp_path, 2023, "pit")
    assert (
        statsapi.saber_splits(tmp_path, 2023, "pit") == saber_doc(2023, "pit")["stats"][0]["splits"]
    )


def test_collect_saber_splits_missing_raw(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        statsapi.saber_splits(tmp_path, 2025, "bat")
    # 시즌 기록 원본만 있고 세이버메트릭스가 없으면 시즌 기록으로 대신 읽지 않는다
    rel = raw_statsapi_path(2025, "bat")
    (tmp_path / rel).parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / rel, tmp_path / rel)
    with pytest.raises(FileNotFoundError):
        statsapi.saber_splits(tmp_path, 2025, "bat")
    # 타격 세이버메트릭스만 있고 투구가 없다
    saber_rel = raw_saber_path(2025, "bat")
    shutil.copyfile(FIXTURES / saber_rel, tmp_path / saber_rel)
    assert statsapi.saber_splits(tmp_path, 2025, "bat")
    with pytest.raises(FileNotFoundError):
        statsapi.saber_splits(tmp_path, 2025, "pit")


def test_collect_saber_splits_season_outside_raw() -> None:
    with pytest.raises(FileNotFoundError):
        statsapi.saber_splits(FIXTURES, 2018, "bat")


def _ohtani_war(season: int) -> dict[Role, float]:
    found: dict[Role, float] = {}
    for role in ROLES:
        for split in statsapi.saber_splits(FIXTURES, season, role):
            if split["player"]["id"] == OHTANI:
                found[role] = split["stat"]["war"]
    return found


@pytest.mark.parametrize(
    ("season", "roles", "total"),
    [
        (2021, ("bat", "pit"), 7.98),
        (2022, ("bat", "pit"), 9.20),
        (2023, ("bat", "pit"), 8.87),
        (2024, ("bat",), 8.93),  # 2024는 투구 없음 (E7)
        (2025, ("bat", "pit"), 9.37),
        (2026, ("bat", "pit"), 6.63),
    ],
)
def test_collect_saber_splits_ohtani_two_way_war_matches_spec(
    season: int, roles: tuple[Role, ...], total: float
) -> None:
    # SPEC E26의 오타니 투타 합산 WAR — 두 역할 원본에서 각각 읽어 더한다
    war = _ohtani_war(season)
    assert tuple(sorted(war)) == tuple(sorted(roles))
    assert round(sum(war.values()), 2) == pytest.approx(total)


def test_collect_saber_splits_ohtani_2025_components() -> None:
    war = _ohtani_war(2025)
    assert war["bat"] == pytest.approx(7.45112)
    assert war["pit"] == pytest.approx(1.91503)
