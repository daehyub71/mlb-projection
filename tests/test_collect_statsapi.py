"""MLB Stats API 수집(collect/statsapi.py) 테스트 (SPEC F1·E1·E20).

요청은 가짜 getter가 계약 URL → 픽스처 바이트로 답한다. 원본 디렉토리는 tmp_path이고,
`tests/fixtures/raw`는 읽기만 한다(고칠 때는 tmp_path로 복사한 뒤 고친다).
"""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from mlbproj.collect import raw, statsapi
from mlbproj.collect.http import Fetcher, FetchError
from mlbproj.contracts import (
    PEOPLE_BATCH_SIZE,
    ROLES,
    SEASONS,
    USER_AGENT,
    Role,
    raw_people_path,
    raw_statsapi_path,
    statsapi_people_url,
    statsapi_stats_url,
)

FIXTURES = Path(__file__).parent / "fixtures" / "raw"
B = PEOPLE_BATCH_SIZE

# 픽스처에 든 선수 14명 (SOURCE.md) — people_000(타자 8명)·people_001(투수 6명)과 같다
FIXTURE_IDS = [
    408234,
    425794,
    519381,
    543037,
    571771,
    592663,
    605400,
    628452,
    646240,
    650391,
    660271,
    694973,
    701538,
    805808,
]


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


def stats_doc(season: int, role: Role) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(fixture_bytes(raw_statsapi_path(season, role)))
    return doc


def encode(doc: Any) -> bytes:
    return json.dumps(doc, ensure_ascii=False).encode("utf-8")


def fixture_people() -> list[dict[str, Any]]:
    """픽스처 선수 정보 두 파일의 people를 합친 목록."""
    people: list[dict[str, Any]] = []
    for batch in (0, 1):
        people.extend(json.loads(fixture_bytes(raw_people_path(batch)))["people"])
    return people


def fixture_people_body(ids: Iterable[int]) -> bytes:
    """픽스처 선수 정보에서 고른 ID만 담은 응답 (한 번에 받은 모양)."""
    wanted = set(ids)
    return encode(
        {"copyright": "test", "people": [p for p in fixture_people() if p["id"] in wanted]}
    )


def synthetic_people_body(ids: Iterable[int]) -> bytes:
    """합성 선수 정보 응답 (묶음이 여러 개 필요한 테스트용)."""
    return encode({"people": [{"id": i, "fullName": f"Player {i}"} for i in ids]})


def serve_people(getter: FakeGetter, ids: Iterable[int]) -> list[tuple[str, bytes]]:
    """ids를 받을 때 나갈 묶음별 (URL, 본문)을 getter에 등록하고 묶음 순서대로 돌려준다."""
    unique = sorted(set(ids))
    batches = [unique[i : i + B] for i in range(0, len(unique), B)]
    served = [(statsapi_people_url(b), synthetic_people_body(b)) for b in batches]
    getter.responses.update(served)
    return served


def people_files(raw_dir: Path) -> list[str]:
    return sorted(p.name for p in (raw_dir / "statsapi").glob("people_*.json"))


# ---------------------------------------------------------------------------
# check_stats_body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("season", SEASONS)
def test_statsapi_check_stats_body_accepts_fixtures(season: int, role: Role) -> None:
    statsapi.check_stats_body(fixture_bytes(raw_statsapi_path(season, role)))


def test_statsapi_check_stats_body_accepts_empty_splits() -> None:
    doc = stats_doc(2026, "bat")
    doc["stats"][0]["splits"] = []
    doc["stats"][0]["totalSplits"] = 0
    statsapi.check_stats_body(encode(doc))


def _with_block(doc: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {**doc, "stats": [{**doc["stats"][0], **changes}]}


def _without_block_key(doc: dict[str, Any], key: str) -> dict[str, Any]:
    block = {k: v for k, v in doc["stats"][0].items() if k != key}
    return {**doc, "stats": [block]}


def _truncated(doc: dict[str, Any]) -> dict[str, Any]:
    # limit에 잘린 응답: totalSplits가 받은 행 수보다 크다
    block = doc["stats"][0]
    return _with_block(doc, splits=block["splits"][:-1])


Mutation = Callable[[dict[str, Any]], Any]

BAD_STATS: list[Any] = [
    pytest.param(_truncated, "totalSplits", id="truncated"),
    pytest.param(
        lambda d: _with_block(d, totalSplits=d["stats"][0]["totalSplits"] + 1),
        "totalSplits",
        id="total-more-than-rows",
    ),
    pytest.param(
        lambda d: _with_block(d, totalSplits=d["stats"][0]["totalSplits"] - 1),
        "totalSplits",
        id="total-less-than-rows",
    ),
    pytest.param(lambda d: _without_block_key(d, "totalSplits"), "totalSplits", id="no-total"),
    pytest.param(lambda d: _without_block_key(d, "splits"), "splits", id="no-splits"),
    pytest.param(lambda d: _with_block(d, splits={}), "splits", id="splits-object"),
    pytest.param(lambda d: _with_block(d, splits=None), "splits", id="splits-null"),
    pytest.param(lambda d: {k: v for k, v in d.items() if k != "stats"}, "stats", id="no-stats"),
    pytest.param(lambda d: {**d, "stats": []}, "stats", id="stats-empty"),
    pytest.param(lambda d: {**d, "stats": d["stats"] * 2}, "stats", id="stats-two"),
    pytest.param(lambda d: {**d, "stats": d["stats"][0]}, "stats", id="stats-object"),
    pytest.param(lambda d: {**d, "stats": [1]}, "stats|splits", id="stats-item-not-object"),
    pytest.param(lambda d: [d], None, id="top-level-list"),
    pytest.param(lambda d: None, None, id="top-level-null"),
]


@pytest.mark.parametrize(("mutate", "match"), BAD_STATS)
def test_statsapi_check_stats_body_rejects_bad_structure(
    mutate: Mutation, match: str | None
) -> None:
    body = encode(mutate(stats_doc(2026, "pit")))
    with pytest.raises(ValueError, match=match):
        statsapi.check_stats_body(body)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"<html><body>Too Many Requests</body></html>", id="html"),
        pytest.param(b"\xff\xfe\x00\x01", id="not-utf8"),
        pytest.param(fixture_bytes(raw_statsapi_path(2026, "bat"))[:-40], id="cut-off"),
    ],
)
def test_statsapi_check_stats_body_rejects_non_json(body: bytes) -> None:
    with pytest.raises(ValueError):
        statsapi.check_stats_body(body)


@pytest.mark.parametrize(
    "total",
    [
        pytest.param(True, id="bool"),  # True == 1이어도 정수가 아니다
        pytest.param("1", id="string"),
        pytest.param([1], id="list"),
        pytest.param(None, id="null"),
    ],
)
def test_statsapi_check_stats_body_rejects_non_integer_total(total: Any) -> None:
    doc = stats_doc(2026, "bat")
    one_row = _with_block(doc, splits=doc["stats"][0]["splits"][:1])
    with pytest.raises(ValueError, match="totalSplits"):
        statsapi.check_stats_body(encode(_with_block(one_row, totalSplits=total)))


def test_statsapi_check_stats_body_rejects_bom_like_read_json(tmp_path: Path) -> None:
    # BOM이 붙은 JSON은 raw.read_json이 읽지 못한다 — 검사가 먼저 막아야 저장한 원본이 늘 읽힌다
    body = "﻿".encode() + fixture_bytes(raw_statsapi_path(2026, "bat"))
    with pytest.raises(ValueError):
        statsapi.check_stats_body(body)
    rel = Path("statsapi/bom.json")
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_bytes(body)
    with pytest.raises(ValueError):
        raw.read_json(tmp_path, rel)


# ---------------------------------------------------------------------------
# collect_season_stats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_statsapi_collect_season_stats_saves_raw_with_contract_url(
    tmp_path: Path, role: Role
) -> None:
    url = statsapi_stats_url(2025, role)
    rel = raw_statsapi_path(2025, role)
    body = fixture_bytes(rel)
    getter = FakeGetter({url: body})
    outcome = statsapi.collect_season_stats(getter, tmp_path, 2025, role)
    assert outcome.fetched
    assert outcome.path == rel
    assert getter.calls == [url]
    assert (tmp_path / rel).read_bytes() == body  # 받은 바이트 그대로
    entry = raw.load_manifest(tmp_path)[rel.as_posix()]
    assert entry == outcome.entry
    assert entry.url == url
    assert entry.bytes == len(body)


def test_statsapi_collect_season_stats_uses_cache_and_force(tmp_path: Path) -> None:
    url = statsapi_stats_url(2026, "bat")
    getter = FakeGetter({url: fixture_bytes(raw_statsapi_path(2026, "bat"))})
    first = statsapi.collect_season_stats(getter, tmp_path, 2026, "bat")
    second = statsapi.collect_season_stats(getter, tmp_path, 2026, "bat")
    assert first.fetched
    assert not second.fetched
    assert getter.calls == [url]  # 캐시가 있으면 요청하지 않는다
    assert second.path == first.path
    assert second.entry == first.entry
    forced = statsapi.collect_season_stats(getter, tmp_path, 2026, "bat", force=True)
    assert forced.fetched
    assert getter.calls == [url, url]


def test_statsapi_collect_season_stats_caches_each_season_and_role(tmp_path: Path) -> None:
    pairs: list[tuple[int, Role]] = [(2025, "bat"), (2025, "pit"), (2026, "bat")]
    getter = FakeGetter(
        {statsapi_stats_url(s, r): fixture_bytes(raw_statsapi_path(s, r)) for s, r in pairs}
    )
    for season, role in pairs:
        assert statsapi.collect_season_stats(getter, tmp_path, season, role).fetched
    assert getter.calls == [statsapi_stats_url(s, r) for s, r in pairs]
    for season, role in pairs:
        assert not statsapi.collect_season_stats(getter, tmp_path, season, role).fetched
    assert len(getter.calls) == len(pairs)


def test_statsapi_collect_season_stats_refetches_damaged_raw(tmp_path: Path) -> None:
    url = statsapi_stats_url(2026, "pit")
    rel = raw_statsapi_path(2026, "pit")
    body = fixture_bytes(rel)
    getter = FakeGetter({url: body})
    statsapi.collect_season_stats(getter, tmp_path, 2026, "pit")
    (tmp_path / rel).write_bytes(b"tampered")
    again = statsapi.collect_season_stats(getter, tmp_path, 2026, "pit")
    assert again.fetched
    assert getter.calls == [url, url]
    assert (tmp_path / rel).read_bytes() == body


def test_statsapi_collect_season_stats_rejects_invalid_body_without_saving(
    tmp_path: Path,
) -> None:
    url = statsapi_stats_url(2026, "pit")
    getter = FakeGetter({url: encode(_truncated(stats_doc(2026, "pit")))})
    with pytest.raises(ValueError):
        statsapi.collect_season_stats(getter, tmp_path, 2026, "pit")
    assert not (tmp_path / raw_statsapi_path(2026, "pit")).exists()
    assert raw.load_manifest(tmp_path) == {}


def test_statsapi_collect_season_stats_force_keeps_old_raw_when_new_body_invalid(
    tmp_path: Path,
) -> None:
    url = statsapi_stats_url(2024, "bat")
    rel = raw_statsapi_path(2024, "bat")
    good = fixture_bytes(rel)
    getter = FakeGetter({url: good})
    first = statsapi.collect_season_stats(getter, tmp_path, 2024, "bat")
    getter.responses[url] = b"<html>busy</html>"
    with pytest.raises(ValueError):
        statsapi.collect_season_stats(getter, tmp_path, 2024, "bat", force=True)
    assert (tmp_path / rel).read_bytes() == good
    assert raw.load_manifest(tmp_path)[rel.as_posix()] == first.entry


def test_statsapi_collect_season_stats_propagates_fetch_error(tmp_path: Path) -> None:
    # 받지 못한 파일은 부르는 쪽이 그 파일만 실패로 기록한다(F1) — 삼키지 않고 그대로 올린다
    url = statsapi_stats_url(2026, "bat")
    getter = FakeGetter({url: FetchError(url, 503, 4, "busy")})
    with pytest.raises(FetchError) as excinfo:
        statsapi.collect_season_stats(getter, tmp_path, 2026, "bat")
    assert excinfo.value.url == url
    assert getter.calls == [url]
    assert not (tmp_path / raw_statsapi_path(2026, "bat")).exists()
    assert raw.load_manifest(tmp_path) == {}


class _Timeline:
    """Fetcher에 넣을 가짜 시계·sleep·transport (레인 A 두 모듈을 이어서 확인한다)."""

    def __init__(self, responses: Sequence[tuple[int, bytes]]) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self.responses = list(responses)
        self.headers: list[dict[str, str]] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now

    def transport(self, url: str, headers: Mapping[str, str], timeout: float) -> tuple[int, bytes]:
        self.headers.append(dict(headers))
        return self.responses.pop(0)

    def fetcher(self) -> Fetcher:
        return Fetcher(self.transport, sleep=self.sleep, clock=self.clock)


def test_statsapi_collect_season_stats_through_fetcher_retries_then_caches(tmp_path: Path) -> None:
    rel = raw_statsapi_path(2026, "pit")
    body = fixture_bytes(rel)
    timeline = _Timeline([(503, b"busy"), (200, body)])
    fetcher = timeline.fetcher()
    outcome = statsapi.collect_season_stats(fetcher, tmp_path, 2026, "pit")
    assert outcome.fetched
    assert (tmp_path / rel).read_bytes() == body  # 503 본문은 저장되지 않는다
    assert timeline.sleeps == [1.0]  # 백오프 1초가 간격 1초도 채운다
    assert [h["User-Agent"] for h in timeline.headers] == [USER_AGENT, USER_AGENT]
    again = statsapi.collect_season_stats(fetcher, tmp_path, 2026, "pit")
    assert not again.fetched
    assert fetcher.requests_made == 2  # 캐시가 있으면 transport를 부르지 않는다


def test_statsapi_collect_season_stats_through_fetcher_not_found(tmp_path: Path) -> None:
    timeline = _Timeline([(404, b"not found")])
    with pytest.raises(FetchError) as excinfo:
        statsapi.collect_season_stats(timeline.fetcher(), tmp_path, 2026, "bat")
    assert excinfo.value.status == 404
    assert excinfo.value.url == statsapi_stats_url(2026, "bat")
    assert raw.load_manifest(tmp_path) == {}


# ---------------------------------------------------------------------------
# stats_splits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("season", "role"), [(2019, "bat"), (2020, "pit"), (2026, "bat")])
def test_statsapi_stats_splits_reads_saved_splits(season: int, role: Role) -> None:
    splits = statsapi.stats_splits(FIXTURES, season, role)
    block = stats_doc(season, role)["stats"][0]
    assert splits == block["splits"]
    assert len(splits) == block["totalSplits"]


def test_statsapi_stats_splits_pitching_rows() -> None:
    splits = statsapi.stats_splits(FIXTURES, 2026, "pit")
    assert sorted(s["player"]["id"] for s in splits) == [
        543037,
        571771,
        605400,
        628452,
        660271,
        694973,
    ]
    ohtani = next(s for s in splits if s["player"]["id"] == 660271)
    assert ohtani["position"]["abbreviation"] == "TWP"


def test_statsapi_stats_splits_round_trip_after_collect(tmp_path: Path) -> None:
    url = statsapi_stats_url(2023, "pit")
    getter = FakeGetter({url: fixture_bytes(raw_statsapi_path(2023, "pit"))})
    statsapi.collect_season_stats(getter, tmp_path, 2023, "pit")
    assert (
        statsapi.stats_splits(tmp_path, 2023, "pit") == stats_doc(2023, "pit")["stats"][0]["splits"]
    )


def test_statsapi_stats_splits_missing_raw(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        statsapi.stats_splits(tmp_path, 2026, "bat")
    rel = raw_statsapi_path(2026, "bat")
    (tmp_path / rel).parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / rel, tmp_path / rel)
    with pytest.raises(FileNotFoundError):  # 타격만 있고 투구가 없다
        statsapi.stats_splits(tmp_path, 2026, "pit")


# ---------------------------------------------------------------------------
# stats_player_ids
# ---------------------------------------------------------------------------


def test_statsapi_stats_player_ids_all_fixture_seasons() -> None:
    ids = statsapi.stats_player_ids(FIXTURES, SEASONS)
    assert ids == FIXTURE_IDS
    assert all(type(i) is int for i in ids)
    assert ids == sorted(p["id"] for p in fixture_people())  # 두 선수 정보 파일과 같은 사람들


@pytest.mark.parametrize(
    ("seasons", "expected"),
    [
        pytest.param(
            [2024],
            [543037, 571771, 592663, 605400, 628452, 646240, 650391, 660271, 694973, 701538],
            id="2024-batting-and-pitching",
        ),
        pytest.param(
            [2026],
            [
                543037,
                571771,
                592663,
                605400,
                628452,
                646240,
                650391,
                660271,
                694973,
                701538,
                805808,
            ],
            id="2026",
        ),
        pytest.param(
            [2020, 2019],
            [
                408234,
                425794,
                519381,
                543037,
                571771,
                592663,
                605400,
                628452,
                646240,
                650391,
                660271,
            ],
            id="2019-2020",
        ),
        pytest.param([], [], id="no-seasons"),
    ],
)
def test_statsapi_stats_player_ids_selected_seasons(
    seasons: list[int], expected: list[int]
) -> None:
    assert statsapi.stats_player_ids(FIXTURES, seasons) == expected


def test_statsapi_stats_player_ids_accepts_one_shot_iterable() -> None:
    seasons = (s for s in (2019, 2020))  # 한 번만 돌 수 있는 입력
    assert statsapi.stats_player_ids(FIXTURES, seasons) == [
        408234,
        425794,
        519381,
        543037,
        571771,
        592663,
        605400,
        628452,
        646240,
        650391,
        660271,
    ]


def test_statsapi_stats_player_ids_missing_season_raw(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    shutil.copytree(FIXTURES, raw_dir)
    (raw_dir / raw_statsapi_path(2024, "pit")).unlink()  # 투구 원본 하나만 없다
    with pytest.raises(FileNotFoundError):
        statsapi.stats_player_ids(raw_dir, SEASONS)
    with pytest.raises(FileNotFoundError):
        statsapi.stats_player_ids(raw_dir, [2024])
    assert statsapi.stats_player_ids(raw_dir, [2025, 2026])  # 없는 시즌을 빼면 된다


def test_statsapi_stats_player_ids_season_outside_raw() -> None:
    with pytest.raises(FileNotFoundError):
        statsapi.stats_player_ids(FIXTURES, [2018, 2019])


@pytest.mark.parametrize(
    "player",
    [
        pytest.param(None, id="no-player"),
        pytest.param({"fullName": "No Id"}, id="no-id"),
        pytest.param({"id": "605400"}, id="id-string"),
        pytest.param("605400", id="player-not-object"),
    ],
)
def test_statsapi_stats_player_ids_rejects_split_without_integer_id(
    tmp_path: Path, player: Any
) -> None:
    raw_dir = tmp_path / "raw"
    shutil.copytree(FIXTURES, raw_dir)
    rel = raw_statsapi_path(2025, "pit")
    doc = json.loads((raw_dir / rel).read_bytes())
    split = doc["stats"][0]["splits"][2]
    if player is None:
        del split["player"]
    else:
        split["player"] = player
    (raw_dir / rel).write_bytes(encode(doc))
    # KeyError가 아니라 어느 파일 몇 번째 행인지 적은 ValueError
    with pytest.raises(ValueError, match=r"statsapi/2025_pit\.json splits\[2\].*player\.id"):
        statsapi.stats_player_ids(raw_dir, [2025])


# ---------------------------------------------------------------------------
# people_batches
# ---------------------------------------------------------------------------


def test_statsapi_people_batches_chunks_by_batch_size() -> None:
    ids = list(range(1000, 1000 + 2 * B + 50))  # B=100이면 250명 → 100·100·50
    batches = statsapi.people_batches(list(reversed(ids)))
    assert [len(b) for b in batches] == [B, B, 50]
    assert [i for batch in batches for i in batch] == ids
    assert all(isinstance(b, list) for b in batches)


def test_statsapi_people_batches_dedupes_and_sorts() -> None:
    assert statsapi.people_batches([5, 3, 5, 1, 3, 1]) == [[1, 3, 5]]


def test_statsapi_people_batches_dedupes_before_chunking() -> None:
    ids = list(range(B + 50)) * 2  # 중복을 빼면 B+50명 → 두 묶음
    random.Random(7).shuffle(ids)
    batches = statsapi.people_batches(ids)
    assert batches == [list(range(B)), list(range(B, B + 50))]


@pytest.mark.parametrize(("n", "sizes"), [(1, [1]), (B, [B]), (B + 1, [B, 1]), (2 * B, [B, B])])
def test_statsapi_people_batches_boundaries(n: int, sizes: list[int]) -> None:
    assert [len(b) for b in statsapi.people_batches(list(range(n)))] == sizes


def test_statsapi_people_batches_empty() -> None:
    assert statsapi.people_batches([]) == []


def test_statsapi_people_batches_does_not_modify_input() -> None:
    ids = [9, 2, 9, 4]
    statsapi.people_batches(ids)
    assert ids == [9, 2, 9, 4]


# ---------------------------------------------------------------------------
# check_people_body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [0, 1])
def test_statsapi_check_people_body_accepts_fixtures(batch: int) -> None:
    body = fixture_bytes(raw_people_path(batch))
    ids = [p["id"] for p in json.loads(body)["people"]]
    statsapi.check_people_body(body, ids)
    statsapi.check_people_body(body, list(reversed(ids)))  # 순서는 상관없다


def test_statsapi_check_people_body_allows_extra_people() -> None:
    body = fixture_bytes(raw_people_path(0))
    statsapi.check_people_body(body, [408234, 660271])


def test_statsapi_check_people_body_names_missing_id() -> None:
    body = fixture_bytes(raw_people_path(0))
    ids = [p["id"] for p in json.loads(body)["people"]]
    with pytest.raises(ValueError, match="999999"):
        statsapi.check_people_body(body, [*ids, 999999])


def test_statsapi_check_people_body_names_every_missing_id() -> None:
    body = fixture_bytes(raw_people_path(1))
    with pytest.raises(ValueError) as excinfo:
        statsapi.check_people_body(body, [605400, 111111, 222222])
    message = str(excinfo.value)
    assert "111111" in message
    assert "222222" in message


def test_statsapi_check_people_body_people_from_other_batch_is_missing() -> None:
    # 타자 묶음 응답에 투수 ID를 요청했다면 빠진 것이다
    with pytest.raises(ValueError, match="605400"):
        statsapi.check_people_body(fixture_bytes(raw_people_path(0)), [605400])


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"<html>error</html>", id="html"),
        pytest.param(fixture_bytes(raw_people_path(0))[:-30], id="cut-off"),
        pytest.param(b'{"copyright": "x"}', id="no-people"),
        pytest.param(b'{"people": {"id": 408234}}', id="people-object"),
        pytest.param(b'{"people": null}', id="people-null"),
        pytest.param(b"[]", id="top-level-list"),
        pytest.param(b'{"people": [408234]}', id="entry-not-object"),
        pytest.param(b'{"people": [{"fullName": "No Id"}]}', id="entry-without-id"),
        pytest.param(b'{"people": [{"id": "408234"}]}', id="entry-id-string"),
        pytest.param(b'{"people": [{"id": true}]}', id="entry-id-bool"),
    ],
)
def test_statsapi_check_people_body_rejects_bad_format(body: bytes) -> None:
    with pytest.raises(ValueError):
        statsapi.check_people_body(body, [408234])


@pytest.mark.parametrize(
    ("body", "match"),
    [
        pytest.param(b"[]", "최상위", id="top-level-list"),
        pytest.param(b'{"people": {}}', "people", id="people-object"),
        pytest.param(b'{"people": [1]}', r"people\[0\]", id="entry-not-object"),
        pytest.param(b'{"people": [{"id": 1}, {}]}', r"people\[1\]", id="second-entry-no-id"),
    ],
)
def test_statsapi_check_people_body_message_names_problem(body: bytes, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        statsapi.check_people_body(body, [1])


# ---------------------------------------------------------------------------
# collect_people
# ---------------------------------------------------------------------------


def test_statsapi_collect_people_fixture_players_in_one_batch(tmp_path: Path) -> None:
    url = statsapi_people_url(FIXTURE_IDS)
    body = fixture_people_body(FIXTURE_IDS)
    getter = FakeGetter({url: body})
    outcomes = statsapi.collect_people(getter, tmp_path, [*reversed(FIXTURE_IDS), 660271])
    assert getter.calls == [url]  # 오름차순·중복 없는 ID로 부른다
    assert [o.path for o in outcomes] == [raw_people_path(0)]
    assert outcomes[0].fetched
    assert (tmp_path / raw_people_path(0)).read_bytes() == body
    entry = raw.load_manifest(tmp_path)[raw_people_path(0).as_posix()]
    assert entry == outcomes[0].entry
    assert entry.url == url
    assert people_files(tmp_path) == ["people_000.json"]


def test_statsapi_collect_people_writes_one_file_per_batch(tmp_path: Path) -> None:
    ids = list(range(1, 2 * B + 51)) * 2  # 중복을 빼면 2B+50명 → 세 묶음
    random.Random(3).shuffle(ids)
    getter = FakeGetter()
    served = serve_people(getter, ids)
    assert len(served) == 3
    outcomes = statsapi.collect_people(getter, tmp_path, ids)
    assert [o.path for o in outcomes] == [raw_people_path(i) for i in range(3)]
    assert all(o.fetched for o in outcomes)
    assert getter.calls == [url for url, _ in served]  # 묶음 순서대로 한 번씩
    manifest = raw.load_manifest(tmp_path)
    for i, (url, body) in enumerate(served):
        rel = raw_people_path(i)
        assert (tmp_path / rel).read_bytes() == body
        assert manifest[rel.as_posix()].url == url
        assert outcomes[i].entry == manifest[rel.as_posix()]
    assert people_files(tmp_path) == ["people_000.json", "people_001.json", "people_002.json"]


def test_statsapi_collect_people_caches_each_batch(tmp_path: Path) -> None:
    ids = list(range(1, 2 * B + 51))
    getter = FakeGetter()
    serve_people(getter, ids)
    statsapi.collect_people(getter, tmp_path, ids)
    assert len(getter.calls) == 3
    again = statsapi.collect_people(getter, tmp_path, ids)
    assert [o.fetched for o in again] == [False, False, False]
    assert len(getter.calls) == 3
    # 마지막 묶음만 바뀌면 그 묶음만 다시 받는다
    grown = ids + list(range(2 * B + 51, 2 * B + 61))
    serve_people(getter, grown)
    result = statsapi.collect_people(getter, tmp_path, grown)
    assert [o.fetched for o in result] == [False, False, True]
    assert getter.calls[3:] == [statsapi_people_url(list(range(2 * B + 1, 2 * B + 61)))]
    forced = statsapi.collect_people(getter, tmp_path, grown, force=True)
    assert [o.fetched for o in forced] == [True, True, True]
    assert len(getter.calls) == 7


def test_statsapi_collect_people_removes_stale_batches(tmp_path: Path) -> None:
    # 이전 실행이 남긴 묶음 (가운데 번호가 비어 있어도 지운다)과 지우면 안 되는 시즌 원본
    raw.save_raw(tmp_path, raw_people_path(1), "https://old.test/people?1", b'{"people": []}')
    raw.save_raw(tmp_path, raw_people_path(3), "https://old.test/people?3", b'{"people": []}')
    stats_rel = raw_statsapi_path(2026, "bat")
    raw.save_raw(tmp_path, stats_rel, statsapi_stats_url(2026, "bat"), fixture_bytes(stats_rel))
    getter = FakeGetter({statsapi_people_url(FIXTURE_IDS): fixture_people_body(FIXTURE_IDS)})
    outcomes = statsapi.collect_people(getter, tmp_path, FIXTURE_IDS)
    assert len(outcomes) == 1
    assert people_files(tmp_path) == ["people_000.json"]
    assert set(raw.load_manifest(tmp_path)) == {
        raw_people_path(0).as_posix(),
        stats_rel.as_posix(),
    }
    assert (tmp_path / stats_rel).read_bytes() == fixture_bytes(stats_rel)


def test_statsapi_collect_people_shrinking_list_drops_last_batch(tmp_path: Path) -> None:
    getter = FakeGetter()
    serve_people(getter, range(1, 2 * B + 51))
    statsapi.collect_people(getter, tmp_path, list(range(1, 2 * B + 51)))
    assert people_files(tmp_path) == ["people_000.json", "people_001.json", "people_002.json"]
    fewer = list(range(1, B + 51))  # 두 묶음: 첫 묶음은 같고 둘째는 바뀐다
    served = serve_people(getter, fewer)
    outcomes = statsapi.collect_people(getter, tmp_path, fewer)
    assert [o.fetched for o in outcomes] == [False, True]
    assert people_files(tmp_path) == ["people_000.json", "people_001.json"]
    manifest = raw.load_manifest(tmp_path)
    assert raw_people_path(2).as_posix() not in manifest
    assert manifest[raw_people_path(1).as_posix()].url == served[1][0]


def test_statsapi_collect_people_empty_list(tmp_path: Path) -> None:
    raw.save_raw(tmp_path, raw_people_path(0), "https://old.test/people?0", b'{"people": []}')
    getter = FakeGetter()
    assert statsapi.collect_people(getter, tmp_path, []) == []
    assert getter.calls == []
    assert people_files(tmp_path) == []  # 묶음 0개 → n ≥ 0인 묶음이 모두 남은 것이다
    assert raw.load_manifest(tmp_path) == {}


def test_statsapi_collect_people_rejects_body_missing_requested_id(tmp_path: Path) -> None:
    url = statsapi_people_url(FIXTURE_IDS)
    body = fixture_people_body(i for i in FIXTURE_IDS if i != 805808)
    getter = FakeGetter({url: body})
    with pytest.raises(ValueError, match="805808"):
        statsapi.collect_people(getter, tmp_path, FIXTURE_IDS)
    assert not (tmp_path / raw_people_path(0)).exists()
    assert raw.load_manifest(tmp_path) == {}


@pytest.mark.parametrize("failure", ["invalid-body", "fetch-error"])
def test_statsapi_collect_people_failed_batch_keeps_previous_raw(
    tmp_path: Path, failure: str
) -> None:
    # 세 묶음을 받아 둔 뒤 목록이 두 묶음으로 줄었는데 둘째 묶음에서 실패한다
    # → 아무것도 지우지 않는다 (남은 people_002도, 이전 people_001도 그대로)
    getter = FakeGetter()
    ids = list(range(1, 2 * B + 51))
    serve_people(getter, ids)
    statsapi.collect_people(getter, tmp_path, ids)
    files_before = {n: (tmp_path / "statsapi" / n).read_bytes() for n in people_files(tmp_path)}
    manifest_before = raw.load_manifest(tmp_path)
    fewer = list(range(1, B + 51))
    served = serve_people(getter, fewer)
    second_url = served[1][0]
    if failure == "invalid-body":
        getter.responses[second_url] = synthetic_people_body(range(B + 1, B + 50))  # 150이 빠졌다
        expected: type[Exception] = ValueError
    else:
        getter.responses[second_url] = FetchError(second_url, 503, 4, "busy")
        expected = FetchError
    with pytest.raises(expected):
        statsapi.collect_people(getter, tmp_path, fewer)
    assert getter.calls[-1] == second_url
    files_after = {n: (tmp_path / "statsapi" / n).read_bytes() for n in people_files(tmp_path)}
    assert files_after == files_before
    assert raw.load_manifest(tmp_path) == manifest_before


def test_statsapi_collect_people_accepts_sequence_types(tmp_path: Path) -> None:
    getter = FakeGetter({statsapi_people_url(FIXTURE_IDS): fixture_people_body(FIXTURE_IDS)})
    ids: Sequence[int] = tuple(FIXTURE_IDS)
    outcomes = statsapi.collect_people(getter, tmp_path, ids)
    assert [o.path for o in outcomes] == [raw_people_path(0)]
