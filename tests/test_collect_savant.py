"""Savant 수집(collect/savant.py) 테스트 — 리더보드·OAA·FRV (SPEC F1·E2~E4·E18·E19).

외부 요청은 하지 않는다. 응답은 가짜 getter가 `tests/fixtures/raw/savant`의 실제 CSV를 돌려준다.
고친 원본이 필요한 테스트는 픽스처를 `tmp_path`로 복사해서 바꾼다.
"""

import csv
import io
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from pandas.api import types as ptypes

from mlbproj import contracts as c
from mlbproj.collect import raw, savant
from mlbproj.collect.raw import CollectOutcome, Getter
from mlbproj.contracts import SavantKind

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RAW = ROOT / "tests" / "fixtures" / "raw"
BOM = b"\xef\xbb\xbf"
KINDS: tuple[SavantKind, ...] = ("batter", "pitcher", "oaa", "frv")
LEADERBOARD_KINDS: tuple[SavantKind, ...] = ("batter", "pitcher")

# 픽스처에 있는 Savant 원본 전부: 리더보드는 2019~2026, 수비는 2022~2026
FIXTURE_CASES: list[tuple[int, SavantKind]] = [
    (season, kind) for season in c.SEASONS for kind in LEADERBOARD_KINDS
] + [(season, kind) for season in c.FIELDING_YEARS for kind in ("oaa", "frv")]
LEADERBOARD_CASES = [case for case in FIXTURE_CASES if case[1] in LEADERBOARD_KINDS]

HTML_ERROR_PAGE = (
    b"<!DOCTYPE html>\n<html lang='en'><head><title>Baseball Savant</title></head>\n"
    b"<body><h1>Something went wrong</h1><p>Please try again later.</p></body></html>\n"
)

VENDITTE = 519381  # Savant 투수 2019·2020에 값까지 같은 2행 (E19)

# 수비 수집기: (kind, 수집 함수, URL 함수)
FieldingCollector = Callable[..., CollectOutcome]
FIELDING_COLLECTORS = [
    pytest.param("oaa", savant.collect_oaa, c.savant_oaa_url, id="oaa"),
    pytest.param("frv", savant.collect_frv, c.savant_frv_url, id="frv"),
]


def _collect_bat(
    getter: Getter, raw_dir: Path, season: int, *, force: bool = False
) -> CollectOutcome:
    return savant.collect_leaderboard(getter, raw_dir, season, "bat", force=force)


def _collect_pit(
    getter: Getter, raw_dir: Path, season: int, *, force: bool = False
) -> CollectOutcome:
    return savant.collect_leaderboard(getter, raw_dir, season, "pit", force=force)


# 네 종류 수집기 모두: (kind, 수집 함수, URL 함수)
ALL_COLLECTORS = [
    pytest.param("batter", _collect_bat, lambda s: c.savant_leaderboard_url(s, "bat"), id="batter"),
    pytest.param(
        "pitcher", _collect_pit, lambda s: c.savant_leaderboard_url(s, "pit"), id="pitcher"
    ),
    *FIELDING_COLLECTORS,
]


class FakeGetter:
    """정해 둔 바이트를 돌려주고 요청 URL을 기록하는 가짜 getter."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls: list[str] = []

    def get(self, url: str) -> bytes:
        self.calls.append(url)
        return self.body


def _getter(body: bytes) -> FakeGetter:
    getter = FakeGetter(body)
    _: Getter = getter  # Getter 프로토콜을 따르는지 (정적 검사용)
    return getter


def _fixture_path(season: int, kind: SavantKind) -> Path:
    return FIXTURE_RAW / c.raw_savant_path(season, kind)


def _fixture_bytes(season: int, kind: SavantKind) -> bytes:
    return _fixture_path(season, kind).read_bytes()


def _raw_frame(season: int, kind: SavantKind) -> pd.DataFrame:
    """비교 기준: 손질 없이 읽은 원본 (`collect.raw.read_csv`)."""
    return raw.read_csv(FIXTURE_RAW, c.raw_savant_path(season, kind))


def _header_line(body: bytes) -> bytes:
    """BOM을 포함한 첫 줄 (줄바꿈 제외)."""
    return body.split(b"\n", 1)[0]


def _rewrite_rows(body: bytes, edit: Callable[[list[list[str]]], list[list[str]]]) -> bytes:
    """CSV를 행 목록으로 읽어 `edit`으로 바꾼 뒤 다시 쓴다 (BOM 유지). rows[0]은 머리줄이다."""
    rows = list(csv.reader(io.StringIO(body.decode("utf-8-sig"))))
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(edit(rows))
    return BOM + out.getvalue().encode("utf-8")


def _drop_column(body: bytes, column: str) -> bytes:
    """머리줄과 모든 데이터 행에서 열 하나를 정확히 지운 CSV (BOM 유지)."""

    def drop(rows: list[list[str]]) -> list[list[str]]:
        index = rows[0].index(column)
        return [row[:index] + row[index + 1 :] for row in rows]

    return _rewrite_rows(body, drop)


def _set_field(body: bytes, column: str, value: str, *, row: int = 1) -> bytes:
    """데이터 행 하나(`row`, 1부터)의 필드 하나만 바꾼 CSV."""

    def set_one(rows: list[list[str]]) -> list[list[str]]:
        rows[row][rows[0].index(column)] = value
        return rows

    return _rewrite_rows(body, set_one)


def _add_column(body: bytes, column: str, value: str) -> bytes:
    """모든 데이터 행에 같은 값을 가진 열 하나를 끝에 더한 CSV."""
    return _rewrite_rows(
        body, lambda rows: [[*rows[0], column]] + [[*row, value] for row in rows[1:]]
    )


def _raw_copy_with(tmp_path: Path, season: int, kind: SavantKind, body: bytes) -> Path:
    """픽스처 원본 디렉토리를 복사하고 원본 하나를 `body`로 바꾼다. 복사한 디렉토리를 돌려준다."""
    raw_dir = tmp_path / "raw"
    shutil.copytree(FIXTURE_RAW, raw_dir)
    (raw_dir / c.raw_savant_path(season, kind)).write_bytes(body)
    return raw_dir


def _files_under(path: Path) -> list[Path]:
    return [p for p in path.rglob("*") if p.is_file()]


def _assert_saved(raw_dir: Path, outcome: CollectOutcome, rel: Path, url: str, body: bytes) -> None:
    """받은 바이트가 그대로 저장되고 매니페스트에 요청 URL이 남았는지."""
    assert outcome.path == rel
    assert (raw_dir / rel).read_bytes() == body  # BOM까지 원형 그대로 (D6)
    manifest = raw.load_manifest(raw_dir)
    entry = manifest[rel.as_posix()]
    assert entry.url == url
    assert entry.bytes == len(body)
    assert outcome.entry == entry


# ---------------------------------------------------------------------------
# required_columns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("batter", c.SAVANT_BATTER_COLUMNS),
        ("pitcher", c.SAVANT_PITCHER_COLUMNS),
        ("oaa", c.SAVANT_OAA_COLUMNS),
        ("frv", c.SAVANT_FRV_COLUMNS),
    ],
)
def test_collect_savant_required_columns_match_contracts(
    kind: SavantKind, expected: tuple[str, ...]
) -> None:
    assert savant.required_columns(kind) == expected


# ---------------------------------------------------------------------------
# check_csv_body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("season", "kind"), FIXTURE_CASES)
def test_collect_savant_check_accepts_fixture_with_bom(season: int, kind: SavantKind) -> None:
    body = _fixture_bytes(season, kind)
    assert body.startswith(BOM)  # 전제: 실제 응답은 BOM으로 시작한다
    savant.check_csv_body(body, kind)  # 예외가 없으면 통과


@pytest.mark.parametrize(("season", "kind"), FIXTURE_CASES)
def test_collect_savant_check_accepts_fixture_without_bom(season: int, kind: SavantKind) -> None:
    body = _fixture_bytes(season, kind).removeprefix(BOM)
    assert not body.startswith(BOM)
    savant.check_csv_body(body, kind)  # 예외가 없으면 통과


@pytest.mark.parametrize("kind", KINDS)
def test_collect_savant_check_rejects_html(kind: SavantKind) -> None:
    """Savant는 오류일 때도 200으로 HTML을 돌려줄 수 있다."""
    with pytest.raises(ValueError):
        savant.check_csv_body(HTML_ERROR_PAGE, kind)


@pytest.mark.parametrize(
    ("season", "kind", "column"),
    [
        (2026, "batter", "xwoba"),
        (2026, "batter", "last_name, first_name"),  # 쉼표가 든 열 이름
        (2019, "pitcher", "xera"),
        (2026, "pitcher", "player_id"),
        (2022, "oaa", "outs_above_average"),  # outs_above_average_* 열은 남아 있다
        (2026, "oaa", "year"),
        (2026, "frv", "id"),
        (2024, "frv", "outs_9"),
    ],
)
def test_collect_savant_check_rejects_missing_column(
    season: int, kind: SavantKind, column: str
) -> None:
    body = _drop_column(_fixture_bytes(season, kind), column)
    with pytest.raises(ValueError, match=re.escape(column)):
        savant.check_csv_body(body, kind)


@pytest.mark.parametrize(
    ("season", "body_kind", "checked_kind", "missing"),
    [
        (2026, "pitcher", "batter", "home_run"),
        (2026, "batter", "pitcher", "p_formatted_ip"),
        (2026, "frv", "oaa", "outs_above_average"),
        (2026, "oaa", "frv", "total_runs"),
    ],
)
def test_collect_savant_check_rejects_other_kind(
    season: int, body_kind: SavantKind, checked_kind: SavantKind, missing: str
) -> None:
    """다른 종류의 CSV는 그 종류의 필수 열이 빠져 있어서 거부된다."""
    with pytest.raises(ValueError, match=re.escape(missing)):
        savant.check_csv_body(_fixture_bytes(season, body_kind), checked_kind)


@pytest.mark.parametrize("suffix", [b"", b"\n", b"\n\n"], ids=["no_newline", "newline", "blank"])
@pytest.mark.parametrize(("season", "kind"), [(2026, k) for k in KINDS])
def test_collect_savant_check_rejects_header_only(
    season: int, kind: SavantKind, suffix: bytes
) -> None:
    body = _header_line(_fixture_bytes(season, kind)) + suffix
    with pytest.raises(ValueError):
        savant.check_csv_body(body, kind)


@pytest.mark.parametrize("kind", KINDS)
def test_collect_savant_check_rejects_empty_body(kind: SavantKind) -> None:
    with pytest.raises(ValueError):
        savant.check_csv_body(b"", kind)


@pytest.mark.parametrize("kind", KINDS)
def test_collect_savant_check_rejects_non_utf8(kind: SavantKind) -> None:
    text = _fixture_bytes(2026, kind).decode("utf-8-sig")
    with pytest.raises(ValueError):
        savant.check_csv_body(text.encode("utf-16"), kind)


# ---------------------------------------------------------------------------
# collect_leaderboard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("season", [2019, 2020, 2026])
@pytest.mark.parametrize(("role", "kind"), [("bat", "batter"), ("pit", "pitcher")])
def test_collect_savant_leaderboard_role_maps_to_kind(
    tmp_path: Path, season: int, role: c.Role, kind: SavantKind
) -> None:
    body = _fixture_bytes(season, kind)
    getter = _getter(body)
    outcome = savant.collect_leaderboard(getter, tmp_path, season, role)
    url = c.savant_leaderboard_url(season, role)
    assert getter.calls == [url]
    assert outcome.fetched
    _assert_saved(tmp_path, outcome, c.raw_savant_path(season, kind), url, body)


@pytest.mark.parametrize(("role", "kind"), [("bat", "batter"), ("pit", "pitcher")])
def test_collect_savant_leaderboard_cache_hit_makes_no_request(
    tmp_path: Path, role: c.Role, kind: SavantKind
) -> None:
    body = _fixture_bytes(2026, kind)
    getter = _getter(body)
    first = savant.collect_leaderboard(getter, tmp_path, 2026, role)
    second = savant.collect_leaderboard(getter, tmp_path, 2026, role)
    assert first.fetched and not second.fetched
    assert getter.calls == [c.savant_leaderboard_url(2026, role)]
    assert second.path == first.path and second.entry == first.entry


@pytest.mark.parametrize(("role", "kind"), [("bat", "batter"), ("pit", "pitcher")])
def test_collect_savant_leaderboard_uses_existing_cache(
    tmp_path: Path, role: c.Role, kind: SavantKind
) -> None:
    """이전 실행이 같은 URL로 저장해 둔 원본이 있으면 요청하지 않는다."""
    body = _fixture_bytes(2025, kind)
    rel = c.raw_savant_path(2025, kind)
    url = c.savant_leaderboard_url(2025, role)
    raw.save_raw(tmp_path, rel, url, body)
    getter = _getter(b"should not be requested")
    outcome = savant.collect_leaderboard(getter, tmp_path, 2025, role)
    assert getter.calls == []
    assert not outcome.fetched
    assert outcome.path == rel
    assert (tmp_path / rel).read_bytes() == body


@pytest.mark.parametrize(("role", "kind"), [("bat", "batter"), ("pit", "pitcher")])
def test_collect_savant_leaderboard_force_refetches(
    tmp_path: Path, role: c.Role, kind: SavantKind
) -> None:
    old = _fixture_bytes(2024, kind)
    new = _fixture_bytes(2026, kind).replace(b",2026,", b",2024,")
    getter = _getter(old)
    savant.collect_leaderboard(getter, tmp_path, 2024, role)
    getter.body = new
    forced = savant.collect_leaderboard(getter, tmp_path, 2024, role, force=True)
    url = c.savant_leaderboard_url(2024, role)
    assert getter.calls == [url, url]
    assert forced.fetched
    _assert_saved(tmp_path, forced, c.raw_savant_path(2024, kind), url, new)


@pytest.mark.parametrize(
    "bad_body",
    [
        pytest.param(HTML_ERROR_PAGE, id="html"),
        pytest.param("header_only", id="header_only"),
        pytest.param("other_kind", id="other_kind"),
    ],
)
@pytest.mark.parametrize(("role", "kind"), [("bat", "batter"), ("pit", "pitcher")])
def test_collect_savant_leaderboard_invalid_body_saves_nothing(
    tmp_path: Path, role: c.Role, kind: SavantKind, bad_body: bytes | str
) -> None:
    if bad_body == "header_only":
        body = _header_line(_fixture_bytes(2026, kind)) + b"\n"
    elif bad_body == "other_kind":
        body = _fixture_bytes(2026, "pitcher" if kind == "batter" else "batter")
    else:
        assert isinstance(bad_body, bytes)
        body = bad_body
    getter = _getter(body)
    with pytest.raises(ValueError):
        savant.collect_leaderboard(getter, tmp_path, 2026, role)
    assert getter.calls == [c.savant_leaderboard_url(2026, role)]
    assert not (tmp_path / c.raw_savant_path(2026, kind)).exists()
    assert raw.load_manifest(tmp_path) == {}


# ---------------------------------------------------------------------------
# collect_oaa / collect_frv
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("season", [2019, 2020, 2021, 2027])
@pytest.mark.parametrize(("kind", "collect", "url_of"), FIELDING_COLLECTORS)
def test_collect_savant_fielding_rejects_season_outside_years(
    tmp_path: Path,
    kind: SavantKind,
    collect: FieldingCollector,
    url_of: Callable[[int], str],
    season: int,
) -> None:
    assert season not in c.FIELDING_YEARS
    getter = _getter(_fixture_bytes(2022, kind))
    with pytest.raises(ValueError):
        collect(getter, tmp_path, season)
    assert getter.calls == []  # 요청하지 않는다
    assert _files_under(tmp_path) == []


@pytest.mark.parametrize("season", c.FIELDING_YEARS)
@pytest.mark.parametrize(("kind", "collect", "url_of"), FIELDING_COLLECTORS)
def test_collect_savant_fielding_collects_each_year(
    tmp_path: Path,
    kind: SavantKind,
    collect: FieldingCollector,
    url_of: Callable[[int], str],
    season: int,
) -> None:
    body = _fixture_bytes(season, kind)
    getter = _getter(body)
    outcome = collect(getter, tmp_path, season)
    url = url_of(season)
    assert getter.calls == [url]
    assert outcome.fetched
    _assert_saved(tmp_path, outcome, c.raw_savant_path(season, kind), url, body)


@pytest.mark.parametrize(("kind", "collect", "url_of"), FIELDING_COLLECTORS)
def test_collect_savant_fielding_cache_and_force(
    tmp_path: Path,
    kind: SavantKind,
    collect: FieldingCollector,
    url_of: Callable[[int], str],
) -> None:
    body = _fixture_bytes(2023, kind)
    getter = _getter(body)
    url = url_of(2023)
    first = collect(getter, tmp_path, 2023)
    second = collect(getter, tmp_path, 2023)
    assert first.fetched and not second.fetched
    assert getter.calls == [url]  # 캐시가 있으면 요청하지 않는다
    forced = collect(getter, tmp_path, 2023, force=True)
    assert forced.fetched
    assert getter.calls == [url, url]
    _assert_saved(tmp_path, forced, c.raw_savant_path(2023, kind), url, body)


@pytest.mark.parametrize(
    "bad_body",
    [
        pytest.param(HTML_ERROR_PAGE, id="html"),
        pytest.param("header_only", id="header_only"),
        pytest.param("other_kind", id="other_kind"),
    ],
)
@pytest.mark.parametrize(("kind", "collect", "url_of"), FIELDING_COLLECTORS)
def test_collect_savant_fielding_invalid_body_saves_nothing(
    tmp_path: Path,
    kind: SavantKind,
    collect: FieldingCollector,
    url_of: Callable[[int], str],
    bad_body: bytes | str,
) -> None:
    if bad_body == "header_only":
        body = _header_line(_fixture_bytes(2026, kind)) + b"\n"
    elif bad_body == "other_kind":
        body = _fixture_bytes(2026, "frv" if kind == "oaa" else "oaa")
    else:
        assert isinstance(bad_body, bytes)
        body = bad_body
    getter = _getter(body)
    with pytest.raises(ValueError):
        collect(getter, tmp_path, 2026)
    assert getter.calls == [url_of(2026)]
    assert not (tmp_path / c.raw_savant_path(2026, kind)).exists()
    assert raw.load_manifest(tmp_path) == {}


# ---------------------------------------------------------------------------
# load_savant
# ---------------------------------------------------------------------------


def _raw_id_column(kind: SavantKind) -> str:
    return "id" if kind == "frv" else "player_id"


@pytest.mark.parametrize(("season", "kind"), FIXTURE_CASES)
def test_collect_savant_load_ids_and_year(season: int, kind: SavantKind) -> None:
    df = savant.load_savant(FIXTURE_RAW, season, kind)
    source = _raw_frame(season, kind)
    assert len(df) == len(source)  # 행을 지우거나 더하지 않는다
    assert not any(str(col).startswith("﻿") for col in df.columns)  # BOM을 뗐다
    assert ptypes.is_integer_dtype(df["player_id"].dtype)
    assert df["player_id"].tolist() == source[_raw_id_column(kind)].astype("int64").tolist()
    assert ptypes.is_integer_dtype(df["year"].dtype)
    assert (df["year"] == season).all()


@pytest.mark.parametrize(("season", "kind"), FIXTURE_CASES)
def test_collect_savant_load_keeps_other_columns(season: int, kind: SavantKind) -> None:
    """ID·year 밖의 열은 이름·값·행 순서를 바꾸지 않는다 (정제는 M2)."""
    df = savant.load_savant(FIXTURE_RAW, season, kind)
    source = _raw_frame(season, kind)
    others = [col for col in source.columns if col not in {"id", "player_id", "year"}]
    assert others
    assert set(others) <= set(df.columns)
    pd.testing.assert_frame_equal(
        df[others].reset_index(drop=True), source[others].reset_index(drop=True)
    )


@pytest.mark.parametrize(("season", "kind"), LEADERBOARD_CASES)
def test_collect_savant_load_leaderboard_columns_unchanged(season: int, kind: SavantKind) -> None:
    df = savant.load_savant(FIXTURE_RAW, season, kind)
    assert list(df.columns) == list(_raw_frame(season, kind).columns)
    assert set(savant.required_columns(kind)) <= set(df.columns)


@pytest.mark.parametrize("season", c.FIELDING_YEARS)
def test_collect_savant_load_oaa_fills_empty_year(season: int) -> None:
    source = _raw_frame(season, "oaa")
    assert source["year"].isna().all()  # 전제: 원본 year는 비어 있다 (E4)
    df = savant.load_savant(FIXTURE_RAW, season, "oaa")
    assert list(df.columns) == list(source.columns)
    assert ptypes.is_integer_dtype(df["year"].dtype)
    assert df["year"].tolist() == [season] * len(source)
    assert set(c.SAVANT_OAA_COLUMNS) <= set(df.columns)


@pytest.mark.parametrize("season", c.FIELDING_YEARS)
def test_collect_savant_load_frv_renames_id_and_adds_year(season: int) -> None:
    source = _raw_frame(season, "frv")
    assert "year" not in source.columns and "player_id" not in source.columns  # 전제 (E18)
    df = savant.load_savant(FIXTURE_RAW, season, "frv")
    assert "id" not in df.columns  # 원본 `id`는 `player_id`가 된다
    assert set(df.columns) == (set(source.columns) - {"id"}) | {"player_id", "year"}
    assert len(df.columns) == len(source.columns) + 1
    assert df["player_id"].tolist() == source["id"].astype("int64").tolist()
    assert df["year"].tolist() == [season] * len(source)


@pytest.mark.parametrize("season", [2019, 2020])
def test_collect_savant_load_keeps_identical_duplicate_rows(season: int) -> None:
    """값까지 같은 중복 행(E19)도 그대로 둔다. 합치는 것은 검증(F2 ②)의 일이다."""
    df = savant.load_savant(FIXTURE_RAW, season, "pitcher")
    assert len(df) == len(_raw_frame(season, "pitcher"))
    dup = df[df["player_id"] == VENDITTE]
    assert len(dup) == 2
    assert dup.duplicated(keep=False).all()


@pytest.mark.parametrize("mode", ["last_row", "whole_file"])
@pytest.mark.parametrize("kind", LEADERBOARD_KINDS)
def test_collect_savant_load_leaderboard_year_mismatch_raises(
    tmp_path: Path, kind: SavantKind, mode: str
) -> None:
    raw_dir = tmp_path / "raw"
    shutil.copytree(FIXTURE_RAW, raw_dir)
    target = raw_dir / c.raw_savant_path(2026, kind)
    original = target.read_bytes()
    if mode == "whole_file":
        # 2025 원본을 2026 자리에 둔 경우
        changed = _fixture_bytes(2025, kind)
    else:
        # 마지막 데이터 행 하나만 year가 다르다
        lines = original.split(b"\n")
        last = max(i for i, line in enumerate(lines) if line.strip())
        lines[last] = lines[last].replace(b",2026,", b",2025,", 1)
        changed = b"\n".join(lines)
    assert changed != original
    target.write_bytes(changed)
    with pytest.raises(ValueError):
        savant.load_savant(raw_dir, 2026, kind)


@pytest.mark.parametrize("kind", KINDS)
def test_collect_savant_load_missing_file_raises(tmp_path: Path, kind: SavantKind) -> None:
    with pytest.raises(FileNotFoundError):
        savant.load_savant(tmp_path, 2026, kind)


@pytest.mark.parametrize("kind", ["oaa", "frv"])
def test_collect_savant_load_missing_season_in_copied_dir_raises(
    tmp_path: Path, kind: SavantKind
) -> None:
    """다른 원본이 있어도 요청한 시즌 파일이 없으면 FileNotFoundError."""
    raw_dir = tmp_path / "raw"
    shutil.copytree(FIXTURE_RAW, raw_dir)
    (raw_dir / c.raw_savant_path(2024, kind)).unlink()
    with pytest.raises(FileNotFoundError):
        savant.load_savant(raw_dir, 2024, kind)


@pytest.mark.parametrize("kind", KINDS)
def test_collect_savant_load_does_not_modify_raw(tmp_path: Path, kind: SavantKind) -> None:
    """year 채우기 같은 손질은 읽을 때만 한다. 원본 파일은 받은 그대로 둔다 (D6)."""
    raw_dir = tmp_path / "raw"
    shutil.copytree(FIXTURE_RAW, raw_dir)
    before = {p.relative_to(raw_dir): p.read_bytes() for p in _files_under(raw_dir)}
    savant.load_savant(raw_dir, 2026, kind)
    after = {p.relative_to(raw_dir): p.read_bytes() for p in _files_under(raw_dir)}
    assert after == before


# ---------------------------------------------------------------------------
# 리뷰에서 더한 경계 사례
# ---------------------------------------------------------------------------


def test_collect_savant_check_message_lists_every_missing_column() -> None:
    body = _drop_column(_drop_column(_fixture_bytes(2026, "pitcher"), "xera"), "pa")
    with pytest.raises(ValueError) as info:
        savant.check_csv_body(body, "pitcher")
    assert "'pa'" in str(info.value) and "'xera'" in str(info.value)


@pytest.mark.parametrize("kind", KINDS)
def test_collect_savant_check_html_message_shows_header_preview(kind: SavantKind) -> None:
    """HTML 오류 페이지는 머리줄 앞부분이 메시지에 들어가 바로 알아볼 수 있다."""
    with pytest.raises(ValueError, match="DOCTYPE html"):
        savant.check_csv_body(HTML_ERROR_PAGE, kind)


@pytest.mark.parametrize("kind", KINDS)
def test_collect_savant_check_accepts_crlf_and_trailing_blank_lines(kind: SavantKind) -> None:
    body = _fixture_bytes(2026, kind)
    savant.check_csv_body(body.replace(b"\n", b"\r\n"), kind)
    savant.check_csv_body(body + b"\n\n", kind)


@pytest.mark.parametrize("kind", KINDS)
def test_collect_savant_check_rejects_truncated_last_row(kind: SavantKind) -> None:
    """마지막 행이 중간에 잘린 응답은 저장하지 않는다 (pandas는 빈 값으로 채워 읽는다)."""
    body = _fixture_bytes(2026, kind).rstrip(b"\n")
    truncated = body[: body.rfind(b",")]
    assert truncated.count(b"\n") == body.count(b"\n")  # 전제: 행 수는 같고 마지막 행만 짧다
    with pytest.raises(ValueError, match="잘린"):
        savant.check_csv_body(truncated, kind)


@pytest.mark.parametrize("kind", KINDS)
def test_collect_savant_check_rejects_unclosed_quote_at_end(kind: SavantKind) -> None:
    body = _fixture_bytes(2026, kind) + b'"Cut, Off'
    with pytest.raises(ValueError):
        savant.check_csv_body(body, kind)


@pytest.mark.parametrize(("kind", "collect", "url_of"), ALL_COLLECTORS)
def test_collect_savant_force_with_invalid_body_keeps_cache(
    tmp_path: Path,
    kind: SavantKind,
    collect: FieldingCollector,
    url_of: Callable[[int], str],
) -> None:
    """강제로 다시 받은 응답이 틀리면 전에 받은 원본과 매니페스트를 그대로 둔다."""
    good = _fixture_bytes(2026, kind)
    getter = _getter(good)
    collect(getter, tmp_path, 2026)
    rel = c.raw_savant_path(2026, kind)
    manifest_before = raw.load_manifest(tmp_path)
    getter.body = HTML_ERROR_PAGE
    with pytest.raises(ValueError):
        collect(getter, tmp_path, 2026, force=True)
    assert (tmp_path / rel).read_bytes() == good
    assert raw.load_manifest(tmp_path) == manifest_before
    again = collect(getter, tmp_path, 2026)  # 캐시는 여전히 쓸 수 있다
    assert not again.fetched
    assert getter.calls == [url_of(2026), url_of(2026)]


@pytest.mark.parametrize(
    ("value", "shown"), [("", "(빈 값)"), ("2025", "2025")], ids=["empty", "other_year"]
)
@pytest.mark.parametrize("kind", LEADERBOARD_KINDS)
def test_collect_savant_load_leaderboard_bad_year_raises_with_value(
    tmp_path: Path, kind: SavantKind, value: str, shown: str
) -> None:
    """리더보드 year가 비었거나 다른 해면 ValueError. 메시지에 파일과 읽힌 값이 들어간다."""
    body = _set_field(_fixture_bytes(2026, kind), "year", value, row=2)
    raw_dir = _raw_copy_with(tmp_path, 2026, kind, body)
    with pytest.raises(ValueError) as info:
        savant.load_savant(raw_dir, 2026, kind)
    message = str(info.value)
    assert c.raw_savant_path(2026, kind).as_posix() in message
    assert f"'{shown}'" in message


def test_collect_savant_load_oaa_keeps_filled_year_equal_to_season(tmp_path: Path) -> None:
    """OAA year가 채워져 와도 요청 시즌과 같으면 그대로 읽는다."""
    body = _set_field(_fixture_bytes(2026, "oaa"), "year", "2026")
    raw_dir = _raw_copy_with(tmp_path, 2026, "oaa", body)
    df = savant.load_savant(raw_dir, 2026, "oaa")
    assert df["year"].tolist() == [2026] * len(df)


@pytest.mark.parametrize("kind", ["oaa", "frv"])
def test_collect_savant_load_fielding_filled_other_year_raises(
    tmp_path: Path, kind: SavantKind
) -> None:
    """수비 원본에 다른 해 year가 들어 있으면 요청 시즌으로 덮어쓰지 않고 멈춘다."""
    source = _fixture_bytes(2026, kind)
    body = (
        _set_field(source, "year", "2025")
        if kind == "oaa"
        else _add_column(source, "year", "2025")  # FRV는 원래 year 열이 없다 (E18)
    )
    raw_dir = _raw_copy_with(tmp_path, 2026, kind, body)
    with pytest.raises(ValueError, match="2025"):
        savant.load_savant(raw_dir, 2026, kind)


def test_collect_savant_load_frv_with_empty_year_column_fills_in_place(tmp_path: Path) -> None:
    """FRV에 빈 year 열이 생겨도 열을 겹치지 않고 그 자리를 채운다."""
    body = _add_column(_fixture_bytes(2026, "frv"), "year", "")
    raw_dir = _raw_copy_with(tmp_path, 2026, "frv", body)
    df = savant.load_savant(raw_dir, 2026, "frv")
    assert list(df.columns).count("year") == 1
    assert df["year"].tolist() == [2026] * len(df)


def test_collect_savant_load_frv_with_both_id_columns_raises(tmp_path: Path) -> None:
    body = _add_column(_fixture_bytes(2026, "frv"), "player_id", "1")
    raw_dir = _raw_copy_with(tmp_path, 2026, "frv", body)
    with pytest.raises(ValueError, match="player_id"):
        savant.load_savant(raw_dir, 2026, "frv")


@pytest.mark.parametrize("season", c.FIELDING_YEARS)
def test_collect_savant_load_frv_column_order(season: int) -> None:
    """`id`는 제자리에서 `player_id`가 되고, year는 맨 끝에 붙는다. 다른 열 순서는 원본 그대로."""
    source = _raw_frame(season, "frv")
    df = savant.load_savant(FIXTURE_RAW, season, "frv")
    expected = ["player_id" if col == "id" else col for col in source.columns] + ["year"]
    assert list(df.columns) == expected


@pytest.mark.parametrize(
    ("kind", "column"),
    [
        ("batter", "xwoba"),
        ("pitcher", "xera"),
        ("oaa", "outs_above_average"),
        ("oaa", "year"),
        ("frv", "id"),
        ("frv", "total_runs"),
    ],
)
def test_collect_savant_load_missing_required_column_raises(
    tmp_path: Path, kind: SavantKind, column: str
) -> None:
    body = _drop_column(_fixture_bytes(2026, kind), column)
    raw_dir = _raw_copy_with(tmp_path, 2026, kind, body)
    with pytest.raises(ValueError, match=re.escape(column)):
        savant.load_savant(raw_dir, 2026, kind)


@pytest.mark.parametrize("value", ["", "abc", "646240.5"], ids=["empty", "text", "fraction"])
@pytest.mark.parametrize("kind", KINDS)
def test_collect_savant_load_bad_player_id_raises(
    tmp_path: Path, kind: SavantKind, value: str
) -> None:
    """선수 ID가 정수로 읽히지 않으면 pandas 오류 대신 파일 이름이 든 ValueError로 멈춘다."""
    body = _set_field(_fixture_bytes(2026, kind), _raw_id_column(kind), value, row=2)
    raw_dir = _raw_copy_with(tmp_path, 2026, kind, body)
    with pytest.raises(ValueError, match="선수 ID") as info:
        savant.load_savant(raw_dir, 2026, kind)
    assert c.raw_savant_path(2026, kind).as_posix() in str(info.value)


@pytest.mark.parametrize(("season", "kind"), [(2026, k) for k in KINDS])
def test_collect_savant_load_is_repeatable(season: int, kind: SavantKind) -> None:
    """같은 원본을 두 번 읽으면 같은 표가 나온다."""
    pd.testing.assert_frame_equal(
        savant.load_savant(FIXTURE_RAW, season, kind),
        savant.load_savant(FIXTURE_RAW, season, kind),
    )
