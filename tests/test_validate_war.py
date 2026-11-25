"""세이버메트릭스(WAR) 수집 완전성 검사(validate_war.py, SPEC v0.5 F2 ⑦·E26) 테스트.

기준 상태는 `tests/fixtures/raw/statsapi/{season}_{role}_saber.json`(실제 응답에서 시즌 기록 픽스처의
선수만 잘라 낸 것, SOURCE.md)이다. 잘라 냈으므로 리그 WAR 합 항등식(570 · 430)은 성립하지 않는다 —
기준 통과는 `league_war`에 픽스처 합을 넘겨서 보고, 다르게 넘겨 실패도 본다.
원본을 바꾸는 테스트는 tmp_path에 복사본을 만들어 고친다(픽스처 자체는 건드리지 않는다).
검사는 실패해도 예외가 아니라 실패한 검사로 담겨야 한다.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import math
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from mlbproj.contracts import (
    FULL_SEASON_GAMES,
    LEAGUE_WAR,
    LEAGUE_WAR_TOLERANCE,
    ROLES,
    SABER_SPLIT_FIELDS,
    SABER_STAT_FIELDS,
    SEASONS,
    SHORT_SEASON,
    WAR_CHECK_NAMES,
    Role,
    raw_saber_path,
    raw_statsapi_path,
    season_games,
)
from mlbproj.validate import (
    CHECK_NAMES,
    CheckResult,
    SeasonReport,
    ValidationError,
    write_report,
)
from mlbproj.validate_war import (
    WarSeasonReport,
    assert_war_validated,
    read_war_report,
    validate_war_season,
    validate_war_seasons,
    write_war_report,
)

FIXTURE_RAW = Path(__file__).parent / "fixtures" / "raw"

COMPLETE, WAR_PRESENT, COVERS_ROLES, LEAGUE_SUM = WAR_CHECK_NAMES

# 원본을 고치는 테스트의 기본 시즌 (오타니가 타격·투구 모두 있고, 등판한 야수 571771의 투구 행이 있다)
S = 2025

DEVERS = 646240
REALMUTO = 592663
OHTANI = 660271
NOLA = 605400
IGLESIAS = 628452
VENDITTE = 519381
HERNANDEZ = 571771  # 등판한 야수 (투구 포지션 3B)

# 세이버메트릭스에만 있는 가짜 선수 ID (시즌 기록 픽스처에 없다)
EXTRA_ID = 999001

# 여러 시즌을 한 번에 검사할 때 리그 합 검사를 사실상 끄는 허용치 (시즌마다 픽스처 합이 달라서)
HUGE_TOLERANCE = 1e9


# ---------------------------------------------------------------------------
# 공통 도우미
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_raw(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """읽기만 하는 테스트가 함께 쓰는 픽스처 복사본."""
    dst = tmp_path_factory.mktemp("shared") / "raw"
    shutil.copytree(FIXTURE_RAW, dst)
    return dst


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """테스트 하나가 고쳐 쓰는 픽스처 복사본."""
    dst = tmp_path / "raw"
    shutil.copytree(FIXTURE_RAW, dst)
    return dst


def _load(path: Path) -> Any:
    return json.loads(path.read_bytes().decode("utf-8"))


def _write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # NaN·Infinity도 그대로 쓴다 (json 모듈 기본값) — 유한하지 않은 war 검사용
    path.write_bytes((json.dumps(doc, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


def _saber(raw_dir: Path, season: int, role: Role) -> dict[str, Any]:
    doc: dict[str, Any] = _load(raw_dir / raw_saber_path(season, role))
    return doc


def _save_saber(raw_dir: Path, season: int, role: Role, doc: dict[str, Any]) -> None:
    _write_json(raw_dir / raw_saber_path(season, role), doc)


def _splits(doc: dict[str, Any]) -> list[dict[str, Any]]:
    splits: list[dict[str, Any]] = doc["stats"][0]["splits"]
    return splits


def _split_of(doc: dict[str, Any], player_id: int) -> dict[str, Any]:
    hits = [s for s in _splits(doc) if s["player"]["id"] == player_id]
    assert len(hits) == 1, f"{player_id} 행이 정확히 하나여야 한다"
    return hits[0]


def _remove_saber_row(raw_dir: Path, season: int, role: Role, player_id: int) -> None:
    """세이버메트릭스에서 한 선수의 행을 빼고 totalSplits를 맞춘다 (형식은 그대로 맞다)."""
    doc = _saber(raw_dir, season, role)
    block = doc["stats"][0]
    before = len(block["splits"])
    block["splits"] = [s for s in block["splits"] if s["player"]["id"] != player_id]
    assert len(block["splits"]) == before - 1, (
        f"{season} {role} 세이버메트릭스에 {player_id}가 없다"
    )
    block["totalSplits"] = len(block["splits"])
    _save_saber(raw_dir, season, role, doc)


def _add_saber_row(
    raw_dir: Path, season: int, role: Role, player_id: int, war: float, position: str = "P"
) -> None:
    """기존 행을 복사해 세이버메트릭스에만 있는 행을 더한다 (투수 타격·야수 등판 같은 행, E26)."""
    doc = _saber(raw_dir, season, role)
    block = doc["stats"][0]
    row = copy.deepcopy(block["splits"][0])
    row["player"]["id"] = player_id
    row["player"]["fullName"] = f"Extra {player_id}"
    row["position"]["abbreviation"] = position
    row["stat"]["war"] = war
    block["splits"].append(row)
    block["totalSplits"] = len(block["splits"])
    _save_saber(raw_dir, season, role, doc)


def _edit_saber(
    raw_dir: Path, season: int, role: Role, edit: Callable[[dict[str, Any]], object]
) -> None:
    """세이버메트릭스 문서 전체를 edit(doc)로 제자리에서 고친다."""
    doc = _saber(raw_dir, season, role)
    edit(doc)
    _save_saber(raw_dir, season, role, doc)


def _war_sum(raw_dir: Path, season: int, role: Role) -> float:
    """원본 전체 행의 war 합 (역할 규칙으로 거르기 전)."""
    return math.fsum(float(s["stat"]["war"]) for s in _splits(_saber(raw_dir, season, role)))


def _league_for(season: int, totals: dict[Role, float]) -> dict[Role, float]:
    """역할별 "원본 war 합 기준값"이 totals가 되도록 league_war(162경기 기준)를 만든다."""
    scale = FULL_SEASON_GAMES / season_games(season)
    return {role: totals[role] * scale for role in ROLES}


def _matched(
    raw_dir: Path, season: int, offsets: dict[Role, float] | None = None
) -> dict[Role, float]:
    """픽스처 합(+ offsets)에 맞춘 league_war."""
    shift = offsets or {}
    return _league_for(
        season, {role: _war_sum(raw_dir, season, role) + shift.get(role, 0.0) for role in ROLES}
    )


def _failed(report: WarSeasonReport) -> set[str]:
    return {c.name for c in report.checks if not c.passed}


def _check(report: WarSeasonReport, name: str) -> CheckResult:
    matches = [c for c in report.checks if c.name == name]
    assert len(matches) == 1, f"{name} 결과가 정확히 하나여야 한다: {report.checks}"
    return matches[0]


def _assert_shape(report: WarSeasonReport, season: int) -> None:
    """보고서가 시즌과 검사 4개(WAR_CHECK_NAMES 순서)를 갖췄는지 본다."""
    assert isinstance(report, WarSeasonReport)
    assert report.season == season
    assert tuple(c.name for c in report.checks) == WAR_CHECK_NAMES
    for c in report.checks:
        assert isinstance(c, CheckResult)
        assert isinstance(c.passed, bool)
        assert isinstance(c.detail, str)
        assert c.detail.strip(), f"{c.name} detail이 비어 있다"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _season_rels(season: int) -> list[Path]:
    """검사에 쓰는 원본 네 개: 세이버메트릭스 두 파일 + 시즌 기록 두 파일."""
    return [
        *(raw_saber_path(season, role) for role in ROLES),
        *(raw_statsapi_path(season, role) for role in ROLES),
    ]


def _snapshot(raw_dir: Path) -> dict[str, bytes]:
    return {
        p.relative_to(raw_dir).as_posix(): p.read_bytes()
        for p in sorted(raw_dir.rglob("*"))
        if p.is_file()
    }


def _passing(raw_dir: Path, season: int) -> WarSeasonReport:
    return validate_war_season(raw_dir, season, league_war=_matched(raw_dir, season))


def _all_ok(season: int) -> WarSeasonReport:
    return WarSeasonReport(season, tuple(CheckResult(n, True, "ok") for n in WAR_CHECK_NAMES), {})


# ---------------------------------------------------------------------------
# WarSeasonReport.passed
# ---------------------------------------------------------------------------


def test_validate_war_report_passed_property() -> None:
    ok = _all_ok(S)
    assert ok.passed is True
    one_failed = dataclasses.replace(
        ok, checks=(*ok.checks[:2], CheckResult(COVERS_ROLES, False, "x"), ok.checks[3])
    )
    assert one_failed.passed is False
    # 검사가 빠지거나 순서가 다르면 통과가 아니다
    assert dataclasses.replace(ok, checks=ok.checks[:-1]).passed is False
    assert dataclasses.replace(ok, checks=tuple(reversed(ok.checks))).passed is False
    assert dataclasses.replace(ok, checks=()).passed is False


# ---------------------------------------------------------------------------
# 기준 상태: 픽스처 합을 기준값으로 주면 모든 시즌이 통과한다
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("season", SEASONS)
def test_validate_war_baseline_every_season_passes(shared_raw: Path, season: int) -> None:
    report = _passing(shared_raw, season)
    _assert_shape(report, season)
    assert _failed(report) == set()
    assert report.passed


@pytest.mark.parametrize("season", SEASONS)
def test_validate_war_baseline_inputs_are_four_raw_files_with_sha256(
    shared_raw: Path, season: int
) -> None:
    report = _passing(shared_raw, season)
    assert report.inputs == {
        rel.as_posix(): _sha256(shared_raw / rel) for rel in _season_rels(season)
    }


@pytest.mark.parametrize("season", SEASONS)
def test_validate_war_default_league_war_fails_only_league_sum_on_fixture(
    shared_raw: Path, season: int
) -> None:
    # 잘라 낸 픽스처의 합은 570 · 430과 거리가 멀다 — 리그 합 검사만 실패하고 나머지는 통과
    report = validate_war_season(shared_raw, season)
    _assert_shape(report, season)
    assert _failed(report) == {LEAGUE_SUM}
    assert not report.passed


def test_validate_war_season_does_not_modify_raw_dir(raw_dir: Path) -> None:
    before = _snapshot(raw_dir)
    validate_war_season(raw_dir, S)
    _passing(raw_dir, S)
    assert _snapshot(raw_dir) == before


def test_validate_war_season_default_arguments_are_contract_values() -> None:
    params = inspect.signature(validate_war_season).parameters
    assert params["league_war"].default == LEAGUE_WAR
    assert params["tolerance"].default == LEAGUE_WAR_TOLERANCE
    params_many = inspect.signature(validate_war_seasons).parameters
    assert params_many["league_war"].default == LEAGUE_WAR
    assert params_many["tolerance"].default == LEAGUE_WAR_TOLERANCE


# ---------------------------------------------------------------------------
# saber_league_sum — war 합 = league_war × 경기 수 / 162 ± 허용치
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offset", "ok"),
    [(0.0, True), (0.9, True), (-0.9, True), (1.1, False), (-1.1, False), (25.0, False)],
)
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("season", [S, SHORT_SEASON])
def test_validate_war_league_sum_tolerance_band(
    shared_raw: Path, season: int, role: Role, offset: float, ok: bool
) -> None:
    # offset은 경기 수 비율을 곱한 뒤의 기준값과 원본 합의 차이다 (2020은 60/162로 줄여서 비교)
    league = _matched(shared_raw, season, {role: offset})
    report = validate_war_season(shared_raw, season, league_war=league)
    _assert_shape(report, season)
    assert _check(report, LEAGUE_SUM).passed is ok
    assert _failed(report) - {LEAGUE_SUM} == set()  # 다른 검사는 기준값과 상관없다
    assert report.passed is ok


@pytest.mark.parametrize(("tolerance", "ok"), [(2.0, True), (1.0, False), (1.4, False)])
def test_validate_war_league_sum_custom_tolerance(
    shared_raw: Path, tolerance: float, ok: bool
) -> None:
    league = _matched(shared_raw, S, {"bat": 1.5, "pit": -1.5})
    report = validate_war_season(shared_raw, S, league_war=league, tolerance=tolerance)
    assert _check(report, LEAGUE_SUM).passed is ok


@pytest.mark.parametrize("role", ROLES)
def test_validate_war_league_sum_one_role_off_fails(shared_raw: Path, role: Role) -> None:
    # 한 역할만 어긋나도 실패
    league = _matched(shared_raw, S, {role: 5.0})
    report = validate_war_season(shared_raw, S, league_war=league)
    assert _failed(report) == {LEAGUE_SUM}


def test_validate_war_league_sum_uses_season_games_for_short_season(shared_raw: Path) -> None:
    # 2020(60경기)에서 경기 수 비율을 빼먹고 162경기 기준값과 그대로 비교하면 통과하는 기준값 → 실패여야 한다
    unscaled = {role: _war_sum(shared_raw, SHORT_SEASON, role) for role in ROLES}
    report = validate_war_season(shared_raw, SHORT_SEASON, league_war=unscaled)
    assert _check(report, LEAGUE_SUM).passed is False
    scaled = _league_for(SHORT_SEASON, unscaled)
    assert validate_war_season(shared_raw, SHORT_SEASON, league_war=scaled).passed


def _fill_to(raw_dir: Path, season: int, totals: dict[Role, float]) -> None:
    """세이버메트릭스에만 있는 행 하나씩을 더해 역할별 원본 war 합을 totals로 만든다."""
    for offset, role in enumerate(ROLES):
        need = totals[role] - _war_sum(raw_dir, season, role)
        _add_saber_row(raw_dir, season, role, EXTRA_ID + offset, need)


@pytest.mark.parametrize("season", [S, SHORT_SEASON])
def test_validate_war_league_sum_default_identity_passes(raw_dir: Path, season: int) -> None:
    # 실데이터처럼 원본 합이 570 · 430 × 경기 수/162이면 기본 기준값으로 통과한다 (E26)
    ratio = season_games(season) / FULL_SEASON_GAMES
    _fill_to(raw_dir, season, {role: LEAGUE_WAR[role] * ratio for role in ROLES})
    report = validate_war_season(raw_dir, season)
    _assert_shape(report, season)
    assert _failed(report) == set()
    assert report.passed


def test_validate_war_league_sum_detail_reports_totals(raw_dir: Path) -> None:
    # 통과 detail에는 핵심 수치(역할별 합 · 기준 · 허용치)를, 실패 detail에는 어긋난 역할과 합을 적는다
    _fill_to(raw_dir, S, dict(LEAGUE_WAR))
    passed = _check(validate_war_season(raw_dir, S), LEAGUE_SUM)
    assert passed.passed
    assert passed.detail == "타격 570.0 · 투구 430.0 (기준 570.0 · 430.0, ±1.0)"
    failed = _check(
        validate_war_season(raw_dir, S, league_war={"bat": 570.0, "pit": 420.0}), LEAGUE_SUM
    )
    assert not failed.passed
    assert "투구" in failed.detail and "430.00" in failed.detail and "420.00" in failed.detail
    assert "타격" not in failed.detail  # 맞는 역할은 실패 사유에 적지 않는다


def test_validate_war_league_sum_short_season_full_totals_fail(raw_dir: Path) -> None:
    # 2020의 원본 합이 162경기 값(570 · 430)이면 기준(211.1 · 159.3)에서 벗어난다
    _fill_to(raw_dir, SHORT_SEASON, dict(LEAGUE_WAR))
    report = validate_war_season(raw_dir, SHORT_SEASON)
    assert _failed(report) == {LEAGUE_SUM}


@pytest.mark.parametrize(("delta", "ok"), [(0.95, True), (-0.95, True), (1.05, False)])
def test_validate_war_league_sum_default_tolerance_around_identity(
    raw_dir: Path, delta: float, ok: bool
) -> None:
    _fill_to(raw_dir, S, {"bat": LEAGUE_WAR["bat"] + delta, "pit": LEAGUE_WAR["pit"]})
    report = validate_war_season(raw_dir, S)
    assert _check(report, LEAGUE_SUM).passed is ok


def test_validate_war_league_sum_counts_saber_only_rows(raw_dir: Path) -> None:
    # 합은 역할 규칙으로 거르기 전 원본 전체 행이다 — 투수 타격 같은 세이버메트릭스 전용 행도 더한다
    original = _matched(raw_dir, S)
    _add_saber_row(raw_dir, S, "bat", EXTRA_ID, 3.0, position="P")
    report = validate_war_season(raw_dir, S, league_war=original)
    assert _failed(report) == {LEAGUE_SUM}
    report = validate_war_season(raw_dir, S, league_war=_matched(raw_dir, S))
    assert report.passed


# ---------------------------------------------------------------------------
# saber_complete — 두 원본 · JSON 형식 · totalSplits = 행 수 · 필수 필드
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
def test_validate_war_complete_missing_saber_file_fails_without_exception(
    raw_dir: Path, role: Role
) -> None:
    league = _matched(raw_dir, S)
    (raw_dir / raw_saber_path(S, role)).unlink()
    report = validate_war_season(raw_dir, S, league_war=league)
    _assert_shape(report, S)
    assert COMPLETE in _failed(report)
    assert not report.passed


def test_validate_war_complete_both_saber_files_missing_fails(raw_dir: Path) -> None:
    for role in ROLES:
        (raw_dir / raw_saber_path(S, role)).unlink()
    report = validate_war_season(raw_dir, S)
    _assert_shape(report, S)
    assert COMPLETE in _failed(report)
    assert not report.passed


def test_validate_war_season_outside_raw_fails_without_exception(shared_raw: Path) -> None:
    # 원본이 하나도 없는 시즌도 예외가 아니라 실패한 보고서
    report = validate_war_season(shared_raw, 2018)
    _assert_shape(report, 2018)
    assert COMPLETE in _failed(report)
    assert not report.passed


def test_validate_war_season_without_any_raw_file_fails(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    report = validate_war_season(empty, S)
    _assert_shape(report, S)
    assert not report.passed


_BAD_BODIES: list[Any] = [
    pytest.param(b"", id="empty"),
    pytest.param(b"<html><body>busy</body></html>", id="html"),
    pytest.param(b"\xff\xfe\x00\x01", id="not-utf8"),
    pytest.param(b"[{", id="cut-off"),
    pytest.param(b"null", id="null"),
    pytest.param(b"[]", id="top-level-list"),
    pytest.param(b'{"stats": []}', id="stats-empty"),
    pytest.param(b'{"stats": {}}', id="stats-object"),
    pytest.param(b'{"stats": [{"totalSplits": 0}]}', id="no-splits"),
    pytest.param(b'{"stats": [{"splits": {}, "totalSplits": 0}]}', id="splits-object"),
    pytest.param(b'{"stats": [1]}', id="stats-item-not-object"),
]


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("body", _BAD_BODIES)
def test_validate_war_complete_malformed_saber_fails_without_exception(
    raw_dir: Path, body: bytes, role: Role
) -> None:
    (raw_dir / raw_saber_path(S, role)).write_bytes(body)
    report = validate_war_season(raw_dir, S, tolerance=HUGE_TOLERANCE)
    _assert_shape(report, S)
    assert COMPLETE in _failed(report)
    assert not report.passed


@pytest.mark.parametrize("delta", [1, -1])
@pytest.mark.parametrize("role", ROLES)
def test_validate_war_complete_total_splits_mismatch_fails(
    raw_dir: Path, role: Role, delta: int
) -> None:
    # 잘린 응답(totalSplits ≠ 행 수)
    league = _matched(raw_dir, S)

    def edit(doc: dict[str, Any]) -> None:
        doc["stats"][0]["totalSplits"] += delta

    _edit_saber(raw_dir, S, role, edit)
    report = validate_war_season(raw_dir, S, league_war=league)
    assert COMPLETE in _failed(report)
    assert not report.passed


@pytest.mark.parametrize("role", ROLES)
def test_validate_war_complete_truncated_rows_fail(raw_dir: Path, role: Role) -> None:
    # 행이 잘렸는데 totalSplits는 원래 값 — 역할 행 누락이 아니라 형식(잘림) 실패로 잡힌다
    def edit(doc: dict[str, Any]) -> None:
        doc["stats"][0]["splits"] = doc["stats"][0]["splits"][:-1]

    _edit_saber(raw_dir, S, role, edit)
    report = validate_war_season(raw_dir, S, tolerance=HUGE_TOLERANCE)
    assert COMPLETE in _failed(report)
    assert not report.passed


@pytest.mark.parametrize("total", [None, "5", True, 5.0], ids=["null", "string", "bool", "float"])
def test_validate_war_complete_non_integer_total_splits_fails(raw_dir: Path, total: Any) -> None:
    def edit(doc: dict[str, Any]) -> None:
        if total is None:
            del doc["stats"][0]["totalSplits"]
        else:
            doc["stats"][0]["totalSplits"] = total

    assert len(_splits(_saber(raw_dir, S, "bat"))) == 5  # "5"·5.0이 행 수와 같아 보이게 고른 값
    _edit_saber(raw_dir, S, "bat", edit)
    report = validate_war_season(raw_dir, S, tolerance=HUGE_TOLERANCE)
    assert COMPLETE in _failed(report)


_SPLIT_FIELD_CASES = [(role, field) for role in ROLES for field in SABER_SPLIT_FIELDS]


@pytest.mark.parametrize(("role", "field"), _SPLIT_FIELD_CASES)
def test_validate_war_complete_missing_split_field_fails(
    raw_dir: Path, role: Role, field: str
) -> None:
    def edit(doc: dict[str, Any]) -> None:
        del _splits(doc)[1][field]

    _edit_saber(raw_dir, S, role, edit)
    report = validate_war_season(raw_dir, S, tolerance=HUGE_TOLERANCE)
    _assert_shape(report, S)
    assert COMPLETE in _failed(report)
    assert not report.passed


_STAT_FIELD_CASES = [(role, field) for role in ROLES for field in SABER_STAT_FIELDS[role]]


@pytest.mark.parametrize(("role", "field"), _STAT_FIELD_CASES)
def test_validate_war_complete_missing_stat_field_fails(
    raw_dir: Path, role: Role, field: str
) -> None:
    def edit(doc: dict[str, Any]) -> None:
        del _split_of(doc, OHTANI)["stat"][field]

    _edit_saber(raw_dir, S, role, edit)
    report = validate_war_season(raw_dir, S, tolerance=HUGE_TOLERANCE)
    _assert_shape(report, S)
    assert COMPLETE in _failed(report)
    assert not report.passed


@pytest.mark.parametrize(
    ("role", "level", "field"),
    [
        ("bat", "split", "team"),
        ("bat", "split", "rank"),
        ("bat", "stat", "woba"),
        ("bat", "stat", "rar"),
        ("pit", "split", "league"),
        ("pit", "stat", "fip"),
        ("pit", "stat", "xfip"),
    ],
)
def test_validate_war_complete_unrequired_field_removed_passes(
    raw_dir: Path, role: Role, level: str, field: str
) -> None:
    # 필수가 아닌 필드는 빠져도 통과한다 (SABER_SPLIT_FIELDS · SABER_STAT_FIELDS만 필수)
    assert field not in SABER_SPLIT_FIELDS and field not in SABER_STAT_FIELDS[role]

    def edit(doc: dict[str, Any]) -> None:
        for split in _splits(doc):
            target = split if level == "split" else split["stat"]
            target.pop(field, None)

    _edit_saber(raw_dir, S, role, edit)
    report = _passing(raw_dir, S)
    assert _failed(report) == set()


@pytest.mark.parametrize(("role", "player_id"), [("bat", DEVERS), ("pit", HERNANDEZ)])
def test_validate_war_complete_duplicate_player_fails(
    raw_dir: Path, role: Role, player_id: int
) -> None:
    # 같은 선수가 한 원본에 두 번 있으면 정제(clean.war)가 멈춘다 — 검증에서 먼저 막는다.
    # 역할 행이 아닌 행(등판한 야수)의 중복도 원본 전체 검사라 잡는다
    def duplicate(doc: dict[str, Any]) -> None:
        block = doc["stats"][0]
        block["splits"].append(copy.deepcopy(_split_of(doc, player_id)))
        block["totalSplits"] = len(block["splits"])

    _edit_saber(raw_dir, S, role, duplicate)
    report = _passing(raw_dir, S)  # 리그 합은 바뀐 원본에 맞춘다
    _assert_shape(report, S)
    assert _failed(report) == {COMPLETE}
    assert str(player_id) in _check(report, COMPLETE).detail


@pytest.mark.parametrize("value", ["2024", "2026", 2025, ""], ids=["prev", "next", "int", "empty"])
@pytest.mark.parametrize("role", ROLES)
def test_validate_war_complete_wrong_season_value_fails(
    raw_dir: Path, role: Role, value: Any
) -> None:
    # 리그 합은 시즌마다 같아(570 · 430) 다른 시즌의 응답이 이 자리에 저장돼도 합으로는 못 잡는다
    def edit(doc: dict[str, Any]) -> None:
        _split_of(doc, OHTANI)["season"] = value

    _edit_saber(raw_dir, S, role, edit)
    report = _passing(raw_dir, S)
    assert _failed(report) == {COMPLETE}
    assert str(OHTANI) in _check(report, COMPLETE).detail


def test_validate_war_complete_whole_file_from_other_season_fails(raw_dir: Path) -> None:
    # 2024 투구 응답이 2025 자리에 저장된 경우 — 2025 시즌 기록의 역할 행과 겹쳐도 시즌 값으로 잡는다
    shutil.copyfile(raw_dir / raw_saber_path(2024, "pit"), raw_dir / raw_saber_path(S, "pit"))
    report = validate_war_season(raw_dir, S, tolerance=HUGE_TOLERANCE)
    assert COMPLETE in _failed(report)
    assert "2024" in _check(report, COMPLETE).detail


# ---------------------------------------------------------------------------
# saber_war_present — 모든 행의 stat.war가 유한한 숫자 (bool 아님)
# ---------------------------------------------------------------------------


_BAD_WAR: list[Any] = [
    pytest.param(None, id="null"),
    pytest.param("1.5", id="string"),
    pytest.param("", id="empty-string"),
    pytest.param(True, id="true"),
    pytest.param(False, id="false"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="-inf"),
    pytest.param([1.0], id="list"),
    pytest.param({"value": 1.0}, id="object"),
]


@pytest.mark.parametrize(
    ("role", "player_id"), [("bat", DEVERS), ("bat", OHTANI), ("pit", NOLA), ("pit", HERNANDEZ)]
)
@pytest.mark.parametrize("value", _BAD_WAR)
def test_validate_war_present_bad_war_fails_and_names_player(
    raw_dir: Path, value: Any, role: Role, player_id: int
) -> None:
    # 역할 행이 아닌 행(등판한 야수 571771의 투구 행)도 원본의 모든 행이라 검사 대상이다
    def edit(doc: dict[str, Any]) -> None:
        _split_of(doc, player_id)["stat"]["war"] = value

    _edit_saber(raw_dir, S, role, edit)
    report = validate_war_season(raw_dir, S, tolerance=HUGE_TOLERANCE)
    _assert_shape(report, S)
    check = _check(report, WAR_PRESENT)
    assert check.passed is False
    assert str(player_id) in check.detail
    assert not report.passed


@pytest.mark.parametrize("role", ROLES)
def test_validate_war_present_missing_war_key_fails(raw_dir: Path, role: Role) -> None:
    def edit(doc: dict[str, Any]) -> None:
        del _split_of(doc, OHTANI)["stat"]["war"]

    _edit_saber(raw_dir, S, role, edit)
    report = validate_war_season(raw_dir, S, tolerance=HUGE_TOLERANCE)
    _assert_shape(report, S)
    assert {COMPLETE, WAR_PRESENT} <= _failed(report)


def test_validate_war_present_integer_war_passes(raw_dir: Path) -> None:
    # 정수 0도 유한한 숫자다 (Venditte 2019 타격 행은 war 0.0)
    def edit(doc: dict[str, Any]) -> None:
        split = _split_of(doc, VENDITTE)
        assert split["stat"]["war"] == 0.0
        split["stat"]["war"] = 0

    _edit_saber(raw_dir, 2019, "bat", edit)
    report = _passing(raw_dir, 2019)
    assert _check(report, WAR_PRESENT).passed
    assert report.passed


def test_validate_war_present_negative_war_passes(shared_raw: Path) -> None:
    # 음수 WAR(대체 선수보다 못함)은 정상 값이다 — 픽스처 2025 투구에 −0.14298이 있다
    assert any(s["stat"]["war"] < 0 for s in _splits(_saber(shared_raw, S, "pit")))
    assert _check(_passing(shared_raw, S), WAR_PRESENT).passed


# ---------------------------------------------------------------------------
# saber_covers_roles — 시즌 기록의 역할 행(D8)이 모두 세이버메트릭스에 있다
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("season", "role", "player_id"),
    [
        pytest.param(S, "bat", DEVERS, id="batter"),
        pytest.param(S, "bat", REALMUTO, id="catcher"),
        pytest.param(S, "bat", OHTANI, id="ohtani-dh"),
        pytest.param(S, "pit", NOLA, id="pitcher"),
        pytest.param(S, "pit", OHTANI, id="ohtani-twp"),
        pytest.param(SHORT_SEASON, "pit", VENDITTE, id="2020-pitcher"),
        pytest.param(2026, "bat", 805808, id="2026-rookie"),
    ],
)
def test_validate_war_covers_roles_missing_role_row_fails(
    raw_dir: Path, season: int, role: Role, player_id: int
) -> None:
    _remove_saber_row(raw_dir, season, role, player_id)
    report = _passing(raw_dir, season)  # 리그 합은 바뀐 원본에 맞춘다
    _assert_shape(report, season)
    assert _failed(report) == {COVERS_ROLES}  # 형식은 맞다 (totalSplits도 맞췄다)
    assert str(player_id) in _check(report, COVERS_ROLES).detail
    assert not report.passed


@pytest.mark.parametrize(
    ("season", "role", "player_id"),
    [
        pytest.param(2019, "bat", NOLA, id="pitcher-batting"),
        pytest.param(2019, "bat", VENDITTE, id="pitcher-batting-zero-pa"),
        pytest.param(2026, "bat", IGLESIAS, id="pitcher-batting-2026"),
        pytest.param(S, "pit", HERNANDEZ, id="position-player-pitching"),
        pytest.param(2024, "pit", HERNANDEZ, id="position-player-pitching-2024"),
    ],
)
def test_validate_war_covers_roles_non_role_row_missing_passes(
    raw_dir: Path, season: int, role: Role, player_id: int
) -> None:
    # 투수의 타격(포지션 P)·등판한 야수는 역할 행이 아니다 — 세이버메트릭스에 없어도 된다
    _remove_saber_row(raw_dir, season, role, player_id)
    report = _passing(raw_dir, season)
    assert _failed(report) == set()


def test_validate_war_covers_roles_zero_pa_batter_is_not_role_row(raw_dir: Path) -> None:
    # D8 타자 역할은 포지션이 P가 아니고 **타석 1 이상**이다(clean.batting.is_batter_split과 같다).
    # 0타석 야수(대주자 등)는 정제 타자 표에 없으므로 세이버메트릭스에 없어도 된다
    stats_path = raw_dir / raw_statsapi_path(S, "bat")
    stats = _load(stats_path)
    _split_of(stats, DEVERS)["stat"]["plateAppearances"] = 0
    _write_json(stats_path, stats)
    _remove_saber_row(raw_dir, S, "bat", DEVERS)
    assert _failed(_passing(raw_dir, S)) == set()
    # 1타석이면 역할 행 → 세이버메트릭스에 없으면 실패
    _split_of(stats, DEVERS)["stat"]["plateAppearances"] = 1
    _write_json(stats_path, stats)
    report = _passing(raw_dir, S)
    assert _failed(report) == {COVERS_ROLES}
    assert str(DEVERS) in _check(report, COVERS_ROLES).detail


@pytest.mark.parametrize("pa", [None, "727", True], ids=["null", "string", "bool"])
def test_validate_war_covers_roles_unreadable_pa_fails(raw_dir: Path, pa: Any) -> None:
    # 타석을 읽지 못하면 역할을 정할 수 없다 — 조용히 역할 밖으로 두지 않고 실패
    stats_path = raw_dir / raw_statsapi_path(S, "bat")
    stats = _load(stats_path)
    _split_of(stats, OHTANI)["stat"]["plateAppearances"] = pa
    _write_json(stats_path, stats)
    report = _passing(raw_dir, S)
    assert _failed(report) == {COVERS_ROLES}
    assert str(OHTANI) in _check(report, COVERS_ROLES).detail


@pytest.mark.parametrize("role", ROLES)
def test_validate_war_covers_roles_saber_only_row_passes(raw_dir: Path, role: Role) -> None:
    # 세이버메트릭스에만 있는 ID는 실패가 아니다 (E26: 투수 타격·야수 등판)
    _add_saber_row(raw_dir, S, role, EXTRA_ID, 0.5)
    report = _passing(raw_dir, S)
    assert _failed(report) == set()


def test_validate_war_covers_roles_follows_season_record_position(raw_dir: Path) -> None:
    # 역할은 세이버메트릭스의 포지션이 아니라 시즌 기록 원본의 포지션으로 정한다 (오타니 타격: 시즌 기록 DH, 세이버 TWP)
    assert _split_of(_saber(raw_dir, S, "bat"), OHTANI)["position"]["abbreviation"] == "TWP"
    stats_path = raw_dir / raw_statsapi_path(S, "bat")
    stats = _load(stats_path)
    assert _split_of(stats, OHTANI)["position"]["abbreviation"] == "DH"
    # 시즌 기록에서 Devers를 투수(P)로 바꾸면 그 타격 행은 역할 행이 아니게 된다
    _split_of(stats, DEVERS)["position"]["abbreviation"] = "P"
    _write_json(stats_path, stats)
    _remove_saber_row(raw_dir, S, "bat", DEVERS)
    assert _failed(_passing(raw_dir, S)) == set()


def test_validate_war_covers_roles_position_player_becomes_pitcher_role(raw_dir: Path) -> None:
    # 시즌 기록에서 등판한 야수의 투구 포지션을 P로 바꾸면 역할 행 → 세이버메트릭스에 없으면 실패
    stats_path = raw_dir / raw_statsapi_path(S, "pit")
    stats = _load(stats_path)
    _split_of(stats, HERNANDEZ)["position"]["abbreviation"] = "P"
    _write_json(stats_path, stats)
    assert _failed(_passing(raw_dir, S)) == set()
    _remove_saber_row(raw_dir, S, "pit", HERNANDEZ)
    report = _passing(raw_dir, S)
    assert _failed(report) == {COVERS_ROLES}
    assert str(HERNANDEZ) in _check(report, COVERS_ROLES).detail


@pytest.mark.parametrize("role", ROLES)
def test_validate_war_covers_roles_missing_season_record_fails_without_exception(
    raw_dir: Path, role: Role
) -> None:
    league = _matched(raw_dir, S)
    (raw_dir / raw_statsapi_path(S, role)).unlink()
    report = validate_war_season(raw_dir, S, league_war=league)
    _assert_shape(report, S)
    assert COVERS_ROLES in _failed(report)
    assert not report.passed


@pytest.mark.parametrize("body", [b"<html>busy</html>", b"[]", b'{"stats": []}'])
def test_validate_war_covers_roles_malformed_season_record_fails_without_exception(
    raw_dir: Path, body: bytes
) -> None:
    league = _matched(raw_dir, S)
    (raw_dir / raw_statsapi_path(S, "pit")).write_bytes(body)
    report = validate_war_season(raw_dir, S, league_war=league)
    _assert_shape(report, S)
    assert COVERS_ROLES in _failed(report)
    assert not report.passed


# ---------------------------------------------------------------------------
# validate_war_seasons
# ---------------------------------------------------------------------------


def test_validate_war_seasons_sorted_and_unique(shared_raw: Path) -> None:
    reports = validate_war_seasons(shared_raw, [2025, 2023, 2025, 2019], tolerance=HUGE_TOLERANCE)
    assert [r.season for r in reports] == [2019, 2023, 2025]
    for report in reports:
        _assert_shape(report, report.season)
        assert report.passed


def test_validate_war_seasons_accepts_one_shot_iterable(shared_raw: Path) -> None:
    seasons = (s for s in (2024, 2022))
    reports = validate_war_seasons(shared_raw, seasons, tolerance=HUGE_TOLERANCE)
    assert [r.season for r in reports] == [2022, 2024]


def test_validate_war_seasons_empty(shared_raw: Path) -> None:
    assert validate_war_seasons(shared_raw, []) == []


def test_validate_war_seasons_every_fixture_season(shared_raw: Path) -> None:
    reports = validate_war_seasons(shared_raw, SEASONS, tolerance=HUGE_TOLERANCE)
    assert [r.season for r in reports] == list(SEASONS)
    assert all(r.passed for r in reports)


def test_validate_war_seasons_forwards_league_war_and_tolerance(shared_raw: Path) -> None:
    league = _matched(shared_raw, S)
    (matched,) = validate_war_seasons(shared_raw, [S], league_war=league)
    assert matched.passed
    (default,) = validate_war_seasons(shared_raw, [S])
    assert _failed(default) == {LEAGUE_SUM}
    off = _matched(shared_raw, S, {"bat": 3.0})
    (strict,) = validate_war_seasons(shared_raw, [S], league_war=off)
    assert _failed(strict) == {LEAGUE_SUM}
    (loose,) = validate_war_seasons(shared_raw, [S], league_war=off, tolerance=4.0)
    assert loose.passed


def test_validate_war_seasons_matches_single_season(shared_raw: Path) -> None:
    league = _matched(shared_raw, S)
    (many,) = validate_war_seasons(shared_raw, [S], league_war=league)
    assert many == validate_war_season(shared_raw, S, league_war=league)


def test_validate_war_seasons_missing_season_fails_without_exception(shared_raw: Path) -> None:
    reports = validate_war_seasons(shared_raw, [2018, S], tolerance=HUGE_TOLERANCE)
    assert [r.season for r in reports] == [2018, S]
    assert not reports[0].passed and COMPLETE in _failed(reports[0])
    assert reports[1].passed


# ---------------------------------------------------------------------------
# 보고서 파일 — write_war_report · read_war_report
# ---------------------------------------------------------------------------


def _mixed_reports(raw_dir: Path) -> list[WarSeasonReport]:
    """통과한 시즌(2023 · 2024)과 리그 합이 실패한 시즌(2025, 기본 기준값)."""
    return [_passing(raw_dir, 2023), _passing(raw_dir, 2024), validate_war_season(raw_dir, S)]


def test_validate_war_report_round_trip(shared_raw: Path, tmp_path: Path) -> None:
    reports = _mixed_reports(shared_raw)
    assert [r.passed for r in reports] == [True, True, False]
    path = tmp_path / "war_validation_report.json"
    write_war_report(reports, path)
    back = read_war_report(path)
    assert back == reports
    assert [r.passed for r in back] == [True, True, False]
    assert all(isinstance(r, WarSeasonReport) for r in back)
    assert all(isinstance(r.checks, tuple) for r in back)


def test_validate_war_report_round_trip_keeps_failure_detail(raw_dir: Path, tmp_path: Path) -> None:
    _remove_saber_row(raw_dir, S, "bat", DEVERS)
    report = _passing(raw_dir, S)
    path = tmp_path / "report.json"
    write_war_report([report], path)
    (back,) = read_war_report(path)
    assert _check(back, COVERS_ROLES) == _check(report, COVERS_ROLES)
    assert back.inputs == report.inputs


def test_validate_war_report_same_input_same_bytes(shared_raw: Path, tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_war_report(_mixed_reports(shared_raw), a)
    write_war_report(_mixed_reports(shared_raw), b)  # 새로 만든 같은 보고서
    assert a.read_bytes() == b.read_bytes()
    write_war_report(_mixed_reports(shared_raw), a)  # 덮어써도 같다
    assert a.read_bytes() == b.read_bytes()


def test_validate_war_report_is_indented_utf8_json(shared_raw: Path, tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_war_report(_mixed_reports(shared_raw), path)
    text = path.read_bytes().decode("utf-8")
    json.loads(text)
    assert len(text.splitlines()) > 3  # 사람이 읽게 줄을 나눈다


def test_validate_war_report_empty_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_war_report([], path)
    assert read_war_report(path) == []


def test_validate_war_report_hand_made_round_trip(tmp_path: Path) -> None:
    report = WarSeasonReport(
        2022,
        (
            CheckResult(COMPLETE, True, "타격 6행 · 투구 5행"),
            CheckResult(WAR_PRESENT, True, "결측 0"),
            CheckResult(COVERS_ROLES, False, "타격 646240 없음"),
            CheckResult(LEAGUE_SUM, True, "타격 570.0 · 투구 430.0 (기준 570.0 · 430.0, ±1.0)"),
        ),
        {raw_saber_path(2022, "bat").as_posix(): "a" * 64},
    )
    path = tmp_path / "report.json"
    write_war_report([report], path)
    assert read_war_report(path) == [report]


def _season_entries(doc: Any) -> Iterator[dict[str, Any]]:
    """보고서 JSON 안에서 시즌 보고서 객체(`season`과 `checks`를 가진 객체)를 모두 찾는다."""
    if isinstance(doc, dict):
        if "season" in doc and "checks" in doc:
            yield doc
        for value in doc.values():
            yield from _season_entries(value)
    elif isinstance(doc, list):
        for value in doc:
            yield from _season_entries(value)


def _check_entries(doc: Any) -> Iterator[dict[str, Any]]:
    """보고서 JSON 안에서 검사 결과 객체(`name`이 WAR_CHECK_NAMES 중 하나)를 모두 찾는다."""
    if isinstance(doc, dict):
        if doc.get("name") in WAR_CHECK_NAMES and "passed" in doc:
            yield doc
        for value in doc.values():
            yield from _check_entries(value)
    elif isinstance(doc, list):
        for value in doc:
            yield from _check_entries(value)


def _rewrite(path: Path, edit: Callable[[Any], object]) -> None:
    doc = _load(path)
    edit(doc)
    path.write_bytes(json.dumps(doc, ensure_ascii=False).encode("utf-8"))


def _first_season_entry(doc: Any) -> dict[str, Any]:
    entries = list(_season_entries(doc))
    assert entries, "보고서 JSON에서 season·checks를 가진 시즌 보고서를 찾지 못했다"
    return entries[0]


def _first_check_entry(doc: Any) -> dict[str, Any]:
    entries = list(_check_entries(doc))
    assert entries, "보고서 JSON에서 name·passed를 가진 검사 결과를 찾지 못했다"
    return entries[0]


_BAD_REPORT_BODIES: dict[str, bytes] = {
    "empty": b"",
    "not-json": b"[{",
    "not-utf8": b"\xff\xfe[]",
    "null": b"null",
    "number": b"42",
    "string": b'"report"',
    "bool": b"true",
    "list-of-numbers": b"[2025]",
    "list-of-strings": b'["2025"]',
}


@pytest.mark.parametrize("body", _BAD_REPORT_BODIES.values(), ids=_BAD_REPORT_BODIES.keys())
def test_validate_war_read_report_malformed_body_raises_value_error(
    tmp_path: Path, body: bytes
) -> None:
    path = tmp_path / "report.json"
    path.write_bytes(body)
    with pytest.raises(ValueError):
        read_war_report(path)


_BAD_EDITS: dict[str, Callable[[Any], object]] = {
    "season-string": lambda d: _first_season_entry(d).update(season=str(S)),
    "season-bool": lambda d: _first_season_entry(d).update(season=True),
    "season-null": lambda d: _first_season_entry(d).update(season=None),
    "checks-not-list": lambda d: _first_season_entry(d).update(checks={}),
    "checks-item-not-object": lambda d: _first_season_entry(d).update(checks=["ok"]),
    "check-passed-string": lambda d: _first_check_entry(d).update(passed="true"),
    "check-passed-null": lambda d: _first_check_entry(d).update(passed=None),
    "check-name-number": lambda d: _first_check_entry(d).update(name=1),
    "check-detail-missing": lambda d: _first_check_entry(d).pop("detail"),
    "check-detail-number": lambda d: _first_check_entry(d).update(detail=3),
}


@pytest.mark.parametrize("edit", _BAD_EDITS.values(), ids=_BAD_EDITS.keys())
def test_validate_war_read_report_malformed_entry_raises_value_error(
    shared_raw: Path, tmp_path: Path, edit: Callable[[Any], object]
) -> None:
    path = tmp_path / "report.json"
    write_war_report([_passing(shared_raw, S)], path)
    _rewrite(path, edit)
    with pytest.raises(ValueError):
        read_war_report(path)


def test_validate_war_read_report_recomputes_passed_from_checks(
    shared_raw: Path, tmp_path: Path
) -> None:
    # 파일에 시즌 단위 `passed`를 적었다면 그것은 사람이 읽기 위한 값이다 — true로 고쳐도 checks로 다시 계산한다
    path = tmp_path / "report.json"
    write_war_report([validate_war_season(shared_raw, S)], path)

    def forge(doc: Any) -> None:
        for entry in _season_entries(doc):
            if "passed" in entry:
                entry["passed"] = True

    _rewrite(path, forge)
    (report,) = read_war_report(path)
    assert report.passed is False
    assert _failed(report) == {LEAGUE_SUM}


# ---------------------------------------------------------------------------
# assert_war_validated
# ---------------------------------------------------------------------------


def _write_passing(raw_dir: Path, tmp_path: Path, seasons: tuple[int, ...] = (2023, S)) -> Path:
    """시즌마다 픽스처 합을 기준값으로 검증해 보고서를 쓰고 경로를 돌려준다.

    스텁이 NotImplementedError를 내면 픽스처 오류가 아니라 테스트 실패로 보이도록
    픽스처 대신 테스트 본문에서 부른다.
    """
    report_path = tmp_path / "war_validation_report.json"
    write_war_report([_passing(raw_dir, s) for s in seasons], report_path)
    return report_path


def test_validate_war_assert_passes_on_fresh_report(raw_dir: Path, tmp_path: Path) -> None:
    report_path = _write_passing(raw_dir, tmp_path)
    assert_war_validated(raw_dir, report_path, [2023, S])
    assert_war_validated(raw_dir, report_path, [S])
    assert_war_validated(raw_dir, report_path, iter([S, 2023]))
    assert_war_validated(raw_dir, report_path, [S, S])


def test_validate_war_assert_without_report_raises(raw_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        assert_war_validated(raw_dir, tmp_path / "missing.json", [S])


def test_validate_war_assert_season_missing_from_report_raises(
    raw_dir: Path, tmp_path: Path
) -> None:
    report_path = _write_passing(raw_dir, tmp_path)
    with pytest.raises(ValidationError, match="2024"):
        assert_war_validated(raw_dir, report_path, [2024, S])


def test_validate_war_assert_failed_season_raises(raw_dir: Path, tmp_path: Path) -> None:
    report_path = tmp_path / "war_validation_report.json"
    reports = [_passing(raw_dir, 2024), validate_war_season(raw_dir, S)]
    assert reports[0].passed and not reports[1].passed
    write_war_report(reports, report_path)
    with pytest.raises(ValidationError, match=str(S)):
        assert_war_validated(raw_dir, report_path, [2024, S])
    # 요청하지 않은 시즌의 실패는 막지 않는다
    assert_war_validated(raw_dir, report_path, [2024])


def test_validate_war_assert_failed_covers_roles_raises(raw_dir: Path, tmp_path: Path) -> None:
    _remove_saber_row(raw_dir, S, "pit", OHTANI)
    report_path = _write_passing(raw_dir, tmp_path, (S,))
    with pytest.raises(ValidationError, match=str(S)):
        assert_war_validated(raw_dir, report_path, [S])


@pytest.mark.parametrize("rel", _season_rels(S), ids=lambda p: p.as_posix())
def test_validate_war_assert_changed_raw_raises(raw_dir: Path, tmp_path: Path, rel: Path) -> None:
    # 검증 뒤에 세이버메트릭스나 시즌 기록 원본이 바뀌면(sha256 불일치) 막는다
    report_path = _write_passing(raw_dir, tmp_path)
    path = raw_dir / rel
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValidationError) as exc:
        assert_war_validated(raw_dir, report_path, [S])
    assert rel.as_posix() in str(exc.value)
    assert_war_validated(raw_dir, report_path, [2023])  # 다른 시즌은 그대로 통과


@pytest.mark.parametrize("role", ROLES)
def test_validate_war_assert_deleted_raw_raises(raw_dir: Path, tmp_path: Path, role: Role) -> None:
    report_path = _write_passing(raw_dir, tmp_path)
    (raw_dir / raw_saber_path(S, role)).unlink()
    with pytest.raises(ValidationError):
        assert_war_validated(raw_dir, report_path, [S])


def test_validate_war_assert_ignores_files_outside_inputs(raw_dir: Path, tmp_path: Path) -> None:
    report_path = _write_passing(raw_dir, tmp_path)
    for rel in (raw_saber_path(2019, "bat"), raw_statsapi_path(2026, "pit")):
        other = raw_dir / rel
        other.write_bytes(other.read_bytes() + b"\n")
    assert_war_validated(raw_dir, report_path, [2023, S])


@pytest.mark.parametrize("body", _BAD_REPORT_BODIES.values(), ids=_BAD_REPORT_BODIES.keys())
def test_validate_war_assert_malformed_report_raises_validation_error(
    raw_dir: Path, tmp_path: Path, body: bytes
) -> None:
    # 읽지 못하는 보고서는 ValueError가 아니라 ValidationError로 막는다
    report_path = tmp_path / "war_validation_report.json"
    report_path.write_bytes(body)
    with pytest.raises(ValidationError):
        assert_war_validated(raw_dir, report_path, [S])


@pytest.mark.parametrize("edit", _BAD_EDITS.values(), ids=_BAD_EDITS.keys())
def test_validate_war_assert_malformed_entry_raises_validation_error(
    raw_dir: Path, tmp_path: Path, edit: Callable[[Any], object]
) -> None:
    report_path = _write_passing(raw_dir, tmp_path, (S,))
    _rewrite(report_path, edit)
    with pytest.raises(ValidationError):
        assert_war_validated(raw_dir, report_path, [S])


def test_validate_war_assert_forged_passed_flag_raises(raw_dir: Path, tmp_path: Path) -> None:
    # 실패한 시즌의 `passed`만 true로 고쳐도 막는다 (checks로 판정)
    report_path = tmp_path / "war_validation_report.json"
    write_war_report([validate_war_season(raw_dir, S)], report_path)

    def forge(doc: Any) -> None:
        for entry in _season_entries(doc):
            if "passed" in entry:
                entry["passed"] = True

    _rewrite(report_path, forge)
    with pytest.raises(ValidationError):
        assert_war_validated(raw_dir, report_path, [S])


def test_validate_war_assert_report_with_missing_check_raises(
    raw_dir: Path, tmp_path: Path
) -> None:
    # 검사가 하나 빠진 보고서(남은 검사는 모두 통과)도 통과로 읽지 않는다
    report_path = _write_passing(raw_dir, tmp_path, (S,))

    def drop_last_check(doc: Any) -> None:
        entry = _first_season_entry(doc)
        assert isinstance(entry["checks"], list)
        entry["checks"] = entry["checks"][:-1]

    _rewrite(report_path, drop_last_check)
    with pytest.raises(ValidationError):
        assert_war_validated(raw_dir, report_path, [S])


@pytest.mark.parametrize("rel", _season_rels(S), ids=lambda p: p.as_posix())
def test_validate_war_assert_unrecorded_input_raises(
    raw_dir: Path, tmp_path: Path, rel: Path
) -> None:
    # 통과로 적혀 있어도 원본 sha256이 빠진 보고서로는 "그 뒤로 바뀌지 않았다"를 대조할 수 없다
    report_path = _write_passing(raw_dir, tmp_path)

    def drop_input(doc: Any) -> None:
        entry = next(e for e in _season_entries(doc) if e["season"] == S)
        assert isinstance(entry.get("inputs"), dict), "시즌 보고서에 inputs 사전이 없다"
        del entry["inputs"][rel.as_posix()]

    _rewrite(report_path, drop_input)
    with pytest.raises(ValidationError) as exc:
        assert_war_validated(raw_dir, report_path, [S])
    assert rel.as_posix() in str(exc.value)
    assert_war_validated(raw_dir, report_path, [2023])


def test_validate_war_assert_does_not_accept_f2_report(raw_dir: Path, tmp_path: Path) -> None:
    # 기존 F2 검증 보고서(검사 6개)는 WAR 검사 4개가 아니다 — WAR 보고서로 통과시키지 않는다
    f2 = SeasonReport(
        S,
        tuple(CheckResult(n, True, "ok") for n in CHECK_NAMES),
        (),
        {rel.as_posix(): _sha256(raw_dir / rel) for rel in _season_rels(S)},
    )
    report_path = tmp_path / "validation_report.json"
    write_report([f2], report_path)
    with pytest.raises(ValidationError):
        assert_war_validated(raw_dir, report_path, [S])
