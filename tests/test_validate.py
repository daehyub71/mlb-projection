"""수집 완전성 검증(validate.py, SPEC F2) 테스트.

기준 상태는 `tests/fixtures/raw`(실제 응답에서 14명만 잘라 낸 원본, SOURCE.md)다. 잘라 낸 원본은
리그 합계가 현실적이지 않으므로 기준 통과는 범위를 넓히고 PA = BF 항등식을 끈 `RELAXED`로 본다.
원본을 바꾸는 테스트는 tmp_path에 복사본을 만들어 고친다(픽스처 자체는 건드리지 않는다).
"""

import ast
import copy
import csv
import dataclasses
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from mlbproj import validate
from mlbproj.collect import raw
from mlbproj.contracts import (
    FIELDING_YEARS,
    PEOPLE_FIELDS,
    ROLES,
    SAVANT_BATTER_COLUMNS,
    SAVANT_FRV_COLUMNS,
    SAVANT_OAA_COLUMNS,
    SAVANT_PITCHER_COLUMNS,
    SEASONS,
    SHORT_SEASON,
    STATSAPI_HITTING_STAT_FIELDS,
    STATSAPI_PITCHING_STAT_FIELDS,
    STATSAPI_SPLIT_FIELDS,
    Role,
    SavantKind,
    raw_people_path,
    raw_savant_path,
    raw_statsapi_path,
)
from mlbproj.validate import (
    CHECK_NAMES,
    DEFAULT_LIMITS,
    CheckResult,
    LeagueLimits,
    SeasonReport,
    ValidationError,
    assert_validated,
    read_report,
    validate_season,
    validate_seasons,
    write_report,
)

FIXTURE_RAW = Path(__file__).parent / "fixtures" / "raw"

# 잘라 낸 픽스처용 기준값 — 리그 합계 범위를 사실상 없애고 PA = BF 항등식을 끈다
RELAXED = LeagueLimits(
    pa_full=(0, 10**7),
    pa_short=(0, 10**7),
    outs_full=(0, 10**7),
    outs_short=(0, 10**7),
    pa_bf_identity=False,
)
# 합성 원본용 — 범위는 넓게, 항등식은 켠다
IDENTITY = dataclasses.replace(RELAXED, pa_bf_identity=True)

F2_1, F2_2, F2_3, F2_4, F2_5, F2_6 = CHECK_NAMES

# 원본을 고치는 테스트의 기본 시즌 (수비 원본이 있고 중복 행이 없는 시즌)
S = 2024

SAVANT_COLUMNS: dict[SavantKind, tuple[str, ...]] = {
    "batter": SAVANT_BATTER_COLUMNS,
    "pitcher": SAVANT_PITCHER_COLUMNS,
    "oaa": SAVANT_OAA_COLUMNS,
    "frv": SAVANT_FRV_COLUMNS,
}


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


def _failed(report: SeasonReport) -> set[str]:
    """실패한 검사 이름 집합."""
    return {c.name for c in report.checks if not c.passed}


def _check(report: SeasonReport, name: str) -> CheckResult:
    """이름으로 검사 결과 하나를 꺼낸다."""
    matches = [c for c in report.checks if c.name == name]
    assert len(matches) == 1, f"{name} 결과가 정확히 하나여야 한다: {report.checks}"
    return matches[0]


def _assert_shape(report: SeasonReport, season: int) -> None:
    """보고서가 시즌과 검사 6개(CHECK_NAMES 순서)를 갖췄는지 본다."""
    assert report.season == season
    assert tuple(c.name for c in report.checks) == CHECK_NAMES
    for c in report.checks:
        assert isinstance(c.passed, bool)
        assert isinstance(c.detail, str)
        assert c.detail.strip(), f"{c.name} detail이 비어 있다"


def _write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def _load_api(raw_dir: Path, season: int, role: Role) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(
        (raw_dir / raw_statsapi_path(season, role)).read_text(encoding="utf-8")
    )
    return doc


def _save_api(raw_dir: Path, season: int, role: Role, doc: dict[str, Any]) -> None:
    _write_json(raw_dir / raw_statsapi_path(season, role), doc)


def _splits(doc: dict[str, Any]) -> list[dict[str, Any]]:
    splits: list[dict[str, Any]] = doc["stats"][0]["splits"]
    return splits


def _split_of(doc: dict[str, Any], player_id: int) -> dict[str, Any]:
    matches = [s for s in _splits(doc) if s["player"]["id"] == player_id]
    assert len(matches) == 1, f"픽스처에 {player_id}가 한 번 있어야 한다"
    return matches[0]


def _api_totals(raw_dir: Path, season: int) -> tuple[int, int, int]:
    """픽스처에서 직접 센 (리그 PA 합, 아웃 합, BF 합)."""
    bat = _splits(_load_api(raw_dir, season, "bat"))
    pit = _splits(_load_api(raw_dir, season, "pit"))
    pa = sum(int(s["stat"]["plateAppearances"]) for s in bat)
    outs = sum(int(s["stat"]["outs"]) for s in pit)
    bf = sum(int(s["stat"]["battersFaced"]) for s in pit)
    return pa, outs, bf


def _edit_people(
    raw_dir: Path, batch: int, edit: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
) -> None:
    path = raw_dir / raw_people_path(batch)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["people"] = edit(doc["people"])
    _write_json(path, doc)


# Savant CSV는 줄 단위로 고친다. 머리글의 BOM과 나머지 줄의 원문 표기는 그대로 둔다.


def _savant_lines(raw_dir: Path, season: int, kind: SavantKind) -> tuple[Path, str, list[str]]:
    path = raw_dir / raw_savant_path(season, kind)
    text = path.read_bytes().decode("utf-8")
    bom = "﻿" if text.startswith("﻿") else ""
    return path, bom, text.removeprefix("﻿").splitlines()


def _write_lines(path: Path, bom: str, lines: list[str]) -> None:
    path.write_bytes((bom + "\n".join(lines) + "\n").encode("utf-8"))


def _parse(line: str) -> list[str]:
    return next(csv.reader([line]))


def _format(fields: list[str]) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(fields)
    return buf.getvalue()


def _id_column(kind: SavantKind) -> str:
    return "id" if kind == "frv" else "player_id"


def _row_index(lines: list[str], kind: SavantKind, player_id: int) -> int:
    col = _parse(lines[0]).index(_id_column(kind))
    for i, line in enumerate(lines[1:], start=1):
        if _parse(line)[col] == str(player_id):
            return i
    raise AssertionError(f"Savant {kind}에 {player_id}가 없다")


def _savant_ids(raw_dir: Path, season: int, kind: SavantKind) -> list[str]:
    _, _, lines = _savant_lines(raw_dir, season, kind)
    col = _parse(lines[0]).index(_id_column(kind))
    return [_parse(line)[col] for line in lines[1:]]


def _remove_savant_row(raw_dir: Path, season: int, kind: SavantKind, player_id: int) -> None:
    path, bom, lines = _savant_lines(raw_dir, season, kind)
    del lines[_row_index(lines, kind, player_id)]
    _write_lines(path, bom, lines)


def _duplicate_savant_row(
    raw_dir: Path,
    season: int,
    kind: SavantKind,
    player_id: int,
    *,
    copies: int = 2,
    change: dict[str, str] | None = None,
) -> None:
    """선수 행을 복제해 `copies`행으로 만든다. `change`가 있으면 복제한 행의 값만 바꾼다."""
    path, bom, lines = _savant_lines(raw_dir, season, kind)
    line = lines[_row_index(lines, kind, player_id)]
    if change:
        header = _parse(lines[0])
        fields = _parse(line)
        for column, value in change.items():
            assert fields[header.index(column)] != value
            fields[header.index(column)] = value
        line = _format(fields)
    lines.extend([line] * (copies - 1))
    _write_lines(path, bom, lines)


def _add_savant_row(raw_dir: Path, season: int, kind: SavantKind, player_id: int) -> None:
    """첫 데이터 행을 본떠 ID만 바꾼 행을 덧붙인다."""
    path, bom, lines = _savant_lines(raw_dir, season, kind)
    header = _parse(lines[0])
    fields = _parse(lines[1])
    fields[header.index(_id_column(kind))] = str(player_id)
    for name_col in ("last_name, first_name", "name"):
        if name_col in header:
            fields[header.index(name_col)] = "Extra, Player"
    lines.append(_format(fields))
    _write_lines(path, bom, lines)


def _drop_savant_column(raw_dir: Path, season: int, kind: SavantKind, column: str) -> None:
    path, bom, lines = _savant_lines(raw_dir, season, kind)
    rows = [_parse(line) for line in lines]
    idx = rows[0].index(column)
    for row in rows:
        del row[idx]
    _write_lines(path, bom, [_format(row) for row in rows])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_inputs(raw_dir: Path, season: int) -> dict[str, str]:
    """시즌 검사에 쓰는 원본: API 타격·투구, Savant 타자·투수, 수비 시즌이면 OAA·FRV, 선수 정보 전부."""
    rels = [
        raw_statsapi_path(season, "bat"),
        raw_statsapi_path(season, "pit"),
        raw_savant_path(season, "batter"),
        raw_savant_path(season, "pitcher"),
    ]
    if season in FIELDING_YEARS:
        rels += [raw_savant_path(season, "oaa"), raw_savant_path(season, "frv")]
    rels += sorted(p.relative_to(raw_dir) for p in (raw_dir / "statsapi").glob("people_*.json"))
    return {rel.as_posix(): _sha256(raw_dir / rel) for rel in rels}


def _snapshot(raw_dir: Path) -> dict[str, bytes]:
    return {
        p.relative_to(raw_dir).as_posix(): p.read_bytes() for p in raw_dir.rglob("*") if p.is_file()
    }


# ---------------------------------------------------------------------------
# 기준 상태: 잘라 낸 픽스처는 모든 시즌에서 여섯 검사를 통과한다
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("season", SEASONS)
def test_validate_baseline_every_season_passes(shared_raw: Path, season: int) -> None:
    report = validate_season(shared_raw, season, limits=RELAXED)
    _assert_shape(report, season)
    assert _failed(report) == set()
    assert report.passed is True


@pytest.mark.parametrize("season", SEASONS)
def test_validate_baseline_merged_duplicates_only_venditte(shared_raw: Path, season: int) -> None:
    # E19: Savant 투수 2019·2020에 Pat Venditte(519381)가 같은 값 2행. 다른 시즌·파일엔 중복이 없다
    report = validate_season(shared_raw, season, limits=RELAXED)
    expected = (f"savant/{season}_pitcher.csv: 519381 ×2",) if season in (2019, 2020) else ()
    assert report.merged_duplicates == expected


@pytest.mark.parametrize("season", SEASONS)
def test_validate_baseline_inputs_map_files_to_sha256(shared_raw: Path, season: int) -> None:
    report = validate_season(shared_raw, season, limits=RELAXED)
    assert report.inputs == _expected_inputs(shared_raw, season)


@pytest.mark.parametrize("season", [s for s in SEASONS if s not in FIELDING_YEARS])
def test_validate_f2_1_pre_fielding_seasons_need_no_fielding_files(
    shared_raw: Path, season: int
) -> None:
    for kind in ("oaa", "frv"):
        assert not (shared_raw / raw_savant_path(season, kind)).exists()
    report = validate_season(shared_raw, season, limits=RELAXED)
    assert _check(report, F2_1).passed
    assert not any(key.endswith(("_oaa.csv", "_frv.csv")) for key in report.inputs)


def test_validate_season_report_passed_property() -> None:
    ok = tuple(CheckResult(name, True, "ok") for name in CHECK_NAMES)
    assert SeasonReport(S, ok, (), {}).passed is True
    for i in range(len(CHECK_NAMES)):
        checks = tuple(dataclasses.replace(c, passed=(j != i)) for j, c in enumerate(ok))
        assert SeasonReport(S, checks, (), {}).passed is False


def test_validate_season_does_not_modify_raw_dir(raw_dir: Path) -> None:
    before = _snapshot(raw_dir)
    validate_seasons(raw_dir, SEASONS, limits=RELAXED)
    validate_season(raw_dir, S)
    assert _snapshot(raw_dir) == before


def test_validate_default_limits_cover_spec_league_totals() -> None:
    # SPEC F2 ④: 한 명 30 BF 이하·합계 0.5% 이하. 실데이터는 항등식을 항상 검사한다
    assert DEFAULT_LIMITS.max_missing_pitcher_bf == 30
    assert DEFAULT_LIMITS.max_missing_pitcher_share == 0.005
    assert DEFAULT_LIMITS.pa_bf_identity is True
    # SPEC E6·E20 실측 리그 합계가 기본 범위 안에 든다
    lo, hi = DEFAULT_LIMITS.pa_full
    for pa in (186_517, 181_818, 184_104, 183_849):
        assert lo <= pa <= hi
    lo, hi = DEFAULT_LIMITS.pa_short
    assert lo <= 66_506 <= hi
    lo, hi = DEFAULT_LIMITS.outs_full
    for outs in (127_845, 130_270):
        assert lo <= outs <= hi
    lo, hi = DEFAULT_LIMITS.outs_short
    assert lo <= 46_406 <= hi


# ---------------------------------------------------------------------------
# F2-1 필수 필드 — 파일·형식이 틀려도 예외 없이 실패한 검사로 담는다
# ---------------------------------------------------------------------------

_SEASON_FILES = [
    raw_statsapi_path(S, "bat"),
    raw_statsapi_path(S, "pit"),
    raw_savant_path(S, "batter"),
    raw_savant_path(S, "pitcher"),
    raw_savant_path(S, "oaa"),
    raw_savant_path(S, "frv"),
]


@pytest.mark.parametrize("rel", _SEASON_FILES, ids=lambda p: p.as_posix())
def test_validate_f2_1_missing_file_fails_without_exception(raw_dir: Path, rel: Path) -> None:
    (raw_dir / rel).unlink()
    report = validate_season(raw_dir, S, limits=RELAXED)
    _assert_shape(report, S)
    assert F2_1 in _failed(report)
    assert rel.name in _check(report, F2_1).detail
    assert report.passed is False


@pytest.mark.parametrize("kind", ["oaa", "frv"])
@pytest.mark.parametrize("season", FIELDING_YEARS)
def test_validate_f2_1_missing_fielding_file_fails_in_fielding_season(
    raw_dir: Path, season: int, kind: SavantKind
) -> None:
    (raw_dir / raw_savant_path(season, kind)).unlink()
    report = validate_season(raw_dir, season, limits=RELAXED)
    assert F2_1 in _failed(report)
    assert report.passed is False


@pytest.mark.parametrize(
    "rel, body",
    [
        (raw_statsapi_path(S, "bat"), b"{not json"),
        (raw_statsapi_path(S, "pit"), b'{"copyright": "x", "stats": []}'),
        (raw_statsapi_path(S, "bat"), b'{"copyright": "x"}'),
        (raw_statsapi_path(S, "bat"), b'{"stats": [{"totalSplits": 0, "splits": []}]}'),
        (raw_statsapi_path(S, "pit"), b'{"stats": [{"totalSplits": 1, "splits": [7]}]}'),
        (raw_statsapi_path(S, "bat"), b"[]"),
        (raw_statsapi_path(S, "pit"), b"\xff\xfe not utf-8"),
        (raw_savant_path(S, "batter"), b""),
        (raw_savant_path(S, "frv"), b""),
    ],
    ids=[
        "bat-not-json",
        "pit-empty-stats",
        "bat-no-stats",
        "bat-empty-splits",
        "pit-splits-not-objects",
        "bat-top-level-list",
        "pit-not-utf8",
        "batter-empty-csv",
        "frv-empty-csv",
    ],
)
def test_validate_f2_1_malformed_file_fails_without_exception(
    raw_dir: Path, rel: Path, body: bytes
) -> None:
    (raw_dir / rel).write_bytes(body)
    report = validate_season(raw_dir, S, limits=RELAXED)
    _assert_shape(report, S)
    assert F2_1 in _failed(report)
    assert report.passed is False


def test_validate_season_without_any_raw_file_fails_without_exception(shared_raw: Path) -> None:
    report = validate_season(shared_raw, 2018, limits=RELAXED)
    _assert_shape(report, 2018)
    assert F2_1 in _failed(report)
    assert report.passed is False


_API_FIELD_CASES = [
    *(("bat", "split", f) for f in STATSAPI_SPLIT_FIELDS),
    *(("pit", "split", f) for f in STATSAPI_SPLIT_FIELDS),
    *(("bat", "stat", f) for f in STATSAPI_HITTING_STAT_FIELDS),
    *(("pit", "stat", f) for f in STATSAPI_PITCHING_STAT_FIELDS),
]


def _drop_api_field(raw_dir: Path, role: Role, level: str, field: str) -> None:
    """S 시즌 API 원본의 첫 split에서 필드 하나를 뺀다."""
    doc = _load_api(raw_dir, S, role)
    target = _splits(doc)[0]
    del (target if level == "split" else target["stat"])[field]
    _save_api(raw_dir, S, role, doc)


@pytest.mark.parametrize("role, level, field", _API_FIELD_CASES)
def test_validate_f2_1_dropped_api_field_fails(
    raw_dir: Path, role: Role, level: str, field: str
) -> None:
    _drop_api_field(raw_dir, role, level, field)
    report = validate_season(raw_dir, S, limits=RELAXED)
    _assert_shape(report, S)
    assert F2_1 in _failed(report)
    assert field in _check(report, F2_1).detail


@pytest.mark.parametrize(
    "role, level, field",
    [
        ("bat", "split", "numTeams"),
        ("bat", "stat", "intentionalWalks"),
        ("pit", "split", "team"),
        ("pit", "stat", "saves"),
    ],
)
def test_validate_f2_1_dropped_unrelated_field_fails_only_f2_1(
    raw_dir: Path, role: Role, level: str, field: str
) -> None:
    # 다른 검사가 쓰지 않는 필드라면 F2-1만 실패한다
    _drop_api_field(raw_dir, role, level, field)
    assert _failed(validate_season(raw_dir, S, limits=RELAXED)) == {F2_1}


@pytest.mark.parametrize("delta", [1, -1])
@pytest.mark.parametrize("role", ROLES)
def test_validate_f2_1_total_splits_mismatch_fails(raw_dir: Path, role: Role, delta: int) -> None:
    doc = _load_api(raw_dir, S, role)
    doc["stats"][0]["totalSplits"] = len(_splits(doc)) + delta
    _save_api(raw_dir, S, role, doc)
    report = validate_season(raw_dir, S, limits=RELAXED)
    assert _failed(report) == {F2_1}


@pytest.mark.parametrize(
    "field, value",
    [
        ("inningsPitched", "199.2"),
        ("inningsPitched", "200.1"),
        ("outs", 599),
        ("inningsPitched", "199.3"),  # 소수부는 아웃 0~2만
        ("inningsPitched", 199.1),  # 문자열 표기가 아니다
        ("inningsPitched", "199"),
    ],
)
def test_validate_f2_1_innings_pitched_disagrees_with_outs_fails(
    raw_dir: Path, field: str, value: Any
) -> None:
    # 2024 Nola(605400): 199.1 이닝 = 199×3+1 = 598 아웃
    doc = _load_api(raw_dir, S, "pit")
    stat = _split_of(doc, 605400)["stat"]
    assert (stat["inningsPitched"], stat["outs"]) == ("199.1", 598)
    stat[field] = value
    _save_api(raw_dir, S, "pit", doc)
    report = validate_season(raw_dir, S, limits=RELAXED)
    assert _failed(report) == {F2_1}
    assert "605400" in _check(report, F2_1).detail


@pytest.mark.parametrize(
    "role, player_id, field, convert",
    [
        ("bat", 646240, "plateAppearances", str),
        ("pit", 605400, "battersFaced", float),
        ("pit", 605400, "outs", lambda _: None),
        ("bat", 646240, "plateAppearances", lambda _: True),
    ],
    ids=["pa-string", "bf-float", "outs-null", "pa-bool"],
)
def test_validate_f2_1_non_integer_counting_stat_fails(
    raw_dir: Path, role: Role, player_id: int, field: str, convert: Callable[[Any], Any]
) -> None:
    # F2-3~F2-5가 더하는 값이 정수가 아니면 조용히 0으로 세지 않고 F2-1이 실패한다
    doc = _load_api(raw_dir, S, role)
    stat = _split_of(doc, player_id)["stat"]
    stat[field] = convert(stat[field])
    _save_api(raw_dir, S, role, doc)
    report = validate_season(raw_dir, S, limits=RELAXED)
    assert F2_1 in _failed(report)
    assert f"stat.{field}" in _check(report, F2_1).detail


def test_validate_f2_1_player_id_not_integer_fails(raw_dir: Path) -> None:
    doc = _load_api(raw_dir, S, "bat")
    _split_of(doc, 646240)["player"]["id"] = "646240"
    _save_api(raw_dir, S, "bat", doc)
    report = validate_season(raw_dir, S, limits=RELAXED)
    assert F2_1 in _failed(report)
    assert "player.id" in _check(report, F2_1).detail


@pytest.mark.parametrize("kind", ["batter", "pitcher", "oaa", "frv"])
def test_validate_f2_1_header_only_csv_fails(raw_dir: Path, kind: SavantKind) -> None:
    # 머리글만 있고 데이터 행이 없으면 잘린 응답이다
    path, bom, lines = _savant_lines(raw_dir, S, kind)
    _write_lines(path, bom, lines[:1])
    report = validate_season(raw_dir, S, limits=RELAXED)
    _assert_shape(report, S)
    assert F2_1 in _failed(report)
    assert path.name in _check(report, F2_1).detail


@pytest.mark.parametrize("value", ["", "abc", "592663.5"])
@pytest.mark.parametrize("kind", ["batter", "pitcher", "oaa", "frv"])
def test_validate_f2_1_savant_id_not_integer_fails_without_exception(
    raw_dir: Path, kind: SavantKind, value: str
) -> None:
    path, bom, lines = _savant_lines(raw_dir, S, kind)
    header, fields = _parse(lines[0]), _parse(lines[1])
    fields[header.index(_id_column(kind))] = value
    lines[1] = _format(fields)
    _write_lines(path, bom, lines)
    report = validate_season(raw_dir, S, limits=RELAXED)
    _assert_shape(report, S)
    assert F2_1 in _failed(report)
    detail = _check(report, F2_1).detail
    assert path.name in detail
    assert _id_column(kind) in detail
    assert report.passed is False


_SAVANT_COLUMN_CASES = [(kind, col) for kind, cols in SAVANT_COLUMNS.items() for col in cols]


@pytest.mark.parametrize("kind, column", _SAVANT_COLUMN_CASES)
def test_validate_f2_1_missing_savant_column_fails(
    raw_dir: Path, kind: SavantKind, column: str
) -> None:
    _drop_savant_column(raw_dir, S, kind, column)
    report = validate_season(raw_dir, S, limits=RELAXED)
    _assert_shape(report, S)
    assert F2_1 in _failed(report)
    assert column in _check(report, F2_1).detail


@pytest.mark.parametrize(
    "kind, column",
    [
        ("batter", "barrel_batted_rate"),
        ("pitcher", "xera"),
        ("oaa", "outs_above_average"),
        ("frv", "framing_runs"),
    ],
)
def test_validate_f2_1_missing_stat_column_fails_only_f2_1(
    raw_dir: Path, kind: SavantKind, column: str
) -> None:
    _drop_savant_column(raw_dir, S, kind, column)
    assert _failed(validate_season(raw_dir, S, limits=RELAXED)) == {F2_1}


# ---------------------------------------------------------------------------
# F2-2 ID 중복 — 같은 값 중복은 합치고 기록, 값이 다르면 실패
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role, player_id", [("bat", 646240), ("pit", 605400)])
def test_validate_f2_2_api_duplicate_player_fails(
    raw_dir: Path, role: Role, player_id: int
) -> None:
    # 팀별로 나뉜 행처럼 같은 선수가 두 번 (응답 완전성은 맞춰 둔다)
    doc = _load_api(raw_dir, S, role)
    extra = copy.deepcopy(_split_of(doc, player_id))
    extra["team"] = {"id": 999, "name": "Other Team", "link": "/api/v1/teams/999"}
    _splits(doc).append(extra)
    doc["stats"][0]["totalSplits"] = len(_splits(doc))
    _save_api(raw_dir, S, role, doc)
    report = validate_season(raw_dir, S, limits=RELAXED)
    assert _failed(report) == {F2_2}
    assert str(player_id) in _check(report, F2_2).detail
    assert report.merged_duplicates == ()


_SAVANT_DUP_CASES = [
    ("batter", 646240, "pa", "999"),
    ("pitcher", 605400, "pa", "999"),
    ("oaa", 646240, "outs_above_average", "99"),
    ("frv", 701538, "total_runs", "99.5"),
]


@pytest.mark.parametrize("kind, player_id, column, value", _SAVANT_DUP_CASES)
def test_validate_f2_2_savant_duplicate_with_different_values_fails(
    raw_dir: Path, kind: SavantKind, player_id: int, column: str, value: str
) -> None:
    _duplicate_savant_row(raw_dir, S, kind, player_id, change={column: value})
    report = validate_season(raw_dir, S, limits=RELAXED)
    assert _failed(report) == {F2_2}
    assert str(player_id) in _check(report, F2_2).detail
    assert report.merged_duplicates == ()


@pytest.mark.parametrize("kind, player_id", [(k, p) for k, p, _, _ in _SAVANT_DUP_CASES])
def test_validate_f2_2_savant_identical_duplicate_passes_and_is_recorded(
    raw_dir: Path, kind: SavantKind, player_id: int
) -> None:
    # OAA의 year 칸은 비어 있다(E4) — 빈 칸끼리도 같은 값이다
    _duplicate_savant_row(raw_dir, S, kind, player_id)
    report = validate_season(raw_dir, S, limits=RELAXED)
    assert _failed(report) == set()
    assert report.merged_duplicates == (f"savant/{S}_{kind}.csv: {player_id} ×2",)


def test_validate_f2_2_identical_duplicate_records_row_count(raw_dir: Path) -> None:
    _duplicate_savant_row(raw_dir, S, "batter", 646240, copies=3)
    report = validate_season(raw_dir, S, limits=RELAXED)
    assert _check(report, F2_2).passed
    assert report.merged_duplicates == (f"savant/{S}_batter.csv: 646240 ×3",)


def test_validate_f2_2_venditte_rows_that_differ_fail(raw_dir: Path) -> None:
    # 2019 Venditte 2행 중 하나의 값을 바꾸면 더 이상 같은 행이 아니다
    assert _savant_ids(raw_dir, 2019, "pitcher").count("519381") == 2
    _remove_savant_row(raw_dir, 2019, "pitcher", 519381)
    _duplicate_savant_row(raw_dir, 2019, "pitcher", 519381, change={"xera": "9.99"})
    report = validate_season(raw_dir, 2019, limits=RELAXED)
    assert _failed(report) == {F2_2}
    assert "519381" in _check(report, F2_2).detail
    assert report.merged_duplicates == ()


# ---------------------------------------------------------------------------
# F2-3 타자 ID 일치 — API PA>0 타자 == Savant 타자
# ---------------------------------------------------------------------------


def test_validate_f2_3_batter_missing_from_savant_fails(raw_dir: Path) -> None:
    _remove_savant_row(raw_dir, S, "batter", 646240)
    report = validate_season(raw_dir, S, limits=RELAXED)
    assert _failed(report) == {F2_3}
    assert "646240" in _check(report, F2_3).detail


def test_validate_f2_3_savant_only_batter_fails(raw_dir: Path) -> None:
    _add_savant_row(raw_dir, S, "batter", 999999)
    report = validate_season(raw_dir, S, limits=RELAXED)
    assert _failed(report) == {F2_3}
    assert "999999" in _check(report, F2_3).detail


def test_validate_f2_3_detail_counts_ids_and_lists_only_a_few(raw_dir: Path) -> None:
    # 한쪽에만 있는 ID가 많으면 수는 모두 세고 ID는 앞의 몇 개만 적는다
    extra = [990000 + 7 * i for i in range(30)]
    for pid in extra:
        _add_savant_row(raw_dir, S, "batter", pid)
    report = validate_season(raw_dir, S, limits=RELAXED)
    assert _failed(report) == {F2_3}
    detail = _check(report, F2_3).detail
    assert re.search(r"(?<!\d)30(?!\d)", detail), detail
    assert str(min(extra)) in detail
    assert 0 < sum(str(pid) in detail for pid in extra) < len(extra)


def test_validate_f2_3_zero_pa_batter_absent_from_savant_passes(shared_raw: Path) -> None:
    # 2020 Iglesias(628452)·Nola(605400)는 API 타격 행이 PA 0이고 Savant 타자에 없다
    doc = _load_api(shared_raw, 2020, "bat")
    savant = _savant_ids(shared_raw, 2020, "batter")
    for pid in (628452, 605400):
        assert _split_of(doc, pid)["stat"]["plateAppearances"] == 0
        assert str(pid) not in savant
    assert _check(validate_season(shared_raw, 2020, limits=RELAXED), F2_3).passed


def test_validate_f2_3_zero_pa_batter_present_in_savant_fails(raw_dir: Path) -> None:
    _add_savant_row(raw_dir, 2020, "batter", 628452)
    report = validate_season(raw_dir, 2020, limits=RELAXED)
    assert _failed(report) == {F2_3}
    assert "628452" in _check(report, F2_3).detail


def test_validate_f2_3_batter_with_plate_appearances_missing_from_savant_fails(
    raw_dir: Path,
) -> None:
    doc = _load_api(raw_dir, 2020, "bat")
    _split_of(doc, 628452)["stat"]["plateAppearances"] = 3
    _save_api(raw_dir, 2020, "bat", doc)
    report = validate_season(raw_dir, 2020, limits=RELAXED)
    assert _failed(report) == {F2_3}
    assert "628452" in _check(report, F2_3).detail


# ---------------------------------------------------------------------------
# F2-4 투수 포함 관계 — Savant 투수 ⊆ API 투구, 빠진 실제 투수는 BF 상한 안
# ---------------------------------------------------------------------------


def test_validate_f2_4_savant_only_pitcher_fails(raw_dir: Path) -> None:
    # Devers(646240)는 2024 API 타격에만 있고 투구 기록이 없다
    assert all(s["player"]["id"] != 646240 for s in _splits(_load_api(raw_dir, S, "pit")))
    _add_savant_row(raw_dir, S, "pitcher", 646240)
    report = validate_season(raw_dir, S, limits=RELAXED)
    assert _failed(report) == {F2_4}
    assert "646240" in _check(report, F2_4).detail


@pytest.mark.parametrize(
    "season, player_id, position",
    [(2024, 605400, "P"), (2023, 660271, "TWP")],
)
def test_validate_f2_4_missing_real_pitcher_over_bf_limit_fails(
    raw_dir: Path, season: int, player_id: int, position: str
) -> None:
    split = _split_of(_load_api(raw_dir, season, "pit"), player_id)
    assert split["position"]["abbreviation"] == position
    assert split["stat"]["battersFaced"] > RELAXED.max_missing_pitcher_bf
    _remove_savant_row(raw_dir, season, "pitcher", player_id)
    report = validate_season(raw_dir, season, limits=RELAXED)
    assert _failed(report) == {F2_4}
    assert str(player_id) in _check(report, F2_4).detail


@pytest.mark.parametrize("max_bf, ok", [(259, True), (258, False)])
def test_validate_f2_4_missing_pitcher_bf_limit_is_inclusive(
    raw_dir: Path, max_bf: int, ok: bool
) -> None:
    # 2024 Iglesias(628452, P) BF 259
    assert _split_of(_load_api(raw_dir, S, "pit"), 628452)["stat"]["battersFaced"] == 259
    _remove_savant_row(raw_dir, S, "pitcher", 628452)
    limits = dataclasses.replace(
        RELAXED, max_missing_pitcher_bf=max_bf, max_missing_pitcher_share=1.0
    )
    assert _check(validate_season(raw_dir, S, limits=limits), F2_4).passed is ok


@pytest.mark.parametrize("margin, ok", [(0.01, True), (-0.01, False)])
def test_validate_f2_4_missing_pitcher_share_limit(raw_dir: Path, margin: float, ok: bool) -> None:
    # 한 명 상한은 넉넉히 두고, 빠진 실제 투수 BF 합 / 리그 BF 합(API 투구 battersFaced 합)만 본다
    missing = (628452, 543037)
    for pid in missing:
        _remove_savant_row(raw_dir, S, "pitcher", pid)
    pit = _splits(_load_api(raw_dir, S, "pit"))
    league_bf = sum(int(s["stat"]["battersFaced"]) for s in pit)
    missing_bf = sum(int(s["stat"]["battersFaced"]) for s in pit if s["player"]["id"] in missing)
    limits = dataclasses.replace(
        RELAXED,
        max_missing_pitcher_bf=10**6,
        max_missing_pitcher_share=missing_bf / league_bf + margin,
    )
    assert _check(validate_season(raw_dir, S, limits=limits), F2_4).passed is ok


STRICT_PITCHERS = dataclasses.replace(
    RELAXED, max_missing_pitcher_bf=0, max_missing_pitcher_share=0.0
)


def test_validate_f2_4_missing_position_player_has_no_bf_limit(raw_dir: Path) -> None:
    # 2024 Hernández(571771)의 투구 행은 포지션 3B이고 Savant 투수에 없다(E2). BF가 커도 제한이 없다
    doc = _load_api(raw_dir, S, "pit")
    split = _split_of(doc, 571771)
    assert split["position"]["abbreviation"] == "3B"
    assert "571771" not in _savant_ids(raw_dir, S, "pitcher")
    split["stat"]["battersFaced"] = 5000
    _save_api(raw_dir, S, "pit", doc)
    report = validate_season(raw_dir, S, limits=STRICT_PITCHERS)
    assert _failed(report) == set()


def test_validate_f2_4_position_player_removed_from_savant_passes(raw_dir: Path) -> None:
    # 2025 Hernández는 Savant 투수에 있다. 빼도 등판한 야수라 통과한다
    assert "571771" in _savant_ids(raw_dir, 2025, "pitcher")
    _remove_savant_row(raw_dir, 2025, "pitcher", 571771)
    assert _failed(validate_season(raw_dir, 2025, limits=STRICT_PITCHERS)) == set()


# ---------------------------------------------------------------------------
# F2-5 리그 합계 — 범위(2020은 *_short)와 PA 합 == BF 합
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("season", SEASONS)
def test_validate_f2_5_default_limits_reject_fixture_totals(shared_raw: Path, season: int) -> None:
    # 14명만 남긴 픽스처의 합계는 실제 리그 범위에 한참 못 미친다. 기본 limits가 기본값이다
    assert _failed(validate_season(shared_raw, season)) == {F2_5}


@pytest.mark.parametrize("bounds", ["exact", "above", "below"])
@pytest.mark.parametrize("metric", ["pa", "outs"])
@pytest.mark.parametrize("season", [2023, SHORT_SEASON])
def test_validate_f2_5_range_is_inclusive(
    shared_raw: Path, season: int, metric: str, bounds: str
) -> None:
    pa, outs, _ = _api_totals(shared_raw, season)
    total = pa if metric == "pa" else outs
    rng = {"exact": (total, total), "above": (total + 1, 10**7), "below": (0, total - 1)}[bounds]
    field = f"{metric}_{'short' if season == SHORT_SEASON else 'full'}"
    limits = dataclasses.replace(RELAXED, **{field: rng})
    assert _check(validate_season(shared_raw, season, limits=limits), F2_5).passed is (
        bounds == "exact"
    )


def test_validate_f2_5_short_season_uses_short_ranges(shared_raw: Path) -> None:
    full_closed = dataclasses.replace(RELAXED, pa_full=(0, 0), outs_full=(0, 0))
    short_closed = dataclasses.replace(RELAXED, pa_short=(0, 0), outs_short=(0, 0))
    assert _check(validate_season(shared_raw, SHORT_SEASON, limits=full_closed), F2_5).passed
    assert not _check(validate_season(shared_raw, 2021, limits=full_closed), F2_5).passed
    assert not _check(validate_season(shared_raw, SHORT_SEASON, limits=short_closed), F2_5).passed
    assert _check(validate_season(shared_raw, 2021, limits=short_closed), F2_5).passed


# 합성 원본: 타자 2명·투수 2명짜리 한 시즌. 필수 필드만 갖추고 F2-1~F2-6을 모두 만족한다
SYN_SEASON = 2021  # 수비 원본이 필요 없는 시즌
SYN_BATTERS = ((1001, "1B"), (1002, "LF"))
SYN_PITCHERS = (2001, 2002)
SYN_OUTS = (60, 45)


def _syn_api_doc(group: str, splits: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "copyright": "synthetic",
        "stats": [
            {
                "type": {"displayName": "season"},
                "group": {"displayName": group},
                "totalSplits": len(splits),
                "exemptions": [],
                "splits": splits,
                "splitsTiedWithOffset": [],
                "splitsTiedWithLimit": [],
                "playerPool": "ALL",
            }
        ],
    }


def _syn_split(player_id: int, position: str, stat: dict[str, Any]) -> dict[str, Any]:
    pitcher = position == "P"
    return {
        "season": str(SYN_SEASON),
        "stat": stat,
        "team": {"id": 1, "name": "Team A", "link": "/api/v1/teams/1"},
        "player": {
            "id": player_id,
            "fullName": f"Player {player_id}",
            "link": f"/api/v1/people/{player_id}",
        },
        "league": {"id": 103, "name": "AL", "link": "/api/v1/league/103"},
        "sport": {"id": 1, "link": "/api/v1/sports/1", "abbreviation": "MLB"},
        "numTeams": 1,
        "position": {
            "code": "1" if pitcher else "3",
            "name": "Pitcher" if pitcher else "Fielder",
            "type": "Pitcher" if pitcher else "Infielder",
            "abbreviation": position,
        },
    }


def _syn_person(player_id: int, position: str) -> dict[str, Any]:
    return {
        "id": player_id,
        "fullName": f"Player {player_id}",
        "birthDate": "1995-05-05",
        "primaryPosition": {"code": "1", "name": position, "type": "x", "abbreviation": position},
        "batSide": {"code": "R", "description": "Right"},
        "pitchHand": {"code": "R", "description": "Right"},
        "active": True,
    }


def _syn_csv(path: Path, header: tuple[str, ...], rows: list[list[Any]]) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("﻿" + buf.getvalue()).encode("utf-8"))


def _build_synthetic(raw_dir: Path, *, pa: tuple[int, int], bf: tuple[int, int]) -> None:
    bat = []
    for (pid, pos), n in zip(SYN_BATTERS, pa, strict=True):
        stat = dict.fromkeys(STATSAPI_HITTING_STAT_FIELDS, 0)
        stat.update(age=28, plateAppearances=n, atBats=n, obp=".300", slg=".400", ops=".700")
        bat.append(_syn_split(pid, pos, stat))
    pit = []
    for pid, n, outs in zip(SYN_PITCHERS, bf, SYN_OUTS, strict=True):
        stat = dict.fromkeys(STATSAPI_PITCHING_STAT_FIELDS, 0)
        stat.update(
            age=30,
            gamesPitched=10,
            inningsPitched=f"{outs // 3}.{outs % 3}",
            outs=outs,
            battersFaced=n,
        )
        pit.append(_syn_split(pid, "P", stat))
    _write_json(raw_dir / raw_statsapi_path(SYN_SEASON, "bat"), _syn_api_doc("hitting", bat))
    _write_json(raw_dir / raw_statsapi_path(SYN_SEASON, "pit"), _syn_api_doc("pitching", pit))
    filler = ["1"] * (len(SAVANT_BATTER_COLUMNS) - 3)
    _syn_csv(
        raw_dir / raw_savant_path(SYN_SEASON, "batter"),
        SAVANT_BATTER_COLUMNS,
        [[f"Player, {pid}", pid, SYN_SEASON, *filler] for pid, _ in SYN_BATTERS],
    )
    filler = ["1"] * (len(SAVANT_PITCHER_COLUMNS) - 3)
    _syn_csv(
        raw_dir / raw_savant_path(SYN_SEASON, "pitcher"),
        SAVANT_PITCHER_COLUMNS,
        [[f"Player, {pid}", pid, SYN_SEASON, *filler] for pid in SYN_PITCHERS],
    )
    people = [_syn_person(pid, pos) for pid, pos in SYN_BATTERS]
    people += [_syn_person(pid, "P") for pid in SYN_PITCHERS]
    _write_json(raw_dir / raw_people_path(0), {"copyright": "synthetic", "people": people})


def test_validate_f2_5_identity_holds_on_synthetic(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _build_synthetic(raw_dir, pa=(100, 50), bf=(90, 60))  # PA 150 = BF 150
    report = validate_season(raw_dir, SYN_SEASON, limits=IDENTITY)
    _assert_shape(report, SYN_SEASON)
    assert _failed(report) == set()
    assert report.passed is True


@pytest.mark.parametrize(
    "pa, bf, numbers",
    [((100, 50), (90, 61), ("150", "151")), ((101, 50), (90, 60), ("151", "150"))],
    ids=["bf-plus-one", "pa-plus-one"],
)
def test_validate_f2_5_identity_off_by_one_fails(
    tmp_path: Path, pa: tuple[int, int], bf: tuple[int, int], numbers: tuple[str, str]
) -> None:
    raw_dir = tmp_path / "raw"
    _build_synthetic(raw_dir, pa=pa, bf=bf)
    report = validate_season(raw_dir, SYN_SEASON, limits=IDENTITY)
    assert _failed(report) == {F2_5}
    detail = _check(report, F2_5).detail
    for number in numbers:
        assert number in detail
    # 항등식을 끄면 같은 원본이 통과한다
    assert validate_season(raw_dir, SYN_SEASON, limits=RELAXED).passed is True


# ---------------------------------------------------------------------------
# F2-6 선수 정보 — 그 시즌 모든 선수가 people_*.json에 PEOPLE_FIELDS를 갖추고 있다
# ---------------------------------------------------------------------------


def test_validate_f2_6_player_missing_from_people_fails(raw_dir: Path) -> None:
    _edit_people(raw_dir, 0, lambda people: [p for p in people if p["id"] != 646240])
    report = validate_season(raw_dir, 2025, limits=RELAXED)
    assert _failed(report) == {F2_6}
    assert "646240" in _check(report, F2_6).detail


def test_validate_f2_6_only_players_of_the_season_matter(raw_dir: Path) -> None:
    # McGonigle(805808)은 2026 데뷔 — 2025 검사에는 필요 없다
    _edit_people(raw_dir, 0, lambda people: [p for p in people if p["id"] != 805808])
    assert _failed(validate_season(raw_dir, 2025, limits=RELAXED)) == set()
    report = validate_season(raw_dir, 2026, limits=RELAXED)
    assert _failed(report) == {F2_6}
    assert "805808" in _check(report, F2_6).detail


def _without_field(
    player_id: int, field: str
) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    def edit(people: list[dict[str, Any]]) -> list[dict[str, Any]]:
        target = [p for p in people if p.get("id") == player_id]
        assert len(target) == 1
        del target[0][field]
        return people

    return edit


@pytest.mark.parametrize("field", [f for f in PEOPLE_FIELDS if f != "id"])
def test_validate_f2_6_person_missing_field_fails(raw_dir: Path, field: str) -> None:
    # Skenes(694973, people_001)는 2024 데뷔 — 2023에는 없다
    _edit_people(raw_dir, 1, _without_field(694973, field))
    report = validate_season(raw_dir, 2024, limits=RELAXED)
    assert _failed(report) == {F2_6}
    detail = _check(report, F2_6).detail
    assert "694973" in detail
    assert field in detail
    assert _failed(validate_season(raw_dir, 2023, limits=RELAXED)) == set()


def test_validate_f2_6_person_without_id_fails(raw_dir: Path) -> None:
    _edit_people(raw_dir, 1, _without_field(694973, "id"))
    assert _failed(validate_season(raw_dir, 2024, limits=RELAXED)) == {F2_6}


def test_validate_f2_6_reads_every_people_batch(raw_dir: Path) -> None:
    # Skenes를 세 번째 묶음(people_002.json)으로 옮겨도 찾는다
    moved: list[dict[str, Any]] = []

    def take(people: list[dict[str, Any]]) -> list[dict[str, Any]]:
        moved.extend(p for p in people if p["id"] == 694973)
        return [p for p in people if p["id"] != 694973]

    _edit_people(raw_dir, 1, take)
    _write_json(raw_dir / raw_people_path(2), {"copyright": "x", "people": moved})
    report = validate_season(raw_dir, 2024, limits=RELAXED)
    assert _failed(report) == set()
    assert report.inputs == _expected_inputs(raw_dir, 2024)
    assert "statsapi/people_002.json" in report.inputs


def test_validate_f2_6_no_people_files_fails_without_exception(raw_dir: Path) -> None:
    for batch in (0, 1):
        (raw_dir / raw_people_path(batch)).unlink()
    report = validate_season(raw_dir, S, limits=RELAXED)
    _assert_shape(report, S)
    assert F2_6 in _failed(report)
    assert report.passed is False


@pytest.mark.parametrize(
    "body",
    [b"{", b'{"copyright": "x"}', b'{"people": {"id": 1}}', b"[]"],
    ids=["not-json", "no-people", "people-not-list", "top-level-list"],
)
def test_validate_f2_6_malformed_people_file_fails_without_exception(
    raw_dir: Path, body: bytes
) -> None:
    (raw_dir / raw_people_path(1)).write_bytes(body)
    report = validate_season(raw_dir, S, limits=RELAXED)
    _assert_shape(report, S)
    assert F2_6 in _failed(report)
    assert raw_people_path(1).name in _check(report, F2_6).detail
    assert report.passed is False


def _person(raw_dir: Path, batch: int, player_id: int) -> dict[str, Any]:
    doc = json.loads((raw_dir / raw_people_path(batch)).read_text(encoding="utf-8"))
    matches = [p for p in doc["people"] if p["id"] == player_id]
    assert len(matches) == 1
    person: dict[str, Any] = matches[0]
    return person


def test_validate_f2_6_identical_person_in_two_batches_passes(raw_dir: Path) -> None:
    # 같은 선수 정보가 두 묶음에 그대로 있으면 하나로 본다
    skenes = _person(raw_dir, 1, 694973)
    _write_json(raw_dir / raw_people_path(2), {"copyright": "x", "people": [skenes]})
    report = validate_season(raw_dir, 2024, limits=RELAXED)
    assert _failed(report) == set()


@pytest.mark.parametrize("name", ["people_002.json", "people_old.json"])
def test_validate_f2_6_conflicting_person_in_two_batches_fails(raw_dir: Path, name: str) -> None:
    # 지워지지 않은 옛 묶음 등으로 같은 선수의 값이 엇갈리면 어느 쪽을 쓸지 정할 수 없다.
    # 이름 규칙에 딱 맞지 않는 people_*.json도 읽고 sha256을 남긴다
    stale = _person(raw_dir, 1, 694973) | {"active": False}
    _write_json(raw_dir / "statsapi" / name, {"copyright": "x", "people": [stale]})
    report = validate_season(raw_dir, 2024, limits=RELAXED)
    assert _failed(report) == {F2_6}
    assert "694973" in _check(report, F2_6).detail
    assert f"statsapi/{name}" in report.inputs


def test_validate_f2_6_people_problem_kept_when_api_unreadable(raw_dir: Path) -> None:
    # API 원본을 읽지 못해 F2-6을 검사하지 못해도 선수 정보 묶음의 문제는 남긴다
    (raw_dir / raw_statsapi_path(S, "bat")).unlink()
    (raw_dir / raw_people_path(1)).write_bytes(b"{")
    report = validate_season(raw_dir, S, limits=RELAXED)
    _assert_shape(report, S)
    assert {F2_1, F2_6} <= _failed(report)
    assert raw_people_path(1).name in _check(report, F2_6).detail
    assert raw_people_path(1).name not in _check(report, F2_1).detail


# ---------------------------------------------------------------------------
# validate_seasons
# ---------------------------------------------------------------------------


def test_validate_seasons_sorts_by_season(shared_raw: Path) -> None:
    order = [2024, 2019, 2026, 2020]
    reports = validate_seasons(shared_raw, iter(order), limits=RELAXED)
    assert [r.season for r in reports] == sorted(order)
    assert reports == [validate_season(shared_raw, s, limits=RELAXED) for s in sorted(order)]


def test_validate_seasons_repeated_season_is_checked_once(shared_raw: Path) -> None:
    reports = validate_seasons(shared_raw, [2024, 2023, 2024], limits=RELAXED)
    assert [r.season for r in reports] == [2023, 2024]


def test_validate_seasons_default_limits(shared_raw: Path) -> None:
    reports = validate_seasons(shared_raw, [S])
    assert reports == [validate_season(shared_raw, S, limits=DEFAULT_LIMITS)]
    assert _failed(reports[0]) == {F2_5}


# ---------------------------------------------------------------------------
# write_report / read_report
# ---------------------------------------------------------------------------


def _mixed_reports(raw_dir: Path) -> list[SeasonReport]:
    """통과 시즌과 실패 시즌(2020은 기본 limits → F2-5 실패)이 섞인 보고서."""
    return [
        validate_season(raw_dir, s, limits=DEFAULT_LIMITS if s == SHORT_SEASON else RELAXED)
        for s in SEASONS
    ]


def _dicts(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _dicts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _dicts(value)


def test_validate_report_round_trip_is_equal(shared_raw: Path, tmp_path: Path) -> None:
    reports = _mixed_reports(shared_raw)
    assert any(not r.passed for r in reports)
    path = tmp_path / "validation_report.json"
    write_report(reports, path)
    assert read_report(path) == reports


def test_validate_report_round_trip_handwritten(tmp_path: Path) -> None:
    checks = tuple(
        CheckResult(name, i != 4, f'설명 {i}: "따옴표", PA 183,849 = BF 183,849 ×2')
        for i, name in enumerate(CHECK_NAMES)
    )
    late = SeasonReport(
        2026,
        checks,
        ("savant/2026_batter.csv: 1 ×3", "savant/2026_pitcher.csv: 2 ×2"),
        {"statsapi/2026_bat.json": "0" * 64, "savant/2026_frv.csv": "f" * 64},
    )
    early = SeasonReport(2019, tuple(dataclasses.replace(c, passed=True) for c in checks), (), {})
    path = tmp_path / "report.json"
    write_report([late, early], path)
    assert read_report(path) == [early, late]


def test_validate_report_bytes_are_deterministic(shared_raw: Path, tmp_path: Path) -> None:
    reports = _mixed_reports(shared_raw)
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    write_report(reports, a)
    write_report(reports, a)  # 덮어써도 같다
    write_report(list(reversed(reports)), b)  # 순서가 달라도 시즌 오름차순으로 쓴다
    write_report(_mixed_reports(shared_raw), c)  # 새로 만든 같은 보고서
    assert a.read_bytes() == b.read_bytes() == c.read_bytes()
    assert [r.season for r in read_report(b)] == sorted(SEASONS)


def test_validate_report_json_is_utf8_with_sorted_keys(shared_raw: Path, tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_report(_mixed_reports(shared_raw), path)
    doc = json.loads(path.read_bytes().decode("utf-8"))
    found = list(_dicts(doc))
    assert found
    for d in found:
        assert list(d) == sorted(d)


def test_validate_write_report_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "data" / "interim" / "validation_report.json"
    report = SeasonReport(S, tuple(CheckResult(n, True, "ok") for n in CHECK_NAMES), (), {})
    write_report([report], path)
    assert read_report(path) == [report]


def test_validate_write_report_rejects_repeated_season(tmp_path: Path) -> None:
    ok = SeasonReport(S, tuple(CheckResult(n, True, "ok") for n in CHECK_NAMES), (), {})
    bad = dataclasses.replace(ok, checks=ok.checks[:-1])
    path = tmp_path / "report.json"
    with pytest.raises(ValueError, match=str(S)):
        write_report([ok, bad], path)
    assert not path.exists()


def _report_doc(season: int = S) -> dict[str, Any]:
    """read_report가 읽을 수 있는 최소 시즌 보고서 JSON."""
    return {
        "season": season,
        "checks": [{"name": n, "passed": True, "detail": "ok"} for n in CHECK_NAMES],
        "merged_duplicates": [],
        "inputs": {raw_statsapi_path(season, "bat").as_posix(): "0" * 64},
    }


def _report_file(edit: Callable[[dict[str, Any]], object] | None = None) -> bytes:
    """최소 시즌 보고서 하나를 `edit`로 고쳐 보고서 파일 바이트(시즌 목록)로 만든다."""
    doc = _report_doc()
    if edit is not None:
        edit(doc)
    return json.dumps([doc]).encode("utf-8")


# 형식이 틀린 보고서 파일 — read_report는 ValueError, assert_validated는 ValidationError
_BAD_REPORTS: dict[str, bytes] = {
    "not-json": b"[{",
    "not-utf8": b"\xff\xfe[]",
    "top-level-object": json.dumps(_report_doc()).encode("utf-8"),
    "season-not-object": b"[2024]",
    "season-string": _report_file(lambda d: d.update(season=str(S))),
    "season-bool": _report_file(lambda d: d.update(season=True)),
    "season-missing": _report_file(lambda d: d.pop("season")),
    "checks-not-list": _report_file(lambda d: d.update(checks={})),
    "check-not-object": _report_file(lambda d: d.update(checks=["ok"])),
    "check-passed-string": _report_file(lambda d: d["checks"][0].update(passed="true")),
    "check-detail-missing": _report_file(lambda d: d["checks"][0].pop("detail")),
    "merged-not-strings": _report_file(lambda d: d.update(merged_duplicates=[1])),
    "inputs-not-object": _report_file(lambda d: d.update(inputs=["statsapi/2024_bat.json"])),
    "inputs-sha-not-string": _report_file(lambda d: d.update(inputs={"statsapi/2024_bat.json": 1})),
    "repeated-season": json.dumps([_report_doc(), _report_doc()]).encode("utf-8"),
}


def test_validate_read_report_minimal_doc_is_readable(tmp_path: Path) -> None:
    # 아래 형식 오류 테스트의 기준: 고치지 않은 최소 보고서는 읽힌다
    path = tmp_path / "report.json"
    path.write_bytes(_report_file())
    (report,) = read_report(path)
    assert report.season == S
    assert report.passed is True


@pytest.mark.parametrize("body", _BAD_REPORTS.values(), ids=_BAD_REPORTS.keys())
def test_validate_read_report_malformed_raises_value_error(tmp_path: Path, body: bytes) -> None:
    path = tmp_path / "report.json"
    path.write_bytes(body)
    with pytest.raises(ValueError):
        read_report(path)


def test_validate_read_report_recomputes_passed_from_checks(
    shared_raw: Path, tmp_path: Path
) -> None:
    # 파일의 `passed`는 사람이 읽기 위한 값이다. 손으로 true로 고쳐도 checks로 다시 계산한다
    path = tmp_path / "report.json"
    write_report([validate_season(shared_raw, S)], path)
    doc = json.loads(path.read_bytes().decode("utf-8"))
    assert doc[0]["passed"] is False
    doc[0]["passed"] = True
    path.write_bytes(json.dumps(doc).encode("utf-8"))
    (report,) = read_report(path)
    assert report.passed is False
    assert _failed(report) == {F2_5}


# ---------------------------------------------------------------------------
# assert_validated
# ---------------------------------------------------------------------------


def _validate_and_write(raw_dir: Path, tmp_path: Path) -> Path:
    """2023·2024를 검증해 보고서를 쓰고 보고서 경로를 돌려준다.

    스텁이 NotImplementedError를 내면 픽스처 오류가 아니라 테스트 실패로 보이도록
    픽스처 대신 테스트 본문에서 부른다.
    """
    report_path = tmp_path / "validation_report.json"
    write_report(validate_seasons(raw_dir, [2023, 2024], limits=RELAXED), report_path)
    return report_path


def test_validate_assert_validated_passes_on_fresh_report(raw_dir: Path, tmp_path: Path) -> None:
    report_path = _validate_and_write(raw_dir, tmp_path)
    assert_validated(raw_dir, report_path, [2023, 2024])
    assert_validated(raw_dir, report_path, [2024])
    assert_validated(raw_dir, report_path, iter([2024, 2023]))


def test_validate_assert_validated_without_report_raises(raw_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        assert_validated(raw_dir, tmp_path / "missing.json", [S])


def test_validate_assert_validated_season_missing_from_report_raises(
    raw_dir: Path,
    tmp_path: Path,
) -> None:
    report_path = _validate_and_write(raw_dir, tmp_path)
    with pytest.raises(ValidationError, match="2025"):
        assert_validated(raw_dir, report_path, [2024, 2025])


def test_validate_assert_validated_failed_season_raises(raw_dir: Path, tmp_path: Path) -> None:
    report_path = tmp_path / "validation_report.json"
    reports = [validate_season(raw_dir, 2024, limits=RELAXED), validate_season(raw_dir, 2025)]
    assert reports[0].passed and not reports[1].passed
    write_report(reports, report_path)
    with pytest.raises(ValidationError, match="2025"):
        assert_validated(raw_dir, report_path, [2024, 2025])
    # 요청하지 않은 시즌의 실패는 막지 않는다
    assert_validated(raw_dir, report_path, [2024])


@pytest.mark.parametrize(
    "rel",
    [
        raw_statsapi_path(2024, "pit"),
        raw_savant_path(2024, "batter"),
        raw_savant_path(2024, "frv"),
        raw_people_path(1),
    ],
    ids=lambda p: p.as_posix(),
)
def test_validate_assert_validated_changed_raw_raises(
    raw_dir: Path, tmp_path: Path, rel: Path
) -> None:
    report_path = _validate_and_write(raw_dir, tmp_path)
    path = raw_dir / rel
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValidationError) as exc:
        assert_validated(raw_dir, report_path, [2024])
    assert rel.as_posix() in str(exc.value)


def test_validate_assert_validated_deleted_raw_raises(raw_dir: Path, tmp_path: Path) -> None:
    report_path = _validate_and_write(raw_dir, tmp_path)
    (raw_dir / raw_savant_path(2024, "oaa")).unlink()
    with pytest.raises(ValidationError):
        assert_validated(raw_dir, report_path, [2024])


def test_validate_assert_validated_ignores_files_outside_inputs(
    raw_dir: Path,
    tmp_path: Path,
) -> None:
    report_path = _validate_and_write(raw_dir, tmp_path)
    other = raw_dir / raw_savant_path(2019, "batter")
    other.write_bytes(other.read_bytes() + b"\n")
    assert_validated(raw_dir, report_path, [2023, 2024])


@pytest.mark.parametrize("name", _BAD_REPORTS)
def test_validate_assert_validated_malformed_report_raises(
    raw_dir: Path, tmp_path: Path, name: str
) -> None:
    report_path = tmp_path / "validation_report.json"
    report_path.write_bytes(_BAD_REPORTS[name])
    with pytest.raises(ValidationError, match="검증 보고서"):
        assert_validated(raw_dir, report_path, [S])


def test_validate_assert_validated_forged_passed_flag_raises(raw_dir: Path, tmp_path: Path) -> None:
    # 실패한 시즌의 `passed`만 true로 고쳐도 막는다
    report_path = tmp_path / "validation_report.json"
    write_report([validate_season(raw_dir, S)], report_path)
    doc = json.loads(report_path.read_bytes().decode("utf-8"))
    doc[0]["passed"] = True
    report_path.write_bytes(json.dumps(doc).encode("utf-8"))
    with pytest.raises(ValidationError, match=F2_5):
        assert_validated(raw_dir, report_path, [S])


@pytest.mark.parametrize("name", ["people_002.json", "people_old.json"])
def test_validate_assert_validated_new_people_file_raises(
    raw_dir: Path, tmp_path: Path, name: str
) -> None:
    # 검증 뒤에 생긴 선수 정보 묶음은 검증하지 않은 입력이다
    report_path = _validate_and_write(raw_dir, tmp_path)
    _write_json(raw_dir / "statsapi" / name, {"copyright": "x", "people": []})
    with pytest.raises(ValidationError) as exc:
        assert_validated(raw_dir, report_path, [2024])
    assert f"statsapi/{name}" in str(exc.value)


@pytest.mark.parametrize(
    "rel",
    [raw_statsapi_path(2024, "bat"), raw_savant_path(2024, "oaa"), raw_people_path(0)],
    ids=lambda p: p.as_posix(),
)
def test_validate_assert_validated_unrecorded_input_raises(
    raw_dir: Path, tmp_path: Path, rel: Path
) -> None:
    # 통과로 적혀 있어도 시즌 원본의 sha256이 빠진 보고서로는 대조할 수 없다
    report_path = _validate_and_write(raw_dir, tmp_path)
    doc = json.loads(report_path.read_bytes().decode("utf-8"))
    entry = next(d for d in doc if d["season"] == 2024)
    del entry["inputs"][rel.as_posix()]
    report_path.write_bytes(json.dumps(doc).encode("utf-8"))
    with pytest.raises(ValidationError) as exc:
        assert_validated(raw_dir, report_path, [2024])
    assert rel.as_posix() in str(exc.value)
    assert_validated(raw_dir, report_path, [2023])


def test_validate_assert_validated_message_names_every_reason(
    raw_dir: Path, tmp_path: Path
) -> None:
    report_path = tmp_path / "validation_report.json"
    write_report(
        [validate_season(raw_dir, 2023, limits=RELAXED), validate_season(raw_dir, 2024)],
        report_path,
    )
    changed = raw_dir / raw_savant_path(2023, "pitcher")
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValidationError) as exc:
        assert_validated(raw_dir, report_path, [2023, 2024, 2025])
    message = str(exc.value)
    assert "2025" in message  # 보고서에 없는 시즌
    assert "2024" in message and F2_5 in message  # 실패한 시즌과 검사
    assert raw_savant_path(2023, "pitcher").as_posix() in message  # 바뀐 원본


# ---------------------------------------------------------------------------
# 모듈 경계 — 수집 모듈(statsapi·savant)은 다른 레인, 원본은 collect.raw로만 읽는다
# ---------------------------------------------------------------------------

_FORBIDDEN_MODULES = {"mlbproj.collect.statsapi", "mlbproj.collect.savant"}


def _imported_names(tree: ast.AST) -> set[str]:
    """import한 모듈과 `from X import Y`의 `X.Y`를 모은다 (상대 import는 mlbproj 기준)."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                base = ".".join(["mlbproj", *([node.module] if node.module else [])])
            found.add(base)
            found |= {f"{base}.{alias.name}" for alias in node.names}
    return found


def test_validate_module_does_not_import_collectors() -> None:
    source = Path(validate.__file__).read_text(encoding="utf-8")
    imported = _imported_names(ast.parse(source))
    assert not any(
        name == mod or name.startswith(mod + ".") for name in imported for mod in _FORBIDDEN_MODULES
    )


def test_validate_import_does_not_load_collectors() -> None:
    code = (
        "import sys, mlbproj.validate; "
        "print(sorted(m for m in sys.modules if m.startswith('mlbproj.collect')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    loaded = set(ast.literal_eval(out.strip()))
    assert not loaded & _FORBIDDEN_MODULES


def test_validate_reads_raw_only_through_collect_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 검사할 디렉토리의 원본은 모두 가짜 바이트다. collect.raw.read_json·read_csv만 진짜 내용을 준다.
    # 원본을 다른 길로 읽으면 가짜 바이트를 보게 되어 검사가 실패한다.
    real = tmp_path / "real"
    shutil.copytree(FIXTURE_RAW, real)
    decoy = tmp_path / "decoy"
    for src in real.rglob("*"):
        if src.is_file():
            dst = decoy / src.relative_to(real)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"decoy, not raw data\n")

    orig_json, orig_csv = raw.read_json, raw.read_csv
    seen: set[str] = set()

    def target(raw_dir: Path, rel: Path) -> Path:
        rel_path = Path(rel)
        if rel_path.is_absolute():
            rel_path = rel_path.relative_to(Path(raw_dir))
        seen.add(rel_path.as_posix())
        return rel_path

    def json_spy(raw_dir: Path, rel: Path) -> Any:
        return orig_json(real, target(raw_dir, rel))

    def csv_spy(raw_dir: Path, rel: Path) -> Any:
        return orig_csv(real, target(raw_dir, rel))

    # `from mlbproj.collect.raw import read_json` 같은 이름 가져오기도 바꿔 끼운다
    for name, value in list(vars(validate).items()):
        if value is orig_json:
            monkeypatch.setattr(validate, name, json_spy)
        elif value is orig_csv:
            monkeypatch.setattr(validate, name, csv_spy)
    monkeypatch.setattr(raw, "read_json", json_spy)
    monkeypatch.setattr(raw, "read_csv", csv_spy)

    report = validate.validate_season(decoy, S, limits=RELAXED)
    assert _failed(report) == set()
    assert set(_expected_inputs(real, S)) <= seen
