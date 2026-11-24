"""수비 시즌 표(clean/fielding.py) 테스트 — OAA·FRV 바깥 결합 (SPEC D21·E4·E18, §6).

기대값은 `tests/fixtures/raw/savant`의 OAA·FRV CSV를 이 파일에서 `csv` 모듈로 직접 읽어 정한다.
고친 원본이 필요한 테스트는 픽스처를 `tmp_path`로 복사해서 바꾼다(픽스처 자체는 건드리지 않는다).

픽스처 사례 (SOURCE.md)
- J.T. Realmuto(592663): 포수. FRV에는 2022~2026 모두 있고, OAA에는 2022(1B)만 있다 → 2023~2026은 FRV에만.
- Miguel Cabrera(408234): 2023 OAA에만 있다(FRV 없음).
- Eloy Jiménez(650391): 2025 결장 → 2025 OAA·FRV 모두 없다.
"""

import csv
import math
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlbproj import contracts as c
from mlbproj.clean.fielding import build_fielding
from mlbproj.contracts import SavantKind
from mlbproj.interim import read_table, write_table

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RAW = ROOT / "tests" / "fixtures" / "raw"

REALMUTO = 592663
CABRERA = 408234
MERRILL = 701538
DEVERS = 646240
JIMENEZ = 650391

# Savant FRV CSV 열 → 정제 열 (stub의 FRV_COLUMNS와 같은 내용을 여기서 따로 적는다)
FRV_MAP: dict[str, str] = {
    "total_runs": "frv_total",
    "range_runs": "frv_range",
    "arm_runs": "frv_arm",
    "dp_runs": "frv_dp",
    "framing_runs": "frv_framing",
    "blocking_runs": "frv_blocking",
    "throwing_runs": "frv_throwing",
}
FRV_VALUE_COLUMNS: list[str] = list(FRV_MAP.values())
OUTS_COLUMNS: list[str] = [f"outs_{p}" for p in c.FRV_OUT_POSITIONS]

CsvRows = list[dict[str, str]]


# ---------------------------------------------------------------------------
# 도우미 — 원본 CSV를 직접 읽고 고친다
# ---------------------------------------------------------------------------


def _csv_path(raw_dir: Path, season: int, kind: SavantKind) -> Path:
    return raw_dir / c.raw_savant_path(season, kind)


def _read_csv(raw_dir: Path, season: int, kind: SavantKind) -> tuple[list[str], CsvRows]:
    """원본 CSV의 머리줄과 행(열 이름 → 원문)을 돌려준다."""
    with _csv_path(raw_dir, season, kind).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    return header, rows


def _write_csv(
    raw_dir: Path, season: int, kind: SavantKind, header: list[str], rows: CsvRows
) -> None:
    """고친 행을 BOM이 붙은 CSV로 다시 쓴다."""
    with _csv_path(raw_dir, season, kind).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _edit_csv(
    raw_dir: Path, season: int, kind: SavantKind, edit: Callable[[CsvRows], CsvRows]
) -> None:
    header, rows = _read_csv(raw_dir, season, kind)
    _write_csv(raw_dir, season, kind, header, edit(rows))


def _frv_by_id(raw_dir: Path, season: int) -> dict[int, dict[str, str]]:
    return {int(r["id"]): r for r in _read_csv(raw_dir, season, "frv")[1]}


def _oaa_by_id(raw_dir: Path, season: int) -> dict[int, dict[str, str]]:
    return {int(r["player_id"]): r for r in _read_csv(raw_dir, season, "oaa")[1]}


def _num(text: str) -> float:
    """CSV 원문을 실수로. 빈 칸은 NaN."""
    return float(text) if text.strip() else math.nan


def _outs(text: str) -> int:
    """아웃 수 원문을 정수로. 빈 칸은 0 (stub docstring)."""
    return int(float(text)) if text.strip() else 0


def _copy_raw(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    shutil.copytree(FIXTURE_RAW, raw_dir)
    return raw_dir


def _row(df: pd.DataFrame, player_id: int, season: int) -> pd.Series:
    """(player_id, season) 행 하나. 정확히 한 행이 있어야 한다."""
    hit = df[(df["player_id"] == player_id) & (df["season"] == season)]
    assert len(hit) == 1, f"({player_id}, {season}) 행이 {len(hit)}개다"
    return hit.iloc[0]


def _assert_float(actual: Any, expected: float, where: str) -> None:
    if math.isnan(expected):
        assert pd.isna(actual), f"{where}: 결측이어야 하는데 {actual!r}"
    else:
        assert not pd.isna(actual), f"{where}: {expected}이어야 하는데 결측"
        assert float(actual) == pytest.approx(expected, rel=1e-12, abs=1e-12), where


def _expected_keys(raw_dir: Path) -> list[tuple[int, int]]:
    """CSV에서 직접 구한 (season, player_id) 목록 — OAA ∪ FRV, season·player_id 순."""
    keys: set[tuple[int, int]] = set()
    for season in c.FIELDING_YEARS:
        ids = set(_frv_by_id(raw_dir, season)) | set(_oaa_by_id(raw_dir, season))
        keys.update((season, pid) for pid in ids)
    return sorted(keys)


def _keys(df: pd.DataFrame) -> list[tuple[int, int]]:
    return [(int(s), int(p)) for s, p in zip(df["season"], df["player_id"], strict=True)]


# ---------------------------------------------------------------------------
# 픽스처 원본 그대로
# ---------------------------------------------------------------------------


def test_clean_fielding_contract() -> None:
    df = build_fielding(FIXTURE_RAW)
    assert c.check_frame(df, c.FIELDING_SEASONS) == []
    assert set(df["league"]) == {c.LEAGUE}


def test_clean_fielding_seasons_are_fielding_years() -> None:
    df = build_fielding(FIXTURE_RAW)
    assert sorted(set(int(s) for s in df["season"])) == list(c.FIELDING_YEARS)
    assert list(c.FIELDING_YEARS) == [2022, 2023, 2024, 2025, 2026]


def test_clean_fielding_sorted_by_season_then_player() -> None:
    df = build_fielding(FIXTURE_RAW)
    keys = _keys(df)
    assert keys == sorted(keys)
    assert df.index.equals(pd.RangeIndex(len(df)))  # 시즌을 이어 붙인 뒤 색인을 새로 매긴다


def test_clean_fielding_interim_round_trip(tmp_path: Path) -> None:
    # 저장(write_table)·읽기(read_table)를 거쳐도 값·형식이 그대로다 (정수가 실수로 바뀌지 않는다)
    df = build_fielding(FIXTURE_RAW)
    path = tmp_path / "fielding_seasons.csv"
    write_table(df, c.FIELDING_SEASONS, path)
    back = read_table(c.FIELDING_SEASONS, path)
    by_keys = df.sort_values(list(c.FIELDING_SEASONS.keys), kind="mergesort")
    assert_frame_equal(back, by_keys.reset_index(drop=True))


def test_clean_fielding_rows_are_union_of_oaa_and_frv() -> None:
    df = build_fielding(FIXTURE_RAW)
    expected = _expected_keys(FIXTURE_RAW)
    assert _keys(df) == expected
    # 픽스처에서 손으로 센 값: 2022 4명, 2023 5명(Cabrera 포함), 2024 5명, 2025 4명, 2026 5명
    assert len(expected) == 23
    per_season = df.groupby("season").size().to_dict()
    assert per_season == {2022: 4, 2023: 5, 2024: 5, 2025: 4, 2026: 5}
    # 2025 결장한 Jiménez는 2025 행이 없다
    assert (2025, JIMENEZ) not in expected


def test_clean_fielding_frv_values_match_csv() -> None:
    df = build_fielding(FIXTURE_RAW)
    checked = 0
    for season in c.FIELDING_YEARS:
        for pid, src in _frv_by_id(FIXTURE_RAW, season).items():
            row = _row(df, pid, season)
            for frv_col, clean_col in FRV_MAP.items():
                _assert_float(row[clean_col], _num(src[frv_col]), f"{pid} {season} {clean_col}")
            for col in OUTS_COLUMNS:
                assert int(row[col]) == _outs(src[col]), f"{pid} {season} {col}"
            checked += 1
    assert checked == 23 - 1  # Cabrera 2023만 FRV에 없다


def test_clean_fielding_outs_are_ints() -> None:
    df = build_fielding(FIXTURE_RAW)
    for col in OUTS_COLUMNS:
        assert pd.api.types.is_integer_dtype(df[col].dtype), col
        assert not df[col].isna().any(), col


def test_clean_fielding_realmuto_catcher_detail() -> None:
    df = build_fielding(FIXTURE_RAW)
    for season in c.FIELDING_YEARS:
        src = _frv_by_id(FIXTURE_RAW, season)[REALMUTO]
        row = _row(df, REALMUTO, season)
        for frv_col in ("framing_runs", "blocking_runs", "throwing_runs"):
            clean_col = FRV_MAP[frv_col]
            expected = _num(src[frv_col])
            assert not math.isnan(expected)  # 포수라 세부가 모두 채워져 있다
            _assert_float(row[clean_col], expected, f"{season} {clean_col}")
        assert int(row["outs_2"]) == int(src["outs_2"])
    # 원문에서 옮겨 적은 기준값: 2022 framing 2.4229…, throwing 9.2529…, blocking 3.4850…, outs_2 3395
    row_2022 = _row(df, REALMUTO, 2022)
    assert float(row_2022["frv_framing"]) == pytest.approx(2.422912026174164)
    assert float(row_2022["frv_throwing"]) == pytest.approx(9.252993493466626)
    assert float(row_2022["frv_blocking"]) == pytest.approx(3.485090995454291)
    assert int(row_2022["outs_2"]) == 3395
    assert int(row_2022["outs_3"]) == 18


def test_clean_fielding_oaa_is_outs_above_average() -> None:
    df = build_fielding(FIXTURE_RAW)
    checked = 0
    for season in c.FIELDING_YEARS:
        for pid, src in _oaa_by_id(FIXTURE_RAW, season).items():
            row = _row(df, pid, season)
            _assert_float(row["oaa"], float(src["outs_above_average"]), f"{pid} {season} oaa")
            checked += 1
    assert checked == 19
    # fielding_runs_prevented(-6)가 아니라 outs_above_average(-8)다
    assert float(_row(df, DEVERS, 2023)["oaa"]) == -8.0


def test_clean_fielding_oaa_only_player_has_no_frv() -> None:
    df = build_fielding(FIXTURE_RAW)
    assert CABRERA in _oaa_by_id(FIXTURE_RAW, 2023)
    assert CABRERA not in _frv_by_id(FIXTURE_RAW, 2023)
    assert df.loc[df["player_id"] == CABRERA, "season"].tolist() == [2023]
    row = _row(df, CABRERA, 2023)
    assert float(row["oaa"]) == 0.0  # 원문 outs_above_average = 0 (결측이 아니다)
    for col in FRV_VALUE_COLUMNS:
        assert pd.isna(row[col]), col
    for col in OUTS_COLUMNS:
        assert int(row[col]) == 0, col


def test_clean_fielding_frv_only_player_has_no_oaa() -> None:
    df = build_fielding(FIXTURE_RAW)
    for season in (2023, 2024, 2025, 2026):
        assert REALMUTO not in _oaa_by_id(FIXTURE_RAW, season)
        row = _row(df, REALMUTO, season)
        assert pd.isna(row["oaa"]), season
        assert not pd.isna(row["frv_total"]), season
    # 2022는 OAA에도 있다
    assert float(_row(df, REALMUTO, 2022)["oaa"]) == -2.0


def test_clean_fielding_frv_and_oaa_not_summed() -> None:
    df = build_fielding(FIXTURE_RAW)
    for season in c.FIELDING_YEARS:
        frv = _frv_by_id(FIXTURE_RAW, season)
        oaa = _oaa_by_id(FIXTURE_RAW, season)
        for pid in set(frv) & set(oaa):
            row = _row(df, pid, season)
            assert float(row["oaa"]) == float(oaa[pid]["outs_above_average"])
            assert float(row["frv_total"]) == pytest.approx(float(frv[pid]["total_runs"]))
    # Merrill 2024: OAA 12, FRV total 10.04… — 22.04 같은 합은 어디에도 없다
    merrill = _row(df, MERRILL, 2024)
    assert float(merrill["oaa"]) == 12.0
    assert float(merrill["frv_total"]) == pytest.approx(10.044080316664084)
    assert list(df.columns) == c.FIELDING_SEASONS.column_names


# ---------------------------------------------------------------------------
# 고친 원본 (tmp_path 복사본)
# ---------------------------------------------------------------------------


def test_clean_fielding_outer_join_synthetic_players(tmp_path: Path) -> None:
    raw_dir = _copy_raw(tmp_path)
    oaa_only, frv_only = 999001, 999002

    def add_oaa(rows: CsvRows) -> CsvRows:
        new = dict(rows[0])
        new.update(
            {
                "last_name, first_name": "Only, Oaa",
                "player_id": str(oaa_only),
                "fielding_runs_prevented": "4",
                "outs_above_average": "5",
            }
        )
        return [*rows, new]

    def add_frv(rows: CsvRows) -> CsvRows:
        new = {key: "" for key in rows[0]}
        new.update(
            {
                "name": "Only, Frv",
                "id": str(frv_only),
                "total_runs": "1.5",
                "framing_runs": "2.25",
                "blocking_runs": "-0.5",
                "throwing_runs": "-0.25",
                "outs_2": "1200",
            }
        )
        return [*rows, new]

    _edit_csv(raw_dir, 2026, "oaa", add_oaa)
    _edit_csv(raw_dir, 2026, "frv", add_frv)
    df = build_fielding(raw_dir)
    assert c.check_frame(df, c.FIELDING_SEASONS) == []

    row = _row(df, oaa_only, 2026)
    assert float(row["oaa"]) == 5.0
    for col in FRV_VALUE_COLUMNS:
        assert pd.isna(row[col]), col
    for col in OUTS_COLUMNS:
        assert int(row[col]) == 0, col

    row = _row(df, frv_only, 2026)
    assert pd.isna(row["oaa"])
    assert float(row["frv_total"]) == 1.5
    assert float(row["frv_framing"]) == 2.25
    assert float(row["frv_blocking"]) == -0.5
    assert float(row["frv_throwing"]) == -0.25
    for col in ("frv_range", "frv_arm", "frv_dp"):
        assert pd.isna(row[col]), col
    assert int(row["outs_2"]) == 1200
    for col in OUTS_COLUMNS[1:]:
        assert int(row[col]) == 0, col  # 빈 칸 → 0

    keys = _keys(df)
    assert keys == sorted(keys)
    assert len(keys) == 23 + 2


def test_clean_fielding_blank_outs_become_zero(tmp_path: Path) -> None:
    raw_dir = _copy_raw(tmp_path)
    header, rows = _read_csv(raw_dir, 2024, "frv")
    for row in rows:
        row["outs_9"] = ""  # 열 전체가 빈 칸 (Jiménez의 24도 지운다)
        if int(row["id"]) == MERRILL:
            for col in OUTS_COLUMNS[:6]:  # outs_2 ~ outs_7
                row[col] = ""
    _write_csv(raw_dir, 2024, "frv", header, rows)

    df = build_fielding(raw_dir)
    assert c.check_frame(df, c.FIELDING_SEASONS) == []
    for col in OUTS_COLUMNS:
        assert pd.api.types.is_integer_dtype(df[col].dtype), col

    season_2024 = df[df["season"] == 2024]
    assert season_2024["outs_9"].tolist() == [0] * len(season_2024)
    merrill = _row(df, MERRILL, 2024)
    assert [int(merrill[col]) for col in OUTS_COLUMNS] == [0, 0, 0, 0, 0, 0, 3970, 0]
    # 다른 해는 그대로다 (Jiménez 2023 outs_9 = 329)
    assert int(_row(df, JIMENEZ, 2023)["outs_9"]) == 329


def test_clean_fielding_ignores_other_seasons(tmp_path: Path) -> None:
    raw_dir = _copy_raw(tmp_path)
    for kind in ("oaa", "frv"):
        source = _csv_path(raw_dir, 2022, kind)
        shutil.copyfile(source, _csv_path(raw_dir, 2021, kind))
        shutil.copyfile(source, _csv_path(raw_dir, 2027, kind))
    df = build_fielding(raw_dir)
    assert sorted(set(int(s) for s in df["season"])) == list(c.FIELDING_YEARS)
    assert _keys(df) == _expected_keys(FIXTURE_RAW)


def test_clean_fielding_identical_duplicate_collapses(tmp_path: Path) -> None:
    raw_dir = _copy_raw(tmp_path)
    # Realmuto 2023 FRV 행은 빈 칸(range_runs 등)이 섞여 있다 — 결측끼리도 같은 값으로 본다
    _edit_csv(
        raw_dir,
        2023,
        "frv",
        lambda rows: [*rows, dict(next(r for r in rows if int(r["id"]) == REALMUTO))],
    )
    _edit_csv(
        raw_dir,
        2024,
        "oaa",
        lambda rows: [dict(next(r for r in rows if int(r["player_id"]) == DEVERS)), *rows],
    )
    df = build_fielding(raw_dir)
    assert c.check_frame(df, c.FIELDING_SEASONS) == []
    assert _keys(df) == _expected_keys(FIXTURE_RAW)

    src = _frv_by_id(FIXTURE_RAW, 2023)[REALMUTO]
    row = _row(df, REALMUTO, 2023)
    for frv_col, clean_col in FRV_MAP.items():
        _assert_float(row[clean_col], _num(src[frv_col]), f"2023 {clean_col}")
    assert float(_row(df, DEVERS, 2024)["oaa"]) == -7.0

    baseline = build_fielding(FIXTURE_RAW)
    assert_frame_equal(df.reset_index(drop=True), baseline.reset_index(drop=True))


@pytest.mark.parametrize(
    ("kind", "season", "id_column", "player_id", "value_column"),
    [
        pytest.param("frv", 2023, "id", REALMUTO, "framing_runs", id="frv"),
        pytest.param("oaa", 2024, "player_id", DEVERS, "outs_above_average", id="oaa"),
    ],
)
def test_clean_fielding_conflicting_duplicate_raises(
    tmp_path: Path,
    kind: SavantKind,
    season: int,
    id_column: str,
    player_id: int,
    value_column: str,
) -> None:
    raw_dir = _copy_raw(tmp_path)

    def add_conflict(rows: CsvRows) -> CsvRows:
        dup = dict(next(r for r in rows if int(r[id_column]) == player_id))
        dup[value_column] = str(float(dup[value_column]) + 3.0)
        return [*rows, dup]

    _edit_csv(raw_dir, season, kind, add_conflict)
    # 어느 파일의 어느 선수인지 메시지에 적는다
    with pytest.raises(ValueError, match=rf"{season}_{kind}\.csv.*{player_id}"):
        build_fielding(raw_dir)


@pytest.mark.parametrize(
    ("kind", "season", "id_column", "player_id", "text_column"),
    [
        pytest.param("frv", 2023, "id", REALMUTO, "name", id="frv-name"),
        pytest.param("oaa", 2024, "player_id", DEVERS, "display_team_name", id="oaa-team"),
    ],
)
def test_clean_fielding_duplicate_differing_in_unused_column_raises(
    tmp_path: Path,
    kind: SavantKind,
    season: int,
    id_column: str,
    player_id: int,
    text_column: str,
) -> None:
    # E19는 "모든 값이 같은" 중복만 합친다 — 표에 옮기지 않는 열이 달라도 하나로 합치지 않는다
    # (validate F2-2와 같은 규칙)
    raw_dir = _copy_raw(tmp_path)
    header, rows = _read_csv(raw_dir, season, kind)
    assert text_column in header
    dup = dict(next(r for r in rows if int(r[id_column]) == player_id))
    dup[text_column] = dup[text_column] + " (dup)"
    _write_csv(raw_dir, season, kind, header, [*rows, dup])
    with pytest.raises(ValueError, match=rf"{season}_{kind}\.csv.*{player_id}"):
        build_fielding(raw_dir)


@pytest.mark.parametrize(
    ("kind", "season", "id_column", "player_id", "column", "text"),
    [
        pytest.param("frv", 2024, "id", MERRILL, "outs_8", "3970.5", id="fractional-outs"),
        pytest.param("frv", 2024, "id", MERRILL, "outs_8", "--", id="text-outs"),
        pytest.param("frv", 2023, "id", REALMUTO, "framing_runs", "abc", id="text-frv"),
        pytest.param("oaa", 2024, "player_id", DEVERS, "outs_above_average", "--", id="text-oaa"),
    ],
)
def test_clean_fielding_bad_number_raises(
    tmp_path: Path,
    kind: SavantKind,
    season: int,
    id_column: str,
    player_id: int,
    column: str,
    text: str,
) -> None:
    # 숫자가 아닌 값을 결측으로, 소수 아웃 수를 버림으로 조용히 바꾸지 않는다
    raw_dir = _copy_raw(tmp_path)

    def corrupt(rows: CsvRows) -> CsvRows:
        for row in rows:
            if int(row[id_column]) == player_id:
                row[column] = text
        return rows

    _edit_csv(raw_dir, season, kind, corrupt)
    with pytest.raises(ValueError, match=rf"{season}_{kind}\.csv.*{column}"):
        build_fielding(raw_dir)


@pytest.mark.parametrize("kind", ["oaa", "frv"])
def test_clean_fielding_missing_file_raises(tmp_path: Path, kind: SavantKind) -> None:
    raw_dir = _copy_raw(tmp_path)
    _csv_path(raw_dir, 2025, kind).unlink()
    with pytest.raises(FileNotFoundError, match=rf"2025_{kind}\.csv"):
        build_fielding(raw_dir)
