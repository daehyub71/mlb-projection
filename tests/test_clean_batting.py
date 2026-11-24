"""타자 시즌 표(clean/batting.py, SPEC F3·D8·D9, §6 `batting_seasons`) 테스트.

기준 상태는 `tests/fixtures/raw`(실제 응답에서 14명만 잘라 낸 원본, SOURCE.md)다.
기댓값은 구현을 돌려서가 아니라 픽스처 JSON·CSV를 직접 읽어 만든다.
원본을 바꾸는 테스트는 tmp_path에 복사본을 만들어 고친다(픽스처 자체는 건드리지 않는다).
"""

import csv
import io
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mlbproj.clean.batting import (
    BATTING_SAVANT_COLUMNS,
    BATTING_STAT_FIELDS,
    build_batting,
    is_batter_split,
    parse_rate,
)
from mlbproj.contracts import (
    BATTER_EXCLUDED_POSITIONS,
    BATTING_SEASONS,
    LEAGUE,
    SEASONS,
    SHORT_SEASON,
    SHORT_SEASON_SCALE,
    check_frame,
    raw_savant_path,
    raw_statsapi_path,
)
from mlbproj.interim import read_table, write_table

FIXTURE_RAW = Path(__file__).parent / "fixtures" / "raw"

WAINWRIGHT = 425794
VENDITTE = 519381
COLE = 543037
NOLA = 605400
IGLESIAS = 628452
DEVERS = 646240
OHTANI = 660271
REALMUTO = 592663
# 투수의 타격 행(포지션 P)만 있는 선수 — 타자 표에 한 번도 나오면 안 된다 (D8·E10)
PITCHERS = (NOLA, WAINWRIGHT, VENDITTE, IGLESIAS, COLE)

RATE_COLUMNS = ("obp", "slg", "ops")
STATCAST_COLUMNS = tuple(BATTING_SAVANT_COLUMNS.values())

Key = tuple[int, int]  # (player_id, season)


# ---------------------------------------------------------------------------
# 픽스처 읽기·고치기 도우미
# ---------------------------------------------------------------------------


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _splits(raw_dir: Path, season: int) -> list[dict[str, Any]]:
    splits: list[dict[str, Any]] = _load(raw_dir / raw_statsapi_path(season, "bat"))["stats"][0][
        "splits"
    ]
    return splits


def _split(raw_dir: Path, season: int, player_id: int) -> dict[str, Any]:
    for split in _splits(raw_dir, season):
        if split["player"]["id"] == player_id:
            return split
    raise AssertionError(f"{season} 타격 원본에 {player_id}가 없다")


def _edit_split(raw_dir: Path, season: int, player_id: int, edit: Any) -> None:
    """타격 원본에서 한 선수의 split을 고친다 (edit(split)가 제자리에서 고친다)."""
    path = raw_dir / raw_statsapi_path(season, "bat")
    doc = _load(path)
    hits = [s for s in doc["stats"][0]["splits"] if s["player"]["id"] == player_id]
    assert len(hits) == 1
    edit(hits[0])
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _expected_batter_splits(raw_dir: Path) -> dict[Key, dict[str, Any]]:
    """포지션이 P가 아니고 타석이 1 이상인 타격 split (D8, contracts의 역할 판정 주석)."""
    out: dict[Key, dict[str, Any]] = {}
    for season in SEASONS:
        for split in _splits(raw_dir, season):
            position = split["position"]["abbreviation"]
            if position not in BATTER_EXCLUDED_POSITIONS and split["stat"]["plateAppearances"] > 0:
                key = (split["player"]["id"], season)
                assert key not in out
                out[key] = split
    return out


def _savant_path(raw_dir: Path, season: int) -> Path:
    return raw_dir / raw_savant_path(season, "batter")


def _savant_values(raw_dir: Path, season: int) -> dict[int, dict[str, float]]:
    """Savant 타자 CSV를 csv 모듈로 읽어 정제 열 이름 → 실수로 돌려준다."""
    with _savant_path(raw_dir, season).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[int, dict[str, float]] = {}
    for row in rows:
        pid = int(row["player_id"])
        assert pid not in out
        out[pid] = {clean: float(row[src]) for src, clean in BATTING_SAVANT_COLUMNS.items()}
    return out


def _savant_lines(raw_dir: Path, season: int) -> list[str]:
    return _savant_path(raw_dir, season).read_text(encoding="utf-8").splitlines()


def _write_savant_lines(raw_dir: Path, season: int, lines: list[str]) -> None:
    _savant_path(raw_dir, season).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _savant_line_index(lines: list[str], player_id: int) -> int:
    hits = [i for i, line in enumerate(lines) if f",{player_id}," in line]
    assert len(hits) == 1
    return hits[0]


def _rows(df: pd.DataFrame) -> dict[Key, dict[str, Any]]:
    """(player_id, season) → 행 (값은 파이썬 기본형)."""
    records = [{str(k): v for k, v in row.items()} for row in df.to_dict("records")]
    return {(int(r["player_id"]), int(r["season"])): r for r in records}


@pytest.fixture
def raw_copy(tmp_path: Path) -> Path:
    """고쳐 쓸 원본 복사본."""
    dst = tmp_path / "raw"
    shutil.copytree(FIXTURE_RAW, dst)
    return dst


# ---------------------------------------------------------------------------
# is_batter_split
# ---------------------------------------------------------------------------


def _synthetic_split(position: str, pa: int) -> dict[str, Any]:
    return {"position": {"abbreviation": position}, "stat": {"plateAppearances": pa}}


@pytest.mark.parametrize(
    ("season", "player_id"),
    [(2019, NOLA), (2021, NOLA), (2019, WAINWRIGHT), (2021, WAINWRIGHT)],
)
def test_clean_batting_is_batter_split_pitcher_hitting_row_false(
    season: int, player_id: int
) -> None:
    # 2021년까지 투수도 타석에 섰다(E10). 포지션 P인 타격 행은 타석이 있어도 타자가 아니다
    split = _split(FIXTURE_RAW, season, player_id)
    assert split["position"]["abbreviation"] == "P"
    assert split["stat"]["plateAppearances"] > 0

    assert is_batter_split(split) is False


@pytest.mark.parametrize("position", ["DH", "C", "X", "TWP", "1B", "SS", "CF"])
def test_clean_batting_is_batter_split_non_pitcher_position_true(position: str) -> None:
    assert is_batter_split(_synthetic_split(position, 1)) is True


@pytest.mark.parametrize(("season", "player_id"), [(2024, OHTANI), (2019, REALMUTO)])
def test_clean_batting_is_batter_split_fixture_batter_true(season: int, player_id: int) -> None:
    split = _split(FIXTURE_RAW, season, player_id)
    assert split["position"]["abbreviation"] in {"DH", "C"}

    assert is_batter_split(split) is True


@pytest.mark.parametrize("position", ["DH", "C", "X", "TWP", "P"])
def test_clean_batting_is_batter_split_zero_pa_false(position: str) -> None:
    # 타석 0인 행(대주자 등)은 기록으로 치지 않는다
    assert is_batter_split(_synthetic_split(position, 0)) is False


@pytest.mark.parametrize(("season", "player_id"), [(2019, VENDITTE), (2026, IGLESIAS)])
def test_clean_batting_is_batter_split_fixture_zero_pa_false(season: int, player_id: int) -> None:
    split = _split(FIXTURE_RAW, season, player_id)
    assert split["stat"]["plateAppearances"] == 0

    assert is_batter_split(split) is False


# ---------------------------------------------------------------------------
# parse_rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [(".350", 0.35), ("1.000", 1.0), (".000", 0.0), (".916", 0.916), ("1.066", 1.066)],
)
def test_clean_batting_parse_rate_number(text: str, expected: float) -> None:
    value = parse_rate(text)

    assert isinstance(value, float)
    assert value == expected


@pytest.mark.parametrize("text", [".---", "-.--", "", None])
def test_clean_batting_parse_rate_placeholder_is_nan(text: str | None) -> None:
    value = parse_rate(text)

    assert isinstance(value, float)
    assert math.isnan(value)


# ---------------------------------------------------------------------------
# build_batting — 픽스처 그대로
# ---------------------------------------------------------------------------


def test_clean_batting_build_batting_contract() -> None:
    batting = build_batting(FIXTURE_RAW)
    assert check_frame(batting, BATTING_SEASONS) == []


def test_clean_batting_build_batting_sorted_by_season_then_player() -> None:
    batting = build_batting(FIXTURE_RAW)
    keys = [(int(s), int(p)) for s, p in zip(batting["season"], batting["player_id"], strict=True)]

    assert keys == sorted(keys)


def test_clean_batting_build_batting_rows_are_batter_splits() -> None:
    batting = build_batting(FIXTURE_RAW)
    expected = _expected_batter_splits(FIXTURE_RAW)

    keys = [(int(p), int(s)) for p, s in zip(batting["player_id"], batting["season"], strict=True)]
    assert len(keys) == len(expected)
    assert set(keys) == set(expected)
    assert set(batting["league"]) == {LEAGUE}


def test_clean_batting_build_batting_excludes_pitcher_hitting() -> None:
    batting = build_batting(FIXTURE_RAW)
    ids = {int(v) for v in batting["player_id"]}

    for pitcher in PITCHERS:
        assert pitcher not in ids


def test_clean_batting_build_batting_two_way_player_has_rows() -> None:
    batting = build_batting(FIXTURE_RAW)
    # 오타니의 타격 포지션은 DH — 8시즌 모두 타자 행이 있다 (E7·F12)
    ohtani = batting[batting["player_id"] == OHTANI]

    assert sorted(int(s) for s in ohtani["season"]) == list(SEASONS)


def test_clean_batting_build_batting_traded_player_one_row() -> None:
    batting = build_batting(FIXTURE_RAW)
    # 2025 시즌 중 이적 — API가 준 합산 1행 그대로 (E1)
    split = _split(FIXTURE_RAW, 2025, DEVERS)
    assert split["numTeams"] == 2

    row = _rows(batting)[(DEVERS, 2025)]

    assert int(row["n_teams"]) == 2
    assert row["team_name"] == split["team"]["name"] == "San Francisco Giants"
    assert len(batting[(batting["player_id"] == DEVERS) & (batting["season"] == 2025)]) == 1


def test_clean_batting_build_batting_values_match_api() -> None:
    batting = build_batting(FIXTURE_RAW)
    rows = _rows(batting)

    for key, split in _expected_batter_splits(FIXTURE_RAW).items():
        row = rows[key]
        stat = split["stat"]
        assert int(row["age"]) == stat["age"], key
        assert row["team_name"] == split["team"]["name"], key
        assert int(row["n_teams"]) == split["numTeams"], key
        for column, field in BATTING_STAT_FIELDS.items():
            assert int(row[column]) == stat[field], (key, column)
        for column in RATE_COLUMNS:
            assert row[column] == float(stat[column]), (key, column)


def test_clean_batting_build_batting_short_season_adjustment() -> None:
    batting = build_batting(FIXTURE_RAW)
    for key, row in _rows(batting).items():
        season = key[1]
        pa = int(row["pa"])
        if season == SHORT_SEASON:
            assert bool(row["is_short_season"]) is True, key
            # pa × SHORT_SEASON_SCALE를 그대로 (예: 오타니 2020 175 PA → 175 * (162/60))
            assert row["pa_adj"] == pa * SHORT_SEASON_SCALE, key
        else:
            assert bool(row["is_short_season"]) is False, key
            assert row["pa_adj"] == float(pa), key
    assert (batting["season"] == SHORT_SEASON).any()


def test_clean_batting_build_batting_statcast_matches_savant() -> None:
    # 매핑의 정제 열이 계약의 Statcast 열과 같다 (SPEC §6)
    assert STATCAST_COLUMNS == ("woba", "xwoba", "ev_avg", "la_avg", "hard_hit_pct", "barrel_pct")
    assert set(STATCAST_COLUMNS) <= set(BATTING_SEASONS.column_names)
    batting = build_batting(FIXTURE_RAW)
    rows = _rows(batting)

    checked = 0
    for season in SEASONS:
        savant = _savant_values(FIXTURE_RAW, season)
        for (pid, row_season), row in rows.items():
            if row_season != season:
                continue
            # F2 ③: 픽스처에서는 모든 타자 행이 Savant에 있다
            assert pid in savant, (pid, season)
            for column in STATCAST_COLUMNS:
                assert row[column] == savant[pid][column], (pid, season, column)
            checked += 1
    assert checked == len(batting)


# ---------------------------------------------------------------------------
# build_batting — 고친 원본
# ---------------------------------------------------------------------------


def test_clean_batting_build_batting_missing_savant_row_is_null(raw_copy: Path) -> None:
    lines = _savant_lines(raw_copy, 2024)
    del lines[_savant_line_index(lines, DEVERS)]
    _write_savant_lines(raw_copy, 2024, lines)

    df = build_batting(raw_copy)

    assert check_frame(df, BATTING_SEASONS) == []
    rows = _rows(df)
    # API 행은 그대로 남고 Statcast 열만 결측 (F3)
    assert (
        int(rows[(DEVERS, 2024)]["pa"])
        == _split(FIXTURE_RAW, 2024, DEVERS)["stat"]["plateAppearances"]
    )
    for column in STATCAST_COLUMNS:
        assert pd.isna(rows[(DEVERS, 2024)][column]), column
    # 다른 시즌·다른 선수는 그대로
    devers_2023 = _savant_values(FIXTURE_RAW, 2023)[DEVERS]
    ohtani_2024 = _savant_values(FIXTURE_RAW, 2024)[OHTANI]
    for column in STATCAST_COLUMNS:
        assert rows[(DEVERS, 2023)][column] == devers_2023[column]
        assert rows[(OHTANI, 2024)][column] == ohtani_2024[column]
    assert len(df) == len(_expected_batter_splits(FIXTURE_RAW))


def test_clean_batting_build_batting_identical_savant_duplicate_collapses(raw_copy: Path) -> None:
    batting = build_batting(FIXTURE_RAW)
    lines = _savant_lines(raw_copy, 2024)
    lines.append(lines[_savant_line_index(lines, DEVERS)])
    _write_savant_lines(raw_copy, 2024, lines)

    df = build_batting(raw_copy)

    assert check_frame(df, BATTING_SEASONS) == []
    pd.testing.assert_frame_equal(df.reset_index(drop=True), batting.reset_index(drop=True))


def test_clean_batting_build_batting_conflicting_savant_duplicate_raises(raw_copy: Path) -> None:
    lines = _savant_lines(raw_copy, 2024)
    line = lines[_savant_line_index(lines, DEVERS)]
    header = next(csv.reader([lines[0].lstrip("﻿")]))
    fields = next(csv.reader([line]))
    fields[header.index("woba")] = ".300"
    out = io.StringIO()
    csv.writer(out, lineterminator="").writerow(fields)
    lines.append(out.getvalue())
    _write_savant_lines(raw_copy, 2024, lines)

    with pytest.raises(ValueError):
        build_batting(raw_copy)


def test_clean_batting_build_batting_rate_placeholder_is_null(raw_copy: Path) -> None:
    def blank_rates(split: dict[str, Any]) -> None:
        split["stat"]["obp"] = ".---"
        split["stat"]["slg"] = "-.--"
        split["stat"]["ops"] = ""

    _edit_split(raw_copy, 2023, REALMUTO, blank_rates)

    df = build_batting(raw_copy)

    assert check_frame(df, BATTING_SEASONS) == []
    row = _rows(df)[(REALMUTO, 2023)]
    for column in RATE_COLUMNS:
        assert pd.isna(row[column]), column
    assert int(row["pa"]) == _split(FIXTURE_RAW, 2023, REALMUTO)["stat"]["plateAppearances"]


def test_clean_batting_build_batting_missing_team_is_null(raw_copy: Path) -> None:
    def drop_team(split: dict[str, Any]) -> None:
        del split["team"]

    _edit_split(raw_copy, 2025, DEVERS, drop_team)

    df = build_batting(raw_copy)

    assert check_frame(df, BATTING_SEASONS) == []
    rows = _rows(df)
    assert pd.isna(rows[(DEVERS, 2025)]["team_name"])
    assert int(rows[(DEVERS, 2025)]["n_teams"]) == 2
    assert rows[(DEVERS, 2024)]["team_name"] == "Boston Red Sox"


def test_clean_batting_build_batting_zero_pa_non_pitcher_row_excluded(raw_copy: Path) -> None:
    # 야수의 타석 0인 행(대주자 등)은 타자 표에 넣지 않는다 (픽스처에 없어서 만든다)
    def zero_pa(split: dict[str, Any]) -> None:
        for field in BATTING_STAT_FIELDS.values():
            split["stat"][field] = 0

    _edit_split(raw_copy, 2026, REALMUTO, zero_pa)

    df = build_batting(raw_copy)

    assert (REALMUTO, 2026) not in _rows(df)
    assert (REALMUTO, 2025) in _rows(df)
    assert len(df) == len(_expected_batter_splits(FIXTURE_RAW)) - 1


# ---------------------------------------------------------------------------
# 조용한 변환 막기 — 형식이 틀린 값은 멈춘다
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["abc", "nan", "inf", "-inf", True])
def test_clean_batting_parse_rate_malformed_raises(text: Any) -> None:
    # 결측은 자리표시(".---" 등)와 None만이다. "nan"·"inf"·참거짓이 조용히 수가 되지 않는다
    with pytest.raises(ValueError):
        parse_rate(text)


@pytest.mark.parametrize("pa", ["3", 2.5, None, True])
def test_clean_batting_is_batter_split_non_int_pa_raises(pa: Any) -> None:
    # 2.5를 2로, "3"을 3으로, True를 1로 바꿔 판정하지 않는다
    with pytest.raises(ValueError):
        is_batter_split(_synthetic_split("DH", pa))


@pytest.mark.parametrize(
    ("field", "value"),
    [("homeRuns", 3.5), ("homeRuns", "28"), ("strikeOuts", None), ("atBats", True)],
)
def test_clean_batting_build_batting_non_int_count_raises(
    raw_copy: Path, field: str, value: Any
) -> None:
    def set_count(split: dict[str, Any]) -> None:
        split["stat"][field] = value

    _edit_split(raw_copy, 2024, DEVERS, set_count)

    with pytest.raises(ValueError, match=str(DEVERS)):
        build_batting(raw_copy)


def test_clean_batting_build_batting_missing_age_is_null(raw_copy: Path) -> None:
    def drop_age(split: dict[str, Any]) -> None:
        del split["stat"]["age"]

    _edit_split(raw_copy, 2024, DEVERS, drop_age)

    df = build_batting(raw_copy)

    assert check_frame(df, BATTING_SEASONS) == []
    # 결측이 있어도 정수 열이 실수로 바뀌지 않는다 (nullable Int64)
    assert df["age"].dtype == "Int64"
    rows = _rows(df)
    assert pd.isna(rows[(DEVERS, 2024)]["age"])
    assert int(rows[(DEVERS, 2023)]["age"]) == _split(FIXTURE_RAW, 2023, DEVERS)["stat"]["age"]


def test_clean_batting_build_batting_duplicate_api_row_raises(raw_copy: Path) -> None:
    # API 기록은 선수당 1행이다(E1). 겹치면 계약(키 중복)을 어긴 표를 돌려주지 않는다
    path = raw_copy / raw_statsapi_path(2024, "bat")
    doc = _load(path)
    block = doc["stats"][0]
    block["splits"].append(json.loads(json.dumps(_split(raw_copy, 2024, DEVERS))))
    block["totalSplits"] = len(block["splits"])
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        build_batting(raw_copy)


# ---------------------------------------------------------------------------
# 형식·순서·재현성
# ---------------------------------------------------------------------------


def test_clean_batting_build_batting_dtypes() -> None:
    df = build_batting(FIXTURE_RAW)

    for column in ("player_id", "season", "n_teams", *BATTING_STAT_FIELDS):
        assert df[column].dtype == "int64", column
    assert df["age"].dtype == "Int64"
    for column in (*RATE_COLUMNS, "pa_adj", *STATCAST_COLUMNS):
        assert df[column].dtype == "float64", column
    assert df["is_short_season"].dtype == "bool"
    for column in ("league", "team_name"):
        assert pd.api.types.is_string_dtype(df[column].dtype), column
        assert not pd.api.types.is_object_dtype(df[column].dtype), column


def test_clean_batting_build_batting_roundtrip_through_interim(tmp_path: Path) -> None:
    df = build_batting(FIXTURE_RAW)
    path = tmp_path / "batting_seasons.csv"

    write_table(df, BATTING_SEASONS, path)

    # write_table은 키(player_id·season·league) 순으로 쓴다. 그 순서로 맞추면 값·형식이 그대로
    keyed = df.sort_values(list(BATTING_SEASONS.keys), kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(read_table(BATTING_SEASONS, path), keyed)


def test_clean_batting_build_batting_independent_of_raw_order(raw_copy: Path) -> None:
    # 원본 안의 행 순서(API split·Savant 줄)가 바뀌어도 같은 표가 나온다
    expected = build_batting(FIXTURE_RAW)
    for season in SEASONS:
        path = raw_copy / raw_statsapi_path(season, "bat")
        doc = _load(path)
        doc["stats"][0]["splits"].reverse()
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        lines = _savant_lines(raw_copy, season)
        _write_savant_lines(raw_copy, season, [lines[0], *lines[:0:-1]])

    pd.testing.assert_frame_equal(build_batting(raw_copy), expected)


def test_clean_batting_build_batting_rows_use_only_their_season(raw_copy: Path) -> None:
    # 시즌 s의 행은 시즌 s 원본으로만 정해진다 — 2026 원본을 바꿔도 2025 이전 행은 그대로 (N2의 바탕)
    expected = build_batting(FIXTURE_RAW)

    def change(split: dict[str, Any]) -> None:
        split["stat"]["homeRuns"] += 10
        split["stat"]["ops"] = "1.234"

    _edit_split(raw_copy, 2026, DEVERS, change)
    lines = _savant_lines(raw_copy, 2026)
    del lines[_savant_line_index(lines, DEVERS)]
    _write_savant_lines(raw_copy, 2026, lines)

    df = build_batting(raw_copy)

    before = expected[expected["season"] < 2026].reset_index(drop=True)
    pd.testing.assert_frame_equal(df[df["season"] < 2026].reset_index(drop=True), before)
    row = _rows(df)[(DEVERS, 2026)]
    assert int(row["hr"]) == _split(FIXTURE_RAW, 2026, DEVERS)["stat"]["homeRuns"] + 10
    assert row["ops"] == 1.234
    assert pd.isna(row["woba"])


def test_clean_batting_build_batting_short_season_known_value() -> None:
    # 오타니 2020: 175 PA → 175 × 162/60 = 472.5 (손 계산, D9). 비율·셈 열은 환산하지 않는다
    split = _split(FIXTURE_RAW, 2020, OHTANI)
    assert split["stat"]["plateAppearances"] == 175

    row = _rows(build_batting(FIXTURE_RAW))[(OHTANI, 2020)]

    assert row["pa_adj"] == pytest.approx(472.5)
    assert int(row["pa"]) == 175
    assert row["ops"] == float(split["stat"]["ops"])
    assert bool(row["is_short_season"]) is True


# ---------------------------------------------------------------------------
# Savant 결합 — 고친 원본
# ---------------------------------------------------------------------------


def test_clean_batting_build_batting_savant_only_row_dropped(raw_copy: Path) -> None:
    # Savant에만 있는 행(투수의 타격 등)은 표에 들어오지 않는다
    lines = _savant_lines(raw_copy, 2024)
    fake = lines[_savant_line_index(lines, DEVERS)].replace(f",{DEVERS},", ",999001,")
    lines.append(fake)
    _write_savant_lines(raw_copy, 2024, lines)

    df = build_batting(raw_copy)

    assert 999001 not in {int(v) for v in df["player_id"]}
    pd.testing.assert_frame_equal(df, build_batting(FIXTURE_RAW))


def test_clean_batting_build_batting_blank_statcast_value_is_null(raw_copy: Path) -> None:
    # Savant 값이 빈 칸이면 그 열만 결측이고 나머지 Statcast 열은 그대로다 (실데이터에 있는 경우)
    lines = _savant_lines(raw_copy, 2024)
    index = _savant_line_index(lines, DEVERS)
    header = next(csv.reader([lines[0].lstrip("﻿")]))
    fields = next(csv.reader([lines[index]]))
    fields[header.index("exit_velocity_avg")] = ""
    out = io.StringIO()
    csv.writer(out, lineterminator="").writerow(fields)
    lines[index] = out.getvalue()
    _write_savant_lines(raw_copy, 2024, lines)

    df = build_batting(raw_copy)

    assert check_frame(df, BATTING_SEASONS) == []
    assert df["ev_avg"].dtype == "float64"
    row = _rows(df)[(DEVERS, 2024)]
    assert pd.isna(row["ev_avg"])
    savant = _savant_values(FIXTURE_RAW, 2024)[DEVERS]
    for column in STATCAST_COLUMNS:
        if column != "ev_avg":
            assert row[column] == savant[column], column


@pytest.mark.parametrize("source", ["woba", "home_run", "pa", "last_name, first_name"])
def test_clean_batting_build_batting_savant_duplicate_any_column_conflict_raises(
    raw_copy: Path, source: str
) -> None:
    # E19: 값까지 완전히 같은 중복만 합친다. Statcast 밖의 열(pa·이름)이 달라도 충돌이다
    lines = _savant_lines(raw_copy, 2024)
    header = next(csv.reader([lines[0].lstrip("﻿")]))
    fields = next(csv.reader([lines[_savant_line_index(lines, DEVERS)]]))
    position = header.index(source)
    fields[position] = "Devers, Raffy" if source.startswith("last_name") else "1"
    out = io.StringIO()
    csv.writer(out, lineterminator="").writerow(fields)
    lines.append(out.getvalue())
    _write_savant_lines(raw_copy, 2024, lines)

    with pytest.raises(ValueError, match=str(DEVERS)):
        build_batting(raw_copy)


@pytest.mark.parametrize(
    "missing",
    [raw_statsapi_path(2019, "bat"), raw_savant_path(2026, "batter")],
    ids=["api", "savant"],
)
def test_clean_batting_build_batting_missing_raw_file_raises(raw_copy: Path, missing: Path) -> None:
    # 원본이 없으면 Statcast를 모두 결측으로 채우거나 시즌을 건너뛰지 않고 멈춘다
    (raw_copy / missing).unlink()

    with pytest.raises(FileNotFoundError):
        build_batting(raw_copy)
