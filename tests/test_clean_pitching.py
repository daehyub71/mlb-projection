"""투수 시즌 표(clean/pitching.py) 테스트 (SPEC F3·D8·D9·E3·E7·E19·E20, §6 `pitching_seasons`).

외부 요청은 하지 않는다. `tests/fixtures/raw`를 원본 디렉토리로 그대로 쓰고, 고친 원본이 필요한
테스트는 픽스처를 `tmp_path`로 복사해서 바꾼다. 기대값은 구현을 돌려서 얻지 않고, 픽스처
JSON·CSV를 테스트에서 직접 읽어서 만든다.
"""

import csv
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mlbproj import contracts as c
from mlbproj.clean import pitching
from mlbproj.clean.pitching import build_pitching, ip_to_outs, is_pitcher_split
from mlbproj.interim import read_table, write_table

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RAW = ROOT / "tests" / "fixtures" / "raw"

# 픽스처 선수 (tests/fixtures/SOURCE.md)
NOLA = 605400  # 8시즌 모두 선발
IGLESIAS = 628452  # 마무리, 2022 이적 (numTeams 2)
SKENES = 694973  # 2024 데뷔
COLE = 543037  # 2025 결장
OHTANI = 660271  # 투타 겸업, 투구 포지션 TWP, 2019·2024 투구 없음 (E7)
HERNANDEZ = 571771  # 등판한 야수, 2024~2026 투구 행 포지션 3B (D8)
VENDITTE = 519381  # 양손 투수, Savant 2019·2020에 값까지 같은 2행 (E19)

# 계약 사본: 정제 열 ← API stat 필드 (스텁의 PITCHING_STAT_FIELDS와 같아야 한다)
EXPECTED_STAT_FIELDS: dict[str, str] = {
    "g": "gamesPitched",
    "gs": "gamesStarted",
    "outs": "outs",
    "bf": "battersFaced",
    "er": "earnedRuns",
    "h": "hits",
    "hr": "homeRuns",
    "bb": "baseOnBalls",
    "hbp": "hitByPitch",
    "so": "strikeOuts",
    "sv": "saves",
}
# 계약 사본: Savant 투수 CSV 열 → 정제 열 (스텁의 PITCHING_SAVANT_COLUMNS와 같아야 한다)
EXPECTED_SAVANT_COLUMNS: dict[str, str] = {
    "xwoba": "xwoba_against",
    "whiff_percent": "whiff_pct",
    "hard_hit_percent": "hard_hit_pct",
    "barrel_batted_rate": "barrel_pct",
    "xera": "xera",
}
STATCAST_COLUMNS: tuple[str, ...] = tuple(EXPECTED_SAVANT_COLUMNS.values())

Key = tuple[int, int]  # (player_id, season)


# ---------------------------------------------------------------------------
# 기대값 — 픽스처 파일을 직접 읽는다
# ---------------------------------------------------------------------------


def _read_pitching_doc(raw_dir: Path, season: int) -> dict[str, Any]:
    """API 투구 원본 JSON 문서를 그대로 읽는다."""
    path = raw_dir / c.raw_statsapi_path(season, "pit")
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return doc


def _all_pitching_splits(raw_dir: Path = FIXTURE_RAW) -> list[tuple[int, dict[str, Any]]]:
    """모든 시즌의 투구 split을 (시즌, split)으로 돌려준다 (등판한 야수 포함)."""
    rows: list[tuple[int, dict[str, Any]]] = []
    for season in c.SEASONS:
        for split in _read_pitching_doc(raw_dir, season)["stats"][0]["splits"]:
            rows.append((season, split))
    return rows


def _expected_pitcher_splits(raw_dir: Path = FIXTURE_RAW) -> dict[Key, dict[str, Any]]:
    """투수 역할(포지션 P·TWP) split을 (player_id, season) → split으로 돌려준다."""
    expected: dict[Key, dict[str, Any]] = {}
    for season, split in _all_pitching_splits(raw_dir):
        if split["position"]["abbreviation"] in {"P", "TWP"}:
            key = (int(split["player"]["id"]), season)
            assert key not in expected, f"픽스처에 같은 선수가 두 번: {key}"
            expected[key] = split
    return expected


def _savant_csv_rows(raw_dir: Path, season: int) -> list[dict[str, str]]:
    """Savant 투수 CSV를 표준 csv 모듈로 읽는다 (BOM 제거)."""
    path = raw_dir / c.raw_savant_path(season, "pitcher")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _expected_statcast(raw_dir: Path = FIXTURE_RAW) -> dict[Key, dict[str, float]]:
    """Savant 투수 CSV의 Statcast 값을 (player_id, season) → {정제 열: 값}으로 돌려준다.

    같은 ID가 두 번 나오면 값이 모두 같아야 한다(E19, 픽스처 사실 확인).
    """
    expected: dict[Key, dict[str, float]] = {}
    raw_rows: dict[Key, dict[str, str]] = {}
    for season in c.SEASONS:
        for row in _savant_csv_rows(raw_dir, season):
            key = (int(row["player_id"]), season)
            if key in raw_rows:
                assert raw_rows[key] == row, f"픽스처의 Savant 중복 값이 다르다: {key}"
                continue
            raw_rows[key] = row
            expected[key] = {
                clean: float(row[source]) for source, clean in EXPECTED_SAVANT_COLUMNS.items()
            }
    return expected


def _outs_from_notation(text: str) -> int:
    """테스트 자체의 이닝 표기 해석 ("172.1" → 517). 구현과 독립적으로 기대값을 만든다."""
    whole, _, part = text.partition(".")
    return int(whole) * 3 + int(part or "0")


def _rows_by_key(df: pd.DataFrame) -> dict[Key, dict[Any, Any]]:
    """표를 (player_id, season) → 행 사전으로 바꾼다."""
    return {(int(r["player_id"]), int(r["season"])): r for r in df.to_dict("records")}


# ---------------------------------------------------------------------------
# 복사본 원본 고치기 — 픽스처는 건드리지 않는다
# ---------------------------------------------------------------------------


def _copy_raw(tmp_path: Path) -> Path:
    """픽스처 원본 디렉토리를 tmp_path로 복사한다."""
    raw_dir = tmp_path / "raw"
    shutil.copytree(FIXTURE_RAW, raw_dir)
    return raw_dir


def _edit_split(
    raw_dir: Path, season: int, player_id: int, edit: Callable[[dict[str, Any]], None]
) -> None:
    """복사본 투구 원본에서 선수 한 명의 split을 고친다."""
    path = raw_dir / c.raw_statsapi_path(season, "pit")
    doc = _read_pitching_doc(raw_dir, season)
    matches = [s for s in doc["stats"][0]["splits"] if s["player"]["id"] == player_id]
    assert len(matches) == 1, f"{season} 투구 원본에 {player_id}가 한 번 있어야 한다"
    edit(matches[0])
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def _savant_lines(raw_dir: Path, season: int) -> tuple[Path, list[str]]:
    """복사본 Savant 투수 CSV의 경로와 줄 목록(줄바꿈 포함, BOM은 첫 줄에 남는다)."""
    path = raw_dir / c.raw_savant_path(season, "pitcher")
    return path, path.read_text(encoding="utf-8").splitlines(keepends=True)


def _player_line_indexes(lines: list[str], player_id: int) -> list[int]:
    """CSV 줄 목록에서 선수 ID 필드가 player_id인 줄 번호들."""
    indexes: list[int] = []
    for index, line in enumerate(lines[1:], start=1):
        fields = next(csv.reader([line]))
        if fields and fields[1] == str(player_id):
            indexes.append(index)
    return indexes


def _replace_field(line: str, column: int, value: str) -> str:
    """CSV 한 줄의 column번째 필드를 value로 바꾼다 (나머지 표기는 그대로 둔다)."""
    fields = next(csv.reader([line]))
    fields[column] = value
    # 원본처럼 쉼표가 든 이름 필드만 따옴표로 감싼다
    return ",".join(f'"{f}"' if "," in f else f for f in fields) + "\n"


def _savant_column_index(name: str) -> int:
    """Savant 투수 CSV 머리줄에서 열 번호."""
    return list(c.SAVANT_PITCHER_COLUMNS).index(name)


# ---------------------------------------------------------------------------
# 계약 상수
# ---------------------------------------------------------------------------


def test_clean_pitching_mappings_match_contract() -> None:
    """열 대응표가 계약 그대로이고, 대상 열은 정제 표에, 원본 열은 원본 필수 필드에 있다."""
    assert dict(pitching.PITCHING_STAT_FIELDS) == EXPECTED_STAT_FIELDS
    assert dict(pitching.PITCHING_SAVANT_COLUMNS) == EXPECTED_SAVANT_COLUMNS
    columns = c.PITCHING_SEASONS.column_names
    assert set(EXPECTED_STAT_FIELDS) <= set(columns)
    assert set(EXPECTED_STAT_FIELDS.values()) <= set(c.STATSAPI_PITCHING_STAT_FIELDS)
    assert set(EXPECTED_SAVANT_COLUMNS.values()) <= set(columns)
    assert set(EXPECTED_SAVANT_COLUMNS) <= set(c.SAVANT_PITCHER_COLUMNS)


# ---------------------------------------------------------------------------
# ip_to_outs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ip", "expected"),
    [
        ("172.1", 517),
        ("0.2", 2),
        ("5", 15),
        ("0.0", 0),
        ("166.0", 498),
        ("1.2", 5),
        (172.1, 517),
        (0.2, 2),
        (172.2, 518),
        (5, 15),
        (5.0, 15),
        (np.int64(5), 15),
        (np.float64(172.1), 517),
    ],
)
def test_clean_pitching_ip_to_outs_converts_notation(ip: Any, expected: int) -> None:
    """이닝 표기의 소수 자리는 아웃 1·2개다. 문자열과 숫자를 같게 바꾼다."""
    result = ip_to_outs(ip)
    assert result == expected
    assert isinstance(result, int) and not isinstance(result, bool)


@pytest.mark.parametrize(
    "ip",
    [
        "1.3",
        "1.5",
        "-1.0",
        "-0.1",
        "abc",
        "",
        "1.2.1",
        None,
        1.3,
        -1.0,
        -3,
        float("nan"),
        float("inf"),
        True,
        [5],
    ],
)
def test_clean_pitching_ip_to_outs_rejects_invalid(ip: Any) -> None:
    """소수 부분이 0·1·2가 아니거나, 음수이거나, 숫자가 아니면 ValueError (bool은 이닝이 아니다)."""
    with pytest.raises(ValueError):
        ip_to_outs(ip)


def test_clean_pitching_ip_to_outs_matches_fixture_outs() -> None:
    """픽스처의 모든 투구 행(등판한 야수 포함)에서 inningsPitched → outs가 API outs와 같다 (E20)."""
    splits = _all_pitching_splits()
    assert splits
    for season, split in splits:
        stat = split["stat"]
        assert _outs_from_notation(stat["inningsPitched"]) == stat["outs"]  # 픽스처 사실
        assert ip_to_outs(stat["inningsPitched"]) == stat["outs"], (season, split["player"]["id"])


# ---------------------------------------------------------------------------
# is_pitcher_split
# ---------------------------------------------------------------------------


def test_clean_pitching_is_pitcher_split_pitcher_positions() -> None:
    """포지션 P(투수)와 TWP(오타니) 투구 행은 투수 역할이다."""
    expected = _expected_pitcher_splits()
    nola = expected[(NOLA, 2022)]
    assert nola["position"]["abbreviation"] == "P"
    assert is_pitcher_split(nola) is True
    ohtani_rows = [split for (pid, _), split in expected.items() if pid == OHTANI]
    assert ohtani_rows
    for split in ohtani_rows:
        assert split["position"]["abbreviation"] == "TWP"
        assert is_pitcher_split(split) is True


def test_clean_pitching_is_pitcher_split_position_player() -> None:
    """등판한 야수(Hernández, 2024~2026 투구 행 포지션 3B)는 투수 역할이 아니다 (D8)."""
    rows = [
        (season, split)
        for season, split in _all_pitching_splits()
        if split["player"]["id"] == HERNANDEZ
    ]
    assert [season for season, _ in rows] == [2024, 2025, 2026]
    for _, split in rows:
        assert split["position"]["abbreviation"] == "3B"
        assert is_pitcher_split(split) is False


@pytest.mark.parametrize(
    ("abbreviation", "expected"),
    [
        ("P", True),
        ("TWP", True),
        ("C", False),
        ("1B", False),
        ("SS", False),
        ("LF", False),
        ("DH", False),
        ("X", False),
        ("OF", False),
    ],
)
def test_clean_pitching_is_pitcher_split_by_abbreviation(abbreviation: str, expected: bool) -> None:
    """판정은 position.abbreviation이 PITCHER_POSITIONS(P·TWP)에 드는지로 한다."""
    split = json.loads(json.dumps(_expected_pitcher_splits()[(NOLA, 2022)]))
    split["position"] = {"abbreviation": abbreviation}
    assert is_pitcher_split(split) is expected


# ---------------------------------------------------------------------------
# build_pitching — 픽스처 원본 그대로
# ---------------------------------------------------------------------------


def test_clean_pitching_build_passes_contract() -> None:
    """결과 표가 PITCHING_SEASONS 계약(열 순서·형식·결측·키 중복)을 지킨다."""
    df = build_pitching(FIXTURE_RAW)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == c.PITCHING_SEASONS.column_names
    assert c.check_frame(df, c.PITCHING_SEASONS) == []


def test_clean_pitching_build_sorted_by_season_then_player() -> None:
    """행은 season, player_id 순으로 정렬되어 있다."""
    df = build_pitching(FIXTURE_RAW)
    keys = [(int(s), int(p)) for s, p in zip(df["season"], df["player_id"], strict=True)]
    assert keys == sorted(keys)
    assert keys == sorted((season, pid) for pid, season in _expected_pitcher_splits())


def test_clean_pitching_build_rows_are_pitcher_splits() -> None:
    """행은 모든 시즌의 P·TWP 투구 split과 하나씩 짝이 맞고, league는 MLB다."""
    df = build_pitching(FIXTURE_RAW)
    expected = _expected_pitcher_splits()
    assert len(df) == len(expected) == 39
    assert set(_rows_by_key(df)) == set(expected)
    assert set(df["season"]) == set(c.SEASONS)
    assert (df["league"] == "MLB").all()


def test_clean_pitching_build_excludes_position_player() -> None:
    """등판한 야수는 Savant 2025 투수 목록에 있어도 표에 나오지 않는다 (D8·E2)."""
    assert HERNANDEZ in {int(r["player_id"]) for r in _savant_csv_rows(FIXTURE_RAW, 2025)}
    df = build_pitching(FIXTURE_RAW)
    assert HERNANDEZ not in set(df["player_id"])


def test_clean_pitching_build_two_way_player_rows() -> None:
    """오타니는 TWP 투구 행이 있는 2020~2023·2025·2026에만 나온다 (2019·2024 투구 없음, E7)."""
    df = build_pitching(FIXTURE_RAW)
    seasons = sorted(int(s) for s in df.loc[df["player_id"] == OHTANI, "season"])
    assert seasons == [2020, 2021, 2022, 2023, 2025, 2026]


def test_clean_pitching_build_api_columns() -> None:
    """age·team_name·n_teams와 셈 열은 API split 값 그대로다 (셈 열은 정수)."""
    df = build_pitching(FIXTURE_RAW)
    rows = _rows_by_key(df)
    for key, split in _expected_pitcher_splits().items():
        row = rows[key]
        stat = split["stat"]
        assert int(row["age"]) == stat["age"], key
        assert row["team_name"] == split["team"]["name"], key
        assert int(row["n_teams"]) == split["numTeams"], key
        for column, field in EXPECTED_STAT_FIELDS.items():
            assert int(row[column]) == stat[field], (key, column)
    for column in EXPECTED_STAT_FIELDS:
        assert pd.api.types.is_integer_dtype(df[column].dtype), column


def test_clean_pitching_build_gs_from_api() -> None:
    """선발 등판은 API gamesStarted에서 온다 (Savant p_gs는 비어 있다, E3)."""
    df = build_pitching(FIXTURE_RAW)
    rows = _rows_by_key(df)
    assert int(rows[(NOLA, 2019)]["gs"]) == 34
    assert int(rows[(OHTANI, 2021)]["gs"]) == 23
    assert int(rows[(SKENES, 2024)]["gs"]) == 23
    iglesias = df.loc[df["player_id"] == IGLESIAS, "gs"]
    assert len(iglesias) == 8
    assert (iglesias == 0).all()
    for key, split in _expected_pitcher_splits().items():
        assert int(rows[key]["gs"]) == split["stat"]["gamesStarted"], key


def test_clean_pitching_build_outs_match_innings() -> None:
    """outs는 API outs와 같고, inningsPitched를 아웃으로 바꾼 값과도 같다."""
    df = build_pitching(FIXTURE_RAW)
    rows = _rows_by_key(df)
    for key, split in _expected_pitcher_splits().items():
        stat = split["stat"]
        outs = int(rows[key]["outs"])
        assert outs == stat["outs"], key
        assert outs == _outs_from_notation(stat["inningsPitched"]), key
        assert outs == ip_to_outs(stat["inningsPitched"]), key
    assert int(rows[(NOLA, 2019)]["outs"]) == 607  # "202.1"
    assert int(rows[(VENDITTE, 2020)]["outs"]) == 13  # "4.1"


def test_clean_pitching_build_short_season_scaling() -> None:
    """2020 행만 is_short_season이고 outs_adj = outs × 162/60, 그 밖엔 outs_adj = outs (D9)."""
    df = build_pitching(FIXTURE_RAW)
    short = df[df["season"] == 2020]
    regular = df[df["season"] != 2020]
    assert len(short) == 6
    assert short["is_short_season"].map(bool).all()
    assert not regular["is_short_season"].map(bool).any()
    for outs, outs_adj in zip(short["outs"], short["outs_adj"], strict=True):
        assert float(outs_adj) == pytest.approx(int(outs) * 162 / 60, rel=1e-12)
    for outs, outs_adj in zip(regular["outs"], regular["outs_adj"], strict=True):
        assert float(outs_adj) == float(outs)
    rows = _rows_by_key(df)
    assert float(rows[(COLE, 2020)]["outs_adj"]) == pytest.approx(219 * 162 / 60, rel=1e-12)
    assert float(rows[(COLE, 2019)]["outs_adj"]) == 637.0


def test_clean_pitching_build_traded_closer() -> None:
    """이적한 마무리(Iglesias 2022)는 합산 1행, n_teams 2, API가 준 팀 이름, 세이브 그대로다."""
    df = build_pitching(FIXTURE_RAW)
    iglesias = df[(df["player_id"] == IGLESIAS) & (df["season"] == 2022)]
    assert len(iglesias) == 1
    row = iglesias.iloc[0]
    assert int(row["n_teams"]) == 2
    assert row["team_name"] == "Atlanta Braves"
    assert int(row["sv"]) == 17
    assert int(row["g"]) == 67
    assert int(row["outs"]) == 186


def test_clean_pitching_build_statcast_columns() -> None:
    """Statcast 열은 Savant 열 이름을 바꿔 붙인 것이고 값은 CSV 그대로다."""
    df = build_pitching(FIXTURE_RAW)
    rows = _rows_by_key(df)
    statcast = _expected_statcast()
    for key in _expected_pitcher_splits():
        # 픽스처의 투수 행은 모두 Savant에 있다
        assert key in statcast, key
        for column, value in statcast[key].items():
            assert float(rows[key][column]) == pytest.approx(value, rel=1e-12), (key, column)
    nola = rows[(NOLA, 2019)]
    assert float(nola["xwoba_against"]) == pytest.approx(0.310)
    assert float(nola["whiff_pct"]) == pytest.approx(26.7)
    assert float(nola["hard_hit_pct"]) == pytest.approx(39.5)
    assert float(nola["barrel_pct"]) == pytest.approx(7.3)
    assert float(nola["xera"]) == pytest.approx(4.23)
    for column in STATCAST_COLUMNS:
        assert pd.api.types.is_float_dtype(df[column].dtype), column


def test_clean_pitching_build_merges_identical_savant_duplicates() -> None:
    """Savant에 값까지 같은 2행인 Venditte는 2019·2020 각각 한 행이고 그 값을 갖는다 (E19)."""
    for season in (2019, 2020):
        ids = [int(r["player_id"]) for r in _savant_csv_rows(FIXTURE_RAW, season)]
        assert ids.count(VENDITTE) == 2  # 픽스처 사실
    df = build_pitching(FIXTURE_RAW)
    venditte = df[df["player_id"] == VENDITTE]
    assert sorted(int(s) for s in venditte["season"]) == [2019, 2020]
    rows = _rows_by_key(df)
    expected = {
        2019: {
            "xwoba_against": 0.394,
            "whiff_pct": 17.9,
            "hard_hit_pct": 8.3,
            "barrel_pct": 10.0,
            "xera": 7.37,
        },
        2020: {
            "xwoba_against": 0.233,
            "whiff_pct": 43.5,
            "hard_hit_pct": 25.0,
            "barrel_pct": 0.0,
            "xera": 2.26,
        },
    }
    for season, values in expected.items():
        row = rows[(VENDITTE, season)]
        for column, value in values.items():
            assert float(row[column]) == pytest.approx(value), (season, column)


# ---------------------------------------------------------------------------
# build_pitching — 고친 복사본
# ---------------------------------------------------------------------------


def test_clean_pitching_build_missing_savant_row_is_nan(tmp_path: Path) -> None:
    """Savant에 없는 투수 행은 Statcast 열만 결측이고 행과 API 열은 그대로다 (F3)."""
    raw_dir = _copy_raw(tmp_path)
    path, lines = _savant_lines(raw_dir, 2024)
    indexes = _player_line_indexes(lines, SKENES)
    assert len(indexes) == 1
    del lines[indexes[0]]
    path.write_text("".join(lines), encoding="utf-8")

    df = build_pitching(raw_dir)
    assert c.check_frame(df, c.PITCHING_SEASONS) == []
    rows = _rows_by_key(df)
    skenes = rows[(SKENES, 2024)]
    for column in STATCAST_COLUMNS:
        assert pd.isna(skenes[column]), column
    assert int(skenes["outs"]) == 399
    assert int(skenes["gs"]) == 23
    # 같은 시즌 다른 투수는 값이 그대로 붙는다
    assert float(rows[(NOLA, 2024)]["xera"]) == pytest.approx(3.75)
    assert float(rows[(SKENES, 2025)]["xera"]) == pytest.approx(2.65)
    assert len(df) == len(_expected_pitcher_splits())


def test_clean_pitching_build_missing_team_is_nan(tmp_path: Path) -> None:
    """split에 team이 없으면 team_name은 결측이다."""
    raw_dir = _copy_raw(tmp_path)
    _edit_split(raw_dir, 2023, NOLA, lambda split: split.pop("team"))

    df = build_pitching(raw_dir)
    assert c.check_frame(df, c.PITCHING_SEASONS) == []
    rows = _rows_by_key(df)
    assert pd.isna(rows[(NOLA, 2023)]["team_name"])
    assert rows[(NOLA, 2022)]["team_name"] == "Philadelphia Phillies"


def test_clean_pitching_build_integer_savant_column_stays_float(tmp_path: Path) -> None:
    """한 시즌 Savant 열 값이 모두 정수로 적혀 있어도 Statcast 열은 실수형이다."""
    raw_dir = _copy_raw(tmp_path)
    path, lines = _savant_lines(raw_dir, 2024)
    column = _savant_column_index("barrel_batted_rate")
    lines = [lines[0]] + [_replace_field(line, column, "7") for line in lines[1:]]
    path.write_text("".join(lines), encoding="utf-8")

    df = build_pitching(raw_dir)
    assert c.check_frame(df, c.PITCHING_SEASONS) == []
    assert pd.api.types.is_float_dtype(df["barrel_pct"].dtype)
    season_2024 = df.loc[df["season"] == 2024, "barrel_pct"]
    assert len(season_2024) == 4
    assert (season_2024 == 7.0).all()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("inningsPitched", "205.1", id="innings-one-out-more"),
        pytest.param("outs", 616, id="outs-one-more"),
        pytest.param("inningsPitched", "205.3", id="innings-bad-notation"),
    ],
)
def test_clean_pitching_build_rejects_innings_outs_mismatch(
    tmp_path: Path, field: str, value: Any
) -> None:
    """inningsPitched를 아웃으로 바꾼 값이 API outs와 다르면 ValueError (Nola 2022 "205.0" = 615)."""
    raw_dir = _copy_raw(tmp_path)
    build_pitching(raw_dir)  # 고치기 전 복사본은 통과한다

    def edit(split: dict[str, Any]) -> None:
        assert split["stat"]["inningsPitched"] == "205.0"
        assert split["stat"]["outs"] == 615
        split["stat"][field] = value

    _edit_split(raw_dir, 2022, NOLA, edit)
    # 어느 시즌·선수 행인지 메시지에 적는다
    with pytest.raises(ValueError, match=rf"2022\b.*\b{NOLA}"):
        build_pitching(raw_dir)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        pytest.param("xera", "7.40", id="mapped-column-xera"),
        pytest.param("whiff_percent", "18.0", id="mapped-column-whiff"),
        pytest.param("pa", "20", id="unmapped-column-pa"),
    ],
)
def test_clean_pitching_build_rejects_conflicting_savant_duplicate(
    tmp_path: Path, column: str, value: str
) -> None:
    """Savant 투수 중복 행의 값이 하나라도 다르면 ValueError (E19: 같은 값 중복만 합친다)."""
    raw_dir = _copy_raw(tmp_path)
    build_pitching(raw_dir)  # 고치기 전 복사본은 통과한다

    path, lines = _savant_lines(raw_dir, 2019)
    indexes = _player_line_indexes(lines, VENDITTE)
    assert len(indexes) == 2
    second = indexes[1]
    changed = _replace_field(lines[second], _savant_column_index(column), value)
    assert changed != lines[second]
    lines[second] = changed
    path.write_text("".join(lines), encoding="utf-8")

    with pytest.raises(ValueError):
        build_pitching(raw_dir)


def test_clean_pitching_build_merges_any_identical_duplicate(tmp_path: Path) -> None:
    """값이 같은 중복은 Venditte가 아니어도 하나로 합친다 (Cole 2022 줄을 그대로 한 번 더 넣는다)."""
    raw_dir = _copy_raw(tmp_path)
    path, lines = _savant_lines(raw_dir, 2022)
    indexes = _player_line_indexes(lines, COLE)
    assert len(indexes) == 1
    lines.append(lines[indexes[0]])
    path.write_text("".join(lines), encoding="utf-8")

    df = build_pitching(raw_dir)
    assert c.check_frame(df, c.PITCHING_SEASONS) == []
    cole = df[(df["player_id"] == COLE) & (df["season"] == 2022)]
    assert len(cole) == 1
    row = cole.iloc[0]
    assert float(row["xwoba_against"]) == pytest.approx(0.289)
    assert float(row["whiff_pct"]) == pytest.approx(34.1)
    assert float(row["hard_hit_pct"]) == pytest.approx(40.2)
    assert float(row["barrel_pct"]) == pytest.approx(9.6)
    assert float(row["xera"]) == pytest.approx(3.37)
    assert len(df) == len(_expected_pitcher_splits())


def _set_stat(field: str, value: Any) -> Callable[[dict[str, Any]], None]:
    """split.stat의 field를 value로 바꾸는 편집 함수."""

    def edit(split: dict[str, Any]) -> None:
        assert field in split["stat"]
        split["stat"][field] = value

    return edit


def _set_top(path: tuple[str, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    """split의 중첩 필드(path)를 value로 바꾸는 편집 함수."""

    def edit(split: dict[str, Any]) -> None:
        target = split
        for key in path[:-1]:
            target = target[key]
        assert path[-1] in target
        target[path[-1]] = value

    return edit


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(_set_stat("gamesStarted", 32.5), id="count-float-truncates"),
        pytest.param(_set_stat("saves", "0"), id="count-string"),
        pytest.param(_set_stat("hitByPitch", True), id="count-bool"),
        pytest.param(_set_stat("earnedRuns", None), id="count-null"),
        pytest.param(_set_stat("age", 29.5), id="age-float"),
        pytest.param(_set_top(("numTeams",), 1.5), id="num-teams-float"),
        pytest.param(_set_top(("player", "id"), str(NOLA)), id="player-id-string"),
        pytest.param(_set_top(("team", "name"), 143), id="team-name-not-string"),
    ],
)
def test_clean_pitching_build_rejects_non_integer_fields(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None]
) -> None:
    """정수 필드가 정수가 아니면 정수형으로 조용히 바꾸지 않고 ValueError (32.5 → 32, "0" → 0 금지)."""
    raw_dir = _copy_raw(tmp_path)
    _edit_split(raw_dir, 2022, NOLA, edit)
    with pytest.raises(ValueError, match="2022"):
        build_pitching(raw_dir)


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda split: split["stat"].pop("age"), id="age-missing"),
        pytest.param(_set_stat("age", None), id="age-null"),
    ],
)
def test_clean_pitching_build_missing_age_is_na(
    tmp_path: Path, edit: Callable[[dict[str, Any]], None]
) -> None:
    """stat.age가 없거나 null이면 age만 결측이고 열은 결측 허용 정수형(Int64)이다."""
    raw_dir = _copy_raw(tmp_path)
    _edit_split(raw_dir, 2022, NOLA, edit)

    df = build_pitching(raw_dir)
    assert c.check_frame(df, c.PITCHING_SEASONS) == []
    assert df["age"].dtype == "Int64"
    rows = _rows_by_key(df)
    assert pd.isna(rows[(NOLA, 2022)]["age"])
    assert int(rows[(NOLA, 2022)]["outs"]) == 615
    assert int(rows[(NOLA, 2023)]["age"]) == _expected_pitcher_splits()[(NOLA, 2023)]["stat"]["age"]


def test_clean_pitching_build_rejects_duplicate_api_split(tmp_path: Path) -> None:
    """API 투구 원본에 같은 선수가 두 번 나오면 값이 같아도 ValueError (E1, F2-2)."""
    raw_dir = _copy_raw(tmp_path)
    path = raw_dir / c.raw_statsapi_path(2022, "pit")
    doc = _read_pitching_doc(raw_dir, 2022)
    splits = doc["stats"][0]["splits"]
    nola = [s for s in splits if s["player"]["id"] == NOLA]
    assert len(nola) == 1
    splits.append(json.loads(json.dumps(nola[0])))
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    with pytest.raises(ValueError, match=str(NOLA)):
        build_pitching(raw_dir)


def test_clean_pitching_build_blank_statcast_cell_is_nan(tmp_path: Path) -> None:
    """Savant 행이 있어도 칸이 비어 있으면 그 열만 결측이고 나머지 Statcast 값은 그대로다."""
    raw_dir = _copy_raw(tmp_path)
    path, lines = _savant_lines(raw_dir, 2023)
    indexes = _player_line_indexes(lines, NOLA)
    assert len(indexes) == 1
    lines[indexes[0]] = _replace_field(lines[indexes[0]], _savant_column_index("xera"), "")
    path.write_text("".join(lines), encoding="utf-8")

    df = build_pitching(raw_dir)
    assert c.check_frame(df, c.PITCHING_SEASONS) == []
    nola = _rows_by_key(df)[(NOLA, 2023)]
    assert pd.isna(nola["xera"])
    expected = _expected_statcast()[(NOLA, 2023)]
    for column in ("xwoba_against", "whiff_pct", "hard_hit_pct", "barrel_pct"):
        assert float(nola[column]) == pytest.approx(expected[column], rel=1e-12), column


def test_clean_pitching_build_rejects_non_numeric_statcast(tmp_path: Path) -> None:
    """Statcast 칸에 숫자가 아닌 값이 있으면 결측으로 바꾸지 않고 ValueError."""
    raw_dir = _copy_raw(tmp_path)
    path, lines = _savant_lines(raw_dir, 2023)
    index = _player_line_indexes(lines, NOLA)[0]
    lines[index] = _replace_field(lines[index], _savant_column_index("whiff_percent"), "--")
    path.write_text("".join(lines), encoding="utf-8")

    with pytest.raises(ValueError, match="whiff_percent"):
        build_pitching(raw_dir)


@pytest.mark.parametrize(
    "rel",
    [
        pytest.param(c.raw_savant_path(2023, "pitcher"), id="savant-pitcher"),
        pytest.param(c.raw_statsapi_path(2023, "pit"), id="statsapi-pit"),
    ],
)
def test_clean_pitching_build_missing_raw_file_raises(tmp_path: Path, rel: Path) -> None:
    """한 시즌 원본이 없으면 그 시즌을 결측·미출전으로 두지 않고 FileNotFoundError (F2)."""
    raw_dir = _copy_raw(tmp_path)
    (raw_dir / rel).unlink()
    with pytest.raises(FileNotFoundError):
        build_pitching(raw_dir)


def test_clean_pitching_build_independent_of_input_order(tmp_path: Path) -> None:
    """원본의 행 순서가 달라도 결과 표는 같다 (정렬·RangeIndex, N1)."""
    expected = build_pitching(FIXTURE_RAW)
    assert expected.index.equals(pd.RangeIndex(len(expected)))

    raw_dir = _copy_raw(tmp_path)
    for season in c.SEASONS:
        path = raw_dir / c.raw_statsapi_path(season, "pit")
        doc = _read_pitching_doc(raw_dir, season)
        doc["stats"][0]["splits"].reverse()
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        csv_path, lines = _savant_lines(raw_dir, season)
        csv_path.write_text(lines[0] + "".join(reversed(lines[1:])), encoding="utf-8")

    pd.testing.assert_frame_equal(build_pitching(raw_dir), expected)


def test_clean_pitching_build_seasons_do_not_leak(tmp_path: Path) -> None:
    """2026 원본을 바꿔도 2019~2025 행은 그대로다 — 행은 자기 시즌 원본에서만 온다 (N2)."""
    before = build_pitching(FIXTURE_RAW)
    raw_dir = _copy_raw(tmp_path)

    def edit(split: dict[str, Any]) -> None:
        split["stat"]["gamesStarted"] += 1
        split["stat"]["saves"] += 3

    _edit_split(raw_dir, 2026, NOLA, edit)
    path, lines = _savant_lines(raw_dir, 2026)
    index = _player_line_indexes(lines, NOLA)[0]
    lines[index] = _replace_field(lines[index], _savant_column_index("xera"), "9.99")
    path.write_text("".join(lines), encoding="utf-8")

    after = build_pitching(raw_dir)
    earlier = before["season"] < 2026
    pd.testing.assert_frame_equal(
        after[after["season"] < 2026].reset_index(drop=True),
        before[earlier].reset_index(drop=True),
    )
    nola_2026 = _rows_by_key(after)[(NOLA, 2026)]
    assert float(nola_2026["xera"]) == pytest.approx(9.99)
    assert int(nola_2026["sv"]) == _expected_pitcher_splits()[(NOLA, 2026)]["stat"]["saves"] + 3


def test_clean_pitching_build_round_trips_through_interim(tmp_path: Path) -> None:
    """정제 표를 CSV로 쓰고 읽어도 값·형식이 같다 (age Int64, 정수 int64, 실수 float64 — 형식 변화 없음)."""
    df = build_pitching(FIXTURE_RAW)
    path = c.interim_path(c.PITCHING_SEASONS, tmp_path)
    write_table(df, c.PITCHING_SEASONS, path)
    back = read_table(c.PITCHING_SEASONS, path)
    keys = list(c.PITCHING_SEASONS.keys)
    expected = df.sort_values(keys, kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(back, expected)
