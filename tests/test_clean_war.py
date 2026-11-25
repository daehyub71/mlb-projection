"""WAR 정제 표(clean/war.py, SPEC v0.5 F3·D8·D33, contracts `WAR_SEASONS`) 테스트.

기준 상태는 `tests/fixtures/raw`(실제 응답에서 14명만 잘라 낸 원본, SOURCE.md)다. 역할 행은 기존 정제 함수
(`clean.batting.build_batting` · `clean.pitching.build_pitching`)로 만든 시즌 표가 정한다.
기댓값은 구현을 돌려서가 아니라 세이버메트릭스 픽스처 JSON을 직접 읽어 만든다.
원본을 바꾸는 테스트는 tmp_path에 복사본을 만들어 고친다(픽스처 자체는 건드리지 않는다).
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mlbproj.clean.batting import build_batting
from mlbproj.clean.pitching import build_pitching
from mlbproj.clean.war import build_war_seasons
from mlbproj.contracts import (
    LEAGUE,
    ROLES,
    SABER_COLUMN_MAP,
    SEASONS,
    SHORT_SEASON,
    WAR_SEASONS,
    Role,
    check_frame,
    raw_saber_path,
)

FIXTURE_RAW = Path(__file__).parent / "fixtures" / "raw"

OHTANI = 660271
DEVERS = 646240
NOLA = 605400
WAINWRIGHT = 425794
VENDITTE = 519381
IGLESIAS = 628452
COLE = 543037
SKENES = 694973
HERNANDEZ = 571771  # 등판한 야수 — 투구 세이버메트릭스 행은 있지만 투수 표에는 없다
# 투수의 타격 행(포지션 P)만 있는 선수 — 타격 세이버메트릭스 행은 있지만 타자 표에는 없다
PITCHERS = (NOLA, WAINWRIGHT, VENDITTE, IGLESIAS, COLE)

KEYS = ["player_id", "season", "league", "role"]
BAT_COLUMNS = tuple(col for _, col in SABER_COLUMN_MAP["bat"] if col != "war")
PIT_COLUMNS = tuple(col for _, col in SABER_COLUMN_MAP["pit"] if col != "war")

Key = tuple[int, int, str]  # (player_id, season, role)


# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _season_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """픽스처 원본으로 만든 정제 타자·투수 시즌 표 (M2 정제 함수)."""
    return build_batting(FIXTURE_RAW), build_pitching(FIXTURE_RAW)


@pytest.fixture
def batting(_season_tables: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """테스트마다 새 사본 (구현이 입력을 고쳐도 다른 테스트에 번지지 않게)."""
    return _season_tables[0].copy()


@pytest.fixture
def pitching(_season_tables: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return _season_tables[1].copy()


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """테스트 하나가 고쳐 쓰는 픽스처 복사본."""
    dst = tmp_path / "raw"
    shutil.copytree(FIXTURE_RAW, dst)
    return dst


def _load(path: Path) -> Any:
    return json.loads(path.read_bytes().decode("utf-8"))


def _saber_stats(raw_dir: Path, season: int, role: Role) -> dict[int, dict[str, Any]]:
    """세이버메트릭스 원본의 선수 ID → stat (픽스처 JSON을 직접 읽는다)."""
    doc = _load(raw_dir / raw_saber_path(season, role))
    out: dict[int, dict[str, Any]] = {}
    for split in doc["stats"][0]["splits"]:
        pid = split["player"]["id"]
        assert pid not in out
        out[pid] = split["stat"]
    return out


def _edit_saber(
    raw_dir: Path, season: int, role: Role, edit: Callable[[list[dict[str, Any]]], object]
) -> None:
    """세이버메트릭스 splits를 edit(splits)로 제자리에서 고치고 totalSplits를 맞춘다."""
    path = raw_dir / raw_saber_path(season, role)
    doc = _load(path)
    block = doc["stats"][0]
    edit(block["splits"])
    block["totalSplits"] = len(block["splits"])
    path.write_bytes((json.dumps(doc, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


def _keys(df: pd.DataFrame) -> list[Key]:
    return [
        (int(pid), int(season), str(role))
        for pid, season, role in zip(df["player_id"], df["season"], df["role"], strict=True)
    ]


def _role_keys(batting: pd.DataFrame, pitching: pd.DataFrame) -> set[Key]:
    bat = {
        (int(p), int(s), "bat")
        for p, s in zip(batting["player_id"], batting["season"], strict=True)
    }
    pit = {
        (int(p), int(s), "pit")
        for p, s in zip(pitching["player_id"], pitching["season"], strict=True)
    }
    return bat | pit


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): v for k, v in rec.items()} for rec in df.to_dict("records")]


def _with_extra_row(table: pd.DataFrame, player_id: int, season: int) -> pd.DataFrame:
    """시즌 표의 첫 행을 복사해 (player_id, season)만 바꾼 행을 더한다 (형식은 그대로)."""
    extra = table.iloc[[0]].copy()
    extra["player_id"] = player_id
    extra["season"] = season
    return pd.concat([table, extra], ignore_index=True)


# ---------------------------------------------------------------------------
# 표 모양 · 계약
# ---------------------------------------------------------------------------


def test_clean_war_columns_follow_contract(batting: pd.DataFrame, pitching: pd.DataFrame) -> None:
    out = build_war_seasons(FIXTURE_RAW, SEASONS, batting, pitching)
    assert list(out.columns) == WAR_SEASONS.column_names
    assert check_frame(out, WAR_SEASONS) == []


def test_clean_war_rows_are_role_rows_of_season_tables(
    batting: pd.DataFrame, pitching: pd.DataFrame
) -> None:
    # 역할 행(D8)은 정제 시즌 표가 정한다 — 타자 표의 각 행 → bat 한 줄, 투수 표의 각 행 → pit 한 줄
    out = build_war_seasons(FIXTURE_RAW, SEASONS, batting, pitching)
    keys = _keys(out)
    assert len(keys) == len(set(keys))
    assert set(keys) == _role_keys(batting, pitching)
    assert len(out) == len(batting) + len(pitching)
    assert set(out["role"]) == set(ROLES)


def test_clean_war_league_is_contract_league(batting: pd.DataFrame, pitching: pd.DataFrame) -> None:
    # 세이버메트릭스의 league(아메리칸·내셔널 리그)가 아니라 계약의 LEAGUE
    out = build_war_seasons(FIXTURE_RAW, SEASONS, batting, pitching)
    assert set(out["league"]) == {LEAGUE}


def test_clean_war_sorted_by_keys(batting: pd.DataFrame, pitching: pd.DataFrame) -> None:
    out = build_war_seasons(FIXTURE_RAW, SEASONS, batting, pitching)
    rows = list(zip(*(out[k] for k in KEYS), strict=True))
    assert rows == sorted(rows)
    # 선수 ID가 첫 키다 — 오타니의 행은 2019 bat부터 2026 pit까지 붙어 있다
    ohtani = [i for i, pid in enumerate(out["player_id"]) if pid == OHTANI]
    assert ohtani == list(range(ohtani[0], ohtani[0] + len(ohtani)))


def test_clean_war_war_is_never_missing(batting: pd.DataFrame, pitching: pd.DataFrame) -> None:
    out = build_war_seasons(FIXTURE_RAW, SEASONS, batting, pitching)
    assert not out["war"].isna().any()
    assert all(math.isfinite(float(v)) for v in out["war"])


# ---------------------------------------------------------------------------
# 값 — SABER_COLUMN_MAP 그대로 (환산·반올림 없음)
# ---------------------------------------------------------------------------


def test_clean_war_batting_values_follow_column_map(
    batting: pd.DataFrame, pitching: pd.DataFrame
) -> None:
    out = build_war_seasons(FIXTURE_RAW, SEASONS, batting, pitching)
    bat = out[out["role"] == "bat"]
    assert len(bat) == len(batting)
    stats = {season: _saber_stats(FIXTURE_RAW, season, "bat") for season in SEASONS}
    for row in _records(bat):
        stat = stats[row["season"]][row["player_id"]]
        for src, col in SABER_COLUMN_MAP["bat"]:
            assert row[col] == pytest.approx(stat[src], abs=1e-12), (row["player_id"], col)
        assert pd.isna(row["war_ra9"])  # 투수 열은 결측


def test_clean_war_pitching_values_follow_column_map(
    batting: pd.DataFrame, pitching: pd.DataFrame
) -> None:
    out = build_war_seasons(FIXTURE_RAW, SEASONS, batting, pitching)
    pit = out[out["role"] == "pit"]
    assert len(pit) == len(pitching)
    stats = {season: _saber_stats(FIXTURE_RAW, season, "pit") for season in SEASONS}
    for row in _records(pit):
        stat = stats[row["season"]][row["player_id"]]
        for src, col in SABER_COLUMN_MAP["pit"]:
            assert row[col] == pytest.approx(stat[src], abs=1e-12), (row["player_id"], col)
        for col in BAT_COLUMNS:
            assert pd.isna(row[col]), (row["player_id"], col)  # 타자 구성 열은 결측


def test_clean_war_role_specific_columns_are_filled(
    batting: pd.DataFrame, pitching: pd.DataFrame
) -> None:
    # 픽스처의 모든 행에 구성 항목이 있으므로 그 역할의 열은 결측이 없다
    out = build_war_seasons(FIXTURE_RAW, SEASONS, batting, pitching)
    bat, pit = out[out["role"] == "bat"], out[out["role"] == "pit"]
    for col in BAT_COLUMNS:
        assert not bat[col].isna().any(), col
        assert pit[col].isna().all(), col
    for col in PIT_COLUMNS:
        assert not pit[col].isna().any(), col
        assert bat[col].isna().all(), col


def test_clean_war_short_season_values_are_not_scaled(
    batting: pd.DataFrame, pitching: pd.DataFrame
) -> None:
    # 2020(60경기)도 원본 WAR 그대로 — 출전량 환산(D9)은 WAR에 쓰지 않는다
    out = build_war_seasons(FIXTURE_RAW, [SHORT_SEASON], batting, pitching)
    nola = out[(out["player_id"] == NOLA) & (out["role"] == "pit")]
    assert len(nola) == 1
    expected = _saber_stats(FIXTURE_RAW, SHORT_SEASON, "pit")[NOLA]
    assert float(nola["war"].iloc[0]) == pytest.approx(expected["war"], abs=1e-12)
    assert float(nola["war_ra9"].iloc[0]) == pytest.approx(expected["ra9War"], abs=1e-12)
    ohtani = out[(out["player_id"] == OHTANI) & (out["role"] == "bat")]
    assert float(ohtani["war"].iloc[0]) == pytest.approx(-0.03624, abs=1e-12)


def test_clean_war_skenes_values(batting: pd.DataFrame, pitching: pd.DataFrame) -> None:
    out = build_war_seasons(FIXTURE_RAW, [2025], batting, pitching)
    row = out[(out["player_id"] == SKENES) & (out["season"] == 2025)]
    assert list(row["role"]) == ["pit"]
    assert float(row["war"].iloc[0]) == pytest.approx(6.53143, abs=1e-12)
    assert float(row["war_ra9"].iloc[0]) == pytest.approx(8.22494, abs=1e-12)


def test_clean_war_ohtani_2025_batting_components(
    batting: pd.DataFrame, pitching: pd.DataFrame
) -> None:
    out = build_war_seasons(FIXTURE_RAW, [2025], batting, pitching)
    row = _records(out[(out["player_id"] == OHTANI) & (out["role"] == "bat")])
    assert len(row) == 1
    expected = {
        "war": 7.45112,
        "war_batting": 62.0982,
        "war_baserunning": 3.63297,
        "war_fielding": 0.0,  # 지명타자 (E26)
        "war_positional": -17.0679,
        "war_league": 2.02232,
        "war_replacement": 22.1415,
    }
    for col, value in expected.items():
        assert row[0][col] == pytest.approx(value, abs=1e-12), col
    assert pd.isna(row[0]["war_ra9"])


# ---------------------------------------------------------------------------
# 투타 겸업 · 역할 밖 행
# ---------------------------------------------------------------------------


def test_clean_war_ohtani_has_bat_and_pit_rows(
    batting: pd.DataFrame, pitching: pd.DataFrame
) -> None:
    out = build_war_seasons(FIXTURE_RAW, SEASONS, batting, pitching)
    ohtani = out[out["player_id"] == OHTANI]
    roles = {
        season: sorted(ohtani.loc[ohtani["season"] == season, "role"])
        for season in sorted(set(ohtani["season"]))
    }
    assert roles == {
        2019: ["bat"],
        2020: ["bat", "pit"],
        2021: ["bat", "pit"],
        2022: ["bat", "pit"],
        2023: ["bat", "pit"],
        2024: ["bat"],  # 2024는 투구 없음 (E7)
        2025: ["bat", "pit"],
        2026: ["bat", "pit"],
    }


@pytest.mark.parametrize(
    ("season", "total"),
    [(2021, 7.98), (2022, 9.20), (2023, 8.87), (2024, 8.93), (2025, 9.37), (2026, 6.63)],
)
def test_clean_war_ohtani_two_way_total_matches_spec(
    batting: pd.DataFrame, pitching: pd.DataFrame, season: int, total: float
) -> None:
    # SPEC E26의 오타니 투타 합산 WAR (두 행의 war를 더한 값)
    out = build_war_seasons(FIXTURE_RAW, [season], batting, pitching)
    ohtani = out[out["player_id"] == OHTANI]
    assert round(float(ohtani["war"].sum()), 2) == pytest.approx(total)


def test_clean_war_drops_saber_rows_outside_season_tables(
    batting: pd.DataFrame, pitching: pd.DataFrame
) -> None:
    # 투수의 타격 행 · 등판한 야수의 투구 행은 세이버메트릭스에만 있다 (E26) — 버린다
    assert NOLA in _saber_stats(FIXTURE_RAW, 2019, "bat")
    assert IGLESIAS in _saber_stats(FIXTURE_RAW, 2026, "bat")
    assert HERNANDEZ in _saber_stats(FIXTURE_RAW, 2025, "pit")
    out = build_war_seasons(FIXTURE_RAW, SEASONS, batting, pitching)
    bat_ids = set(out.loc[out["role"] == "bat", "player_id"])
    pit_ids = set(out.loc[out["role"] == "pit", "player_id"])
    assert bat_ids.isdisjoint(PITCHERS)
    assert HERNANDEZ not in pit_ids
    assert HERNANDEZ in bat_ids


def test_clean_war_season_table_defines_rows(batting: pd.DataFrame, pitching: pd.DataFrame) -> None:
    # 시즌 표에서 뺀 행은 세이버메트릭스에 있어도 만들지 않는다
    fewer_bat = batting[~((batting["player_id"] == DEVERS) & (batting["season"] == 2025))]
    fewer_pit = pitching[~((pitching["player_id"] == OHTANI) & (pitching["season"] == 2025))]
    out = build_war_seasons(FIXTURE_RAW, [2025], fewer_bat, fewer_pit)
    keys = set(_keys(out))
    assert (DEVERS, 2025, "bat") not in keys
    assert (OHTANI, 2025, "pit") not in keys
    assert (OHTANI, 2025, "bat") in keys
    assert len(out) == int(
        (fewer_bat["season"] == 2025).sum() + (fewer_pit["season"] == 2025).sum()
    )


# ---------------------------------------------------------------------------
# 시즌 선택
# ---------------------------------------------------------------------------


def test_clean_war_selected_seasons_only(batting: pd.DataFrame, pitching: pd.DataFrame) -> None:
    out = build_war_seasons(FIXTURE_RAW, [2024, 2025], batting, pitching)
    assert set(out["season"]) == {2024, 2025}
    expected = {key for key in _role_keys(batting, pitching) if key[1] in (2024, 2025)}
    assert set(_keys(out)) == expected
    assert len(out) == len(expected)
    assert check_frame(out, WAR_SEASONS) == []


def test_clean_war_reads_only_selected_season_raw(
    raw_dir: Path, batting: pd.DataFrame, pitching: pd.DataFrame
) -> None:
    # 요청하지 않은 시즌의 세이버메트릭스 원본은 없어도 된다
    for season in SEASONS:
        if season not in (2024, 2025):
            for role in ROLES:
                (raw_dir / raw_saber_path(season, role)).unlink()
    out = build_war_seasons(raw_dir, [2024, 2025], batting, pitching)
    assert set(out["season"]) == {2024, 2025}


def test_clean_war_accepts_one_shot_iterable(batting: pd.DataFrame, pitching: pd.DataFrame) -> None:
    seasons = (s for s in (2025, 2023))
    out = build_war_seasons(FIXTURE_RAW, seasons, batting, pitching)
    assert set(out["season"]) == {2023, 2025}
    assert len(out) == int(
        batting["season"].isin([2023, 2025]).sum() + pitching["season"].isin([2023, 2025]).sum()
    )


def test_clean_war_season_order_does_not_matter(
    batting: pd.DataFrame, pitching: pd.DataFrame
) -> None:
    a = build_war_seasons(FIXTURE_RAW, [2021, 2025], batting, pitching)
    b = build_war_seasons(FIXTURE_RAW, [2025, 2021], batting, pitching)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_clean_war_empty_seasons(batting: pd.DataFrame, pitching: pd.DataFrame) -> None:
    out = build_war_seasons(FIXTURE_RAW, [], batting, pitching)
    assert len(out) == 0
    assert list(out.columns) == WAR_SEASONS.column_names
    assert check_frame(out, WAR_SEASONS) == []


def test_clean_war_same_input_same_output(batting: pd.DataFrame, pitching: pd.DataFrame) -> None:
    a = build_war_seasons(FIXTURE_RAW, SEASONS, batting, pitching)
    b = build_war_seasons(FIXTURE_RAW, SEASONS, batting, pitching)
    pd.testing.assert_frame_equal(a, b)


def test_clean_war_does_not_modify_inputs(
    raw_dir: Path, batting: pd.DataFrame, pitching: pd.DataFrame
) -> None:
    bat_before, pit_before = batting.copy(), pitching.copy()
    files_before = {p: p.read_bytes() for p in sorted(raw_dir.rglob("*")) if p.is_file()}
    build_war_seasons(raw_dir, SEASONS, batting, pitching)
    pd.testing.assert_frame_equal(batting, bat_before)
    pd.testing.assert_frame_equal(pitching, pit_before)
    assert {p: p.read_bytes() for p in sorted(raw_dir.rglob("*")) if p.is_file()} == files_before


# ---------------------------------------------------------------------------
# 오류 — F2 ⑦을 통과했다면 생기지 않는 원본
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("season", "role", "player_id"),
    [
        pytest.param(2025, "bat", DEVERS, id="batter"),
        pytest.param(2025, "bat", OHTANI, id="ohtani-bat"),
        pytest.param(2025, "pit", OHTANI, id="ohtani-pit"),
        pytest.param(SHORT_SEASON, "pit", VENDITTE, id="2020-pitcher"),
    ],
)
def test_clean_war_missing_saber_row_raises(
    raw_dir: Path,
    batting: pd.DataFrame,
    pitching: pd.DataFrame,
    season: int,
    role: Role,
    player_id: int,
) -> None:
    def remove(splits: list[dict[str, Any]]) -> None:
        splits[:] = [s for s in splits if s["player"]["id"] != player_id]

    _edit_saber(raw_dir, season, role, remove)
    assert player_id not in _saber_stats(raw_dir, season, role)
    with pytest.raises(ValueError) as exc:
        build_war_seasons(raw_dir, [season], batting, pitching)
    message = str(exc.value)
    assert str(player_id) in message
    assert str(season) in message


@pytest.mark.parametrize("role", ROLES)
def test_clean_war_season_table_row_without_saber_raises(
    batting: pd.DataFrame, pitching: pd.DataFrame, role: Role
) -> None:
    # 시즌 표에 있는데 세이버메트릭스에 없는 (선수, 시즌) — 조용히 결측으로 두지 않는다
    if role == "bat":
        batting = _with_extra_row(batting, 999999, 2025)
    else:
        pitching = _with_extra_row(pitching, 999999, 2025)
    with pytest.raises(ValueError, match="999999"):
        build_war_seasons(FIXTURE_RAW, [2025], batting, pitching)


def test_clean_war_extra_row_outside_selected_seasons_is_ignored(
    batting: pd.DataFrame, pitching: pd.DataFrame
) -> None:
    # 요청하지 않은 시즌의 시즌 표 행은 보지 않는다
    batting = _with_extra_row(batting, 999999, 2023)
    out = build_war_seasons(FIXTURE_RAW, [2025], batting, pitching)
    assert 999999 not in set(out["player_id"])


@pytest.mark.parametrize("role", ROLES)
def test_clean_war_duplicate_season_table_row_raises(
    batting: pd.DataFrame, pitching: pd.DataFrame, role: Role
) -> None:
    # 시즌 표에 같은 (선수, 시즌)이 두 번 있으면 WAR 행도 두 번 생긴다 — 키 중복으로 멈춘다
    if role == "bat":
        batting = pd.concat(
            [batting, batting[(batting["player_id"] == OHTANI) & (batting["season"] == 2025)]],
            ignore_index=True,
        )
    else:
        pitching = pd.concat(
            [pitching, pitching[(pitching["player_id"] == SKENES) & (pitching["season"] == 2025)]],
            ignore_index=True,
        )
    with pytest.raises(ValueError, match="중복"):
        build_war_seasons(FIXTURE_RAW, [2025], batting, pitching)


_BAD_COMPONENT: list[Any] = [
    pytest.param("62.1", id="string"),
    pytest.param(True, id="bool"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("-inf"), id="-inf"),
    pytest.param({"value": 1.0}, id="object"),
]


@pytest.mark.parametrize(
    ("role", "player_id", "field"),
    [("bat", OHTANI, "batting"), ("bat", DEVERS, "positional"), ("pit", SKENES, "ra9War")],
)
@pytest.mark.parametrize("value", _BAD_COMPONENT)
def test_clean_war_bad_component_raises(
    raw_dir: Path,
    batting: pd.DataFrame,
    pitching: pd.DataFrame,
    value: Any,
    role: Role,
    player_id: int,
    field: str,
) -> None:
    # 구성 항목도 숫자가 아니면 조용히 결측으로 바꾸지 않는다 (어느 선수·필드인지 적는다)
    def edit(splits: list[dict[str, Any]]) -> None:
        next(s for s in splits if s["player"]["id"] == player_id)["stat"][field] = value

    _edit_saber(raw_dir, 2025, role, edit)
    with pytest.raises(ValueError) as exc:
        build_war_seasons(raw_dir, [2025], batting, pitching)
    assert str(player_id) in str(exc.value)
    assert field in str(exc.value)


def test_clean_war_null_component_is_missing_only_there(
    raw_dir: Path, batting: pd.DataFrame, pitching: pd.DataFrame
) -> None:
    # F2 ⑦은 구성 항목의 키만 본다 — 값이 null이면 그 칸만 결측이고 war와 다른 칸은 그대로다
    def edit(splits: list[dict[str, Any]]) -> None:
        next(s for s in splits if s["player"]["id"] == OHTANI)["stat"]["fielding"] = None

    _edit_saber(raw_dir, 2025, "bat", edit)
    out = build_war_seasons(raw_dir, [2025], batting, pitching)
    row = _records(out[(out["player_id"] == OHTANI) & (out["role"] == "bat")])
    assert len(row) == 1
    assert pd.isna(row[0]["war_fielding"])
    assert row[0]["war"] == pytest.approx(7.45112, abs=1e-12)
    assert row[0]["war_positional"] == pytest.approx(-17.0679, abs=1e-12)
    others = out[~((out["player_id"] == OHTANI) & (out["role"] == "bat"))]
    assert not others.loc[others["role"] == "bat", "war_fielding"].isna().any()


def test_clean_war_ignores_bad_values_in_dropped_rows(
    raw_dir: Path, batting: pd.DataFrame, pitching: pd.DataFrame
) -> None:
    # 정제 표에 들어가지 않는 행(등판한 야수의 투구 행)의 값은 보지 않는다 — 원본 검사는 F2 ⑦ 몫이다
    def edit(splits: list[dict[str, Any]]) -> None:
        next(s for s in splits if s["player"]["id"] == HERNANDEZ)["stat"]["war"] = None

    _edit_saber(raw_dir, 2025, "pit", edit)
    out = build_war_seasons(raw_dir, [2025], batting, pitching)
    assert HERNANDEZ not in set(out.loc[out["role"] == "pit", "player_id"])
    assert check_frame(out, WAR_SEASONS) == []


@pytest.mark.parametrize(("role", "player_id"), [("bat", DEVERS), ("pit", NOLA)])
def test_clean_war_duplicate_saber_player_raises(
    raw_dir: Path, batting: pd.DataFrame, pitching: pd.DataFrame, role: Role, player_id: int
) -> None:
    def duplicate(splits: list[dict[str, Any]]) -> None:
        splits.append(next(s for s in splits if s["player"]["id"] == player_id))

    _edit_saber(raw_dir, 2025, role, duplicate)
    with pytest.raises(ValueError, match=str(player_id)):
        build_war_seasons(raw_dir, [2025], batting, pitching)


_BAD_WAR: list[Any] = [
    pytest.param(None, id="null"),
    pytest.param("1.5", id="string"),
    pytest.param(True, id="bool"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param([1.0], id="list"),
]


@pytest.mark.parametrize(("role", "player_id"), [("bat", OHTANI), ("pit", SKENES)])
@pytest.mark.parametrize("value", _BAD_WAR)
def test_clean_war_non_finite_war_raises(
    raw_dir: Path,
    batting: pd.DataFrame,
    pitching: pd.DataFrame,
    value: Any,
    role: Role,
    player_id: int,
) -> None:
    def edit(splits: list[dict[str, Any]]) -> None:
        next(s for s in splits if s["player"]["id"] == player_id)["stat"]["war"] = value

    _edit_saber(raw_dir, 2025, role, edit)
    with pytest.raises(ValueError, match=str(player_id)):
        build_war_seasons(raw_dir, [2025], batting, pitching)


def test_clean_war_missing_war_key_raises(
    raw_dir: Path, batting: pd.DataFrame, pitching: pd.DataFrame
) -> None:
    def edit(splits: list[dict[str, Any]]) -> None:
        del next(s for s in splits if s["player"]["id"] == DEVERS)["stat"]["war"]

    _edit_saber(raw_dir, 2025, "bat", edit)
    with pytest.raises(ValueError, match=str(DEVERS)):
        build_war_seasons(raw_dir, [2025], batting, pitching)


@pytest.mark.parametrize("role", ROLES)
def test_clean_war_missing_saber_file_raises(
    raw_dir: Path, batting: pd.DataFrame, pitching: pd.DataFrame, role: Role
) -> None:
    (raw_dir / raw_saber_path(2025, role)).unlink()
    with pytest.raises(FileNotFoundError):
        build_war_seasons(raw_dir, [2025], batting, pitching)
