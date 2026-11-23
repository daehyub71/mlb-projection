"""계약(contracts.py) 테스트 — 상수·경로·URL·표 검사."""

from pathlib import Path

import pandas as pd
import pytest

from mlbproj import contracts as c


def test_contracts_seasons_and_windows() -> None:
    assert c.SEASONS == (2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)
    assert c.FIELDING_YEARS == (2022, 2023, 2024, 2025, 2026)
    assert c.TARGET_SEASON - c.WINDOW == 2022  # 2027 입력 = 2022~2026 (D3)
    assert c.FINAL_TEST_TARGET not in c.TUNE_TARGETS
    assert c.SHORT_SEASON_SCALE == pytest.approx(2.7)
    assert c.RATE_MIN_OUTS == 50 * 3


def test_contracts_raw_paths() -> None:
    assert c.raw_statsapi_path(2020, "bat") == Path("data/raw/statsapi/2020_bat.json")
    assert c.raw_people_path(3) == Path("data/raw/statsapi/people_003.json")
    assert c.raw_savant_path(2026, "frv") == Path("data/raw/savant/2026_frv.csv")
    assert c.web_player_path(660271) == Path("web/public/data/players/660271.json")


def test_contracts_urls_match_measured() -> None:
    """SPEC E1·E4·E18 실측에 쓴 URL과 같아야 한다."""
    assert c.statsapi_stats_url(2026, "pit") == (
        "https://statsapi.mlb.com/api/v1/stats?stats=season&group=pitching&season=2026"
        "&sportId=1&playerPool=ALL&limit=3000"
    )
    assert c.savant_oaa_url(2026) == (
        "https://baseballsavant.mlb.com/leaderboard/outs_above_average?type=Fielder"
        "&startYear=2026&endYear=2026&split=no&team=&range=year&min=1&pos=&roles=&viz=hide"
        "&csv=true"
    )
    assert c.savant_frv_url(2022) == (
        "https://baseballsavant.mlb.com/leaderboard/fielding-run-value?gameType=Regular"
        "&seasonStart=2022&seasonEnd=2022&type=fielder&position=&minInnings=0&minResults=1"
        "&csv=true"
    )


def test_contracts_savant_url_same_columns_every_year() -> None:
    """F1: 연도마다 같은 열을 요청한다. p_gs는 비어 있어서 받지 않는다(E3)."""
    urls = {season: c.savant_leaderboard_url(season, "pit") for season in c.SEASONS}
    selections = {url.split("selections=")[1] for url in urls.values()}
    assert len(selections) == 1
    assert "p_gs" not in next(iter(selections))
    assert "min=1" in urls[2019]
    assert "type=batter" in c.savant_leaderboard_url(2019, "bat")


def test_contracts_people_url_batches() -> None:
    assert c.statsapi_people_url([1, 22]).endswith("people?personIds=1,22")
    with pytest.raises(ValueError):
        c.statsapi_people_url([])
    with pytest.raises(ValueError):
        c.statsapi_people_url(list(range(c.PEOPLE_BATCH_SIZE + 1)))


def test_contracts_user_agent_has_no_personal_info() -> None:
    assert "@" not in c.USER_AGENT
    assert c.USER_AGENT.startswith("mlbproj/")


def test_contracts_table_specs_consistent() -> None:
    names = [t.name for t in c.TABLES]
    assert len(names) == len(set(names))
    for table in c.TABLES:
        cols = table.column_names
        assert len(cols) == len(set(cols)), table.name
        assert set(table.keys) <= set(cols), table.name
        for key in table.keys:
            assert not next(col for col in table.columns if col.name == key).nullable


def _players_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2],
            "name": ["A", "B"],
            "birth_date": ["1990-01-01", None],
            "primary_position": ["SS", "P"],
            "bats": ["R", "L"],
            "throws": ["R", "L"],
            "active": [True, False],
            "last_mlb_season": [2026, 2023],
        }
    )


def test_contracts_check_frame_passes_valid() -> None:
    assert c.check_frame(_players_frame(), c.PLAYERS) == []


def test_contracts_check_frame_accepts_nullable_int() -> None:
    df = pd.DataFrame(
        {
            "target_season": [2025],
            "role": ["bat"],
            "model": ["league"],
            "metric": ["pa"],
            "group": ["all"],
            "n": pd.array([10], dtype="Int64"),
            "mae": [None],
            "mae_weighted": [None],
            "baseline_mae_league": [None],
            "baseline_mae_prev": [3.0],
            "excluded": [0],
        }
    ).astype({"mae": "float64", "mae_weighted": "float64", "baseline_mae_league": "float64"})
    assert c.check_frame(df, c.EVALUATIONS) == []


def test_contracts_check_frame_detects_problems() -> None:
    missing = _players_frame().drop(columns=["active"])
    assert any("열 불일치" in p for p in c.check_frame(missing, c.PLAYERS))

    reordered = _players_frame()[["name", *[n for n in c.PLAYERS.column_names if n != "name"]]]
    assert c.check_frame(reordered, c.PLAYERS)

    wrong_kind = _players_frame().assign(last_mlb_season=["2026", "2023"])
    assert any("last_mlb_season" in p for p in c.check_frame(wrong_kind, c.PLAYERS))

    null_name = _players_frame().assign(name=["A", None])
    assert any("name" in p and "결측" in p for p in c.check_frame(null_name, c.PLAYERS))

    dup = pd.concat([_players_frame(), _players_frame().iloc[[0]]], ignore_index=True)
    assert any("중복" in p for p in c.check_frame(dup, c.PLAYERS))


FIXTURE_RAW = Path(__file__).resolve().parent / "fixtures" / "raw"


@pytest.mark.parametrize("season", c.SEASONS)
def test_contracts_fixture_statsapi_has_required_fields(season: int) -> None:
    import json

    for role, stat_fields in (
        ("bat", c.STATSAPI_HITTING_STAT_FIELDS),
        ("pit", c.STATSAPI_PITCHING_STAT_FIELDS),
    ):
        path = FIXTURE_RAW / c.raw_statsapi_path(season, role).relative_to(c.RAW_DIR)
        splits = json.loads(path.read_text(encoding="utf-8"))["stats"][0]["splits"]
        assert splits
        for split in splits:
            assert set(c.STATSAPI_SPLIT_FIELDS) <= set(split)
            assert set(stat_fields) <= set(split["stat"]), (role, split["player"]["id"])


def test_contracts_fixture_people_has_required_fields() -> None:
    import json

    for batch in (0, 1):
        path = FIXTURE_RAW / c.raw_people_path(batch).relative_to(c.RAW_DIR)
        for person in json.loads(path.read_text(encoding="utf-8"))["people"]:
            assert set(c.PEOPLE_FIELDS) <= set(person), person["id"]


@pytest.mark.parametrize(
    ("kind", "columns", "years"),
    [
        ("batter", c.SAVANT_BATTER_COLUMNS, c.SEASONS),
        ("pitcher", c.SAVANT_PITCHER_COLUMNS, c.SEASONS),
        ("oaa", c.SAVANT_OAA_COLUMNS, c.FIELDING_YEARS),
        ("frv", c.SAVANT_FRV_COLUMNS, c.FIELDING_YEARS),
    ],
)
def test_contracts_fixture_savant_has_required_columns(
    kind: str, columns: tuple[str, ...], years: tuple[int, ...]
) -> None:
    for season in years:
        df = pd.read_csv(FIXTURE_RAW / "savant" / f"{season}_{kind}.csv", encoding="utf-8-sig")
        assert set(columns) <= set(df.columns), (kind, season)
