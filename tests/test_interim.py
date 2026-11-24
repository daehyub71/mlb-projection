"""정제 표 CSV 입출력(interim.py) 테스트 — 계약 형식 왕복과 재현성(N1)."""

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlbproj import contracts as c
from mlbproj.interim import read_table, write_table


def _players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [660271, 408234],
            "name": ["Shohei Ohtani", "Miguel Cabrera"],
            "birth_date": ["1994-07-05", None],
            "primary_position": ["TWP", "1B"],
            "bats": ["L", None],
            "throws": ["R", "R"],
            "active": [True, False],
            "last_mlb_season": [2026, 2023],
        }
    )


def _cohorts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [2, 1],
            "role": ["pit", "bat"],
            "target_season": [2027, 2027],
            "window_first": [2022, 2024],
            "window_last": [2026, 2026],
            "n_window": [5, 3],
            "input_first": [2022, 2024],
            "input_last": [2025, 2026],
            "n_seasons": [3, 3],
            "played_prev": [False, True],
            "sample": [402, 1740],
            "sample_outs": pd.array([288, None], dtype="Int64"),
        }
    )


def test_interim_round_trip_players(tmp_path: Path) -> None:
    path = tmp_path / "players.csv"
    write_table(_players(), c.PLAYERS, path)
    back = read_table(c.PLAYERS, path)
    assert c.check_frame(back, c.PLAYERS) == []
    expected = _players().sort_values("player_id").reset_index(drop=True)
    assert back["player_id"].tolist() == expected["player_id"].tolist()
    assert back["birth_date"].isna().tolist() == [True, False]
    assert back["active"].tolist() == [False, True]
    assert back["name"].tolist() == expected["name"].tolist()


def test_interim_round_trip_nullable_int_and_bool(tmp_path: Path) -> None:
    path = tmp_path / "cohorts.csv"
    write_table(_cohorts(), c.COHORTS, path)
    back = read_table(c.COHORTS, path)
    assert c.check_frame(back, c.COHORTS) == []
    assert str(back["sample_outs"].dtype) == "Int64"
    assert back["sample_outs"].isna().tolist() == [True, False]  # 키 순서: (1, bat), (2, pit)
    assert back["played_prev"].tolist() == [True, False]


def test_interim_floats_round_trip_exactly(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "target_season": [2025, 2025],
            "role": ["bat", "bat"],
            "model": ["weighted", "league"],
            "metric": ["ops", "ops"],
            "group": ["all", "all"],
            "n": [10, 10],
            "mae": [0.1 + 0.2, float("nan")],
            "mae_weighted": [1 / 3, 2 / 3],
            "baseline_mae_league": [0.07345678901234567, 0.0],
            "baseline_mae_prev": [float("nan"), 1e-12],
            "excluded": [0, 3],
        }
    )
    path = tmp_path / "e.csv"
    write_table(df, c.EVALUATIONS, path)
    back = read_table(c.EVALUATIONS, path)
    expected = df.sort_values(list(c.EVALUATIONS.keys)).reset_index(drop=True)
    assert_frame_equal(
        back[["mae", "mae_weighted", "baseline_mae_league", "baseline_mae_prev"]],
        expected[["mae", "mae_weighted", "baseline_mae_league", "baseline_mae_prev"]],
    )


def test_interim_write_is_deterministic_and_sorted(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_table(_cohorts(), c.COHORTS, a)
    write_table(_cohorts().iloc[::-1].reset_index(drop=True), c.COHORTS, b)
    assert a.read_bytes() == b.read_bytes()
    lines = a.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(c.COHORTS.column_names)
    assert lines[1].startswith("1,bat,2027")
    assert not a.read_bytes().endswith(b"\r\n")


def test_interim_write_rejects_contract_violation(tmp_path: Path) -> None:
    bad = _players().drop(columns=["active"])
    with pytest.raises(ValueError, match="열 불일치"):
        write_table(bad, c.PLAYERS, tmp_path / "p.csv")
    assert not (tmp_path / "p.csv").exists()


def test_interim_read_rejects_wrong_header(tmp_path: Path) -> None:
    path = tmp_path / "p.csv"
    path.write_text("player_id,name\n1,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="열"):
        read_table(c.PLAYERS, path)


def test_interim_read_rejects_null_in_required_column(tmp_path: Path) -> None:
    path = tmp_path / "p.csv"
    write_table(_players(), c.PLAYERS, path)
    text = path.read_text(encoding="utf-8").replace("Shohei Ohtani", "")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="name"):
        read_table(c.PLAYERS, path)


def test_interim_read_rejects_empty_or_odd_bool(tmp_path: Path) -> None:
    path = tmp_path / "p.csv"
    write_table(_players(), c.PLAYERS, path)
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace(",False,", ",,", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="active"):
        read_table(c.PLAYERS, path)
    path.write_text(original.replace(",False,", ",yes,", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="active"):
        read_table(c.PLAYERS, path)
