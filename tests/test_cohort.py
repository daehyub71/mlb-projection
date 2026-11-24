"""예측 대상과 사용 기간(cohort.py) 테스트 (SPEC F4·F5·F12·D3·N2).

입력은 이 파일에서 만든 합성 정제 표다. `_season_frame`이 `contracts.BATTING_SEASONS`·
`PITCHING_SEASONS`의 모든 열을 유효한 값으로 채운다(정제 모듈은 쓰지 않는다).
기대값은 아래 선수 표(`BAT_PA`·`PIT_BF_OUTS`)에서 손으로 계산해 `EXPECTED`에 적었다.
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandas.api import types as ptypes
from pandas.testing import assert_frame_equal

from mlbproj import contracts as c
from mlbproj.cohort import PLAYING_TIME_COLUMNS, build_cohort, expand_window
from mlbproj.contracts import Role
from mlbproj.interim import read_table, write_table

# ---------------------------------------------------------------------------
# 합성 선수 표
# ---------------------------------------------------------------------------

# 타자: player_id → {season: pa}
BAT_PA: dict[int, dict[int, int]] = {
    101: {2022: 500, 2023: 510, 2024: 520, 2025: 530, 2026: 540},  # 2022~2026 다섯 시즌 모두
    102: {2026: 150},  # 2026 데뷔
    103: {2024: 300, 2025: 310, 2026: 320},  # 2024 데뷔
    104: {2021: 400, 2022: 410, 2023: 420},  # 2023 이후 미출전
    105: {2022: 200, 2023: 210, 2024: 220, 2026: 230},  # 2025만 결장
    106: {2019: 600, 2020: 220, 2021: 610},  # 2021까지만 — 2027 대상 아님
    # 107: 2019~2026 여덟 시즌 모두
    107: {
        2019: 100,
        2020: 40,
        2021: 101,
        2022: 102,
        2023: 103,
        2024: 104,
        2025: 105,
        2026: 106,
    },
    108: {2020: 50, 2024: 60},  # 첫 기록이 기간보다 오래됐다
    109: {2019: 90, 2021: 95},  # 2020 결장
}

# 투수: player_id → {season: (bf, outs)}. 101은 투타 겸업(F12) — 투수 첫 기록은 2023, 2024 미출전
PIT_BF_OUTS: dict[int, dict[int, tuple[int, int]]] = {
    101: {2023: (300, 210), 2025: (310, 220), 2026: (320, 230)},
    201: {
        2022: (700, 500),
        2023: (710, 510),
        2024: (720, 520),
        2025: (730, 530),
        2026: (740, 540),
    },
    202: {2022: (100, 60), 2023: (110, 70)},  # 2023 이후 미출전
    204: {2020: (80, 50)},  # 2020만 — 2027 대상 아님
    205: {2026: (50, 33)},  # 2026 데뷔
}

# 출전 행 일부는 Statcast·나이·팀을 비워 둔다 (그대로 결측으로 옮겨져야 한다)
OVERRIDES: dict[Role, dict[tuple[int, int], dict[str, Any]]] = {
    "bat": {(105, 2023): {"woba": None, "xwoba": None, "age": None, "team_name": None}},
    "pit": {(202, 2022): {"xera": None, "whiff_pct": None, "age": None}},
}

# 기대 결과: (player_id, window_first, window_last, n_window, input_first, input_last,
#            n_seasons, played_prev, sample, sample_outs)
CohortRow = tuple[int, int, int, int, int, int, int, bool, int, int | None]
EXPECTED: dict[tuple[Role, int], list[CohortRow]] = {
    ("bat", 2027): [
        (101, 2022, 2026, 5, 2022, 2026, 5, True, 2600, None),
        (102, 2026, 2026, 1, 2026, 2026, 1, True, 150, None),
        (103, 2024, 2026, 3, 2024, 2026, 3, True, 930, None),
        (104, 2022, 2026, 5, 2022, 2023, 2, False, 830, None),
        (105, 2022, 2026, 5, 2022, 2026, 4, True, 860, None),
        (107, 2022, 2026, 5, 2022, 2026, 5, True, 520, None),
        (108, 2022, 2026, 5, 2024, 2024, 1, False, 60, None),
    ],
    ("bat", 2024): [
        (101, 2022, 2023, 2, 2022, 2023, 2, True, 1010, None),
        (104, 2021, 2023, 3, 2021, 2023, 3, True, 1230, None),
        (105, 2022, 2023, 2, 2022, 2023, 2, True, 410, None),
        (106, 2019, 2023, 5, 2019, 2021, 3, False, 1430, None),
        (107, 2019, 2023, 5, 2019, 2023, 5, True, 446, None),
        (108, 2020, 2023, 4, 2020, 2020, 1, False, 50, None),
        (109, 2019, 2023, 5, 2019, 2021, 2, False, 185, None),
    ],
    ("pit", 2027): [
        (101, 2023, 2026, 4, 2023, 2026, 3, True, 930, 660),
        (201, 2022, 2026, 5, 2022, 2026, 5, True, 3600, 2600),
        (202, 2022, 2026, 5, 2022, 2023, 2, False, 210, 130),
        (205, 2026, 2026, 1, 2026, 2026, 1, True, 50, 33),
    ],
    ("pit", 2024): [
        (101, 2023, 2023, 1, 2023, 2023, 1, True, 300, 210),
        (201, 2022, 2023, 2, 2022, 2023, 2, True, 1410, 1010),
        (202, 2022, 2023, 2, 2022, 2023, 2, True, 210, 130),
        (204, 2020, 2023, 4, 2020, 2020, 1, False, 80, 50),
    ],
}
CASES = list(EXPECTED)
CASE_IDS = [f"{role}-{target}" for role, target in CASES]

COHORT_FIELDS: tuple[str, ...] = (
    "player_id",
    "window_first",
    "window_last",
    "n_window",
    "input_first",
    "input_last",
    "n_seasons",
    "played_prev",
    "sample",
    "sample_outs",
)
SPECS: dict[Role, c.TableSpec] = {"bat": c.BATTING_SEASONS, "pit": c.PITCHING_SEASONS}
KEY_COLUMNS = ("player_id", "season", "league")
EXPAND_LEAD = ["player_id", "role", "target_season", "season", "played"]
POISON = 99_999  # 목표 시즌 이후 행에 넣는 "독" 값 (N2)
NEW_PLAYER = 999  # 목표 시즌 이후에만 기록이 있는 선수


# ---------------------------------------------------------------------------
# 도우미 — 계약 형식의 합성 정제 표
# ---------------------------------------------------------------------------


def _dtype(col: c.Column) -> str:
    if col.kind == "int":
        return "Int64" if col.nullable else "int64"
    if col.kind == "float":
        return "float64"
    if col.kind == "bool":
        return "bool"
    return "str"


_ADJUSTED = {"pa_adj": "pa", "outs_adj": "outs"}


def _fill(spec: c.TableSpec, row: Mapping[str, Any]) -> dict[str, Any]:
    """계약의 모든 열을 채운 행. row에 있는 값이 먼저다."""
    pid, season = int(row["player_id"]), int(row["season"])
    out: dict[str, Any] = {}
    for i, col in enumerate(spec.columns):
        name = col.name
        if name in row:
            out[name] = row[name]
        elif name == "league":
            out[name] = c.LEAGUE
        elif name == "team_name":
            out[name] = f"T{pid}"
        elif name == "age":
            out[name] = 22 + (season - c.SEASON_FIRST) + pid % 5
        elif name == "n_teams":
            out[name] = 1
        elif name == "is_short_season":
            out[name] = season == c.SHORT_SEASON
        elif name in _ADJUSTED:
            continue  # 아래에서 출전량으로 계산
        elif col.kind == "int":
            out[name] = i + season % 100
        elif col.kind == "float":
            out[name] = round(0.1 + 0.01 * i + 0.001 * (season - c.SEASON_FIRST) + 1e-5 * pid, 6)
        else:
            raise AssertionError(f"채울 수 없는 열: {name}")
    for adj, base in _ADJUSTED.items():
        if adj in spec.column_names and adj not in out:
            scale = c.SHORT_SEASON_SCALE if season == c.SHORT_SEASON else 1.0
            out[adj] = float(out[base]) * scale
    return out


def _frame(spec: c.TableSpec, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    filled = [_fill(spec, row) for row in rows]
    df = pd.DataFrame(
        {
            col.name: pd.Series([r[col.name] for r in filled], dtype=_dtype(col))
            for col in spec.columns
        }
    )
    assert c.check_frame(df, spec) == []
    return df


def _base_rows(role: Role) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if role == "bat":
        for pid, by_season in BAT_PA.items():
            for season, pa in by_season.items():
                rows.append({"player_id": pid, "season": season, "pa": pa})
    else:
        for pid, by_season_pit in PIT_BF_OUTS.items():
            for season, (bf, outs) in by_season_pit.items():
                rows.append({"player_id": pid, "season": season, "bf": bf, "outs": outs})
    for row in rows:
        row.update(OVERRIDES[role].get((row["player_id"], row["season"]), {}))
    return rows


def _season_frame(role: Role) -> pd.DataFrame:
    """합성 정제 표 전체 (2019~2026)."""
    return _frame(SPECS[role], _base_rows(role))


def _played(role: Role) -> set[tuple[int, int]]:
    data: Mapping[int, Mapping[int, Any]] = BAT_PA if role == "bat" else PIT_BF_OUTS
    return {(pid, season) for pid, by_season in data.items() for season in by_season}


def _clean_and_poisoned(role: Role, target: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(season < target 행만, 같은 행 + season ≥ target "독" 행).

    독 행: 모든 선수의 target~target+2 시즌에 출전량·통산 기록을 POISON으로, 그리고
    target 시즌에만 기록이 있는 새 선수. 독 행을 앞에 둬서 입력 순서도 섞는다.
    """
    clean_rows = [r for r in _base_rows(role) if r["season"] < target]
    data: Mapping[int, Any] = BAT_PA if role == "bat" else PIT_BF_OUTS
    huge = {name: POISON for name in PLAYING_TIME_COLUMNS[role] if name not in _ADJUSTED}
    poison_rows: list[dict[str, Any]] = [
        {"player_id": pid, "season": season, **huge}
        for pid in data
        for season in range(target, target + 3)
    ]
    poison_rows.append({"player_id": NEW_PLAYER, "season": target, **huge})
    spec = SPECS[role]
    return _frame(spec, clean_rows), _frame(spec, poison_rows + clean_rows)


def _cohort_frame(role: Role, target: int) -> pd.DataFrame:
    """`EXPECTED`를 그대로 옮긴 COHORTS 계약 표 (expand_window를 build_cohort와 따로 시험한다)."""
    rows = EXPECTED[(role, target)]
    columns = list(zip(*rows, strict=True))
    data: dict[str, pd.Series] = {}
    for name, values in zip(COHORT_FIELDS, columns, strict=True):
        if name == "played_prev":
            data[name] = pd.Series(values, dtype="bool")
        elif name == "sample_outs":
            data[name] = pd.Series(values, dtype="Int64")
        else:
            data[name] = pd.Series(values, dtype="int64")
    data["role"] = pd.Series([role] * len(rows), dtype="str")
    data["target_season"] = pd.Series([target] * len(rows), dtype="int64")
    df = pd.DataFrame({name: data[name] for name in c.COHORTS.column_names})
    assert c.check_frame(df, c.COHORTS) == []
    return df


def _tuples(df: pd.DataFrame) -> list[CohortRow]:
    out: list[CohortRow] = []
    for rec in df.to_dict("records"):
        sample_outs = rec["sample_outs"]
        out.append(
            (
                int(rec["player_id"]),
                int(rec["window_first"]),
                int(rec["window_last"]),
                int(rec["n_window"]),
                int(rec["input_first"]),
                int(rec["input_last"]),
                int(rec["n_seasons"]),
                bool(rec["played_prev"]),
                int(rec["sample"]),
                None if pd.isna(sample_outs) else int(sample_outs),
            )
        )
    return out


def _by_player(df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    return {t[0]: dict(zip(COHORT_FIELDS, t, strict=True)) for t in _tuples(df)}


def _value_columns(role: Role) -> list[c.Column]:
    return [col for col in SPECS[role].columns if col.name not in KEY_COLUMNS]


def _same(actual: Any, expected: Any) -> bool:
    if pd.isna(expected):
        return bool(pd.isna(actual))
    if pd.isna(actual):
        return False
    if isinstance(expected, float):
        return math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
    return bool(actual == expected)


def _reset(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# build_cohort
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("role", "target"), CASES, ids=CASE_IDS)
def test_cohort_build_full_table(role: Role, target: int) -> None:
    result = build_cohort(_season_frame(role), role, target)
    assert c.check_frame(result, c.COHORTS) == []
    assert _tuples(result) == EXPECTED[(role, target)]
    assert set(result["role"]) == {role}
    assert set(result["target_season"]) == {target}


@pytest.mark.parametrize(("role", "target"), CASES, ids=CASE_IDS)
def test_cohort_build_sorted_by_player_and_invariants(role: Role, target: int) -> None:
    result = build_cohort(_season_frame(role), role, target)
    ids = result["player_id"].tolist()
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    for row in _by_player(result).values():
        assert row["window_last"] == target - 1
        assert row["window_first"] >= target - c.WINDOW
        assert row["n_window"] == row["window_last"] - row["window_first"] + 1
        assert row["window_first"] <= row["input_first"] <= row["input_last"] <= row["window_last"]
        assert 1 <= row["n_seasons"] <= row["n_window"] <= c.WINDOW


def test_cohort_build_five_season_player() -> None:
    row = _by_player(build_cohort(_season_frame("bat"), "bat", 2027))[101]
    assert (row["window_first"], row["window_last"]) == (2022, 2026)
    assert row["n_window"] == 5
    assert row["n_seasons"] == 5
    assert row["played_prev"] is True
    assert row["sample"] == 500 + 510 + 520 + 530 + 540


def test_cohort_build_debut_2026() -> None:
    row = _by_player(build_cohort(_season_frame("bat"), "bat", 2027))[102]
    assert (row["window_first"], row["window_last"]) == (2026, 2026)
    assert (row["input_first"], row["input_last"]) == (2026, 2026)
    assert row["n_window"] == 1
    assert row["n_seasons"] == 1
    assert row["played_prev"] is True
    assert row["sample"] == 150


def test_cohort_build_debut_2024() -> None:
    row = _by_player(build_cohort(_season_frame("bat"), "bat", 2027))[103]
    assert row["window_first"] == 2024
    assert row["n_window"] == 3
    assert row["n_seasons"] == 3
    assert row["sample"] == 300 + 310 + 320


def test_cohort_build_last_seen_2023_is_still_target() -> None:
    row = _by_player(build_cohort(_season_frame("bat"), "bat", 2027))[104]
    assert row["played_prev"] is False
    assert (row["input_first"], row["input_last"]) == (2022, 2023)
    assert (row["window_first"], row["window_last"]) == (2022, 2026)
    assert row["n_window"] == 5  # 미출전 2024~2026도 기간에 든다 (F5)
    assert row["n_seasons"] == 2
    assert row["sample"] == 410 + 420  # 2021은 기간 밖


def test_cohort_build_absent_one_season() -> None:
    row = _by_player(build_cohort(_season_frame("bat"), "bat", 2027))[105]
    assert row["n_window"] == 5
    assert row["n_seasons"] == 4
    assert row["played_prev"] is True
    assert row["sample"] == 200 + 210 + 220 + 230


def test_cohort_build_old_rows_only_not_target() -> None:
    result = build_cohort(_season_frame("bat"), "bat", 2027)
    assert 106 not in result["player_id"].tolist()  # 2019~2021만
    assert 109 not in result["player_id"].tolist()  # 2019·2021만
    pit = build_cohort(_season_frame("pit"), "pit", 2027)
    assert 204 not in pit["player_id"].tolist()  # 2020만


def test_cohort_build_first_season_rule() -> None:
    # 2020·2024 기록: 2027 목표면 기간은 Y−5=2022부터 (첫 기록이 더 오래됐다)
    row_2027 = _by_player(build_cohort(_season_frame("bat"), "bat", 2027))[108]
    assert row_2027["window_first"] == 2022
    assert (row_2027["input_first"], row_2027["input_last"]) == (2024, 2024)
    assert row_2027["n_window"] == 5
    assert row_2027["n_seasons"] == 1
    assert row_2027["played_prev"] is False
    # 2024 목표면 max(2019, 2020) = 2020 — Y−5(2019)가 아니라 첫 기록부터
    by_player_2024 = _by_player(build_cohort(_season_frame("bat"), "bat", 2024))
    row_2024 = by_player_2024[108]
    assert row_2024["window_first"] == 2020
    assert row_2024["window_last"] == 2023
    assert row_2024["n_window"] == 4
    # 첫 기록이 2021이면 2021부터, 2019 이전부터 있으면 2019부터
    assert by_player_2024[104]["window_first"] == 2021
    assert by_player_2024[104]["n_window"] == 3
    assert by_player_2024[107]["window_first"] == 2019
    assert by_player_2024[107]["n_window"] == 5


def test_cohort_build_sample_counts_window_only() -> None:
    row_2027 = _by_player(build_cohort(_season_frame("bat"), "bat", 2027))[107]
    assert row_2027["sample"] == 102 + 103 + 104 + 105 + 106  # 2019~2021은 빼고
    # 2020은 환산 전 pa를 더한다 (pa_adj가 아니다)
    row_2024 = _by_player(build_cohort(_season_frame("bat"), "bat", 2024))[107]
    assert row_2024["sample"] == 100 + 40 + 101 + 102 + 103
    assert _by_player(build_cohort(_season_frame("bat"), "bat", 2024))[108]["sample"] == 50


def test_cohort_build_bat_sample_outs_is_na() -> None:
    result = build_cohort(_season_frame("bat"), "bat", 2027)
    assert ptypes.is_integer_dtype(result["sample_outs"].dtype)
    assert result["sample_outs"].isna().all()
    assert not result["sample"].isna().any()


def test_cohort_build_pitching_sample_and_outs() -> None:
    result = build_cohort(_season_frame("pit"), "pit", 2027)
    assert c.check_frame(result, c.COHORTS) == []
    rows = _by_player(result)
    assert rows[201]["sample"] == 700 + 710 + 720 + 730 + 740  # bf 합
    assert rows[201]["sample_outs"] == 500 + 510 + 520 + 530 + 540  # outs 합
    assert rows[202]["sample"] == 210
    assert rows[202]["sample_outs"] == 130
    assert rows[202]["played_prev"] is False
    assert not result["sample_outs"].isna().any()


def test_cohort_build_two_way_player_roles_independent() -> None:
    # F12: 같은 ID라도 역할마다 첫 기록·사용 시즌 수를 따로 센다
    bat = _by_player(build_cohort(_season_frame("bat"), "bat", 2027))[101]
    pit = _by_player(build_cohort(_season_frame("pit"), "pit", 2027))[101]
    assert (bat["window_first"], bat["n_window"], bat["n_seasons"]) == (2022, 5, 5)
    assert (pit["window_first"], pit["n_window"], pit["n_seasons"]) == (2023, 4, 3)
    assert pit["sample"] == 300 + 310 + 320
    assert pit["sample_outs"] == 210 + 220 + 230


def test_cohort_build_input_order_irrelevant() -> None:
    frame = _season_frame("bat")
    shuffled = frame.sample(frac=1.0, random_state=7)
    assert_frame_equal(
        _reset(build_cohort(shuffled, "bat", 2027)), _reset(build_cohort(frame, "bat", 2027))
    )


@pytest.mark.parametrize(
    ("frame_role", "role"),
    [pytest.param("bat", "pit", id="bat-as-pit"), pytest.param("pit", "bat", id="pit-as-bat")],
)
def test_cohort_build_role_mismatch_raises(frame_role: Role, role: Role) -> None:
    with pytest.raises(ValueError):
        build_cohort(_season_frame(frame_role), role, 2027)


def test_cohort_build_contract_violation_raises() -> None:
    frame = _season_frame("bat")
    with pytest.raises(ValueError):
        build_cohort(frame.drop(columns=["woba"]), "bat", 2027)
    with pytest.raises(ValueError):
        build_cohort(pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "bat", 2027)


@pytest.mark.parametrize(("role", "target"), CASES, ids=CASE_IDS)
def test_cohort_build_ignores_future_rows(role: Role, target: int) -> None:
    # N2: season ≥ Y 행(새 선수·큰 출전량)을 섞어도 결과가 같다
    clean, poisoned = _clean_and_poisoned(role, target)
    expected = build_cohort(clean, role, target)
    assert _tuples(expected) == EXPECTED[(role, target)]
    result = build_cohort(poisoned, role, target)
    assert_frame_equal(_reset(result), _reset(expected))
    assert NEW_PLAYER not in result["player_id"].tolist()
    # 원래 표(2024 목표면 2024~2026 행이 섞여 있다)도 같은 결과
    assert_frame_equal(_reset(build_cohort(_season_frame(role), role, target)), _reset(expected))


# ---------------------------------------------------------------------------
# expand_window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("role", "target"), CASES, ids=CASE_IDS)
def test_cohort_expand_window_one_row_per_window_season(role: Role, target: int) -> None:
    cohort = _cohort_frame(role, target)
    result = expand_window(cohort, _season_frame(role), role)
    expected_keys = [
        (pid, season)
        for pid, first, last, *_ in EXPECTED[(role, target)]
        for season in range(first, last + 1)
    ]
    keys = [(int(p), int(s)) for p, s in zip(result["player_id"], result["season"], strict=True)]
    assert keys == expected_keys  # (player_id, season) 순, 빠짐·중복 없음
    assert len(result) == int(cohort["n_window"].sum())
    played = _played(role)
    assert result["played"].tolist() == [key in played for key in expected_keys]
    assert set(result["role"]) == {role}
    assert set(result["target_season"]) == {target}


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_cohort_expand_window_columns_and_dtypes(role: Role) -> None:
    result = expand_window(_cohort_frame(role, 2027), _season_frame(role), role)
    value_columns = _value_columns(role)
    assert list(result.columns) == EXPAND_LEAD + [col.name for col in value_columns]
    for name in ("player_id", "target_season", "season"):
        assert ptypes.is_integer_dtype(result[name].dtype), name
        assert not result[name].isna().any(), name
    assert ptypes.is_bool_dtype(result["played"].dtype)
    assert not result["played"].isna().any()
    assert ptypes.is_string_dtype(result["role"].dtype) or ptypes.is_object_dtype(
        result["role"].dtype
    )
    for col in value_columns:
        dtype = result[col.name].dtype
        if col.kind == "int":
            assert str(dtype) == "Int64", f"{col.name}: {dtype}"
        elif col.kind == "float":
            assert ptypes.is_float_dtype(dtype), f"{col.name}: {dtype}"
        elif col.kind == "bool":
            assert ptypes.is_bool_dtype(dtype), f"{col.name}: {dtype}"
        else:
            assert ptypes.is_string_dtype(dtype) or ptypes.is_object_dtype(dtype), col.name


@pytest.mark.parametrize(("role", "target"), CASES, ids=CASE_IDS)
def test_cohort_expand_window_absent_seasons(role: Role, target: int) -> None:
    result = expand_window(_cohort_frame(role, target), _season_frame(role), role)
    absent = result[~result["played"].astype(bool)]
    assert len(absent) > 0
    zero_columns = set(PLAYING_TIME_COLUMNS[role])
    for rec in absent.to_dict("records"):
        where = f"({rec['player_id']}, {rec['season']})"
        for col in _value_columns(role):
            value = rec[col.name]
            if col.name in zero_columns:
                assert not pd.isna(value) and value == 0, f"{where} {col.name}: {value!r}"
            elif col.name == "is_short_season":
                assert bool(value) is (rec["season"] == c.SHORT_SEASON), where
            else:
                assert pd.isna(value), f"{where} {col.name}: 결측이어야 하는데 {value!r}"


def test_cohort_expand_window_absent_rows_bat_2027() -> None:
    result = expand_window(_cohort_frame("bat", 2027), _season_frame("bat"), "bat")
    rows = {(int(r["player_id"]), int(r["season"])): r for r in result.to_dict("records")}
    # 104: 2023 이후 미출전 → 2024~2026은 pa·pa_adj 0, 나머지 결측
    for season in (2024, 2025, 2026):
        rec = rows[(104, season)]
        assert not rec["played"]
        assert rec["pa"] == 0
        assert rec["pa_adj"] == 0.0
        assert pd.isna(rec["ops"]) and pd.isna(rec["hr"]) and pd.isna(rec["team_name"])
    # 105: 2025만 결장
    assert not rows[(105, 2025)]["played"]
    assert rows[(105, 2026)]["played"]
    assert rows[(105, 2026)]["pa"] == 230
    # 기간 밖(104의 2021)은 펼치지 않는다
    assert (104, 2021) not in rows


@pytest.mark.parametrize(("role", "target"), CASES, ids=CASE_IDS)
def test_cohort_expand_window_present_seasons_copy_values(role: Role, target: int) -> None:
    seasons = _season_frame(role)
    result = expand_window(_cohort_frame(role, target), seasons, role)
    source = {
        (int(rec["player_id"]), int(rec["season"])): rec for rec in seasons.to_dict("records")
    }
    present = result[result["played"].astype(bool)]
    assert len(present) > 0
    for rec in present.to_dict("records"):
        key = (int(rec["player_id"]), int(rec["season"]))
        for col in _value_columns(role):
            assert _same(rec[col.name], source[key][col.name]), (
                f"{key} {col.name}: {rec[col.name]!r} != {source[key][col.name]!r}"
            )


def test_cohort_expand_window_keeps_missing_values_of_present_rows() -> None:
    result = expand_window(_cohort_frame("bat", 2027), _season_frame("bat"), "bat")
    rec = result[(result["player_id"] == 105) & (result["season"] == 2023)].iloc[0]
    assert bool(rec["played"])
    assert rec["pa"] == 210
    assert pd.isna(rec["woba"]) and pd.isna(rec["xwoba"])
    assert pd.isna(rec["age"]) and pd.isna(rec["team_name"])
    assert not pd.isna(rec["ev_avg"])


def test_cohort_expand_window_short_season_flag() -> None:
    result = expand_window(_cohort_frame("bat", 2024), _season_frame("bat"), "bat")

    def row(pid: int, season: int) -> pd.Series:
        hit = result[(result["player_id"] == pid) & (result["season"] == season)]
        assert len(hit) == 1
        return hit.iloc[0]

    # 109는 2020 결장: 행이 없어도 is_short_season은 True, 출전량 0
    absent = row(109, 2020)
    assert not bool(absent["played"])
    assert bool(absent["is_short_season"]) is True
    assert absent["pa"] == 0
    assert absent["pa_adj"] == 0.0
    # 108의 2020 출전: 표의 환산값을 그대로 쓴다 (50 × 162/60 = 135)
    present = row(108, 2020)
    assert bool(present["played"])
    assert bool(present["is_short_season"]) is True
    assert present["pa"] == 50
    assert float(present["pa_adj"]) == pytest.approx(135.0)
    # 2020이 아닌 시즌은 모두 False
    other = result[result["season"] != c.SHORT_SEASON]
    assert not other["is_short_season"].astype(bool).any()
    assert result.loc[result["season"] == c.SHORT_SEASON, "is_short_season"].astype(bool).all()


def test_cohort_expand_window_pitching_absent_rows() -> None:
    result = expand_window(_cohort_frame("pit", 2027), _season_frame("pit"), "pit")
    rec = result[(result["player_id"] == 101) & (result["season"] == 2024)].iloc[0]
    assert not bool(rec["played"])
    for name in ("g", "gs", "outs", "bf"):
        assert rec[name] == 0, name
    assert rec["outs_adj"] == 0.0
    for name in ("er", "so", "sv", "n_teams", "age", "team_name", "xera", "whiff_pct"):
        assert pd.isna(rec[name]), name
    assert bool(rec["is_short_season"]) is False
    present = result[(result["player_id"] == 202) & (result["season"] == 2023)].iloc[0]
    assert bool(present["played"])
    assert present["bf"] == 110
    assert present["outs"] == 70


def test_cohort_expand_window_input_order_irrelevant() -> None:
    cohort = _cohort_frame("bat", 2027)
    seasons = _season_frame("bat")
    expected = expand_window(cohort, seasons, "bat")
    shuffled = expand_window(
        cohort.sample(frac=1.0, random_state=3), seasons.sample(frac=1.0, random_state=5), "bat"
    )
    assert_frame_equal(_reset(shuffled), _reset(expected))


@pytest.mark.parametrize(("role", "target"), CASES, ids=CASE_IDS)
def test_cohort_expand_window_ignores_future_rows(role: Role, target: int) -> None:
    # N2: 기간 뒤(season ≥ Y) 행을 섞어도 펼친 결과가 같다
    cohort = _cohort_frame(role, target)
    clean, poisoned = _clean_and_poisoned(role, target)
    expected = expand_window(cohort, clean, role)
    result = expand_window(cohort, poisoned, role)
    assert_frame_equal(_reset(result), _reset(expected))
    assert_frame_equal(_reset(expand_window(cohort, _season_frame(role), role)), _reset(expected))
    assert int(result["season"].max()) == target - 1
    assert NEW_PLAYER not in result["player_id"].tolist()
    assert POISON not in result[PLAYING_TIME_COLUMNS[role][0]].dropna().tolist()


def test_cohort_expand_window_after_build_cohort() -> None:
    seasons = _season_frame("pit")
    cohort = build_cohort(seasons, "pit", 2027)
    result = expand_window(cohort, seasons, "pit")
    assert len(result) == int(cohort["n_window"].sum())
    by_player = result.groupby("player_id")["played"].sum().astype(int).to_dict()
    assert by_player == {row[0]: row[6] for row in EXPECTED[("pit", 2027)]}


# ---------------------------------------------------------------------------
# 경계 사례 · 저장 형식 · 입력 보존
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_cohort_build_empty_result_keeps_contract(role: Role) -> None:
    # 기간 안 기록이 없는 목표 시즌, 빈 입력 표 모두 계약을 지키는 빈 표다
    seasons = _season_frame(role)
    far = build_cohort(seasons, role, c.SEASON_LAST + 10)
    empty_input = build_cohort(seasons.iloc[0:0], role, 2027)
    for result in (far, empty_input):
        assert len(result) == 0
        assert c.check_frame(result, c.COHORTS) == []
    expanded = expand_window(far, seasons, role)
    assert len(expanded) == 0
    assert list(expanded.columns) == EXPAND_LEAD + [col.name for col in _value_columns(role)]


@pytest.mark.parametrize(("role", "target"), CASES, ids=CASE_IDS)
def test_cohort_build_interim_round_trip(role: Role, target: int, tmp_path: Path) -> None:
    # 저장·읽기를 거쳐도 같은 표다 — 정수 열이 Int64·실수로, sample_outs가 다른 형으로 바뀌지 않는다
    result = build_cohort(_season_frame(role), role, target)
    path = tmp_path / "cohorts.csv"
    write_table(result, c.COHORTS, path)
    assert_frame_equal(read_table(c.COHORTS, path), result)


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_cohort_inputs_not_modified(role: Role) -> None:
    seasons = _season_frame(role)
    seasons_before = seasons.copy(deep=True)
    cohort = build_cohort(seasons, role, 2027)
    cohort_before = cohort.copy(deep=True)
    expand_window(cohort, seasons, role)
    assert_frame_equal(seasons, seasons_before)
    assert_frame_equal(cohort, cohort_before)


# ---------------------------------------------------------------------------
# expand_window 입력 검사 (N2 — 대상 표의 기간을 그대로 펼치므로 잘못된 기간을 막는다)
# ---------------------------------------------------------------------------


def _shift_window(cohort: pd.DataFrame, player_id: int, first: int, last: int) -> pd.DataFrame:
    """한 대상의 기간을 바꾼 대상 표 (n_window도 맞춰서 COHORTS 계약은 지킨다)."""
    out = cohort.copy()
    hit = out["player_id"] == player_id
    out.loc[hit, "window_first"] = first
    out.loc[hit, "window_last"] = last
    out.loc[hit, "n_window"] = last - first + 1
    assert c.check_frame(out, c.COHORTS) == []
    return out


@pytest.mark.parametrize(
    ("first", "last"),
    [
        pytest.param(2022, 2027, id="last-is-target"),
        pytest.param(2023, 2028, id="last-after-target"),
        pytest.param(2021, 2026, id="first-before-window"),
        pytest.param(2026, 2025, id="first-after-last"),
    ],
)
def test_cohort_expand_window_rejects_window_outside_target(first: int, last: int) -> None:
    cohort = _shift_window(_cohort_frame("bat", 2027), 101, first, last)
    with pytest.raises(ValueError, match="101"):
        expand_window(cohort, _season_frame("bat"), "bat")


def test_cohort_expand_window_rejects_cohort_role_mismatch() -> None:
    with pytest.raises(ValueError, match="역할"):
        expand_window(_cohort_frame("pit", 2027), _season_frame("bat"), "bat")


def test_cohort_expand_window_rejects_several_targets() -> None:
    cohort = pd.concat([_cohort_frame("bat", 2027), _cohort_frame("bat", 2024)], ignore_index=True)
    assert c.check_frame(cohort, c.COHORTS) == []
    with pytest.raises(ValueError, match="목표 시즌"):
        expand_window(cohort, _season_frame("bat"), "bat")


def test_cohort_expand_window_rejects_contract_violations() -> None:
    cohort = _cohort_frame("bat", 2027)
    seasons = _season_frame("bat")
    with pytest.raises(ValueError):
        expand_window(cohort.drop(columns=["sample_outs"]), seasons, "bat")
    with pytest.raises(ValueError):
        expand_window(cohort, seasons.drop(columns=["woba"]), "bat")
    with pytest.raises(ValueError):
        expand_window(cohort, _season_frame("pit"), "bat")  # 투수 표를 타자로


# --- 데뷔 연도로 기간 시작 보완 (F5, E21 — 데이터가 2019부터라 그 전 기록은 데뷔 연도로 안다) ---


def _bat_rows(pid: int, seasons: Sequence[int]) -> pd.DataFrame:
    return _frame(c.BATTING_SEASONS, [{"player_id": pid, "season": s, "pa": 300} for s in seasons])


def test_cohort_build_debut_extends_window_to_target_minus_five() -> None:
    """2008 데뷔, 2019·2020 결장 뒤 2021 복귀: 목표 2024의 기간은 2021이 아니라 2019부터."""
    df = _bat_rows(1, (2021, 2022, 2023))
    assert int(build_cohort(df, "bat", 2024).iloc[0]["window_first"]) == 2021
    row = build_cohort(df, "bat", 2024, debut_seasons={1: 2008}).iloc[0]
    assert (int(row["window_first"]), int(row["n_window"])) == (2019, 5)
    assert (int(row["input_first"]), int(row["n_seasons"]), int(row["sample"])) == (2021, 3, 900)


def test_cohort_build_debut_inside_window() -> None:
    """2022 데뷔(그해 행 없음), 2023부터 행: 목표 2025의 기간은 2022부터."""
    row = build_cohort(_bat_rows(2, (2023, 2024)), "bat", 2025, debut_seasons={2: 2022}).iloc[0]
    assert (int(row["window_first"]), int(row["n_window"])) == (2022, 3)


def test_cohort_build_debut_later_than_first_row_uses_first_row() -> None:
    """데뷔 연도가 첫 행보다 늦게 적혀 있어도(잘못된 값) 첫 행보다 늦게 시작하지 않는다."""
    row = build_cohort(_bat_rows(3, (2021, 2022)), "bat", 2024, debut_seasons={3: 2023}).iloc[0]
    assert int(row["window_first"]) == 2021


def test_cohort_build_debut_unknown_player_falls_back_to_first_row() -> None:
    row = build_cohort(_bat_rows(4, (2022, 2023)), "bat", 2024, debut_seasons={999: 2001}).iloc[0]
    assert int(row["window_first"]) == 2022


def test_cohort_build_debut_does_not_change_targets_or_leak() -> None:
    """데뷔 연도는 대상을 늘리지 않는다 — 기간 안에 행이 없는 선수는 여전히 대상이 아니다(F4·N2)."""
    df = pd.concat(
        [_bat_rows(5, (2021,)), _bat_rows(6, (2025, 2026)), _bat_rows(7, (2019,))],
        ignore_index=True,
    )
    out = build_cohort(df, "bat", 2025, debut_seasons={5: 2010, 6: 2015, 7: 2010})
    assert out["player_id"].tolist() == [5]  # 6은 목표 시즌 이후만, 7은 기간(2020~2024) 전만
    assert int(out.iloc[0]["window_first"]) == 2020


def test_cohort_expand_window_debut_fills_absent_early_seasons() -> None:
    df = _bat_rows(1, (2021, 2022, 2023))
    cohort = build_cohort(df, "bat", 2024, debut_seasons={1: 2008})
    wide = expand_window(cohort, df, "bat")
    assert wide["season"].tolist() == [2019, 2020, 2021, 2022, 2023]
    assert wide["played"].tolist() == [False, False, True, True, True]
    assert wide["pa"].tolist()[:2] == [0, 0]
    assert bool(wide.loc[wide["season"] == 2020, "is_short_season"].iloc[0])
