"""과거 검증 평가(evaluate.py) 테스트 (SPEC F10·D13·N2, PLAN R8, `contracts.EVALUATIONS`).

목표 시즌 Y = 2025. 대상 표·정제 표·예측 표는 이 파일에서 손으로 만든다(모델 모듈은 쓰지 않는다).
모델은 "league"·"baseline"·"weighted" 셋이다. 선수별 예측·실제 값과 그룹별 MAE는 아래 주석에 손으로
계산해 `BAT_EXPECTED`·`PIT_EXPECTED`에 적었다.

집단 규칙 (evaluate 모듈 docstring)
- 비율 지표 mae: 대상 중 Y 실제 pa ≥ RATE_MIN_PA(타자) / outs ≥ RATE_MIN_OUTS(투수)인 선수의 평균 |오차|.
- 비율 지표 mae_weighted: 대상 중 Y 분모(ops는 pa, era는 outs, k_bb_pct는 bf) > 0인 선수 전원의 가중 평균.
- excluded = 그룹 대상 수 − n. 출전량 지표는 대상 전원(미출전 실제 0), mae_weighted 결측, excluded 0.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlbproj import contracts as c
from mlbproj.contracts import Role
from mlbproj.evaluate import METRICS, actual_values, evaluate

TARGET = 2025
POISON = 99_999

# ---------------------------------------------------------------------------
# 도우미 — 계약 형식의 합성 표
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {"league": c.LEAGUE, "team_name": "TST", "age": 27, "n_teams": 1}
_ADJUSTED = {"pa_adj": "pa", "outs_adj": "outs"}


def _dtype(col: c.Column) -> str:
    if col.kind == "int":
        return "Int64" if col.nullable else "int64"
    if col.kind == "float":
        return "float64"
    if col.kind == "bool":
        return "bool"
    return "str"


def _value(col: c.Column, row: Mapping[str, Any]) -> Any:
    """행에 없는 열의 기본값. 정수 0, 결측 허용 열은 결측, 2020 출전량은 ×162/60."""
    name = col.name
    if name in row:
        return row[name]
    if name in _DEFAULTS:
        return _DEFAULTS[name]
    if name == "is_short_season":
        return int(row["season"]) == c.SHORT_SEASON
    if name in _ADJUSTED:
        scale = c.SHORT_SEASON_SCALE if int(row["season"]) == c.SHORT_SEASON else 1.0
        return float(row.get(_ADJUSTED[name], 0)) * scale
    if col.kind == "int" and not col.nullable:
        return 0
    if col.nullable:
        return None
    raise AssertionError(f"채울 수 없는 열: {name}")


def _frame(spec: c.TableSpec, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """계약의 모든 열을 채운 표 (`check_frame` 통과를 확인한다)."""
    df = pd.DataFrame(
        {
            col.name: pd.Series([_value(col, row) for row in rows], dtype=_dtype(col))
            for col in spec.columns
        }
    )
    assert c.check_frame(df, spec) == []
    return df


def _cohort(
    role: Role, players: Sequence[tuple[int, int, bool]], *, target: int = TARGET
) -> pd.DataFrame:
    """대상 표. players = (player_id, n_seasons, played_prev)."""
    rows = [
        {
            "player_id": pid,
            "role": role,
            "target_season": target,
            "window_first": target - c.WINDOW,
            "window_last": target - 1,
            "n_window": c.WINDOW,
            "input_first": target - c.WINDOW,
            "input_last": target - 1 if played_prev else target - 2,
            "n_seasons": n_seasons,
            "played_prev": played_prev,
            "sample": 1000,
            "sample_outs": 300 if role == "pit" else None,
        }
        for pid, n_seasons, played_prev in players
    ]
    return _frame(c.COHORTS, rows)


def _projections(
    role: Role,
    cohort: pd.DataFrame,
    preds: Mapping[str, Mapping[int, tuple[float, float, float]]],
    *,
    target: int = TARGET,
) -> pd.DataFrame:
    """모델 여러 개의 예측 표를 이어 붙인 것.

    preds: 모델 → player_id → 타자 (ops, pa, hr) / 투수 (era, k_bb_pct, ip).
    """
    # 대상 표에 없는 선수(오류 사례)는 임의의 기간 값을 쓴다
    info: dict[int, dict[Any, Any]] = {int(r["player_id"]): r for r in cohort.to_dict("records")}
    other = {"input_first": 2020, "input_last": 2024, "n_seasons": 1}
    rows: list[dict[str, Any]] = []
    for model, by_player in preds.items():
        for pid, (a, b, d) in by_player.items():
            span = info.get(pid, other)
            base: dict[str, Any] = {
                "player_id": pid,
                "role": role,
                "target_season": target,
                "model": model,
                "input_first": int(span["input_first"]),
                "input_last": int(span["input_last"]),
                "n_seasons": int(span["n_seasons"]),
            }
            if role == "bat":
                base.update(
                    {
                        "sample_pa": 1000,
                        "pa": b,
                        "hr": d,
                        "obp": a * 0.45,
                        "slg": a - a * 0.45,
                        "ops": a,
                        "lg_ops": 0.720,
                    }
                )
            else:
                base.update(
                    {
                        "sample_bf": 1000,
                        "sample_outs": 300,
                        "ip": d,
                        "era": a,
                        "k_bb_pct": b,
                        "lg_era": 4.0,
                        "lg_k_bb_pct": 15.0,
                    }
                )
            rows.append(base)
    return _frame(c.BAT_PROJECTIONS if role == "bat" else c.PIT_PROJECTIONS, rows)


# ---------------------------------------------------------------------------
# 타자 사례 (Y = 2025)
# ---------------------------------------------------------------------------

# 대상: (player_id, n_seasons, played_prev)
#  그룹 all {1..6}, seasons_1_2 {3,4,6}, seasons_3_5 {1,2,5}, prev_played {1,2,3,6}, prev_absent {4,5}
BAT_COHORT: list[tuple[int, int, bool]] = [
    (1, 5, True),
    (2, 4, True),
    (3, 2, True),
    (4, 1, False),
    (5, 3, False),
    (6, 2, True),
]

# Y 실제 기록: player_id → (pa, hr, ops). 5는 Y 미출전. 99는 대상이 아닌 신인 (무시해야 한다)
#  비율 대표 집단 (pa ≥ 200): {1, 2, 4, 6} — 3은 150타석으로 빠지고, 6은 정확히 200이라 들어간다
#  가중 집단 (pa > 0): {1, 2, 3, 4, 6}
BAT_ACTUAL: dict[int, tuple[int, int, float]] = {
    1: (600, 30, 0.800),
    2: (250, 10, 0.700),
    3: (150, 3, 0.650),
    4: (300, 12, 0.750),
    6: (200, 8, 0.900),
    99: (500, 25, 0.950),
}

# 예측: 모델 → player_id → (ops, pa, hr). league 모델의 출전량은 baseline과 같다
BAT_PREDS: dict[str, dict[int, tuple[float, float, float]]] = {
    "league": {
        1: (0.720, 580, 28),
        2: (0.720, 300, 12),
        3: (0.720, 100, 2),
        4: (0.720, 0, 0),
        5: (0.720, 0, 0),
        6: (0.720, 250, 10),
    },
    "baseline": {
        1: (0.850, 580, 28),
        2: (0.680, 300, 12),
        3: (0.600, 100, 2),
        4: (0.720, 0, 0),
        5: (0.720, 0, 0),
        6: (0.800, 250, 10),
    },
    "weighted": {
        1: (0.810, 590, 29),
        2: (0.690, 260, 11),
        3: (0.640, 170, 5),
        4: (0.740, 100, 4),
        5: (0.700, 60, 2),
        6: (0.860, 220, 9),
    },
}

# |오차| (선수 1, 2, 3, 4, 5, 6)
#  ops  league   .08 .02 .07 .03  —  .18
#       baseline .05 .02 .05 .03  —  .10
#       weighted .01 .01 .01 .01  —  .04
#  pa   baseline  20  50  50 300   0  50   (실제 600 250 150 300 0 200)
#       weighted  10  10  20 200  60  20
#  hr   baseline   2   2   1  12   0   2   (실제 30 10 3 12 0 8)
#       weighted   1   1   2   8   2   1
# ops 가중치(pa): 1→600, 2→250, 3→150, 4→300, 6→200
#  가중 |오차| 합: league   48 + 5 + 10.5 + 9 + 36 = 108.5
#                  baseline 30 + 5 +  7.5 + 9 + 20 =  71.5
#                  weighted  6 + 2.5 + 1.5 + 3 + 8 =  21
# (model, metric, group) → (n, mae, mae_weighted, excluded). None = 결측
Expected = dict[tuple[str, str, str], tuple[int, float | None, float | None, int]]
BAT_EXPECTED: Expected = {
    # ops · all: 대표 {1,2,4,6} n 4, 제외 6 − 4 = 2, 가중 Σpa 1500
    ("league", "ops", "all"): (4, (0.08 + 0.02 + 0.03 + 0.18) / 4, 108.5 / 1500, 2),
    ("baseline", "ops", "all"): (4, (0.05 + 0.02 + 0.03 + 0.10) / 4, 71.5 / 1500, 2),
    ("weighted", "ops", "all"): (4, (0.01 + 0.01 + 0.01 + 0.04) / 4, 21 / 1500, 2),
    # ops · seasons_1_2 {3,4,6}: 대표 {4,6} n 2, 제외 1, 가중 {3,4,6} Σpa 650
    ("league", "ops", "seasons_1_2"): (2, (0.03 + 0.18) / 2, (10.5 + 9 + 36) / 650, 1),
    ("baseline", "ops", "seasons_1_2"): (2, (0.03 + 0.10) / 2, (7.5 + 9 + 20) / 650, 1),
    ("weighted", "ops", "seasons_1_2"): (2, (0.01 + 0.04) / 2, (1.5 + 3 + 8) / 650, 1),
    # ops · seasons_3_5 {1,2,5}: 대표 {1,2} n 2, 제외 1, 가중 {1,2} Σpa 850
    ("league", "ops", "seasons_3_5"): (2, (0.08 + 0.02) / 2, (48 + 5) / 850, 1),
    ("baseline", "ops", "seasons_3_5"): (2, (0.05 + 0.02) / 2, (30 + 5) / 850, 1),
    ("weighted", "ops", "seasons_3_5"): (2, (0.01 + 0.01) / 2, (6 + 2.5) / 850, 1),
    # ops · prev_played {1,2,3,6}: 대표 {1,2,6} n 3, 제외 1, 가중 {1,2,3,6} Σpa 1200
    ("league", "ops", "prev_played"): (3, (0.08 + 0.02 + 0.18) / 3, (48 + 5 + 10.5 + 36) / 1200, 1),
    ("baseline", "ops", "prev_played"): (
        3,
        (0.05 + 0.02 + 0.10) / 3,
        (30 + 5 + 7.5 + 20) / 1200,
        1,
    ),
    ("weighted", "ops", "prev_played"): (
        3,
        (0.01 + 0.01 + 0.04) / 3,
        (6 + 2.5 + 1.5 + 8) / 1200,
        1,
    ),
    # ops · prev_absent {4,5}: 대표 {4} n 1, 제외 1, 가중 {4}
    ("league", "ops", "prev_absent"): (1, 0.03, 0.03, 1),
    ("baseline", "ops", "prev_absent"): (1, 0.03, 0.03, 1),
    ("weighted", "ops", "prev_absent"): (1, 0.01, 0.01, 1),
    # pa (출전량) — n = 그룹 대상 수, 제외 0, 가중 결측
    ("baseline", "pa", "all"): (6, (20 + 50 + 50 + 300 + 0 + 50) / 6, None, 0),
    ("weighted", "pa", "all"): (6, (10 + 10 + 20 + 200 + 60 + 20) / 6, None, 0),
    ("baseline", "pa", "seasons_1_2"): (3, (50 + 300 + 50) / 3, None, 0),
    ("weighted", "pa", "seasons_1_2"): (3, (20 + 200 + 20) / 3, None, 0),
    ("baseline", "pa", "seasons_3_5"): (3, (20 + 50 + 0) / 3, None, 0),
    ("weighted", "pa", "seasons_3_5"): (3, (10 + 10 + 60) / 3, None, 0),
    ("baseline", "pa", "prev_played"): (4, (20 + 50 + 50 + 50) / 4, None, 0),
    ("weighted", "pa", "prev_played"): (4, (10 + 10 + 20 + 20) / 4, None, 0),
    ("baseline", "pa", "prev_absent"): (2, (300 + 0) / 2, None, 0),
    ("weighted", "pa", "prev_absent"): (2, (200 + 60) / 2, None, 0),
    # hr (출전량)
    ("baseline", "hr", "all"): (6, (2 + 2 + 1 + 12 + 0 + 2) / 6, None, 0),
    ("weighted", "hr", "all"): (6, (1 + 1 + 2 + 8 + 2 + 1) / 6, None, 0),
    ("baseline", "hr", "seasons_1_2"): (3, (1 + 12 + 2) / 3, None, 0),
    ("weighted", "hr", "seasons_1_2"): (3, (2 + 8 + 1) / 3, None, 0),
    ("baseline", "hr", "seasons_3_5"): (3, (2 + 2 + 0) / 3, None, 0),
    ("weighted", "hr", "seasons_3_5"): (3, (1 + 1 + 2) / 3, None, 0),
    ("baseline", "hr", "prev_played"): (4, (2 + 2 + 1 + 2) / 4, None, 0),
    ("weighted", "hr", "prev_played"): (4, (1 + 1 + 2 + 1) / 4, None, 0),
    ("baseline", "hr", "prev_absent"): (2, (12 + 0) / 2, None, 0),
    ("weighted", "hr", "prev_absent"): (2, (8 + 2) / 2, None, 0),
}


def _bat_seasons(*, poison_other_seasons: bool = False) -> pd.DataFrame:
    """타자 정제 표: Y 실제 행 + Y−1 행 (평가에 쓰이면 안 된다).

    poison_other_seasons: Y−1 행을 엉뚱한 값으로 바꾸고 Y+1 행(대상 전원 + 새 선수)을 더한다.
    """
    rows: list[dict[str, Any]] = [
        {"player_id": pid, "season": TARGET, "pa": pa, "hr": hr, "ops": ops}
        for pid, (pa, hr, ops) in BAT_ACTUAL.items()
    ]
    prev = {1: 550, 2: 320, 3: 120, 6: 260}
    for pid, pa in prev.items():
        value = POISON if poison_other_seasons else pa
        rows.append({"player_id": pid, "season": TARGET - 1, "pa": value, "hr": 1, "ops": 0.5})
    if poison_other_seasons:
        rows += [
            {"player_id": pid, "season": TARGET + 1, "pa": POISON, "hr": POISON, "ops": 9.9}
            for pid in (1, 2, 3, 4, 5, 6, 777)
        ]
        rows.reverse()
    return _frame(c.BATTING_SEASONS, rows)


def _bat_case() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(예측 표, 정제 표, 대상 표)."""
    cohort = _cohort("bat", BAT_COHORT)
    return _projections("bat", cohort, BAT_PREDS), _bat_seasons(), cohort


# ---------------------------------------------------------------------------
# 투수 사례 (Y = 2025)
# ---------------------------------------------------------------------------

# 대상: 전원 n_seasons ≥ 3 → seasons_1_2 그룹은 비어 있다 (n = 0)
#  그룹 all = seasons_3_5 = {21..25}, prev_played {21,22,23,25}, prev_absent {24}
PIT_COHORT: list[tuple[int, int, bool]] = [
    (21, 5, True),
    (22, 3, True),
    (23, 4, True),
    (24, 3, False),
    (25, 4, True),
]

# Y 실제 기록: player_id → (outs, bf, er, so, bb). 24는 Y 미출전, 98은 대상이 아닌 신인
#  21: era 27·60/540 = 3.0, k_bb 100·(200 − 50)/750 = 20.0, ip 540/3 = 180
#  22: era 27·25/150 = 4.5, k_bb 100·(42 − 21)/210 = 10.0, ip 50  (outs 150 = 기준선 → 대표 집단)
#  23: era 27·20/90 = 6.0,  k_bb 100·(28 − 14)/140 = 10.0, ip 30  (outs 90 < 150 → 대표 집단 밖)
#  25: outs 0 → era 결측, k_bb 100·(0 − 3)/3 = −100, ip 0
#  비율 대표 집단 (outs ≥ 150): {21, 22} — k_bb_pct도 outs 기준이다
#  era 가중 집단 (outs > 0): {21, 22, 23}, 가중치 540·150·90 (합 780)
#  k_bb 가중 집단 (bf > 0): {21, 22, 23, 25}, 가중치 750·210·140·3 (합 1103)
PIT_ACTUAL: dict[int, tuple[int, int, int, int, int]] = {
    21: (540, 750, 60, 200, 50),
    22: (150, 210, 25, 42, 21),
    23: (90, 140, 20, 28, 14),
    25: (0, 3, 2, 0, 3),
    98: (300, 400, 30, 100, 30),
}

# 예측: 모델 → player_id → (era, k_bb_pct, ip)
PIT_PREDS: dict[str, dict[int, tuple[float, float, float]]] = {
    "league": {
        21: (4.0, 15.0, 170.0),
        22: (4.0, 15.0, 60.0),
        23: (4.0, 15.0, 40.0),
        24: (4.0, 15.0, 0.0),
        25: (4.0, 15.0, 1.0),
    },
    "baseline": {
        21: (3.5, 18.0, 170.0),
        22: (4.0, 12.0, 60.0),
        23: (5.0, 8.0, 40.0),
        24: (4.0, 15.0, 0.0),
        25: (4.0, -20.0, 1.0),
    },
    "weighted": {
        21: (3.2, 19.0, 175.0),
        22: (4.4, 11.0, 55.0),
        23: (5.5, 9.0, 35.0),
        24: (4.1, 14.0, 20.0),
        25: (4.2, 5.0, 10.0),
    },
}

# |오차| (선수 21, 22, 23, 24, 25)
#  era      league   1.0 0.5 2.0  —   —
#           baseline 0.5 0.5 1.0  —   —
#           weighted 0.2 0.1 0.5  —   —
#  k_bb_pct league     5   5   5  —  115
#           baseline   2   2   2  —   80
#           weighted   1   1   1  —  105
#  ip       baseline  10  10  10   0   1   (실제 180 50 30 0 0)
#           weighted   5   5   5  20  10
# era 가중 |오차| 합: league 540·1 + 150·0.5 + 90·2 = 795, baseline 270 + 75 + 90 = 435,
#                     weighted 108 + 15 + 45 = 168
# k_bb 가중 |오차| 합: league 3750 + 1050 + 700 + 345 = 5845, baseline 1500 + 420 + 280 + 240 = 2440,
#                      weighted 750 + 210 + 140 + 315 = 1415
_PIT_RATE_ROWS: dict[tuple[str, str], tuple[float, float]] = {
    # (model, metric) → (mae, mae_weighted) — all·seasons_3_5·prev_played 공통
    ("league", "era"): ((1.0 + 0.5) / 2, 795 / 780),
    ("baseline", "era"): ((0.5 + 0.5) / 2, 435 / 780),
    ("weighted", "era"): ((0.2 + 0.1) / 2, 168 / 780),
    ("league", "k_bb_pct"): ((5 + 5) / 2, 5845 / 1103),
    ("baseline", "k_bb_pct"): ((2 + 2) / 2, 2440 / 1103),
    ("weighted", "k_bb_pct"): ((1 + 1) / 2, 1415 / 1103),
}
PIT_EXPECTED: Expected = {}
for (_model, _metric), (_mae, _mw) in _PIT_RATE_ROWS.items():
    # all {21..25}: n 2, 제외 5 − 2 = 3 / seasons_3_5 = all / prev_played {21,22,23,25}: 제외 4 − 2 = 2
    PIT_EXPECTED[(_model, _metric, "all")] = (2, _mae, _mw, 3)
    PIT_EXPECTED[(_model, _metric, "seasons_3_5")] = (2, _mae, _mw, 3)
    PIT_EXPECTED[(_model, _metric, "prev_played")] = (2, _mae, _mw, 2)
    # prev_absent {24}: Y 미출전 → n 0, 가중 집단도 비어 결측, 제외 1
    PIT_EXPECTED[(_model, _metric, "prev_absent")] = (0, None, None, 1)
    # seasons_1_2: 그룹이 비었다 → n 0, 제외 0
    PIT_EXPECTED[(_model, _metric, "seasons_1_2")] = (0, None, None, 0)
PIT_EXPECTED.update(
    {
        ("baseline", "ip", "all"): (5, (10 + 10 + 10 + 0 + 1) / 5, None, 0),
        ("weighted", "ip", "all"): (5, (5 + 5 + 5 + 20 + 10) / 5, None, 0),
        ("baseline", "ip", "seasons_3_5"): (5, (10 + 10 + 10 + 0 + 1) / 5, None, 0),
        ("weighted", "ip", "seasons_3_5"): (5, (5 + 5 + 5 + 20 + 10) / 5, None, 0),
        ("baseline", "ip", "prev_played"): (4, (10 + 10 + 10 + 1) / 4, None, 0),
        ("weighted", "ip", "prev_played"): (4, (5 + 5 + 5 + 10) / 4, None, 0),
        ("baseline", "ip", "prev_absent"): (1, 0.0, None, 0),
        ("weighted", "ip", "prev_absent"): (1, 20.0, None, 0),
        ("baseline", "ip", "seasons_1_2"): (0, None, None, 0),
        ("weighted", "ip", "seasons_1_2"): (0, None, None, 0),
    }
)


def _pit_seasons() -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"player_id": pid, "season": TARGET, "outs": o, "bf": bf, "er": er, "so": so, "bb": bb}
        for pid, (o, bf, er, so, bb) in PIT_ACTUAL.items()
    ]
    rows.append({"player_id": 24, "season": TARGET - 1, "outs": 300, "bf": 400, "er": 40})
    return _frame(c.PITCHING_SEASONS, rows)


def _pit_case() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort = _cohort("pit", PIT_COHORT)
    return _projections("pit", cohort, PIT_PREDS), _pit_seasons(), cohort


# ---------------------------------------------------------------------------
# 결과 확인 도우미
# ---------------------------------------------------------------------------


def _same(actual: Any, expected: float | None) -> bool:
    if expected is None:
        return bool(pd.isna(actual))
    return bool(not pd.isna(actual) and actual == pytest.approx(expected, rel=1e-9, abs=1e-12))


def _rows(out: pd.DataFrame) -> dict[tuple[str, str, str], dict[Any, Any]]:
    return {(str(r["model"]), str(r["metric"]), str(r["group"])): r for r in out.to_dict("records")}


def _assert_table(out: pd.DataFrame, expected: Expected, role: Role) -> None:
    assert c.check_frame(out, c.EVALUATIONS) == []
    rows = _rows(out)
    assert set(rows) == set(expected)
    for key, (n, mae, mae_weighted, excluded) in expected.items():
        row = rows[key]
        assert int(row["target_season"]) == TARGET, key
        assert row["role"] == role, key
        assert int(row["n"]) == n, (key, "n", row["n"])
        assert _same(row["mae"], mae), (key, "mae", row["mae"], mae)
        assert _same(row["mae_weighted"], mae_weighted), (key, "mae_weighted", row["mae_weighted"])
        assert int(row["excluded"]) == excluded, (key, "excluded", row["excluded"])


def _assert_sorted_by_keys(out: pd.DataFrame) -> None:
    keys = [tuple(r) for r in out[list(c.EVALUATIONS.keys)].itertuples(index=False)]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# actual_values
# ---------------------------------------------------------------------------


def test_evaluate_metrics_cover_contract_metrics() -> None:
    # METRICS는 역할별 지표(contracts)와 같은 이름을 쓴다
    assert {m.name for m in METRICS if m.role == "bat"} == set(c.BAT_METRICS)
    assert {m.name for m in METRICS if m.role == "pit"} == set(c.PIT_METRICS)
    assert {m.name for m in METRICS if m.kind == "rate"} == set(c.RATE_METRICS)


def test_evaluate_actual_values_bat() -> None:
    rows: list[dict[str, Any]] = [
        {"player_id": 1, "season": 2025, "pa": 600, "ab": 540, "hr": 30, "ops": 0.800},
        # 희생번트 1타석 — 정제 표 ops가 결측이면 그대로 결측
        {"player_id": 2, "season": 2025, "pa": 1, "sh": 1, "hr": 0, "ops": None},
        {"player_id": 1, "season": 2024, "pa": 500, "hr": 20, "ops": 0.700},  # 다른 시즌
        {"player_id": 3, "season": 2026, "pa": 400, "hr": 10, "ops": 0.750},  # 다른 시즌
    ]
    out = actual_values(_frame(c.BATTING_SEASONS, rows), "bat", 2025)
    # 열: player_id, 지표(ops, pa, hr — METRICS 순), 분모 pa (지표 pa와 같은 열)
    assert list(out.columns) == ["player_id", "ops", "pa", "hr"]
    out = out.sort_values("player_id").reset_index(drop=True)
    # Y에 행이 있는 선수만
    assert out["player_id"].tolist() == [1, 2]
    ops = [float(v) for v in out["ops"]]
    assert ops[0] == pytest.approx(0.800)  # 정제 표 ops 그대로
    assert pd.isna(ops[1])
    assert [float(v) for v in out["pa"]] == [600.0, 1.0]
    assert [float(v) for v in out["hr"]] == [30.0, 0.0]


def test_evaluate_actual_values_pit() -> None:
    rows = [
        {"player_id": 21, "season": 2025, "outs": 540, "bf": 750, "er": 60, "so": 200, "bb": 50},
        {"player_id": 25, "season": 2025, "outs": 0, "bf": 3, "er": 2, "so": 0, "bb": 3},
        {"player_id": 26, "season": 2025, "outs": 0, "bf": 0},  # bf·outs 모두 0
        {"player_id": 21, "season": 2024, "outs": 300, "bf": 400, "er": 10},  # 다른 시즌
    ]
    out = actual_values(_frame(c.PITCHING_SEASONS, rows), "pit", 2025)
    # 열: player_id, 지표(era, k_bb_pct, ip — METRICS 순), 분모 outs·bf
    assert list(out.columns) == ["player_id", "era", "k_bb_pct", "ip", "outs", "bf"]
    rec = {int(r["player_id"]): r for r in out.to_dict("records")}
    assert set(rec) == {21, 25, 26}
    # 21: era = 27·60/540 = 3.0, k_bb_pct = 100·(200 − 50)/750 = 20.0, ip = 540/3 = 180
    assert rec[21]["era"] == pytest.approx(3.0)
    assert rec[21]["k_bb_pct"] == pytest.approx(20.0)
    assert rec[21]["ip"] == pytest.approx(180.0)
    assert (float(rec[21]["outs"]), float(rec[21]["bf"])) == (540.0, 750.0)
    # 25: outs 0 → era 결측, k_bb_pct = 100·(0 − 3)/3 = −100, ip = 0
    assert bool(pd.isna(rec[25]["era"]))
    assert rec[25]["k_bb_pct"] == pytest.approx(-100.0)
    assert rec[25]["ip"] == 0.0
    # 26: bf 0 → k_bb_pct도 결측
    assert bool(pd.isna(rec[26]["era"])) and bool(pd.isna(rec[26]["k_bb_pct"]))
    assert rec[26]["ip"] == 0.0


def test_evaluate_actual_values_no_rows_in_target() -> None:
    rows = [{"player_id": 1, "season": 2024, "pa": 500, "hr": 20, "ops": 0.700}]
    out = actual_values(_frame(c.BATTING_SEASONS, rows), "bat", 2025)
    assert len(out) == 0
    assert list(out.columns) == ["player_id", "ops", "pa", "hr"]


# ---------------------------------------------------------------------------
# evaluate — 손 계산 표
# ---------------------------------------------------------------------------


def test_evaluate_bat_matches_hand_computed_table() -> None:
    out = evaluate(*_bat_case(), "bat")
    _assert_table(out, BAT_EXPECTED, "bat")
    _assert_sorted_by_keys(out)


def test_evaluate_pit_matches_hand_computed_table() -> None:
    out = evaluate(*_pit_case(), "pit")
    _assert_table(out, PIT_EXPECTED, "pit")
    _assert_sorted_by_keys(out)


def test_evaluate_empty_group_has_nan_mae() -> None:
    # 투수 seasons_1_2 그룹은 비었다 → 모든 지표에서 n 0, MAE 열 결측, 제외 0
    out = evaluate(*_pit_case(), "pit")
    empty = out[out["group"] == "seasons_1_2"]
    assert len(empty) == 3 * 2 + 2  # 비율 2지표 × 3모델 + ip × 2모델
    assert (empty["n"] == 0).all()
    assert (empty["excluded"] == 0).all()
    for col in ("mae", "mae_weighted", "baseline_mae_league", "baseline_mae_prev"):
        assert bool(empty[col].isna().all()), col


def test_evaluate_baselines_equal_league_and_prev_model_mae() -> None:
    # 모든 행의 기준선 = 같은 (지표, 그룹)의 "league"·"baseline" 행 mae (n = 0이면 둘 다 결측)
    cases: list[tuple[Role, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]] = [
        ("bat", _bat_case()),
        ("pit", _pit_case()),
    ]
    for role, (projections, seasons, cohort) in cases:
        rows = _rows(evaluate(projections, seasons, cohort, role))
        for (model, metric, group), row in rows.items():
            prev = rows[("baseline", metric, group)]["mae"]
            assert _same(row["baseline_mae_prev"], None if pd.isna(prev) else float(prev)), (
                model,
                metric,
                group,
            )
            if metric in c.RATE_METRICS:
                lg = rows[("league", metric, group)]["mae"]
                assert _same(row["baseline_mae_league"], None if pd.isna(lg) else float(lg)), (
                    model,
                    metric,
                    group,
                )
    # 손 계산 값으로도 확인: 타자 ops · all → 리그 .31/4, 직전 .20/4 (세 모델 행 모두)
    bat_rows = _rows(evaluate(*_bat_case(), "bat"))
    for model in ("league", "baseline", "weighted"):
        row = bat_rows[(model, "ops", "all")]
        assert row["baseline_mae_league"] == pytest.approx(0.31 / 4)
        assert row["baseline_mae_prev"] == pytest.approx(0.20 / 4)
    # 출전량: 타자 pa · prev_absent → 직전 150
    assert bat_rows[("weighted", "pa", "prev_absent")]["baseline_mae_prev"] == pytest.approx(150.0)


def test_evaluate_league_model_has_only_rate_metrics() -> None:
    bat = evaluate(*_bat_case(), "bat")
    assert set(bat.loc[bat["model"] == "league", "metric"]) == {"ops"}
    assert set(bat.loc[bat["model"] == "baseline", "metric"]) == {"ops", "pa", "hr"}
    pit = evaluate(*_pit_case(), "pit")
    assert set(pit.loc[pit["model"] == "league", "metric"]) == {"era", "k_bb_pct"}
    assert set(pit.loc[pit["model"] == "weighted", "metric"]) == {"era", "k_bb_pct", "ip"}


def test_evaluate_rate_population_threshold_boundary() -> None:
    # 선수 6의 Y 타석을 RATE_MIN_PA − 1(199)로 낮추면 ops 대표 집단에서 빠진다 (가중 집단에는 남는다)
    projections, seasons, cohort = _bat_case()
    seasons = seasons.copy()
    mask = (seasons["player_id"] == 6) & (seasons["season"] == TARGET)
    seasons.loc[mask, "pa"] = c.RATE_MIN_PA - 1
    rows = _rows(evaluate(projections, seasons, cohort, "bat"))
    row = rows[("weighted", "ops", "all")]
    # 대표 {1,2,4}: (.01 + .01 + .01)/3, 제외 6 − 3 = 3
    assert int(row["n"]) == 3 and int(row["excluded"]) == 3
    assert row["mae"] == pytest.approx(0.01)
    # 가중: 6 + 2.5 + 1.5 + 3 + 199·0.04 = 20.96, Σpa 1499
    assert row["mae_weighted"] == pytest.approx(20.96 / 1499)


def test_evaluate_ignores_other_seasons() -> None:
    # 평가는 Y 행만 실제 값으로 쓴다: Y−1 행을 바꾸고 Y+1 행을 더해도 결과가 같다 (N2)
    projections, seasons, cohort = _bat_case()
    base = evaluate(projections, seasons, cohort, "bat")
    poisoned = evaluate(projections, _bat_seasons(poison_other_seasons=True), cohort, "bat")
    assert_frame_equal(base.reset_index(drop=True), poisoned.reset_index(drop=True))


def test_evaluate_without_weighted_model() -> None:
    # league·baseline만 있어도 된다 — 표에는 있는 모델만 나온다
    cohort = _cohort("bat", BAT_COHORT)
    preds = {k: BAT_PREDS[k] for k in ("league", "baseline")}
    out = evaluate(_projections("bat", cohort, preds), _bat_seasons(), cohort, "bat")
    assert set(out["model"]) == {"league", "baseline"}
    _assert_table(out, {k: v for k, v in BAT_EXPECTED.items() if k[0] != "weighted"}, "bat")


# ---------------------------------------------------------------------------
# 오류
# ---------------------------------------------------------------------------


def _without(preds: Mapping[str, Any], model: str) -> dict[str, Any]:
    return {k: v for k, v in preds.items() if k != model}


def _bad_cases() -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Role]]:
    cohort = _cohort("bat", BAT_COHORT)
    seasons = _bat_seasons()
    good = _projections("bat", cohort, BAT_PREDS)
    missing_player = {
        **BAT_PREDS,
        "weighted": {k: v for k, v in BAT_PREDS["weighted"].items() if k != 6},
    }
    extra_player = {m: {**p, 77: (0.7, 100.0, 3.0)} for m, p in BAT_PREDS.items()}
    other_target = _projections("bat", cohort, BAT_PREDS, target=TARGET + 1)
    mixed_target = pd.concat(
        [
            _projections("bat", cohort, _without(BAT_PREDS, "weighted")),
            _projections("bat", cohort, {"weighted": BAT_PREDS["weighted"]}, target=TARGET + 1),
        ],
        ignore_index=True,
    )
    mixed_role = good.copy()
    mixed_role.loc[mixed_role["model"] == "weighted", "role"] = "pit"
    cohort_mixed_target = pd.concat(
        [cohort, _cohort("bat", [(9, 2, True)], target=TARGET + 1)], ignore_index=True
    )
    pit_cohort = _cohort("pit", [(pid, n, p) for pid, n, p in BAT_COHORT])
    return {
        "no_league_model": (
            _projections("bat", cohort, _without(BAT_PREDS, "league")),
            seasons,
            cohort,
            "bat",
        ),
        "no_baseline_model": (
            _projections("bat", cohort, _without(BAT_PREDS, "baseline")),
            seasons,
            cohort,
            "bat",
        ),
        "projection_target_differs": (other_target, seasons, cohort, "bat"),
        "projection_mixed_target": (mixed_target, seasons, cohort, "bat"),
        "projection_mixed_role": (mixed_role, seasons, cohort, "bat"),
        "cohort_mixed_target": (good, seasons, cohort_mixed_target, "bat"),
        "cohort_other_role": (good, seasons, pit_cohort, "bat"),
        "role_argument_differs": (good, seasons, cohort, "pit"),
        "player_missing_in_one_model": (
            _projections("bat", cohort, missing_player),
            seasons,
            cohort,
            "bat",
        ),
        "player_not_in_cohort": (_projections("bat", cohort, extra_player), seasons, cohort, "bat"),
        "projection_contract": (good.drop(columns=["lg_ops"]), seasons, cohort, "bat"),
        "seasons_contract": (good, seasons.drop(columns=["ops"]), cohort, "bat"),
        "cohort_contract": (good, seasons, cohort.drop(columns=["n_seasons"]), "bat"),
        # contracts.MODELS에 없는 모델 이름
        "unknown_model": (
            _projections("bat", cohort, {**BAT_PREDS, "marcel": BAT_PREDS["weighted"]}),
            seasons,
            cohort,
            "bat",
        ),
        # 한 선수의 Y 행이 리그 둘 (계약은 통과하지만 실제 값이 모호하다, D1)
        "two_target_rows_for_one_player": (
            good,
            pd.concat([seasons, seasons.iloc[[0]].assign(league="KBO")], ignore_index=True),
            cohort,
            "bat",
        ),
    }


@pytest.mark.parametrize("case", list(_bad_cases()))
def test_evaluate_rejects_bad_inputs(case: str) -> None:
    projections, seasons, cohort, role = _bad_cases()[case]
    with pytest.raises(ValueError):
        evaluate(projections, seasons, cohort, role)


def test_evaluate_expected_tables_are_complete() -> None:
    # 기대 표 자체 점검: 모델 × 지표 × 그룹이 빠짐없다 (league는 비율 지표만)
    for role, expected in (("bat", BAT_EXPECTED), ("pit", PIT_EXPECTED)):
        metrics = [m for m in METRICS if m.role == role]
        keys = {
            (model, m.name, g)
            for m in metrics
            for model in ("league", "baseline", "weighted")
            if m.kind == "rate" or model != "league"
            for g in c.GROUPS
        }
        assert set(expected) == keys


def test_evaluate_player_mismatch_message_names_players() -> None:
    # 어느 모델에서 어느 선수가 빠졌는지·남는지 적는다
    cohort = _cohort("bat", BAT_COHORT)
    missing = {**BAT_PREDS, "weighted": {k: v for k, v in BAT_PREDS["weighted"].items() if k != 6}}
    with pytest.raises(ValueError, match=r"weighted.*예측 없음 \[6\]"):
        evaluate(_projections("bat", cohort, missing), _bat_seasons(), cohort, "bat")
    extra = {m: {**p, 77: (0.7, 100.0, 3.0)} for m, p in BAT_PREDS.items()}
    with pytest.raises(ValueError, match=r"대상 아님 \[77\]"):
        evaluate(_projections("bat", cohort, extra), _bat_seasons(), cohort, "bat")


# ---------------------------------------------------------------------------
# docstring 경계 규칙 고정 (Green이 열어 둔 해석)
# ---------------------------------------------------------------------------


def test_evaluate_volume_rows_have_no_league_baseline() -> None:
    # "league" 모델은 비율 지표 행만 낸다 → 출전량 행의 baseline_mae_league는 결측 (E11 표의 —)
    cases: list[tuple[Role, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]] = [
        ("bat", _bat_case()),
        ("pit", _pit_case()),
    ]
    for role, case in cases:
        out = evaluate(*case, role)
        volume = out[~out["metric"].isin(c.RATE_METRICS)]
        assert len(volume) > 0
        assert bool(volume["baseline_mae_league"].isna().all()), role
        assert not bool((volume["model"] == "league").any()), role


def test_evaluate_zero_representatives_blank_weighted_mae() -> None:
    # 선수 4의 Y 타석을 150으로 낮춘다 → prev_absent {4,5}의 대표 집단(pa ≥ 200)이 빈다
    #   n = 0 → "n = 0이면 MAE 열은 결측": mae·mae_weighted·기준선 모두 결측 (가중 집단 {4}는 있지만)
    #   excluded = 2 − 0 = 2
    projections, seasons, cohort = _bat_case()
    seasons = seasons.copy()
    seasons.loc[(seasons["player_id"] == 4) & (seasons["season"] == TARGET), "pa"] = 150
    rows = _rows(evaluate(projections, seasons, cohort, "bat"))
    for model in ("league", "baseline", "weighted"):
        row = rows[(model, "ops", "prev_absent")]
        assert int(row["n"]) == 0 and int(row["excluded"]) == 2, model
        for col in ("mae", "mae_weighted", "baseline_mae_league", "baseline_mae_prev"):
            assert bool(pd.isna(row[col])), (model, col)
    # all 그룹에는 대표 {1,2,6}가 있어 가중 MAE에 선수 4(150타석)가 그대로 들어간다
    #   league 가중 합 = 48 + 5 + 10.5 + 150·0.03 + 36 = 104, Σpa = 600 + 250 + 150 + 150 + 200 = 1350
    row = rows[("league", "ops", "all")]
    assert int(row["n"]) == 3 and int(row["excluded"]) == 3
    assert row["mae_weighted"] == pytest.approx(104 / 1350)


def test_evaluate_missing_actual_ops_leaves_weighted_population() -> None:
    # 선수 3의 Y 행을 희생번트 2타석(pa 2, ab 0)으로 바꾼다 → 정제 표 ops 결측 → 오차를 잴 수 없다
    #   타석은 있어도 ops 가중 집단에서 빠진다: league 가중 합 108.5 − 10.5 = 98, Σpa 1500 − 150 = 1350
    #   대표 집단(n 4)·제외(2)는 그대로 (3은 원래 150타석이라 대표 밖)
    projections, seasons, cohort = _bat_case()
    seasons = seasons.copy()
    mask = (seasons["player_id"] == 3) & (seasons["season"] == TARGET)
    seasons.loc[mask, ["pa", "sh", "hr"]] = [2, 2, 0]
    seasons.loc[mask, "ops"] = float("nan")
    row = _rows(evaluate(projections, seasons, cohort, "bat"))[("league", "ops", "all")]
    assert int(row["n"]) == 4 and int(row["excluded"]) == 2
    assert row["mae_weighted"] == pytest.approx(98 / 1350)


# ---------------------------------------------------------------------------
# 누설 방지 (N2) · 결정성
# ---------------------------------------------------------------------------


def test_evaluate_ignores_other_seasons_pit() -> None:
    # 투수: Y−1 행을 엉뚱하게 바꾸고 Y+1 행(대상 전원 + 새 선수)을 더해도 결과가 같다 (N2)
    projections, seasons, cohort = _pit_case()
    base = evaluate(projections, seasons, cohort, "pit")
    other = seasons["season"] != TARGET
    poisoned = seasons.copy()
    poisoned.loc[other, ["outs", "bf", "er", "so"]] = POISON
    future = [
        {"player_id": pid, "season": TARGET + 1, "outs": POISON, "bf": POISON, "er": POISON}
        for pid in (21, 22, 23, 24, 25, 555)
    ]
    poisoned = pd.concat([_frame(c.PITCHING_SEASONS, future), poisoned], ignore_index=True)
    assert c.check_frame(poisoned, c.PITCHING_SEASONS) == []
    assert_frame_equal(base, evaluate(projections, poisoned, cohort, "pit"))


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_evaluate_row_order_does_not_matter(role: Role) -> None:
    # 예측·정제·대상 표의 행 순서를 뒤집어도 결과가 정확히 같다 (키 순 정렬)
    projections, seasons, cohort = _bat_case() if role == "bat" else _pit_case()
    assert_frame_equal(
        evaluate(projections, seasons, cohort, role),
        evaluate(_flipped(projections), _flipped(seasons), _flipped(cohort), role),
    )


def _flipped(df: pd.DataFrame) -> pd.DataFrame:
    """행 순서를 뒤집은 표 (색인은 0부터 다시)."""
    return df.iloc[::-1].reset_index(drop=True)
