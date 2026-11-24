"""출전량 모델(models/playing_time.py) 테스트 (SPEC F8·D9·D12·E11·N2).

입력은 이 파일에서 만든 합성 정제 표다. `_frame`이 `contracts.BATTING_SEASONS`·`PITCHING_SEASONS`의
모든 열을 유효한 값으로 채운다(정제 모듈은 쓰지 않는다). 대상 표는 `cohort.build_cohort`로 만든다.

두 가지 합성 데이터를 쓴다.

- 특성 표(`FEAT_*`): 특성 공식(pt1·pt2·gap·n_seasons·gs_share·const)을 손으로 계산해 맞춘다.
- 선형 세계(`*_WORLD`): 모든 학습 행의 정답이 정확히 한 선형 규칙을 따르도록 시즌 기록을 거꾸로
  만들었다. 그래서 최소제곱은 규칙의 계수를 그대로 되찾고, 예측은 규칙으로 손으로 계산된다.

    타자 규칙: pa(T) = 1.5·pt1 − 0.5·pt2 − 50·gap + 0·n_seasons + 100
    투수 규칙: outs(T) = 1.5·pt1 − 0.5·pt2 − 50·gap + 0·n_seasons + 40·gs_share + 100

  규칙 값이 0이면 그 시즌에 행이 없다(정답 0). 2020 기록은 ×162/60 환산값(D9)이 특성이 된다.
"""

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.api import types as ptypes
from pandas.testing import assert_frame_equal

from mlbproj import cohort as cohort_mod
from mlbproj import contracts as c
from mlbproj.cohort import build_cohort
from mlbproj.contracts import Role
from mlbproj.models import playing_time as pt_mod
from mlbproj.models.playing_time import (
    FEATURES,
    fit_coefficients,
    fit_segments,
    playing_time_features,
    predict_playing_time,
    training_frame,
)

SPECS: dict[Role, c.TableSpec] = {"bat": c.BATTING_SEASONS, "pit": c.PITCHING_SEASONS}
PT_SPECS: dict[Role, c.TableSpec] = {"bat": c.BAT_PLAYING_TIME, "pit": c.PIT_PLAYING_TIME}
PT_COLUMN: dict[Role, str] = {"bat": "pa", "pit": "outs"}
TRAIN_LEAD = ["player_id", "train_target"]
FIRST_TRAIN_TARGET = c.SEASON_FIRST + 2  # 2021 — T−2가 데이터(2019~) 안에 있어야 한다
POISON = 99_999  # 목표 시즌 이후 행에 넣는 "독" 값 (N2)
NEW_PLAYER = 999  # 목표 시즌 이후에만 기록이 있는 선수
TOL = 1e-6

# ---------------------------------------------------------------------------
# 특성 표 — 손 계산용
# ---------------------------------------------------------------------------

# 타자: player_id → {season: pa}
FEAT_BAT: dict[int, dict[int, int]] = {
    301: {2022: 500, 2023: 520},
    302: {2019: 300, 2020: 100, 2021: 400},  # 2020 pa 100 → pa_adj 100 × 162/60 = 270
    303: {2022: 250},
    304: {2023: 50},
}
# 투수: player_id → {season: (g, gs, outs)}
FEAT_PIT: dict[int, dict[int, tuple[int, int, int]]] = {
    401: {2019: (10, 10, 300), 2021: (20, 5, 200), 2024: (30, 0, 90)},
    # 2020 outs 50 → outs_adj 50 × 2.7 = 135. 두 시즌 모두 gs/g = 0.5라 G·GS 환산 여부와 무관하다
    402: {2020: (10, 5, 50), 2023: (8, 4, 60)},
    403: {2022: (0, 0, 0)},  # Σg = 0
}

# 목표 T → player_id → FEATURES 순서의 기대값
FEAT_BAT_EXPECTED: dict[int, dict[int, tuple[float, ...]]] = {
    # T=2024, 기간 2019~2023 (input_last 기준 gap = 2023 − input_last)
    2024: {
        301: (520.0, 500.0, 0.0, 2.0, 1.0),  # pt1 = 2023, pt2 = 2022, 2022·2023 두 시즌
        302: (0.0, 0.0, 2.0, 3.0, 1.0),  # 2022·2023 없음, gap = 2023 − 2021 = 2, 2019~2021 세 시즌
        303: (0.0, 250.0, 1.0, 1.0, 1.0),  # gap = 2023 − 2022 = 1
        304: (50.0, 0.0, 0.0, 1.0, 1.0),  # 2023 데뷔 → pt2 = 0
    },
    # T=2021, 기간 2016~2020: 302만 대상. pt1 = 2020 환산값 270 (D9), pt2 = 2019 pa 300
    2021: {302: (270.0, 300.0, 0.0, 2.0, 1.0)},
    # T=2022, 기간 2017~2021: 302만 대상. pt1 = 2021 pa 400, pt2 = 2020 환산값 270
    2022: {302: (400.0, 270.0, 0.0, 3.0, 1.0)},
}
FEAT_PIT_EXPECTED: dict[int, dict[int, tuple[float, ...]]] = {
    # T=2025, 기간 2020~2024 — 401의 2019 행(10/10)은 기간 밖이라 gs_share에 넣지 않는다
    2025: {
        401: (90.0, 0.0, 0.0, 2.0, 0.1, 1.0),  # gs_share = (5 + 0) / (20 + 30) = 0.1 (넣으면 0.25)
        402: (
            0.0,
            60.0,
            1.0,
            2.0,
            0.5,
            1.0,
        ),  # gap = 2024 − 2023 = 1, gs_share = (5 + 4) / (10 + 8)
        403: (0.0, 0.0, 2.0, 1.0, 0.0, 1.0),  # gap = 2024 − 2022 = 2, Σg = 0 → gs_share 0
    },
    # T=2021, 기간 2016~2020: 401(2019), 402(2020)
    2021: {
        401: (0.0, 300.0, 1.0, 1.0, 1.0, 1.0),  # 2020 없음, gap = 2020 − 2019 = 1, 10/10
        402: (135.0, 0.0, 0.0, 1.0, 0.5, 1.0),  # pt1 = 50 × 2.7 = 135 (D9), 5/10
    },
    # T=2022, 기간 2017~2021: 401(2019·2021), 402(2020)
    2022: {
        401: (200.0, 0.0, 0.0, 2.0, 0.5, 1.0),  # gs_share = (10 + 5) / (10 + 20) = 0.5
        402: (0.0, 135.0, 1.0, 1.0, 0.5, 1.0),  # pt2 = 2020 환산값 135, gap = 2021 − 2020 = 1
    },
}

# ---------------------------------------------------------------------------
# 선형 세계 — 정답이 규칙을 정확히 따르는 시즌 기록
# ---------------------------------------------------------------------------

BAT_BETA = (1.5, -0.5, -50.0, 0.0, 100.0)  # FEATURES["bat"] 순서
PIT_BETA = (1.5, -0.5, -50.0, 0.0, 40.0, 100.0)  # FEATURES["pit"] 순서

# 타자: player_id → {season: pa}. 아래 정답은 BAT_TRAIN_2023의 손 계산 참고
BAT_WORLD: dict[int, dict[int, int]] = {
    1: {2019: 100},  # 2019만 — 2021·2022 규칙 값 0이라 행이 없다
    2: {2019: 200, 2020: 40, 2021: 162, 2022: 289},
    3: {2020: 40, 2021: 262, 2022: 439},  # 2020 데뷔
    4: {2021: 100, 2022: 250},  # 2021 데뷔 — T=2021 학습 대상이 아니다
    5: {2019: 60, 2021: 20, 2022: 130},  # 2020 결장 뒤 복귀
    6: {2019: 362, 2020: 20, 2022: 23},  # 2021 결장(규칙 값 0) 뒤 복귀
}
# 투수: player_id → (gs/g 비율, {season: outs}). 시즌마다 g = 10, gs = 10 × 비율이라
# 기간 안 gs_share는 언제나 그 비율이다(2020 G·GS 환산 여부와 무관)
PIT_WORLD: dict[int, tuple[float, dict[int, int]]] = {
    1: (0.0, {2019: 100}),
    2: (1.0, {2019: 200, 2020: 40, 2021: 202, 2022: 389}),
    3: (0.5, {2020: 40, 2021: 282, 2022: 489}),
    4: (0.0, {2021: 100, 2022: 250}),
    5: (0.5, {2019: 60, 2021: 40, 2022: 180}),
    6: (1.0, {2019: 442, 2020: 20, 2022: 63}),
}

# 목표 Y=2023의 학습 행: (train_target, player_id) → (*FEATURES["bat"], label)
# 규칙 L = 1.5·pt1 − 0.5·pt2 − 50·gap + 100 (n_seasons 계수 0)
BAT_TRAIN_2023: dict[tuple[int, int], tuple[float, ...]] = {
    # T=2021 — 기간 2016~2020 안에 행이 있는 1·2·3·5·6
    (2021, 1): (0.0, 100.0, 1.0, 1.0, 1.0, 0.0),  # 0 − 50 − 50 + 100 = 0 → 2021 행 없음
    (2021, 2): (108.0, 200.0, 0.0, 2.0, 1.0, 162.0),  # pt1 40×2.7=108: 162 − 100 + 100 = 162
    (2021, 3): (108.0, 0.0, 0.0, 1.0, 1.0, 262.0),  # 162 + 100 = 262
    (2021, 5): (0.0, 60.0, 1.0, 1.0, 1.0, 20.0),  # −30 − 50 + 100 = 20
    (2021, 6): (54.0, 362.0, 0.0, 2.0, 1.0, 0.0),  # pt1 20×2.7=54: 81 − 181 + 100 = 0
    # T=2022 — 기간 2017~2021, 여섯 명 모두
    (2022, 1): (0.0, 0.0, 2.0, 1.0, 1.0, 0.0),  # gap 2021−2019=2: −100 + 100 = 0
    (2022, 2): (162.0, 108.0, 0.0, 3.0, 1.0, 289.0),  # 243 − 54 + 100 = 289
    (2022, 3): (262.0, 108.0, 0.0, 2.0, 1.0, 439.0),  # 393 − 54 + 100 = 439
    (2022, 4): (100.0, 0.0, 0.0, 1.0, 1.0, 250.0),  # 150 + 100 = 250
    (2022, 5): (20.0, 0.0, 0.0, 2.0, 1.0, 130.0),  # 2020 없음 → pt2 0: 30 + 100 = 130
    (2022, 6): (0.0, 54.0, 1.0, 2.0, 1.0, 23.0),  # gap 2021−2020=1: −27 − 50 + 100 = 23
}
# 규칙 L = 1.5·pt1 − 0.5·pt2 − 50·gap + 40·gs_share + 100
PIT_TRAIN_2023: dict[tuple[int, int], tuple[float, ...]] = {
    (2021, 1): (0.0, 100.0, 1.0, 1.0, 0.0, 1.0, 0.0),  # −50 − 50 + 0 + 100 = 0
    (2021, 2): (108.0, 200.0, 0.0, 2.0, 1.0, 1.0, 202.0),  # 162 − 100 + 40 + 100 = 202
    (2021, 3): (108.0, 0.0, 0.0, 1.0, 0.5, 1.0, 282.0),  # 162 + 20 + 100 = 282
    (2021, 5): (0.0, 60.0, 1.0, 1.0, 0.5, 1.0, 40.0),  # −30 − 50 + 20 + 100 = 40
    (2021, 6): (54.0, 442.0, 0.0, 2.0, 1.0, 1.0, 0.0),  # 81 − 221 + 40 + 100 = 0
    (2022, 1): (0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0),  # −100 + 100 = 0
    (2022, 2): (202.0, 108.0, 0.0, 3.0, 1.0, 1.0, 389.0),  # 303 − 54 + 40 + 100 = 389
    (2022, 3): (282.0, 108.0, 0.0, 2.0, 0.5, 1.0, 489.0),  # 423 − 54 + 20 + 100 = 489
    (2022, 4): (100.0, 0.0, 0.0, 1.0, 0.0, 1.0, 250.0),  # 150 + 100 = 250
    (2022, 5): (40.0, 0.0, 0.0, 2.0, 0.5, 1.0, 180.0),  # 60 + 20 + 100 = 180
    (2022, 6): (0.0, 54.0, 1.0, 2.0, 1.0, 1.0, 63.0),  # −27 − 50 + 40 + 100 = 63
}
TRAIN_2023: dict[Role, dict[tuple[int, int], tuple[float, ...]]] = {
    "bat": BAT_TRAIN_2023,
    "pit": PIT_TRAIN_2023,
}
BETA: dict[Role, tuple[float, ...]] = {"bat": BAT_BETA, "pit": PIT_BETA}

# 목표 Y=2023 예측 (기간 2018~2022): 같은 규칙에 2023 특성을 넣고 0 미만은 0
PRED_2023: dict[Role, dict[int, float]] = {
    "bat": {
        1: 0.0,  # (0, 0, gap 2022−2019=3, 1): −150 + 100 = −50 → 0으로 자른다
        2: 452.5,  # (289, 162, 0, 4): 433.5 − 81 + 100
        3: 627.5,  # (439, 262, 0, 3): 658.5 − 131 + 100
        4: 425.0,  # (250, 100, 0, 2): 375 − 50 + 100
        5: 285.0,  # (130, 20, 0, 3): 195 − 10 + 100
        6: 134.5,  # (23, 0, 0, 3): 34.5 + 100 (2021 결장 → pt2 0)
    },
    "pit": {
        1: 0.0,  # −150 + 0 + 100 = −50 → 0
        2: 622.5,  # (389, 202, 0, 4, 1): 583.5 − 101 + 40 + 100
        3: 712.5,  # (489, 282, 0, 3, 0.5): 733.5 − 141 + 20 + 100
        4: 425.0,  # (250, 100, 0, 2, 0): 375 − 50 + 100
        5: 370.0,  # (180, 40, 0, 3, 0.5): 270 − 20 + 20 + 100
        6: 234.5,  # (63, 0, 0, 3, 1): 94.5 + 40 + 100
    },
}
# 목표 Y=2022 타자: 학습은 T=2021 다섯 행뿐(특성 5개, 계수가 정확히 정해진다).
# 2022 특성은 BAT_TRAIN_2023의 T=2022 행과 같으므로 예측 = 그 행의 정답
BAT_PRED_2022: dict[int, float] = {1: 0.0, 2: 289.0, 3: 439.0, 4: 250.0, 5: 130.0, 6: 23.0}


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
    """계약의 모든 열을 채운 행. row에 있는 값이 먼저고, 환산 출전량은 2020만 ×162/60이다."""
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
            out[name] = round(0.1 + 0.01 * i + 0.001 * (season - c.SEASON_FIRST), 6)
        else:
            raise AssertionError(f"채울 수 없는 열: {name}")
    for adj, base in _ADJUSTED.items():
        if adj in spec.column_names and adj not in out:
            scale = c.SHORT_SEASON_SCALE if season == c.SHORT_SEASON else 1.0
            out[adj] = float(out[base]) * scale
    return out


def _frame(role: Role, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    spec = SPECS[role]
    filled = [_fill(spec, row) for row in rows]
    df = pd.DataFrame(
        {
            col.name: pd.Series([r[col.name] for r in filled], dtype=_dtype(col))
            for col in spec.columns
        }
    )
    assert c.check_frame(df, spec) == []
    return df


def _bat_rows(data: Mapping[int, Mapping[int, int]]) -> list[dict[str, Any]]:
    return [
        {"player_id": pid, "season": season, "pa": pa}
        for pid, by_season in data.items()
        for season, pa in by_season.items()
    ]


def _pit_rows(data: Mapping[int, Mapping[int, tuple[int, int, int]]]) -> list[dict[str, Any]]:
    return [
        {"player_id": pid, "season": season, "g": g, "gs": gs, "outs": outs, "bf": 4 * outs}
        for pid, by_season in data.items()
        for season, (g, gs, outs) in by_season.items()
    ]


def _world_rows(role: Role) -> list[dict[str, Any]]:
    if role == "bat":
        return _bat_rows(BAT_WORLD)
    # 시즌마다 g = 10, gs = 10 × 비율
    pit: dict[int, dict[int, tuple[int, int, int]]] = {
        pid: {season: (10, round(10 * share), outs) for season, outs in by_season.items()}
        for pid, (share, by_season) in PIT_WORLD.items()
    }
    return _pit_rows(pit)


def _feat_rows(role: Role) -> list[dict[str, Any]]:
    return _bat_rows(FEAT_BAT) if role == "bat" else _pit_rows(FEAT_PIT)


def _world(role: Role) -> pd.DataFrame:
    return _frame(role, _world_rows(role))


def _poison_values(role: Role) -> dict[str, int]:
    if role == "bat":
        return {"pa": POISON}
    return {"g": POISON, "gs": POISON, "outs": POISON, "bf": POISON}


def _clean_and_poisoned(
    role: Role, rows: Sequence[Mapping[str, Any]], target: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(season < target 행만, 같은 행 + season ≥ target "독" 행).

    독 행: 모든 선수의 target~target+2 시즌에 출전량을 POISON으로(target 시즌 정답이 아주 커진다),
    그리고 target 시즌에만 기록이 있는 새 선수. 독 행을 앞에 둬서 입력 순서도 섞는다.
    """
    clean_rows = [dict(r) for r in rows if int(r["season"]) < target]
    pids = sorted({int(r["player_id"]) for r in rows})
    huge = _poison_values(role)
    poison_rows: list[dict[str, Any]] = [
        {"player_id": pid, "season": season, **huge}
        for pid in pids
        for season in range(target, target + 3)
    ]
    poison_rows.append({"player_id": NEW_PLAYER, "season": target, **huge})
    return _frame(role, clean_rows), _frame(role, poison_rows + clean_rows)


def _sorted(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    return df.sort_values(list(keys), kind="mergesort").reset_index(drop=True)


def _feature_records(df: pd.DataFrame, role: Role) -> dict[int, tuple[float, ...]]:
    return {
        int(rec["player_id"]): tuple(float(rec[name]) for name in FEATURES[role])
        for rec in df.to_dict("records")
    }


def _train_records(df: pd.DataFrame, role: Role) -> dict[tuple[int, int], tuple[float, ...]]:
    names = [*FEATURES[role], "label"]
    return {
        (int(rec["train_target"]), int(rec["player_id"])): tuple(float(rec[n]) for n in names)
        for rec in df.to_dict("records")
    }


def _assert_records(got: Mapping[Any, tuple[float, ...]], expected: Mapping[Any, Any]) -> None:
    assert sorted(got) == sorted(expected)
    for key, values in expected.items():
        assert got[key] == pytest.approx(values, abs=TOL), key


def _train_frame_by_hand(role: Role, feats: Sequence[Sequence[float]]) -> pd.DataFrame:
    """규칙 BETA_FIT[role]로 정답을 만든 학습 표 (training_frame 결과와 같은 열 순서)."""
    beta = np.array(BETA_FIT[role])
    data: dict[str, Any] = {
        "player_id": pd.Series(range(1, len(feats) + 1), dtype="int64"),
        "train_target": pd.Series([2024] * len(feats), dtype="int64"),
    }
    for j, name in enumerate(FEATURES[role]):
        data[name] = pd.Series([float(f[j]) for f in feats], dtype="float64")
    data["label"] = pd.Series([float(np.dot(f, beta)) for f in feats], dtype="float64")
    return pd.DataFrame(data)


CohortCall = tuple[int, Mapping[int, int] | None, pd.DataFrame]


def _spy_build_cohort(monkeypatch: pytest.MonkeyPatch) -> list[CohortCall]:
    """`build_cohort` 호출을 (목표 시즌, debut_seasons, 결과)로 기록한다. 실제 함수에 넘긴다."""
    calls: list[CohortCall] = []
    real: Callable[..., pd.DataFrame] = cohort_mod.build_cohort
    signature = inspect.signature(real)

    def spy(*args: Any, **kwargs: Any) -> pd.DataFrame:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        result = real(*args, **kwargs)
        calls.append(
            (int(bound.arguments["target_season"]), bound.arguments["debut_seasons"], result)
        )
        return result

    # `from mlbproj.cohort import build_cohort`와 `cohort.build_cohort` 두 방식 모두 잡는다
    monkeypatch.setattr(cohort_mod, "build_cohort", spy)
    monkeypatch.setattr(pt_mod, "build_cohort", spy, raising=False)
    return calls


# ---------------------------------------------------------------------------
# FEATURES 상수
# ---------------------------------------------------------------------------


def test_models_playing_time_features_constant() -> None:
    assert FEATURES["bat"] == ("pt1", "pt2", "gap", "n_seasons", "const")
    assert FEATURES["pit"] == ("pt1", "pt2", "gap", "n_seasons", "gs_share", "const")


# ---------------------------------------------------------------------------
# playing_time_features
# ---------------------------------------------------------------------------

FEAT_CASES: list[tuple[Role, int]] = [
    ("bat", 2024),
    ("bat", 2021),
    ("bat", 2022),
    ("pit", 2025),
    ("pit", 2021),
    ("pit", 2022),
]
FEAT_IDS = [f"{role}-{target}" for role, target in FEAT_CASES]


def _feat_expected(role: Role, target: int) -> dict[int, tuple[float, ...]]:
    return (FEAT_BAT_EXPECTED if role == "bat" else FEAT_PIT_EXPECTED)[target]


@pytest.mark.parametrize(("role", "target"), FEAT_CASES, ids=FEAT_IDS)
def test_models_playing_time_features_hand_values(role: Role, target: int) -> None:
    seasons = _frame(role, _feat_rows(role))
    cohort = build_cohort(seasons, role, target)
    out = playing_time_features(cohort, seasons, role)
    _assert_records(_feature_records(out, role), _feat_expected(role, target))


@pytest.mark.parametrize(("role", "target"), FEAT_CASES, ids=FEAT_IDS)
def test_models_playing_time_features_columns_dtypes_no_nan(role: Role, target: int) -> None:
    seasons = _frame(role, _feat_rows(role))
    cohort = build_cohort(seasons, role, target)
    out = playing_time_features(cohort, seasons, role)
    assert list(out.columns) == ["player_id", *FEATURES[role]]
    assert ptypes.is_integer_dtype(out["player_id"].dtype)
    for name in FEATURES[role]:
        assert ptypes.is_float_dtype(out[name].dtype), name
    assert not bool(out.isna().any().any())
    assert (out["const"] == 1.0).all()
    # 대상 표의 선수 전원, 한 번씩
    assert sorted(out["player_id"].tolist()) == sorted(cohort["player_id"].tolist())


@pytest.mark.parametrize(("role", "target"), FEAT_CASES, ids=FEAT_IDS)
def test_models_playing_time_features_gap_and_n_seasons_follow_cohort(
    role: Role, target: int
) -> None:
    seasons = _frame(role, _feat_rows(role))
    cohort = build_cohort(seasons, role, target)
    got = _feature_records(playing_time_features(cohort, seasons, role), role)
    gap_at = FEATURES[role].index("gap")
    n_at = FEATURES[role].index("n_seasons")
    for rec in cohort.to_dict("records"):
        pid = int(rec["player_id"])
        # gap = (T−1) − input_last, n_seasons = 대상 표 값
        assert got[pid][gap_at] == float(target - 1 - int(rec["input_last"]))
        assert got[pid][n_at] == float(rec["n_seasons"])


def test_models_playing_time_features_2020_uses_adjusted_value() -> None:
    # D9: 2020 행은 환산 출전량이 특성이 된다 — 타자 pa 100 → 270, 투수 outs 50 → 135
    bat = _frame("bat", _bat_rows(FEAT_BAT))
    bat_2021 = playing_time_features(build_cohort(bat, "bat", 2021), bat, "bat")
    bat_2022 = playing_time_features(build_cohort(bat, "bat", 2022), bat, "bat")
    assert _feature_records(bat_2021, "bat")[302][0] == pytest.approx(100 * 162 / 60)  # pt1
    assert _feature_records(bat_2022, "bat")[302][1] == pytest.approx(100 * 162 / 60)  # pt2
    pit = _frame("pit", _pit_rows(FEAT_PIT))
    pit_2021 = playing_time_features(build_cohort(pit, "pit", 2021), pit, "pit")
    pit_2022 = playing_time_features(build_cohort(pit, "pit", 2022), pit, "pit")
    assert _feature_records(pit_2021, "pit")[402][0] == pytest.approx(50 * 162 / 60)  # pt1
    assert _feature_records(pit_2022, "pit")[402][1] == pytest.approx(50 * 162 / 60)  # pt2


def test_models_playing_time_features_gs_share_window_and_zero_games() -> None:
    pit = _frame("pit", _pit_rows(FEAT_PIT))
    out = playing_time_features(build_cohort(pit, "pit", 2025), pit, "pit").set_index("player_id")
    # 401: 기간(2020~2024) 안 2021·2024만 — (5 + 0) / (20 + 30) = 0.1. 2019(10/10)를 넣으면 0.25
    assert out.loc[401, "gs_share"] == pytest.approx(0.1)
    # 403: 기간 안 Σg = 0 → 0 (결측이 아니다)
    assert out.loc[403, "gs_share"] == 0.0


def test_models_playing_time_features_gs_share_uses_raw_2020_counts() -> None:
    # docstring 공식 Σgs / Σg는 정제 표의 원래 G·GS다 (계약에 g_adj·gs_adj가 없다).
    # 501: 2020 (g 10, gs 10), 2021 (g 10, gs 0) → 목표 2022의 gs_share = (10 + 0) / (10 + 10) = 0.5
    # 2020 G·GS도 ×162/60으로 환산하면 27 / (27 + 10) ≈ 0.730이 된다 — 그 방식이 아님을 고정한다
    pit = _frame("pit", _pit_rows({501: {2020: (10, 10, 30), 2021: (10, 0, 30)}}))
    out = playing_time_features(build_cohort(pit, "pit", 2022), pit, "pit").set_index("player_id")
    assert out.loc[501, "gs_share"] == pytest.approx(0.5, abs=TOL)
    # pt1 = 2021 outs_adj 30, pt2 = 2020 outs 30 × 2.7 = 81 (D9 환산은 출전량에만)
    assert out.loc[501, "pt1"] == pytest.approx(30.0)
    assert out.loc[501, "pt2"] == pytest.approx(81.0)


def test_models_playing_time_features_reject_empty_cohort() -> None:
    bat = _frame("bat", _bat_rows(FEAT_BAT))
    empty = build_cohort(bat, "bat", 2024).iloc[0:0]
    assert c.check_frame(empty, c.COHORTS) == []
    with pytest.raises(ValueError, match="비어"):
        playing_time_features(empty, bat, "bat")
    with pytest.raises(ValueError, match="비어"):
        predict_playing_time(empty, bat, "bat")


def test_models_playing_time_features_reject_bad_seasons() -> None:
    bat = _frame("bat", _bat_rows(FEAT_BAT))
    cohort = build_cohort(bat, "bat", 2024)
    with pytest.raises(ValueError):
        playing_time_features(cohort, bat.drop(columns=["pa_adj"]), "bat")
    # 역할과 다른 정제 표 (투수 표를 타자로)
    pit = _frame("pit", _pit_rows(FEAT_PIT))
    with pytest.raises(ValueError):
        playing_time_features(cohort, pit, "bat")


def test_models_playing_time_features_reject_bad_cohort() -> None:
    bat = _frame("bat", _bat_rows(FEAT_BAT))
    pit = _frame("pit", _pit_rows(FEAT_PIT))
    cohort = build_cohort(bat, "bat", 2024)
    with pytest.raises(ValueError):  # 계약 위반
        playing_time_features(cohort.drop(columns=["n_seasons"]), bat, "bat")
    with pytest.raises(ValueError):  # 대상 표 역할이 role과 다르다
        playing_time_features(build_cohort(pit, "pit", 2025), bat, "bat")
    two_targets = pd.concat([cohort, build_cohort(bat, "bat", 2025)], ignore_index=True)
    assert c.check_frame(two_targets, c.COHORTS) == []
    with pytest.raises(ValueError):  # 목표 시즌이 둘
        playing_time_features(two_targets, bat, "bat")


# ---------------------------------------------------------------------------
# training_frame
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_training_frame_hand_rows(role: Role) -> None:
    train = training_frame(_world(role), role, 2023)
    assert len(train) == len(TRAIN_2023[role])
    _assert_records(_train_records(train, role), TRAIN_2023[role])


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_training_frame_columns_and_dtypes(role: Role) -> None:
    train = training_frame(_world(role), role, 2023)
    assert list(train.columns) == [*TRAIN_LEAD, *FEATURES[role], "label"]
    assert ptypes.is_integer_dtype(train["player_id"].dtype)
    assert ptypes.is_integer_dtype(train["train_target"].dtype)
    for name in [*FEATURES[role], "label"]:
        assert ptypes.is_float_dtype(train[name].dtype), name
    assert not bool(train.isna().any().any())


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_training_frame_single_target(role: Role) -> None:
    # Y=2022 → 학습 목표는 2021 하나: TRAIN_2023의 T=2021 행만
    train = training_frame(_world(role), role, 2022)
    expected = {k: v for k, v in TRAIN_2023[role].items() if k[0] == 2021}
    assert set(train["train_target"]) == {2021}
    _assert_records(_train_records(train, role), expected)


def test_models_playing_time_training_frame_targets_run_to_y_minus_1() -> None:
    # Y=2025 → T = 2021, 2022, 2023, 2024. 세계에 2023·2024 행이 없으니 그 정답은 모두 0
    train = training_frame(_world("bat"), "bat", 2025)
    counts = train.groupby("train_target")["player_id"].nunique().to_dict()
    # T=2021: 1·2·3·5·6 / T=2022~2024: 여섯 명 (1은 2019 행이 기간 2019~2023 안에 남는다)
    assert counts == {2021: 5, 2022: 6, 2023: 6, 2024: 6}
    assert (train.loc[train["train_target"] >= 2023, "label"] == 0.0).all()
    got = _train_records(train, "bat")
    # T=2024, 1: 기간 2019~2023, input_last 2019 → gap = 2023 − 2019 = 4, n = 1
    assert got[(2024, 1)] == pytest.approx((0.0, 0.0, 4.0, 1.0, 1.0, 0.0))
    # T=2024, 2: pt1 = 2023 없음 0, pt2 = 2022 pa 289, gap = 2023 − 2022 = 1, n = 4 (2019~2022)
    assert got[(2024, 2)] == pytest.approx((0.0, 289.0, 1.0, 4.0, 1.0, 0.0))
    # T=2023, 1: 기간 2019~2022, gap = 2022 − 2019 = 3
    assert got[(2023, 1)] == pytest.approx((0.0, 0.0, 3.0, 1.0, 1.0, 0.0))


@pytest.mark.parametrize("role", ["bat", "pit"])
@pytest.mark.parametrize("target", [c.SEASON_FIRST, 2020, FIRST_TRAIN_TARGET])
def test_models_playing_time_training_frame_empty_without_targets(role: Role, target: int) -> None:
    # Y ≤ 2021 → 학습 목표 2021 ~ Y−1이 없다 (T−2가 2019 이전이 된다)
    train = training_frame(_world(role), role, target)
    assert len(train) == 0
    assert list(train.columns) == [*TRAIN_LEAD, *FEATURES[role], "label"]


def test_models_playing_time_training_frame_first_target_follows_data_start() -> None:
    # 데이터가 2020부터면 T=2021의 T−2(2019)가 데이터 밖이다 → 학습 목표는 2020 + 2 = 2022부터
    rows = [r for r in _world_rows("bat") if int(r["season"]) >= 2020]
    train = training_frame(_frame("bat", rows), "bat", 2024)
    assert sorted(set(train["train_target"])) == [2022, 2023]


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_training_frame_rejects_bad_seasons(role: Role) -> None:
    world = _world(role)
    with pytest.raises(ValueError):  # 계약 열이 빠졌다
        training_frame(world.drop(columns=[PT_COLUMN[role]]), role, 2023)
    other: Role = "pit" if role == "bat" else "bat"
    with pytest.raises(ValueError):  # 역할과 다른 정제 표
        training_frame(_world(other), role, 2023)


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_row_order_does_not_matter(role: Role) -> None:
    # 정제 표의 행 순서를 섞어도 학습 표(정렬 없이)·계수·예측이 비트 단위로 같다 (N1 재현성)
    world = _world(role)
    shuffled = world.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert not world.equals(shuffled)
    assert_frame_equal(training_frame(shuffled, role, 2023), training_frame(world, role, 2023))
    cohort = build_cohort(world, role, 2023)
    assert_frame_equal(
        predict_playing_time(cohort, shuffled, role), predict_playing_time(cohort, world, role)
    )


def test_models_playing_time_training_frame_uses_build_cohort_per_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_build_cohort(monkeypatch)
    training_frame(_world("bat"), "bat", 2025)
    assert sorted({target for target, _, _ in calls}) == [2021, 2022, 2023, 2024]


def test_models_playing_time_training_frame_passes_debut_seasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world("bat")
    without = training_frame(world, "bat", 2023)
    debut = {2: 2015, 3: 2017}
    calls = _spy_build_cohort(monkeypatch)
    with_debut = training_frame(world, "bat", 2023, debut)
    assert sorted({target for target, _, _ in calls}) == [2021, 2022]
    for _, passed, _ in calls:
        assert passed is not None and dict(passed) == debut
    windows = {
        (target, int(pid)): int(first)
        for target, _, result in calls
        for pid, first in zip(result["player_id"], result["window_first"], strict=True)
    }
    # 데뷔 연도가 학습 대상의 기간을 바꾼다 (E21)
    # T=2021: 2 → max(2016, min(2019, 2015)) = 2016 (없으면 2019), 3 → max(2016, 2017) = 2017 (없으면 2020)
    assert windows[(2021, 2)] == 2016
    assert windows[(2021, 3)] == 2017
    # T=2022: 2 → max(2017, 2015) = 2017, 3 → max(2017, 2017) = 2017
    assert windows[(2022, 2)] == 2017
    assert windows[(2022, 3)] == 2017
    # 첫 행 이전에는 행이 없으니 특성(pt1·pt2·gap·n_seasons)은 그대로다
    keys = ["train_target", "player_id"]
    assert_frame_equal(_sorted(with_debut, keys), _sorted(without, keys))


# ---------------------------------------------------------------------------
# fit_coefficients
# ---------------------------------------------------------------------------

# 손으로 정한 규칙 (FEATURES 순서)
BETA_FIT: dict[Role, tuple[float, ...]] = {
    "bat": (0.8, 0.1, -30.0, 15.0, 60.0),
    "pit": (0.9, 0.05, -20.0, 10.0, 120.0, 30.0),
}
FIT_FEATS: dict[Role, list[tuple[float, ...]]] = {
    # (pt1, pt2, gap, n_seasons, const) → 정답 = 0.8·pt1 + 0.1·pt2 − 30·gap + 15·n + 60
    "bat": [
        (500, 400, 0, 5, 1),  # 400 + 40 + 75 + 60 = 575
        (300, 0, 0, 1, 1),  # 240 + 15 + 60 = 315
        (0, 200, 1, 2, 1),  # 20 − 30 + 30 + 60 = 80
        (0, 0, 2, 1, 1),  # −60 + 15 + 60 = 15
        (100, 600, 0, 4, 1),  # 80 + 60 + 60 + 60 = 260
        (250, 250, 0, 3, 1),  # 200 + 25 + 45 + 60 = 330
    ],
    # (pt1, pt2, gap, n, gs_share, const) → 0.9·pt1 + 0.05·pt2 − 20·gap + 10·n + 120·gs + 30
    "pit": [
        (500, 400, 0, 5, 1.0, 1),  # 450 + 20 + 50 + 120 + 30 = 670
        (300, 0, 0, 1, 0.0, 1),  # 270 + 10 + 30 = 310
        (0, 200, 1, 2, 0.5, 1),  # 10 − 20 + 20 + 60 + 30 = 100
        (0, 0, 2, 1, 0.0, 1),  # −40 + 10 + 30 = 0
        (100, 600, 0, 4, 0.25, 1),  # 90 + 30 + 40 + 30 + 30 = 220
        (250, 250, 0, 3, 0.8, 1),  # 225 + 12.5 + 30 + 96 + 30 = 393.5
        (60, 30, 0, 2, 0.0, 1),  # 54 + 1.5 + 20 + 30 = 105.5
    ],
}
FIT_LABELS: dict[Role, list[float]] = {
    "bat": [575.0, 315.0, 80.0, 15.0, 260.0, 330.0],
    "pit": [670.0, 310.0, 100.0, 0.0, 220.0, 393.5, 105.5],
}


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_fit_recovers_linear_rule(role: Role) -> None:
    train = _train_frame_by_hand(role, FIT_FEATS[role])
    # 손 계산한 정답과 같은지 먼저 확인 (도우미 자체 검사)
    assert train["label"].tolist() == pytest.approx(FIT_LABELS[role])
    coef = fit_coefficients(train, role)
    assert isinstance(coef, np.ndarray)
    assert coef.shape == (len(FEATURES[role]),)
    np.testing.assert_allclose(coef, BETA_FIT[role], rtol=0, atol=TOL)


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_fit_is_least_squares(role: Role) -> None:
    # 같은 특성의 두 행에 정답 ±7 → 최소제곱 해는 평균 정답(= 규칙 값)으로 맞춘 해와 같다
    base = _train_frame_by_hand(role, FIT_FEATS[role])
    up, down = base.copy(), base.copy()
    up["label"] = up["label"] + 7.0
    down["label"] = down["label"] - 7.0
    noisy = pd.concat([up, down], ignore_index=True)
    noisy["player_id"] = range(1, len(noisy) + 1)
    coef = fit_coefficients(noisy, role)
    np.testing.assert_allclose(coef, BETA_FIT[role], rtol=0, atol=TOL)


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_fit_exactly_enough_rows(role: Role) -> None:
    # 행 수 = 특성 수 (타자 5, 투수 6) — 풀 수 있다
    n = len(FEATURES[role])
    coef = fit_coefficients(_train_frame_by_hand(role, FIT_FEATS[role][:n]), role)
    np.testing.assert_allclose(coef, BETA_FIT[role], rtol=0, atol=TOL)


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_fit_too_few_rows(role: Role) -> None:
    n = len(FEATURES[role])
    with pytest.raises(ValueError):
        fit_coefficients(_train_frame_by_hand(role, FIT_FEATS[role][: n - 1]), role)
    with pytest.raises(ValueError):
        fit_coefficients(_train_frame_by_hand(role, []), role)


@pytest.mark.parametrize("role", ["bat", "pit"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_models_playing_time_fit_rejects_non_finite(role: Role, bad: float) -> None:
    train = _train_frame_by_hand(role, FIT_FEATS[role])
    with_label = train.copy()
    with_label.loc[0, "label"] = bad
    with pytest.raises(ValueError, match="결측"):
        fit_coefficients(with_label, role)
    with_feature = train.copy()
    with_feature.loc[1, "pt1"] = bad
    with pytest.raises(ValueError, match="결측"):
        fit_coefficients(with_feature, role)


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_fit_world_rule(role: Role) -> None:
    # 선형 세계의 학습 표(Y=2023, 11행)는 규칙을 정확히 따른다 → 계수를 그대로 되찾는다
    coef = fit_coefficients(training_frame(_world(role), role, 2023), role)
    np.testing.assert_allclose(coef, BETA[role], rtol=0, atol=TOL)


# ---------------------------------------------------------------------------
# predict_playing_time
# ---------------------------------------------------------------------------


def _predictions(out: pd.DataFrame, role: Role) -> dict[int, float]:
    return {
        int(pid): float(v) for pid, v in zip(out["player_id"], out[PT_COLUMN[role]], strict=True)
    }


def _segment_expected(world: pd.DataFrame, role: Role, target: int) -> dict[int, float]:
    """구간 규칙(모듈 docstring)으로 계산한 기대 예측. fit_segments 자체는 아래 구간 테스트가 손 계산으로 검사한다."""
    cohort = build_cohort(world, role, target)
    model = fit_segments(training_frame(world, role, target), role)
    feats = playing_time_features(cohort, world, role).set_index("player_id")
    linear = feats[list(FEATURES[role])].to_numpy() @ model.played
    values = np.maximum(np.where(feats["gap"].to_numpy() > 0, model.absent, linear), 0.0)
    return {int(pid): float(v) for pid, v in zip(feats.index, values, strict=True)}


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_predict_hand_values(role: Role) -> None:
    world = _world(role)
    cohort = build_cohort(world, role, 2023)
    out = predict_playing_time(cohort, world, role)
    assert c.check_frame(out, PT_SPECS[role]) == []
    got = _predictions(out, role)
    expected = _segment_expected(world, role, 2023)
    assert sorted(got) == sorted(expected)
    for pid, value in expected.items():
        assert got[pid] == pytest.approx(value, abs=TOL), pid


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_predict_contract_sorted_nonnegative(role: Role) -> None:
    world = _world(role)
    cohort = build_cohort(world, role, 2023)
    # 대상 표 순서를 뒤집어 넣어도 결과는 player_id 순이다
    shuffled = cohort.iloc[::-1].reset_index(drop=True)
    out = predict_playing_time(shuffled, world, role)
    assert c.check_frame(out, PT_SPECS[role]) == []
    assert list(out.columns) == PT_SPECS[role].column_names
    ids = out["player_id"].tolist()
    assert ids == sorted(ids)
    assert set(ids) == set(cohort["player_id"])
    assert len(ids) == len(cohort)
    assert set(out["target_season"]) == {2023}
    assert bool((out[PT_COLUMN[role]] >= 0).all())


@pytest.mark.parametrize(
    ("role", "target"), [("bat", 2021), ("pit", 2021), ("pit", 2022), ("bat", 2022)]
)
def test_models_playing_time_predict_too_few_training_rows(role: Role, target: int) -> None:
    # Y=2021: 학습 행 0 / Y=2022: T=2021 행 중 직전 시즌 출전 구간 행이 특성 수보다 적다 (구간 모델)
    world = _world(role)
    with pytest.raises(ValueError):
        predict_playing_time(build_cohort(world, role, target), world, role)


def test_models_playing_time_predict_rejects_bad_input() -> None:
    world = _world("bat")
    cohort = build_cohort(world, "bat", 2023)
    with pytest.raises(ValueError):
        predict_playing_time(cohort, world.drop(columns=["pa_adj"]), "bat")
    with pytest.raises(ValueError):
        predict_playing_time(cohort.drop(columns=["sample"]), world, "bat")
    with pytest.raises(ValueError):  # 역할이 다른 정제 표
        predict_playing_time(cohort, _world("pit"), "bat")


def test_models_playing_time_predict_passes_debut_seasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _world("bat")
    cohort = build_cohort(world, "bat", 2023)
    debut = {2: 2015, 3: 2017}
    calls = _spy_build_cohort(monkeypatch)
    out = predict_playing_time(cohort, world, "bat", debut)
    assert {2021, 2022} <= {target for target, _, _ in calls}
    for _, passed, _ in calls:
        assert passed is not None and dict(passed) == debut
    # 데뷔 연도를 넘겨도 예측은 출력 계약을 지킨다 (값은 학습 대상 기간에 따라 달라질 수 있다)
    got = _predictions(out, "bat")
    assert sorted(got) == sorted(int(i) for i in cohort["player_id"])
    assert all(v >= 0 for v in got.values())


# ---------------------------------------------------------------------------
# N2 — 목표 시즌 Y 이후 행은 결과를 바꾸지 않는다
# ---------------------------------------------------------------------------

N2_WORLD_CASES: list[tuple[Role, int]] = [
    ("bat", 2022),
    ("bat", 2023),
    ("pit", 2022),
    ("pit", 2023),
]
N2_WORLD_IDS = [f"{role}-{target}" for role, target in N2_WORLD_CASES]


@pytest.mark.parametrize(("role", "target"), FEAT_CASES, ids=FEAT_IDS)
def test_models_playing_time_n2_features_hand_fixture(role: Role, target: int) -> None:
    clean, poisoned = _clean_and_poisoned(role, _feat_rows(role), target)
    cohort = build_cohort(clean, role, target)
    expected = playing_time_features(cohort, clean, role)
    actual = playing_time_features(cohort, poisoned, role)
    assert_frame_equal(_sorted(actual, ["player_id"]), _sorted(expected, ["player_id"]))
    _assert_records(_feature_records(actual, role), _feat_expected(role, target))


@pytest.mark.parametrize(("role", "target"), N2_WORLD_CASES, ids=N2_WORLD_IDS)
def test_models_playing_time_n2_features_world(role: Role, target: int) -> None:
    clean, poisoned = _clean_and_poisoned(role, _world_rows(role), target)
    cohort = build_cohort(clean, role, target)
    expected = playing_time_features(cohort, clean, role)
    actual = playing_time_features(cohort, poisoned, role)
    assert_frame_equal(_sorted(actual, ["player_id"]), _sorted(expected, ["player_id"]))


@pytest.mark.parametrize(("role", "target"), N2_WORLD_CASES, ids=N2_WORLD_IDS)
def test_models_playing_time_n2_training_frame(role: Role, target: int) -> None:
    clean, poisoned = _clean_and_poisoned(role, _world_rows(role), target)
    keys = ["train_target", "player_id"]
    expected = training_frame(clean, role, target)
    actual = training_frame(poisoned, role, target)
    assert_frame_equal(_sorted(actual, keys), _sorted(expected, keys))
    # 손 계산과도 같다 (목표 2023 학습 표에서 T < target 행)
    hand = {k: v for k, v in TRAIN_2023[role].items() if k[0] < target}
    _assert_records(_train_records(actual, role), hand)


@pytest.mark.parametrize(("role", "target"), N2_WORLD_CASES, ids=N2_WORLD_IDS)
def test_models_playing_time_n2_no_label_from_target_or_later(role: Role, target: int) -> None:
    _, poisoned = _clean_and_poisoned(role, _world_rows(role), target)
    train = training_frame(poisoned, role, target)
    assert set(train["train_target"]) <= set(range(FIRST_TRAIN_TARGET, target))
    assert bool((train["label"] < POISON).all())
    assert NEW_PLAYER not in set(train["player_id"])
    for name in FEATURES[role]:
        assert bool((train[name] < POISON).all()), name


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_n2_training_targets_below_y(
    role: Role, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, poisoned = _clean_and_poisoned(role, _world_rows(role), 2023)
    calls = _spy_build_cohort(monkeypatch)
    training_frame(poisoned, role, 2023)
    assert sorted({target for target, _, _ in calls}) == [2021, 2022]


@pytest.mark.parametrize(
    ("role", "target", "expected"),
    [
        ("bat", 2023, None),
        ("pit", 2023, None),
    ],
    ids=["bat-2023", "pit-2023"],
)
def test_models_playing_time_n2_predict(
    role: Role, target: int, expected: Mapping[int, float] | None
) -> None:
    clean, poisoned = _clean_and_poisoned(role, _world_rows(role), target)
    cohort = build_cohort(clean, role, target)
    from_clean = predict_playing_time(cohort, clean, role)
    from_poisoned = predict_playing_time(cohort, poisoned, role)
    assert_frame_equal(from_poisoned, from_clean, check_exact=True)
    got = _predictions(from_poisoned, role)
    for pid, value in (expected or _segment_expected(clean, role, target)).items():
        assert got[pid] == pytest.approx(value, abs=TOL), pid


@pytest.mark.parametrize(("role", "target"), N2_WORLD_CASES, ids=N2_WORLD_IDS)
def test_models_playing_time_n2_future_rows_in_two_leagues(role: Role, target: int) -> None:
    # 목표 시즌 이후에 같은 (선수, 시즌)이 MLB·KBO 두 행으로 있어도(계약 키는 league까지라 유효)
    # 특성·학습 표·예측이 같아야 한다. 펼치기 전에 자르지 않으면 일대일 병합 오류가 난다
    rows = _world_rows(role)
    clean = _frame(role, [r for r in rows if int(r["season"]) < target])
    huge = _poison_values(role)
    future = [
        {"player_id": pid, "season": season, "league": league, **huge}
        for pid in (2, 3)
        for season in (target, target + 1)
        for league in (c.LEAGUE, "KBO")
    ]
    dirty = _frame(role, [*future, *(r for r in rows if int(r["season"]) < target)])
    cohort = build_cohort(clean, role, target)
    assert_frame_equal(
        playing_time_features(cohort, dirty, role), playing_time_features(cohort, clean, role)
    )
    assert_frame_equal(training_frame(dirty, role, target), training_frame(clean, role, target))
    if target != 2022:  # 2022는 출전 구간 학습 행이 모자라다 (predict_too_few_training_rows)
        assert_frame_equal(
            predict_playing_time(cohort, dirty, role), predict_playing_time(cohort, clean, role)
        )


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_n2_future_debut_years_ignored(role: Role) -> None:
    # 데뷔 연도가 목표 시즌 이후로 적혀 있어도(늦게 적힌 값) 결과가 같다: 첫 행 시즌이 더 이르다
    world = _world(role)
    cohort = build_cohort(world, role, 2023)
    debut = {2: 2030, 3: 2023, NEW_PLAYER: 2023}
    assert_frame_equal(
        predict_playing_time(cohort, world, role, debut), predict_playing_time(cohort, world, role)
    )
    keys = ["train_target", "player_id"]
    assert_frame_equal(
        _sorted(training_frame(world, role, 2023, debut), keys),
        _sorted(training_frame(world, role, 2023), keys),
    )


def test_models_playing_time_n2_predict_future_rows_do_not_add_training_rows() -> None:
    # 투수 Y=2022: 학습 행은 T=2021 다섯 개뿐이라 모자란다. 독 행(2022 정답·새 선수)이
    # 학습에 섞이면 행이 늘어 풀려 버린다 — 둘 다 ValueError여야 한다
    clean, poisoned = _clean_and_poisoned("pit", _world_rows("pit"), 2022)
    cohort = build_cohort(clean, "pit", 2022)
    with pytest.raises(ValueError):
        predict_playing_time(cohort, clean, "pit")
    with pytest.raises(ValueError):
        predict_playing_time(cohort, poisoned, "pit")


# --- 구간 모델 (2026-11-24 M3 조정: 직전 시즌 미출전 선수는 학습 정답의 중앙값) ---


def _segment_train(
    role: Role, played: Sequence[tuple[float, float]], absent_labels: Sequence[float]
) -> pd.DataFrame:
    """특성 pt1만 움직이는 학습 표. played = (pt1, label), absent = gap 1인 행의 정답."""
    names = list(FEATURES[role])
    rows: list[dict[str, float]] = []
    for pt1, label in played:
        row = {n: 0.0 for n in names}
        row.update(pt1=pt1, n_seasons=3.0, const=1.0, label=label)
        rows.append(row)
    for label in absent_labels:
        row = {n: 0.0 for n in names}
        row.update(gap=1.0, n_seasons=2.0, const=1.0, label=label)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.insert(0, "train_target", 2023)
    df.insert(0, "player_id", range(1, len(df) + 1))
    return df


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_segments_fit_played_rows_only(role: Role) -> None:
    # 출전 행은 label = 0.5·pt1 + 30 (n_seasons·const 열이 상수라 계수는 최소노름 해) — 미출전 행(정답 0·0·300)이 섞이면 이 규칙이 깨진다
    played = [
        (100.0, 80.0),
        (200.0, 130.0),
        (300.0, 180.0),
        (400.0, 230.0),
        (500.0, 280.0),
        (600.0, 330.0),
        (700.0, 380.0),
    ]
    model = fit_segments(_segment_train(role, played, [0.0, 0.0, 300.0]), role)
    names = list(FEATURES[role])
    x = np.array([[{"pt1": 250.0, "n_seasons": 3.0, "const": 1.0}.get(n, 0.0) for n in names]])
    assert float((x @ model.played)[0]) == pytest.approx(0.5 * 250 + 30)
    assert model.absent == 0.0  # 중앙값 (0, 0, 300) = 0


def test_models_playing_time_segments_absent_value_is_median() -> None:
    played = [
        (100.0, 80.0),
        (200.0, 130.0),
        (300.0, 180.0),
        (400.0, 230.0),
        (500.0, 280.0),
        (600.0, 330.0),
    ]
    assert fit_segments(_segment_train("bat", played, [5.0, 9.0, 7.0]), "bat").absent == 7.0
    assert (
        fit_segments(_segment_train("bat", played, []), "bat").absent == 0.0
    )  # 미출전 학습 행 없음 → 0


def test_models_playing_time_segments_too_few_played_rows() -> None:
    with pytest.raises(ValueError):
        fit_segments(_segment_train("bat", [(100.0, 80.0)], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "bat")


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_playing_time_predict_uses_segments(role: Role) -> None:
    """예측 = 직전 시즌 출전(gap 0)은 출전 구간 계수, 미출전(gap > 0)은 미출전 구간 값 (0 이상)."""
    target = 2026
    world = _world(role)
    cohort = build_cohort(world, role, target)
    model = fit_segments(training_frame(world, role, target), role)
    feats = playing_time_features(cohort, world, role).set_index("player_id")
    raw = feats[list(FEATURES[role])].to_numpy() @ model.played
    expected = np.maximum(np.where(feats["gap"].to_numpy() > 0, model.absent, raw), 0.0)
    got = _predictions(predict_playing_time(cohort, world, role), role)
    assert [got[i] for i in feats.index] == pytest.approx(list(expected))
