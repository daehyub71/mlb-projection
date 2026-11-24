"""확장 모델(models/extended.py) 테스트 (SPEC F9·D10·D11·N2, PLAN §2-3).

입력은 이 파일에서 만든 합성 정제 표다. `_season_row`가 `contracts.BATTING_SEASONS`·`PITCHING_SEASONS`의
모든 열을 채우고, `_tables`가 `tables.Tables` 묶음을 만든다. 리그 평균은 `models.league`로 계산하고,
대상 표(cohorts)는 비워 두어 `tables.cohort_for`가 `cohort.build_cohort`로 만들게 한다.

세 가지 합성 데이터를 쓴다.

- 특성 손 계산(`HAND_*`, 목표 2025, 타자 3명·투수 3명): 시즌 나이(6월 30일 기준, 생년월일이 없으면
  기간 안 마지막 행의 나이 + 경과 연수), 기간 평균의 가중치(params 가중치 × pa/bf), 결측 행 제외,
  기간 밖·목표 시즌 행 제외를 손으로 계산해 맞춘다.
- 작은 세계(`TINY_*`, 목표 2022): 학습 목표가 2021 하나뿐이다. 회귀 표본 R = 0이고 기간 안 행이 2020
  하나라서 가중 비율 = 그 행의 비율이 되어, 보정 학습 행(특성·y)을 손으로 계산할 수 있다.
- 큰 세계(`_world`, 2019~2024, 타자·투수 각 80명, 시드 고정): 출전량 모델이 맞춰지고 보정 계수가
  0이 아니게 될 만큼 크다. 기대값은 M3 함수(`weighted_rates`·`predict_playing_time`·`actual_values`)와
  이 모듈 공개 함수의 조합(모듈 docstring의 결합 규칙)으로 만든다.

2026 규칙: 실제 데이터는 읽지 않는다. 모든 목표 시즌은 합성 세계 안의 값이다.
"""

import functools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlbproj import contracts as c
from mlbproj.age import season_age
from mlbproj.contracts import Role
from mlbproj.evaluate import actual_values
from mlbproj.models import extended as ext_mod
from mlbproj.models.assemble import assemble_bat, assemble_pit
from mlbproj.models.extended import (
    ROLE_FEATURES,
    Correction,
    corrections,
    extended_features,
    extended_projection,
    fit_correction,
    predict_delta,
    residual_frame,
    weighted_projection_for,
)
from mlbproj.models.league import league_batting, league_pitching
from mlbproj.models.params import NO_ADOPTION, Adoption, Params, RateParams
from mlbproj.models.playing_time import predict_playing_time
from mlbproj.models.weighted import weighted_rates
from mlbproj.tables import Tables, cohort_for

NAN = float("nan")
SPECS: dict[Role, c.TableSpec] = {"bat": c.BATTING_SEASONS, "pit": c.PITCHING_SEASONS}
PROJ_SPECS: dict[Role, c.TableSpec] = {"bat": c.BAT_PROJECTIONS, "pit": c.PIT_PROJECTIONS}
VOLUME_METRICS = frozenset({"pa", "ip"})
ROLE_METRIC_CASES: list[tuple[Role, str]] = [
    (role, metric) for role in c.ROLES for metric in c.EXT_ROLE_METRICS[role]
]
REL = 1e-9
ABS = 1e-12

# 테스트용 파라미터. 손 계산은 타자·투수 가중치가 다르고(역할별 가중치를 쓰는지 본다) 회귀 표본이 0이다.
P_HAND = Params(
    version="hand",
    bat=RateParams(weights=(2.0, 1.0, 1.0, 1.0, 1.0), regress=dict.fromkeys(c.BAT_COMPONENTS, 0.0)),
    pit=RateParams(
        weights=(3.0, 1.0, 1.0, 1.0, 1.0),
        regress={**dict.fromkeys(c.PIT_COMPONENTS, 0.0), "er": 0.0},
    ),
    era_method="er",
)
P_WORLD = Params(
    version="world",
    bat=RateParams(
        weights=(5.0, 4.0, 3.0, 2.0, 1.0), regress=dict.fromkeys(c.BAT_COMPONENTS, 300.0)
    ),
    pit=RateParams(
        weights=(3.0, 2.0, 1.5, 1.0, 0.5),
        regress={**dict.fromkeys(c.PIT_COMPONENTS, 250.0), "er": 200.0},
    ),
    era_method="fip",
)

# ---------------------------------------------------------------------------
# 도우미 — 계약 형식의 합성 표
# ---------------------------------------------------------------------------

_ADJUSTED: dict[Role, tuple[str, str]] = {"bat": ("pa_adj", "pa"), "pit": ("outs_adj", "outs")}


def _dtype(col: c.Column) -> str:
    if col.kind == "int":
        return "Int64" if col.nullable else "int64"
    if col.kind == "float":
        return "float64"
    if col.kind == "bool":
        return "bool"
    return "str"


def _season_row(role: Role, pid: int, season: int, **values: Any) -> dict[str, Any]:
    """계약의 모든 열을 채운 정제 표 한 행. 주지 않은 횟수는 0, 결측 허용 열(age·비율·Statcast)은 결측.

    타자 ab를 주지 않으면 pa − bb − hbp − sf − sh, 투수 g를 주지 않으면 10. 환산 출전량은 2020만 ×162/60.
    """
    spec = SPECS[role]
    unknown = set(values) - set(spec.column_names)
    assert not unknown, f"모르는 열: {unknown}"
    adj, base = _ADJUSTED[role]
    row: dict[str, Any] = {}
    for col in spec.columns:
        name = col.name
        if name in values:
            row[name] = values[name]
        elif name == "player_id":
            row[name] = pid
        elif name == "season":
            row[name] = season
        elif name == "league":
            row[name] = c.LEAGUE
        elif name == "team_name":
            row[name] = f"T{pid}"
        elif name == "n_teams":
            row[name] = 1
        elif name == "is_short_season":
            row[name] = season == c.SHORT_SEASON
        elif name == adj:
            continue
        elif col.kind == "int" and not col.nullable:
            row[name] = 0
        else:
            row[name] = None
    if role == "bat" and "ab" not in values:
        row["ab"] = row["pa"] - row["bb"] - row["hbp"] - row["sf"] - row["sh"]
    if role == "pit" and "g" not in values:
        row["g"] = 10
    scale = c.SHORT_SEASON_SCALE if season == c.SHORT_SEASON else 1.0
    row[adj] = float(row[base]) * scale
    return row


def _season_frame(role: Role, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    spec = SPECS[role]
    df = pd.DataFrame(
        {
            col.name: pd.Series([r[col.name] for r in rows], dtype=_dtype(col))
            for col in spec.columns
        }
    )
    assert c.check_frame(df, spec) == []
    return df


RowSpec = tuple[int, int, dict[str, Any]]


def _rows(role: Role, specs: Sequence[RowSpec]) -> list[dict[str, Any]]:
    return [_season_row(role, pid, season, **values) for pid, season, values in specs]


def _players(births: Mapping[int, str | None], active: bool, last_season: int) -> pd.DataFrame:
    """선수 표. active·last_mlb_season은 표시용이라 모델에 영향이 없어야 한다(N2)."""
    ids = sorted(births)
    n = len(ids)
    df = pd.DataFrame(
        {
            "player_id": pd.Series(ids, dtype="int64"),
            "name": pd.Series([f"Player {i}" for i in ids], dtype="str"),
            "birth_date": pd.Series([births[i] for i in ids], dtype="str"),
            "primary_position": pd.Series([None] * n, dtype="str"),
            "bats": pd.Series([None] * n, dtype="str"),
            "throws": pd.Series([None] * n, dtype="str"),
            "active": pd.Series([active] * n, dtype="bool"),
            "last_mlb_season": pd.Series([last_season] * n, dtype="int64"),
        }
    )
    assert c.check_frame(df, c.PLAYERS) == []
    return df


def _empty_cohorts() -> pd.DataFrame:
    df = pd.DataFrame({col.name: pd.Series(dtype=_dtype(col)) for col in c.COHORTS.columns})
    assert c.check_frame(df, c.COHORTS) == []
    return df


def _tables(
    bat_rows: Sequence[Mapping[str, Any]],
    pit_rows: Sequence[Mapping[str, Any]],
    births: Mapping[int, str | None],
    *,
    active: bool = True,
    last_season: int = 2024,
    cohorts: pd.DataFrame | None = None,
) -> Tables:
    bat = _season_frame("bat", bat_rows)
    pit = _season_frame("pit", pit_rows)
    return Tables(
        seasons={"bat": bat, "pit": pit},
        cohorts=_empty_cohorts() if cohorts is None else cohorts,
        league={"bat": league_batting(bat), "pit": league_pitching(pit)},
        debuts={},
        players=_players(births, active, last_season),
    )


def _projection(
    cohort: pd.DataFrame, role: Role, values: Mapping[int, Mapping[str, float]]
) -> pd.DataFrame:
    """대상 표로 만든 손 예측 표 (model "weighted", 계약 통과). values에 없는 값은 고정 기본값."""
    ids = [int(i) for i in cohort["player_id"]]
    defaults = (
        {"pa": 400.0, "hr": 12.0, "obp": 0.320, "slg": 0.400, "ops": 0.720, "lg_ops": 0.710}
        if role == "bat"
        else {"ip": 60.0, "era": 4.0, "k_bb_pct": 14.0, "lg_era": 4.1, "lg_k_bb_pct": 13.5}
    )
    data: dict[str, Any] = {
        "player_id": pd.Series(ids, dtype="int64"),
        "role": pd.Series([role] * len(ids), dtype="str"),
        "target_season": cohort["target_season"].astype("int64").to_numpy(),
        "model": pd.Series(["weighted"] * len(ids), dtype="str"),
        "input_first": cohort["input_first"].astype("int64").to_numpy(),
        "input_last": cohort["input_last"].astype("int64").to_numpy(),
        "n_seasons": cohort["n_seasons"].astype("int64").to_numpy(),
    }
    if role == "bat":
        data["sample_pa"] = cohort["sample"].astype("int64").to_numpy()
    else:
        data["sample_bf"] = cohort["sample"].astype("int64").to_numpy()
        data["sample_outs"] = cohort["sample_outs"].astype("int64").to_numpy()
    for name, default in defaults.items():
        data[name] = pd.Series(
            [float(values.get(i, {}).get(name, default)) for i in ids], dtype="float64"
        )
    df = pd.DataFrame(data)[PROJ_SPECS[role].column_names]
    assert c.check_frame(df, PROJ_SPECS[role]) == []
    return df


def _floats(df: pd.DataFrame, columns: Sequence[str]) -> npt.NDArray[np.float64]:
    """열들을 실수 배열로 (결측은 NaN)."""
    return df[list(columns)].to_numpy(dtype="float64", na_value=np.nan)


def _as_float(values: Any) -> npt.NDArray[np.float64]:
    """pandas 열·배열·목록을 실수 배열로 (pd.NA도 NaN)."""
    if isinstance(values, pd.Series | pd.DataFrame):
        return values.to_numpy(dtype="float64", na_value=np.nan)
    return np.asarray(values, dtype="float64")


def _assert_close(got: Any, expected: Any, rel: float = REL, abs_: float = ABS) -> None:
    """NaN 위치가 같고, 나머지 값이 가까운지."""
    g = _as_float(got)
    e = _as_float(expected)
    assert g.shape == e.shape, (g.shape, e.shape)
    assert np.array_equal(np.isnan(g), np.isnan(e)), (g, e)
    mask = ~np.isnan(e)
    np.testing.assert_allclose(g[mask], e[mask], rtol=rel, atol=abs_)


def _adopt(*metrics: str) -> Adoption:
    unknown = set(metrics) - set(c.EXT_METRICS)
    assert not unknown, unknown
    return Adoption(version="test", adopt={m: m in metrics for m in c.EXT_METRICS})


def _first_target(metric: str) -> int:
    return c.EXT_TRAIN_FIRST_VOLUME if metric in VOLUME_METRICS else c.EXT_TRAIN_FIRST


def _residual_columns(metric: str) -> list[str]:
    return ["train_target", "player_id", *c.EXT_FEATURES[metric], "y"]


def _residual_records(df: pd.DataFrame, metric: str) -> dict[tuple[int, int], tuple[float, ...]]:
    names = [*c.EXT_FEATURES[metric], "y"]
    values = _floats(df, names)
    keys = zip(df["train_target"].astype(int), df["player_id"].astype(int), strict=True)
    return {
        (int(t), int(p)): tuple(float(v) for v in row)
        for (t, p), row in zip(keys, values, strict=True)
    }


def _residual_y(df: pd.DataFrame) -> dict[tuple[int, int], float]:
    keys = zip(df["train_target"].astype(int), df["player_id"].astype(int), strict=True)
    ys = df["y"].to_numpy(dtype="float64", na_value=np.nan)
    return {(int(t), int(p)): float(y) for (t, p), y in zip(keys, ys, strict=True)}


def _assert_records(
    got: Mapping[tuple[int, int], tuple[float, ...]],
    expected: Mapping[tuple[int, int], tuple[float, ...]],
) -> None:
    assert sorted(got) == sorted(expected)
    for key, values in expected.items():
        try:
            _assert_close(got[key], values)
        except AssertionError as err:  # 어느 행인지 알 수 있게
            raise AssertionError(f"{key}: {got[key]} != {values}") from err


# ---------------------------------------------------------------------------
# 특성 손 계산 (목표 2025, 기간 2020~2024)
# ---------------------------------------------------------------------------

Y_HAND = 2025

# 타자 가중치 (P_HAND.bat): 2024 → 2, 2023 → 1, 2022 → 1, 2021 → 1, 2020 → 1
HAND_BAT: list[RowSpec] = [
    # 11: 생년월일 있음. 2019(기간 밖)·2025(목표 시즌) 행은 극단값이라 섞이면 바로 드러난다
    (11, 2019, {"pa": 600, "so": 300, "age": 23, "barrel_pct": 50.0, "ev_avg": 120.0}),
    (
        11,
        2022,
        {"pa": 400, "so": 100, "age": 26, "barrel_pct": 15.0, "ev_avg": 90.0}
        | {"woba": 0.300, "xwoba": 0.320},
    ),
    # barrel 결측
    (11, 2023, {"pa": 200, "so": 60, "age": 27, "ev_avg": 87.0, "woba": 0.330, "xwoba": 0.260}),
    # ev·woba 결측. 표의 age는 일부러 생년월일과 다르게 — 생년월일이 먼저다
    (11, 2024, {"pa": 500, "so": 100, "age": 40, "barrel_pct": 8.0, "xwoba": 0.400}),
    (
        11,
        2025,
        {"pa": 650, "so": 400, "age": 41, "barrel_pct": 60.0, "ev_avg": 130.0}
        | {"woba": 0.100, "xwoba": 0.900},
    ),
    # 12: players에 있지만 생년월일 결측 → 마지막 행(2023) 나이 32 + (2025 − 2023)
    (12, 2021, {"pa": 300, "so": 90, "age": 30}),
    (12, 2023, {"pa": 100, "so": 20, "age": 32}),
    # 13: players에 없음 → 2024 나이 24 + 1. xwoba 결측
    (13, 2024, {"pa": 50, "so": 5, "age": 24, "barrel_pct": 4.0, "ev_avg": 85.0, "woba": 0.250}),
]
# 투수 가중치 (P_HAND.pit): 2024 → 3, 2023 → 1, 2022 → 1
HAND_PIT: list[RowSpec] = [
    (21, 2019, {"bf": 900, "outs": 600, "age": 29, "whiff_pct": 99.0, "xera": 9.90}),  # 기간 밖
    (21, 2022, {"bf": 100, "outs": 70, "age": 32, "xera": 5.00}),  # whiff 결측
    (21, 2023, {"bf": 400, "outs": 300, "age": 33, "whiff_pct": 30.0, "xera": 3.00}),
    (21, 2024, {"bf": 200, "outs": 150, "age": 34, "whiff_pct": 25.0}),  # xera 결측
    (21, 2025, {"bf": 999, "outs": 700, "age": 35, "whiff_pct": 1.0, "xera": 0.10}),  # 목표 시즌
    (22, 2024, {"bf": 250, "outs": 180, "age": 28}),  # Statcast 없음
    (
        23,
        2023,
        {"bf": 120, "outs": 90, "whiff_pct": 20.0, "xera": 4.50},
    ),  # 나이 결측, players에 없음
]
HAND_BIRTHS: dict[int, str | None] = {
    11: "1995-07-15",  # 7월 15일생 — 2025-06-30엔 29세
    12: None,
    21: "1990-06-30",  # 6월 30일생 — 그날 35세
    22: None,
}
# 투수 가중 era (손 예측 표). xera_minus_proj = x̄(xera) − 이 값
HAND_PIT_ERA: dict[int, float] = {21: 3.9, 22: 4.2, 23: 4.0}

# ROLE_FEATURES["bat"] 순서: age_c, age_c2, barrel_pct, ev_avg, k_pct, xwoba_minus_woba
HAND_BAT_EXPECTED: dict[int, tuple[float, ...]] = {
    # age 29 → age_c 2, age_c2 4 (표 나이로 하면 41)
    # barrel: 2022(w·pa 1·400, 15.0)·2024(2·500 = 1000, 8.0) → (6000 + 8000) / 1400 = 10.0
    # ev: 2022(400, 90.0)·2023(200, 87.0) → (36000 + 17400) / 600 = 89.0
    # k_pct: 100 × (2·100 + 1·60 + 1·100) / (2·500 + 1·200 + 1·400) = 100 × 360 / 1600 = 22.5
    # xwoba−woba: 둘 다 있는 2022·2023만 — x̄(xwoba) = (400·.320 + 200·.260)/600 = .30,
    #   x̄(woba) = (400·.300 + 200·.330)/600 = .31 → −0.01 (2024 xwoba .400을 넣으면 달라진다)
    11: (2.0, 4.0, 10.0, 89.0, 22.5, -0.01),
    # age 34 → 7, 49. k_pct: 100 × (90 + 20) / (300 + 100) = 27.5. Statcast 행 없음 → 결측
    12: (7.0, 49.0, NAN, NAN, 27.5, NAN),
    # age 25 → −2, 4. k_pct 100 × 5 / 50 = 10.0. xwoba 결측 → 차이 결측
    13: (-2.0, 4.0, 4.0, 85.0, 10.0, NAN),
}
# ROLE_FEATURES["pit"] 순서: age_c, age_c2, xera_minus_proj, whiff_pct
HAND_PIT_EXPECTED: dict[int, tuple[float, ...]] = {
    # age 35 → 8, 64
    # xera: 2022(w·bf 1·100, 5.0)·2023(1·400, 3.0) → (500 + 1200) / 500 = 3.4 → 3.4 − 3.9 = −0.5
    # whiff: 2023(400, 30.0)·2024(3·200 = 600, 25.0) → (12000 + 15000) / 1000 = 27.0
    21: (8.0, 64.0, -0.5, 27.0),
    # 생년월일 결측 → 2024 나이 28 + 1 = 29 → 2, 4. Statcast 없음
    22: (2.0, 4.0, NAN, NAN),
    # 생년월일도 표 나이도 없음 → 결측. xera 4.5 − 4.0 = 0.5, whiff 20.0
    23: (NAN, NAN, 0.5, 20.0),
}
HAND_EXPECTED: dict[Role, dict[int, tuple[float, ...]]] = {
    "bat": HAND_BAT_EXPECTED,
    "pit": HAND_PIT_EXPECTED,
}


@functools.cache
def _hand_tables() -> Tables:
    return _tables(_rows("bat", HAND_BAT), _rows("pit", HAND_PIT), HAND_BIRTHS)


def _hand_weighted(role: Role) -> pd.DataFrame:
    tables = _hand_tables()
    cohort = cohort_for(tables, role, Y_HAND)
    values = {pid: {"era": era} for pid, era in HAND_PIT_ERA.items()} if role == "pit" else {}
    return _projection(cohort, role, values)


# ---------------------------------------------------------------------------
# 작은 세계 (목표 2022 → 학습 목표 2021 하나, 기간 2016~2020)
# ---------------------------------------------------------------------------

Y_TINY = 2022

TINY_BAT: list[RowSpec] = [
    # 31: 2020 단타 16 = 24 − 4 − 0 − 4. obp = 42/100, slg = (16 + 8 + 16)/80 = .50 → 가중 ops .92
    (
        31,
        2020,
        {"pa": 100, "ab": 80, "h": 24, "b2": 4, "hr": 4, "bb": 16, "hbp": 2, "sf": 2, "so": 20}
        | {"age": 30, "barrel_pct": 9.0, "ev_avg": 89.0, "woba": 0.380, "xwoba": 0.400},
    ),
    (
        31,
        2021,
        {"pa": 400, "ab": 350, "h": 100, "b2": 20, "b3": 2, "hr": 20, "bb": 40, "hbp": 5}
        | {"sf": 5, "so": 80, "ops": 0.800, "age": 31, "barrel_pct": 12.0, "ev_avg": 91.0},
    ),
    # 목표 시즌 Y — 학습·특성 어디에도 쓰면 안 된다
    (
        31,
        2022,
        {"pa": 650, "h": 200, "hr": 60, "so": 300, "ops": 1.500, "age": 32}
        | {"barrel_pct": 40.0, "ev_avg": 120.0, "woba": 0.900, "xwoba": 0.100},
    ),
    # 32: 생년월일 결측. 2020 단타 8. obp = 19/50 = .38, slg = 12/40 = .30 → 가중 ops .68, hr/pa 0
    (
        32,
        2020,
        {"pa": 50, "ab": 40, "h": 10, "b2": 2, "bb": 8, "hbp": 1, "sf": 1, "so": 15, "age": 25},
    ),
    # 2021 pa 200 — 경계값, 학습 행에 든다
    (
        32,
        2021,
        {"pa": 200, "ab": 180, "h": 45, "hr": 6, "bb": 15, "hbp": 3, "sf": 2, "so": 40}
        | {"ops": 0.650, "age": 26},
    ),
    (33, 2020, {"pa": 30, "h": 6, "so": 10, "age": 20}),
    (33, 2021, {"pa": 199, "h": 50, "hr": 5, "so": 40, "ops": 0.700, "age": 21}),  # 199 < 200
    (34, 2019, {"pa": 300, "h": 70, "hr": 10, "bb": 30, "so": 60, "age": 34}),  # 2021 행 없음
    (35, 2021, {"pa": 500, "h": 140, "hr": 25, "so": 100, "ops": 0.850, "age": 23}),  # 2021 데뷔
]
TINY_PIT: list[RowSpec] = [
    # 41: 가중 era = 27 × 20/150 = 3.6, 가중 K−BB% = 100 × (50 − 10)/200 = 20.0
    (
        41,
        2020,
        {"bf": 200, "outs": 150, "er": 20, "so": 50, "bb": 10, "hbp": 2, "hr": 6, "h": 45}
        | {"age": 27, "whiff_pct": 28.0, "xera": 3.10},
    ),
    # 실제 era = 27 × 30/300 = 2.7, K−BB% = 100 × (100 − 30)/400 = 17.5
    (
        41,
        2021,
        {"bf": 400, "outs": 300, "er": 30, "so": 100, "bb": 30, "hbp": 4, "hr": 10, "h": 85}
        | {"age": 28},
    ),
    (41, 2022, {"bf": 999, "outs": 900, "er": 1, "so": 900, "age": 29, "whiff_pct": 90.0}),  # Y
    # 42: 생년월일 결측. 가중 era = 27 × 5/75 = 1.8, K−BB% = 100 × 25/100 = 25.0
    (42, 2020, {"bf": 100, "outs": 75, "er": 5, "so": 30, "bb": 5, "age": 30}),
    # 실제 outs 150 — 경계값. era = 27 × 10/150 = 1.8, K−BB% = 100 × 21/210 = 10.0
    (42, 2021, {"bf": 210, "outs": 150, "er": 10, "so": 42, "bb": 21, "age": 31}),
    (43, 2020, {"bf": 80, "outs": 60, "er": 6, "so": 20, "bb": 8, "age": 24}),
    (43, 2021, {"bf": 200, "outs": 149, "er": 15, "so": 40, "bb": 15, "age": 25}),  # 149 < 150
    (44, 2019, {"bf": 300, "outs": 210, "er": 25, "so": 60, "bb": 20, "age": 32}),  # 2021 없음
]
TINY_BIRTHS: dict[int, str | None] = {
    31: "1990-03-01",  # 2021 시즌 31세 → age_c 4
    32: None,  # 2020 나이 25 + 1 = 26 → age_c −1
    33: "2000-12-01",
    34: "1985-05-05",
    35: "1998-01-01",
    41: "1992-08-15",  # 2021 시즌 28세 → age_c 1
    42: None,  # 2020 나이 30 + 1 = 31 → age_c 4
    43: "1995-01-01",
    44: "1988-02-02",
}
# (train_target, player_id) → (*EXT_FEATURES[metric], y). 특성은 T = 2021 기준(Y 자리에 T)
TINY_EXPECTED: dict[tuple[Role, str], dict[tuple[int, int], tuple[float, ...]]] = {
    # hr 특성 (age_c, barrel_pct, ev_avg), y = 실제 hr/pa − 가중 hr/pa
    ("bat", "hr"): {
        (2021, 31): (4.0, 9.0, 89.0, 0.01),  # 20/400 − 4/100 = 0.05 − 0.04
        (2021, 32): (-1.0, NAN, NAN, 0.03),  # 6/200 − 0/50
    },
    # ops 특성 (age_c, age_c2, k_pct, xwoba_minus_woba, barrel_pct, ev_avg), y = 실제 ops − 가중 ops
    ("bat", "ops"): {
        # k_pct 100 × 20/100, xwoba−woba .400 − .380, y .800 − .92
        (2021, 31): (4.0, 16.0, 20.0, 0.02, 9.0, 89.0, -0.12),
        # k_pct 100 × 15/50, y .650 − .68
        (2021, 32): (-1.0, 1.0, 30.0, NAN, NAN, NAN, -0.03),
    },
    # era 특성 (age_c, xera_minus_proj, whiff_pct), y = 실제 era − 가중 era
    ("pit", "era"): {
        (2021, 41): (1.0, -0.5, 28.0, -0.9),  # xera 3.10 − 3.6, y 2.7 − 3.6
        (2021, 42): (4.0, NAN, NAN, 0.0),  # 1.8 − 1.8
    },
    # k_bb_pct 특성 (age_c, whiff_pct), y = 실제 K−BB% − 가중 K−BB%
    ("pit", "k_bb_pct"): {
        (2021, 41): (1.0, 28.0, -2.5),  # 17.5 − 20.0
        (2021, 42): (4.0, NAN, -15.0),  # 10.0 − 25.0
    },
}


@functools.cache
def _tiny_tables() -> Tables:
    return _tables(_rows("bat", TINY_BAT), _rows("pit", TINY_PIT), TINY_BIRTHS)


# ---------------------------------------------------------------------------
# 큰 세계 (2019~2024, 시드 고정)
# ---------------------------------------------------------------------------

Y_WORLD = 2024
WORLD_SEED = 20261124
WORLD_N = 80  # 비율 지표 학습 행(OPS 96행 ≥ 70)이 넉넉하도록
WORLD_LAST = 2024
BAT_BASE = 1000
PIT_BASE = 2000
POISON = 99_999  # 목표 시즌 이후 행에 넣는 "독" 값 (N2)
NEW_PLAYER = 9_999  # 목표 시즌에만 기록이 있는 선수


def _world_birth(rng: np.random.Generator) -> str:
    year = int(rng.integers(1984, 2002))
    return f"{year:04d}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 29)):02d}"


def _world_career(rng: np.random.Generator) -> list[int]:
    """기록이 있는 시즌. 첫 시즌 2019~2022, 30%는 2022 이전에 그만두고, 중간 시즌은 10% 쉰다."""
    first = int(rng.integers(2019, 2023))
    last = int(rng.integers(first, 2023)) if rng.random() < 0.3 else WORLD_LAST
    return [s for s in range(first, last + 1) if s == first or rng.random() >= 0.1]


def _statcast(rng: np.random.Generator, values: dict[str, float]) -> dict[str, float | None]:
    """Statcast 값 — 10%는 행 전체 결측, 나머지는 열마다 8% 결측."""
    if rng.random() < 0.1:
        return dict.fromkeys(values)
    return {k: (None if rng.random() < 0.08 else v) for k, v in values.items()}


def _world_bat_row(
    rng: np.random.Generator, pid: int, season: int, age: int | None
) -> dict[str, Any]:
    pa = int(rng.integers(60, 700))
    bb = round(pa * rng.uniform(0.05, 0.13))
    hbp = round(pa * rng.uniform(0.0, 0.02))
    sf = round(pa * rng.uniform(0.0, 0.015))
    ab = pa - bb - hbp - sf
    h = round(ab * rng.uniform(0.20, 0.31))
    b2 = round(h * rng.uniform(0.12, 0.25))
    b3 = round(h * rng.uniform(0.0, 0.03))
    hr = min(round(ab * rng.uniform(0.005, 0.07)), h - b2 - b3)
    so = round(pa * rng.uniform(0.12, 0.32))
    obp = (h + bb + hbp) / (ab + bb + hbp + sf)
    slg = ((h - b2 - b3 - hr) + 2 * b2 + 3 * b3 + 4 * hr) / ab
    woba = obp * 0.88 + rng.normal(0.0, 0.01)
    statcast = _statcast(
        rng,
        {
            "woba": woba,
            "xwoba": woba + rng.normal(0.0, 0.015),
            "ev_avg": rng.uniform(85.0, 93.0),
            "la_avg": rng.uniform(4.0, 20.0),
            "hard_hit_pct": rng.uniform(30.0, 52.0),
            "barrel_pct": rng.uniform(2.0, 16.0),
        },
    )
    return _season_row(
        "bat",
        pid,
        season,
        pa=pa,
        ab=ab,
        h=h,
        b2=b2,
        b3=b3,
        hr=hr,
        bb=bb,
        ibb=min(bb, int(rng.integers(0, 4))),
        hbp=hbp,
        sf=sf,
        so=so,
        obp=obp,
        slg=slg,
        ops=obp + slg,
        age=age,
        **statcast,
    )


def _world_pit_row(
    rng: np.random.Generator, pid: int, season: int, age: int | None, starter: bool
) -> dict[str, Any]:
    g = int(rng.integers(15, 34)) if starter else int(rng.integers(15, 70))
    outs = int(rng.integers(240, 560)) if starter else int(rng.integers(45, 240))
    bf = round(outs * rng.uniform(1.3, 1.5))
    er = round(outs / 27 * rng.uniform(2.5, 6.0))
    statcast = _statcast(
        rng,
        {
            "xwoba_against": rng.uniform(0.26, 0.36),
            "whiff_pct": rng.uniform(18.0, 35.0),
            "hard_hit_pct": rng.uniform(30.0, 45.0),
            "barrel_pct": rng.uniform(4.0, 12.0),
            "xera": max(27 * er / outs + rng.normal(0.0, 0.6), 1.0),
        },
    )
    return _season_row(
        "pit",
        pid,
        season,
        g=g,
        gs=g if starter else 0,
        outs=outs,
        bf=bf,
        er=er,
        h=round(bf * rng.uniform(0.18, 0.26)),
        hr=round(bf * rng.uniform(0.015, 0.045)),
        bb=round(bf * rng.uniform(0.05, 0.12)),
        hbp=round(bf * rng.uniform(0.0, 0.015)),
        so=round(bf * rng.uniform(0.15, 0.33)),
        sv=0 if starter else int(rng.integers(0, 20)),
        age=age,
        **statcast,
    )


@functools.cache
def _world_data() -> tuple[
    tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[tuple[int, str | None], ...]
]:
    """(타자 행, 투수 행, 선수 생년월일). 선수 k마다

    - k % 11 == 5: players에 없음 (나이는 표의 age로)
    - k % 7 == 3: players에 있지만 생년월일 결측 (표의 age로)
    - k % 13 == 6: 생년월일도 표 나이도 없음 → 나이 결측
    """
    rng = np.random.default_rng(WORLD_SEED)
    bat: list[dict[str, Any]] = []
    pit: list[dict[str, Any]] = []
    births: dict[int, str | None] = {}
    for base in (BAT_BASE, PIT_BASE):
        for k in range(WORLD_N):
            pid = base + k + 1
            birth = _world_birth(rng)
            no_age = k % 13 == 6
            if k % 11 != 5:
                births[pid] = None if (k % 7 == 3 or no_age) else birth
            starter = rng.random() < 0.4
            for season in _world_career(rng):
                age = None if no_age else season_age(birth, season)
                if base == BAT_BASE:
                    bat.append(_world_bat_row(rng, pid, season, age))
                else:
                    pit.append(_world_pit_row(rng, pid, season, age, starter))
    return tuple(bat), tuple(pit), tuple(sorted(births.items()))


def _world_ids(role: Role) -> list[int]:
    base = BAT_BASE if role == "bat" else PIT_BASE
    return [base + k + 1 for k in range(WORLD_N)]


@functools.cache
def _world() -> Tables:
    """2019~2024 기록 전체 (목표 2024의 행도 있다 — 목표 2024 입력에 쓰이면 안 된다)."""
    bat, pit, births = _world_data()
    return _tables(bat, pit, dict(births), last_season=WORLD_LAST)


def _poison_row(role: Role, pid: int, season: int) -> dict[str, Any]:
    counts: tuple[str, ...]
    floats: tuple[str, ...]
    if role == "bat":
        counts = ("pa", "ab", "h", "b2", "b3", "hr", "bb", "ibb", "hbp", "sf", "sh", "so")
        floats = (
            "obp",
            "slg",
            "ops",
            "woba",
            "xwoba",
            "ev_avg",
            "la_avg",
            "hard_hit_pct",
            "barrel_pct",
        )
    else:
        counts = ("g", "gs", "outs", "bf", "er", "h", "hr", "bb", "hbp", "so", "sv")
        floats = ("xwoba_against", "whiff_pct", "hard_hit_pct", "barrel_pct", "xera")
    values: dict[str, Any] = {**dict.fromkeys(counts, POISON), **dict.fromkeys(floats, 999.0)}
    return _season_row(role, pid, season, age=99, **values)


@functools.cache
def _world_clean_and_poisoned() -> tuple[Tables, Tables]:
    """(Y 이전 행만, Y 이후 행·리그 평균·표시용 열을 바꾼 묶음).

    독: 모든 선수의 2025·2026 시즌 행(값 POISON), 2024에만 기록이 있는 새 선수, 세계의 실제 2024 행.
    독 행을 앞에 둬서 입력 순서도 섞는다. 리그 평균은 독이 든 표로 다시 계산한다(Y 이후 행이 생긴다).
    players는 active를 뒤집고 last_mlb_season을 2026으로 바꾸고 새 선수를 더한다.
    """
    bat, pit, births = _world_data()
    clean_rows: dict[Role, list[dict[str, Any]]] = {
        "bat": [r for r in bat if r["season"] < Y_WORLD],
        "pit": [r for r in pit if r["season"] < Y_WORLD],
    }
    poisoned_rows: dict[Role, list[dict[str, Any]]] = {}
    role_rows: tuple[tuple[Role, tuple[dict[str, Any], ...]], ...] = (("bat", bat), ("pit", pit))
    for role, rows in role_rows:
        poison = [_poison_row(role, pid, s) for pid in _world_ids(role) for s in (2025, 2026)]
        poison.append(_poison_row(role, NEW_PLAYER, Y_WORLD))
        poisoned_rows[role] = [*poison, *rows]
    clean = _tables(
        clean_rows["bat"], clean_rows["pit"], dict(births), active=True, last_season=Y_WORLD - 1
    )
    poisoned = _tables(
        poisoned_rows["bat"],
        poisoned_rows["pit"],
        {**dict(births), NEW_PLAYER: "1999-01-01"},
        active=False,
        last_season=2026,
    )
    assert int(poisoned.league["bat"]["season"].max()) == 2026
    return clean, poisoned


# ---------------------------------------------------------------------------
# 기대값 도우미 — M3 함수와 모듈 docstring 규칙
# ---------------------------------------------------------------------------


def _rates(tables: Tables, role: Role, target: int, params: Params) -> pd.DataFrame:
    """가중 비율 (player_id 색인)."""
    cohort = cohort_for(tables, role, target)
    rates = weighted_rates(cohort, tables.seasons[role], tables.league[role], role, params)
    return rates.set_index("player_id")


def _playing_time(tables: Tables, role: Role, target: int) -> pd.DataFrame:
    """가중 출전량 (player_id 색인, 타자 pa / 투수 outs)."""
    cohort = cohort_for(tables, role, target)
    pt = predict_playing_time(cohort, tables.seasons[role], role, tables.debuts)
    return pt.set_index("player_id")


def _rate_weighted_frame(tables: Tables, role: Role, target: int, params: Params) -> pd.DataFrame:
    """출전량 없이(0) 가중 비율로만 만든 예측 표 — 출전량 모델이 맞춰지지 않는 학습 목표(2021)용."""
    cohort = cohort_for(tables, role, target)
    rates = weighted_rates(cohort, tables.seasons[role], tables.league[role], role, params)
    column = "pa" if role == "bat" else "outs"
    zero = pd.DataFrame(
        {
            "player_id": cohort["player_id"].astype("int64").to_numpy(),
            "target_season": np.full(len(cohort), target, dtype="int64"),
            column: np.zeros(len(cohort)),
        }
    )
    if role == "bat":
        return assemble_bat(cohort, rates, zero, "weighted")
    return assemble_pit(cohort, rates, zero, "weighted")


def _expected_residuals(
    tables: Tables, role: Role, metric: str, target: int, params: Params
) -> dict[tuple[int, int], float]:
    """(train_target, player_id) → y. 모듈 docstring "보정 대상 y"를 M3 함수로 그대로 계산한다."""
    out: dict[tuple[int, int], float] = {}
    for train in range(_first_target(metric), target):
        cohort = cohort_for(tables, role, train)
        ids = [int(i) for i in cohort["player_id"]]
        actual = actual_values(tables.seasons[role], role, train).set_index("player_id")
        actual = actual.reindex(ids)
        if metric in VOLUME_METRICS:
            column = "pa" if role == "bat" else "outs"
            # 학습 행: T 대상 전원, T에 행이 없으면 실제 0
            y = actual[column].astype("float64").fillna(0.0) - _playing_time(tables, role, train)[
                column
            ].reindex(ids)
            keep = pd.Series(True, index=ids)
        else:
            rates = _rates(tables, role, train, params).reindex(ids)
            if role == "bat":
                keep = actual["pa"] >= c.RATE_MIN_PA
                if metric == "hr":
                    y = actual["hr"] / actual["pa"] - rates["hr_per_pa"]
                else:
                    y = actual["ops"] - rates["ops"]
            else:
                keep = actual["outs"] >= c.RATE_MIN_OUTS
                y = actual[metric] - rates[metric]
        for pid in ids:
            if bool(keep[pid]):  # 실제 행이 없으면(NaN) 비교가 거짓
                out[(train, pid)] = float(y[pid])
    return out


@functools.cache
def _world_residual(role: Role, metric: str, target: int = Y_WORLD) -> pd.DataFrame:
    return residual_frame(_world(), role, metric, target, P_WORLD)


@functools.cache
def _world_weighted(role: Role) -> pd.DataFrame:
    return weighted_projection_for(_world(), role, Y_WORLD, P_WORLD)


@functools.cache
def _world_features(role: Role) -> pd.DataFrame:
    return extended_features(_world(), role, Y_WORLD, P_WORLD, _world_weighted(role))


@functools.cache
def _world_delta(role: Role, metric: str) -> tuple[float, ...]:
    """목표 Y의 Δ = predict_delta(특성, fit_correction(residual_frame)) — 모듈 docstring의 순서."""
    correction = fit_correction(_world_residual(role, metric), metric)
    delta = predict_delta(_world_features(role), correction)
    return tuple(float(v) for v in delta.to_numpy(dtype="float64"))


def _expected_projection(
    role: Role, deltas: Mapping[str, npt.NDArray[np.float64] | float]
) -> pd.DataFrame:
    """가중 예측 + 채택 지표 Δ (모듈 docstring "예측 결합"). deltas에 없는 지표는 채택하지 않음."""
    weighted = _world_weighted(role)
    out = weighted.copy()
    out["model"] = "extended"
    ids = weighted["player_id"].to_numpy()
    if role == "bat":
        hr_per_pa_w = _rates(_world(), "bat", Y_WORLD, P_WORLD)["hr_per_pa"].reindex(ids)
        pa_e = weighted["pa"].to_numpy(dtype="float64")
        hr_per_pa_e = hr_per_pa_w.to_numpy(dtype="float64")
        if "pa" in deltas:
            pa_e = np.maximum(pa_e + deltas["pa"], 0.0)
        if "hr" in deltas:
            hr_per_pa_e = np.maximum(hr_per_pa_e + deltas["hr"], 0.0)
        out["pa"] = pa_e
        out["hr"] = pa_e * hr_per_pa_e
        if "ops" in deltas:
            ops_w = weighted["ops"].to_numpy(dtype="float64")
            ops_e = np.maximum(ops_w + deltas["ops"], 0.0)
            out["obp"] = weighted["obp"].to_numpy(dtype="float64") * ops_e / ops_w
            out["slg"] = weighted["slg"].to_numpy(dtype="float64") * ops_e / ops_w
            out["ops"] = ops_e
    else:
        if "ip" in deltas:
            outs_w = _playing_time(_world(), "pit", Y_WORLD)["outs"].reindex(ids)
            out["ip"] = np.maximum(outs_w.to_numpy(dtype="float64") + deltas["ip"], 0.0) / 3
        if "era" in deltas:
            out["era"] = np.maximum(weighted["era"].to_numpy(dtype="float64") + deltas["era"], 0.0)
        if "k_bb_pct" in deltas:
            out["k_bb_pct"] = weighted["k_bb_pct"].to_numpy(dtype="float64") + deltas["k_bb_pct"]
    return out


# 지표 → 예측 표에서 바뀌는 열
CHANGED: dict[str, tuple[str, ...]] = {
    "pa": ("pa", "hr"),
    "hr": ("hr",),
    "ops": ("obp", "slg", "ops"),
    "ip": ("ip",),
    "era": ("era",),
    "k_bb_pct": ("k_bb_pct",),
}


def _assert_projection(got: pd.DataFrame, expected: pd.DataFrame, adopted: Sequence[str]) -> None:
    """채택 지표가 바꾸는 열은 가깝게, 나머지 열은 가중 값 그대로(정확히) 같아야 한다."""
    role: Role = "bat" if "pa" in expected.columns else "pit"
    assert c.check_frame(got, PROJ_SPECS[role]) == []
    changed = sorted({col for m in adopted for col in CHANGED[m]})
    same = [col for col in expected.columns if col not in changed]
    assert_frame_equal(
        got[same].reset_index(drop=True), expected[same].reset_index(drop=True), check_exact=True
    )
    for col in changed:
        _assert_close(got[col].to_numpy(dtype="float64"), expected[col].to_numpy(dtype="float64"))


def _constant_fit(monkeypatch: pytest.MonkeyPatch, deltas: Mapping[str, float]) -> list[str]:
    """`fit_correction`을 절편만 있는 보정(Δ = 상수)으로 바꾼다. 호출된 지표를 기록한다.

    모듈 docstring대로 extended_projection이 residual_frame → fit_correction → predict_delta로 보정하면
    Δ가 정확히 그 상수가 되어, 결합 규칙(자르기·비율 조정·아웃 단위)을 손으로 확인할 수 있다.
    """
    calls: list[str] = []

    def fake(frame: pd.DataFrame, metric: str) -> Correction:
        calls.append(metric)
        features = c.EXT_FEATURES[metric]
        coef = np.zeros(len(features) + 1, dtype="float64")
        coef[0] = deltas[metric]
        return Correction(metric, features, coef, dict.fromkeys(features, 0.0), len(frame))

    monkeypatch.setattr(ext_mod, "fit_correction", fake)
    return calls


# ---------------------------------------------------------------------------
# 상수·세계 점검 (스텁과 무관 — 테스트 데이터가 충분한지)
# ---------------------------------------------------------------------------


def test_models_extended_role_features_cover_metric_features() -> None:
    for role in c.ROLES:
        seen: list[str] = []
        for metric in c.EXT_ROLE_METRICS[role]:
            seen.extend(f for f in c.EXT_FEATURES[metric] if f not in seen)
        assert ROLE_FEATURES[role] == tuple(seen)


def test_models_extended_world_is_big_enough() -> None:
    """큰 세계의 학습 행 수가 (특성 수 + 1) × 10 이상이라 보정 계수가 0이 아니게 된다.

    또 목표 Y의 가중 출전량이 0인 타자(직전 시즌 쉰 선수 구간 중앙값 0)가 있어야
    "hr = pa_e × 가중 hr_per_pa" 규칙을 hr / pa로 되돌려 쓰는 구현과 구별된다.
    """
    world = _world()
    for role, metric in ROLE_METRIC_CASES:
        rows = _expected_residuals(world, role, metric, Y_WORLD, P_WORLD)
        need = (len(c.EXT_FEATURES[metric]) + 1) * c.EXT_MIN_ROWS_PER_COEF
        assert len(rows) >= need, (role, metric, len(rows), need)
    assert bool((_playing_time(world, "bat", Y_WORLD)["pa"] == 0).any())
    assert bool((_playing_time(world, "bat", Y_WORLD)["pa"] > 0).any())
    # 나이 결측 선수(생년월일·표 나이 모두 없음)가 목표 Y 대상에 있다
    no_age = {BAT_BASE + k + 1 for k in range(WORLD_N) if k % 13 == 6}
    assert no_age & set(cohort_for(world, "bat", Y_WORLD)["player_id"])


# ---------------------------------------------------------------------------
# extended_features
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", c.ROLES)
def test_models_extended_features_hand_values(role: Role) -> None:
    out = extended_features(_hand_tables(), role, Y_HAND, P_HAND, _hand_weighted(role))
    assert list(out.columns) == ["player_id", *ROLE_FEATURES[role]]
    expected = HAND_EXPECTED[role]
    assert [int(i) for i in out["player_id"]] == sorted(expected)  # player_id 순
    values = _floats(out, ROLE_FEATURES[role])
    for row, pid in zip(values, sorted(expected), strict=True):
        try:
            _assert_close(row, expected[pid])
        except AssertionError as err:
            raise AssertionError(f"{role} {pid}: {row} != {expected[pid]}") from err


@pytest.mark.parametrize("role", c.ROLES)
def test_models_extended_features_players_follow_cohort(role: Role) -> None:
    tables = _hand_tables()
    out = extended_features(tables, role, Y_HAND, P_HAND, _hand_weighted(role))
    cohort_ids = sorted(int(i) for i in cohort_for(tables, role, Y_HAND)["player_id"])
    assert [int(i) for i in out["player_id"]] == cohort_ids
    assert out["player_id"].is_unique


def test_models_extended_features_age_uses_birth_date_june_30() -> None:
    """생년월일이 있으면 표의 나이를 쓰지 않는다. 6월 30일 기준이라 같은 해 7월생은 한 살 적다."""
    out = extended_features(_hand_tables(), "bat", Y_HAND, P_HAND, _hand_weighted("bat"))
    age_c = _floats(out.set_index("player_id").loc[[11]], ["age_c"])[0, 0]
    birth_age = season_age("1995-07-15", Y_HAND)
    assert birth_age == 29
    assert age_c == pytest.approx(birth_age - c.AGE_CENTER)  # 29 − 27 = 2, 표 나이 40 + 1이면 14


def test_models_extended_features_xera_uses_passed_weighted_era() -> None:
    """xera_minus_proj = 기간 x̄(xera) − 넘겨받은 가중 era. 가중 era를 바꾸면 그만큼 바뀐다."""
    tables = _hand_tables()
    cohort = cohort_for(tables, "pit", Y_HAND)
    shifted = _projection(
        cohort, "pit", {pid: {"era": era + 1.0} for pid, era in HAND_PIT_ERA.items()}
    )
    base = extended_features(tables, "pit", Y_HAND, P_HAND, _hand_weighted("pit"))
    moved = extended_features(tables, "pit", Y_HAND, P_HAND, shifted)
    _assert_close(moved["xera_minus_proj"], base["xera_minus_proj"] - 1.0)
    other = [f for f in ROLE_FEATURES["pit"] if f != "xera_minus_proj"]
    _assert_close(_floats(moved, other), _floats(base, other))


# 시즌마다 가중치가 다르다: 타자 Y−1 … Y−5 = 16·8·4·2·1, 투수는 거꾸로 1·2·4·8·16 (합 31).
# 가중치를 거꾸로 읽거나 한 칸 밀리면 손 계산 값과 어긋난다.
P_ORDER = Params(
    version="order",
    bat=RateParams(
        weights=(16.0, 8.0, 4.0, 2.0, 1.0), regress=dict.fromkeys(c.BAT_COMPONENTS, 0.0)
    ),
    pit=RateParams(
        weights=(1.0, 2.0, 4.0, 8.0, 16.0),
        regress={**dict.fromkeys(c.PIT_COMPONENTS, 0.0), "er": 0.0},
    ),
    era_method="er",
)
ORDER_BAT: list[RowSpec] = [
    # 51: 2020~2024 모두 실제 pa 100. 2020(단축 시즌)도 pa 100으로 가중한다 (pa_adj 270이 아니다)
    (51, 2020, {"pa": 100, "so": 10, "ev_avg": 50.0}),
    (51, 2021, {"pa": 100, "so": 20, "ev_avg": 60.0}),
    (51, 2022, {"pa": 100, "so": 30, "ev_avg": 70.0}),
    (51, 2023, {"pa": 100, "so": 40, "ev_avg": 80.0}),
    (51, 2024, {"pa": 100, "so": 50, "ev_avg": 90.0}),
    # 52: 생년월일 결측, 기간 안 마지막 행(2024)의 나이가 결측 → 나이 결측
    #     (2022 나이 30 + 3으로 대신하지 않는다 — "기간 안 마지막 행의 age, 그것도 없으면 결측")
    (52, 2022, {"pa": 200, "so": 40, "age": 30}),
    (52, 2024, {"pa": 300, "so": 60}),
    # 53: players에 없음. 마지막 행 2023 나이 27 → 2025 나이 27 + 2 = 29
    (53, 2021, {"pa": 150, "so": 30, "age": 25}),
    (53, 2023, {"pa": 250, "so": 50, "age": 27}),
]
ORDER_PIT: list[RowSpec] = [
    (61, 2020, {"bf": 100, "outs": 75, "whiff_pct": 10.0, "xera": 2.0}),
    (61, 2021, {"bf": 100, "outs": 75, "whiff_pct": 20.0, "xera": 3.0}),
    (61, 2022, {"bf": 100, "outs": 75, "whiff_pct": 30.0, "xera": 4.0}),
    (61, 2023, {"bf": 100, "outs": 75, "whiff_pct": 40.0, "xera": 5.0}),
    (61, 2024, {"bf": 100, "outs": 75, "whiff_pct": 50.0, "xera": 6.0}),
]
ORDER_BIRTHS: dict[int, str | None] = {51: "1996-01-01", 52: None, 61: "1996-01-01"}
ORDER_EXPECTED: dict[Role, dict[int, tuple[float, ...]]] = {
    # ROLE_FEATURES["bat"]: age_c, age_c2, barrel_pct, ev_avg, k_pct, xwoba_minus_woba
    "bat": {
        # 1996-01-01생 → 2025 시즌 29세 → age_c 2
        # ev: pa가 모두 100이라 Σw·ev / Σw = (16·90 + 8·80 + 4·70 + 2·60 + 1·50) / 31
        #     = (1440 + 640 + 280 + 120 + 50) / 31 = 2530 / 31 (거꾸로 읽으면 1810 / 31)
        # k_pct: 100 × (16·50 + 8·40 + 4·30 + 2·20 + 1·10) / (31 · 100) = 100 × 1290 / 3100
        51: (2.0, 4.0, NAN, 2530 / 31, 1290 / 31, NAN),
        # k_pct: 2024 w 16, 2022 w 4 → 100 × (16·60 + 4·40) / (16·300 + 4·200) = 100 × 1120 / 5600
        52: (NAN, NAN, NAN, NAN, 20.0, NAN),
        # k_pct: 2023 w 8, 2021 w 2 → 100 × (8·50 + 2·30) / (8·250 + 2·150) = 100 × 460 / 2300
        53: (2.0, 4.0, NAN, NAN, 20.0, NAN),
    },
    # ROLE_FEATURES["pit"]: age_c, age_c2, xera_minus_proj, whiff_pct
    "pit": {
        # 투수 가중치: 2024 → 1, 2023 → 2, 2022 → 4, 2021 → 8, 2020 → 16 (bf가 모두 100)
        # xera: (16·2 + 8·3 + 4·4 + 2·5 + 1·6) / 31 = 88 / 31 → 가중 era 2.0을 빼면 26 / 31
        # whiff: (16·10 + 8·20 + 4·30 + 2·40 + 1·50) / 31 = 570 / 31
        61: (2.0, 4.0, 26 / 31, 570 / 31),
    },
}


@pytest.mark.parametrize("role", c.ROLES)
def test_models_extended_features_season_weights_and_age_fallback(role: Role) -> None:
    """시즌 Y−k의 가중치 = weights[k−1] (역할별), 분모는 실제 pa·bf, 나이는 기간 안 마지막 행만 대신 쓴다."""
    tables = _tables(_rows("bat", ORDER_BAT), _rows("pit", ORDER_PIT), ORDER_BIRTHS)
    cohort = cohort_for(tables, role, Y_HAND)
    weighted = _projection(cohort, role, {61: {"era": 2.0}} if role == "pit" else {})
    out = extended_features(tables, role, Y_HAND, P_ORDER, weighted)
    expected = ORDER_EXPECTED[role]
    assert [int(i) for i in out["player_id"]] == sorted(expected)
    for row, pid in zip(_floats(out, ROLE_FEATURES[role]), sorted(expected), strict=True):
        try:
            _assert_close(row, expected[pid])
        except AssertionError as err:
            raise AssertionError(f"{role} {pid}: {row} != {expected[pid]}") from err


@pytest.mark.parametrize("case", ["target", "model", "missing", "extra", "role"])
def test_models_extended_features_rejects_mismatched_weighted(case: str) -> None:
    """다른 목표·모델·선수의 예측 표를 받으면 xera_minus_proj가 조용히 결측이 되지 않게 멈춘다."""
    weighted = _hand_weighted("pit")
    bad = {
        "target": lambda: weighted.assign(target_season=Y_HAND - 1),
        "model": lambda: weighted.assign(model="extended"),
        "missing": lambda: weighted.iloc[1:].reset_index(drop=True),
        "extra": lambda: pd.concat(
            [weighted, weighted.iloc[[0]].assign(player_id=999)], ignore_index=True
        ),
        "role": lambda: _hand_weighted("bat"),  # 타자 예측 표 — 투수 계약과 열이 다르다
    }[case]()
    with pytest.raises(ValueError, match="가중 예측 표"):
        extended_features(_hand_tables(), "pit", Y_HAND, P_HAND, bad)


def test_models_extended_features_ignore_rows_from_target_season() -> None:
    """목표 시즌 이후 행을 지워도(또는 남겨도) 특성은 같다 — 손 표에는 2025 행이 있다."""
    tables = _hand_tables()
    trimmed = _tables(
        [r for r in _rows("bat", HAND_BAT) if r["season"] < Y_HAND],
        [r for r in _rows("pit", HAND_PIT) if r["season"] < Y_HAND],
        HAND_BIRTHS,
    )
    for role in c.ROLES:
        weighted = _hand_weighted(role)
        assert_frame_equal(
            extended_features(tables, role, Y_HAND, P_HAND, weighted).reset_index(drop=True),
            extended_features(trimmed, role, Y_HAND, P_HAND, weighted).reset_index(drop=True),
            check_exact=True,
        )


# ---------------------------------------------------------------------------
# residual_frame — 작은 세계 손 계산
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("role", "metric"), sorted(TINY_EXPECTED))
def test_models_extended_residual_hand_rows(role: Role, metric: str) -> None:
    out = residual_frame(_tiny_tables(), role, metric, Y_TINY, P_HAND)
    assert list(out.columns) == _residual_columns(metric)
    _assert_records(_residual_records(out, metric), TINY_EXPECTED[(role, metric)])


def test_models_extended_residual_drops_rate_rows_without_actual_value() -> None:
    """실제 pa ≥ 200인데 실제 ops가 결측이면 ops 학습 행에서 뺀다 (y를 맞출 수 없다 — 평가의 대표 집단과 같다).

    같은 선수는 hr 학습 행에는 든다. 36: 2020 pa 100·hr 2 → 가중 hr/pa = 2·2 / (2·100) = .02 (R = 0),
    2021 pa 300·hr 9·ops 결측 → hr y = 9/300 − .02 = .01. players에 없어 2020 나이 28 + 1 = 29 → age_c 2.
    """
    extra: list[RowSpec] = [
        (36, 2020, {"pa": 100, "h": 20, "hr": 2, "so": 20, "age": 28}),
        (36, 2021, {"pa": 300, "h": 80, "hr": 9, "so": 60}),
    ]
    tables = _tables(_rows("bat", [*TINY_BAT, *extra]), _rows("pit", TINY_PIT), TINY_BIRTHS)
    ops = residual_frame(tables, "bat", "ops", Y_TINY, P_HAND)
    _assert_records(_residual_records(ops, "ops"), TINY_EXPECTED[("bat", "ops")])
    hr = residual_frame(tables, "bat", "hr", Y_TINY, P_HAND)
    expected_hr = {**TINY_EXPECTED[("bat", "hr")], (2021, 36): (2.0, NAN, NAN, 0.01)}
    _assert_records(_residual_records(hr, "hr"), expected_hr)


@pytest.mark.parametrize(
    ("role", "metric", "target"),
    [
        ("bat", "pa", Y_TINY),  # 학습 목표 2023 … 2021 — 없음
        ("pit", "ip", Y_TINY),
        ("bat", "hr", 2021),  # 2021 … 2020 — 없음
        ("bat", "ops", 2021),
        ("pit", "era", 2021),
        ("pit", "k_bb_pct", 2021),
    ],
)
def test_models_extended_residual_empty_without_training_target(
    role: Role, metric: str, target: int
) -> None:
    out = residual_frame(_tiny_tables(), role, metric, target, P_HAND)
    assert out.empty
    assert list(out.columns) == _residual_columns(metric)


@pytest.mark.parametrize(
    ("role", "metric"),
    [("bat", "era"), ("bat", "ip"), ("pit", "ops"), ("pit", "pa"), ("bat", "obp"), ("pit", "")],
)
def test_models_extended_residual_rejects_metric_of_other_role(role: Role, metric: str) -> None:
    with pytest.raises(ValueError):
        residual_frame(_tiny_tables(), role, metric, Y_TINY, P_HAND)


# ---------------------------------------------------------------------------
# residual_frame — 큰 세계
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("role", "metric"), ROLE_METRIC_CASES)
def test_models_extended_residual_world_columns_order_and_targets(role: Role, metric: str) -> None:
    out = _world_residual(role, metric)
    assert list(out.columns) == _residual_columns(metric)
    keys = list(zip(out["train_target"].astype(int), out["player_id"].astype(int), strict=True))
    assert keys == sorted(keys)  # (train_target, player_id) 순
    assert len(set(keys)) == len(keys)
    # 학습 목표: 출전량 2023 … Y−1, 비율 2021 … Y−1 (Y = 2024)
    assert sorted({t for t, _ in keys}) == list(range(_first_target(metric), Y_WORLD))


@pytest.mark.parametrize(("role", "metric"), ROLE_METRIC_CASES)
def test_models_extended_residual_world_population_and_y(role: Role, metric: str) -> None:
    """학습 행 집합(pa·ip: 대상 전원, 비율: 실제 표본 기준 이상)과 y가 M3 함수로 계산한 값과 같다."""
    out = _world_residual(role, metric)
    got = _residual_y(out)
    assert len(got) == len(out)
    expected = _expected_residuals(_world(), role, metric, Y_WORLD, P_WORLD)
    assert sorted(got) == sorted(expected)
    keys = sorted(expected)
    _assert_close([got[k] for k in keys], [expected[k] for k in keys])


@pytest.mark.parametrize(("role", "metric"), ROLE_METRIC_CASES)
def test_models_extended_residual_world_features_match_extended_features(
    role: Role, metric: str
) -> None:
    """학습 행의 특성 = 그 학습 목표 T의 extended_features (결측을 채우기 전 값)."""
    out = _world_residual(role, metric)
    names = list(c.EXT_FEATURES[metric])
    for target in sorted({int(t) for t in out["train_target"]}):
        part = out[out["train_target"] == target]
        weighted = _rate_weighted_frame(_world(), role, target, P_WORLD)
        feats = extended_features(_world(), role, target, P_WORLD, weighted).set_index("player_id")
        expected = feats.loc[[int(i) for i in part["player_id"]], names]
        _assert_close(_floats(part, names), _floats(expected, names))


def test_models_extended_residual_targets_run_to_y_minus_1() -> None:
    """목표 2025면 비율 지표는 2021~2024, 출전량 지표는 2023~2024를 학습한다."""
    for role, metric in [("bat", "pa"), ("bat", "hr"), ("pit", "ip"), ("pit", "era")]:
        out = _world_residual(role, metric, 2025)
        assert sorted(set(out["train_target"].astype(int))) == list(
            range(_first_target(metric), 2025)
        ), (role, metric)


def test_models_extended_residual_uses_cohort_for() -> None:
    """학습 목표 T의 대상은 `tables.cohort_for` — build가 만든 대상 표가 있으면 그것을 쓴다."""
    world = _world()
    built = cohort_for(world, "bat", 2023)
    dropped = int(built["player_id"].iloc[0])
    stored = built[built["player_id"] != dropped].reset_index(drop=True)
    tables = Tables(
        seasons=world.seasons,
        cohorts=stored,
        league=world.league,
        debuts=world.debuts,
        players=world.players,
    )
    for metric in ("pa", "ops"):
        out = residual_frame(tables, "bat", metric, Y_WORLD, P_WORLD)
        rows_2023 = set(out.loc[out["train_target"] == 2023, "player_id"].astype(int))
        assert dropped not in rows_2023, metric
        assert rows_2023 <= set(stored["player_id"].astype(int)), metric
    # 출전량 지표의 2023 학습 행 = 저장된 대상 표 전원
    out = residual_frame(tables, "bat", "pa", Y_WORLD, P_WORLD)
    assert set(out.loc[out["train_target"] == 2023, "player_id"].astype(int)) == set(
        stored["player_id"].astype(int)
    )


# ---------------------------------------------------------------------------
# fit_correction
# ---------------------------------------------------------------------------

# 지표 → 손으로 정한 계수 [절편, 특성 계수…] (EXT_FEATURES 순)
FIT_BETA: dict[str, tuple[float, ...]] = {
    "pa": (5.0, 2.0, -0.5),
    "hr": (0.01, 0.002, 0.001, -0.0005),
    "ops": (0.02, -0.003, 0.0001, 0.001, 0.5, 0.002, -0.0004),
    "ip": (-3.0, 4.0, -1.5),
    "era": (0.1, 0.02, 0.5, -0.01),
    "k_bb_pct": (1.0, -0.2, 0.3),
}


def _feature_value(rng: np.random.Generator, name: str, age_c: float) -> float:
    if name == "age_c":
        return age_c
    if name == "age_c2":
        return age_c * age_c
    ranges = {
        "barrel_pct": (2.0, 16.0),
        "ev_avg": (85.0, 93.0),
        "k_pct": (12.0, 32.0),
        "xwoba_minus_woba": (-0.04, 0.04),
        "xera_minus_proj": (-1.5, 1.5),
        "whiff_pct": (18.0, 35.0),
    }
    low, high = ranges[name]
    return float(rng.uniform(low, high))


def _linear_frame(metric: str, n_rows: int, seed: int = 7) -> pd.DataFrame:
    """y = FIT_BETA[metric] · [1, 특성…]를 정확히 따르는 학습 행 (residual_frame 열 순서)."""
    rng = np.random.default_rng(seed)
    names = c.EXT_FEATURES[metric]
    beta = FIT_BETA[metric]
    data: dict[str, list[float]] = {name: [] for name in names}
    ys: list[float] = []
    for i in range(n_rows):
        age_c = float(i % 17 - 6)  # −6 … 10
        row = [_feature_value(rng, name, age_c) for name in names]
        for name, value in zip(names, row, strict=True):
            data[name].append(value)
        ys.append(beta[0] + sum(b * x for b, x in zip(beta[1:], row, strict=True)))
    frame = pd.DataFrame(
        {
            "train_target": pd.Series([2021 + i % 3 for i in range(n_rows)], dtype="int64"),
            "player_id": pd.Series(range(1, n_rows + 1), dtype="int64"),
            **{name: pd.Series(values, dtype="float64") for name, values in data.items()},
            "y": pd.Series(ys, dtype="float64"),
        }
    )
    return frame.sort_values(["train_target", "player_id"], kind="mergesort").reset_index(drop=True)


def _need(metric: str) -> int:
    return (len(c.EXT_FEATURES[metric]) + 1) * c.EXT_MIN_ROWS_PER_COEF


@pytest.mark.parametrize("metric", c.EXT_METRICS)
def test_models_extended_fit_recovers_linear_rule(metric: str) -> None:
    """학습 행이 정확히 (특성 수 + 1) × 10개면 맞춘다 — 선형 규칙의 계수를 그대로 되찾는다."""
    frame = _linear_frame(metric, _need(metric))
    corr = fit_correction(frame, metric)
    assert isinstance(corr, Correction)
    assert corr.metric == metric
    assert corr.features == c.EXT_FEATURES[metric]
    assert corr.n_train == _need(metric)
    assert corr.coef.shape == (len(c.EXT_FEATURES[metric]) + 1,)
    np.testing.assert_allclose(corr.coef, FIT_BETA[metric], rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("metric", c.EXT_METRICS)
def test_models_extended_fit_too_few_rows_gives_zero(metric: str) -> None:
    frame = _linear_frame(metric, _need(metric) - 1)
    corr = fit_correction(frame, metric)
    assert corr.n_train == _need(metric) - 1
    assert corr.coef.shape == (len(c.EXT_FEATURES[metric]) + 1,)
    assert np.array_equal(corr.coef, np.zeros(len(c.EXT_FEATURES[metric]) + 1))


def test_models_extended_fit_fills_nan_with_training_mean_before_fitting() -> None:
    """k_bb_pct (age_c, whiff_pct), 30행. whiff는 i % 5 == 0인 6행이 결측.

    age_c = i − 10 (i = 0…29) → 평균 (−10 + 19)/2 = 4.5
    whiff = 20 + i (결측 아닌 24행, i 합 435 − 75 = 360 → 평균 i 15) → 평균 35.0
    y = 1 + 2·age_c + 0.5·whiff, 결측 행은 whiff = 35.0(채운 값)으로 계산 →
    평균으로 채운 뒤 맞추면 [1, 2, 0.5]가 정확히 나온다. 0으로 채우면 어긋나고,
    결측 행을 빼면 24행 < 30이라 계수가 0이 된다.
    """
    rows = range(30)
    whiff = [NAN if i % 5 == 0 else 20.0 + i for i in rows]
    filled = [35.0 if i % 5 == 0 else 20.0 + i for i in rows]
    frame = pd.DataFrame(
        {
            "train_target": pd.Series([2023] * 30, dtype="int64"),
            "player_id": pd.Series([100 + i for i in rows], dtype="int64"),
            "age_c": pd.Series([float(i - 10) for i in rows], dtype="float64"),
            "whiff_pct": pd.Series(whiff, dtype="float64"),
            "y": pd.Series(
                [1.0 + 2.0 * (i - 10) + 0.5 * w for i, w in zip(rows, filled, strict=True)]
            ),
        }
    )
    corr = fit_correction(frame, "k_bb_pct")
    assert corr.n_train == 30
    assert corr.fill == pytest.approx({"age_c": 4.5, "whiff_pct": 35.0})
    np.testing.assert_allclose(corr.coef, [1.0, 2.0, 0.5], rtol=1e-9, atol=1e-9)


def test_models_extended_fit_all_nan_feature_fills_zero() -> None:
    """hr (age_c, barrel_pct, ev_avg), 40행, ev_avg 전부 결측 → fill 0, 그 계수는 0(열이 모두 0).

    y = 0.02 + 0.001·age_c + 0.002·barrel (age_c = i % 9 − 3, barrel = 5 + i % 7)
    """
    rows = range(40)
    age_c = [float(i % 9 - 3) for i in rows]
    barrel = [5.0 + i % 7 for i in rows]
    frame = pd.DataFrame(
        {
            "train_target": pd.Series([2021 + i // 20 for i in rows], dtype="int64"),
            "player_id": pd.Series([i % 20 + 1 for i in rows], dtype="int64"),
            "age_c": pd.Series(age_c, dtype="float64"),
            "barrel_pct": pd.Series(barrel, dtype="float64"),
            "ev_avg": pd.Series([NAN] * 40, dtype="float64"),
            "y": pd.Series(
                [0.02 + 0.001 * a + 0.002 * b for a, b in zip(age_c, barrel, strict=True)]
            ),
        }
    )
    corr = fit_correction(frame, "hr")
    assert corr.fill["ev_avg"] == 0.0
    # age_c 평균: i % 9 − 3을 40행 — (0…8)이 4번 + (0…3)이 한 번 → (4·36 + 6)/40 − 3 = 0.75
    assert corr.fill["age_c"] == pytest.approx(0.75)
    # barrel 평균: 5 + i % 7 — (0…6)이 5번 + (0…4)이 한 번 → 5 + (5·21 + 10)/40 = 7.875
    assert corr.fill["barrel_pct"] == pytest.approx(7.875)
    np.testing.assert_allclose(corr.coef, [0.02, 0.001, 0.002, 0.0], rtol=1e-7, atol=1e-10)
    assert np.all(np.isfinite(corr.coef))


@pytest.mark.parametrize("metric", c.EXT_METRICS)
def test_models_extended_fit_empty_frame(metric: str) -> None:
    empty = pd.DataFrame({name: pd.Series(dtype="float64") for name in _residual_columns(metric)})
    corr = fit_correction(empty, metric)
    assert corr.n_train == 0
    assert np.array_equal(corr.coef, np.zeros(len(c.EXT_FEATURES[metric]) + 1))
    assert corr.fill == dict.fromkeys(c.EXT_FEATURES[metric], 0.0)


@pytest.mark.parametrize("bad", [NAN, float("inf")])
def test_models_extended_fit_rejects_non_finite_y(bad: float) -> None:
    """y에 결측·무한값이 하나라도 있으면 최소제곱이 결측 계수를 내므로 멈춘다 (조용히 NaN 예측을 내지 않는다)."""
    frame = _linear_frame("pa", _need("pa"))
    frame.loc[3, "y"] = bad
    with pytest.raises(ValueError, match="y"):
        fit_correction(frame, "pa")


def test_models_extended_fit_is_deterministic() -> None:
    frame = _linear_frame("ops", 90, seed=3)
    noisy = frame.assign(y=frame["y"] + np.sin(np.arange(len(frame))) * 0.01)
    first = fit_correction(noisy, "ops")
    second = fit_correction(noisy.copy(), "ops")
    assert np.array_equal(first.coef, second.coef)
    assert first.fill == second.fill
    assert first.n_train == second.n_train == 90


# ---------------------------------------------------------------------------
# predict_delta
# ---------------------------------------------------------------------------


def test_models_extended_predict_delta_hand_values() -> None:
    """Δ = 절편 + Σ 계수 × 특성(결측은 fill). 쓰지 않는 열(age_c2·k_pct·…)은 무시, 행 순서 그대로."""
    features = pd.DataFrame(
        {
            "player_id": pd.Series([3, 1, 2], dtype="int64"),  # 일부러 player_id 순이 아니다
            "age_c": [2.0, NAN, NAN],
            "age_c2": [4.0, 999.0, NAN],
            "barrel_pct": [10.0, NAN, NAN],
            "ev_avg": [92.0, 88.0, NAN],
            "k_pct": [25.0, 999.0, NAN],
            "xwoba_minus_woba": [0.01, 999.0, NAN],
        }
    )
    corr = Correction(
        metric="hr",
        features=("age_c", "barrel_pct", "ev_avg"),
        coef=np.array([0.01, 0.002, 0.001, -0.0005]),
        fill={"age_c": 1.0, "barrel_pct": 8.0, "ev_avg": 90.0},
        n_train=50,
    )
    delta = predict_delta(features, corr)
    assert isinstance(delta, pd.Series)
    expected = [
        0.01 + 0.002 * 2 + 0.001 * 10 - 0.0005 * 92,  # = −0.022
        0.01 + 0.002 * 1 + 0.001 * 8 - 0.0005 * 88,  # age·barrel 채움 → −0.024
        0.01 + 0.002 * 1 + 0.001 * 8 - 0.0005 * 90,  # 전부 채움 → −0.025
    ]
    _assert_close(delta.to_numpy(dtype="float64"), expected)
    _assert_close(expected, [-0.022, -0.024, -0.025])


def test_models_extended_predict_delta_pit_and_zero_coef() -> None:
    features = pd.DataFrame(
        {
            "player_id": pd.Series([7, 8], dtype="int64"),
            "age_c": [3.0, -1.0],
            "age_c2": [9.0, 1.0],
            "xera_minus_proj": [-0.4, NAN],
            "whiff_pct": [30.0, NAN],
        }
    )
    corr = Correction(
        "era",
        ("age_c", "xera_minus_proj", "whiff_pct"),
        np.array([0.1, 0.02, 0.5, -0.01]),
        {"age_c": 0.0, "xera_minus_proj": 0.2, "whiff_pct": 25.0},
        40,
    )
    # 0.1 + 0.06 − 0.2 − 0.3 = −0.34 / 0.1 − 0.02 + 0.1 − 0.25 = −0.07
    _assert_close(predict_delta(features, corr).to_numpy(dtype="float64"), [-0.34, -0.07])
    zero = corr._replace(coef=np.zeros(4))
    _assert_close(predict_delta(features, zero).to_numpy(dtype="float64"), [0.0, 0.0])


# ---------------------------------------------------------------------------
# extended_projection — 큰 세계
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", c.ROLES)
def test_models_extended_projection_no_adoption_equals_weighted(role: Role) -> None:
    out = extended_projection(_world(), role, Y_WORLD, P_WORLD, NO_ADOPTION)
    assert c.check_frame(out, PROJ_SPECS[role]) == []
    assert set(out["model"]) == {"extended"}
    weighted = _world_weighted(role)
    assert_frame_equal(
        out.drop(columns="model").reset_index(drop=True),
        weighted.drop(columns="model").reset_index(drop=True),
        check_exact=True,
    )


@pytest.mark.parametrize("role", c.ROLES)
def test_models_extended_projection_contract_players_sorted(role: Role) -> None:
    out = extended_projection(_world(), role, Y_WORLD, P_WORLD, _adopt(*c.EXT_ROLE_METRICS[role]))
    assert c.check_frame(out, PROJ_SPECS[role]) == []
    cohort_ids = sorted(int(i) for i in cohort_for(_world(), role, Y_WORLD)["player_id"])
    assert [int(i) for i in out["player_id"]] == cohort_ids
    assert set(out["model"]) == {"extended"}
    assert set(out["target_season"].astype(int)) == {Y_WORLD}
    assert set(out["role"]) == {role}
    numeric = out.select_dtypes("number")
    assert not bool(numeric.isna().any().any())


@pytest.mark.parametrize(
    ("role", "adopted"),
    [
        ("bat", ("pa",)),  # pa와 hr(= pa_e × 가중 hr/pa)이 바뀐다
        ("bat", ("hr",)),  # hr만
        ("bat", ("ops",)),  # obp·slg가 ops_e / ops_w 비율로
        ("bat", ("pa", "hr", "ops")),
        ("pit", ("ip",)),  # 보정은 아웃 단위
        ("pit", ("era", "k_bb_pct")),
        ("pit", ("ip", "era", "k_bb_pct")),
    ],
)
def test_models_extended_projection_adopted_metrics_follow_fitted_delta(
    role: Role, adopted: tuple[str, ...]
) -> None:
    deltas = {m: np.asarray(_world_delta(role, m), dtype="float64") for m in adopted}
    for metric, delta in deltas.items():
        assert bool(np.any(np.abs(delta) > 1e-9)), f"{metric} 보정이 0 — 세계가 작다"
    out = extended_projection(_world(), role, Y_WORLD, P_WORLD, _adopt(*adopted))
    _assert_projection(out, _expected_projection(role, deltas), adopted)
    for metric in adopted:
        for col in CHANGED[metric]:
            weighted = _world_weighted(role)[col].to_numpy(dtype="float64")
            assert not np.allclose(out[col].to_numpy(dtype="float64"), weighted), (metric, col)


BAT_UP = {"pa": 12.5, "hr": 0.005, "ops": 0.04}
PIT_UP = {"ip": 30.0, "era": 0.25, "k_bb_pct": 1.5}


@pytest.mark.parametrize(
    ("role", "deltas"),
    [
        ("bat", BAT_UP),
        ("bat", {"pa": 12.5}),
        ("bat", {"ops": 0.04}),
        ("pit", PIT_UP),
        ("pit", {"ip": 30.0}),
    ],
)
def test_models_extended_projection_constant_delta_formulas(
    monkeypatch: pytest.MonkeyPatch, role: Role, deltas: dict[str, float]
) -> None:
    """Δ가 상수일 때 결합 규칙을 손으로 확인한다.

    타자: pa = pa_w + 12.5, hr = (pa_w + 12.5) × (가중 hr/pa + 0.005) — pa_w = 0인 선수도
    hr = 12.5 × 가중 hr/pa (hr_w / pa_w로 되돌리면 안 된다). ops = ops_w + .04,
    obp = obp_w × (ops_w + .04) / ops_w, slg도 같은 비율.
    투수: ip = (outs_w + 30) / 3 = ip_w + 10 (Δ는 아웃 단위), era + .25, K−BB% + 1.5.
    """
    calls = _constant_fit(monkeypatch, deltas)
    out = extended_projection(_world(), role, Y_WORLD, P_WORLD, _adopt(*deltas))
    assert set(deltas) <= set(calls), "extended_projection이 fit_correction을 거치지 않았다"
    _assert_projection(out, _expected_projection(role, deltas), list(deltas))
    if role == "pit" and "ip" in deltas:
        _assert_close(out["ip"], _world_weighted("pit")["ip"] + 10.0)


def test_models_extended_projection_bat_clips_at_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """pa_e = max(pa_w − 10⁶, 0) = 0 → hr 0. ops_e = max(ops_w − 100, 0) = 0 → obp·slg 0."""
    deltas = {"pa": -1e6, "hr": 0.01, "ops": -100.0}
    _constant_fit(monkeypatch, deltas)
    out = extended_projection(_world(), "bat", Y_WORLD, P_WORLD, _adopt(*deltas))
    for col in ("pa", "hr", "obp", "slg", "ops"):
        assert np.array_equal(out[col].to_numpy(dtype="float64"), np.zeros(len(out))), col
    _assert_projection(out, _expected_projection("bat", deltas), list(deltas))


def test_models_extended_projection_hr_per_pa_clips_at_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """hr만 채택, Δhr = −1 → hr_per_pa_e = max(가중 hr/pa − 1, 0) = 0 → hr 0, pa는 그대로."""
    _constant_fit(monkeypatch, {"hr": -1.0})
    out = extended_projection(_world(), "bat", Y_WORLD, P_WORLD, _adopt("hr"))
    assert np.array_equal(out["hr"].to_numpy(dtype="float64"), np.zeros(len(out)))
    _assert_projection(out, _expected_projection("bat", {"hr": -1.0}), ["hr"])


def test_models_extended_projection_pit_clips_outs_and_era_not_k_bb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """outs_e = max(outs_w − 10⁶, 0) → ip 0, era_e = max(era_w − 100, 0) → 0.

    K−BB%는 자르지 않는다: k_bb_pct_w − 1000 그대로(음수).
    """
    deltas = {"ip": -1e6, "era": -100.0, "k_bb_pct": -1000.0}
    _constant_fit(monkeypatch, deltas)
    out = extended_projection(_world(), "pit", Y_WORLD, P_WORLD, _adopt(*deltas))
    assert np.array_equal(out["ip"].to_numpy(dtype="float64"), np.zeros(len(out)))
    assert np.array_equal(out["era"].to_numpy(dtype="float64"), np.zeros(len(out)))
    assert bool((out["k_bb_pct"] < 0).all())
    _assert_projection(out, _expected_projection("pit", deltas), list(deltas))


def test_models_extended_combine_bat_ops_ratio_and_zero_ops() -> None:
    """ops만 채택했을 때의 결합 (모듈 docstring "예측 결합"). 큰 세계에는 가중 ops가 0인 선수가 없어 직접 본다.

    행 0: ops_w .700 + .070 = .770 → 비율 1.1 → obp .300 × 1.1 = .330, slg .400 × 1.1 = .440
    행 1: ops_w 0 + .050 = .050 → obp·slg는 0 그대로 (ops_w = 0이면 비율을 쓰지 않는다)
    행 2: ops_w .500 − 1 → max(−.5, 0) = 0 → 비율 0 → obp·slg 0
    pa·hr는 채택하지 않아 바뀐 열에 없다.
    """
    out = pd.DataFrame(
        {
            "pa": [100.0, 50.0, 80.0],
            "obp": [0.300, 0.0, 0.200],
            "slg": [0.400, 0.0, 0.300],
            "ops": [0.700, 0.0, 0.500],
        }
    )
    deltas = {"ops": np.array([0.070, 0.050, -1.0])}
    changed = ext_mod._combine_bat(out, np.array([0.03, 0.0, 0.01]), deltas)
    assert sorted(changed) == ["obp", "ops", "slg"]
    _assert_close(changed["ops"], [0.770, 0.050, 0.0])
    _assert_close(changed["obp"], [0.330, 0.0, 0.0])
    _assert_close(changed["slg"], [0.440, 0.0, 0.0])


@pytest.mark.parametrize(
    "adoption",
    [
        Adoption("v1", {"pa": True}),  # 지표가 빠졌다
        Adoption("", dict.fromkeys(c.EXT_METRICS, False)),  # 버전이 비었다
        Adoption("v1", {**dict.fromkeys(c.EXT_METRICS, False), "obp": True}),  # 모르는 지표
        Adoption("v1", {**dict.fromkeys(c.EXT_METRICS, False), "pa": 1}),  # type: ignore[dict-item]
    ],
)
@pytest.mark.parametrize("role", c.ROLES)
def test_models_extended_projection_rejects_invalid_adoption(
    role: Role, adoption: Adoption
) -> None:
    with pytest.raises(ValueError):
        extended_projection(_world(), role, Y_WORLD, P_WORLD, adoption)


# ---------------------------------------------------------------------------
# corrections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "metrics"),
    [
        ("bat", {"pa": True, "hr": False, "ops": True}),
        ("pit", {"ip": False, "era": True, "k_bb_pct": False}),
        ("pit", {"ip": True, "era": True, "k_bb_pct": True}),
        ("bat", {"pa": False, "hr": False, "ops": False}),
    ],
)
def test_models_extended_corrections_true_metrics_only(
    role: Role, metrics: dict[str, bool]
) -> None:
    out = corrections(_world(), role, Y_WORLD, P_WORLD, metrics)
    assert sorted(out) == sorted(m for m, on in metrics.items() if on)
    for metric, corr in out.items():
        expected = fit_correction(_world_residual(role, metric), metric)
        assert corr.metric == metric
        assert corr.features == c.EXT_FEATURES[metric]
        assert np.array_equal(corr.coef, expected.coef)
        assert corr.fill == expected.fill
        assert corr.n_train == expected.n_train == len(_world_residual(role, metric))


@pytest.mark.parametrize(("role", "metric"), [("bat", "era"), ("pit", "ops"), ("bat", "obp")])
def test_models_extended_corrections_rejects_metric_of_other_role(role: Role, metric: str) -> None:
    with pytest.raises(ValueError):
        corrections(_tiny_tables(), role, Y_TINY, P_HAND, {metric: True})


# ---------------------------------------------------------------------------
# N2 — 목표 시즌 이후 기록·리그 평균·표시용 열이 결과에 영향을 주지 않는다
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", c.ROLES)
def test_models_extended_no_leakage(role: Role) -> None:
    clean, poisoned = _world_clean_and_poisoned()
    weighted = {
        name: weighted_projection_for(tables, role, Y_WORLD, P_WORLD)
        for name, tables in (("clean", clean), ("poisoned", poisoned))
    }
    assert_frame_equal(weighted["clean"], weighted["poisoned"], check_exact=True)  # M3 전제

    def same(a: pd.DataFrame, b: pd.DataFrame) -> None:
        assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True)

    same(
        extended_features(clean, role, Y_WORLD, P_WORLD, weighted["clean"]),
        extended_features(poisoned, role, Y_WORLD, P_WORLD, weighted["poisoned"]),
    )
    for metric in c.EXT_ROLE_METRICS[role]:
        frame_clean = residual_frame(clean, role, metric, Y_WORLD, P_WORLD)
        frame_poisoned = residual_frame(poisoned, role, metric, Y_WORLD, P_WORLD)
        same(frame_clean, frame_poisoned)
        fit_clean = fit_correction(frame_clean, metric)
        fit_poisoned = fit_correction(frame_poisoned, metric)
        assert np.array_equal(fit_clean.coef, fit_poisoned.coef), metric
        assert fit_clean.fill == fit_poisoned.fill, metric
        assert fit_clean.n_train == fit_poisoned.n_train, metric
    adoption = _adopt(*c.EXT_ROLE_METRICS[role])
    same(
        extended_projection(clean, role, Y_WORLD, P_WORLD, adoption),
        extended_projection(poisoned, role, Y_WORLD, P_WORLD, adoption),
    )
