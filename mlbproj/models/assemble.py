"""모델 결과 결합 — 대상 + 비율 + 출전량 → 예측 표 (SPEC F8, §6 projections).

비율 모델과 출전량 모델이 따로 내놓은 표를 선수 ID로 합친다. 타자 HR = 예상 PA × 예상 HR/PA,
투수 IP = 예상 아웃 / 3. 세 표의 선수·목표 시즌·역할이 정확히 같아야 한다.
"""

import pandas as pd

from mlbproj.contracts import (
    BAT_PLAYING_TIME,
    BAT_PROJECTIONS,
    BAT_RATES,
    COHORTS,
    MODELS,
    PIT_PLAYING_TIME,
    PIT_PROJECTIONS,
    PIT_RATES,
    ModelName,
    Role,
    TableSpec,
    check_frame,
)


def _check(
    cohort: pd.DataFrame, rates: pd.DataFrame, pt: pd.DataFrame, role: Role, model: str
) -> int:
    if model not in MODELS:
        raise ValueError(f"모델 이름은 {MODELS} 중 하나: {model!r}")
    specs: tuple[tuple[pd.DataFrame, TableSpec], ...] = (
        (cohort, COHORTS),
        (rates, BAT_RATES if role == "bat" else PIT_RATES),
        (pt, BAT_PLAYING_TIME if role == "bat" else PIT_PLAYING_TIME),
    )
    for df, spec in specs:
        problems = check_frame(df, spec)
        if problems:
            raise ValueError("; ".join(problems))
    if set(cohort["role"]) != {role}:
        raise ValueError(f"대상 표의 역할이 {role}이 아니다: {sorted(set(cohort['role']))}")
    targets = {int(t) for df in (cohort, rates, pt) for t in df["target_season"]}
    if len(targets) != 1:
        raise ValueError(f"목표 시즌이 하나가 아니다: {sorted(targets)}")
    ids = [set(df["player_id"]) for df in (cohort, rates, pt)]
    if not ids[0] == ids[1] == ids[2]:
        raise ValueError("대상·비율·출전량 표의 선수가 다르다")
    return targets.pop()


def _base(cohort: pd.DataFrame, model: str) -> pd.DataFrame:
    keep = [
        "player_id",
        "role",
        "target_season",
        "input_first",
        "input_last",
        "n_seasons",
        "sample",
    ]
    base = cohort[keep].copy()
    base["model"] = model
    return base


def assemble_bat(
    cohort: pd.DataFrame, rates: pd.DataFrame, playing_time: pd.DataFrame, model: ModelName
) -> pd.DataFrame:
    """타자 예측 표 (`contracts.BAT_PROJECTIONS`, player_id 순).

    Raises:
        ValueError: 표가 계약을 어기거나, 역할·목표 시즌·선수가 서로 맞지 않거나, 모델 이름이 틀릴 때.
    """
    _check(cohort, rates, playing_time, "bat", model)
    df = _base(cohort, model).rename(columns={"sample": "sample_pa"})
    df = df.merge(playing_time[["player_id", "pa"]], on="player_id").merge(
        rates[["player_id", "obp", "slg", "ops", "hr_per_pa", "lg_ops"]], on="player_id"
    )
    df["hr"] = df["pa"] * df["hr_per_pa"]
    out = df[BAT_PROJECTIONS.column_names].sort_values("player_id").reset_index(drop=True)
    return out


def assemble_pit(
    cohort: pd.DataFrame, rates: pd.DataFrame, playing_time: pd.DataFrame, model: ModelName
) -> pd.DataFrame:
    """투수 예측 표 (`contracts.PIT_PROJECTIONS`, player_id 순).

    Raises:
        ValueError: 표가 계약을 어기거나, 역할·목표 시즌·선수가 서로 맞지 않거나, 모델 이름이 틀릴 때.
    """
    _check(cohort, rates, playing_time, "pit", model)
    df = _base(cohort, model).rename(columns={"sample": "sample_bf"})
    df["sample_outs"] = cohort["sample_outs"].astype("int64").to_numpy()
    df = df.merge(playing_time[["player_id", "outs"]], on="player_id").merge(
        rates[["player_id", "era", "k_bb_pct", "lg_era", "lg_k_bb_pct"]], on="player_id"
    )
    df["ip"] = df["outs"] / 3
    out = df[PIT_PROJECTIONS.column_names].sort_values("player_id").reset_index(drop=True)
    return out
