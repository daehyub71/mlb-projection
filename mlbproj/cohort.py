"""예측 대상과 사용 기간 (SPEC F4·F5·F12·D3·N2). 레인 C가 구현한다.

입력은 정제 표(`contracts.BATTING_SEASONS` / `PITCHING_SEASONS` 형식의 DataFrame)뿐이다.
원본 디렉토리를 읽지 않는다. **목표 시즌 Y 이후(season ≥ Y)의 행은 결과에 영향을 주면 안 된다**(N2) —
그런 행을 섞어 넣어도 결과가 같아야 한다.
"""

from collections.abc import Mapping

import pandas as pd

from mlbproj.contracts import (
    BATTING_SEASONS,
    COHORTS,
    PITCHING_SEASONS,
    SHORT_SEASON,
    WINDOW,
    Role,
    TableSpec,
    check_frame,
)

# 기간 안 미출전 시즌에 0으로 채우는 출전량 열 (F5). 나머지 값 열은 결측으로 둔다.
PLAYING_TIME_COLUMNS: dict[Role, tuple[str, ...]] = {
    "bat": ("pa", "pa_adj"),
    "pit": ("g", "gs", "outs", "outs_adj", "bf"),
}

# 역할별 정제 표 계약
_SPECS: dict[Role, TableSpec] = {"bat": BATTING_SEASONS, "pit": PITCHING_SEASONS}
# sample로 더하는 열: 타석(타자), 상대 타자(투수)
_SAMPLE_COLUMN: dict[Role, str] = {"bat": "pa", "pit": "bf"}
# 펼친 결과의 앞쪽 열
_EXPAND_LEAD: tuple[str, ...] = ("player_id", "role", "target_season", "season", "played")
# 오류 메시지에 적는 선수 ID의 최대 개수
_SHOW_MAX = 5


def _check_seasons(seasons: pd.DataFrame, role: Role) -> TableSpec:
    """정제 표가 그 역할의 계약을 지키는지 확인하고 계약을 돌려준다. 어기면 ValueError."""
    spec = _SPECS[role]
    problems = check_frame(seasons, spec)
    if problems:
        raise ValueError(f"{role} 정제 표가 계약을 어긴다: " + "; ".join(problems))
    return spec


def _check_cohort(cohort: pd.DataFrame, role: Role) -> None:
    """펼칠 대상 표를 확인한다. 어기면 ValueError.

    `COHORTS` 계약, 역할 일치, 목표 시즌 하나, 기간이 Y−WINDOW ~ Y−1 안인지 본다.
    `expand_window`는 대상 표의 기간을 그대로 펼치므로 Y 이후 시즌이 섞이는 것을 여기서 막는다(N2).
    """
    problems = check_frame(cohort, COHORTS)
    if problems:
        raise ValueError("대상 표가 계약을 어긴다: " + "; ".join(problems))
    other_roles = sorted(set(cohort["role"]) - {role})
    if other_roles:
        raise ValueError(f"대상 표에 {role}이 아닌 역할이 있다: {other_roles}")
    targets = sorted({int(t) for t in cohort["target_season"]})
    if len(targets) > 1:
        raise ValueError(f"대상 표에 목표 시즌이 여러 개다: {targets}")
    target = cohort["target_season"]
    outside = (
        (cohort["window_last"] != target - 1)
        | (cohort["window_first"] < target - WINDOW)
        | (cohort["window_first"] > cohort["window_last"])
    )
    if bool(outside.any()):
        ids = sorted({int(i) for i in cohort.loc[outside, "player_id"]})
        raise ValueError(
            f"기간이 목표 시즌의 Y−{WINDOW} ~ Y−1을 벗어난 대상이 있다: {ids[:_SHOW_MAX]}"
        )


def build_cohort(
    seasons: pd.DataFrame,
    role: Role,
    target_season: int,
    debut_seasons: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """목표 시즌 Y·역할 r의 예측 대상과 사용 기간 (`contracts.COHORTS`, `check_frame` 통과, player_id 순).

    - 대상(F4·D3): Y−`WINDOW` ~ Y−1 안에 그 역할의 행이 하나라도 있는 선수 전원. 은퇴·장기 결장도 빼지 않는다.
    - window_first = max(Y−WINDOW, 첫 MLB 기록 시즌), window_last = Y−1. 첫 MLB 기록 시즌은 이 역할의
      첫 행 시즌이고, `debut_seasons`에 그 선수의 MLB 데뷔 연도가 있으면 둘 중 이른 쪽이다 — 정제 표는
      2019부터라 그 전 기록은 데뷔 연도로 보완한다(F5, E21). 데뷔 연도는 대상 선정(F4)에 쓰지 않는다.
      n_window = window_last − window_first + 1 (미출전 시즌 포함, F5: 빈 시즌을 더 오래된 기록으로 채우지 않는다).
    - input_first·input_last = 기간 안 첫·마지막 행 시즌, n_seasons = 기간 안 행이 있는 시즌 수.
    - played_prev = Y−1에 행이 있는지.
    - sample = 기간 안 pa 합(타자) 또는 bf 합(투수). sample_outs = 기간 안 outs 합(투수), 타자는 결측.
    - 첫 행 시즌은 Y 이전 행만으로 정한다(N2).

    Args:
        seasons: 그 역할의 정제 표 (`check_frame`으로 확인한다).
        role: "bat" 또는 "pit". seasons 형식과 맞아야 한다.
        target_season: 목표 시즌 Y.
        debut_seasons: 선수 ID → MLB 데뷔 연도(선수 정보 `mlbDebutDate`). 없으면 첫 행 시즌만 쓴다.

    Raises:
        ValueError: seasons가 role의 계약을 어길 때.
    """
    _check_seasons(seasons, role)
    window_start = target_season - WINDOW
    window_last = target_season - 1
    # N2: 목표 시즌 이후 행은 첫 기록 판정에도 쓰지 않는다
    past = seasons[seasons["season"] < target_season]
    first_season = past.groupby("player_id")["season"].min()
    if debut_seasons:
        # 데뷔 연도가 첫 행보다 이르면 그쪽을 첫 기록으로 본다 (늦게 적힌 값은 무시: min)
        debut = pd.Series(debut_seasons, dtype="int64").reindex(first_season.index)
        first_season = pd.concat([first_season, debut], axis=1).min(axis=1).astype("int64")
    in_window = past[past["season"] >= window_start].groupby("player_id")

    out = pd.DataFrame(
        {
            "input_first": in_window["season"].min(),
            "input_last": in_window["season"].max(),
            "n_seasons": in_window["season"].nunique(),
            "sample": in_window[_SAMPLE_COLUMN[role]].sum(),
        }
    )
    # 기간은 그 역할의 첫 기록부터 (F5) — 더 오래된 기록으로 5시즌을 채우지 않는다
    out["window_first"] = first_season.reindex(out.index).clip(lower=window_start)
    out["window_last"] = window_last
    out["n_window"] = out["window_last"] - out["window_first"] + 1
    out["played_prev"] = out["input_last"] == window_last
    if role == "pit":
        out["sample_outs"] = in_window["outs"].sum().astype("Int64")
    else:
        out["sample_outs"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["role"] = role
    out["target_season"] = target_season
    return out.reset_index()[COHORTS.column_names]


def expand_window(cohort: pd.DataFrame, seasons: pd.DataFrame, role: Role) -> pd.DataFrame:
    """대상마다 기간 안의 시즌을 한 행씩 펼친다 (F5 — 모델 입력).

    결과 열: player_id, role, target_season, season, played(bool), 그다음 seasons의 값 열
    (player_id·season·league를 뺀 나머지, 계약 순서). 행은 (player_id, season) 순이고,
    대상마다 window_first ~ window_last의 모든 시즌이 정확히 한 번씩 있다.
    played = 그 시즌에 행이 있는지. 미출전 시즌은 `PLAYING_TIME_COLUMNS[role]`을 0으로,
    나머지 값 열(비율·Statcast·age·team 등)을 결측으로 둔다. 정수 열은 결측을 담을 수 있게 `Int64`로,
    is_short_season은 행이 없어도 season == SHORT_SEASON으로 채운다.

    Args:
        cohort: `build_cohort` 결과 (한 목표 시즌·역할).
        seasons: 같은 역할의 정제 표.
        role: 역할.

    Raises:
        ValueError: seasons가 role의 계약을 어기거나, cohort가 `COHORTS` 계약·역할·목표 시즌 하나·
            기간(Y−WINDOW ~ Y−1)을 어길 때 (N2).
    """
    _check_cohort(cohort, role)
    spec = _check_seasons(seasons, role)
    # 키 열(player_id·season·league)을 뺀 값 열, 계약 순서
    value_columns = [col for col in spec.columns if col.name not in spec.keys]

    # 대상마다 window_first ~ window_last 시즌을 한 행씩 (기간 밖·목표 시즌 이후는 만들지 않는다, N2)
    grid = cohort[["player_id", "role", "target_season"]].copy()
    grid["season"] = [
        list(range(first, last + 1))
        for first, last in zip(cohort["window_first"], cohort["window_last"], strict=True)
    ]
    grid = grid.explode("season")
    grid["season"] = grid["season"].astype("int64")

    # 결측을 담을 수 있게 정수는 Int64, 참·거짓은 boolean으로 바꾼 뒤 붙인다
    nullable = {
        col.name: "Int64" if col.kind == "int" else "boolean"
        for col in value_columns
        if col.kind in ("int", "bool")
    }
    values = seasons[["player_id", "season", *(col.name for col in value_columns)]].astype(nullable)
    out = grid.merge(
        values, on=["player_id", "season"], how="left", validate="one_to_one", indicator=True
    )
    out["played"] = out.pop("_merge") == "both"

    for name in PLAYING_TIME_COLUMNS[role]:
        out[name] = out[name].fillna(0)
    out["is_short_season"] = (
        out["is_short_season"].fillna(out["season"] == SHORT_SEASON).astype("bool")
    )

    out = out[[*_EXPAND_LEAD, *(col.name for col in value_columns)]]
    return out.sort_values(["player_id", "season"], kind="mergesort").reset_index(drop=True)
