"""과거 검증 평가 (SPEC F10·D13·N2, `contracts.EVALUATIONS`). 레인 A가 구현한다.

목표 시즌 Y·역할 하나에 대해, 여러 모델의 예측 표(`BAT_PROJECTIONS`/`PIT_PROJECTIONS`를 이어 붙인 것)를
Y의 실제 기록과 비교해 MAE 표를 만든다. 예측 표에는 반드시 model "league"와 "baseline"이 있어야 한다 —
이 둘이 기준선(baseline_mae_league·baseline_mae_prev)이다.

지표 (`METRICS`)
- 타자: ops (비율, 실제 = 정제 표 ops, 대표 집단 = 실제 pa ≥ `RATE_MIN_PA`, 가중치 pa),
  pa (출전량, 실제 = pa), hr (출전량, 실제 = hr).
- 투수: era (비율, 실제 = 27·er/outs, 대표 집단 = 실제 outs ≥ `RATE_MIN_OUTS`, 가중치 outs),
  k_bb_pct (비율, 실제 = 100·(so−bb)/bf, 대표 집단 = outs ≥ `RATE_MIN_OUTS`, 가중치 bf),
  ip (출전량, 실제 = outs/3).

집단 (D13)
- 비율 지표: mae = 대표 집단의 평균 |예측 − 실제|. mae_weighted = Y에 분모(pa·outs·bf)가 0보다 큰 대상 전원의
  가중 평균 |오차|. excluded = 그룹 대상 수 − n.
- 출전량 지표: 대상 전원, Y에 기록이 없으면 실제 0. mae_weighted는 결측, excluded = 0.
- 그룹: all, seasons_1_2(n_seasons ≤ 2), seasons_3_5(≥ 3), prev_played, prev_absent(played_prev).
  n = 그 그룹의 평가 집단 크기. n = 0이면 MAE 열은 결측.
- baseline_mae_league·baseline_mae_prev = 같은 집단에서 "league"·"baseline" 모델의 MAE.
  "league" 모델은 비율 지표 행만 낸다(출전량은 baseline과 같으므로).
"""

from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from mlbproj.contracts import (
    BAT_PROJECTIONS,
    BATTING_SEASONS,
    COHORTS,
    EVALUATIONS,
    GROUPS,
    MODELS,
    PIT_PROJECTIONS,
    PITCHING_SEASONS,
    RATE_MIN_OUTS,
    RATE_MIN_PA,
    Role,
    TableSpec,
    check_frame,
)


class Metric(NamedTuple):
    """평가 지표 정의.

    Attributes:
        name: 지표 이름 (EVALUATIONS.metric).
        role: 역할.
        kind: "rate"(비율) 또는 "volume"(출전량).
        weight: 비율 지표의 가중치 열 ("pa"·"outs"·"bf"). 출전량은 None.
    """

    name: str
    role: Role
    kind: str
    weight: str | None


METRICS: tuple[Metric, ...] = (
    Metric("ops", "bat", "rate", "pa"),
    Metric("pa", "bat", "volume", None),
    Metric("hr", "bat", "volume", None),
    Metric("era", "pit", "rate", "outs"),
    Metric("k_bb_pct", "pit", "rate", "bf"),
    Metric("ip", "pit", "volume", None),
)

_SEASON_SPECS: dict[Role, TableSpec] = {"bat": BATTING_SEASONS, "pit": PITCHING_SEASONS}
_PROJECTION_SPECS: dict[Role, TableSpec] = {"bat": BAT_PROJECTIONS, "pit": PIT_PROJECTIONS}
_LEAGUE_MODEL = "league"  # baseline_mae_league
_PREV_MODEL = "baseline"  # baseline_mae_prev
# 비율 지표 대표 집단: Y 실제 출전량 열 ≥ 기준 (D13). 투수는 era·k_bb_pct 모두 아웃 기준이다
_RATE_POPULATION: dict[Role, tuple[str, int]] = {
    "bat": ("pa", RATE_MIN_PA),
    "pit": ("outs", RATE_MIN_OUTS),
}
_SHORT_CAREER_MAX = 2  # seasons_1_2 그룹의 n_seasons 상한
_OUTS_PER_GAME = 27  # ERA = 9이닝(27아웃)당 자책점
_OUTS_PER_INNING = 3
_SHOW_MAX = 5  # 오류 메시지에 적는 선수 ID의 최대 개수


def _require(df: pd.DataFrame, spec: TableSpec, what: str) -> None:
    """표가 계약을 지키는지 확인한다. 어기면 ValueError (메시지 앞에 what을 붙인다)."""
    problems = check_frame(df, spec)
    if problems:
        raise ValueError(f"{what} 계약 위반: " + "; ".join(problems))


def _show(ids: Iterable[int]) -> list[int]:
    """오류 메시지에 적을 선수 ID (정렬, 최대 `_SHOW_MAX`개)."""
    return sorted(int(i) for i in ids)[:_SHOW_MAX]


def _role_metrics(role: Role) -> list[Metric]:
    """그 역할의 지표 (METRICS 순)."""
    return [m for m in METRICS if m.role == role]


def _actual_columns(role: Role) -> list[str]:
    """`actual_values`의 열: player_id, 지표(METRICS 순), 아직 없는 분모 열(가중치 순)."""
    names = [m.name for m in _role_metrics(role)]
    for m in _role_metrics(role):
        if m.weight is not None and m.weight not in names:
            names.append(m.weight)
    return ["player_id", *names]


def actual_values(seasons: pd.DataFrame, role: Role, target_season: int) -> pd.DataFrame:
    """목표 시즌의 실제 값. 열: player_id, 그 역할 지표들(METRICS 순), 분모 열(타자 pa / 투수 outs·bf).

    분모가 0이면 그 비율은 결측. Y에 행이 없는 선수는 결과에 없다.
    """
    _require(seasons, _SEASON_SPECS[role], f"{role} 정제 표")
    rows = seasons[seasons["season"] == target_season].sort_values("player_id", kind="mergesort")
    # 선수당 한 행이어야 대상 표와 맞출 수 있다 (v0.1은 MLB만, D1)
    repeated = rows.loc[rows["player_id"].duplicated(), "player_id"]
    if not repeated.empty:
        raise ValueError(
            f"{target_season} 시즌에 행이 둘 이상인 선수가 있다 (리그 여럿?): {_show(set(repeated))}"
        )
    values: dict[str, pd.Series]
    if role == "bat":
        # 실제 ops는 정제 표 값 그대로 (타석 0이면 결측)
        values = {"ops": rows["ops"].where(rows["pa"] > 0), "pa": rows["pa"], "hr": rows["hr"]}
    else:
        outs = rows["outs"].where(rows["outs"] > 0)
        bf = rows["bf"].where(rows["bf"] > 0)
        values = {
            "era": _OUTS_PER_GAME * rows["er"] / outs,
            "k_bb_pct": 100 * (rows["so"] - rows["bb"]) / bf,
            "ip": rows["outs"] / _OUTS_PER_INNING,
            "outs": rows["outs"],
            "bf": rows["bf"],
        }
    out = pd.DataFrame({"player_id": rows["player_id"], **values})
    return out[_actual_columns(role)].reset_index(drop=True)


def _check_inputs(projections: pd.DataFrame, cohort: pd.DataFrame, role: Role) -> int:
    """예측 표·대상 표를 확인하고 목표 시즌 Y를 돌려준다. 어기면 ValueError."""
    _require(projections, _PROJECTION_SPECS[role], f"{role} 예측 표")
    _require(cohort, COHORTS, "대상 표")
    for what, df in (("예측 표", projections), ("대상 표", cohort)):
        roles = sorted(set(df["role"]))
        if roles != [role]:
            raise ValueError(f"{what}의 역할이 {role} 하나가 아니다: {roles}")
    targets = sorted({int(t) for t in cohort["target_season"]})
    if len(targets) != 1:
        raise ValueError(f"대상 표의 목표 시즌이 하나가 아니다: {targets}")
    projection_targets = sorted({int(t) for t in projections["target_season"]})
    if projection_targets != targets:
        raise ValueError(f"예측 표의 목표 시즌 {projection_targets}이 대상 표 {targets}와 다르다")
    models = set(projections["model"])
    unknown = sorted(models - set(MODELS))
    if unknown:
        raise ValueError(f"모델 이름은 {MODELS} 중 하나: {unknown}")
    missing = [m for m in (_LEAGUE_MODEL, _PREV_MODEL) if m not in models]
    if missing:
        raise ValueError(f"기준선 모델이 예측 표에 없다: {missing}")
    players = {int(i) for i in cohort["player_id"]}
    for model, rows in projections.groupby("model", sort=True):
        predicted = {int(i) for i in rows["player_id"]}
        if predicted != players:
            raise ValueError(
                f"{model} 모델 예측의 선수가 대상 표와 다르다 — "
                f"예측 없음 {_show(players - predicted)}, 대상 아님 {_show(predicted - players)}"
            )
    return targets[0]


def _group_members(cohort: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """대상(행, cohort 순) × 그룹(열, GROUPS 순) 소속 (D13·PLAN R8)."""
    short = (cohort["n_seasons"] <= _SHORT_CAREER_MAX).to_numpy(dtype=bool)
    played = cohort["played_prev"].to_numpy(dtype=bool)
    rules = {
        "all": np.ones(len(cohort), dtype=bool),
        "seasons_1_2": short,
        "seasons_3_5": ~short,  # n_seasons ≥ 3
        "prev_played": played,
        "prev_absent": ~played,
    }
    return np.column_stack([rules[g] for g in GROUPS])


def _group_mean(
    err: npt.NDArray[np.float64],
    weight: npt.NDArray[np.float64],
    in_population: npt.NDArray[np.bool_],
    members: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """그룹마다 (평가 집단 ∩ 그룹)의 인원과 가중 평균 |오차|. 가중치 합이 0이면 결측."""
    mask = members & in_population[:, None]
    total = np.where(mask, (err * weight)[:, None], 0.0).sum(axis=0)
    weight_sum = np.where(mask, weight[:, None], 0.0).sum(axis=0)
    mean = np.divide(total, weight_sum, out=np.full(weight_sum.shape, np.nan), where=weight_sum > 0)
    return mask.sum(axis=0).astype(np.int64), mean


def _score(
    metric: Metric,
    pred: pd.Series,
    actual: pd.DataFrame,
    members: npt.NDArray[np.bool_],
    role: Role,
) -> pd.DataFrame:
    """한 모델·한 지표의 그룹별 n·mae·mae_weighted·excluded (GROUPS 순).

    pred·actual은 같은 대상 순서다. actual에서 Y 미출전 대상은 결측이다.
    """
    ones = np.ones(len(pred))
    if metric.kind == "rate" and metric.weight is not None:
        # 실제 값이 결측(분모 0·미출전)이면 오차도 결측 — 어느 집단에도 들지 않는다
        err = (pred - actual[metric.name]).abs().to_numpy(dtype=float)
        known = ~np.isnan(err)
        sample_column, threshold = _RATE_POPULATION[role]
        representative = known & (actual[sample_column] >= threshold).to_numpy(dtype=bool)
        weight = actual[metric.weight].fillna(0.0).to_numpy(dtype=float)
        n, mae = _group_mean(err, ones, representative, members)
        _, mae_weighted = _group_mean(err, weight, known & (weight > 0), members)
        mae_weighted = np.where(n > 0, mae_weighted, np.nan)  # n = 0이면 MAE 열은 결측
        excluded = members.sum(axis=0) - n
    else:
        # 출전량: 대상 전원, Y 미출전은 실제 0
        err = (pred - actual[metric.name].fillna(0.0)).abs().to_numpy(dtype=float)
        n, mae = _group_mean(err, ones, np.ones(len(err), dtype=bool), members)
        mae_weighted = np.full(len(GROUPS), np.nan)
        excluded = np.zeros(len(GROUPS), dtype=np.int64)
    return pd.DataFrame(
        {
            "metric": metric.name,
            "group": list(GROUPS),
            "n": n,
            "mae": mae,
            "mae_weighted": mae_weighted,
            "excluded": excluded.astype(np.int64),
        }
    )


def _baseline_column(scores: pd.DataFrame, model: str, column: str) -> pd.DataFrame:
    """기준선 모델의 (metric, group)별 mae를 column 이름으로."""
    rows = scores.loc[scores["model"] == model, ["metric", "group", "mae"]]
    return rows.rename(columns={"mae": column})


def evaluate(
    projections: pd.DataFrame, seasons: pd.DataFrame, cohort: pd.DataFrame, role: Role
) -> pd.DataFrame:
    """평가 표 (`contracts.EVALUATIONS`, `check_frame` 통과, 키 순).

    Args:
        projections: 한 역할·한 목표 시즌의 예측 표들(모델 여러 개, "league"·"baseline" 포함).
        seasons: 같은 역할 정제 표 (목표 시즌 행이 실제 값이다).
        cohort: 같은 목표 시즌·역할의 대상 표 (그룹 판정).
        role: 역할.

    Raises:
        ValueError: 표가 계약을 어기거나, 목표 시즌·역할이 섞였거나, 기준선 모델이 없거나,
            예측 표의 선수가 대상 표와 다를 때.
    """
    target = _check_inputs(projections, cohort, role)
    info = cohort.set_index("player_id").sort_index()
    members = _group_members(info)
    # 실제 값은 Y 행만 (N2 — 다른 시즌 행은 보지 않는다). 대상이 아닌 선수(신인)는 빠진다
    actual = actual_values(seasons, role, target).set_index("player_id").reindex(info.index)

    frames: list[pd.DataFrame] = []
    for model in sorted(set(projections["model"])):
        pred = projections[projections["model"] == model].set_index("player_id").reindex(info.index)
        for metric in _role_metrics(role):
            if metric.kind != "rate" and model == _LEAGUE_MODEL:
                continue  # 리그 평균 모델은 비율 지표만 (출전량은 baseline과 같다)
            frames.append(
                _score(metric, pred[metric.name], actual, members, role).assign(model=model)
            )
    scores = pd.concat(frames, ignore_index=True)

    # 기준선: 같은 (지표, 그룹)의 league·baseline 모델 MAE. 출전량에는 league 행이 없어 결측
    keys = ["metric", "group"]
    out = scores.merge(
        _baseline_column(scores, _LEAGUE_MODEL, "baseline_mae_league"),
        on=keys,
        how="left",
        validate="many_to_one",
    ).merge(
        _baseline_column(scores, _PREV_MODEL, "baseline_mae_prev"),
        on=keys,
        how="left",
        validate="many_to_one",
    )
    out["target_season"] = target
    out["role"] = role
    out = out[EVALUATIONS.column_names]
    return out.sort_values(list(EVALUATIONS.keys), kind="mergesort").reset_index(drop=True)
