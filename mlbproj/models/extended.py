"""확장 모델 (SPEC F9·D10·D11) — 가중 모델 위의 지표별 잔차 보정. 레인 A가 구현한다.

목표 시즌 Y, 역할 r의 확장 예측 = 가중 모델 예측 + 지표별 선형 보정 Δ. Δ는 채택표(`Adoption`)에서
True인 지표에만 더한다. 모든 지표가 False면 확장 예측의 값은 가중 예측과 같다(model 이름만 "extended").

가중 예측 (목표 시즌 하나)
- 대상: `tables.cohort_for(tables, r, Y)`.
- 비율: `weighted_rates(cohort, tables.seasons[r], tables.league[r], r, params)`.
- 출전량: `predict_playing_time(cohort, tables.seasons[r], r, tables.debuts)`.
- 예측 표: `assemble_bat`/`assemble_pit`. 아래에서 "가중 pa·hr_per_pa·ops·outs·era·k_bb_pct"는 이 값이다.

특성 (`contracts.EXT_FEATURES`, 목표 Y의 대상 기준, Y−1 이하 기록만 — N2)
- age: Y 시즌 나이 = `age.season_age(players.birth_date, Y)`. 생년월일이 없으면(또는 players에 없으면)
  기간 안 마지막 행의 정제 표 `age` + (Y − 그 시즌). 그것도 없으면 결측.
- age_c = age − `AGE_CENTER`, age_c2 = age_c².
- 기간 평균 x̄: 대상 기간(window_first ~ window_last) 안의 그 역할 행만 쓴다. 가중치는
  params.<r>.weights[Y − 1 − season] × 분모(타자 pa, 투수 bf)이고, x가 결측인 행은 뺀다. 쓸 행이 없으면 결측.
  - 타자: k_pct = 100 · Σ(w·so) / Σ(w·pa) (w는 시즌 가중치만), barrel_pct = x̄(barrel_pct), ev_avg = x̄(ev_avg),
    xwoba_minus_woba = x̄(xwoba) − x̄(woba) — 두 값이 모두 있는 행만으로 각각 계산한다.
  - 투수: whiff_pct = x̄(whiff_pct), xera_minus_proj = x̄(xera) − 가중 era(Y 예측).
- 결측 특성은 학습 행의 그 특성 평균(`Correction.fill`)으로 채운다. 학습 행에도 값이 없으면 0.

보정 대상 y (학습 목표 T의 실제값 − T의 가중 예측; 실제값은 `evaluate.actual_values`)
- "pa": 실제 pa(T에 행이 없으면 0) − 가중 pa. 학습 행: T 대상 전원.
- "hr": 실제 hr / 실제 pa − 가중 hr_per_pa. 학습 행: 실제 pa ≥ `RATE_MIN_PA`.
- "ops": 실제 ops − 가중 ops. 학습 행: 실제 pa ≥ `RATE_MIN_PA`.
- "ip": 실제 outs(없으면 0) − 가중 outs (단위는 아웃). 학습 행: T 대상 전원.
- "era": 실제 era − 가중 era. 학습 행: 실제 outs ≥ `RATE_MIN_OUTS`.
- "k_bb_pct": 실제 k_bb_pct − 가중 k_bb_pct. 학습 행: 실제 outs ≥ `RATE_MIN_OUTS`.

학습 목표 T = `EXT_TRAIN_FIRST`(pa·ip는 `EXT_TRAIN_FIRST_VOLUME`) … Y−1. T마다 대상·가중 예측·특성을 위
규칙 그대로(Y 자리에 T) 만든다. T의 입력은 T−1 이하라 학습 전체가 Y−1 이하 기록만 쓴다(N2).
계수: 절편 + 특성(`EXT_FEATURES[metric]` 순), 최소제곱(`numpy.linalg.lstsq`). 학습 행은
(train_target, player_id) 순. 학습 행이 (특성 수 + 1) × `EXT_MIN_ROWS_PER_COEF`보다 적으면 계수는 모두 0.

예측 결합 (채택 지표만 Δ를 더한다. 채택하지 않은 지표는 가중 값 그대로)
- 타자: pa_e = max(pa_w + Δpa, 0), hr_per_pa_e = max(hr_per_pa_w + Δhr, 0), hr = pa_e × hr_per_pa_e.
  ops_e = max(ops_w + Δops, 0)이고 obp·slg는 ops_e / ops_w 비율로 같이 바꾼다(ops_w = 0이면 그대로).
- 투수: outs_e = max(outs_w + Δip, 0), ip = outs_e / 3. era_e = max(era_w + Δera, 0), k_bb_pct_e = k_bb_pct_w + Δ.
- 결과: `assemble_bat`/`assemble_pit`(model "extended")와 같은 계약(`BAT_PROJECTIONS`·`PIT_PROJECTIONS`).
"""

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from mlbproj.age import season_age
from mlbproj.contracts import (
    AGE_CENTER,
    BAT_PROJECTIONS,
    BATTING_SEASONS,
    EXT_FEATURES,
    EXT_MIN_ROWS_PER_COEF,
    EXT_ROLE_METRICS,
    EXT_TRAIN_FIRST,
    EXT_TRAIN_FIRST_VOLUME,
    PIT_PROJECTIONS,
    PITCHING_SEASONS,
    RATE_MIN_OUTS,
    RATE_MIN_PA,
    WINDOW,
    Role,
    TableSpec,
    check_frame,
)
from mlbproj.evaluate import actual_values
from mlbproj.models.assemble import assemble_bat, assemble_pit
from mlbproj.models.params import Adoption, Params, validate_adoption
from mlbproj.models.playing_time import predict_playing_time
from mlbproj.models.weighted import weighted_rates
from mlbproj.tables import Tables, cohort_for

# 역할별 특성 열 (EXT_ROLE_METRICS 순서로 처음 나온 차례). extended_features의 열 순서다.
ROLE_FEATURES: dict[Role, tuple[str, ...]] = {
    "bat": ("age_c", "age_c2", "barrel_pct", "ev_avg", "k_pct", "xwoba_minus_woba"),
    "pit": ("age_c", "age_c2", "xera_minus_proj", "whiff_pct"),
}

# 출전량 지표 → 실제·예측 출전량 열 (타자 pa, 투수 아웃). 학습 목표는 EXT_TRAIN_FIRST_VOLUME부터다
_VOLUME_COLUMN: dict[str, str] = {"pa": "pa", "ip": "outs"}
# 비율 지표 학습 행: T 실제 출전량 열 ≥ 기준 (evaluate의 대표 집단과 같다)
_RATE_POPULATION: dict[Role, tuple[str, int]] = {
    "bat": ("pa", RATE_MIN_PA),
    "pit": ("outs", RATE_MIN_OUTS),
}
# 비율 지표 y의 가중 예측 열 (가중 비율 표). hr은 타석당 홈런이다
_RATE_PREDICTION: dict[str, str] = {
    "hr": "hr_per_pa",
    "ops": "ops",
    "era": "era",
    "k_bb_pct": "k_bb_pct",
}
# 기간 평균 x̄의 행 가중치 = 시즌 가중치 × 이 열
_MEAN_DENOMINATOR: dict[Role, str] = {"bat": "pa", "pit": "bf"}
# 기간 평균: 평균 이름 → (값 열, 값이 모두 있어야 하는 열). 결측 행은 분자·분모에서 뺀다
_WINDOW_MEANS: dict[Role, dict[str, tuple[str, tuple[str, ...]]]] = {
    "bat": {
        "barrel_pct": ("barrel_pct", ("barrel_pct",)),
        "ev_avg": ("ev_avg", ("ev_avg",)),
        # xwoba_minus_woba는 두 값이 모두 있는 행만으로 각각 평균한다
        "xwoba": ("xwoba", ("xwoba", "woba")),
        "woba": ("woba", ("xwoba", "woba")),
    },
    "pit": {
        "whiff_pct": ("whiff_pct", ("whiff_pct",)),
        "xera": ("xera", ("xera",)),
    },
}
# 정제 표 키 — 더하기 전에 이 순서로 정렬해 입력 행 순서와 무관하게 같은 비트를 낸다
_SEASON_KEYS: dict[Role, list[str]] = {
    "bat": list(BATTING_SEASONS.keys),
    "pit": list(PITCHING_SEASONS.keys),
}
_OUTS_PER_INNING = 3
# extended_features에 넘기는 가중 예측 표의 계약
_PROJECTION_SPECS: dict[Role, TableSpec] = {"bat": BAT_PROJECTIONS, "pit": PIT_PROJECTIONS}
_SHOW_MAX = 5  # 오류 메시지에 적는 선수 ID의 최대 개수


def _metric_features(metric: str) -> tuple[str, ...]:
    """지표의 보정 특성 (`EXT_FEATURES`). 모르는 지표면 ValueError."""
    if metric not in EXT_FEATURES:
        raise ValueError(f"지표는 {sorted(EXT_FEATURES)} 중 하나: {metric!r}")
    return EXT_FEATURES[metric]


def _ratio(num: npt.NDArray[np.float64], den: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """num / den. 분모가 0 이하인 곳(쓸 행이 없음)은 결측."""
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _floats(series: pd.Series) -> npt.NDArray[np.float64]:
    """열을 실수 배열로 (결측·pd.NA는 NaN)."""
    return series.to_numpy(dtype="float64", na_value=np.nan)


def _window_rows(
    seasons: pd.DataFrame, cohort: pd.DataFrame, role: Role, target: int, params: Params
) -> pd.DataFrame:
    """대상마다 기간 안 행 (키 순)과 시즌 가중치 열 `_w`.

    시즌 Y−k의 가중치는 params.<role>.weights[k−1]이다. Y−WINDOW ~ Y−1 밖(Y 이후 포함, N2)이거나
    대상의 기간(window_first ~ window_last) 밖인 행은 쓰지 않는다.
    """
    weights = (params.bat if role == "bat" else params.pit).weights
    weight_of = {target - k: w for k, w in zip(range(1, WINDOW + 1), weights, strict=True)}
    window = cohort[["player_id", "window_first", "window_last"]]
    rows = seasons.merge(window, on="player_id", how="inner", validate="many_to_one")
    keep = rows["season"].isin(list(weight_of)) & rows["season"].between(
        rows["window_first"], rows["window_last"]
    )
    rows = rows[keep].sort_values(_SEASON_KEYS[role], kind="mergesort")
    return rows.assign(_w=rows["season"].map(weight_of).astype("float64"))


def _window_sums(rows: pd.DataFrame, role: Role, ids: pd.Index) -> pd.DataFrame:
    """대상마다 기간 평균의 분자(`<이름>_num`)·분모(`<이름>_den`) 합 (index = ids, 행이 없으면 0)."""
    season_w = _floats(rows["_w"])
    weight = season_w * _floats(rows[_MEAN_DENOMINATOR[role]])
    parts: dict[str, npt.NDArray[np.float64]] = {}
    if role == "bat":
        # k_pct의 가중치는 시즌 가중치만이다: 100 · Σ(w·so) / Σ(w·pa)
        parts["k_pct_num"] = season_w * _floats(rows["so"])
        parts["k_pct_den"] = season_w * _floats(rows["pa"])
    for name, (column, need) in _WINDOW_MEANS[role].items():
        present = rows[list(need)].notna().all(axis=1).to_numpy(dtype=bool)
        parts[f"{name}_num"] = np.where(present, weight * _floats(rows[column]), 0.0)
        parts[f"{name}_den"] = np.where(present, weight, 0.0)
    sums = pd.DataFrame(parts).groupby(rows["player_id"].to_numpy()).sum()
    return sums.reindex(ids, fill_value=0.0)


def _birth_age(birth_date: object, target: int) -> float:
    """생년월일로 Y 시즌 나이. 생년월일이 없으면(NaN·None) 결측."""
    age = season_age(birth_date, target) if isinstance(birth_date, str) else None
    return np.nan if age is None else float(age)


def _season_ages(
    players: pd.DataFrame, rows: pd.DataFrame, ids: pd.Index, target: int
) -> npt.NDArray[np.float64]:
    """대상마다 Y 시즌 나이.

    생년월일(players)이 먼저다. 없으면(또는 players에 없으면) 기간 안 마지막 행의 정제 표 age +
    (Y − 그 시즌), 그것도 없으면 결측.
    """
    # rows는 키 순이라 선수마다 마지막 행이 기간 안 마지막 시즌이다 (결측 age를 건너뛰지 않는다)
    last = rows.drop_duplicates("player_id", keep="last").set_index("player_id").reindex(ids)
    table_age = _floats(last["age"]) + (target - _floats(last["season"]))
    if "birth_date" in players.columns:
        births = players.set_index("player_id")["birth_date"].reindex(ids)
        birth_age = np.array([_birth_age(b, target) for b in births], dtype="float64")
    else:
        birth_age = np.full(len(ids), np.nan)
    return np.where(np.isnan(birth_age), table_age, birth_age)


def _features(
    tables: Tables,
    role: Role,
    target: int,
    params: Params,
    cohort: pd.DataFrame,
    era: pd.Series | None,
) -> pd.DataFrame:
    """대상 표(cohort) 순서의 특성 표 (모듈 docstring "특성"). era는 투수 가중 era (player_id 색인)."""
    ids = pd.Index(cohort["player_id"].astype("int64"), name="player_id")
    rows = _window_rows(tables.seasons[role], cohort, role, target, params)
    sums = _window_sums(rows, role, ids)

    def mean(name: str) -> npt.NDArray[np.float64]:
        return _ratio(_floats(sums[f"{name}_num"]), _floats(sums[f"{name}_den"]))

    age_c = _season_ages(tables.players, rows, ids, target) - AGE_CENTER
    values: dict[str, npt.NDArray[np.float64]] = {"age_c": age_c, "age_c2": age_c * age_c}
    if role == "bat":
        values["k_pct"] = 100 * mean("k_pct")
        values["barrel_pct"] = mean("barrel_pct")
        values["ev_avg"] = mean("ev_avg")
        values["xwoba_minus_woba"] = mean("xwoba") - mean("woba")
    else:
        if era is None:
            raise ValueError("투수 특성에는 가중 era가 필요하다")
        values["whiff_pct"] = mean("whiff_pct")
        values["xera_minus_proj"] = mean("xera") - _floats(era.reindex(ids))
    return pd.DataFrame(
        {"player_id": ids.to_numpy(), **{name: values[name] for name in ROLE_FEATURES[role]}}
    )


def _check_weighted(weighted: pd.DataFrame, cohort: pd.DataFrame, role: Role, target: int) -> None:
    """extended_features의 가중 예측 표가 같은 역할·목표·대상의 "weighted" 표인지 확인한다.

    다른 목표·모델의 표나 선수가 빠진 표를 받으면 xera_minus_proj가 조용히 결측(→ 학습 평균으로 채움)이나
    엉뚱한 값이 되므로 ValueError로 멈춘다.
    """
    problems = check_frame(weighted, _PROJECTION_SPECS[role])
    if problems:
        raise ValueError("가중 예측 표가 계약을 어긴다: " + "; ".join(problems))
    roles = sorted(set(weighted["role"]))
    targets = sorted({int(t) for t in weighted["target_season"]})
    models = sorted(set(weighted["model"]))
    if roles != [role] or targets != [target] or models != ["weighted"]:
        raise ValueError(
            f"가중 예측 표는 역할 {role}·목표 시즌 {target}·모델 weighted 하나여야 한다: "
            f"역할 {roles}, 목표 시즌 {targets}, 모델 {models}"
        )
    expected = {int(i) for i in cohort["player_id"]}
    got = {int(i) for i in weighted["player_id"]}
    if got != expected:
        raise ValueError(
            f"가중 예측 표의 선수가 {target} {role} 대상 표와 다르다 — "
            f"예측 없음 {sorted(expected - got)[:_SHOW_MAX]}, "
            f"대상 아님 {sorted(got - expected)[:_SHOW_MAX]}"
        )


def _empty_residuals(metric: str) -> pd.DataFrame:
    """학습 목표가 없을 때의 빈 학습 표 (`residual_frame`과 같은 열)."""
    return pd.DataFrame(
        {
            "train_target": pd.Series(dtype="int64"),
            "player_id": pd.Series(dtype="int64"),
            **{name: pd.Series(dtype="float64") for name in EXT_FEATURES[metric]},
            "y": pd.Series(dtype="float64"),
        }
    )


def _training_rows(
    tables: Tables, role: Role, metric: str, train: int, params: Params
) -> pd.DataFrame:
    """학습 목표 T 하나의 보정 학습 행 (모듈 docstring "보정 대상 y", Y 자리에 T)."""
    cohort = cohort_for(tables, role, train)
    seasons = tables.seasons[role]
    ids = pd.Index(cohort["player_id"].astype("int64"), name="player_id")
    rates = weighted_rates(cohort, seasons, tables.league[role], role, params)
    rates = rates.set_index("player_id").reindex(ids)
    features = _features(
        tables, role, train, params, cohort, rates["era"] if role == "pit" else None
    )
    # 실제값은 T 행만. T에 행이 없는 대상은 결측(NaN)이다
    actual = actual_values(seasons, role, train).set_index("player_id").reindex(ids)
    if metric in _VOLUME_COLUMN:
        column = _VOLUME_COLUMN[metric]
        pt = predict_playing_time(cohort, seasons, role, tables.debuts)
        predicted = _floats(pt.set_index("player_id")[column].reindex(ids))
        # 대상 전원. T 미출전은 실제 0
        y = np.nan_to_num(_floats(actual[column]), nan=0.0) - predicted
        keep = np.ones(len(ids), dtype=bool)
    else:
        sample, threshold = _RATE_POPULATION[role]
        if metric == "hr":
            observed = _ratio(_floats(actual["hr"]), _floats(actual["pa"]))
        else:
            observed = _floats(actual[metric])
        y = observed - _floats(rates[_RATE_PREDICTION[metric]])
        # 실제 표본 기준 이상만 (미출전 NaN 비교는 거짓). 실제값이 결측인 행은 맞출 수 없어 뺀다
        keep = (_floats(actual[sample]) >= threshold) & ~np.isnan(y)
    part = features.loc[keep, ["player_id", *EXT_FEATURES[metric]]]
    part.insert(0, "train_target", np.int64(train))
    part["y"] = y[keep]
    return part


def _combine_bat(
    out: pd.DataFrame,
    hr_per_pa: npt.NDArray[np.float64],
    deltas: Mapping[str, npt.NDArray[np.float64]],
) -> dict[str, npt.NDArray[np.float64]]:
    """타자 채택 지표의 새 열 값 (모듈 docstring "예측 결합"). 채택하지 않은 열은 돌려주지 않는다."""
    changed: dict[str, npt.NDArray[np.float64]] = {}
    pa_e = _floats(out["pa"])
    hr_per_pa_e = hr_per_pa
    if "pa" in deltas:
        pa_e = np.maximum(pa_e + deltas["pa"], 0.0)
        changed["pa"] = pa_e
    if "hr" in deltas:
        hr_per_pa_e = np.maximum(hr_per_pa + deltas["hr"], 0.0)
    if "pa" in deltas or "hr" in deltas:
        # hr = pa_e × hr_per_pa_e (가중 hr / pa로 되돌리지 않는다 — 가중 pa가 0인 선수가 있다)
        changed["hr"] = pa_e * hr_per_pa_e
    if "ops" in deltas:
        ops_w = _floats(out["ops"])
        ops_e = np.maximum(ops_w + deltas["ops"], 0.0)
        # obp·slg는 ops_e / ops_w 비율로 같이 바꾼다. ops_w = 0이면 그대로
        nonzero = ops_w != 0
        safe = np.where(nonzero, ops_w, 1.0)
        for column in ("obp", "slg"):
            base = _floats(out[column])
            changed[column] = np.where(nonzero, base * ops_e / safe, base)
        changed["ops"] = ops_e
    return changed


def _combine_pit(
    out: pd.DataFrame, outs: npt.NDArray[np.float64], deltas: Mapping[str, npt.NDArray[np.float64]]
) -> dict[str, npt.NDArray[np.float64]]:
    """투수 채택 지표의 새 열 값. IP 보정은 아웃 단위, ERA는 0에서 자르고 K−BB%는 자르지 않는다."""
    changed: dict[str, npt.NDArray[np.float64]] = {}
    if "ip" in deltas:
        changed["ip"] = np.maximum(outs + deltas["ip"], 0.0) / _OUTS_PER_INNING
    if "era" in deltas:
        changed["era"] = np.maximum(_floats(out["era"]) + deltas["era"], 0.0)
    if "k_bb_pct" in deltas:
        changed["k_bb_pct"] = _floats(out["k_bb_pct"]) + deltas["k_bb_pct"]
    return changed


class Correction(NamedTuple):
    """지표 하나의 선형 보정.

    Attributes:
        metric: 지표 (`EXT_METRICS`).
        features: 특성 이름 (`EXT_FEATURES[metric]`).
        coef: [절편, 특성 계수…] (길이 = 특성 수 + 1). 학습 행이 부족하면 모두 0.
        fill: 특성 → 결측을 채우는 값 (학습 행 평균, 없으면 0).
        n_train: 학습 행 수.
    """

    metric: str
    features: tuple[str, ...]
    coef: npt.NDArray[np.float64]
    fill: dict[str, float]
    n_train: int


def weighted_projection_for(
    tables: Tables, role: Role, target: int, params: Params
) -> pd.DataFrame:
    """목표 시즌 하나의 가중 예측 표 (model "weighted", 모듈 docstring "가중 예측"). 메인이 구현해 둔다."""
    cohort = cohort_for(tables, role, target)
    seasons = tables.seasons[role]
    rates = weighted_rates(cohort, seasons, tables.league[role], role, params)
    playing_time = predict_playing_time(cohort, seasons, role, tables.debuts)
    if role == "bat":
        return assemble_bat(cohort, rates, playing_time, "weighted")
    return assemble_pit(cohort, rates, playing_time, "weighted")


def extended_features(
    tables: Tables, role: Role, target: int, params: Params, weighted: pd.DataFrame
) -> pd.DataFrame:
    """목표 Y 대상의 특성 표. 열: player_id + `ROLE_FEATURES[role]`, player_id 순, 결측 허용.

    Args:
        weighted: 같은 목표의 가중 예측 표 (`weighted_projection_for`). 투수 xera_minus_proj에 쓴다.

    Raises:
        ValueError: weighted가 예측 표 계약을 어기거나, 역할·목표 시즌·모델("weighted")·선수가
            목표 Y 대상과 다를 때.
    """
    cohort = cohort_for(tables, role, target)
    _check_weighted(weighted, cohort, role, target)
    era = weighted.set_index("player_id")["era"] if role == "pit" else None
    return _features(tables, role, target, params, cohort, era)


def residual_frame(
    tables: Tables, role: Role, metric: str, target: int, params: Params
) -> pd.DataFrame:
    """목표 Y의 보정 학습 행. 열: train_target, player_id, `EXT_FEATURES[metric]`…, y.

    특성은 결측을 채우기 전 값이다. (train_target, player_id) 순. 학습 목표가 없으면 빈 표(같은 열).

    Raises:
        ValueError: metric이 그 역할의 `EXT_ROLE_METRICS`가 아닐 때.
    """
    if metric not in EXT_ROLE_METRICS[role]:
        raise ValueError(f"{role} 보정 지표는 {EXT_ROLE_METRICS[role]} 중 하나: {metric!r}")
    first = EXT_TRAIN_FIRST_VOLUME if metric in _VOLUME_COLUMN else EXT_TRAIN_FIRST
    # 학습 목표 T ≤ Y−1 — T의 입력은 T−1 이하라 학습 전체가 Y−1 이하 기록만 쓴다(N2)
    parts = [_training_rows(tables, role, metric, train, params) for train in range(first, target)]
    if not parts:
        return _empty_residuals(metric)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["train_target", "player_id"], kind="mergesort").reset_index(drop=True)


def fit_correction(frame: pd.DataFrame, metric: str) -> Correction:
    """`residual_frame` 결과로 보정 계수를 맞춘다 (모듈 docstring "계수").

    Raises:
        ValueError: metric이 `EXT_FEATURES`에 없거나, y에 결측·무한값이 있을 때 (최소제곱이 결측 계수를 낸다).
    """
    features = _metric_features(metric)
    n_train = len(frame)
    target = frame["y"].to_numpy(dtype="float64", na_value=np.nan)
    bad_y = int((~np.isfinite(target)).sum())
    if bad_y:
        raise ValueError(f"{metric} 학습 행 {n_train}개 중 {bad_y}개의 y가 결측·무한값이다")
    values = frame[list(features)].to_numpy(dtype="float64", na_value=np.nan)
    known = ~np.isnan(values)
    # 결측 채움값 = 학습 행의 특성 평균 (값이 하나도 없으면 0)
    counts = known.sum(axis=0)
    sums = np.where(known, values, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros(len(features)), where=counts > 0)
    fill = {name: float(value) for name, value in zip(features, means, strict=True)}
    coef = np.zeros(len(features) + 1, dtype="float64")
    if n_train >= (len(features) + 1) * EXT_MIN_ROWS_PER_COEF:
        design = np.column_stack([np.ones(n_train), np.where(known, values, means)])
        # 값이 모두 채움값 0인 특성은 최소 노름 해라 계수가 0이 된다
        coef = np.asarray(np.linalg.lstsq(design, target, rcond=None)[0], dtype="float64")
    return Correction(metric, features, coef, fill, n_train)


def predict_delta(features: pd.DataFrame, correction: Correction) -> pd.Series:
    """특성 표(`extended_features`)의 행마다 Δ = 절편 + Σ 계수 × 특성(결측은 fill). 행 순서 그대로."""
    names = list(correction.features)
    values = features[names].to_numpy(dtype="float64", na_value=np.nan)
    fill = np.array([correction.fill[name] for name in names], dtype="float64")
    filled = np.where(np.isnan(values), fill, values)
    delta = correction.coef[0] + filled @ correction.coef[1:]
    return pd.Series(delta, index=features.index, name=correction.metric, dtype="float64")


def extended_projection(
    tables: Tables, role: Role, target: int, params: Params, adoption: Adoption
) -> pd.DataFrame:
    """목표 Y의 확장 예측 표 (model "extended", `BAT_PROJECTIONS`·`PIT_PROJECTIONS`, player_id 순).

    채택한 지표만 `residual_frame` → `fit_correction` → `predict_delta`로 보정한다.

    Raises:
        ValueError: 채택표가 형식을 어길 때 (`validate_adoption`).
    """
    validate_adoption(adoption)
    # 가중 예측 (weighted_projection_for와 같은 계산). hr_per_pa·아웃은 결합에 따로 쓴다
    cohort = cohort_for(tables, role, target)
    seasons = tables.seasons[role]
    rates = weighted_rates(cohort, seasons, tables.league[role], role, params)
    playing_time = predict_playing_time(cohort, seasons, role, tables.debuts)
    assemble = assemble_bat if role == "bat" else assemble_pit
    out = assemble(cohort, rates, playing_time, "extended")

    adopted = {metric: adoption.adopt[metric] for metric in EXT_ROLE_METRICS[role]}
    fitted = corrections(tables, role, target, params, adopted)
    if not fitted:
        return out  # 채택 지표가 없으면 값은 가중 예측 그대로
    ids = pd.Index(out["player_id"].astype("int64"), name="player_id")
    rates = rates.set_index("player_id").reindex(ids)
    features = _features(
        tables, role, target, params, cohort, rates["era"] if role == "pit" else None
    )
    features = features.set_index("player_id").reindex(ids).reset_index()
    deltas = {
        metric: predict_delta(features, correction).to_numpy(dtype="float64")
        for metric, correction in fitted.items()
    }
    if role == "bat":
        changed = _combine_bat(out, _floats(rates["hr_per_pa"]), deltas)
    else:
        outs = _floats(playing_time.set_index("player_id")["outs"].reindex(ids))
        changed = _combine_pit(out, outs, deltas)
    return out.assign(**changed)


def corrections(
    tables: Tables, role: Role, target: int, params: Params, metrics: Mapping[str, bool]
) -> dict[str, Correction]:
    """metrics에서 True인 지표들의 보정 (지표 → Correction). 설명·검증 기록용."""
    return {
        metric: fit_correction(residual_frame(tables, role, metric, target, params), metric)
        for metric, on in metrics.items()
        if on
    }
