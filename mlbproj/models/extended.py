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

from mlbproj.contracts import Role
from mlbproj.models.assemble import assemble_bat, assemble_pit
from mlbproj.models.params import Adoption, Params
from mlbproj.models.playing_time import predict_playing_time
from mlbproj.models.weighted import weighted_rates
from mlbproj.tables import Tables, cohort_for

# 역할별 특성 열 (EXT_ROLE_METRICS 순서로 처음 나온 차례). extended_features의 열 순서다.
ROLE_FEATURES: dict[Role, tuple[str, ...]] = {
    "bat": ("age_c", "age_c2", "barrel_pct", "ev_avg", "k_pct", "xwoba_minus_woba"),
    "pit": ("age_c", "age_c2", "xera_minus_proj", "whiff_pct"),
}


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
    """
    raise NotImplementedError


def residual_frame(
    tables: Tables, role: Role, metric: str, target: int, params: Params
) -> pd.DataFrame:
    """목표 Y의 보정 학습 행. 열: train_target, player_id, `EXT_FEATURES[metric]`…, y.

    특성은 결측을 채우기 전 값이다. (train_target, player_id) 순. 학습 목표가 없으면 빈 표(같은 열).

    Raises:
        ValueError: metric이 그 역할의 `EXT_ROLE_METRICS`가 아닐 때.
    """
    raise NotImplementedError


def fit_correction(frame: pd.DataFrame, metric: str) -> Correction:
    """`residual_frame` 결과로 보정 계수를 맞춘다 (모듈 docstring "계수")."""
    raise NotImplementedError


def predict_delta(features: pd.DataFrame, correction: Correction) -> pd.Series:
    """특성 표(`extended_features`)의 행마다 Δ = 절편 + Σ 계수 × 특성(결측은 fill). 행 순서 그대로."""
    raise NotImplementedError


def extended_projection(
    tables: Tables, role: Role, target: int, params: Params, adoption: Adoption
) -> pd.DataFrame:
    """목표 Y의 확장 예측 표 (model "extended", `BAT_PROJECTIONS`·`PIT_PROJECTIONS`, player_id 순).

    채택한 지표만 `residual_frame` → `fit_correction` → `predict_delta`로 보정한다.

    Raises:
        ValueError: 채택표가 형식을 어길 때 (`validate_adoption`).
    """
    raise NotImplementedError


def corrections(
    tables: Tables, role: Role, target: int, params: Params, metrics: Mapping[str, bool]
) -> dict[str, Correction]:
    """metrics에서 True인 지표들의 보정 (지표 → Correction). 설명·검증 기록용."""
    raise NotImplementedError
