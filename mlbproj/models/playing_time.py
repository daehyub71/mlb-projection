"""출전량 모델 (SPEC F8·D9·D12·N2). 레인 C가 구현한다.

교과서 마르셀 공식을 쓰지 않고(E11), 선형 회귀 계수를 **목표 시즌 Y보다 앞선 목표들**로 적합한다.
numpy 최소제곱(`numpy.linalg.lstsq`, pandas에 포함)만 쓴다.

특성 (대상 표의 한 선수, 목표 시즌 T 기준 — 같은 규칙으로 학습·예측에 쓴다)
- 타자: pt1 = T−1의 pa_adj, pt2 = T−2의 pa_adj (행이 없으면 0, D9 환산값), gap = (T−1) − input_last,
  n_seasons(대상 표), 상수항 1.
- 투수: pt1·pt2 = outs_adj, gap, n_seasons, gs_share = 기간 안 Σgs / Σg (Σg = 0이면 0), 상수항 1.
- 특성 열 순서는 `FEATURES[role]`.

학습 (목표 Y를 예측할 때)
- 학습 목표 T = max(SEASON_FIRST + 2, …) ~ Y−1 (T−2가 데이터 안에 있어야 한다: T ≥ 2021).
- 각 T마다 `cohort.build_cohort(seasons, role, T, debut_seasons)`로 대상을 만들고, 정답은 T의 pa_adj
  (투수 outs_adj), 행이 없으면 0.
- 모든 T의 행을 합쳐 최소제곱으로 계수를 구한다. **season ≥ Y 행은 특성·정답 어디에도 쓰지 않는다**(N2).
- 예측 = 특성 · 계수, 0 미만은 0으로 자른다.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from mlbproj.contracts import Role

FEATURES: dict[Role, tuple[str, ...]] = {
    "bat": ("pt1", "pt2", "gap", "n_seasons", "const"),
    "pit": ("pt1", "pt2", "gap", "n_seasons", "gs_share", "const"),
}


def playing_time_features(cohort: pd.DataFrame, seasons: pd.DataFrame, role: Role) -> pd.DataFrame:
    """대상 표의 선수마다 특성을 만든다. 열: player_id, 그다음 `FEATURES[role]` 순서 (모두 실수, 결측 없음).

    Raises:
        ValueError: 입력 표가 계약을 어기거나 대상 표의 역할·목표 시즌이 하나가 아닐 때.
    """
    raise NotImplementedError


def training_frame(
    seasons: pd.DataFrame,
    role: Role,
    target_season: int,
    debut_seasons: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """목표 Y의 학습 표: 열 = player_id, train_target, `FEATURES[role]`, label. 학습 목표가 없으면 빈 표."""
    raise NotImplementedError


def fit_coefficients(train: pd.DataFrame, role: Role) -> np.ndarray:
    """최소제곱 계수 (길이 = len(FEATURES[role])).

    Raises:
        ValueError: 학습 행이 특성 수보다 적을 때.
    """
    raise NotImplementedError


def predict_playing_time(
    cohort: pd.DataFrame,
    seasons: pd.DataFrame,
    role: Role,
    debut_seasons: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """대상 표의 출전량 예측. 타자 `contracts.BAT_PLAYING_TIME`(pa), 투수 `PIT_PLAYING_TIME`(outs).

    학습(`training_frame` → `fit_coefficients`)과 예측을 한 번에 한다. 결과는 player_id 순, 0 이상.

    Raises:
        ValueError: 입력 표가 계약을 어기거나 학습 행이 모자랄 때.
    """
    raise NotImplementedError
