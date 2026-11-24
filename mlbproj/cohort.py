"""예측 대상과 사용 기간 (SPEC F4·F5·F12·D3·N2). 레인 C가 구현한다.

입력은 정제 표(`contracts.BATTING_SEASONS` / `PITCHING_SEASONS` 형식의 DataFrame)뿐이다.
원본 디렉토리를 읽지 않는다. **목표 시즌 Y 이후(season ≥ Y)의 행은 결과에 영향을 주면 안 된다**(N2) —
그런 행을 섞어 넣어도 결과가 같아야 한다.
"""

import pandas as pd

from mlbproj.contracts import Role

# 기간 안 미출전 시즌에 0으로 채우는 출전량 열 (F5). 나머지 값 열은 결측으로 둔다.
PLAYING_TIME_COLUMNS: dict[Role, tuple[str, ...]] = {
    "bat": ("pa", "pa_adj"),
    "pit": ("g", "gs", "outs", "outs_adj", "bf"),
}


def build_cohort(seasons: pd.DataFrame, role: Role, target_season: int) -> pd.DataFrame:
    """목표 시즌 Y·역할 r의 예측 대상과 사용 기간 (`contracts.COHORTS`, `check_frame` 통과, player_id 순).

    - 대상(F4·D3): Y−`WINDOW` ~ Y−1 안에 그 역할의 행이 하나라도 있는 선수 전원. 은퇴·장기 결장도 빼지 않는다.
    - window_first = max(Y−WINDOW, 그 선수의 이 역할 첫 행 시즌), window_last = Y−1,
      n_window = window_last − window_first + 1 (미출전 시즌 포함, F5: 빈 시즌을 더 오래된 기록으로 채우지 않는다).
    - input_first·input_last = 기간 안 첫·마지막 행 시즌, n_seasons = 기간 안 행이 있는 시즌 수.
    - played_prev = Y−1에 행이 있는지.
    - sample = 기간 안 pa 합(타자) 또는 bf 합(투수). sample_outs = 기간 안 outs 합(투수), 타자는 결측.
    - 첫 행 시즌은 Y 이전 행만으로 정한다(N2).

    Args:
        seasons: 그 역할의 정제 표 (`check_frame`으로 확인한다).
        role: "bat" 또는 "pit". seasons 형식과 맞아야 한다.
        target_season: 목표 시즌 Y.

    Raises:
        ValueError: seasons가 role의 계약을 어길 때.
    """
    raise NotImplementedError


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
    """
    raise NotImplementedError
