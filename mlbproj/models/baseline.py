"""기준 모델 — 직전 시즌과 리그 평균 (SPEC F6, D13 기준선). 레인 A가 구현한다.

모두 목표 시즌 Y의 대상 표(`contracts.COHORTS`) 한 장과 같은 역할의 정제 표, 리그 평균 표를 받는다.
**Y 이후(season ≥ Y) 행은 쓰지 않는다**(N2). 결과의 선수 집합은 대상 표와 정확히 같고 player_id 순이다.

비율 공식은 `models.league` 모듈 docstring과 같다(한 선수·한 시즌의 합으로 계산). 분모(pa·ab·bf·outs)가
0이면 그 비율은 리그 평균(Y−1)으로 채운다 — 결과에 결측이 없다.
"""

import pandas as pd

from mlbproj.contracts import Role


def baseline_rates(
    cohort: pd.DataFrame, seasons: pd.DataFrame, league: pd.DataFrame, role: Role
) -> pd.DataFrame:
    """직전 시즌 비율 (F6). 타자 `BAT_RATES`, 투수 `PIT_RATES` 형식.

    - Y−1에 행이 있으면 그 시즌의 실제 성분·비율(r_*, obp·slg·ops·hr_per_pa / er_per_out·era·k_bb_pct),
      없으면 Y−1 리그 평균.
    - lg_ops / lg_era·lg_k_bb_pct = Y−1 리그 평균. target_season = Y.

    Raises:
        ValueError: 입력 표가 계약을 어기거나 대상 표의 역할·목표 시즌이 하나가 아닐 때.
        KeyError: 리그 평균에 Y−1이 없을 때.
    """
    raise NotImplementedError


def league_model_rates(cohort: pd.DataFrame, league: pd.DataFrame, role: Role) -> pd.DataFrame:
    """리그 평균 모델의 비율 — 모든 대상에게 Y−1 리그 평균 (D13 기준선, model "league")."""
    raise NotImplementedError


def baseline_playing_time(cohort: pd.DataFrame, seasons: pd.DataFrame, role: Role) -> pd.DataFrame:
    """직전 시즌 출전량 (F6). 타자 `BAT_PLAYING_TIME`(pa = Y−1 pa_adj), 투수 `PIT_PLAYING_TIME`
    (outs = Y−1 outs_adj). Y−1에 행이 없으면 0.
    """
    raise NotImplementedError
