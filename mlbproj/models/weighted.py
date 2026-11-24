"""가중 + 리그 평균 회귀 비율 모델 (SPEC F7·D9·D10·D11). 레인 B가 구현한다.

입력은 목표 시즌 Y의 대상 표(`COHORTS`), 같은 역할 정제 표, 리그 평균 표(`LEAGUE_BAT`/`LEAGUE_PIT`),
파라미터(`models.params.Params`)다. **Y 이후 행·리그 평균은 쓰지 않는다**(N2). 리그 평균은 Y−1 행만 쓴다.

규칙
- 대상마다 기간(window_first ~ window_last, 최대 5시즌) 안의 행만 쓴다. 시즌 Y−k의 가중치는
  `params.<role>.weights[k−1]` (k = 1…5). 행이 없는 시즌은 더하지 않는다.
- 비율 지표는 **실제 표본**(pa·bf·outs, 환산 전)으로 가중한다(D9 — 2020 환산은 출전량에만).
- 성분 x의 회귀 비율 = (Σ w·x + R_x · lg_x) / (Σ w·분모 + R_x). R_x = `params.<role>.regress[x]`,
  lg_x = Y−1 리그의 r_x. 타자 분모는 pa, 투수 성분 분모는 bf.
- 타자: 회귀한 r_h1…r_ab로 obp = (r_h1+r_b2+r_b3+r_hr+r_bb+r_hbp)/(r_ab+r_bb+r_hbp+r_sf),
  slg = (r_h1+2r_b2+3r_b3+4r_hr)/r_ab, ops = obp + slg, hr_per_pa = r_hr. lg_ops = Y−1 리그 ops.
- 투수: er_per_out = (Σw·er + R_er·lg_er_per_out)/(Σw·outs + R_er). k_bb_pct = 100(r_so − r_bb).
  era는 `params.era_method`로 고른다(D11):
  - "er": era = 27 × er_per_out.
  - "fip": 선수 bf_per_out = (Σw·bf + R_er·lg_bf_per_out)/(Σw·outs + R_er),
    era = (13r_hr + 3(r_bb + r_hbp) − 2r_so) × bf_per_out × 3 + Y−1 리그 fip_const.
  lg_era·lg_k_bb_pct = Y−1 리그 값.
- 결과의 선수 집합은 대상 표와 같고 player_id 순, 결측 없음. target_season = Y.
"""

import pandas as pd

from mlbproj.contracts import Role
from mlbproj.models.params import Params


def weighted_rates(
    cohort: pd.DataFrame,
    seasons: pd.DataFrame,
    league: pd.DataFrame,
    role: Role,
    params: Params,
) -> pd.DataFrame:
    """가중·회귀 비율. 타자 `contracts.BAT_RATES`, 투수 `contracts.PIT_RATES` 형식 (`check_frame` 통과).

    Raises:
        ValueError: 입력 표가 계약을 어기거나, 대상 표의 역할·목표 시즌이 하나가 아니거나, 파라미터가 틀릴 때.
        KeyError: 리그 평균에 Y−1이 없을 때.
    """
    raise NotImplementedError
