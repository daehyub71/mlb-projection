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

from typing import NamedTuple

import pandas as pd

from mlbproj.contracts import Role


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


def actual_values(seasons: pd.DataFrame, role: Role, target_season: int) -> pd.DataFrame:
    """목표 시즌의 실제 값. 열: player_id, 그 역할 지표들(METRICS 순), 분모 열(타자 pa / 투수 outs·bf).

    분모가 0이면 그 비율은 결측. Y에 행이 없는 선수는 결과에 없다.
    """
    raise NotImplementedError


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
    raise NotImplementedError
