"""시즌별 리그 평균 (SPEC F6·F7·D11, `contracts.LEAGUE_BAT`·`LEAGUE_PIT`). 레인 A가 구현한다.

입력은 정제 표(`BATTING_SEASONS`·`PITCHING_SEASONS`)다. 리그 평균은 그 역할 표의 **모든 행 합**으로
계산한다(선수 평균의 평균이 아니다). 목표 시즌 Y의 모델은 Y−1 행만 쓴다 — 한 시즌의 행은 그 시즌
기록만으로 정해지므로 미래가 섞이지 않는다(N2).

공식 (합은 그 시즌 모든 행)
- 타자: h1 = h − b2 − b3 − hr. r_<성분> = Σ성분 / Σpa (성분 = `contracts.BAT_COMPONENTS`).
  obp = (Σh + Σbb + Σhbp) / (Σab + Σbb + Σhbp + Σsf), slg = (Σh1 + 2Σb2 + 3Σb3 + 4Σhr) / Σab,
  ops = obp + slg, hr_per_pa = Σhr / Σpa.
- 투수: r_<성분> = Σ성분 / Σbf (`contracts.PIT_COMPONENTS`), er_per_out = Σer / Σouts,
  era = 27 × er_per_out, k_bb_pct = 100 × (r_so − r_bb),
  fip_const = era − (13Σhr + 3(Σbb + Σhbp) − 2Σso) / (Σouts / 3), bf_per_out = Σbf / Σouts.
"""

import pandas as pd


def league_batting(batting: pd.DataFrame) -> pd.DataFrame:
    """시즌별 타자 리그 평균 (`contracts.LEAGUE_BAT`, `check_frame` 통과, season 순).

    Args:
        batting: `contracts.BATTING_SEASONS` 표.

    Raises:
        ValueError: 입력이 계약을 어길 때, 또는 어떤 시즌의 Σpa·Σab가 0일 때.
    """
    raise NotImplementedError


def league_pitching(pitching: pd.DataFrame) -> pd.DataFrame:
    """시즌별 투수 리그 평균 (`contracts.LEAGUE_PIT`, `check_frame` 통과, season 순).

    Args:
        pitching: `contracts.PITCHING_SEASONS` 표.

    Raises:
        ValueError: 입력이 계약을 어길 때, 또는 어떤 시즌의 Σbf·Σouts가 0일 때.
    """
    raise NotImplementedError


def league_row(league: pd.DataFrame, season: int) -> pd.Series:
    """리그 평균 표에서 한 시즌 행을 꺼낸다.

    Raises:
        KeyError: 그 시즌이 없을 때 (메시지에 시즌을 적는다).
    """
    raise NotImplementedError
