"""WAR 정제 표 — SPEC v0.5 F3, D8·D33. 레인 A가 구현한다.

세이버메트릭스 원본을 `contracts.WAR_SEASONS` 표로 만든다. 역할 규칙(D8)은 이미 정제된 시즌 표
(`BATTING_SEASONS`·`PITCHING_SEASONS`)를 기준으로 삼는다 — 그 표에 있는 (선수, 시즌) 행만 WAR 행을 만든다.
"""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def build_war_seasons(
    raw_dir: Path, seasons: Iterable[int], batting: pd.DataFrame, pitching: pd.DataFrame
) -> pd.DataFrame:
    """WAR 정제 표를 만든다.

    - 역할 "bat": `batting`(BATTING_SEASONS)의 각 (player_id, season) 행에 그 시즌 타격 세이버메트릭스의 값을
      `SABER_COLUMN_MAP["bat"]`대로 붙인다. 투수 열(`war_ra9`)은 결측.
    - 역할 "pit": `pitching`의 각 행에 투구 세이버메트릭스를 `SABER_COLUMN_MAP["pit"]`대로 붙인다. 타자 구성 열은 결측.
    - league = `contracts.LEAGUE`. 시즌 표에 없는 세이버메트릭스 행(투수 타격·야수 등판)은 버린다.
    - 결과는 `WAR_SEASONS` 열 순서, 키(player_id, season, league, role) 오름차순, `check_frame` 통과.

    Args:
        raw_dir: 원본 디렉토리.
        seasons: 만들 시즌들. 시즌 표에서 이 시즌들의 행만 본다.
        batting: 정제 타자 시즌 표.
        pitching: 정제 투수 시즌 표.

    Raises:
        ValueError: 시즌 표의 행에 맞는 세이버메트릭스 행이 없거나, 같은 선수가 한 원본에 두 번 있거나,
            war가 유한한 숫자가 아닐 때(F2 ⑦을 통과했다면 없다). 어느 선수·시즌인지 적는다.
        FileNotFoundError: 원본이 없을 때.
    """
    raise NotImplementedError
