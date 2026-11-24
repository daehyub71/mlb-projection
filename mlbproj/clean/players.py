"""선수 표 (SPEC §6 `players`, D23). 레인 A가 구현한다.

원본은 `collect.raw.read_json`과 `collect.statsapi.stats_splits`로만 읽는다.
"""

from pathlib import Path

import pandas as pd


def people_records(raw_dir: Path) -> list[dict[str, object]]:
    """선수 정보 원본 묶음을 모두 읽어 `people` 항목을 이어 붙인다.

    묶음은 `contracts.raw_people_path(i)`를 i = 0부터 파일이 없을 때까지 읽는다(이름이 다른
    파일은 읽지 않는다 — TASKS 미해소 이슈 ③). 같은 ID가 여러 묶음에 있으면 값이 완전히 같을 때만
    하나로 두고, 다르면 ValueError.

    Raises:
        FileNotFoundError: 묶음이 하나도 없을 때.
        ValueError: 묶음끼리 같은 선수의 값이 다를 때.
    """
    raise NotImplementedError


def build_players(raw_dir: Path) -> pd.DataFrame:
    """선수 표를 만든다 (`contracts.PLAYERS`, `check_frame` 통과, player_id 오름차순).

    - 대상: `contracts.SEASONS` 모든 시즌의 API 타격·투구 원본에 **행이 하나라도 있는** 선수 전원
      (타석·상대 타자가 0이어도 행이 있으면 MLB 기록이다).
    - name = fullName, birth_date = birthDate(YYYY-MM-DD, 없으면 결측), primary_position =
      primaryPosition.abbreviation, bats = batSide.code, throws = pitchHand.code (없으면 결측).
    - active = 선수 정보의 active 그대로(bool). 은퇴 판정이 아니다 — 표시용(D23, E16).
    - last_mlb_season = 그 선수가 타격 또는 투구 원본에 행으로 나온 마지막 시즌.

    Raises:
        ValueError: 원본에 나온 선수가 선수 정보에 없을 때 (빠진 ID를 적는다).
    """
    raise NotImplementedError
