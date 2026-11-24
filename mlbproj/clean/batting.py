"""타자 시즌 표 (SPEC F3·D8·D9, §6 `batting_seasons`). 레인 A가 구현한다.

원본은 `collect.statsapi.stats_splits`와 `collect.savant.load_savant`로만 읽는다.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

# 정제 열 ← API stat 필드
BATTING_STAT_FIELDS: Mapping[str, str] = {
    "pa": "plateAppearances",
    "ab": "atBats",
    "h": "hits",
    "b2": "doubles",
    "b3": "triples",
    "hr": "homeRuns",
    "bb": "baseOnBalls",
    "ibb": "intentionalWalks",
    "hbp": "hitByPitch",
    "sf": "sacFlies",
    "sh": "sacBunts",
    "so": "strikeOuts",
}
# Savant 타자 CSV 열 → 정제 열 (값은 그대로 — 퍼센트 단위·소수 단위 유지)
BATTING_SAVANT_COLUMNS: Mapping[str, str] = {
    "woba": "woba",
    "xwoba": "xwoba",
    "exit_velocity_avg": "ev_avg",
    "launch_angle_avg": "la_avg",
    "hard_hit_percent": "hard_hit_pct",
    "barrel_batted_rate": "barrel_pct",
}


def is_batter_split(split: Mapping[str, Any]) -> bool:
    """타격 기록 한 행이 타자 역할인지 (D8, contracts의 역할 판정 주석).

    포지션(`position.abbreviation`)이 `contracts.BATTER_EXCLUDED_POSITIONS`가 아니고
    `stat.plateAppearances` > 0이면 True. 포지션 X·DH·TWP도 타자다.
    """
    raise NotImplementedError


def parse_rate(text: Any) -> float:
    """API의 비율 문자열(".350", "1.000")을 실수로. ".---"·"-.--"·빈 값·None은 NaN."""
    raise NotImplementedError


def build_batting(raw_dir: Path) -> pd.DataFrame:
    """타자 시즌 표를 만든다 (`contracts.BATTING_SEASONS`, `check_frame` 통과, season·player_id 순).

    - 시즌: `contracts.SEASONS`. 행: `is_batter_split`이 True인 API 타격 split. league = "MLB".
    - age = stat.age, team_name = team.name(없으면 결측), n_teams = numTeams.
    - 셈 열은 `BATTING_STAT_FIELDS`대로 정수. obp·slg·ops는 `parse_rate`.
    - pa_adj = pa × `SHORT_SEASON_SCALE` (season == `SHORT_SEASON`), 그 밖엔 pa (실수). is_short_season.
    - Statcast: (player_id, season)으로 `load_savant(raw_dir, season, "batter")`를 붙여
      `BATTING_SAVANT_COLUMNS`대로 이름을 바꾼다. Savant에 없는 행은 결측(F3).
      Savant 타자 CSV에 같은 ID가 두 번 있으면 값이 같을 때만 하나로 두고, 다르면 ValueError.
    """
    raise NotImplementedError
