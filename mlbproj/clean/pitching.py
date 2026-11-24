"""투수 시즌 표 (SPEC F3·D8·D9·E3, §6 `pitching_seasons`). 레인 B가 구현한다.

원본은 `collect.statsapi.stats_splits`와 `collect.savant.load_savant`로만 읽는다.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

# 정제 열 ← API stat 필드. GS는 API에서 받는다(Savant p_gs는 비어 있다, E3).
PITCHING_STAT_FIELDS: Mapping[str, str] = {
    "g": "gamesPitched",
    "gs": "gamesStarted",
    "outs": "outs",
    "bf": "battersFaced",
    "er": "earnedRuns",
    "h": "hits",
    "hr": "homeRuns",
    "bb": "baseOnBalls",
    "hbp": "hitByPitch",
    "so": "strikeOuts",
    "sv": "saves",
}
# Savant 투수 CSV 열 → 정제 열
PITCHING_SAVANT_COLUMNS: Mapping[str, str] = {
    "xwoba": "xwoba_against",
    "whiff_percent": "whiff_pct",
    "hard_hit_percent": "hard_hit_pct",
    "barrel_batted_rate": "barrel_pct",
    "xera": "xera",
}


def ip_to_outs(ip: Any) -> int:
    """이닝 표기를 아웃 수로 바꾼다. "172.1" → 517, "0.2" → 2, "5" → 15, 숫자 172.1도 같다.

    Raises:
        ValueError: 소수 부분이 0·1·2가 아니거나(예: "1.3"), 음수이거나, 숫자가 아닐 때.
    """
    raise NotImplementedError


def is_pitcher_split(split: Mapping[str, Any]) -> bool:
    """투구 기록 한 행이 투수 역할인지 (D8): 포지션이 `contracts.PITCHER_POSITIONS`(P·TWP)면 True.

    등판한 야수(포지션이 그 밖)는 False.
    """
    raise NotImplementedError


def build_pitching(raw_dir: Path) -> pd.DataFrame:
    """투수 시즌 표를 만든다 (`contracts.PITCHING_SEASONS`, `check_frame` 통과, season·player_id 순).

    - 시즌: `contracts.SEASONS`. 행: `is_pitcher_split`이 True인 API 투구 split. league = "MLB".
    - age = stat.age, team_name = team.name(없으면 결측), n_teams = numTeams.
    - 셈 열은 `PITCHING_STAT_FIELDS`대로 정수. outs는 API `outs`를 쓰고, `ip_to_outs(inningsPitched)`와
      다르면 ValueError (F2-1이 막지만 정제에서도 확인한다).
    - outs_adj = outs × `SHORT_SEASON_SCALE` (season == `SHORT_SEASON`), 그 밖엔 outs (실수). is_short_season.
    - Statcast: (player_id, season)으로 `load_savant(raw_dir, season, "pitcher")`를 붙여
      `PITCHING_SAVANT_COLUMNS`대로 바꾼다. 없는 행은 결측. 양손 투수처럼 **값이 모두 같은 중복 행**(E19)은
      하나로 합치고, 값이 다른 중복은 ValueError.
    """
    raise NotImplementedError
