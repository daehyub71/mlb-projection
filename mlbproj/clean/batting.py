"""타자 시즌 표 (SPEC F3·D8·D9, §6 `batting_seasons`). 레인 A가 구현한다.

원본은 `collect.statsapi.stats_splits`와 `collect.savant.load_savant`로만 읽는다.
"""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from mlbproj.collect.savant import load_savant
from mlbproj.collect.statsapi import stats_splits
from mlbproj.contracts import (
    BATTER_EXCLUDED_POSITIONS,
    BATTING_SEASONS,
    LEAGUE,
    SEASONS,
    SHORT_SEASON,
    SHORT_SEASON_SCALE,
    Column,
    check_frame,
    raw_savant_path,
)

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

# 비율 열. API stat 필드 이름과 정제 열 이름이 같다
_RATE_COLUMNS: tuple[str, ...] = ("obp", "slg", "ops")
# 비율 자리표시(".---"·"-.--")와 빈 값은 점·빼기 기호만으로 이루어져 있다
_PLACEHOLDER_CHARS = frozenset(".-")
# Statcast 열 (정제 이름). API 행에는 없고 Savant에서 붙인다
_STATCAST_COLUMNS = frozenset(BATTING_SAVANT_COLUMNS.values())
_JOIN_KEYS = ["player_id", "season"]


def _json_int(value: Any, what: str) -> int:
    """JSON 정수를 그대로 돌려준다.

    bool·실수·문자열은 정수로 바꾸지 않고 멈춘다. `astype("int64")`는 3.5를 3으로,
    "3"을 3으로, True를 1로 조용히 바꾸기 때문이다.

    Raises:
        ValueError: 정수가 아닐 때.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what}: 정수가 아니다 ({value!r})")
    return value


def is_batter_split(split: Mapping[str, Any]) -> bool:
    """타격 기록 한 행이 타자 역할인지 (D8, contracts의 역할 판정 주석).

    포지션(`position.abbreviation`)이 `contracts.BATTER_EXCLUDED_POSITIONS`가 아니고
    `stat.plateAppearances` > 0이면 True. 포지션 X·DH·TWP도 타자다.
    """
    position = split["position"]["abbreviation"]
    plate_appearances = _json_int(split["stat"]["plateAppearances"], "plateAppearances")
    return position not in BATTER_EXCLUDED_POSITIONS and plate_appearances > 0


def parse_rate(text: Any) -> float:
    """API의 비율 문자열(".350", "1.000")을 실수로. ".---"·"-.--"·빈 값·None은 NaN."""
    if text is None or (isinstance(text, str) and set(text.strip()) <= _PLACEHOLDER_CHARS):
        return math.nan
    if isinstance(text, bool):
        raise ValueError(f"비율 값이 참·거짓이다 ({text!r})")
    value = float(text)
    # "nan"·"inf" 같은 문자열이 결측·무한대로 조용히 바뀌지 않게 한다 (결측은 자리표시만)
    if not math.isfinite(value):
        raise ValueError(f"비율 값이 유한한 수가 아니다 ({text!r})")
    return value


def _api_row(split: Mapping[str, Any], season: int) -> dict[str, Any]:
    """타자 split 하나를 정제 열(Statcast 제외)의 행으로 바꾼다."""
    stat = split["stat"]
    player_id = split["player"]["id"]
    where = f"{season} 타격 원본의 선수 {player_id}"
    team = split.get("team")
    age = stat.get("age")
    counts = {
        column: _json_int(stat[field], f"{where} {field}")
        for column, field in BATTING_STAT_FIELDS.items()
    }
    is_short = season == SHORT_SEASON
    return {
        "player_id": player_id,
        "season": season,
        "league": LEAGUE,
        "age": None if age is None else _json_int(age, f"{where} age"),
        "team_name": team.get("name") if isinstance(team, dict) else None,
        "n_teams": _json_int(split["numTeams"], f"{where} numTeams"),
        **counts,
        **{column: parse_rate(stat.get(column)) for column in _RATE_COLUMNS},
        # 출전량만 환산한다 (D9)
        "pa_adj": counts["pa"] * SHORT_SEASON_SCALE if is_short else float(counts["pa"]),
        "is_short_season": is_short,
    }


def _savant_batter(raw_dir: Path, season: int) -> pd.DataFrame:
    """Savant 타자 CSV에서 (player_id, season)과 Statcast 열만 정제 이름으로 꺼낸다.

    같은 ID가 두 번 있으면 행 전체 값(Statcast 밖의 열 포함)이 같을 때만 하나로 둔다(E19).

    Raises:
        ValueError: 같은 ID의 행끼리 값이 다를 때.
    """
    df = load_savant(raw_dir, season, "batter").drop_duplicates()
    duplicated = df["player_id"].duplicated(keep=False)
    if bool(duplicated.any()):
        ids = sorted({int(v) for v in df.loc[duplicated, "player_id"]})
        name = raw_savant_path(season, "batter").as_posix()
        raise ValueError(f"{name}: 값이 다른 중복 선수 ID가 있다: {ids}")
    selected = df[["player_id", "year", *BATTING_SAVANT_COLUMNS]]
    return selected.rename(columns={"year": "season", **BATTING_SAVANT_COLUMNS})


def _dtype(col: Column) -> str:
    """계약의 열 → pandas 형식 (`interim.read_table`이 읽어 들이는 형식과 같다)."""
    if col.kind == "int":
        return "Int64" if col.nullable else "int64"
    if col.kind == "float":
        return "float64"
    if col.kind == "bool":
        return "boolean" if col.nullable else "bool"
    # 문자열 열: 모두 결측이어도 object가 아니라 문자열 형식(결측은 NaN)으로 둔다
    return "str"


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
    # 각 행은 그 시즌의 원본만으로 정해진다 (다른 시즌 값이 섞이지 않는다 — N2의 바탕)
    rows = [
        _api_row(split, season)
        for season in SEASONS
        for split in stats_splits(raw_dir, season, "bat")
        if is_batter_split(split)
    ]
    api_columns = [c for c in BATTING_SEASONS.column_names if c not in _STATCAST_COLUMNS]
    api = pd.DataFrame(rows, columns=api_columns)
    savant = pd.concat([_savant_batter(raw_dir, season) for season in SEASONS], ignore_index=True)
    # Savant에만 있는 행(투수의 타격 등)은 버리고, Savant에 없는 API 행은 Statcast 결측 (F3)
    merged = api.merge(savant, how="left", on=_JOIN_KEYS, validate="many_to_one")
    out = merged[BATTING_SEASONS.column_names].astype(
        {col.name: _dtype(col) for col in BATTING_SEASONS.columns}
    )
    out = out.sort_values(["season", "player_id"], kind="mergesort").reset_index(drop=True)
    # API 행이 겹치는 등 계약을 어기면 쓰기 전에 여기서 멈춘다
    problems = check_frame(out, BATTING_SEASONS)
    if problems:
        raise ValueError("; ".join(problems))
    return out
