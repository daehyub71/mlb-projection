"""투수 시즌 표 (SPEC F3·D8·D9·E3, §6 `pitching_seasons`). 레인 B가 구현한다.

원본은 `collect.statsapi.stats_splits`와 `collect.savant.load_savant`로만 읽는다.
"""

import numbers
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api import types as ptypes

from mlbproj.collect.savant import load_savant
from mlbproj.collect.statsapi import stats_splits
from mlbproj.contracts import (
    LEAGUE,
    PITCHER_POSITIONS,
    PITCHING_SEASONS,
    SEASONS,
    SHORT_SEASON,
    SHORT_SEASON_SCALE,
    Column,
)

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

# 이닝 표기: 정수 이닝 + 선택적 소수 한 자리(아웃 0·1·2개)
_INNINGS_PATTERN = re.compile(r"([0-9]+)(?:\.([012]))?")

# API에서 만드는 열 (계약 순서). 나머지는 Savant에서 붙는다
_API_COLUMNS: tuple[str, ...] = tuple(
    name
    for name in PITCHING_SEASONS.column_names
    if name not in set(PITCHING_SAVANT_COLUMNS.values())
)


def ip_to_outs(ip: Any) -> int:
    """이닝 표기를 아웃 수로 바꾼다. "172.1" → 517, "0.2" → 2, "5" → 15, 숫자 172.1도 같다.

    Raises:
        ValueError: 소수 부분이 0·1·2가 아니거나(예: "1.3"), 음수이거나, 숫자가 아닐 때.
    """
    if isinstance(ip, bool):
        raise ValueError(f"이닝 표기가 아니다: {ip!r}")
    if isinstance(ip, numbers.Integral):
        if int(ip) < 0:
            raise ValueError(f"이닝이 음수다: {ip!r}")
        return int(ip) * 3
    if isinstance(ip, numbers.Real):
        # 실수는 가장 짧은 표기(172.1 → "172.1")로 바꿔 문자열과 같은 규칙으로 읽는다.
        # 음수·NaN·무한대는 이 표기에서 걸러진다
        text = repr(float(ip))
    elif isinstance(ip, str):
        text = ip
    else:
        raise ValueError(f"이닝 표기가 아니다: {ip!r}")
    match = _INNINGS_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(f"이닝 표기가 아니다 (소수 부분은 0·1·2, 음수 불가): {ip!r}")
    whole, part = match.groups()
    return int(whole) * 3 + int(part or "0")


def is_pitcher_split(split: Mapping[str, Any]) -> bool:
    """투구 기록 한 행이 투수 역할인지 (D8): 포지션이 `contracts.PITCHER_POSITIONS`(P·TWP)면 True.

    등판한 야수(포지션이 그 밖)는 False.
    """
    return split["position"]["abbreviation"] in PITCHER_POSITIONS


def _require_int(value: Any, where: str) -> int:
    """JSON 정수만 받는다 (bool 제외).

    표를 만들 때 `astype("int64")`는 3.5를 3으로, "3"을 3으로, True를 1로 조용히 바꾼다.
    F2-1은 `outs`·`battersFaced`만 정수인지 보므로 정제에서 모든 정수 필드를 확인한다.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}가 정수가 아니다: {value!r}")
    return value


def _api_row(season: int, split: Mapping[str, Any]) -> dict[str, Any]:
    """투수 split 하나를 정제 표의 API 열로 바꾼다.

    Raises:
        ValueError: 정수 필드가 정수가 아니거나, inningsPitched를 아웃으로 바꾼 값이 outs와
            다를 때. 메시지에 시즌·선수 ID·필드를 적는다.
    """
    stat = split["stat"]
    player_id = _require_int(split["player"]["id"], f"{season} 투구 player.id")
    where = f"{season} 투구 {player_id}"
    counts = {
        column: _require_int(stat[field], f"{where} stat.{field}")
        for column, field in PITCHING_STAT_FIELDS.items()
    }
    innings = stat["inningsPitched"]
    try:
        innings_outs = ip_to_outs(innings)
    except ValueError as err:
        raise ValueError(f"{where}: inningsPitched {innings!r}를 아웃 수로 바꿀 수 없다") from err
    outs = counts["outs"]
    if innings_outs != outs:
        raise ValueError(
            f"{where}: inningsPitched {innings!r}(아웃 {innings_outs})가 outs {outs}와 다르다"
        )
    age = stat.get("age")
    team = split.get("team")
    team_name = team.get("name") if isinstance(team, Mapping) else None
    if team_name is not None and not isinstance(team_name, str):
        raise ValueError(f"{where} team.name이 문자열이 아니다: {team_name!r}")
    return {
        "player_id": player_id,
        "season": season,
        "league": LEAGUE,
        "age": None if age is None else _require_int(age, f"{where} stat.age"),
        "team_name": team_name,
        "n_teams": _require_int(split["numTeams"], f"{where} numTeams"),
        **counts,
        # 출전량 입력만 2020을 162경기로 환산한다 (D9)
        "outs_adj": float(outs) * SHORT_SEASON_SCALE if season == SHORT_SEASON else float(outs),
        "is_short_season": season == SHORT_SEASON,
    }


def _check_unique_api(api: pd.DataFrame) -> None:
    """API 투구 기록은 선수·시즌당 1행이다(E1). 중복이면 ValueError (F2-2: 값이 같아도 실패)."""
    duplicated = api.duplicated(["season", "player_id"], keep=False)
    if bool(duplicated.any()):
        seasons = api.loc[duplicated, "season"].tolist()
        ids = api.loc[duplicated, "player_id"].tolist()
        pairs = sorted(set(zip(seasons, ids, strict=True)))
        raise ValueError(f"API 투구 원본에 같은 선수가 두 번 이상 있다 (시즌, ID): {pairs}")


def _statcast(raw_dir: Path, season: int) -> pd.DataFrame:
    """한 시즌 Savant 투수 CSV에서 (player_id, season) + Statcast 열을 꺼낸다.

    값이 모두 같은 중복 행(E19)은 하나로 합치고(대응하지 않는 열까지 같아야 한다, F2-2와 같은
    규칙), 값이 다른 중복은 ValueError. Statcast 열에 숫자가 아닌 값이 있어도 ValueError —
    빈 칸만 결측이다.
    """
    savant = load_savant(raw_dir, season, "pitcher").drop_duplicates()
    conflicts = savant["player_id"].duplicated(keep=False)
    if bool(conflicts.any()):
        ids = sorted({int(pid) for pid in savant.loc[conflicts, "player_id"]})
        raise ValueError(f"{season} Savant 투수 CSV에 값이 다른 ID 중복이 있다: {ids}")
    for source in PITCHING_SAVANT_COLUMNS:
        column = savant[source]
        if ptypes.is_bool_dtype(column) or not ptypes.is_numeric_dtype(column):
            numbers_only = pd.to_numeric(column, errors="coerce")
            bad = sorted({str(v) for v in column[column.notna() & numbers_only.isna()]})
            raise ValueError(
                f"{season} Savant 투수 CSV의 {source} 열에 숫자가 아닌 값이 있다: {bad[:5]}"
            )
    columns = {"year": "season", **PITCHING_SAVANT_COLUMNS}
    return savant[["player_id", *columns]].rename(columns=columns)


def _dtype(col: Column) -> str:
    """계약 열 종류에 맞는 pandas 형식. 결측을 허용하는 정수는 `Int64`."""
    if col.kind == "int":
        return "Int64" if col.nullable else "int64"
    if col.kind == "float":
        return "float64"
    return col.kind


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
    rows = [
        _api_row(season, split)
        for season in SEASONS
        for split in stats_splits(raw_dir, season, "pit")
        if is_pitcher_split(split)
    ]
    api = pd.DataFrame(rows, columns=list(_API_COLUMNS))
    _check_unique_api(api)
    statcast = pd.concat([_statcast(raw_dir, season) for season in SEASONS], ignore_index=True)
    # Savant에 없는 투수는 Statcast 결측, API 투수가 아닌 Savant 행(등판한 야수)은 버린다
    df = api.merge(statcast, on=["player_id", "season"], how="left", validate="one_to_one")
    df = df.sort_values(["season", "player_id"], kind="mergesort").reset_index(drop=True)
    dtypes = {col.name: _dtype(col) for col in PITCHING_SEASONS.columns}
    return df[PITCHING_SEASONS.column_names].astype(dtypes)
