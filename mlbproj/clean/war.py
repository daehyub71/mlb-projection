"""WAR 정제 표 — SPEC v0.5 F3, D8·D33. 레인 A가 구현한다.

세이버메트릭스 원본을 `contracts.WAR_SEASONS` 표로 만든다. 역할 규칙(D8)은 이미 정제된 시즌 표
(`BATTING_SEASONS`·`PITCHING_SEASONS`)를 기준으로 삼는다 — 그 표에 있는 (선수, 시즌) 행만 WAR 행을 만든다.
"""

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from mlbproj.collect.statsapi import saber_splits
from mlbproj.contracts import (
    LEAGUE,
    ROLES,
    SABER_COLUMN_MAP,
    WAR_SEASONS,
    Column,
    Role,
    check_frame,
    raw_saber_path,
)

_ROLE_LABEL: dict[Role, str] = {"bat": "타격", "pit": "투구"}
# 두 역할의 구성 항목 열 (war 제외). 행마다 결측으로 채운 뒤 그 역할의 열만 값을 넣는다
_COMPONENTS: tuple[str, ...] = tuple(
    col for role in ROLES for src, col in SABER_COLUMN_MAP[role] if src != "war"
)


def _dtype(col: Column) -> str:
    """계약 열 종류에 맞는 pandas 형식 (WAR_SEASONS는 결측 없는 정수·문자열과 실수뿐이다)."""
    if col.kind == "int":
        return "Int64" if col.nullable else "int64"
    if col.kind == "float":
        return "float64"
    return col.kind


def _finite_float(value: Any) -> float | None:
    """유한한 JSON 숫자면 실수로, 아니면 None (bool·문자열·NaN·무한대는 숫자로 보지 않는다)."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _saber_stats(raw_dir: Path, season: int, role: Role) -> dict[int, dict[str, Any]]:
    """세이버메트릭스 원본의 선수 ID → stat. 원본 전체에서 같은 선수가 두 번 나오면 ValueError."""
    where = (
        f"{season} {_ROLE_LABEL[role]} 세이버메트릭스({raw_saber_path(season, role).as_posix()})"
    )
    stats: dict[int, dict[str, Any]] = {}
    for index, split in enumerate(saber_splits(raw_dir, season, role)):
        player = split.get("player")
        pid = player.get("id") if isinstance(player, dict) else None
        if not isinstance(pid, int) or isinstance(pid, bool):
            raise ValueError(f"{where} splits[{index}]에 정수 player.id가 없다 ({pid!r})")
        if pid in stats:
            raise ValueError(f"{where}에 선수 {pid}가 두 번 있다")
        stat = split.get("stat")
        if not isinstance(stat, dict):
            raise ValueError(f"{where}의 선수 {pid} 행에 stat 객체가 없다")
        stats[pid] = stat
    return stats


def _war_row(
    player_id: int, season: int, role: Role, stats: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """시즌 표의 역할 행 하나에 세이버메트릭스 값을 붙인 WAR_SEASONS 행."""
    where = f"{season} {_ROLE_LABEL[role]} 세이버메트릭스의 선수 {player_id}"
    stat = stats.get(player_id)
    if stat is None:
        # 조용히 결측으로 두지 않는다 — F2 ⑦(saber_covers_roles)을 통과했다면 생기지 않는다
        raise ValueError(f"{where} 행이 없다 (시즌 표의 역할 행에는 WAR이 있어야 한다, F2 ⑦)")
    war = _finite_float(stat.get("war"))
    if war is None:
        raise ValueError(f"{where}: war가 유한한 숫자가 아니다 ({stat.get('war')!r})")
    row: dict[str, Any] = {
        "player_id": player_id,
        "season": season,
        "league": LEAGUE,
        "role": role,
        "war": war,
        **dict.fromkeys(_COMPONENTS, math.nan),
    }
    for src, col in SABER_COLUMN_MAP[role]:
        if src == "war":
            continue
        value = stat.get(src)
        if value is None:
            continue  # 구성 항목이 없으면 결측 (필드가 있는지는 F2 ⑦이 본다)
        number = _finite_float(value)
        if number is None:
            raise ValueError(f"{where}: {src}가 유한한 숫자가 아니다 ({value!r})")
        row[col] = number
    return row


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
    tables: dict[Role, pd.DataFrame] = {"bat": batting, "pit": pitching}
    rows: list[dict[str, Any]] = []
    for season in sorted(set(seasons)):
        for role in ROLES:
            # 행이 없는 시즌·역할이라도 요청한 시즌의 원본은 읽는다 (없으면 FileNotFoundError)
            stats = _saber_stats(raw_dir, season, role)
            table = tables[role]
            player_ids = table.loc[table["season"] == season, "player_id"]
            rows += [_war_row(int(pid), season, role, stats) for pid in player_ids]
    frame = pd.DataFrame(rows, columns=WAR_SEASONS.column_names)
    out = frame.astype({col.name: _dtype(col) for col in WAR_SEASONS.columns})
    out = out.sort_values(list(WAR_SEASONS.keys), kind="mergesort").reset_index(drop=True)
    # 시즌 표에 같은 (선수, 시즌)이 두 번 있는 등 계약을 어기면 여기서 멈춘다
    problems = check_frame(out, WAR_SEASONS)
    if problems:
        raise ValueError("; ".join(problems))
    return out
