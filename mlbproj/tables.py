"""모델·평가·내보내기가 함께 읽는 정제 표 묶음 (PLAN §2-3).

`backtest`·`tune`·`models.extended`·`final`·`export`가 같은 묶음을 쓴다. 메인 세션만 고친다.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mlbproj import contracts as c
from mlbproj.clean.players import people_records
from mlbproj.cohort import build_cohort
from mlbproj.interim import read_table
from mlbproj.models.league import league_batting, league_pitching


@dataclass(frozen=True)
class Tables:
    """정제 표 묶음.

    Attributes:
        seasons: 역할 → 정제 시즌 표 (`BATTING_SEASONS`·`PITCHING_SEASONS`).
        cohorts: 대상 표 (build가 만든 목표 시즌들, `COHORTS`).
        league: 역할 → 시즌별 리그 평균 표 (`LEAGUE_BAT`·`LEAGUE_PIT`).
        debuts: 선수 ID → MLB 데뷔 연도 (F5, E21).
        players: 선수 표 (`PLAYERS`). 생년월일은 시즌 나이에 쓴다. active·last_mlb_season은
            표시용이라 모델 입력으로 쓰지 않는다(N2).
        fielding: 수비 표 (`FIELDING_SEASONS`, 표시용).
    """

    seasons: dict[c.Role, pd.DataFrame]
    cohorts: pd.DataFrame
    league: dict[c.Role, pd.DataFrame]
    debuts: dict[int, int]
    players: pd.DataFrame = field(default_factory=pd.DataFrame)
    fielding: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_tables(interim_dir: Path = c.INTERIM_DIR, raw_dir: Path = c.RAW_DIR) -> Tables:
    """정제 표와 리그 평균, 데뷔 연도를 읽는다."""
    batting = read_table(c.BATTING_SEASONS, c.interim_path(c.BATTING_SEASONS, interim_dir))
    pitching = read_table(c.PITCHING_SEASONS, c.interim_path(c.PITCHING_SEASONS, interim_dir))
    debuts = {
        int(str(p["id"])): int(str(p["mlbDebutDate"])[:4])
        for p in people_records(raw_dir)
        if p.get("mlbDebutDate")
    }
    return Tables(
        seasons={"bat": batting, "pit": pitching},
        cohorts=read_table(c.COHORTS, c.interim_path(c.COHORTS, interim_dir)),
        league={"bat": league_batting(batting), "pit": league_pitching(pitching)},
        debuts=debuts,
        players=read_table(c.PLAYERS, c.interim_path(c.PLAYERS, interim_dir)),
        fielding=read_table(c.FIELDING_SEASONS, c.interim_path(c.FIELDING_SEASONS, interim_dir)),
    )


def cohort_for(tables: Tables, role: c.Role, target: int) -> pd.DataFrame:
    """한 목표 시즌·역할의 대상 표 (player_id 순).

    build가 만든 대상 표에 있으면 그 행을, 없으면(확장 모델의 학습 목표 2021~2023 등) 같은 규칙
    (`cohort.build_cohort`, 데뷔 연도 보완)으로 바로 만든다.

    Raises:
        ValueError: 그 역할의 정제 시즌 표가 없거나 대상이 0명일 때.
    """
    cohorts = tables.cohorts
    mask = (cohorts["target_season"] == target) & (cohorts["role"] == role)
    out = cohorts[mask]
    if out.empty and role in tables.seasons:
        out = build_cohort(tables.seasons[role], role, target, debut_seasons=tables.debuts)
    if out.empty:
        raise ValueError(f"{target} {role} 대상 표가 없다 (build를 먼저 돌린다)")
    return out.sort_values("player_id", kind="mergesort").reset_index(drop=True)
