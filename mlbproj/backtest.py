"""과거 검증 파이프라인 (SPEC F10·D13·N2, PLAN §2-3).

정제 표(`data/interim`)와 파라미터로 목표 시즌·역할마다 세 모델을 만들고 평가한다.

- "league": 비율 = Y−1 리그 평균, 출전량 = 직전 시즌 (기준선)
- "baseline": 비율·출전량 = 직전 시즌 (F6, 기준선)
- "weighted": 비율 = 가중 + 리그 평균 회귀(F7), 출전량 = 구간 모델(F8)

**2026은 여기서 돌리지 않는다** — 최종 검증은 파라미터를 커밋한 뒤 `final-test`로 한 번만(M4).
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mlbproj import contracts as c
from mlbproj.clean.players import people_records
from mlbproj.evaluate import evaluate
from mlbproj.interim import read_table
from mlbproj.models.assemble import assemble_bat, assemble_pit
from mlbproj.models.baseline import baseline_playing_time, baseline_rates, league_model_rates
from mlbproj.models.league import league_batting, league_pitching
from mlbproj.models.params import Params
from mlbproj.models.playing_time import predict_playing_time
from mlbproj.models.weighted import weighted_rates


@dataclass(frozen=True)
class Tables:
    """과거 검증에 쓰는 정제 표 묶음.

    Attributes:
        seasons: 역할 → 정제 시즌 표.
        cohorts: 대상 표 (모든 목표 시즌·역할).
        league: 역할 → 시즌별 리그 평균 표.
        debuts: 선수 ID → MLB 데뷔 연도 (F5, E21).
    """

    seasons: dict[c.Role, pd.DataFrame]
    cohorts: pd.DataFrame
    league: dict[c.Role, pd.DataFrame]
    debuts: dict[int, int]


def load_tables(interim_dir: Path = c.INTERIM_DIR, raw_dir: Path = c.RAW_DIR) -> Tables:
    """정제 표와 리그 평균, 데뷔 연도를 읽는다."""
    batting = read_table(c.BATTING_SEASONS, c.interim_path(c.BATTING_SEASONS, interim_dir))
    pitching = read_table(c.PITCHING_SEASONS, c.interim_path(c.PITCHING_SEASONS, interim_dir))
    cohorts = read_table(c.COHORTS, c.interim_path(c.COHORTS, interim_dir))
    debuts = {
        int(str(p["id"])): int(str(p["mlbDebutDate"])[:4])
        for p in people_records(raw_dir)
        if p.get("mlbDebutDate")
    }
    return Tables(
        seasons={"bat": batting, "pit": pitching},
        cohorts=cohorts,
        league={"bat": league_batting(batting), "pit": league_pitching(pitching)},
        debuts=debuts,
    )


def cohort_for(tables: Tables, role: c.Role, target: int) -> pd.DataFrame:
    """한 목표 시즌·역할의 대상 표 (player_id 순)."""
    cohorts = tables.cohorts
    mask = (cohorts["target_season"] == target) & (cohorts["role"] == role)
    out = cohorts[mask].sort_values("player_id").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{target} {role} 대상 표가 없다 (build를 먼저 돌린다)")
    return out


def _assemble(
    role: c.Role, cohort: pd.DataFrame, rates: pd.DataFrame, pt: pd.DataFrame, model: c.ModelName
) -> pd.DataFrame:
    if role == "bat":
        return assemble_bat(cohort, rates, pt, model)
    return assemble_pit(cohort, rates, pt, model)


def baseline_projections(tables: Tables, role: c.Role, target: int) -> pd.DataFrame:
    """기준선 두 모델("league", "baseline")의 예측 표."""
    cohort = cohort_for(tables, role, target)
    seasons, league = tables.seasons[role], tables.league[role]
    pt = baseline_playing_time(cohort, seasons, role)
    return pd.concat(
        [
            _assemble(role, cohort, league_model_rates(cohort, league, role), pt, "league"),
            _assemble(role, cohort, baseline_rates(cohort, seasons, league, role), pt, "baseline"),
        ],
        ignore_index=True,
    )


def weighted_projection(
    tables: Tables,
    role: c.Role,
    target: int,
    params: Params,
    playing_time: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """가중 모델("weighted")의 예측 표. playing_time을 주면 출전량 학습을 건너뛴다(조정 중 재사용)."""
    cohort = cohort_for(tables, role, target)
    seasons = tables.seasons[role]
    rates = weighted_rates(cohort, seasons, tables.league[role], role, params)
    pt = (
        playing_time
        if playing_time is not None
        else predict_playing_time(cohort, seasons, role, tables.debuts)
    )
    return _assemble(role, cohort, rates, pt, "weighted")


def run_backtest(tables: Tables, params: Params, targets: Iterable[int]) -> pd.DataFrame:
    """목표 시즌들의 평가표 (`contracts.EVALUATIONS`, 키 순).

    Raises:
        ValueError: 목표에 최종 검증 시즌(2026) 이후가 들어 있을 때 — 그건 final-test만 한다.
    """
    rows: list[pd.DataFrame] = []
    for target in targets:
        if target >= c.FINAL_TEST_TARGET:
            raise ValueError(
                f"{target}은 과거 검증으로 돌리지 않는다 — 최종 검증은 final-test로 한 번만 (SPEC F10)"
            )
        for role in c.ROLES:
            projections = pd.concat(
                [
                    baseline_projections(tables, role, target),
                    weighted_projection(tables, role, target, params),
                ],
                ignore_index=True,
            )
            rows.append(
                evaluate(projections, tables.seasons[role], cohort_for(tables, role, target), role)
            )
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(list(c.EVALUATIONS.keys), kind="mergesort").reset_index(drop=True)


def summary(evaluations: pd.DataFrame, group: str = "all") -> pd.DataFrame:
    """한 그룹의 MAE를 지표 × 목표 시즌 × 모델로 펼친 요약."""
    rows = evaluations[evaluations["group"] == group]
    return rows.pivot_table(index=["metric", "target_season"], columns="model", values="mae").round(
        4
    )
