"""WAR 검증 · 2026 WAR 시험(한 번) · 웹 war.json — SPEC v0.5 F20·F21·F22·D33~D36·D39. 레인 C가 구현한다.

`models.war`(레인 B)의 함수는 이름과 모양만 계약이다 — 이 모듈의 테스트는 그 함수들을 monkeypatch로
가짜로 바꿔 끼워 레인 B와 동시에 돈다. 기존 `final.py`의 `committed_head`·실행 시각 규칙을 재사용한다.

검증 (F20, `evaluate_war`) — 역할 r, 목표 Y
- metric "war": 예측 = proj.war, 실제 = 목표 Y의 war_seasons war(역할 행 없으면 0), 대상 = proj의 선수 전원.
  기준선 prev = Y−1 war(없으면 0), league = league_rate × proj.pt. 그룹: "all", "seasons_1_2"(n_seasons ≤ 2),
  "seasons_3_5", "war2plus"(실제 war ≥ `WAR_STAR_MIN`).
- metric "war_full": 그룹 "regulars"만 — 목표 Y 실제 출전(`WAR_UNIT`) ≥ `WAR_REGULAR_MIN[r]`이고 Y−1 출전 > 0인 선수.
  실제 = 실제 war / 실제 출전 × proj.full_units, 예측 = proj.war_full,
  prev = Y−1 war / Y−1 출전 × full_units, league = league_rate × full_units.
- n = 그 그룹 선수 수, n = 0이면 mae·기준선 결측. 결과는 `WAR_EVALUATIONS` 열 순서, 키 순.

과거 검증 (`backtest_war`): targets마다 두 역할 `models.war.war_projections` → `evaluate_war`.
`FINAL_TEST_TARGET`(2026) 이상은 **데이터를 읽기 전에** ValueError("final-test-war로 한 번만").

2026 WAR 시험 (`run_final_test_war`, D36) — `final.run_final_test`와 같은 장치, 파일만 다르다
1. `final.committed_head([params_path, war_params_path], repo, git)` — 두 파일이 커밋돼 있어야 한다.
2. 넘긴 params·war_params가 커밋된 파일의 값과 다르면 거부.
3. out_path가 있고 rerun_reason이 없으면 거부("이미 한 번 돌렸다 … --rerun 사유"). 기록이 없는데 사유를 주면 거부.
   다시 돌릴 때 war_params_sha256·params_sha256이 첫 기록과 다르면 거부.
4. 목표 2026을 `evaluate_war`로 평가해 기록(JSON indent=1):
   {"target_season": 2026, "ran_at", "params_sha256", "war_params_sha256", "commit",
    "reruns": [{"ran_at", "reason", "commit"}…], "rows": [WAR_EVALUATIONS 행…(NaN → null)]}
   다시 돌리면 ran_at·commit은 첫 값, reruns에 이번 실행을 붙이고 rows를 바꾼다. 반환은 기록(dict).
   **`contracts.FINAL_TEST_PATH`(기존 여섯 지표 기록)는 읽지도 쓰지도 않는다.**

웹 (`build_war_web`, F21·F22) — `contracts/web-data.schema.json`의 WarFile 모양. 반올림은 `WEB_ROUND`.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mlbproj.contracts import TUNE_TARGETS, Role
from mlbproj.final import GitRunner
from mlbproj.models.params import Params
from mlbproj.models.war import WarParams
from mlbproj.tables import Tables


class WarTestError(RuntimeError):
    """2026 WAR 시험을 돌릴 수 없을 때 (커밋 안 됨 · 값 불일치 · 이미 돌림 · 사유 문제)."""


def actual_war(
    war_seasons: pd.DataFrame, seasons: pd.DataFrame, role: Role, season: int
) -> pd.DataFrame:
    """그 시즌 역할 실제 WAR과 출전. 열: player_id, war, units(`WAR_UNIT[role]`). 시즌 표에 행이 있는 선수만, player_id 순."""
    raise NotImplementedError


def evaluate_war(
    proj: pd.DataFrame,
    war_seasons: pd.DataFrame,
    seasons: pd.DataFrame,
    role: Role,
    target: int,
    league_rate: float,
) -> pd.DataFrame:
    """한 역할·한 목표의 WAR 검증 행 (모듈 docstring "검증") → `WAR_EVALUATIONS`.

    Raises:
        ValueError: proj의 목표·역할이 인자와 다를 때.
    """
    raise NotImplementedError


def backtest_war(
    tables: Tables,
    war_seasons: pd.DataFrame,
    params: Params,
    war_params: WarParams,
    targets: Iterable[int] = TUNE_TARGETS,
) -> pd.DataFrame:
    """과거 검증표 (`WAR_EVALUATIONS`, 두 역할 × targets). 2026 이상은 읽기 전에 ValueError."""
    raise NotImplementedError


def run_final_test_war(
    tables: Tables,
    war_seasons: pd.DataFrame,
    params: Params,
    war_params: WarParams,
    *,
    repo: Path,
    params_path: Path,
    war_params_path: Path,
    out_path: Path,
    now: str,
    rerun_reason: str | None = None,
    git: GitRunner | None = None,
) -> dict[str, Any]:
    """2026 WAR 시험을 한 번 돌리고 기록한다 (모듈 docstring "2026 WAR 시험").

    Raises:
        WarTestError: 위 거부 조건.
        ValueError: now가 "YYYY-MM-DDTHH:MM:SSZ"가 아닐 때.
    """
    raise NotImplementedError


@dataclass(frozen=True)
class WarExportInput:
    """war.json을 만드는 데 필요한 것.

    Attributes:
        meta: 다른 웹 파일과 같은 Meta(dict, 스키마 Meta 모양 — 기존 내보내기에서 받는다).
        rankings: 목표 2027 `WAR_RANKINGS`.
        projections: 역할 → 목표 2027 `WAR_PROJECTIONS`(pt·slot을 읽는다).
        cohorts: 역할 → 목표 2027 cohort(n_seasons·played_prev).
        players: `PLAYERS` 표(이름·포지션·active·last_mlb_season).
        player_summaries: player_id → 스키마 PlayerSummary(dict) — 기존 내보내기와 같은 값(나이·팀 포함).
        war_seasons: `WAR_SEASONS`(history — 입력 기간 5시즌).
        evaluations: `WAR_EVALUATIONS`(2024·2025 과거 검증 + 있으면 2026 시험 rows).
        final_test: 2026 WAR 시험 기록(dict) 또는 None.
        war_params_sha256: `config/war_v1.json`의 sha256.
    """

    meta: dict[str, Any]
    rankings: pd.DataFrame
    projections: dict[Role, pd.DataFrame]
    cohorts: dict[Role, pd.DataFrame]
    players: pd.DataFrame
    player_summaries: dict[int, dict[str, Any]]
    war_seasons: pd.DataFrame
    evaluations: pd.DataFrame
    final_test: dict[str, Any] | None
    war_params_sha256: str


def build_war_web(inp: WarExportInput) -> dict[str, Any]:
    """war.json 내용(WarFile). rows는 rankings 순서 그대로.

    - pa = 타자 pt, ip = 투수 pt / 3(소수 이닝), slot = 투수 slot(투수 역할이 없으면 null).
    - group = 두 역할 n_seasons 중 큰 값으로 seasons_1_2(≤2) / seasons_3_5. played_prev = 어느 역할이든 Y−1 기록.
    - history = 입력 기간(2022~2026) 시즌 중 그 선수 WAR 행이 있는 시즌, 오름차순 {season, war_bat, war_pit}.
    - final_test = 기록이 있으면 {"target_season", "ran_at", "war_params_sha256", "reruns": [{"ran_at", "reason"}]}.
    - full_time = {"pa": 600, "sp_ip": 180, "rp_ip": 65}, source = `WAR_SOURCE`.
    """
    raise NotImplementedError


def write_war_csv(inp: WarExportInput, path: Path) -> None:
    """검토용 CSV(`WAR_PROJECTION_CSV`): rankings 열 + name, 같은 입력이면 같은 바이트."""
    raise NotImplementedError
