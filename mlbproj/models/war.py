"""WAR 예측 — SPEC v0.5 F18·F19·D34·D35·D36, PLAN v0.3 §2-3 "WAR 모델". 레인 B가 구현한다.

기존 여섯 지표 모델 위에 얹기만 한다. `config/params_v1.json`(시즌 가중치)과 기존 출전량 모델
(`models.playing_time.predict_playing_time`)을 그대로 쓰고, WAR 파라미터는 `config/war_v1.json` 하나다.

비율 (F18) — 역할 r, 목표 Y, 출전 단위 u = `WAR_UNIT[r]`(타자 pa, 투수 outs — 2020도 실제 값, 환산 안 함)
    war_rate = (Σ wₖ·warₖ + R·lg) / (Σ wₖ·uₖ + R)
    k = 입력 기간 시즌(cohort의 window_first ~ window_last). 가중치 wₖ = params.<r>.weights[Y−1−season]
    (weights[0]이 Y−1). 기간 안 미출전 시즌은 행이 없으니 더하지 않는다. lg = `league_war_rate(…, Y−1)`.
예측 (F19)
    war = war_rate × pt,  pt = 예측 출전량(타자 PA, 투수 아웃 — `predict_playing_time` 결과 그대로)
    war_full = war_rate × FULL_TIME_UNITS[slot],  slot = 타자 "bat", 투수 `pitcher_slots`("sp"·"rp")
투타 합산 (D34) — `combine_roles`: 선수마다 한 행, 결측 역할은 0으로 더한다.
누설 방지 (N2): 목표 Y의 계산에 season ≥ Y 행을 쓰지 않는다. 테스트로 강제한다(미래 행을 바꿔도 결과 비트 동일).
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mlbproj.contracts import TUNE_TARGETS, WAR_TUNE_GRID, Role
from mlbproj.models.params import Params
from mlbproj.tables import Tables


@dataclass(frozen=True)
class WarParams:
    """WAR 파라미터 (`config/war_v1.json`).

    Attributes:
        version: 이름 (예: "v1").
        regress: 역할 → 회귀 표본 R (타자 타석, 투수 아웃). 두 역할 모두 있어야 하고 양수.
    """

    version: str
    regress: dict[Role, float]


DEFAULT_WAR_PARAMS = WarParams(version="v0", regress={"bat": 1200.0, "pit": 600.0})


def war_params_to_json(p: WarParams) -> str:
    """정규화된 JSON 문자열 (키 정렬·indent=2·끝 줄바꿈). 같은 값이면 같은 문자열 — sha256의 기준."""
    raise NotImplementedError


def war_params_from_json(text: str) -> WarParams:
    """`war_params_to_json`의 역.

    Raises:
        ValueError: 키가 빠졌거나 남거나, 역할이 bat·pit 둘이 아니거나, R이 양의 유한수가 아닐 때.
    """
    raise NotImplementedError


def war_params_sha256(p: WarParams) -> str:
    """`war_params_to_json(p)`의 UTF-8 sha256 (소문자 16진 64자)."""
    raise NotImplementedError


def save_war_params(p: WarParams, path: Path) -> None:
    """파일로 쓴다(부모 디렉토리를 만든다)."""
    raise NotImplementedError


def load_war_params(path: Path) -> WarParams:
    """파일에서 읽는다. 형식 오류는 ValueError."""
    raise NotImplementedError


def league_war_rate(
    war_seasons: pd.DataFrame, seasons: pd.DataFrame, role: Role, season: int
) -> float:
    """리그 WAR 비율 = 그 시즌 역할 WAR 합 / 출전 합 (F18의 lg).

    war_seasons(`WAR_SEASONS`)의 (role, season) 행 war 합을 seasons(그 역할 정제 시즌 표)의 같은 시즌
    `WAR_UNIT[role]` 합으로 나눈다.

    Raises:
        ValueError: 그 시즌 출전 합이 0이거나 행이 없을 때.
    """
    raise NotImplementedError


def war_rates(
    cohort: pd.DataFrame,
    seasons: pd.DataFrame,
    war_seasons: pd.DataFrame,
    role: Role,
    weights: tuple[float, ...],
    regress: float,
    league_rate: float,
) -> pd.DataFrame:
    """대상(cohort — 한 역할·한 목표)의 WAR 비율 (F18). 열: player_id, war_rate. player_id 순.

    Raises:
        ValueError: cohort가 여러 목표·역할을 섞었거나, 기간 안 시즌 행에 맞는 WAR 행이 없을 때(어느 선수인지).
    """
    raise NotImplementedError


def pitcher_slots(
    cohort: pd.DataFrame, pit_seasons: pd.DataFrame, weights: tuple[float, ...]
) -> pd.DataFrame:
    """투수 자리 (D35). 열: player_id, slot. 입력 기간 Σw·gs / Σw·g ≥ `STARTER_SHARE_MIN`이면 "sp", 아니면 "rp"
    (Σw·g = 0이면 "rp"). 가중치는 `war_rates`와 같은 규칙. player_id 순."""
    raise NotImplementedError


def project_war(
    cohort: pd.DataFrame,
    rates: pd.DataFrame,
    playing_time: pd.DataFrame,
    role: Role,
    slots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """예측 WAR·풀타임 환산 (F19) → `WAR_PROJECTIONS` 표(열 순서·키 순).

    playing_time은 `BAT_PLAYING_TIME`(pa) 또는 `PIT_PLAYING_TIME`(outs). 타자는 slot "bat"(slots 무시),
    투수는 slots가 필요하다. n_seasons는 cohort 그대로.

    Raises:
        ValueError: 세 표의 선수 집합이 다르거나, 투수인데 slots가 없을 때.
    """
    raise NotImplementedError


def war_projections(
    tables: Tables,
    war_seasons: pd.DataFrame,
    params: Params,
    war_params: WarParams,
    role: Role,
    target: int,
    playing_time: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """한 역할·한 목표의 `WAR_PROJECTIONS`. 순서: `tables.cohort_for` → `league_war_rate(Y−1)` →
    `war_rates` → (투수) `pitcher_slots` → 출전량(playing_time이 없으면
    `predict_playing_time(cohort, seasons, role, tables.debuts)`) → `project_war`."""
    raise NotImplementedError


def combine_roles(
    bat: pd.DataFrame | None, pit: pd.DataFrame | None, war_seasons: pd.DataFrame, target: int
) -> pd.DataFrame:
    """두 역할 예측(`WAR_PROJECTIONS`, 같은 목표)을 투타 합산 순위 표 `WAR_RANKINGS`로 (D34).

    roles = "both"(두 역할 다) / "bat" / "pit". war = war_bat + war_pit(결측 0), war_full도 같다.
    prev_war = war_seasons에서 목표 Y−1의 그 선수 행(두 역할) war 합, 행이 없으면 결측.
    정렬: war 내림차순, 같으면 player_id 오름차순.

    Raises:
        ValueError: 두 입력의 목표 시즌이 target과 다를 때.
    """
    raise NotImplementedError


def tune_war(
    tables: Tables,
    war_seasons: pd.DataFrame,
    params: Params,
    targets: Iterable[int] = TUNE_TARGETS,
    grid: Mapping[Role, tuple[float, ...]] = WAR_TUNE_GRID,
) -> tuple[WarParams, pd.DataFrame]:
    """WAR 회귀 표본 조정 (D36). 역할마다 grid의 R 후보로 targets 예측 WAR의 MAE를 내고(대상 전원,
    목표 시즌 실제 WAR — 역할 행이 없으면 0), 목표 평균 MAE가 가장 작은 R을 고른다(같으면 작은 R).
    출전량은 목표마다 한 번만 예측해 후보 사이에 재사용한다.

    Returns:
        (WarParams(version="v1", regress=고른 값), 보고 표 — 열: role, regress, target_season, mae,
         mean_mae, chosen(bool). role·regress·target_season 순.)

    Raises:
        ValueError: targets에 `FINAL_TEST_TARGET`(2026) 이상이 있으면 **데이터를 읽기 전에**.
    """
    raise NotImplementedError
