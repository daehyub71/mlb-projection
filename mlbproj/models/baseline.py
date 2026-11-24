"""기준 모델 — 직전 시즌과 리그 평균 (SPEC F6, D13 기준선). 레인 A가 구현한다.

모두 목표 시즌 Y의 대상 표(`contracts.COHORTS`) 한 장과 같은 역할의 정제 표, 리그 평균 표를 받는다.
**Y 이후(season ≥ Y) 행은 쓰지 않는다**(N2). 결과의 선수 집합은 대상 표와 정확히 같고 player_id 순이다.

비율 공식은 `models.league` 모듈 docstring과 같다(한 선수·한 시즌의 합으로 계산). 분모(pa·ab·bf·outs)가
0이면 그 비율은 리그 평균(Y−1)으로 채운다 — 결과에 결측이 없다.
"""

import pandas as pd

from mlbproj.contracts import (
    BAT_PLAYING_TIME,
    BAT_RATES,
    BATTING_SEASONS,
    COHORTS,
    LEAGUE_BAT,
    LEAGUE_PIT,
    PIT_PLAYING_TIME,
    PIT_RATES,
    PITCHING_SEASONS,
    Role,
    TableSpec,
)
from mlbproj.models.league import _bat_rates, _pit_rates, _require, league_row

_SEASON_SPECS: dict[Role, TableSpec] = {"bat": BATTING_SEASONS, "pit": PITCHING_SEASONS}
_LEAGUE_SPECS: dict[Role, TableSpec] = {"bat": LEAGUE_BAT, "pit": LEAGUE_PIT}
_RATE_SPECS: dict[Role, TableSpec] = {"bat": BAT_RATES, "pit": PIT_RATES}
_PLAYING_TIME_SPECS: dict[Role, TableSpec] = {"bat": BAT_PLAYING_TIME, "pit": PIT_PLAYING_TIME}
# 출전량 예측 열 ← Y−1 정제 표의 환산 출전량 열 (D9)
_PLAYING_TIME_SOURCE: dict[Role, tuple[str, str]] = {
    "bat": ("pa", "pa_adj"),
    "pit": ("outs", "outs_adj"),
}
# 리그 평균 기준선 열 ← 리그 평균 표의 열
_LEAGUE_COLUMNS: dict[Role, dict[str, str]] = {
    "bat": {"lg_ops": "ops"},
    "pit": {"lg_era": "era", "lg_k_bb_pct": "k_bb_pct"},
}
_SHOW_MAX = 5  # 오류 메시지에 적는 선수 ID의 최대 개수


def _rate_columns(role: Role) -> list[str]:
    """비율 표의 예측 열 (player_id·target_season·lg_*를 뺀 나머지, 계약 순서)."""
    skip = {"player_id", "target_season", *_LEAGUE_COLUMNS[role]}
    return [name for name in _RATE_SPECS[role].column_names if name not in skip]


def _target_season(cohort: pd.DataFrame, role: Role) -> int:
    """대상 표를 확인하고 목표 시즌 Y를 돌려준다. 계약·역할·목표 시즌 하나를 어기면 ValueError."""
    _require(cohort, COHORTS, "대상 표")
    roles = sorted(set(cohort["role"]))
    if roles != [role]:
        raise ValueError(f"대상 표의 역할이 {role} 하나가 아니다: {roles}")
    targets = sorted({int(t) for t in cohort["target_season"]})
    if len(targets) != 1:
        raise ValueError(f"대상 표의 목표 시즌이 하나가 아니다: {targets}")
    return targets[0]


def _player_index(cohort: pd.DataFrame) -> pd.Index:
    """결과의 행 = 대상 선수 (player_id 순)."""
    return pd.Index(sorted(int(i) for i in cohort["player_id"]), name="player_id", dtype="int64")


def _prev_season(seasons: pd.DataFrame, target: int, index: pd.Index) -> pd.DataFrame:
    """대상 선수의 Y−1 행 (index 순). 행이 없는 선수는 모두 결측. Y 이후 행은 보지 않는다(N2).

    한 선수의 Y−1 행이 둘 이상이면(리그가 여럿) ValueError — v0.1은 MLB만 다룬다(D1).
    """
    prev = seasons[seasons["season"] == target - 1]
    repeated = prev.loc[prev["player_id"].duplicated(), "player_id"]
    if not repeated.empty:
        ids = sorted({int(i) for i in repeated})
        raise ValueError(
            f"{target - 1} 시즌에 행이 둘 이상인 선수가 있다 (리그 여럿?): {ids[:_SHOW_MAX]}"
        )
    return prev.set_index("player_id").reindex(index)


def _finish(rates: pd.DataFrame, lg: pd.Series, target: int, role: Role) -> pd.DataFrame:
    """비율 예측에 리그 평균 기준선·목표 시즌을 붙여 계약 순서로 돌려준다."""
    out = rates.copy()
    for column, source in _LEAGUE_COLUMNS[role].items():
        out[column] = float(lg[source])
    out["target_season"] = target
    return out.reset_index()[_RATE_SPECS[role].column_names]


def baseline_rates(
    cohort: pd.DataFrame, seasons: pd.DataFrame, league: pd.DataFrame, role: Role
) -> pd.DataFrame:
    """직전 시즌 비율 (F6). 타자 `BAT_RATES`, 투수 `PIT_RATES` 형식.

    - Y−1에 행이 있으면 그 시즌의 실제 성분·비율(r_*, obp·slg·ops·hr_per_pa / er_per_out·era·k_bb_pct),
      없으면 Y−1 리그 평균.
    - lg_ops / lg_era·lg_k_bb_pct = Y−1 리그 평균. target_season = Y.

    Raises:
        ValueError: 입력 표가 계약을 어기거나 대상 표의 역할·목표 시즌이 하나가 아닐 때.
        KeyError: 리그 평균에 Y−1이 없을 때.
    """
    target = _target_season(cohort, role)
    _require(seasons, _SEASON_SPECS[role], f"{role} 정제 표")
    _require(league, _LEAGUE_SPECS[role], f"{role} 리그 평균 표")
    lg = league_row(league, target - 1)
    columns = _rate_columns(role)
    # 정제 표의 obp·slg·ops 열이 아니라 개수 열로 계산한다 (리그 평균과 같은 공식, 2020도 실제 표본 — D9)
    prev = _prev_season(seasons, target, _player_index(cohort))
    own = _bat_rates(prev) if role == "bat" else _pit_rates(prev)
    # 행이 없거나 분모가 0이라 결측인 비율은 Y−1 리그 평균
    rates = own[columns].fillna(lg[columns])
    if role == "bat":
        # obp·slg 중 하나만 리그로 채워졌어도 ops = obp + slg를 지킨다
        rates["ops"] = rates["obp"] + rates["slg"]
    return _finish(rates, lg, target, role)


def league_model_rates(cohort: pd.DataFrame, league: pd.DataFrame, role: Role) -> pd.DataFrame:
    """리그 평균 모델의 비율 — 모든 대상에게 Y−1 리그 평균 (D13 기준선, model "league")."""
    target = _target_season(cohort, role)
    _require(league, _LEAGUE_SPECS[role], f"{role} 리그 평균 표")
    lg = league_row(league, target - 1)
    index = _player_index(cohort)
    rates = pd.DataFrame({name: float(lg[name]) for name in _rate_columns(role)}, index=index)
    return _finish(rates, lg, target, role)


def baseline_playing_time(cohort: pd.DataFrame, seasons: pd.DataFrame, role: Role) -> pd.DataFrame:
    """직전 시즌 출전량 (F6). 타자 `BAT_PLAYING_TIME`(pa = Y−1 pa_adj), 투수 `PIT_PLAYING_TIME`
    (outs = Y−1 outs_adj). Y−1에 행이 없으면 0.
    """
    target = _target_season(cohort, role)
    _require(seasons, _SEASON_SPECS[role], f"{role} 정제 표")
    column, source = _PLAYING_TIME_SOURCE[role]
    prev = _prev_season(seasons, target, _player_index(cohort))
    out = pd.DataFrame({column: prev[source].fillna(0.0).astype("float64")})
    out["target_season"] = target
    return out.reset_index()[_PLAYING_TIME_SPECS[role].column_names]
