"""WAR 예측(models/war.py) 테스트 (SPEC v0.5 F18·F19·D34·D35·D36·N2, PLAN §2-3 "WAR 모델").

입력은 모두 이 파일에서 만든 작은 합성 표다.
- 정제 시즌 표: `_seasons`가 `BATTING_SEASONS`·`PITCHING_SEASONS`의 모든 열을 채운다. 의미가 있는 값은
  출전량(타자 pa, 투수 outs)과 선발 판정용 g·gs뿐이다. 2020의 pa_adj·outs_adj는 162/60배라(D9),
  모델이 환산값을 쓰면 기대값과 어긋난다.
- WAR 표: `_war_seasons`가 `WAR_SEASONS` 형식으로 만든다. 구성 항목 열(war_batting…·war_ra9)은
  엉뚱한 값(`DECOY`)이라, 모델이 war 대신 이 열을 쓰면 기대값과 어긋난다.
- 대상 표: `cohort.build_cohort`(M2 완료)로 만든다.

규칙 (war.py docstring) — 목표 Y, 역할 r, 출전 단위 u(타자 pa, 투수 outs — 2020도 실제 값)
    war_rate = (Σ wₖ·warₖ + R·lg) / (Σ wₖ·uₖ + R),  wₖ = weights[Y−1−season]  (weights[0] = Y−1)
기대값은 `HAND`에 (Σw·war, Σw·u)로, 리그 비율은 `LEAGUE_SUMS`에 (WAR 합, 출전 합)으로 손으로 계산해 적었다.
식은 각 항목 주석에 있다. 목표 시즌 2026은 어떤 데이터로도 돌리지 않는다(2026을 거부하는 동작만 본다).
"""

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlbproj import contracts as c
from mlbproj.cohort import build_cohort
from mlbproj.contracts import Role
from mlbproj.models import playing_time as playing_time_module
from mlbproj.models import war as war_module
from mlbproj.models.params import DEFAULT_PARAMS, Params, RateParams, validate_params
from mlbproj.models.war import (
    DEFAULT_WAR_PARAMS,
    WarParams,
    combine_roles,
    league_war_rate,
    load_war_params,
    pitcher_slots,
    project_war,
    save_war_params,
    tune_war,
    war_params_from_json,
    war_params_sha256,
    war_params_to_json,
    war_projections,
    war_rates,
)
from mlbproj.tables import Tables

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

TARGET = 2025  # 기본 목표 시즌 Y — 입력 기간 2020~2024, 리그 비율은 2024
BAT_WEIGHTS: tuple[float, ...] = (8.0, 4.0, 2.0, 1.0, 0.5)  # [Y−1, Y−2, Y−3, Y−4, Y−5]
PIT_WEIGHTS: tuple[float, ...] = (5.0, 4.0, 3.0, 2.0, 1.0)
BAT_R = 100.0  # 회귀 표본 (타석)
PIT_R = 50.0  # 회귀 표본 (아웃)
TWO_WAY = 1001  # 타자·투수 둘 다 (오타니 사례, F12·D34)
NEW_BAT = 1999  # 목표 시즌 이후에만 기록이 있는 선수 (N2)
NEW_PIT = 2999
POISON_UNIT = 9_999  # 목표 시즌 이후 행에 넣는 "독" 출전량
POISON_WAR = 99.0  # 목표 시즌 이후 행에 넣는 "독" WAR
DECOY = 7.0  # WAR 구성 항목 열의 값 — 모델이 쓰면 안 된다
NAN = float("nan")
REL = 1e-12
ABS = 1e-15

# 타자: player_id → {season: (pa, war)}
BAT_WORLD: dict[int, dict[int, tuple[int, float]]] = {
    # 투타 겸업. 2022 데뷔 → 기간 2022~2024. 2025·2026은 목표 이후
    TWO_WAY: {
        2022: (500, 3.0),
        2023: (400, 1.0),
        2024: (600, 4.0),
        2025: (650, 5.0),
        2026: (620, 4.5),
    },
    # 2019는 Y=2025의 기간(2020~2024) 밖, 2020은 단축 시즌(pa_adj = 540), 2021~2023 결장
    1002: {2019: (100, 0.2), 2020: (200, 1.5), 2024: (300, 0.5), 2025: (400, 1.2)},
    # 2023 한 시즌, WAR 음수, 2024 이후 미출전
    1003: {2023: (50, -0.4)},
}

# 투수: player_id → {season: (g, gs, outs, war)}
PIT_WORLD: dict[int, dict[int, tuple[int, int, int, float]]] = {
    TWO_WAY: {2024: (10, 10, 150, 1.5), 2025: (12, 12, 200, 1.8)},
    # 선발. 2019는 Y=2025의 기간 밖(불펜 100경기 — 섞이면 자리가 바뀐다)
    2001: {
        2019: (100, 0, 300, 9.0),
        2021: (30, 30, 540, 3.0),
        2022: (28, 28, 500, 2.5),
        2023: (20, 15, 300, 1.0),
        2024: (32, 32, 600, 4.0),
        2025: (33, 33, 570, 5.0),
    },
    # 불펜. 2020은 단축 시즌(outs_adj = 162)
    2002: {
        2020: (25, 0, 60, 0.3),
        2023: (65, 0, 195, 0.8),
        2024: (60, 0, 180, 0.6),
        2025: (62, 0, 200, 0.9),
    },
    # 스윙맨 — 가중하면 선발(Σw·gs/Σw·g = 150/290), 같은 가중이면 불펜(30/65)
    2003: {2023: (35, 0, 210, 0.4), 2024: (30, 30, 480, 2.2)},
    # 경계 — 선발 비중이 정확히 0.5 (≥ 이면 선발)
    2004: {2024: (20, 10, 300, 1.0)},
    # 2020 한 시즌(Y=2025의 Y−5) — 그 가중치가 0이면 Σw·g = 0
    2005: {2020: (12, 12, 200, 1.2)},
}

# (역할, 목표) → {player_id: (Σw·war, Σw·u)} — 기간 안 출전 시즌만, wₖ = weights[Y−1−season]
HAND: dict[tuple[Role, int], dict[int, tuple[float, float]]] = {
    # 타자 Y=2024 (기간 2019~2023, w: 2023=8, 2022=4, 2021=2, 2020=1, 2019=0.5)
    ("bat", 2024): {
        TWO_WAY: (20.0, 5200.0),  # 8·1.0 + 4·3.0 / 8·400 + 4·500
        1002: (1.6, 250.0),  # 0.5·0.2 + 1·1.5 / 0.5·100 + 1·200
        1003: (-3.2, 400.0),  # 8·(−0.4) / 8·50
    },
    # 타자 Y=2025 (기간 2020~2024, w: 2024=8, 2023=4, 2022=2, 2021=1, 2020=0.5)
    ("bat", 2025): {
        TWO_WAY: (42.0, 7400.0),  # 8·4.0 + 4·1.0 + 2·3.0 / 8·600 + 4·400 + 2·500
        1002: (4.75, 2500.0),  # 8·0.5 + 0.5·1.5 / 8·300 + 0.5·200 (2019 제외, 2020은 pa 200)
        1003: (-1.6, 200.0),  # 4·(−0.4) / 4·50
    },
    # 투수 Y=2024 (기간 2019~2023, w: 2023=5, 2022=4, 2021=3, 2020=2, 2019=1)
    ("pit", 2024): {
        2001: (33.0, 5420.0),  # 1·9.0 + 3·3.0 + 4·2.5 + 5·1.0 / 1·300 + 3·540 + 4·500 + 5·300
        2002: (4.6, 1095.0),  # 2·0.3 + 5·0.8 / 2·60 + 5·195
        2003: (2.0, 1050.0),  # 5·0.4 / 5·210
        2005: (2.4, 400.0),  # 2·1.2 / 2·200
    },
    # 투수 Y=2025 (기간 2020~2024, w: 2024=5, 2023=4, 2022=3, 2021=2, 2020=1)
    ("pit", 2025): {
        TWO_WAY: (7.5, 750.0),  # 5·1.5 / 5·150
        2001: (37.5, 6780.0),  # 5·4.0 + 4·1.0 + 3·2.5 + 2·3.0 / 5·600 + 4·300 + 3·500 + 2·540
        2002: (6.5, 1740.0),  # 5·0.6 + 4·0.8 + 1·0.3 / 5·180 + 4·195 + 1·60 (2020은 outs 60)
        2003: (12.6, 3240.0),  # 5·2.2 + 4·0.4 / 5·480 + 4·210
        2004: (5.0, 1500.0),  # 5·1.0 / 5·300
        2005: (1.2, 200.0),  # 1·1.2 / 1·200
    },
}

# (역할, 시즌) → (그 시즌 역할 WAR 합, 출전 합) — 리그 비율 = WAR 합 / 출전 합
LEAGUE_SUMS: dict[tuple[Role, int], tuple[float, float]] = {
    ("bat", 2020): (1.5, 200.0),  # 1002 — pa 200 (pa_adj 540 아님)
    ("bat", 2023): (0.6, 450.0),  # 1001 1.0 + 1003 −0.4 / 400 + 50
    ("bat", 2024): (4.5, 900.0),  # 1001 4.0 + 1002 0.5 / 600 + 300 (1001 투수 WAR 1.5 제외)
    ("pit", 2020): (1.5, 260.0),  # 2002 0.3 + 2005 1.2 / 60 + 200 (outs_adj 아님)
    ("pit", 2023): (2.2, 705.0),  # 2001 1.0 + 2002 0.8 + 2003 0.4 / 300 + 195 + 210
    ("pit", 2024): (9.3, 1710.0),  # 1.5 + 4.0 + 0.6 + 2.2 + 1.0 / 150 + 600 + 180 + 480 + 300
}

# Y=2025 대상의 n_seasons (cohort 그대로)
N_SEASONS_2025: dict[Role, dict[int, int]] = {
    "bat": {TWO_WAY: 3, 1002: 2, 1003: 1},
    "pit": {TWO_WAY: 1, 2001: 4, 2002: 3, 2003: 2, 2004: 1, 2005: 1},
}

# Y=2025 투수 자리 (PIT_WEIGHTS) — Σw·gs / Σw·g
SLOTS_2025: dict[int, str] = {
    TWO_WAY: "sp",  # 50/50
    2001: "sp",  # 364/384 (기간 밖 2019 불펜 행 제외)
    2002: "rp",  # 0
    2003: "sp",  # 5·30 / (5·30 + 4·35) = 150/290 ≈ .517
    2004: "sp",  # 5·10 / 5·20 = .5 (경계, ≥)
    2005: "sp",  # 12/12
}

# 주입하는 예측 출전량 (타자 PA, 투수 아웃) — 목표와 무관하게 같은 값
PT: dict[Role, dict[int, float]] = {
    "bat": {TWO_WAY: 550.0, 1002: 300.0, 1003: 100.0},
    "pit": {TWO_WAY: 300.0, 2001: 540.0, 2002: 190.0, 2003: 420.0, 2004: 240.0, 2005: 60.0},
}

PARAMS = Params(
    version="test",
    bat=RateParams(weights=BAT_WEIGHTS, regress=dict(DEFAULT_PARAMS.bat.regress)),
    pit=RateParams(weights=PIT_WEIGHTS, regress=dict(DEFAULT_PARAMS.pit.regress)),
    era_method="er",
)
WAR_PARAMS = WarParams(version="test", regress={"bat": BAT_R, "pit": PIT_R})
WEIGHTS: dict[Role, tuple[float, ...]] = {"bat": BAT_WEIGHTS, "pit": PIT_WEIGHTS}
REGRESS: dict[Role, float] = {"bat": BAT_R, "pit": PIT_R}

SPECS: dict[Role, c.TableSpec] = {"bat": c.BATTING_SEASONS, "pit": c.PITCHING_SEASONS}
PT_SPECS: dict[Role, c.TableSpec] = {"bat": c.BAT_PLAYING_TIME, "pit": c.PIT_PLAYING_TIME}
PT_COLUMN: dict[Role, str] = {"bat": "pa", "pit": "outs"}
WAR_COMPONENT_COLUMNS: tuple[str, ...] = (
    "war_batting",
    "war_baserunning",
    "war_fielding",
    "war_positional",
    "war_league",
    "war_replacement",
)

# tune_war 격자 (일부러 순서를 섞었다). 손 계산 결과(`_oracle_mae`): 타자 두 해 평균 MAE
# 300 → .8990, 3000 → .8263, 10000 → .9391 · 투수 200 → .8160, 2000 → .7736, 10000 → .8531.
# 목표별 최솟값(타자 2025 = 300, 투수 2024 = 200)과 평균 최솟값이 다르다 — 평균으로 골라야 한다.
TUNE_GRID: dict[Role, tuple[float, ...]] = {
    "bat": (10000.0, 3000.0, 300.0),
    "pit": (200.0, 2000.0, 10000.0),
}
TUNE_CHOSEN: dict[Role, float] = {"bat": 3000.0, "pit": 2000.0}
REPORT_COLUMNS = ["role", "regress", "target_season", "mae", "mean_mae", "chosen"]

# ---------------------------------------------------------------------------
# 도우미 — 계약 형식의 합성 표
# ---------------------------------------------------------------------------

World = Mapping[int, Mapping[int, tuple[Any, ...]]]


def _dtype(col: c.Column) -> str:
    if col.kind == "int":
        return "Int64" if col.nullable else "int64"
    if col.kind == "float":
        return "float64"
    return col.kind  # "bool" · "str"


_ADJUSTED = {"pa_adj": "pa", "outs_adj": "outs"}


def _fill(spec: c.TableSpec, row: Mapping[str, Any]) -> dict[str, Any]:
    """계약의 모든 열을 채운 정제 표 행. row에 있는 값이 먼저다. 2020 환산 열만 162/60배."""
    pid, season = int(row["player_id"]), int(row["season"])
    out: dict[str, Any] = {}
    for i, col in enumerate(spec.columns):
        name = col.name
        if name in row:
            out[name] = row[name]
        elif name == "league":
            out[name] = c.LEAGUE
        elif name == "team_name":
            out[name] = f"T{pid}"
        elif name == "age":
            out[name] = 22 + (season - c.SEASON_FIRST) + pid % 5
        elif name == "n_teams":
            out[name] = 1
        elif name == "is_short_season":
            out[name] = season == c.SHORT_SEASON
        elif name in _ADJUSTED:
            continue  # 아래에서 출전량으로 계산
        elif col.kind == "int":
            out[name] = i + season % 100
        elif col.kind == "float":
            out[name] = round(0.1 + 0.01 * i + 0.001 * (season - c.SEASON_FIRST), 6)
        else:
            raise AssertionError(f"채울 수 없는 열: {name}")
    for adj, base in _ADJUSTED.items():
        if adj in spec.column_names:
            scale = c.SHORT_SEASON_SCALE if season == c.SHORT_SEASON else 1.0
            out[adj] = float(out[base]) * scale
    return out


def _frame(spec: c.TableSpec, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """계약의 모든 열을 계약 순서·형식으로 담은 표. 계약을 지키는지 바로 확인한다."""
    df = pd.DataFrame(
        {
            col.name: pd.Series([row[col.name] for row in rows], dtype=_dtype(col))
            for col in spec.columns
        }
    )
    assert c.check_frame(df, spec) == []
    return df


def _seasons(role: Role, world: World | None = None) -> pd.DataFrame:
    """합성 정제 시즌 표. world가 없으면 그 역할의 기본 선수 기록."""
    if world is None:
        world = BAT_WORLD if role == "bat" else PIT_WORLD
    rows: list[dict[str, Any]] = []
    for pid, by_season in world.items():
        for season, values in by_season.items():
            if role == "bat":
                rows.append({"player_id": pid, "season": season, "pa": values[0]})
            else:
                g, gs, outs = values[0], values[1], values[2]
                rows.append({"player_id": pid, "season": season, "g": g, "gs": gs, "outs": outs})
    return _frame(SPECS[role], [_fill(SPECS[role], row) for row in rows])


def _war_frame(records: Sequence[tuple[int, int, str, float]]) -> pd.DataFrame:
    """(player_id, season, role, war) → `WAR_SEASONS` 표. 구성 항목은 엉뚱한 값(DECOY)."""
    rows: list[dict[str, Any]] = []
    for pid, season, role, war in records:
        row: dict[str, Any] = {
            "player_id": pid,
            "season": season,
            "league": c.LEAGUE,
            "role": role,
            "war": war,
        }
        for name in WAR_COMPONENT_COLUMNS:
            row[name] = DECOY if role == "bat" else NAN
        row["war_ra9"] = DECOY if role == "pit" else NAN
        rows.append(row)
    return _frame(c.WAR_SEASONS, rows)


def _war_records(role: Role, world: World) -> list[tuple[int, int, str, float]]:
    return [
        (pid, season, role, float(values[-1]))
        for pid, by_season in world.items()
        for season, values in by_season.items()
    ]


def _war_seasons(
    bat_world: World | None = None,
    pit_world: World | None = None,
    drop: frozenset[tuple[int, int, str]] = frozenset(),
) -> pd.DataFrame:
    """두 역할의 WAR 표(타자 행 다음 투수 행). drop의 (player_id, season, role) 행은 뺀다."""
    records = [
        *_war_records("bat", BAT_WORLD if bat_world is None else bat_world),
        *_war_records("pit", PIT_WORLD if pit_world is None else pit_world),
    ]
    return _war_frame([r for r in records if (r[0], r[1], r[2]) not in drop])


def _cohort(role: Role, target: int = TARGET, world: World | None = None) -> pd.DataFrame:
    cohort = build_cohort(_seasons(role, world), role, target)
    assert c.check_frame(cohort, c.COHORTS) == []
    return cohort


def _empty_cohorts() -> pd.DataFrame:
    return _frame(c.COHORTS, [])


def _tables(
    bat_world: World | None = None,
    pit_world: World | None = None,
    *,
    cohorts: pd.DataFrame | None = None,
    debuts: Mapping[int, int] | None = None,
) -> Tables:
    """정제 표 묶음. cohorts가 없으면 빈 표 — `cohort_for`가 build_cohort로 바로 만든다."""
    return Tables(
        seasons={"bat": _seasons("bat", bat_world), "pit": _seasons("pit", pit_world)},
        cohorts=_empty_cohorts() if cohorts is None else cohorts,
        league={},
        debuts={} if debuts is None else dict(debuts),
    )


def _playing_time(role: Role, target: int, pids: Sequence[int]) -> pd.DataFrame:
    """주입하는 출전량 표 (`BAT_PLAYING_TIME` · `PIT_PLAYING_TIME`)."""
    df = pd.DataFrame(
        {
            "player_id": pd.Series(list(pids), dtype="int64"),
            "target_season": pd.Series([target] * len(pids), dtype="int64"),
            PT_COLUMN[role]: pd.Series([PT[role][p] for p in pids], dtype="float64"),
        }
    )[PT_SPECS[role].column_names]
    assert c.check_frame(df, PT_SPECS[role]) == []
    return df


def _rates_frame(rates: Mapping[int, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": pd.Series(list(rates), dtype="int64"),
            "war_rate": pd.Series(list(rates.values()), dtype="float64"),
        }
    )


def _slots_frame(slots: Mapping[int, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": pd.Series(list(slots), dtype="int64"),
            "slot": pd.Series(list(slots.values()), dtype="str"),
        }
    )


def _lg(role: Role, season: int) -> float:
    war, units = LEAGUE_SUMS[(role, season)]
    return war / units


def _rate(role: Role, target: int, pid: int, regress: float, lg: float) -> float:
    """손 계산 합으로 만든 기대 비율 (F18)."""
    a, b = HAND[(role, target)][pid]
    return (a + regress * lg) / (b + regress)


def _rate_map(out: pd.DataFrame) -> dict[int, float]:
    return {int(p): float(r) for p, r in zip(out["player_id"], out["war_rate"], strict=True)}


def _actual(role: Role, target: int, pid: int) -> float:
    """목표 시즌 실제 WAR (역할 행이 없으면 0)."""
    world: World = BAT_WORLD if role == "bat" else PIT_WORLD
    values = world.get(pid, {}).get(target)
    return 0.0 if values is None else float(values[-1])


def _oracle_mae(role: Role, target: int, regress: float) -> float:
    """대상 전원의 |비율 × 주입 출전량 − 실제 WAR| 평균 (D36)."""
    lg = _lg(role, target - 1)
    errors = [
        abs(_rate(role, target, pid, regress, lg) * PT[role][pid] - _actual(role, target, pid))
        for pid in HAND[(role, target)]
    ]
    return sum(errors) / len(errors)


def _oracle_choice(role: Role, grid: Sequence[float], targets: Sequence[int]) -> float:
    """두 해 평균 MAE 최소, 같으면 작은 R."""
    means = {r: sum(_oracle_mae(role, t, r) for t in targets) / len(targets) for r in grid}
    return min(sorted(means), key=lambda r: means[r])


def _poison_values(role: Role) -> tuple[Any, ...]:
    # 투수는 불펜 90경기로 — 섞이면 선발 판정도 바뀐다
    return (POISON_UNIT, POISON_WAR) if role == "bat" else (90, 0, POISON_UNIT, POISON_WAR)


def _poison_world(role: Role, cutoff: int) -> dict[int, dict[int, tuple[Any, ...]]]:
    """season ≥ cutoff 행만 바꾸고 더한 세계 (N2). cutoff 전 행은 순서까지 그대로다."""
    world: World = BAT_WORLD if role == "bat" else PIT_WORLD
    out: dict[int, dict[int, tuple[Any, ...]]] = {
        pid: {s: (_poison_values(role) if s >= cutoff else v) for s, v in by_season.items()}
        for pid, by_season in world.items()
    }
    extra = 1003 if role == "bat" else 2004  # 기존 선수에게 미래 시즌을 더한다
    out[extra] = {**out[extra], cutoff + 1: _poison_values(role)}
    new = NEW_BAT if role == "bat" else NEW_PIT  # 미래에만 있는 선수
    out[new] = {cutoff: _poison_values(role), cutoff + 1: _poison_values(role)}
    return out


def _poisoned(cutoff: int) -> tuple[World, World, pd.DataFrame]:
    """(타자 세계, 투수 세계, WAR 표). WAR 표에서는 미래 행 일부도 지운다(기간 밖이라 오류가 아니다)."""
    bat, pit = _poison_world("bat", cutoff), _poison_world("pit", cutoff)
    drop = frozenset({(TWO_WAY, cutoff, "bat"), (NEW_PIT, cutoff + 1, "pit")})
    return bat, pit, _war_seasons(bat, pit, drop)


def _install_fake_playing_time(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """`predict_playing_time`을 PT를 돌려주는 가짜로 바꾼다 (학습을 피한다). 호출 기록을 돌려준다."""
    calls: list[dict[str, Any]] = []

    def fake(
        cohort: pd.DataFrame,
        seasons: pd.DataFrame,
        role: Role,
        debut_seasons: Mapping[int, int] | None = None,
    ) -> pd.DataFrame:
        targets = sorted({int(t) for t in cohort["target_season"]})
        assert len(targets) == 1, targets
        pids = sorted(int(p) for p in cohort["player_id"])
        calls.append(
            {
                "role": role,
                "target": targets[0],
                "pids": pids,
                "seasons": seasons,
                "debut_seasons": debut_seasons,
            }
        )
        return _playing_time(role, targets[0], pids)

    monkeypatch.setattr(playing_time_module, "predict_playing_time", fake)
    monkeypatch.setattr(war_module, "predict_playing_time", fake, raising=False)
    return calls


def _expected_projection(
    out: pd.DataFrame,
    role: Role,
    target: int,
    rates: Mapping[int, float],
    pts: Mapping[int, float],
    slots: Mapping[int, str],
    n_seasons: Mapping[int, int],
) -> None:
    """`WAR_PROJECTIONS` 결과가 손 계산(F19)과 같은지 본다."""
    assert list(out.columns) == c.WAR_PROJECTIONS.column_names
    assert c.check_frame(out, c.WAR_PROJECTIONS) == []
    assert out["player_id"].tolist() == sorted(rates)
    for row in out.to_dict("records"):
        pid = int(row["player_id"])
        units = c.FULL_TIME_UNITS[slots[pid]]
        assert row["role"] == role
        assert row["target_season"] == target
        assert row["n_seasons"] == n_seasons[pid]
        assert row["slot"] == slots[pid]
        assert row["war_rate"] == pytest.approx(rates[pid], rel=REL, abs=ABS)
        assert row["pt"] == pytest.approx(pts[pid], rel=REL)
        assert row["war"] == pytest.approx(rates[pid] * pts[pid], rel=REL, abs=ABS)
        assert row["full_units"] == units
        assert row["war_full"] == pytest.approx(rates[pid] * units, rel=REL, abs=ABS)


def _same(got: Any, expected: float) -> bool:
    if math.isnan(expected):
        return bool(pd.isna(got))
    return not pd.isna(got) and float(got) == pytest.approx(expected, rel=REL, abs=ABS)


# ---------------------------------------------------------------------------
# 도우미 자체 확인 — 손 계산 합이 합성 세계와 맞는지
# ---------------------------------------------------------------------------


def test_models_war_helpers_match_hand_sums() -> None:
    for (role, target), sums in HAND.items():
        world: World = BAT_WORLD if role == "bat" else PIT_WORLD
        unit_index = 0 if role == "bat" else 2
        cohort = _cohort(role, target)
        assert sorted(int(p) for p in cohort["player_id"]) == sorted(sums)
        for pid, (war_sum, unit_sum) in sums.items():
            first = int(cohort.loc[cohort["player_id"] == pid, "window_first"].iloc[0])
            rows = [(s, v) for s, v in world[pid].items() if first <= s < target]
            weights = WEIGHTS[role]
            assert sum(weights[target - 1 - s] * v[-1] for s, v in rows) == pytest.approx(war_sum)
            assert sum(weights[target - 1 - s] * v[unit_index] for s, v in rows) == pytest.approx(
                unit_sum
            )
    for (role, season), (war_sum, unit_sum) in LEAGUE_SUMS.items():
        league_world: World = BAT_WORLD if role == "bat" else PIT_WORLD
        unit_index = 0 if role == "bat" else 2
        values = [
            v for by_season in league_world.values() for s, v in by_season.items() if s == season
        ]
        assert sum(v[-1] for v in values) == pytest.approx(war_sum)
        assert sum(v[unit_index] for v in values) == pytest.approx(unit_sum)
    for role, counts in N_SEASONS_2025.items():
        cohort = _cohort(role)
        assert dict(zip(cohort["player_id"], cohort["n_seasons"], strict=True)) == counts
    assert c.check_frame(_war_seasons(), c.WAR_SEASONS) == []


# ---------------------------------------------------------------------------
# WarParams JSON (config/war_v1.json)
# ---------------------------------------------------------------------------


def test_models_war_params_json_is_normalized() -> None:
    text = war_params_to_json(DEFAULT_WAR_PARAMS)
    assert text.endswith("\n") and not text.endswith("\n\n")
    doc = json.loads(text)
    assert doc == {"regress": {"bat": 1200.0, "pit": 600.0}, "version": "v0"}
    # 키 정렬 · indent=2 · 끝 줄바꿈
    assert text == json.dumps(doc, indent=2, sort_keys=True) + "\n"


def test_models_war_params_json_same_value_same_text() -> None:
    base = WarParams(version="v1", regress={"bat": 1200.0, "pit": 600.0})
    reordered = WarParams(version="v1", regress={"pit": 600.0, "bat": 1200.0})
    as_int = WarParams(version="v1", regress={"bat": 1200, "pit": 600})
    assert war_params_to_json(base) == war_params_to_json(reordered) == war_params_to_json(as_int)


@pytest.mark.parametrize(
    "params",
    [
        DEFAULT_WAR_PARAMS,
        WarParams(version="v1", regress={"bat": 2400.0, "pit": 150.0}),
        WarParams(version="v-test", regress={"bat": 333.25, "pit": 0.5}),
    ],
    ids=["default", "grid", "fractional"],
)
def test_models_war_params_json_round_trip(params: WarParams) -> None:
    text = war_params_to_json(params)
    back = war_params_from_json(text)
    assert back == params
    assert all(isinstance(v, float) for v in back.regress.values())
    assert war_params_to_json(back) == text


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"version": "v1"}',
        '{"regress": {"bat": 1.0, "pit": 1.0}}',
        '{"regress": {"bat": 1.0, "pit": 1.0}, "version": "v1", "extra": 0}',
        '{"regress": {"bat": 1.0}, "version": "v1"}',
        '{"regress": {"pit": 1.0}, "version": "v1"}',
        '{"regress": {"bat": 1.0, "pit": 1.0, "two": 1.0}, "version": "v1"}',
        '{"regress": [1.0, 1.0], "version": "v1"}',
        '{"regress": {"bat": 0.0, "pit": 1.0}, "version": "v1"}',
        '{"regress": {"bat": 1.0, "pit": -5.0}, "version": "v1"}',
        '{"regress": {"bat": NaN, "pit": 1.0}, "version": "v1"}',
        '{"regress": {"bat": 1.0, "pit": Infinity}, "version": "v1"}',
        '{"regress": {"bat": "abc", "pit": 1.0}, "version": "v1"}',
        '{"regress": {"bat": null, "pit": 1.0}, "version": "v1"}',
        '{"regress": {"bat": true, "pit": 1.0}, "version": "v1"}',
        '{"regress": {"bat": 1.0, "pit": 1.0}, "version": ""}',
        '{"regress": {"bat": 1.0, "pit": 1.0}, "version": 1}',
        # 같은 키 두 번 — json 기본은 마지막 값을 조용히 쓴다(커밋된 R이 읽는 사람 눈과 달라진다)
        '{"regress": {"bat": 1.0, "pit": 1.0}, "regress": {"bat": 2.0, "pit": 2.0}, "version": "v1"}',
        '{"regress": {"bat": 1.0, "bat": 2.0, "pit": 1.0}, "version": "v1"}',
    ],
    ids=[
        "invalid_json",
        "not_object",
        "missing_regress",
        "missing_version",
        "extra_key",
        "missing_pit",
        "missing_bat",
        "extra_role",
        "regress_not_object",
        "zero",
        "negative",
        "nan",
        "infinity",
        "string",
        "null",
        "bool",
        "empty_version",
        "version_not_string",
        "duplicate_top_key",
        "duplicate_role",
    ],
)
def test_models_war_params_from_json_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        war_params_from_json(text)


@pytest.mark.parametrize(
    "params",
    [
        WarParams(version="v1", regress=cast(dict[Role, float], {"bat": 1.0})),
        WarParams(version="v1", regress={"bat": 1.0, "pit": 0.0}),
        WarParams(version="v1", regress={"bat": float("inf"), "pit": 1.0}),
        WarParams(version="", regress={"bat": 1.0, "pit": 1.0}),
    ],
    ids=["missing_pit", "zero", "infinity", "empty_version"],
)
def test_models_war_params_to_json_rejects_invalid(params: WarParams, tmp_path: Path) -> None:
    # 쓰는 쪽도 같은 규칙 — 잘못된 값이 config 파일·해시로 새지 않는다
    with pytest.raises(ValueError):
        war_params_to_json(params)
    with pytest.raises(ValueError):
        war_params_sha256(params)
    path = tmp_path / "war_v1.json"
    with pytest.raises(ValueError):
        save_war_params(params, path)
    assert not path.exists()


def test_models_war_params_sha256_is_stable_hex() -> None:
    digest = war_params_sha256(DEFAULT_WAR_PARAMS)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    expected = hashlib.sha256(war_params_to_json(DEFAULT_WAR_PARAMS).encode("utf-8")).hexdigest()
    assert digest == expected
    # 같은 값이면 같은 해시 (역할 순서·int/float 무관)
    assert war_params_sha256(WarParams(version="v0", regress={"pit": 600, "bat": 1200})) == digest
    assert war_params_sha256(DEFAULT_WAR_PARAMS) == digest


def test_models_war_params_sha256_changes_with_values() -> None:
    base = war_params_sha256(DEFAULT_WAR_PARAMS)
    changed = [
        WarParams(version="v1", regress={"bat": 1200.0, "pit": 600.0}),
        WarParams(version="v0", regress={"bat": 1201.0, "pit": 600.0}),
        WarParams(version="v0", regress={"bat": 1200.0, "pit": 601.0}),
    ]
    digests = {war_params_sha256(p) for p in changed}
    assert base not in digests
    assert len(digests) == len(changed)


def test_models_war_params_save_load_round_trip(tmp_path: Path) -> None:
    params = WarParams(version="v1", regress={"bat": 2400.0, "pit": 300.0})
    path = tmp_path / "config" / "nested" / "war_v1.json"  # 부모 디렉토리를 만든다
    save_war_params(params, path)
    assert path.read_text(encoding="utf-8") == war_params_to_json(params)
    assert load_war_params(path) == params


@pytest.mark.parametrize(
    "text",
    ["{", '{"version": "v1"}', '{"regress": {"bat": -1.0, "pit": 1.0}, "version": "v1"}'],
    ids=["broken", "missing_regress", "negative"],
)
def test_models_war_params_load_rejects_malformed_file(tmp_path: Path, text: str) -> None:
    path = tmp_path / "war_v1.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_war_params(path)


# ---------------------------------------------------------------------------
# 리그 WAR 비율 (F18의 lg)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(LEAGUE_SUMS), ids=lambda k: f"{k[0]}-{k[1]}")
def test_models_war_league_rate_hand_computed(key: tuple[Role, int]) -> None:
    role, season = key
    # 다른 역할 행(1001 투수 WAR 등)·다른 시즌 행·2020 환산값을 쓰면 어긋난다
    got = league_war_rate(_war_seasons(), _seasons(role), role, season)
    assert got == pytest.approx(_lg(role, season), rel=REL)


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_war_league_rate_rejects_missing_season(role: Role) -> None:
    with pytest.raises(ValueError):
        league_war_rate(_war_seasons(), _seasons(role), role, 2018)


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_war_league_rate_rejects_zero_units(role: Role) -> None:
    world: dict[int, dict[int, tuple[Any, ...]]] = (
        {7: {2024: (0, 0.5)}} if role == "bat" else {7: {2024: (1, 0, 0, 0.5)}}
    )
    war = _war_frame(_war_records(role, world))
    with pytest.raises(ValueError):
        league_war_rate(war, _seasons(role, world), role, 2024)


# ---------------------------------------------------------------------------
# WAR 비율 (F18)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("regress", "lg"),
    [(BAT_R, 0.005), (2500.0, 0.0), (1.0, -0.002)],
    ids=["base", "zero_league", "tiny_regress"],
)
def test_models_war_rates_bat_hand_computed(regress: float, lg: float) -> None:
    # 예: R=100, lg=.005 → 1001 (42 + .5) / (7400 + 100) = 42.5/7500,
    # 1002 (4.75 + .5) / (2500 + 100) = 5.25/2600, 1003 (−1.6 + .5) / (200 + 100) = −1.1/300
    cohort = _cohort("bat")
    out = war_rates(cohort, _seasons("bat"), _war_seasons(), "bat", BAT_WEIGHTS, regress, lg)
    expected = {pid: _rate("bat", TARGET, pid, regress, lg) for pid in HAND[("bat", TARGET)]}
    assert _rate_map(out) == pytest.approx(expected, rel=REL, abs=ABS)


def test_models_war_rates_bat_hand_values_literal() -> None:
    out = war_rates(
        _cohort("bat"), _seasons("bat"), _war_seasons(), "bat", BAT_WEIGHTS, 100.0, 0.005
    )
    expected = {TWO_WAY: 42.5 / 7500, 1002: 5.25 / 2600, 1003: -1.1 / 300}
    assert _rate_map(out) == pytest.approx(expected, rel=REL)


@pytest.mark.parametrize(
    ("regress", "lg"), [(PIT_R, 0.004), (600.0, 0.006)], ids=["base", "larger_regress"]
)
def test_models_war_rates_pit_hand_computed(regress: float, lg: float) -> None:
    # 예: R=50, lg=.004 → 2001 (37.5 + .2) / (6780 + 50) = 37.7/6830, 2005 (1.2 + .2) / (200 + 50)
    cohort = _cohort("pit")
    out = war_rates(cohort, _seasons("pit"), _war_seasons(), "pit", PIT_WEIGHTS, regress, lg)
    expected = {pid: _rate("pit", TARGET, pid, regress, lg) for pid in HAND[("pit", TARGET)]}
    assert _rate_map(out) == pytest.approx(expected, rel=REL, abs=ABS)


def test_models_war_rates_target_2024_window() -> None:
    # 목표 Y가 바뀌면 가중치도 Y−1부터 다시 센다 (Y=2024: 2023 = weights[0], 2019 = weights[4])
    for role in c.ROLES:
        lg = _lg(role, 2023)
        out = war_rates(
            _cohort(role, 2024), _seasons(role), _war_seasons(), role, WEIGHTS[role], 300.0, lg
        )
        expected = {pid: _rate(role, 2024, pid, 300.0, lg) for pid in HAND[(role, 2024)]}
        assert _rate_map(out) == pytest.approx(expected, rel=REL, abs=ABS)


def test_models_war_rates_weights_follow_recency() -> None:
    # 가중치를 거꾸로 주면 값이 달라진다 — weights[0]이 Y−1이다
    reversed_weights = tuple(reversed(BAT_WEIGHTS))
    out = war_rates(
        _cohort("bat"), _seasons("bat"), _war_seasons(), "bat", reversed_weights, BAT_R, 0.005
    )
    # 1001: w 2024=0.5, 2023=1, 2022=2 → (0.5·4 + 1·1 + 2·3 + .5) / (0.5·600 + 400 + 2·500 + 100)
    assert _rate_map(out)[TWO_WAY] == pytest.approx(9.5 / 1800, rel=REL)


def test_models_war_rates_short_season_uses_actual_units() -> None:
    # 2020 한 시즌뿐인 2005: 실제 아웃 200 (환산 540이 아니다). w(2020) = weights[4] = 1
    out = war_rates(
        _cohort("pit"), _seasons("pit"), _war_seasons(), "pit", PIT_WEIGHTS, 50.0, 0.004
    )
    assert _rate_map(out)[2005] == pytest.approx((1.2 + 0.2) / (200 + 50), rel=REL)
    # 1002의 2020: 실제 타석 200 (환산 540이 아니다), w(2020) = 0.5
    out = war_rates(
        _cohort("bat"), _seasons("bat"), _war_seasons(), "bat", BAT_WEIGHTS, 100.0, 0.005
    )
    assert _rate_map(out)[1002] == pytest.approx((4.0 + 0.75 + 0.5) / (2400 + 100 + 100), rel=REL)


def test_models_war_rates_huge_regress_gives_league_rate() -> None:
    out = war_rates(
        _cohort("bat"), _seasons("bat"), _war_seasons(), "bat", BAT_WEIGHTS, 1e12, 0.005
    )
    for value in _rate_map(out).values():
        assert value == pytest.approx(0.005, rel=1e-6)


def test_models_war_rates_ignore_other_role_and_components() -> None:
    # 투타 겸업 1001의 투수 WAR·WAR 구성 항목 열을 바꿔도 타자 비율은 비트까지 같다 (반대도)
    base_bat = war_rates(
        _cohort("bat"), _seasons("bat"), _war_seasons(), "bat", BAT_WEIGHTS, BAT_R, 0.005
    )
    base_pit = war_rates(
        _cohort("pit"), _seasons("pit"), _war_seasons(), "pit", PIT_WEIGHTS, PIT_R, 0.004
    )
    other = _war_seasons()
    two_way_pit = (other["player_id"] == TWO_WAY) & (other["role"] == "pit")
    other.loc[two_way_pit, "war"] = 50.0
    other.loc[other["role"] == "bat", list(WAR_COMPONENT_COLUMNS)] = -3.0
    other.loc[other["role"] == "pit", "war_ra9"] = -3.0
    changed_bat = war_rates(
        _cohort("bat"), _seasons("bat"), other, "bat", BAT_WEIGHTS, BAT_R, 0.005
    )
    assert_frame_equal(changed_bat, base_bat, check_exact=True)

    other = _war_seasons()
    other.loc[(other["player_id"] == TWO_WAY) & (other["role"] == "bat"), "war"] = 50.0
    changed_pit = war_rates(
        _cohort("pit"), _seasons("pit"), other, "pit", PIT_WEIGHTS, PIT_R, 0.004
    )
    assert_frame_equal(changed_pit, base_pit, check_exact=True)
    assert changed_pit.set_index("player_id").loc[TWO_WAY, "war_rate"] == pytest.approx(
        _rate("pit", TARGET, TWO_WAY, PIT_R, 0.004), rel=REL
    )


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_war_rates_output_columns_and_order(role: Role) -> None:
    cohort = _cohort(role).iloc[::-1].reset_index(drop=True)  # 일부러 거꾸로
    out = war_rates(cohort, _seasons(role), _war_seasons(), role, WEIGHTS[role], 100.0, 0.005)
    assert list(out.columns) == ["player_id", "war_rate"]
    assert out["player_id"].tolist() == sorted(HAND[(role, TARGET)])
    assert pd.api.types.is_integer_dtype(out["player_id"])
    assert pd.api.types.is_float_dtype(out["war_rate"])
    assert not out["war_rate"].isna().any()


def test_models_war_rates_rejects_mixed_targets() -> None:
    cohort = pd.concat([_cohort("bat", 2025), _cohort("bat", 2024)], ignore_index=True)
    with pytest.raises(ValueError):
        war_rates(cohort, _seasons("bat"), _war_seasons(), "bat", BAT_WEIGHTS, BAT_R, 0.005)


def test_models_war_rates_rejects_mixed_roles() -> None:
    cohort = pd.concat([_cohort("bat"), _cohort("pit")], ignore_index=True)
    with pytest.raises(ValueError):
        war_rates(cohort, _seasons("bat"), _war_seasons(), "bat", BAT_WEIGHTS, BAT_R, 0.005)


@pytest.mark.parametrize(
    ("role", "pid", "season"),
    [
        ("bat", 1002, 2020),  # 단축 시즌 행
        ("bat", TWO_WAY, 2024),  # 투수 WAR 행(1001·2024·pit)은 남아 있다 — 대신 쓰면 안 된다
        ("pit", 2003, 2023),
        ("pit", TWO_WAY, 2024),  # 타자 WAR 행(1001·2024·bat)은 남아 있다
    ],
    ids=["bat_short_season", "bat_two_way", "pit", "pit_two_way"],
)
def test_models_war_rates_rejects_missing_war_row(role: Role, pid: int, season: int) -> None:
    war = _war_seasons(drop=frozenset({(pid, season, role)}))
    with pytest.raises(ValueError, match=str(pid)):
        war_rates(_cohort(role), _seasons(role), war, role, WEIGHTS[role], 100.0, 0.005)


def test_models_war_rates_missing_war_row_outside_window_is_ignored() -> None:
    # 기간 밖(2019) WAR 행이 없어도 오류가 아니고 결과도 같다
    cases: tuple[tuple[Role, int], ...] = (("bat", 1002), ("pit", 2001))
    for role, pid in cases:
        base = war_rates(
            _cohort(role), _seasons(role), _war_seasons(), role, WEIGHTS[role], 100.0, 0.005
        )
        war = _war_seasons(drop=frozenset({(pid, 2019, role)}))
        out = war_rates(_cohort(role), _seasons(role), war, role, WEIGHTS[role], 100.0, 0.005)
        assert_frame_equal(out, base, check_exact=True)


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_war_rates_no_leak(role: Role) -> None:
    # N2: season ≥ Y 행(시즌 표·WAR 표)을 바꾸고 더하고 지워도 결과 비트가 같다
    cohort = _cohort(role)
    clean = war_rates(cohort, _seasons(role), _war_seasons(), role, WEIGHTS[role], 100.0, 0.005)
    bat, pit, war = _poisoned(TARGET)
    poisoned_seasons = _seasons(role, bat if role == "bat" else pit)
    assert_frame_equal(build_cohort(poisoned_seasons, role, TARGET), cohort, check_exact=True)
    dirty = war_rates(cohort, poisoned_seasons, war, role, WEIGHTS[role], 100.0, 0.005)
    assert_frame_equal(dirty, clean, check_exact=True)


def _shuffled(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_war_rates_input_row_order_is_bit_stable(role: Role) -> None:
    # 같은 입력 같은 바이트 — 시즌 표·WAR 표·대상 표의 행 순서를 섞어도 비트까지 같다
    base = war_rates(
        _cohort(role), _seasons(role), _war_seasons(), role, WEIGHTS[role], 100.0, 0.005
    )
    for seed in (1, 2, 3):
        out = war_rates(
            _shuffled(_cohort(role), seed),
            _shuffled(_seasons(role), seed + 10),
            _shuffled(_war_seasons(), seed + 20),
            role,
            WEIGHTS[role],
            100.0,
            0.005,
        )
        assert_frame_equal(out, base, check_exact=True)


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_war_league_rate_input_row_order_is_bit_stable(role: Role) -> None:
    base = league_war_rate(_war_seasons(), _seasons(role), role, 2024)
    for seed in (1, 2, 3):
        got = league_war_rate(
            _shuffled(_war_seasons(), seed), _shuffled(_seasons(role), seed + 10), role, 2024
        )
        assert got == base  # 근삿값이 아니라 같은 비트


# 더하는 순서에 따라 비트가 달라지는 값 (0.1 + 0.2 + 0.7 ≠ 0.7 + 0.2 + 0.1). 11은 기간 2022~2024에
# 0.1·0.2·0.7, 2024 리그는 11·12·13이 0.7·0.2·0.1 — 행 순서를 뒤집으면 정렬 없이 더한 합이 달라진다.
INEXACT_WORLD: dict[int, dict[int, tuple[Any, ...]]] = {
    11: {2022: (100, 0.1), 2023: (100, 0.2), 2024: (100, 0.7)},
    12: {2024: (100, 0.2)},
    13: {2024: (100, 0.1)},
}


def test_models_war_sums_do_not_depend_on_row_order() -> None:
    # 도우미 확인: 이 값들은 정말 순서에 민감하다 (아니면 아래 비교가 늘 통과한다)
    league = [0.7, 0.2, 0.1]  # 2024 리그 WAR, player_id 순
    assert float(pd.Series(league).sum()) != float(pd.Series(league[::-1]).sum())
    rows = pd.DataFrame({"k": [11, 11, 11, 12, 13], "v": [0.1, 0.2, 0.7, 0.2, 0.1]})
    flipped_rows = rows.iloc[::-1].reset_index(drop=True)
    assert float(rows.groupby("k")["v"].sum().loc[11]) != float(
        flipped_rows.groupby("k")["v"].sum().loc[11]
    )

    seasons = _seasons("bat", INEXACT_WORLD)
    war = _war_frame(_war_records("bat", INEXACT_WORLD))
    flipped_seasons = seasons.iloc[::-1].reset_index(drop=True)
    flipped_war = war.iloc[::-1].reset_index(drop=True)
    # 리그 비율 — 키(player_id) 순으로 더한다
    assert league_war_rate(war, seasons, "bat", 2024) == league_war_rate(
        flipped_war, flipped_seasons, "bat", 2024
    )
    # WAR 비율 — 선수마다 시즌 순으로 더한다 (R·lg가 합을 가리지 않게 작게)
    cohort = build_cohort(seasons, "bat", TARGET)
    flat = (1.0, 1.0, 1.0, 1.0, 1.0)
    base = war_rates(cohort, seasons, war, "bat", flat, 1.0, 0.0)
    flipped = war_rates(cohort, flipped_seasons, flipped_war, "bat", flat, 1.0, 0.0)
    assert_frame_equal(flipped, base, check_exact=True)


@pytest.mark.parametrize(
    ("weights", "regress", "lg"),
    [
        ((8.0, 4.0, 2.0, 1.0), BAT_R, 0.005),  # 가중치 4개 (WINDOW = 5)
        ((8.0, 4.0, 2.0, 1.0, 0.5, 0.25), BAT_R, 0.005),
        ((8.0, 4.0, -2.0, 1.0, 0.5), BAT_R, 0.005),
        ((8.0, 4.0, NAN, 1.0, 0.5), BAT_R, 0.005),
        (BAT_WEIGHTS, 0.0, 0.005),  # R = 0 — 출전 0인 선수에서 0 나누기
        (BAT_WEIGHTS, -100.0, 0.005),
        (BAT_WEIGHTS, NAN, 0.005),
        (BAT_WEIGHTS, BAT_R, NAN),
        (BAT_WEIGHTS, BAT_R, float("inf")),
    ],
    ids=[
        "weights_short",
        "weights_long",
        "weights_negative",
        "weights_nan",
        "regress_zero",
        "regress_negative",
        "regress_nan",
        "league_nan",
        "league_inf",
    ],
)
def test_models_war_rates_rejects_bad_parameters(
    weights: tuple[float, ...], regress: float, lg: float
) -> None:
    with pytest.raises(ValueError):
        war_rates(_cohort("bat"), _seasons("bat"), _war_seasons(), "bat", weights, regress, lg)


def _leaky(cohort: pd.DataFrame, case: str) -> pd.DataFrame:
    """기간이 목표 Y의 Y−5 ~ Y−1을 벗어난 대상 표 (N2 — 계약 형식은 지킨다)."""
    out = cohort.copy()
    first = out["window_first"].to_numpy(dtype="int64", copy=True)
    if case == "last_is_target":  # 목표 시즌까지 읽는다
        out["window_last"] = out["target_season"]
    elif case == "first_too_early":  # Y−6부터
        first[0] = int(out["target_season"].iloc[0]) - c.WINDOW - 1
    else:  # first > last
        first[0] = int(out["window_last"].iloc[0]) + 1
    out["window_first"] = pd.Series(first, dtype="int64")
    assert c.check_frame(out, c.COHORTS) == []
    return out


LEAKY_CASES = ["last_is_target", "first_too_early", "first_after_last"]


@pytest.mark.parametrize("case", LEAKY_CASES)
@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_war_rates_rejects_cohort_window_outside_target(role: Role, case: str) -> None:
    # 기간이 Y 이후를 덮는 대상 표는 계산 전에 막는다 (N2 — season < Y 필터와 별도의 이중 장치)
    with pytest.raises(ValueError, match="벗어난"):
        war_rates(
            _leaky(_cohort(role), case),
            _seasons(role),
            _war_seasons(),
            role,
            WEIGHTS[role],
            100.0,
            0.005,
        )


@pytest.mark.parametrize("case", LEAKY_CASES)
def test_models_war_pitcher_slots_rejects_cohort_window_outside_target(case: str) -> None:
    with pytest.raises(ValueError, match="벗어난"):
        pitcher_slots(_leaky(_cohort("pit"), case), _seasons("pit"), PIT_WEIGHTS)


def test_models_war_pitcher_slots_rejects_batter_cohort() -> None:
    with pytest.raises(ValueError):
        pitcher_slots(_cohort("bat"), _seasons("pit"), PIT_WEIGHTS)


# ---------------------------------------------------------------------------
# 투수 자리 (D35)
# ---------------------------------------------------------------------------


def _slot_map(out: pd.DataFrame) -> dict[int, str]:
    return {int(p): str(s) for p, s in zip(out["player_id"], out["slot"], strict=True)}


def test_models_war_pitcher_slots_hand_computed() -> None:
    out = pitcher_slots(_cohort("pit"), _seasons("pit"), PIT_WEIGHTS)
    assert _slot_map(out) == SLOTS_2025


def test_models_war_pitcher_slots_follow_weights() -> None:
    # 같은 가중이면 2003은 30/65 < .5 → 불펜 (가중하면 선발이었다)
    out = pitcher_slots(_cohort("pit"), _seasons("pit"), (1.0, 1.0, 1.0, 1.0, 1.0))
    assert _slot_map(out) == {**SLOTS_2025, 2003: "rp"}
    # 가중치를 거꾸로 주면 2003은 1·30 / (1·30 + 2·35) = .3 → 불펜 — weights[0]이 Y−1이다
    out = pitcher_slots(_cohort("pit"), _seasons("pit"), tuple(reversed(PIT_WEIGHTS)))
    assert _slot_map(out)[2003] == "rp"


def test_models_war_pitcher_slots_zero_weighted_games_is_rp() -> None:
    # Y−5(2020) 가중치가 0이면 2020만 던진 2005는 Σw·g = 0 → 불펜 (선발 12/12여도)
    out = pitcher_slots(_cohort("pit"), _seasons("pit"), (5.0, 4.0, 3.0, 2.0, 0.0))
    assert _slot_map(out) == {**SLOTS_2025, 2005: "rp"}


def test_models_war_pitcher_slots_zero_games_is_rp() -> None:
    world: dict[int, dict[int, tuple[Any, ...]]] = {
        8: {2024: (0, 0, 3, 0.0)},  # g = 0 (합성 경계) → 나누지 않고 불펜
        9: {2024: (1, 1, 15, 0.1)},
    }
    cohort = build_cohort(_seasons("pit", world), "pit", TARGET)
    out = pitcher_slots(cohort, _seasons("pit", world), PIT_WEIGHTS)
    assert _slot_map(out) == {8: "rp", 9: "sp"}


def test_models_war_pitcher_slots_output_columns_and_order() -> None:
    cohort = _cohort("pit").iloc[::-1].reset_index(drop=True)
    out = pitcher_slots(cohort, _seasons("pit"), PIT_WEIGHTS)
    assert list(out.columns) == ["player_id", "slot"]
    assert out["player_id"].tolist() == sorted(SLOTS_2025)
    assert set(out["slot"]) <= {"sp", "rp"}


def test_models_war_pitcher_slots_no_leak() -> None:
    cohort = _cohort("pit")
    clean = pitcher_slots(cohort, _seasons("pit"), PIT_WEIGHTS)
    _, pit, _ = _poisoned(TARGET)
    dirty = pitcher_slots(cohort, _seasons("pit", pit), PIT_WEIGHTS)
    assert_frame_equal(dirty, clean, check_exact=True)


# ---------------------------------------------------------------------------
# 예측 WAR · 풀타임 환산 (F19)
# ---------------------------------------------------------------------------

BAT_RATES_IN: dict[int, float] = {TWO_WAY: 0.006, 1002: 0.002, 1003: -0.0035}
PIT_RATES_IN: dict[int, float] = {
    TWO_WAY: 0.01,
    2001: 0.0075,
    2002: 0.004,
    2003: 0.005,
    2004: 0.0035,
    2005: -0.002,
}
PIT_SLOTS_IN: dict[int, str] = {
    TWO_WAY: "sp",
    2001: "sp",
    2002: "rp",
    2003: "sp",
    2004: "sp",
    2005: "rp",
}


def test_models_war_project_bat_hand_computed() -> None:
    # 1001: war = .006 × 550 = 3.3, war_full = .006 × 600 = 3.6 · 1003: 출전 100, 음수 비율
    cohort = _cohort("bat")
    pts = PT["bat"]
    out = project_war(
        cohort, _rates_frame(BAT_RATES_IN), _playing_time("bat", TARGET, sorted(pts)), "bat"
    )
    slots = dict.fromkeys(BAT_RATES_IN, "bat")
    _expected_projection(out, "bat", TARGET, BAT_RATES_IN, pts, slots, N_SEASONS_2025["bat"])
    assert out.set_index("player_id").loc[TWO_WAY, "war"] == pytest.approx(3.3, rel=REL)
    assert out.set_index("player_id").loc[TWO_WAY, "war_full"] == pytest.approx(3.6, rel=REL)


def test_models_war_project_pit_hand_computed() -> None:
    # 2001(선발): .0075 × 540 = 4.05, 풀타임 .0075 × 540 · 2002(불펜): .004 × 190, 풀타임 .004 × 195
    cohort = _cohort("pit")
    pts = PT["pit"]
    out = project_war(
        cohort,
        _rates_frame(PIT_RATES_IN),
        _playing_time("pit", TARGET, sorted(pts)),
        "pit",
        _slots_frame(PIT_SLOTS_IN),
    )
    _expected_projection(out, "pit", TARGET, PIT_RATES_IN, pts, PIT_SLOTS_IN, N_SEASONS_2025["pit"])
    by_pid = out.set_index("player_id")
    assert by_pid.loc[2002, "full_units"] == 195.0
    assert by_pid.loc[2002, "war_full"] == pytest.approx(0.004 * 195, rel=REL)
    assert by_pid.loc[2001, "full_units"] == 540.0


def test_models_war_project_bat_ignores_slots() -> None:
    cohort = _cohort("bat")
    out = project_war(
        cohort,
        _rates_frame(BAT_RATES_IN),
        _playing_time("bat", TARGET, sorted(PT["bat"])),
        "bat",
        _slots_frame(dict.fromkeys(BAT_RATES_IN, "rp")),
    )
    assert set(out["slot"]) == {"bat"}
    assert set(out["full_units"]) == {c.FULL_TIME_UNITS["bat"]}


def test_models_war_project_pit_requires_slots() -> None:
    with pytest.raises(ValueError):
        project_war(
            _cohort("pit"),
            _rates_frame(PIT_RATES_IN),
            _playing_time("pit", TARGET, sorted(PT["pit"])),
            "pit",
        )


@pytest.mark.parametrize(
    "case", ["rates_missing", "rates_extra", "pt_missing", "pt_extra", "cohort_missing"]
)
def test_models_war_project_rejects_player_mismatch(case: str) -> None:
    cohort = _cohort("bat")
    rates = dict(BAT_RATES_IN)
    pt_pids = sorted(PT["bat"])
    if case == "rates_missing":
        del rates[1003]
    elif case == "rates_extra":
        rates[4242] = 0.001
    elif case == "pt_missing":
        pt_pids = pt_pids[:-1]
    elif case == "cohort_missing":
        cohort = cohort[cohort["player_id"] != 1002].reset_index(drop=True)
    pt = _playing_time("bat", TARGET, pt_pids)
    if case == "pt_extra":
        extra = pd.DataFrame(
            {
                "player_id": pd.Series([4242], dtype="int64"),
                "target_season": pd.Series([TARGET], dtype="int64"),
                "pa": pd.Series([10.0], dtype="float64"),
            }
        )
        pt = pd.concat([pt, extra], ignore_index=True)
    with pytest.raises(ValueError):
        project_war(cohort, _rates_frame(rates), pt, "bat")


@pytest.mark.parametrize(
    "case", ["slots_missing", "slots_extra", "slots_duplicate", "slot_bat", "slot_unknown"]
)
def test_models_war_project_pit_rejects_bad_slots(case: str) -> None:
    slots = _slots_frame(PIT_SLOTS_IN)
    if case == "slots_missing":
        slots = slots[slots["player_id"] != 2005].reset_index(drop=True)
    elif case == "slots_extra":
        slots = pd.concat([slots, _slots_frame({4242: "sp"})], ignore_index=True)
    elif case == "slots_duplicate":
        slots = pd.concat([slots, _slots_frame({2005: "rp"})], ignore_index=True)
    elif case == "slot_bat":  # 투수에게 타자 기준(600타석)을 붙이면 안 된다
        slots = _slots_frame({**PIT_SLOTS_IN, 2005: "bat"})
    else:
        slots = _slots_frame({**PIT_SLOTS_IN, 2005: "closer"})
    with pytest.raises(ValueError):
        project_war(
            _cohort("pit"),
            _rates_frame(PIT_RATES_IN),
            _playing_time("pit", TARGET, sorted(PT["pit"])),
            "pit",
            slots,
        )


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_war_project_rejects_playing_time_target_mismatch(role: Role) -> None:
    # 다른 목표의 출전량(예: 2024 예측)을 섞으면 막는다
    rates = BAT_RATES_IN if role == "bat" else PIT_RATES_IN
    slots = None if role == "bat" else _slots_frame(PIT_SLOTS_IN)
    with pytest.raises(ValueError):
        project_war(
            _cohort(role),
            _rates_frame(rates),
            _playing_time(role, TARGET - 1, sorted(PT[role])),
            role,
            slots,
        )


def test_models_war_project_rejects_duplicate_rates() -> None:
    rates = pd.concat([_rates_frame(BAT_RATES_IN), _rates_frame({1003: 0.0})], ignore_index=True)
    with pytest.raises(ValueError):
        project_war(_cohort("bat"), rates, _playing_time("bat", TARGET, sorted(PT["bat"])), "bat")


def test_models_war_project_rejects_nan_rate() -> None:
    # 결측 비율이 결과 표로 새지 않는다 (WAR_PROJECTIONS는 결측 불허)
    with pytest.raises(ValueError):
        project_war(
            _cohort("bat"),
            _rates_frame({**BAT_RATES_IN, 1003: NAN}),
            _playing_time("bat", TARGET, sorted(PT["bat"])),
            "bat",
        )


@pytest.mark.parametrize("case", LEAKY_CASES)
def test_models_war_project_rejects_cohort_window_outside_target(case: str) -> None:
    with pytest.raises(ValueError, match="벗어난"):
        project_war(
            _leaky(_cohort("bat"), case),
            _rates_frame(BAT_RATES_IN),
            _playing_time("bat", TARGET, sorted(PT["bat"])),
            "bat",
        )


def test_models_war_project_sorted_by_key() -> None:
    # 입력 순서가 섞여 있어도 키(player_id) 순
    cohort = _cohort("pit").iloc[::-1].reset_index(drop=True)
    rates = _rates_frame(PIT_RATES_IN).iloc[[3, 0, 5, 1, 4, 2]].reset_index(drop=True)
    pt = _playing_time("pit", TARGET, sorted(PT["pit"], reverse=True))
    slots = _slots_frame(PIT_SLOTS_IN).iloc[::-1].reset_index(drop=True)
    out = project_war(cohort, rates, pt, "pit", slots)
    _expected_projection(
        out, "pit", TARGET, PIT_RATES_IN, PT["pit"], PIT_SLOTS_IN, N_SEASONS_2025["pit"]
    )


# ---------------------------------------------------------------------------
# 한 역할·한 목표 전체 (war_projections)
# ---------------------------------------------------------------------------


def _expected_full(role: Role, target: int = TARGET) -> dict[int, float]:
    lg = _lg(role, target - 1)
    return {pid: _rate(role, target, pid, REGRESS[role], lg) for pid in HAND[(role, target)]}


def _slots_for(role: Role, pids: Sequence[int]) -> dict[int, str]:
    return dict.fromkeys(pids, "bat") if role == "bat" else {p: SLOTS_2025[p] for p in pids}


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_war_projections_hand_computed(role: Role) -> None:
    # lg = Y−1(2024) 역할 행 WAR 합 / 출전 합, R = war_params.regress[역할], w = params.<역할>.weights
    pids = sorted(HAND[(role, TARGET)])
    out = war_projections(
        _tables(),
        _war_seasons(),
        PARAMS,
        WAR_PARAMS,
        role,
        TARGET,
        playing_time=_playing_time(role, TARGET, pids),
    )
    _expected_projection(
        out,
        role,
        TARGET,
        _expected_full(role),
        PT[role],
        _slots_for(role, pids),
        N_SEASONS_2025[role],
    )


def test_models_war_projections_bat_literal_values() -> None:
    # R=100, lg=4.5/900=.005 → 1001 42.5/7500 × 550, 풀타임 × 600
    pids = sorted(HAND[("bat", TARGET)])
    out = war_projections(
        _tables(),
        _war_seasons(),
        PARAMS,
        WAR_PARAMS,
        "bat",
        TARGET,
        playing_time=_playing_time("bat", TARGET, pids),
    ).set_index("player_id")
    assert out.loc[TWO_WAY, "war"] == pytest.approx(42.5 / 7500 * 550, rel=REL)
    assert out.loc[TWO_WAY, "war_full"] == pytest.approx(42.5 / 7500 * 600, rel=REL)
    assert out.loc[1003, "war"] == pytest.approx(-1.1 / 300 * 100, rel=REL)


def test_models_war_projections_uses_tables_cohort() -> None:
    # build가 만든 대상 표가 있으면 그 행만 (tables.cohort_for)
    full = build_cohort(_seasons("bat"), "bat", TARGET)
    cohorts = full[full["player_id"].isin([TWO_WAY, 1003])].reset_index(drop=True)
    pids = [TWO_WAY, 1003]
    out = war_projections(
        _tables(cohorts=cohorts),
        _war_seasons(),
        PARAMS,
        WAR_PARAMS,
        "bat",
        TARGET,
        playing_time=_playing_time("bat", TARGET, pids),
    )
    expected = {p: v for p, v in _expected_full("bat").items() if p in pids}
    _expected_projection(
        out, "bat", TARGET, expected, PT["bat"], _slots_for("bat", pids), N_SEASONS_2025["bat"]
    )


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_war_projections_predicts_playing_time_when_missing(
    role: Role, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_playing_time(monkeypatch)
    debuts = {TWO_WAY: 2018}
    tables = _tables(debuts=debuts)
    out = war_projections(tables, _war_seasons(), PARAMS, WAR_PARAMS, role, TARGET)
    assert len(calls) == 1
    call = calls[0]
    assert call["role"] == role
    assert call["target"] == TARGET
    assert call["pids"] == sorted(HAND[(role, TARGET)])
    assert_frame_equal(call["seasons"], tables.seasons[role])
    assert dict(call["debut_seasons"] or {}) == debuts
    pids = sorted(HAND[(role, TARGET)])
    _expected_projection(
        out,
        role,
        TARGET,
        _expected_full(role),
        PT[role],
        _slots_for(role, pids),
        N_SEASONS_2025[role],
    )


def test_models_war_projections_injected_playing_time_skips_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise AssertionError("출전량을 주입했는데 모델을 불렀다")

    monkeypatch.setattr(playing_time_module, "predict_playing_time", fail)
    monkeypatch.setattr(war_module, "predict_playing_time", fail, raising=False)
    pids = sorted(HAND[("pit", TARGET)])
    out = war_projections(
        _tables(),
        _war_seasons(),
        PARAMS,
        WAR_PARAMS,
        "pit",
        TARGET,
        playing_time=_playing_time("pit", TARGET, pids),
    )
    assert out["player_id"].tolist() == pids


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_war_projections_no_leak(role: Role) -> None:
    # N2: 시즌 표·WAR 표의 season ≥ Y 행을 바꾸고 더하고 지워도 결과 비트가 같다
    pids = sorted(HAND[(role, TARGET)])
    pt = _playing_time(role, TARGET, pids)
    clean = war_projections(
        _tables(), _war_seasons(), PARAMS, WAR_PARAMS, role, TARGET, playing_time=pt
    )
    bat, pit, war = _poisoned(TARGET)
    dirty = war_projections(
        _tables(bat, pit), war, PARAMS, WAR_PARAMS, role, TARGET, playing_time=pt
    )
    assert_frame_equal(dirty, clean, check_exact=True)


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_war_projections_input_row_order_is_bit_stable(role: Role) -> None:
    # 같은 입력 같은 바이트 — 정제 표·WAR 표·출전량 표의 행 순서와 상관없다
    pids = sorted(HAND[(role, TARGET)])
    pt = _playing_time(role, TARGET, pids)
    base = war_projections(
        _tables(), _war_seasons(), PARAMS, WAR_PARAMS, role, TARGET, playing_time=pt
    )
    tables = _tables()
    shuffled = Tables(
        seasons={r: _shuffled(df, 7) for r, df in tables.seasons.items()},
        cohorts=tables.cohorts,
        league=tables.league,
        debuts=tables.debuts,
    )
    out = war_projections(
        shuffled,
        _shuffled(_war_seasons(), 8),
        PARAMS,
        WAR_PARAMS,
        role,
        TARGET,
        playing_time=_shuffled(pt, 9),
    )
    assert_frame_equal(out, base, check_exact=True)


@pytest.mark.parametrize(
    "war_params",
    [
        WarParams(version="test", regress=cast(dict[Role, float], {"bat": BAT_R})),
        WarParams(version="test", regress={"bat": BAT_R, "pit": 0.0}),
        WarParams(version="", regress={"bat": BAT_R, "pit": PIT_R}),
    ],
    ids=["missing_pit", "zero_pit", "empty_version"],
)
def test_models_war_projections_rejects_bad_war_params(war_params: WarParams) -> None:
    # 역할과 상관없이 WarParams 형식부터 본다 (타자 예측이어도 투수 R이 잘못되면 거부)
    pids = sorted(HAND[("bat", TARGET)])
    with pytest.raises(ValueError):
        war_projections(
            _tables(),
            _war_seasons(),
            PARAMS,
            war_params,
            "bat",
            TARGET,
            playing_time=_playing_time("bat", TARGET, pids),
        )


def test_models_war_projections_rejects_missing_league_season() -> None:
    # 목표 Y−1의 WAR 행이 없으면 리그 비율을 낼 수 없다 — 조용히 0으로 두지 않는다
    war = _war_seasons()
    war = war[~((war["role"] == "bat") & (war["season"] == TARGET - 1))].reset_index(drop=True)
    pids = sorted(HAND[("bat", TARGET)])
    with pytest.raises(ValueError):
        war_projections(
            _tables(),
            war,
            PARAMS,
            WAR_PARAMS,
            "bat",
            TARGET,
            playing_time=_playing_time("bat", TARGET, pids),
        )


# ---------------------------------------------------------------------------
# 투타 합산 순위 (D34)
# ---------------------------------------------------------------------------


def _proj(role: Role, target: int, rows: Sequence[tuple[int, float, float]]) -> pd.DataFrame:
    """(player_id, war, war_full) → `WAR_PROJECTIONS` 표 (비율·출전량은 서로 맞게 채운다)."""
    slot = "bat" if role == "bat" else "sp"
    units = c.FULL_TIME_UNITS[slot]
    records: list[dict[str, Any]] = []
    for pid, war, war_full in rows:
        rate = war_full / units
        records.append(
            {
                "player_id": pid,
                "role": role,
                "target_season": target,
                "n_seasons": 3,
                "slot": slot,
                "war_rate": rate,
                "pt": war / rate,
                "war": war,
                "full_units": units,
                "war_full": war_full,
            }
        )
    return _frame(c.WAR_PROJECTIONS, records)


COMBINE_BAT = [(7, 4.0, 4.4), (1, 5.0, 5.5), (5, 3.0, 4.0)]
COMBINE_PIT = [(7, 2.5, 3.5), (3, 3.0, 3.6), (4, 1.0, 2.0)]
# Y−1(2024) 실제 WAR: 7은 두 역할 합 3.25 + 1.5, 1은 0.0(결측 아님), 3은 2025 행(9.0)을 더하면 안 된다,
# 5는 2023에만, 4는 2025에만 → 결측
COMBINE_WAR: list[tuple[int, int, str, float]] = [
    (7, 2024, "bat", 3.25),
    (7, 2024, "pit", 1.5),
    (1, 2024, "bat", 0.0),
    (3, 2024, "pit", 2.0),
    (5, 2023, "bat", 2.0),
    (3, 2025, "pit", 9.0),
    (4, 2025, "pit", 1.1),
    (1, 2025, "bat", 7.0),
]
RankRow = tuple[int, str, float, float, float, float, float, float, float]
# (player_id, roles, war_bat, war_pit, war, war_full_bat, war_full_pit, war_full, prev_war)
# 정렬: war 내림차순, 같으면 player_id — 3과 5는 war 3.0 동률(war_full로 정렬하면 5가 앞선다)
COMBINE_EXPECTED: list[RankRow] = [
    (7, "both", 4.0, 2.5, 6.5, 4.4, 3.5, 7.9, 4.75),
    (1, "bat", 5.0, NAN, 5.0, 5.5, NAN, 5.5, 0.0),
    (3, "pit", NAN, 3.0, 3.0, NAN, 3.6, 3.6, 2.0),
    (5, "bat", 3.0, NAN, 3.0, 4.0, NAN, 4.0, NAN),
    (4, "pit", NAN, 1.0, 1.0, NAN, 2.0, 2.0, NAN),
]
RANK_VALUE_COLUMNS = (
    "war_bat",
    "war_pit",
    "war",
    "war_full_bat",
    "war_full_pit",
    "war_full",
    "prev_war",
)


def _assert_rankings(out: pd.DataFrame, expected: Sequence[RankRow], target: int) -> None:
    assert list(out.columns) == c.WAR_RANKINGS.column_names
    assert c.check_frame(out, c.WAR_RANKINGS) == []
    assert out["player_id"].tolist() == [row[0] for row in expected]
    assert out["roles"].tolist() == [row[1] for row in expected]
    assert out["target_season"].tolist() == [target] * len(expected)
    for got, row in zip(out.to_dict("records"), expected, strict=True):
        for name, value in zip(RANK_VALUE_COLUMNS, row[2:], strict=True):
            assert _same(got[name], value), (row[0], name, got[name], value)


def test_models_war_combine_roles_hand_computed() -> None:
    out = combine_roles(
        _proj("bat", TARGET, COMBINE_BAT),
        _proj("pit", TARGET, COMBINE_PIT),
        _war_frame(COMBINE_WAR),
        TARGET,
    )
    _assert_rankings(out, COMBINE_EXPECTED, TARGET)


def test_models_war_combine_roles_input_order_does_not_matter() -> None:
    out = combine_roles(
        _proj("bat", TARGET, COMBINE_BAT[::-1]),
        _proj("pit", TARGET, COMBINE_PIT[::-1]),
        _war_frame(COMBINE_WAR[::-1]),
        TARGET,
    )
    _assert_rankings(out, COMBINE_EXPECTED, TARGET)


def test_models_war_combine_roles_bat_only() -> None:
    out = combine_roles(_proj("bat", TARGET, COMBINE_BAT), None, _war_frame(COMBINE_WAR), TARGET)
    # 7의 prev_war는 두 역할 합 그대로(투수 예측이 없어도 Y−1 실제 WAR은 투타 합)
    expected: list[RankRow] = [
        (1, "bat", 5.0, NAN, 5.0, 5.5, NAN, 5.5, 0.0),
        (7, "bat", 4.0, NAN, 4.0, 4.4, NAN, 4.4, 4.75),
        (5, "bat", 3.0, NAN, 3.0, 4.0, NAN, 4.0, NAN),
    ]
    _assert_rankings(out, expected, TARGET)


def test_models_war_combine_roles_pit_only() -> None:
    out = combine_roles(None, _proj("pit", TARGET, COMBINE_PIT), _war_frame(COMBINE_WAR), TARGET)
    expected: list[RankRow] = [
        (3, "pit", NAN, 3.0, 3.0, NAN, 3.6, 3.6, 2.0),
        (7, "pit", NAN, 2.5, 2.5, NAN, 3.5, 3.5, 4.75),
        (4, "pit", NAN, 1.0, 1.0, NAN, 2.0, 2.0, NAN),
    ]
    _assert_rankings(out, expected, TARGET)


@pytest.mark.parametrize(
    ("bat_target", "pit_target", "target"),
    [(2024, 2025, 2025), (2025, 2024, 2025), (2025, 2025, 2024)],
    ids=["bat_differs", "pit_differs", "both_differ"],
)
def test_models_war_combine_roles_rejects_target_mismatch(
    bat_target: int, pit_target: int, target: int
) -> None:
    with pytest.raises(ValueError):
        combine_roles(
            _proj("bat", bat_target, COMBINE_BAT),
            _proj("pit", pit_target, COMBINE_PIT),
            _war_frame(COMBINE_WAR),
            target,
        )


def test_models_war_combine_roles_rejects_single_input_mismatch() -> None:
    with pytest.raises(ValueError):
        combine_roles(None, _proj("pit", 2024, COMBINE_PIT), _war_frame(COMBINE_WAR), TARGET)


def test_models_war_combine_roles_rejects_no_inputs() -> None:
    with pytest.raises(ValueError):
        combine_roles(None, None, _war_frame(COMBINE_WAR), TARGET)


def test_models_war_combine_roles_rejects_swapped_roles() -> None:
    # 투수 예측을 타자 자리에 넘기면(또는 반대) 역할 표기가 틀린 순위가 나온다 — 막는다
    with pytest.raises(ValueError):
        combine_roles(_proj("pit", TARGET, COMBINE_PIT), None, _war_frame(COMBINE_WAR), TARGET)
    with pytest.raises(ValueError):
        combine_roles(None, _proj("bat", TARGET, COMBINE_BAT), _war_frame(COMBINE_WAR), TARGET)


def test_models_war_combine_roles_no_leak() -> None:
    # prev_war는 Y−1 행만 — season ≥ Y 행을 바꾸고 더해도 결과 비트가 같다
    bat, pit = _proj("bat", TARGET, COMBINE_BAT), _proj("pit", TARGET, COMBINE_PIT)
    clean = combine_roles(bat, pit, _war_frame(COMBINE_WAR), TARGET)
    poisoned = [
        (pid, season, role, POISON_WAR if season >= TARGET else war)
        for pid, season, role, war in COMBINE_WAR
    ] + [(5, TARGET, "bat", POISON_WAR), (9, TARGET, "pit", POISON_WAR)]
    dirty = combine_roles(bat, pit, _war_frame(poisoned), TARGET)
    assert_frame_equal(dirty, clean, check_exact=True)


# ---------------------------------------------------------------------------
# WAR 회귀 표본 조정 (D36)
# ---------------------------------------------------------------------------


def test_models_war_tune_oracle_choice_is_interior() -> None:
    # 도우미 확인: 고를 값이 격자의 가장 작은·큰 R이 아니고, 목표별 최솟값과 다르다
    for role in c.ROLES:
        assert _oracle_choice(role, TUNE_GRID[role], c.TUNE_TARGETS) == TUNE_CHOSEN[role]
        assert min(TUNE_GRID[role]) < TUNE_CHOSEN[role] < max(TUNE_GRID[role])
    best_2025 = min(TUNE_GRID["bat"], key=lambda r: _oracle_mae("bat", 2025, r))
    assert best_2025 != TUNE_CHOSEN["bat"]


def test_models_war_tune_picks_lowest_mean_mae(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_playing_time(monkeypatch)
    chosen, _ = tune_war(_tables(), _war_seasons(), PARAMS, grid=TUNE_GRID)
    assert isinstance(chosen, WarParams)
    assert chosen.version == "v1"
    assert chosen.regress == TUNE_CHOSEN


def test_models_war_tune_report_table(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_playing_time(monkeypatch)
    _, report = tune_war(_tables(), _war_seasons(), PARAMS, c.TUNE_TARGETS, TUNE_GRID)
    assert list(report.columns) == REPORT_COLUMNS
    expected: list[tuple[str, float, int, float, float, bool]] = []
    for role in c.ROLES:
        for regress in sorted(TUNE_GRID[role]):
            maes = [_oracle_mae(role, t, regress) for t in c.TUNE_TARGETS]
            mean = sum(maes) / len(maes)
            for target, mae in zip(c.TUNE_TARGETS, maes, strict=True):
                expected.append((role, regress, target, mae, mean, regress == TUNE_CHOSEN[role]))
    assert len(report) == len(expected)
    assert report["role"].tolist() == [e[0] for e in expected]
    assert report["regress"].tolist() == pytest.approx([e[1] for e in expected])
    assert report["target_season"].tolist() == [e[2] for e in expected]
    assert report["mae"].tolist() == pytest.approx([e[3] for e in expected], rel=1e-9)
    assert report["mean_mae"].tolist() == pytest.approx([e[4] for e in expected], rel=1e-9)
    assert report["chosen"].tolist() == [e[5] for e in expected]
    assert pd.api.types.is_bool_dtype(report["chosen"])
    assert pd.api.types.is_integer_dtype(report["target_season"])
    assert pd.api.types.is_float_dtype(report["regress"])


def test_models_war_tune_targets_order_does_not_matter(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_playing_time(monkeypatch)
    chosen_a, report_a = tune_war(_tables(), _war_seasons(), PARAMS, (2024, 2025), TUNE_GRID)
    chosen_b, report_b = tune_war(_tables(), _war_seasons(), PARAMS, (2025, 2024), TUNE_GRID)
    assert chosen_a == chosen_b
    assert_frame_equal(report_a, report_b)


def test_models_war_tune_tie_picks_smaller_regress(monkeypatch: pytest.MonkeyPatch) -> None:
    # 2025 이전 WAR이 모두 0이면 비율·lg가 R과 무관하게 정확히 0 → MAE가 모든 후보에서 같다
    def zeroed(world: World) -> dict[int, dict[int, tuple[Any, ...]]]:
        return {
            pid: {s: (*v[:-1], 0.0) if s < 2025 else v for s, v in by_season.items()}
            for pid, by_season in world.items()
        }

    bat, pit = zeroed(BAT_WORLD), zeroed(PIT_WORLD)
    _install_fake_playing_time(monkeypatch)
    grid: dict[Role, tuple[float, ...]] = {
        "bat": (300.0, 30.0, 3000.0),
        "pit": (200.0, 20.0, 2000.0),
    }
    chosen, report = tune_war(_tables(bat, pit), _war_seasons(bat, pit), PARAMS, grid=grid)
    assert chosen.regress == {"bat": 30.0, "pit": 20.0}
    for role in c.ROLES:
        rows = report[report["role"] == role]
        assert rows["mean_mae"].nunique() == 1
        assert rows.loc[rows["chosen"], "regress"].unique().tolist() == [min(grid[role])]


def test_models_war_tune_predicts_playing_time_once_per_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_playing_time(monkeypatch)
    tune_war(_tables(), _war_seasons(), PARAMS, grid=TUNE_GRID)
    # 후보 세 개 × 목표 둘 × 역할 둘이어도 출전량 예측은 (역할, 목표)마다 한 번
    assert len(calls) == len(c.ROLES) * len(c.TUNE_TARGETS)
    assert sorted((call["role"], call["target"]) for call in calls) == [
        ("bat", 2024),
        ("bat", 2025),
        ("pit", 2024),
        ("pit", 2025),
    ]
    for call in calls:
        assert call["pids"] == sorted(HAND[(call["role"], call["target"])])


def test_models_war_tune_default_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_playing_time(monkeypatch)
    chosen, report = tune_war(_tables(), _war_seasons(), PARAMS)
    for role in c.ROLES:
        rows = report[report["role"] == role]
        assert sorted(set(rows["regress"])) == sorted(c.WAR_TUNE_GRID[role])
        assert sorted(set(rows["target_season"])) == list(c.TUNE_TARGETS)
        assert chosen.regress[role] == _oracle_choice(role, c.WAR_TUNE_GRID[role], c.TUNE_TARGETS)
    assert len(report) == sum(len(g) for g in c.WAR_TUNE_GRID.values()) * len(c.TUNE_TARGETS)


def test_models_war_tune_ignores_rows_after_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    # 목표 2024·2025 조정은 2026 이후 행을 쓰지 않는다 (2026 시즌 행을 바꾸고 더하고 지워도 같다)
    _install_fake_playing_time(monkeypatch)
    clean = tune_war(_tables(), _war_seasons(), PARAMS, grid=TUNE_GRID)
    bat, pit, war = _poisoned(2026)
    dirty = tune_war(_tables(bat, pit), war, PARAMS, grid=TUNE_GRID)
    assert dirty[0] == clean[0]
    assert_frame_equal(dirty[1], clean[1], check_exact=True)


def _empty_tables() -> Tables:
    return Tables(seasons={}, cohorts=_empty_cohorts(), league={}, debuts={})


@pytest.mark.parametrize("targets", [(2024, 2026), (2025, 2026)], ids=["2024_2026", "2025_2026"])
def test_models_war_tune_rejects_final_target_with_empty_tables(
    targets: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    # 빈 Tables로도 "대상 표가 없다"(2024·2025)보다 2026 거부가 먼저 난다
    _install_fake_playing_time(monkeypatch)
    with pytest.raises(ValueError, match="2026"):
        tune_war(_empty_tables(), _war_frame([]), PARAMS, targets, TUNE_GRID)


class _Untouchable:
    """읽으면 실패하는 자리표시 — 오류가 데이터를 읽기 전에 나는지 본다."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"데이터를 읽었다: {name}")

    def __getitem__(self, key: Any) -> Any:
        raise AssertionError(f"데이터를 읽었다: [{key!r}]")

    def __iter__(self) -> Iterator[Any]:
        raise AssertionError("데이터를 읽었다: iter")

    def __len__(self) -> int:
        raise AssertionError("데이터를 읽었다: len")


@pytest.mark.parametrize(
    "targets",
    [(2026,), (2024, 2026), (2024, 2027), (2027,)],
    ids=["2026", "2024_2026", "2024_2027", "2027"],
)
def test_models_war_tune_rejects_final_target_before_reading(
    targets: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise AssertionError("출전량 모델을 불렀다")

    monkeypatch.setattr(playing_time_module, "predict_playing_time", fail)
    monkeypatch.setattr(war_module, "predict_playing_time", fail, raising=False)
    untouchable_tables = cast(Tables, _Untouchable())
    untouchable_war = cast(pd.DataFrame, _Untouchable())
    with pytest.raises(ValueError):
        tune_war(untouchable_tables, untouchable_war, PARAMS, targets, TUNE_GRID)


def test_models_war_tune_rejects_final_target_from_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # targets는 Iterable — 한 번만 도는 생성기여도 2026을 거부한다
    _install_fake_playing_time(monkeypatch)
    targets = (t for t in (2024, 2026))
    with pytest.raises(ValueError, match="2026"):
        tune_war(_empty_tables(), _war_frame([]), PARAMS, targets, TUNE_GRID)


@pytest.mark.parametrize(
    ("targets", "grid"),
    [
        ((), TUNE_GRID),
        (c.TUNE_TARGETS, {"bat": TUNE_GRID["bat"]}),
        (c.TUNE_TARGETS, {"bat": TUNE_GRID["bat"], "pit": TUNE_GRID["pit"], "two": (1.0,)}),
        (c.TUNE_TARGETS, {**TUNE_GRID, "pit": ()}),
        (c.TUNE_TARGETS, {**TUNE_GRID, "bat": (300.0, 0.0)}),
        (c.TUNE_TARGETS, {**TUNE_GRID, "pit": (200.0, -1.0)}),
        (c.TUNE_TARGETS, {**TUNE_GRID, "bat": (300.0, NAN)}),
    ],
    ids=[
        "no_targets",
        "grid_missing_pit",
        "grid_extra_role",
        "grid_empty",
        "grid_zero",
        "grid_negative",
        "grid_nan",
    ],
)
def test_models_war_tune_rejects_bad_targets_or_grid(
    targets: tuple[int, ...], grid: Mapping[str, tuple[float, ...]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_playing_time(monkeypatch)
    with pytest.raises(ValueError):
        tune_war(
            _tables(),
            _war_seasons(),
            PARAMS,
            targets,
            cast(Mapping[Role, tuple[float, ...]], grid),
        )


def test_models_war_tune_duplicate_grid_values_report_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 격자에 같은 R이 두 번 있어도 보고 표에는 한 번 (키 role·regress·target_season 중복 없음)
    _install_fake_playing_time(monkeypatch)
    grid: dict[Role, tuple[float, ...]] = {
        "bat": (*TUNE_GRID["bat"], 3000.0),
        "pit": (*TUNE_GRID["pit"], 200.0),
    }
    chosen, report = tune_war(_tables(), _war_seasons(), PARAMS, grid=grid)
    assert chosen.regress == TUNE_CHOSEN
    assert not report.duplicated(["role", "regress", "target_season"]).any()
    assert len(report) == sum(len(set(g)) for g in grid.values()) * len(c.TUNE_TARGETS)


def test_models_war_params_default_is_valid_json() -> None:
    # 기본값(출발점)도 형식 규칙을 지킨다 — 조정 전 war_projections에 그대로 쓸 수 있다
    assert war_params_from_json(war_params_to_json(DEFAULT_WAR_PARAMS)) == DEFAULT_WAR_PARAMS
    assert set(DEFAULT_WAR_PARAMS.regress) == set(c.ROLES)
    validate_params(PARAMS)
