"""WAR 검증·2026 WAR 시험·war.json(war_eval.py) 테스트 (SPEC v0.5 F20·F21·F22·D33~D36·D39).

규칙: 손으로 만든 합성 표와 합성 샘플(`tests/fixtures/synthetic/web/war.json`)만 쓴다. 실데이터(`data/`)를
읽지 않는다. 목표 2026은 ① 과거 검증이 거부하는지 ② 가짜 평가로 감싼 `run_final_test_war`의 기록 장치만
본다 — 2026 WAR 계산은 합성 세계에서도 하지 않는다(실제 시험은 메인이 한 번만 돌린다).

바꿔 끼우는 곳 — `models.war`(레인 B)는 아직 스텁이다
- `_patch_war`는 같은 가짜를 `mlbproj.models.war`의 속성과 `mlbproj.war_eval`의 같은 이름에 함께 끼운다.
  war_eval이 `from mlbproj.models import war` 뒤 `war.war_projections(...)`로 부르든,
  `from mlbproj.models.war import war_projections`로 부르든 가짜가 불린다.
- 가짜: `war_projections`(정해 둔 예측 표), `league_war_rate`(스텁 docstring의 식 그대로 계산),
  WAR 파라미터 입출력(`war_params_to_json`·`war_params_from_json`·`war_params_sha256`·`save_war_params`·
  `load_war_params` — 정렬 JSON과 그 UTF-8 sha256).
- `run_final_test_war` 시험은 `mlbproj.war_eval.evaluate_war`도 가짜로 바꾼다(모듈 전역 이름으로 부른다).
- git은 `final.committed_head`가 부르는 세 명령(rev-parse·ls-files·status --porcelain)만 흉내 낸다.

합성 세계 (`BAT_ROWS`·`PIT_ROWS`·`WAR_ROWS`, 출전 = 타자 PA · 투수 아웃)
- 2024 리그 WAR 비율은 일부러 딱 떨어지게 맞췄다: 타자 6.0 / 2,000 PA = 0.003, 투수 4.0 / 2,000 아웃 = 0.002
  (채우기 선수 98·97). 그래서 목표 2025 과거 검증의 리그 기준선은 0.003 × pt · 0.002 × pt다.
- 타자(목표 2025 예측): 1(3시즌) · 2(1) · 3(2, 2025 미출전 → 실제 0) · 4(5, 2024 미출전, 2025 499 PA·1.99 WAR —
  두 경계 바로 아래) · 5(4, 2025 WAR −0.5) · 6(3, 2024 미출전 → 주전급 제외). 2는 2025 500 PA·2.0 WAR(두 경계).
  99는 예측 대상이 아니다. 1의 투수 역할 행(2025 0.7)은 타자 검증에 섞이지 않는다.
- 투수(목표 2025 예측): 11(sp, 540아웃) · 12(rp, 180아웃) · 13(sp, 2025 500아웃이지만 2024 0아웃 → 주전급 제외) ·
  14(rp, 2025 450아웃 = 경계, 풀타임 195아웃) · 15(rp, 기록 없음 → 실제 0). 실제 WAR 2 이상이 없다(n = 0).
"""

import hashlib
import json
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlbproj import contracts as c
from mlbproj import war_eval
from mlbproj.contracts import Role
from mlbproj.models import war as war_model
from mlbproj.models.league import league_batting, league_pitching
from mlbproj.models.params import DEFAULT_PARAMS, Params, params_sha256, save_params
from mlbproj.models.war import WarParams
from mlbproj.schema import load_schema, validate
from mlbproj.tables import Tables
from mlbproj.war_eval import WarExportInput, WarTestError

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_WAR = ROOT / "tests" / "fixtures" / "synthetic" / "web" / "war.json"

PARAMS: Params = replace(DEFAULT_PARAMS, version="v1")
PARAMS2: Params = replace(DEFAULT_PARAMS, version="v2")
WAR_PARAMS = WarParams(version="v1", regress={"bat": 1200.0, "pit": 600.0})
WAR_PARAMS2 = WarParams(version="v1", regress={"bat": 600.0, "pit": 600.0})
HEAD = "0123456789abcdef0123456789abcdef01234567"
HEAD2 = "89abcdef0123456789abcdef0123456789abcdef"
NOW1 = "2026-11-25T01:02:03Z"
NOW2 = "2026-11-26T04:05:06Z"
NOW3 = "2026-11-27T07:08:09Z"
RECORD_KEYS = [
    "target_season",
    "ran_at",
    "params_sha256",
    "war_params_sha256",
    "commit",
    "reruns",
    "rows",
]
LG_BAT_2024 = 0.003  # 6.0 WAR / 2,000 PA
LG_PIT_2024 = 0.002  # 4.0 WAR / 2,000 아웃
_KEYS = list(c.WAR_EVALUATIONS.keys)

# ---------------------------------------------------------------------------
# 도우미 — 계약 형식의 합성 표
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {"league": c.LEAGUE, "team_name": "TST", "age": 27, "n_teams": 1}
_ADJUSTED = {"pa_adj": "pa", "outs_adj": "outs"}


def _dtype(col: c.Column) -> str:
    if col.kind == "int":
        return "Int64" if col.nullable else "int64"
    if col.kind == "float":
        return "float64"
    if col.kind == "bool":
        return "bool"
    return "str"


def _value(col: c.Column, row: Mapping[str, Any]) -> Any:
    """행에 없는 열의 기본값. 정수 0, 결측 허용 열은 결측, 2020 출전량은 ×162/60."""
    name = col.name
    if name in row:
        return row[name]
    if name in _DEFAULTS:
        return _DEFAULTS[name]
    if name == "is_short_season":
        return int(row["season"]) == c.SHORT_SEASON
    if name in _ADJUSTED:
        scale = c.SHORT_SEASON_SCALE if int(row["season"]) == c.SHORT_SEASON else 1.0
        return float(row.get(_ADJUSTED[name], 0)) * scale
    if col.kind == "int" and not col.nullable:
        return 0
    if col.nullable:
        return None
    raise AssertionError(f"채울 수 없는 열: {name}")


def _frame(spec: c.TableSpec, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """계약의 모든 열을 채운 표 (`check_frame` 통과를 확인한다)."""
    df = pd.DataFrame(
        {
            col.name: pd.Series([_value(col, row) for row in rows], dtype=_dtype(col))
            for col in spec.columns
        }
    )
    assert c.check_frame(df, spec) == []
    return df


def _bat(pid: int, season: int, pa: int) -> dict[str, Any]:
    return {"player_id": pid, "season": season, "pa": pa, "ab": pa}


def _pit(pid: int, season: int, outs: int, g: int = 20, gs: int = 0) -> dict[str, Any]:
    return {"player_id": pid, "season": season, "outs": outs, "bf": outs + 10, "g": g, "gs": gs}


def _war(pid: int, season: int, role: Role, war: float) -> dict[str, Any]:
    return {"player_id": pid, "season": season, "role": role, "war": war}


BAT_ROWS = [
    _bat(1, 2023, 580),
    _bat(3, 2023, 300),
    _bat(6, 2023, 400),
    _bat(1, 2024, 600),
    _bat(2, 2024, 200),
    _bat(3, 2024, 450),
    _bat(5, 2024, 100),
    _bat(98, 2024, 650),  # 채우기 — 2024 리그 비율을 0.003으로
    _bat(1, 2025, 620),
    _bat(2, 2025, 500),  # 주전급 경계(500 PA) 포함
    _bat(4, 2025, 499),  # 주전급 경계 바로 아래
    _bat(5, 2025, 650),
    _bat(6, 2025, 560),
    _bat(99, 2025, 600),  # 예측 대상 아님
]
PIT_ROWS = [
    _pit(11, 2023, 400),
    _pit(14, 2023, 150),
    _pit(11, 2024, 450),
    _pit(12, 2024, 120),
    _pit(13, 2024, 0),  # 기록은 있지만 0아웃 — "Y−1 출전 > 0"이 아니다
    _pit(14, 2024, 200),
    _pit(97, 2024, 1230),  # 채우기 — 2024 리그 비율을 0.002로
    _pit(1, 2025, 60),  # 타자 1의 투수 역할 (투수 예측 대상 아님)
    _pit(11, 2025, 540, gs=20),
    _pit(12, 2025, 180),
    _pit(13, 2025, 500, gs=15),
    _pit(14, 2025, 450),  # 주전급 경계(450아웃) 포함
]
WAR_ROWS = [
    _war(1, 2023, "bat", 3.0),
    _war(3, 2023, "bat", 0.6),
    _war(6, 2023, "bat", 2.0),
    _war(1, 2024, "bat", 3.5),
    _war(2, 2024, "bat", 0.5),
    _war(3, 2024, "bat", 1.2),
    _war(5, 2024, "bat", 0.3),
    _war(98, 2024, "bat", 0.5),
    _war(1, 2025, "bat", 4.0),
    _war(2, 2025, "bat", 2.0),
    _war(4, 2025, "bat", 1.99),
    _war(5, 2025, "bat", -0.5),
    _war(6, 2025, "bat", 2.8),
    _war(99, 2025, "bat", 5.0),
    _war(11, 2023, "pit", 1.5),
    _war(14, 2023, "pit", 0.2),
    _war(11, 2024, "pit", 2.5),
    _war(12, 2024, "pit", -0.1),
    _war(13, 2024, "pit", -0.2),
    _war(14, 2024, "pit", 0.6),
    _war(97, 2024, "pit", 1.2),
    _war(1, 2025, "pit", 0.7),
    _war(11, 2025, "pit", 1.9),
    _war(12, 2025, "pit", 0.5),
    _war(13, 2025, "pit", 1.5),
    _war(14, 2025, "pit", 1.8),
]


def _world() -> Tables:
    """합성 세계의 정제 표 묶음 (모듈 docstring)."""
    batting = _frame(c.BATTING_SEASONS, BAT_ROWS)
    pitching = _frame(c.PITCHING_SEASONS, PIT_ROWS)
    return Tables(
        seasons={"bat": batting, "pit": pitching},
        cohorts=_frame(c.COHORTS, []),
        league={"bat": league_batting(batting), "pit": league_pitching(pitching)},
        debuts={},
    )


def _war_seasons() -> pd.DataFrame:
    return _frame(c.WAR_SEASONS, WAR_ROWS)


# (player_id, n_seasons, slot, pt, war, full_units, war_full)
ProjRow = tuple[int, int, str, float, float, float, float]
PROJ_ROWS: dict[tuple[Role, int], list[ProjRow]] = {
    ("bat", 2025): [
        (1, 3, "bat", 600.0, 3.0, 600.0, 3.0),
        (2, 1, "bat", 300.0, 1.0, 600.0, 2.0),
        (3, 2, "bat", 400.0, 1.5, 600.0, 2.25),
        (4, 5, "bat", 200.0, 0.4, 600.0, 1.2),
        (5, 4, "bat", 500.0, 2.0, 600.0, 2.4),
        (6, 3, "bat", 450.0, 1.35, 600.0, 1.8),
    ],
    ("pit", 2025): [
        (11, 3, "sp", 480.0, 2.88, 540.0, 3.24),
        (12, 1, "rp", 150.0, 0.6, 195.0, 0.78),
        (13, 2, "sp", 300.0, 0.9, 540.0, 1.62),
        (14, 4, "rp", 200.0, 1.0, 195.0, 0.975),
        (15, 5, "rp", 90.0, 0.09, 195.0, 0.195),
    ],
    ("bat", 2024): [
        (1, 3, "bat", 550.0, 2.75, 600.0, 3.0),
        (2, 1, "bat", 250.0, 0.5, 600.0, 1.2),
        (3, 2, "bat", 350.0, 0.7, 600.0, 1.2),
        (5, 3, "bat", 150.0, 0.3, 600.0, 1.2),
    ],
    ("pit", 2024): [
        (11, 2, "sp", 420.0, 2.1, 540.0, 2.7),
        (12, 1, "rp", 100.0, 0.2, 195.0, 0.39),
        (13, 1, "rp", 60.0, 0.0, 195.0, 0.0),
        (14, 2, "rp", 150.0, 0.45, 195.0, 0.585),
    ],
}
# 2026 시험용 가짜 예측 (가짜 평가가 받기만 한다)
PROJ_ROWS[("bat", 2026)] = [(1, 3, "bat", 600.0, 3.0, 600.0, 3.0)]
PROJ_ROWS[("pit", 2026)] = [(11, 3, "sp", 480.0, 2.88, 540.0, 3.24)]


def _proj(role: Role, target: int, rows: Sequence[ProjRow] | None = None) -> pd.DataFrame:
    """`WAR_PROJECTIONS` 표. war_rate = war / pt."""
    chosen = PROJ_ROWS[(role, target)] if rows is None else rows
    return _frame(
        c.WAR_PROJECTIONS,
        [
            {
                "player_id": pid,
                "role": role,
                "target_season": target,
                "n_seasons": n,
                "slot": slot,
                "war_rate": war / pt if pt else 0.0,
                "pt": pt,
                "war": war,
                "full_units": full,
                "war_full": war_full,
            }
            for pid, n, slot, pt, war, full, war_full in chosen
        ],
    )


def _ev(
    target: int,
    role: str,
    metric: str,
    group: str,
    n: int,
    mae: float | None,
    prev: float | None,
    league: float | None,
) -> dict[str, Any]:
    """`WAR_EVALUATIONS` 한 행 (열 순서 그대로)."""
    return {
        "target_season": target,
        "role": role,
        "metric": metric,
        "group": group,
        "n": n,
        "mae": mae,
        "baseline_mae_prev": prev,
        "baseline_mae_league": league,
    }


def _expected_2025(role: Role) -> pd.DataFrame:
    """목표 2025 손 계산 (리그 비율 = 2024 값, 모듈 docstring의 세계)."""
    if role == "bat":
        # 선수별 |예측 − 실제| / |작년 − 실제| / |0.003·pt − 실제|
        # 1: 1.0 / 0.5 / 2.2   2: 1.0 / 1.5 / 1.1   3: 1.5 / 1.2 / 1.2
        # 4: 1.59 / 1.99 / 1.39   5: 2.5 / 0.8 / 2.0   6: 1.45 / 2.8 / 1.45
        act1, act2, act5 = 4.0 / 620 * 600, 2.0 / 500 * 600, -0.5 / 650 * 600
        rows = [
            _ev(
                2025,
                "bat",
                "war",
                "all",
                6,
                (1.0 + 1.0 + 1.5 + 1.59 + 2.5 + 1.45) / 6,
                (0.5 + 1.5 + 1.2 + 1.99 + 0.8 + 2.8) / 6,
                (2.2 + 1.1 + 1.2 + 1.39 + 2.0 + 1.45) / 6,
            ),
            _ev(2025, "bat", "war", "seasons_1_2", 2, 1.25, 1.35, 1.15),  # 2·3
            _ev(
                2025,
                "bat",
                "war",
                "seasons_3_5",
                4,  # 1·4·5·6
                (1.0 + 1.59 + 2.5 + 1.45) / 4,
                (0.5 + 1.99 + 0.8 + 2.8) / 4,
                (2.2 + 1.39 + 2.0 + 1.45) / 4,
            ),
            _ev(
                2025,
                "bat",
                "war",
                "war2plus",
                3,  # 1(4.0)·2(2.0)·6(2.8), 4(1.99)는 빠진다
                (1.0 + 1.0 + 1.45) / 3,
                (0.5 + 1.5 + 2.8) / 3,
                (2.2 + 1.1 + 1.45) / 3,
            ),
            # 주전급 1·2·5 (실제·작년은 600 PA당, 리그는 0.003 × 600)
            _ev(
                2025,
                "bat",
                "war_full",
                "regulars",
                3,
                (abs(3.0 - act1) + abs(2.0 - act2) + abs(2.4 - act5)) / 3,
                (
                    abs(3.5 / 600 * 600 - act1)
                    + abs(0.5 / 200 * 600 - act2)
                    + abs(0.3 / 100 * 600 - act5)
                )
                / 3,
                (abs(1.8 - act1) + abs(1.8 - act2) + abs(1.8 - act5)) / 3,
            ),
        ]
    else:
        # 11: 0.98 / 0.6 / 0.94   12: 0.1 / 0.6 / 0.2   13: 0.6 / 1.7 / 0.9
        # 14: 0.8 / 1.2 / 1.4   15: 0.09 / 0.0 / 0.18
        act11, act14 = 1.9 / 540 * 540, 1.8 / 450 * 195
        rows = [
            _ev(
                2025,
                "pit",
                "war",
                "all",
                5,
                (0.98 + 0.1 + 0.6 + 0.8 + 0.09) / 5,
                (0.6 + 0.6 + 1.7 + 1.2 + 0.0) / 5,
                (0.94 + 0.2 + 0.9 + 1.4 + 0.18) / 5,
            ),
            _ev(2025, "pit", "war", "seasons_1_2", 2, 0.35, 1.15, 0.55),  # 12·13
            _ev(
                2025,
                "pit",
                "war",
                "seasons_3_5",
                3,  # 11·14·15
                (0.98 + 0.8 + 0.09) / 3,
                (0.6 + 1.2 + 0.0) / 3,
                (0.94 + 1.4 + 0.18) / 3,
            ),
            _ev(2025, "pit", "war", "war2plus", 0, None, None, None),  # 실제 2 이상 없음
            # 주전급 11·14 (14는 불펜 195아웃으로 환산)
            _ev(
                2025,
                "pit",
                "war_full",
                "regulars",
                2,
                (abs(3.24 - act11) + abs(0.975 - act14)) / 2,
                (abs(2.5 / 450 * 540 - act11) + abs(0.6 / 200 * 195 - act14)) / 2,
                (abs(0.002 * 540 - act11) + abs(0.002 * 195 - act14)) / 2,
            ),
        ]
    return _frame(c.WAR_EVALUATIONS, rows)


def _assert_evaluations(out: pd.DataFrame, expected: pd.DataFrame) -> None:
    """계약(열 순서·종류) + 손 계산 값 + 키 순."""
    assert list(out.columns) == c.WAR_EVALUATIONS.column_names
    assert c.check_frame(out, c.WAR_EVALUATIONS) == []
    assert_frame_equal(
        out.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-9,
        atol=1e-12,
    )


def _lg(war_seasons: pd.DataFrame, seasons: pd.DataFrame, role: Role, season: int) -> float:
    """리그 WAR 비율 (`league_war_rate` docstring의 식 그대로)."""
    mine = war_seasons[(war_seasons["role"] == role) & (war_seasons["season"] == season)]
    units = seasons.loc[seasons["season"] == season, c.WAR_UNIT[role]]
    return float(mine["war"].sum()) / float(units.sum())


# ---------------------------------------------------------------------------
# models.war 가짜
# ---------------------------------------------------------------------------


def _fake_war_json(p: WarParams) -> str:
    doc = {"regress": {r: float(p.regress[r]) for r in sorted(p.regress)}, "version": p.version}
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def _fake_war_from_json(text: str) -> WarParams:
    doc = json.loads(text)
    return WarParams(
        version=str(doc["version"]),
        regress={"bat": float(doc["regress"]["bat"]), "pit": float(doc["regress"]["pit"])},
    )


def _fake_war_sha(p: WarParams) -> str:
    return hashlib.sha256(_fake_war_json(p).encode("utf-8")).hexdigest()


def _fake_save_war(p: WarParams, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_fake_war_json(p), encoding="utf-8")


def _fake_load_war(path: Path) -> WarParams:
    return _fake_war_from_json(path.read_text(encoding="utf-8"))


def _patch_war(monkeypatch: pytest.MonkeyPatch, name: str, fake: Callable[..., Any]) -> None:
    """`models.war`의 함수와 war_eval이 직접 import한 같은 이름을 함께 가짜로 바꾼다."""
    monkeypatch.setattr(war_model, name, fake)
    monkeypatch.setattr(war_eval, name, fake, raising=False)


@dataclass(frozen=True)
class ProjectionCall:
    role: Role
    target: int
    params: Params
    war_params: WarParams


class WarFakes:
    """models.war 가짜 묶음 — 정해 둔 예측 표를 돌려주고 부른 인자를 남긴다."""

    def __init__(self, log: list[str] | None = None) -> None:
        self.log = log
        self.projection_calls: list[ProjectionCall] = []

    def war_projections(
        self,
        tables: Tables,
        war_seasons: pd.DataFrame,
        params: Params,
        war_params: WarParams,
        role: Role,
        target: int,
        playing_time: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        self.projection_calls.append(ProjectionCall(role, int(target), params, war_params))
        if self.log is not None:
            self.log.append("project")
        return _proj(role, int(target))

    @staticmethod
    def league_war_rate(
        war_seasons: pd.DataFrame, seasons: pd.DataFrame, role: Role, season: int
    ) -> float:
        return _lg(war_seasons, seasons, role, season)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fakes: dict[str, Callable[..., Any]] = {
            "war_projections": self.war_projections,
            "league_war_rate": self.league_war_rate,
            "war_params_to_json": _fake_war_json,
            "war_params_from_json": _fake_war_from_json,
            "war_params_sha256": _fake_war_sha,
            "save_war_params": _fake_save_war,
            "load_war_params": _fake_load_war,
        }
        for name, fake in fakes.items():
            _patch_war(monkeypatch, name, fake)


# ---------------------------------------------------------------------------
# actual_war
# ---------------------------------------------------------------------------


def test_war_eval_actual_war_batter_rows_of_season() -> None:
    orphan = _frame(c.WAR_SEASONS, [_war(50, 2025, "bat", 0.3)])  # 시즌 표에 행이 없다
    war_seasons = pd.concat([_war_seasons(), orphan], ignore_index=True)
    out = war_eval.actual_war(war_seasons, _world().seasons["bat"], "bat", 2025)
    assert list(out.columns) == ["player_id", "war", "units"]
    # 시즌 표에 2025 행이 있는 선수만, player_id 순. 1의 투수 행(0.7)은 섞이지 않는다
    assert out["player_id"].tolist() == [1, 2, 4, 5, 6, 99]
    assert out["war"].tolist() == pytest.approx([4.0, 2.0, 1.99, -0.5, 2.8, 5.0])
    assert out["units"].tolist() == pytest.approx([620, 500, 499, 650, 560, 600])


def test_war_eval_actual_war_pitcher_units_are_outs() -> None:
    out = war_eval.actual_war(_war_seasons(), _world().seasons["pit"], "pit", 2025)
    assert list(out.columns) == ["player_id", "war", "units"]
    assert out["player_id"].tolist() == [1, 11, 12, 13, 14]
    assert out["war"].tolist() == pytest.approx([0.7, 1.9, 0.5, 1.5, 1.8])
    assert out["units"].tolist() == pytest.approx([60, 540, 180, 500, 450])


def test_war_eval_actual_war_sorted_by_player_id() -> None:
    seasons = _world().seasons["bat"].iloc[::-1].reset_index(drop=True)
    war_seasons = _war_seasons().iloc[::-1].reset_index(drop=True)
    out = war_eval.actual_war(war_seasons, seasons, "bat", 2024)
    assert out["player_id"].tolist() == [1, 2, 3, 5, 98]
    assert out["war"].tolist() == pytest.approx([3.5, 0.5, 1.2, 0.3, 0.5])
    assert out["units"].tolist() == pytest.approx([600, 200, 450, 100, 650])


@pytest.mark.parametrize(("role", "units"), [("bat", 200), ("pit", 90)])
def test_war_eval_actual_war_short_season_not_scaled(role: Role, units: int) -> None:
    # 2020 출전량 환산(pa_adj·outs_adj)은 쓰지 않는다 — 실제 PA·아웃 그대로
    if role == "bat":
        seasons = _frame(c.BATTING_SEASONS, [_bat(7, 2020, 200)])
        assert float(seasons["pa_adj"].iloc[0]) != 200
    else:
        seasons = _frame(c.PITCHING_SEASONS, [_pit(8, 2020, 90)])
        assert float(seasons["outs_adj"].iloc[0]) != 90
    pid = 7 if role == "bat" else 8
    war_seasons = _frame(c.WAR_SEASONS, [_war(pid, 2020, role, 1.25)])
    out = war_eval.actual_war(war_seasons, seasons, role, 2020)
    assert out["player_id"].tolist() == [pid]
    assert out["war"].tolist() == pytest.approx([1.25])
    assert out["units"].tolist() == pytest.approx([units])


def _without_war(pid: int, season: int, role: Role) -> pd.DataFrame:
    """합성 WAR 표에서 한 역할 행을 뺀다 (정제 표 계약 위반 상황)."""
    ws = _war_seasons()
    drop = (ws["player_id"] == pid) & (ws["season"] == season) & (ws["role"] == role)
    return ws[~drop].reset_index(drop=True)


def test_war_eval_actual_war_missing_war_row_is_error() -> None:
    # 4는 2025 타격 행이 있는데 WAR 행이 없다 — 미출전(0)으로 넘기지 않고 멈춘다
    with pytest.raises(ValueError, match=r"\[4\]"):
        war_eval.actual_war(_without_war(4, 2025, "bat"), _world().seasons["bat"], "bat", 2025)


@pytest.mark.parametrize("season", [2025, 2024], ids=["target", "prev"])
def test_war_eval_evaluate_war_missing_war_row_is_error(season: int) -> None:
    # 목표 시즌·작년 어느 쪽이든 시즌 행에 WAR 행이 없으면 0으로 치지 않는다 (1은 두 시즌 다 출전)
    with pytest.raises(ValueError, match=r"\[1\]"):
        war_eval.evaluate_war(
            _proj("bat", 2025),
            _without_war(1, season, "bat"),
            _world().seasons["bat"],
            "bat",
            2025,
            LG_BAT_2024,
        )


# ---------------------------------------------------------------------------
# evaluate_war — 손 계산
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_war_eval_evaluate_war_hand_computed(role: Role) -> None:
    lg = LG_BAT_2024 if role == "bat" else LG_PIT_2024
    out = war_eval.evaluate_war(
        _proj(role, 2025), _war_seasons(), _world().seasons[role], role, 2025, lg
    )
    _assert_evaluations(out, _expected_2025(role))


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_war_eval_evaluate_war_row_order_is_keys(role: Role) -> None:
    lg = LG_BAT_2024 if role == "bat" else LG_PIT_2024
    shuffled = _proj(role, 2025).iloc[::-1].reset_index(drop=True)  # 입력 순서와 무관
    out = war_eval.evaluate_war(shuffled, _war_seasons(), _world().seasons[role], role, 2025, lg)
    # 그룹은 "war" 넷 + "war_full"의 regulars 하나, 키 순
    assert list(zip(out["metric"], out["group"], strict=True)) == [
        ("war", "all"),
        ("war", "seasons_1_2"),
        ("war", "seasons_3_5"),
        ("war", "war2plus"),
        ("war_full", "regulars"),
    ]
    assert out["target_season"].tolist() == [2025] * 5
    assert out["role"].tolist() == [role] * 5
    keyed = out.sort_values(_KEYS, kind="mergesort").reset_index(drop=True)
    assert_frame_equal(out.reset_index(drop=True), keyed)
    _assert_evaluations(out, _expected_2025(role))


def test_war_eval_evaluate_war_empty_group_is_missing() -> None:
    out = war_eval.evaluate_war(
        _proj("pit", 2025), _war_seasons(), _world().seasons["pit"], "pit", 2025, LG_PIT_2024
    )
    row = out[(out["metric"] == "war") & (out["group"] == "war2plus")]
    assert len(row) == 1  # n = 0이어도 행은 있다
    assert int(row["n"].iloc[0]) == 0
    for col in ("mae", "baseline_mae_prev", "baseline_mae_league"):
        assert pd.isna(row[col].iloc[0])


def test_war_eval_evaluate_war_absent_actual_counts_zero() -> None:
    # 3은 2025 타격 행이 없다 → 실제 0, 대상에는 남는다
    only3 = _proj("bat", 2025, [(3, 2, "bat", 400.0, 1.5, 600.0, 2.25)])
    out = war_eval.evaluate_war(
        only3, _war_seasons(), _world().seasons["bat"], "bat", 2025, LG_BAT_2024
    )
    all_row = out[(out["metric"] == "war") & (out["group"] == "all")].iloc[0]
    assert int(all_row["n"]) == 1
    assert float(all_row["mae"]) == pytest.approx(1.5)  # |1.5 − 0|
    assert float(all_row["baseline_mae_prev"]) == pytest.approx(1.2)  # |1.2 − 0|
    assert float(all_row["baseline_mae_league"]) == pytest.approx(1.2)  # |0.003·400 − 0|
    regulars = out[(out["metric"] == "war_full") & (out["group"] == "regulars")].iloc[0]
    assert int(regulars["n"]) == 0
    assert pd.isna(regulars["mae"])


def test_war_eval_evaluate_war_regulars_use_full_units_per_player() -> None:
    # 14는 불펜(195아웃) — 실제 비율 1.8/450을 195아웃으로 환산한다 (540이 아니다)
    only14 = _proj("pit", 2025, [(14, 4, "rp", 200.0, 1.0, 195.0, 0.975)])
    out = war_eval.evaluate_war(
        only14, _war_seasons(), _world().seasons["pit"], "pit", 2025, LG_PIT_2024
    )
    regulars = out[(out["metric"] == "war_full") & (out["group"] == "regulars")].iloc[0]
    actual = 1.8 / 450 * 195
    assert int(regulars["n"]) == 1
    assert float(regulars["mae"]) == pytest.approx(abs(0.975 - actual))
    assert float(regulars["baseline_mae_prev"]) == pytest.approx(abs(0.6 / 200 * 195 - actual))
    assert float(regulars["baseline_mae_league"]) == pytest.approx(abs(0.002 * 195 - actual))


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_war_eval_evaluate_war_ignores_later_seasons(role: Role) -> None:
    # N2 — 목표 2025 검증은 2026 행(시즌 표·WAR 표)을 보지 않는다. 극단값을 넣어도 손 계산 그대로
    world = _world()
    if role == "bat":
        later = _frame(c.BATTING_SEASONS, [_bat(pid, 2026, 700) for pid in (1, 2, 3, 4, 5, 6)])
    else:
        later = _frame(c.PITCHING_SEASONS, [_pit(pid, 2026, 600) for pid in (11, 12, 13, 14, 15)])
    pids = later["player_id"].tolist()
    later_war = _frame(c.WAR_SEASONS, [_war(pid, 2026, role, 9.0 - pid) for pid in pids])
    seasons = pd.concat([world.seasons[role], later], ignore_index=True)
    war_seasons = pd.concat([_war_seasons(), later_war], ignore_index=True)
    lg = LG_BAT_2024 if role == "bat" else LG_PIT_2024
    out = war_eval.evaluate_war(_proj(role, 2025), war_seasons, seasons, role, 2025, lg)
    _assert_evaluations(out, _expected_2025(role))


@pytest.mark.parametrize("case", ["nan-war", "nan-pt", "nan-full", "nan-league", "inf-league"])
def test_war_eval_evaluate_war_rejects_missing_inputs(case: str) -> None:
    # 결측 예측·리그 비율이 MAE를 조용히 NaN으로 만들지 않게 멈춘다
    proj = _proj("bat", 2025)
    lg = LG_BAT_2024
    if case == "nan-war":
        proj.loc[0, "war"] = float("nan")
    elif case == "nan-pt":
        proj.loc[1, "pt"] = float("nan")
    elif case == "nan-full":
        proj.loc[2, "war_full"] = float("nan")
    elif case == "nan-league":
        lg = float("nan")
    else:
        lg = float("inf")
    with pytest.raises(ValueError):
        war_eval.evaluate_war(proj, _war_seasons(), _world().seasons["bat"], "bat", 2025, lg)


@pytest.mark.parametrize("case", ["target", "role", "mixed-roles"])
def test_war_eval_evaluate_war_rejects_mismatched_projection(case: str) -> None:
    if case == "target":
        proj = _proj("bat", 2024)
    elif case == "role":
        proj = _proj("pit", 2025)
    else:
        proj = pd.concat([_proj("bat", 2025), _proj("pit", 2025)], ignore_index=True)
    with pytest.raises(ValueError):
        war_eval.evaluate_war(
            proj, _war_seasons(), _world().seasons["bat"], "bat", 2025, LG_BAT_2024
        )


# ---------------------------------------------------------------------------
# backtest_war
# ---------------------------------------------------------------------------


def _expected_backtest(
    tables: Tables, war_seasons: pd.DataFrame, targets: Sequence[int]
) -> pd.DataFrame:
    """targets × 두 역할의 evaluate_war (리그 비율 = Y−1), 키 순."""
    frames = [
        war_eval.evaluate_war(
            _proj(role, target),
            war_seasons,
            tables.seasons[role],
            role,
            target,
            _lg(war_seasons, tables.seasons[role], role, target - 1),
        )
        for target in targets
        for role in c.ROLES
    ]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(_KEYS, kind="mergesort").reset_index(drop=True)


def test_war_eval_backtest_war_two_roles_by_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = WarFakes()
    fakes.install(monkeypatch)
    tables, war_seasons = _world(), _war_seasons()

    out = war_eval.backtest_war(tables, war_seasons, PARAMS, WAR_PARAMS, targets=(2024, 2025))

    # 목표마다 두 역할 예측을 한 번씩, 받은 파라미터 그대로
    assert sorted((call.role, call.target) for call in fakes.projection_calls) == [
        ("bat", 2024),
        ("bat", 2025),
        ("pit", 2024),
        ("pit", 2025),
    ]
    for call in fakes.projection_calls:
        assert call.params == PARAMS
        assert call.war_params == WAR_PARAMS
    assert list(out.columns) == c.WAR_EVALUATIONS.column_names
    assert c.check_frame(out, c.WAR_EVALUATIONS) == []
    assert len(out) == 2 * 2 * 5
    keyed = out.sort_values(_KEYS, kind="mergesort").reset_index(drop=True)
    assert_frame_equal(out.reset_index(drop=True), keyed)
    # 2025는 손 계산 그대로 — 리그 기준선이 Y−1(2024) 비율이다
    for role in c.ROLES:
        mine = out[(out["target_season"] == 2025) & (out["role"] == role)]
        _assert_evaluations(mine, _expected_2025(role))
    _assert_evaluations(out, _expected_backtest(tables, war_seasons, (2024, 2025)))


def test_war_eval_backtest_war_default_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = WarFakes()
    fakes.install(monkeypatch)
    tables, war_seasons = _world(), _war_seasons()
    out = war_eval.backtest_war(tables, war_seasons, PARAMS, WAR_PARAMS)
    assert c.TUNE_TARGETS == (2024, 2025)
    assert sorted({call.target for call in fakes.projection_calls}) == [2024, 2025]
    assert sorted(set(out["target_season"].tolist())) == [2024, 2025]
    _assert_evaluations(out, _expected_backtest(tables, war_seasons, (2024, 2025)))


def test_war_eval_backtest_war_accepts_iterable_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    WarFakes().install(monkeypatch)
    targets: Iterator[int] = iter([2025])  # 한 번만 읽히는 반복자
    out = war_eval.backtest_war(_world(), _war_seasons(), PARAMS, WAR_PARAMS, targets=targets)
    expected = pd.concat([_expected_2025("bat"), _expected_2025("pit")], ignore_index=True)
    _assert_evaluations(out, expected)


class _Poison:
    """읽으면 실패하는 자리 — 2026 거부가 데이터를 읽기 전인지 본다."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"2026 거부 전에 데이터를 읽었다 ({name})")

    def __getitem__(self, key: Any) -> Any:
        raise AssertionError(f"2026 거부 전에 데이터를 읽었다 ([{key!r}])")

    def __iter__(self) -> Iterator[Any]:
        raise AssertionError("2026 거부 전에 데이터를 읽었다 (iter)")

    def __len__(self) -> int:
        raise AssertionError("2026 거부 전에 데이터를 읽었다 (len)")


class _PoisonDict(dict[Role, pd.DataFrame]):
    def __getitem__(self, key: Role) -> pd.DataFrame:
        raise AssertionError(f"2026 거부 전에 데이터를 읽었다 (seasons[{key!r}])")


@pytest.mark.parametrize(
    "targets",
    [(2026,), (2024, 2026), (2025, 2027), (2030,)],
    ids=["2026", "2024-2026", "2025-2027", "2030"],
)
def test_war_eval_backtest_war_refuses_final_target_before_reading(
    monkeypatch: pytest.MonkeyPatch, targets: tuple[int, ...]
) -> None:
    fakes = WarFakes()
    fakes.install(monkeypatch)
    poisoned = Tables(
        seasons=_PoisonDict(),
        cohorts=cast(pd.DataFrame, _Poison()),
        league=_PoisonDict(),
        debuts={},
    )
    with pytest.raises(ValueError, match="final-test-war"):
        war_eval.backtest_war(
            poisoned, cast(pd.DataFrame, _Poison()), PARAMS, WAR_PARAMS, targets=targets
        )
    assert fakes.projection_calls == []  # 앞의 목표(2024 등)도 돌리지 않는다


def _poisoned_tables() -> Tables:
    return Tables(
        seasons=_PoisonDict(),
        cohorts=cast(pd.DataFrame, _Poison()),
        league=_PoisonDict(),
        debuts={},
    )


def test_war_eval_backtest_war_refuses_repeated_targets_before_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 같은 목표를 두 번 주면 키가 겹친 표가 된다 — 읽기 전에 멈춘다
    fakes = WarFakes()
    fakes.install(monkeypatch)
    with pytest.raises(ValueError, match="두 번"):
        war_eval.backtest_war(
            _poisoned_tables(),
            cast(pd.DataFrame, _Poison()),
            PARAMS,
            WAR_PARAMS,
            targets=(2024, 2025, 2024),
        )
    assert fakes.projection_calls == []


def test_war_eval_backtest_war_rejects_overlapping_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    # 평가가 두 역할 모두 "bat" 행을 돌려주면 키가 겹친다 — 합친 표의 계약 검사가 잡는다
    WarFakes().install(monkeypatch)

    def always_bat(
        proj: pd.DataFrame,
        war_seasons: pd.DataFrame,
        seasons: pd.DataFrame,
        role: Role,
        target: int,
        league_rate: float,
    ) -> pd.DataFrame:
        return _frame(c.WAR_EVALUATIONS, _final_rows("bat", int(target)))

    monkeypatch.setattr(war_eval, "evaluate_war", always_bat)
    with pytest.raises(ValueError, match="중복"):
        war_eval.backtest_war(_world(), _war_seasons(), PARAMS, WAR_PARAMS, targets=(2025,))


# ---------------------------------------------------------------------------
# run_final_test_war — evaluate_war·models.war는 가짜
# ---------------------------------------------------------------------------

PARAMS_REL = c.PARAMS_PATH.as_posix()
WAR_REL = c.WAR_PARAMS_PATH.as_posix()
CLEAN = {PARAMS_REL: "clean", WAR_REL: "clean"}
_PORCELAIN = {"modified": " M", "staged": "M ", "untracked": "??"}


class FakeGit:
    """`final.GitRunner` 가짜 — `final.committed_head`가 부르는 세 명령만 흉내 낸다.

    상태: "clean" · "modified"(작업 트리만) · "staged"(스테이징이 HEAD와 다름) · "untracked".
    상태표에 없는 경로는 "untracked"다. 모르는 명령은 AssertionError.
    """

    def __init__(
        self,
        states: Mapping[str, str],
        *,
        head: str = HEAD,
        fail: BaseException | None = None,
        log: list[str] | None = None,
    ) -> None:
        self.states = dict(states)
        self.head = head
        self.fail = fail
        self.log = log
        self.calls: list[list[str]] = []

    def _state(self, rel: str) -> str:
        return self.states.get(rel, "untracked")

    def __call__(self, args: Sequence[str]) -> str:
        argv = [str(a) for a in args]
        self.calls.append(argv)
        if self.log is not None:
            self.log.append("git")
        if self.fail is not None:
            raise self.fail
        paths = argv[argv.index("--") + 1 :] if "--" in argv else []
        if argv[:1] == ["rev-parse"]:
            return f"{self.head}\n"
        if argv[:1] == ["ls-files"]:
            return "".join(f"{p}\n" for p in paths if self._state(p) != "untracked")
        if argv[:1] == ["status"] and "--porcelain" in argv:
            return "".join(
                f"{_PORCELAIN[self._state(p)]} {p}\n" for p in paths if self._state(p) in _PORCELAIN
            )
        raise AssertionError(f"FakeGit가 모르는 git 명령: {argv}")


def _final_rows(role: str, target: int, bump: float = 0.0) -> list[dict[str, Any]]:
    """가짜 evaluate_war가 돌려줄 한 역할의 행 (키 순). war2plus는 n = 0 → 결측."""
    return [
        _ev(target, role, "war", "all", 4, 0.5 + bump, 0.625, 0.75),
        _ev(target, role, "war", "seasons_1_2", 1, 0.25 + bump, 0.5, 0.75),
        _ev(target, role, "war", "seasons_3_5", 3, 0.625 + bump, 0.625, 0.5),
        _ev(target, role, "war", "war2plus", 0, None, None, None),
        _ev(target, role, "war_full", "regulars", 2, 1.125 + bump, 1.25, 1.5),
    ]


def _record_rows(bump: float = 0.0) -> list[dict[str, Any]]:
    return _final_rows("bat", 2026, bump) + _final_rows("pit", 2026, bump)


@dataclass(frozen=True)
class EvaluateCall:
    role: Role
    target: int
    league_rate: float
    proj_target: int


class FakeEvaluateWar:
    """evaluate_war 가짜 — 정해 둔 행을 돌려주고 부른 인자를 남긴다 (2026 계산을 하지 않는다)."""

    def __init__(self, bump: float = 0.0, log: list[str] | None = None) -> None:
        self.bump = bump
        self.log = log
        self.calls: list[EvaluateCall] = []

    def __call__(
        self,
        proj: pd.DataFrame,
        war_seasons: pd.DataFrame,
        seasons: pd.DataFrame,
        role: Role,
        target: int,
        league_rate: float,
    ) -> pd.DataFrame:
        targets = sorted({int(t) for t in proj["target_season"]})
        self.calls.append(EvaluateCall(role, int(target), float(league_rate), targets[0]))
        if self.log is not None:
            self.log.append("evaluate")
        return _frame(c.WAR_EVALUATIONS, _final_rows(role, int(target), self.bump))


@dataclass(frozen=True)
class Repo:
    root: Path
    params_path: Path
    war_params_path: Path
    out_path: Path


def _repo(tmp_path: Path, params: Params = PARAMS, war_params: WarParams = WAR_PARAMS) -> Repo:
    """저장소 자리와 커밋된(것으로 칠) 파라미터 두 파일. 기록은 `FINAL_TEST_WAR_PATH` 자리."""
    root = tmp_path / "repo"
    params_path = root / c.PARAMS_PATH
    war_params_path = root / c.WAR_PARAMS_PATH
    save_params(params, params_path)
    _fake_save_war(war_params, war_params_path)
    return Repo(root, params_path, war_params_path, root / c.FINAL_TEST_WAR_PATH)


def _run(
    repo: Repo,
    now: str,
    git: FakeGit,
    reason: str | None = None,
    *,
    params: Params = PARAMS,
    war_params: WarParams = WAR_PARAMS,
    tables: Tables | None = None,
    war_seasons: pd.DataFrame | None = None,
) -> dict[str, Any]:
    return war_eval.run_final_test_war(
        _world() if tables is None else tables,
        _war_seasons() if war_seasons is None else war_seasons,
        params,
        war_params,
        repo=repo.root,
        params_path=repo.params_path,
        war_params_path=repo.war_params_path,
        out_path=repo.out_path,
        now=now,
        rerun_reason=reason,
        git=git,
    )


def _no_nan(text: str) -> Any:
    """NaN·Infinity가 있으면 실패하는 JSON 읽기."""

    def reject(token: str) -> Any:
        raise AssertionError(f"JSON에 {token}이 들어 있다")

    return json.loads(text, parse_constant=reject)


def test_war_eval_run_final_test_war_writes_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    # 기존 여섯 지표의 2026 기록 자리 — 읽지도 쓰지도 않는다 (있어도 첫 실행이다)
    monkeypatch.chdir(repo.root)
    six_metrics = repo.root / c.FINAL_TEST_PATH
    six_metrics.parent.mkdir(parents=True, exist_ok=True)
    six_metrics.write_text('{"sentinel": "six metrics"}\n', encoding="utf-8")
    log: list[str] = []
    fakes = WarFakes(log)
    fakes.install(monkeypatch)
    evaluate = FakeEvaluateWar(log=log)
    monkeypatch.setattr(war_eval, "evaluate_war", evaluate)
    tables, war_seasons = _world(), _war_seasons()

    record = _run(repo, NOW1, FakeGit(CLEAN, log=log), tables=tables, war_seasons=war_seasons)

    expected = {
        "target_season": 2026,
        "ran_at": NOW1,
        "params_sha256": params_sha256(PARAMS),
        "war_params_sha256": _fake_war_sha(WAR_PARAMS),
        "commit": HEAD,
        "reruns": [],
        "rows": _record_rows(),  # 결측은 None (JSON null)
    }
    assert record == expected
    assert list(record) == RECORD_KEYS
    text = repo.out_path.read_text(encoding="utf-8")
    assert _no_nan(text) == expected
    assert text.startswith('{\n "')  # 사람이 읽는 indent=1
    for row in _no_nan(text)["rows"]:
        assert list(row) == c.WAR_EVALUATIONS.column_names
        assert type(row["n"]) is int and type(row["target_season"]) is int

    # 커밋 확인이 예측·평가보다 먼저
    assert "git" in log and "evaluate" in log
    last_git = max(i for i, name in enumerate(log) if name == "git")
    assert all(i > last_git for i, name in enumerate(log) if name in ("project", "evaluate"))
    # 두 역할을 목표 2026으로, 리그 비율은 Y−1(2025)
    assert sorted((call.role, call.target) for call in fakes.projection_calls) == [
        ("bat", 2026),
        ("pit", 2026),
    ]
    assert sorted(call.role for call in evaluate.calls) == ["bat", "pit"]
    for call in evaluate.calls:
        assert call.target == c.FINAL_TEST_TARGET == 2026
        assert call.proj_target == 2026
        lg = _lg(war_seasons, tables.seasons[call.role], call.role, 2025)
        assert call.league_rate == pytest.approx(lg)
    # 기존 기록은 그대로
    assert six_metrics.read_text(encoding="utf-8") == '{"sentinel": "six metrics"}\n'


@pytest.mark.parametrize(
    ("states", "fail"),
    [
        ({**CLEAN, PARAMS_REL: "untracked"}, None),
        ({**CLEAN, WAR_REL: "untracked"}, None),
        ({**CLEAN, WAR_REL: "modified"}, None),
        ({**CLEAN, WAR_REL: "staged"}, None),
        ({**CLEAN, PARAMS_REL: "modified"}, None),
        (CLEAN, subprocess.CalledProcessError(128, ["git"], stderr="fatal: not a git repository")),
    ],
    ids=[
        "params-untracked",
        "war-untracked",
        "war-modified",
        "war-staged",
        "params-modified",
        "git-error",
    ],
)
def test_war_eval_run_final_test_war_requires_committed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    states: dict[str, str],
    fail: BaseException | None,
) -> None:
    repo = _repo(tmp_path)
    fakes = WarFakes()
    fakes.install(monkeypatch)
    evaluate = FakeEvaluateWar()
    monkeypatch.setattr(war_eval, "evaluate_war", evaluate)
    with pytest.raises(WarTestError):
        _run(repo, NOW1, FakeGit(states, fail=fail))
    assert fakes.projection_calls == [] and evaluate.calls == []
    assert not repo.out_path.exists()


@pytest.mark.parametrize("which", ["params", "war_params"])
def test_war_eval_run_final_test_war_rejects_values_unlike_committed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    repo = _repo(tmp_path)  # 파일은 PARAMS·WAR_PARAMS
    fakes = WarFakes()
    fakes.install(monkeypatch)
    evaluate = FakeEvaluateWar()
    monkeypatch.setattr(war_eval, "evaluate_war", evaluate)
    params = PARAMS2 if which == "params" else PARAMS
    war_params = WAR_PARAMS2 if which == "war_params" else WAR_PARAMS
    with pytest.raises(WarTestError):
        _run(repo, NOW1, FakeGit(CLEAN), params=params, war_params=war_params)
    assert fakes.projection_calls == [] and evaluate.calls == []
    assert not repo.out_path.exists()


@pytest.mark.parametrize(
    "reason", [None, "", "  \t "], ids=["no-reason", "empty-reason", "blank-reason"]
)
def test_war_eval_run_final_test_war_refuses_second_run_without_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str | None
) -> None:
    repo = _repo(tmp_path)
    WarFakes().install(monkeypatch)
    monkeypatch.setattr(war_eval, "evaluate_war", FakeEvaluateWar())
    _run(repo, NOW1, FakeGit(CLEAN))
    before = repo.out_path.read_bytes()

    evaluate = FakeEvaluateWar(bump=0.25)
    monkeypatch.setattr(war_eval, "evaluate_war", evaluate)
    with pytest.raises(WarTestError, match="이미"):
        _run(repo, NOW2, FakeGit(CLEAN), reason)
    assert repo.out_path.read_bytes() == before  # 기록은 바이트까지 그대로
    assert evaluate.calls == []


def test_war_eval_run_final_test_war_rerun_keeps_first_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    WarFakes().install(monkeypatch)
    monkeypatch.setattr(war_eval, "evaluate_war", FakeEvaluateWar())
    _run(repo, NOW1, FakeGit(CLEAN))

    reason = "원본 재수집 뒤 다시 확인"
    monkeypatch.setattr(war_eval, "evaluate_war", FakeEvaluateWar(bump=0.25))
    record = _run(repo, NOW2, FakeGit(CLEAN, head=HEAD2), reason)

    assert list(record) == RECORD_KEYS
    assert record["target_season"] == 2026
    assert record["ran_at"] == NOW1  # 첫 실행 시각
    assert record["commit"] == HEAD  # 첫 실행 커밋
    assert record["reruns"] == [{"ran_at": NOW2, "reason": reason, "commit": HEAD2}]
    assert list(record["reruns"][0]) == ["ran_at", "reason", "commit"]
    assert record["rows"] == _record_rows(0.25)  # 새 결과로 바꾼다
    assert record["params_sha256"] == params_sha256(PARAMS)
    assert record["war_params_sha256"] == _fake_war_sha(WAR_PARAMS)
    assert _no_nan(repo.out_path.read_text(encoding="utf-8")) == record

    # 한 번 더 — reruns 끝에 붙는다
    monkeypatch.setattr(war_eval, "evaluate_war", FakeEvaluateWar())
    again = _run(repo, NOW3, FakeGit(CLEAN), "둘째 사유")
    assert again["ran_at"] == NOW1 and again["commit"] == HEAD
    assert again["reruns"] == [
        {"ran_at": NOW2, "reason": reason, "commit": HEAD2},
        {"ran_at": NOW3, "reason": "둘째 사유", "commit": HEAD},
    ]
    assert again["rows"] == _record_rows()
    assert _no_nan(repo.out_path.read_text(encoding="utf-8")) == again


def test_war_eval_run_final_test_war_reason_without_record_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    fakes = WarFakes()
    fakes.install(monkeypatch)
    evaluate = FakeEvaluateWar()
    monkeypatch.setattr(war_eval, "evaluate_war", evaluate)
    with pytest.raises(WarTestError):
        _run(repo, NOW1, FakeGit(CLEAN), "기록이 없는데 사유")
    assert not repo.out_path.exists()
    assert fakes.projection_calls == [] and evaluate.calls == []


@pytest.mark.parametrize("changed", ["params", "war_params"])
def test_war_eval_run_final_test_war_rerun_refuses_changed_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    repo = _repo(tmp_path)
    WarFakes().install(monkeypatch)
    monkeypatch.setattr(war_eval, "evaluate_war", FakeEvaluateWar())
    _run(repo, NOW1, FakeGit(CLEAN))
    before = repo.out_path.read_bytes()

    # 2026을 본 뒤 파라미터를 바꿔 커밋했다 — 파일·값은 서로 맞지만 첫 기록과 다르다
    params, war_params = PARAMS, WAR_PARAMS
    if changed == "params":
        params = PARAMS2
        save_params(params, repo.params_path)
    else:
        war_params = WAR_PARAMS2
        _fake_save_war(war_params, repo.war_params_path)
    evaluate = FakeEvaluateWar(bump=0.25)
    monkeypatch.setattr(war_eval, "evaluate_war", evaluate)
    with pytest.raises(WarTestError):
        _run(
            repo,
            NOW2,
            FakeGit(CLEAN, head=HEAD2),
            "모델을 바꿔 다시",
            params=params,
            war_params=war_params,
        )
    assert repo.out_path.read_bytes() == before
    assert evaluate.calls == []  # 거부는 2026 평가 전이다


def _bad_records() -> dict[str, str]:
    """형식이 틀린 기존 기록 (--rerun이 읽다가 멈춰야 한다)."""
    good: dict[str, Any] = {
        "target_season": 2026,
        "ran_at": NOW1,
        "params_sha256": params_sha256(PARAMS),
        "war_params_sha256": _fake_war_sha(WAR_PARAMS),
        "commit": HEAD,
        "reruns": [],
        "rows": _record_rows(),
    }
    not_2026 = {**good, "rows": [{**good["rows"][0], "target_season": 2025}]}
    six_metrics = {k: v for k, v in good.items() if k != "war_params_sha256"}
    six_metrics["adoption_sha256"] = "b" * 64  # 여섯 지표 기록 모양
    nan_mae = json.dumps(good).replace('"mae": 0.5,', '"mae": NaN,', 1)
    assert "NaN" in nan_mae
    return {
        "not-json": "{not json",
        "missing-keys": json.dumps({"target_season": 2026}),
        "rows-not-2026": json.dumps(not_2026),
        "six-metric-shape": json.dumps(six_metrics),
        "nan-mae": nan_mae,
    }


@pytest.mark.parametrize("case", sorted(_bad_records()))
def test_war_eval_run_final_test_war_rerun_refuses_bad_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    repo = _repo(tmp_path)
    fakes = WarFakes()
    fakes.install(monkeypatch)
    evaluate = FakeEvaluateWar()
    monkeypatch.setattr(war_eval, "evaluate_war", evaluate)
    content = _bad_records()[case]
    repo.out_path.parent.mkdir(parents=True, exist_ok=True)
    repo.out_path.write_text(content, encoding="utf-8")
    with pytest.raises(WarTestError):
        _run(repo, NOW2, FakeGit(CLEAN), "다시 확인")
    assert repo.out_path.read_text(encoding="utf-8") == content  # 고치지 않는다
    assert fakes.projection_calls == [] and evaluate.calls == []


@pytest.mark.parametrize("reason", [None, "다시 확인"], ids=["first-run", "rerun"])
@pytest.mark.parametrize("which", ["record", "first-run-archive"])
def test_war_eval_run_final_test_war_refuses_six_metric_record_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str, reason: str | None
) -> None:
    # 기존 여섯 지표 기록(과 첫 실행 보관본) 자리는 읽지도 쓰지도 않는다 — git 확인 전에 멈춘다
    repo = _repo(tmp_path)
    fakes = WarFakes()
    fakes.install(monkeypatch)
    evaluate = FakeEvaluateWar()
    monkeypatch.setattr(war_eval, "evaluate_war", evaluate)
    six = repo.root / c.FINAL_TEST_PATH
    target = six if which == "record" else six.with_name(f"{six.stem}_run1.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    sentinel = '{"sentinel": "six metrics"}\n'
    target.write_text(sentinel, encoding="utf-8")
    git = FakeGit(CLEAN)
    with pytest.raises(WarTestError, match="여섯 지표"):
        _run(replace(repo, out_path=target), NOW1, git, reason)
    assert target.read_text(encoding="utf-8") == sentinel
    assert git.calls == [] and fakes.projection_calls == [] and evaluate.calls == []


def test_war_eval_run_final_test_war_refuses_rows_not_for_2026(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 다음 --rerun이 읽지 못할 기록(목표 2025 행)은 쓰지 않는다
    repo = _repo(tmp_path)
    WarFakes().install(monkeypatch)

    def wrong_target(
        proj: pd.DataFrame,
        war_seasons: pd.DataFrame,
        seasons: pd.DataFrame,
        role: Role,
        target: int,
        league_rate: float,
    ) -> pd.DataFrame:
        return _frame(c.WAR_EVALUATIONS, _final_rows(role, 2025))

    monkeypatch.setattr(war_eval, "evaluate_war", wrong_target)
    with pytest.raises(ValueError, match="형식"):
        _run(repo, NOW1, FakeGit(CLEAN))
    assert not repo.out_path.exists()


@pytest.mark.parametrize(
    "now",
    [
        "2026-11-25 01:02:03",
        "2026-11-25T01:02:03",
        "2026-11-25T01:02:03+00:00",
        "2026-13-01T00:00:00Z",
        "2026-1-5T1:2:3Z",
        "",
    ],
)
def test_war_eval_run_final_test_war_rejects_bad_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, now: str
) -> None:
    repo = _repo(tmp_path)
    WarFakes().install(monkeypatch)
    monkeypatch.setattr(war_eval, "evaluate_war", FakeEvaluateWar())
    with pytest.raises(ValueError):
        _run(repo, now, FakeGit(CLEAN))
    assert not repo.out_path.exists()


def test_war_eval_run_final_test_war_record_is_byte_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    WarFakes().install(monkeypatch)
    monkeypatch.setattr(war_eval, "evaluate_war", FakeEvaluateWar())
    first = _repo(tmp_path / "a")
    second = _repo(tmp_path / "b")
    _run(first, NOW1, FakeGit(CLEAN))
    _run(second, NOW1, FakeGit(CLEAN))
    assert first.out_path.read_bytes() == second.out_path.read_bytes()


# ---------------------------------------------------------------------------
# build_war_web · write_war_csv
# ---------------------------------------------------------------------------

SUMMARIES: dict[int, dict[str, Any]] = {
    660: {
        "player_id": 660,
        "name": "Two Way",
        "primary_position": "TWP",
        "age": 33,
        "team_name": "Team A",
        "active": True,
        "last_mlb_season": 2026,
    },
    2: {
        "player_id": 2,
        "name": "Pure Pitcher",
        "primary_position": "P",
        "age": 31,
        "team_name": "Team B",
        "active": True,
        "last_mlb_season": 2026,
    },
    4: {
        "player_id": 4,
        "name": "Bench Bat",
        "primary_position": "C",
        "age": 25,
        "team_name": "Team C",
        "active": True,
        "last_mlb_season": 2026,
    },
    3: {
        "player_id": 3,
        "name": "Pure Batter",
        "primary_position": "SS",
        "age": 36,
        "team_name": None,
        "active": False,
        "last_mlb_season": 2025,
    },
}
WAR_SHA = "d" * 64


def _rank(
    pid: int,
    roles: str,
    war_bat: float | None,
    war_pit: float | None,
    full_bat: float | None,
    full_pit: float | None,
    prev_war: float | None,
) -> dict[str, Any]:
    return {
        "player_id": pid,
        "target_season": 2027,
        "roles": roles,
        "war_bat": war_bat,
        "war_pit": war_pit,
        "war": (war_bat or 0.0) + (war_pit or 0.0),
        "war_full_bat": full_bat,
        "war_full_pit": full_pit,
        "war_full": (full_bat or 0.0) + (full_pit or 0.0),
        "prev_war": prev_war,
    }


def _cohort(pid: int, role: Role, n_seasons: int, played_prev: bool) -> dict[str, Any]:
    return {
        "player_id": pid,
        "role": role,
        "target_season": 2027,
        "window_first": 2022,
        "window_last": 2026,
        "n_window": 5,
        "input_first": 2022,
        "input_last": 2026 if played_prev else 2025,
        "n_seasons": n_seasons,
        "played_prev": played_prev,
        "sample": 1000,
        "sample_outs": 300 if role == "pit" else None,
    }


def _player(pid: int, name: str, position: str, active: bool, last: int) -> dict[str, Any]:
    return {
        "player_id": pid,
        "name": name,
        "primary_position": position,
        "active": active,
        "last_mlb_season": last,
    }


def _sample_meta() -> dict[str, Any]:
    meta: dict[str, Any] = json.loads(SAMPLE_WAR.read_text(encoding="utf-8"))["meta"]
    return meta


def _record() -> dict[str, Any]:
    """run_final_test_war 기록 모양 (내보내기 입력)."""
    return {
        "target_season": 2026,
        "ran_at": NOW1,
        "params_sha256": "a" * 64,
        "war_params_sha256": WAR_SHA,
        "commit": HEAD,
        "reruns": [{"ran_at": NOW2, "reason": "원본 재수집", "commit": HEAD2}],
        "rows": [_ev(2026, "bat", "war", "all", 9, 0.48766, 0.55, 0.57)],
    }


def _export_input(final_test: dict[str, Any] | None) -> WarExportInput:
    """war.json 입력 — 선수 660(투타), 2(투수), 4(타자 2시즌), 3(타자, 2026 미출전).

    rankings는 war 내림차순(player_id 순이 아니다)이다. 평가표의 2026 행은 시험 기록이 있을 때만
    기록의 rows 그대로 넣는다(D36 — 2026은 final-test-war 기록으로만 공개한다).
    """
    rankings = _frame(
        c.WAR_RANKINGS,
        [
            _rank(660, "both", 4.1234, 1.8467, 4.9481, 4.3412, 2.9433),
            _rank(2, "pit", None, 0.6149, None, 0.8333, 0.4),
            _rank(4, "bat", 0.2049, None, 1.5371, None, 0.0),
            _rank(3, "bat", -0.3456, None, -1.728, None, None),
        ],
    )
    projections: dict[Role, pd.DataFrame] = {
        "bat": _proj(
            "bat",
            2027,
            [
                (3, 4, "bat", 120.06, -0.3456, 600.0, -1.728),
                (4, 2, "bat", 80.0, 0.2049, 600.0, 1.5371),
                (660, 2, "bat", 499.84, 4.1234, 600.0, 4.9481),
            ],
        ),
        "pit": _proj(
            "pit",
            2027,
            [
                (2, 1, "rp", 143.0, 0.6149, 195.0, 0.8333),
                (660, 3, "sp", 230.0, 1.8467, 540.0, 4.3412),
            ],
        ),
    }
    cohorts: dict[Role, pd.DataFrame] = {
        "bat": _frame(
            c.COHORTS,
            [
                _cohort(3, "bat", 4, False),
                _cohort(4, "bat", 2, True),
                _cohort(660, "bat", 2, False),
            ],
        ),
        "pit": _frame(c.COHORTS, [_cohort(2, "pit", 1, True), _cohort(660, "pit", 3, True)]),
    }
    players = _frame(
        c.PLAYERS,
        [
            _player(2, "Pure Pitcher", "P", True, 2026),
            _player(3, "Pure Batter", "SS", False, 2025),
            _player(4, "Bench Bat", "C", True, 2026),
            _player(660, "Two Way", "TWP", True, 2026),
            _player(999, "Other Player", "1B", True, 2026),
        ],
    )
    war_seasons = _frame(
        c.WAR_SEASONS,
        [
            _war(660, 2021, "bat", 5.02),  # 입력 기간(2022~2026) 밖
            _war(660, 2023, "bat", 3.111),
            _war(660, 2024, "bat", 7.4467),
            _war(660, 2024, "pit", 0.5),
            _war(660, 2025, "pit", 1.9212),
            _war(660, 2026, "pit", 2.9433),
            _war(2, 2026, "pit", 0.4),
            _war(3, 2022, "bat", 1.0),
            _war(3, 2023, "bat", 0.8),
            _war(3, 2024, "bat", 0.25),
            _war(3, 2025, "bat", -0.1234),
            _war(4, 2025, "bat", 0.3),
            _war(4, 2026, "bat", 0.0),
            _war(999, 2026, "bat", 3.0),  # 순위에 없는 선수
        ],
    )
    past = [
        _ev(2024, "bat", "war", "all", 10, 0.123456, 0.591234, 0.575),
        _ev(2024, "bat", "war_full", "regulars", 0, None, None, None),
        _ev(2025, "pit", "war", "war2plus", 3, 1.61234, 1.42494, 1.5),
    ]
    tested = [] if final_test is None else list(final_test["rows"])
    evaluations = _frame(c.WAR_EVALUATIONS, [*past, *tested])
    return WarExportInput(
        meta=_sample_meta(),
        rankings=rankings,
        projections=projections,
        cohorts=cohorts,
        players=players,
        player_summaries={pid: dict(s) for pid, s in SUMMARIES.items()},
        war_seasons=war_seasons,
        evaluations=evaluations,
        final_test=final_test,
        war_params_sha256=WAR_SHA,
    )


EXPECTED_ROWS: list[dict[str, Any]] = [
    {
        "player": SUMMARIES[660],
        "roles": "both",
        "played_prev": True,  # 투수 역할에 2026 기록
        "group": "seasons_3_5",  # 두 역할 n_seasons 중 큰 값(타 2 · 투 3)
        "slot": "sp",
        "pa": 499.8,
        "ip": 76.7,  # 230아웃 / 3
        "war_bat": 4.12,
        "war_pit": 1.85,
        "war": 5.97,
        "war_full_bat": 4.95,
        "war_full_pit": 4.34,
        "war_full": 9.29,
        "prev_war": 2.94,
        "history": [
            {"season": 2023, "war_bat": 3.11, "war_pit": None},
            {"season": 2024, "war_bat": 7.45, "war_pit": 0.5},
            {"season": 2025, "war_bat": None, "war_pit": 1.92},
            {"season": 2026, "war_bat": None, "war_pit": 2.94},
        ],
    },
    {
        "player": SUMMARIES[2],
        "roles": "pit",
        "played_prev": True,
        "group": "seasons_1_2",
        "slot": "rp",
        "pa": None,
        "ip": 47.7,  # 143아웃 / 3
        "war_bat": None,
        "war_pit": 0.61,
        "war": 0.61,
        "war_full_bat": None,
        "war_full_pit": 0.83,
        "war_full": 0.83,
        "prev_war": 0.4,
        "history": [{"season": 2026, "war_bat": None, "war_pit": 0.4}],
    },
    {
        "player": SUMMARIES[4],
        "roles": "bat",
        "played_prev": True,
        "group": "seasons_1_2",  # n_seasons 2 = 경계
        "slot": None,
        "pa": 80.0,
        "ip": None,
        "war_bat": 0.2,
        "war_pit": None,
        "war": 0.2,
        "war_full_bat": 1.54,
        "war_full_pit": None,
        "war_full": 1.54,
        "prev_war": 0.0,
        "history": [
            {"season": 2025, "war_bat": 0.3, "war_pit": None},
            {"season": 2026, "war_bat": 0.0, "war_pit": None},
        ],
    },
    {
        "player": SUMMARIES[3],
        "roles": "bat",
        "played_prev": False,
        "group": "seasons_3_5",
        "slot": None,
        "pa": 120.1,
        "ip": None,
        "war_bat": -0.35,
        "war_pit": None,
        "war": -0.35,
        "war_full_bat": -1.73,
        "war_full_pit": None,
        "war_full": -1.73,
        "prev_war": None,
        "history": [
            {"season": 2022, "war_bat": 1.0, "war_pit": None},
            {"season": 2023, "war_bat": 0.8, "war_pit": None},
            {"season": 2024, "war_bat": 0.25, "war_pit": None},
            {"season": 2025, "war_bat": -0.12, "war_pit": None},
        ],
    },
]
EXPECTED_PAST_EVALUATIONS: list[dict[str, Any]] = [
    _ev(2024, "bat", "war", "all", 10, 0.1235, 0.5912, 0.575),
    _ev(2024, "bat", "war_full", "regulars", 0, None, None, None),
    _ev(2025, "pit", "war", "war2plus", 3, 1.6123, 1.4249, 1.5),
]
EXPECTED_EVALUATIONS: list[dict[str, Any]] = [
    *EXPECTED_PAST_EVALUATIONS,
    _ev(2026, "bat", "war", "all", 9, 0.4877, 0.55, 0.57),  # 시험 기록이 있을 때만
]


@pytest.mark.parametrize("with_record", [False, True], ids=["no-final-test", "final-test"])
def test_war_eval_build_war_web_matches_schema(with_record: bool) -> None:
    out = war_eval.build_war_web(_export_input(_record() if with_record else None))
    schema = load_schema(ROOT / c.WEB_SCHEMA_PATH)
    assert validate(out, "WarFile", schema) == []
    json.dumps(out, allow_nan=False)  # NaN·inf가 없다


def test_war_eval_build_war_web_header() -> None:
    inp = _export_input(None)
    out = war_eval.build_war_web(inp)
    assert out["meta"] == _sample_meta()
    assert out["source"] == c.WAR_SOURCE == "MLB Stats API sabermetrics"
    assert out["war_params_sha256"] == WAR_SHA
    assert out["full_time"] == {"pa": 600, "sp_ip": 180, "rp_ip": 65}
    assert all(type(v) is int for v in out["full_time"].values())
    assert out["final_test"] is None


def test_war_eval_build_war_web_rows_follow_rankings() -> None:
    out = war_eval.build_war_web(_export_input(None))
    assert [row["player"]["player_id"] for row in out["rows"]] == [660, 2, 4, 3]
    assert out["rows"] == EXPECTED_ROWS


def test_war_eval_build_war_web_two_way_row() -> None:
    row = war_eval.build_war_web(_export_input(None))["rows"][0]
    # 오타니형: 두 역할 출전량·자리·구성·합이 모두 있다
    assert row["roles"] == "both"
    assert row["pa"] == 499.8 and row["ip"] == 76.7 and row["slot"] == "sp"
    assert row["war"] == 5.97 and row["war_full"] == 9.29
    assert [h["season"] for h in row["history"]] == [2023, 2024, 2025, 2026]  # 2021은 기간 밖


@pytest.mark.parametrize("with_record", [False, True], ids=["no-final-test", "final-test"])
def test_war_eval_build_war_web_evaluations_rounded(with_record: bool) -> None:
    out = war_eval.build_war_web(_export_input(_record() if with_record else None))
    expected = EXPECTED_EVALUATIONS if with_record else EXPECTED_PAST_EVALUATIONS
    assert out["evaluations"] == expected
    for row in out["evaluations"]:
        assert list(row) == c.WAR_EVALUATIONS.column_names
        assert type(row["n"]) is int and type(row["target_season"]) is int


def test_war_eval_build_war_web_final_test_summary() -> None:
    out = war_eval.build_war_web(_export_input(_record()))
    # commit·params_sha256·rows는 내보내지 않는다
    assert out["final_test"] == {
        "target_season": 2026,
        "ran_at": NOW1,
        "war_params_sha256": WAR_SHA,
        "reruns": [{"ran_at": NOW2, "reason": "원본 재수집"}],
    }


def test_war_eval_build_war_web_is_reproducible() -> None:
    def dump() -> str:
        out = war_eval.build_war_web(_export_input(_record()))
        return json.dumps(out, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    assert dump() == dump()


def test_war_eval_build_war_web_accepts_run_final_test_war_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    WarFakes().install(monkeypatch)
    monkeypatch.setattr(war_eval, "evaluate_war", FakeEvaluateWar())
    _run(repo, NOW1, FakeGit(CLEAN))
    record = _run(repo, NOW2, FakeGit(CLEAN, head=HEAD2), "다시 확인")

    inp = replace(
        _export_input(record),
        meta={
            **_sample_meta(),
            "params_sha256": record["params_sha256"],
        },  # 같은 파라미터로 내보낸다
        evaluations=_frame(c.WAR_EVALUATIONS, _record_rows()),
        war_params_sha256=record["war_params_sha256"],
    )
    out = war_eval.build_war_web(inp)
    assert validate(out, "WarFile", load_schema(ROOT / c.WEB_SCHEMA_PATH)) == []
    assert out["final_test"] == {
        "target_season": 2026,
        "ran_at": NOW1,
        "war_params_sha256": _fake_war_sha(WAR_PARAMS),
        "reruns": [{"ran_at": NOW2, "reason": "다시 확인"}],
    }


def _set_rows(frame: pd.DataFrame, pid: int, **values: Any) -> pd.DataFrame:
    """표 사본에서 선수 pid 행의 값을 바꾼다."""
    out = frame.copy()
    for key, value in values.items():
        out.loc[out["player_id"] == pid, key] = value
    return out


def _with_evaluations(inp: WarExportInput, extra: Sequence[Mapping[str, Any]]) -> WarExportInput:
    rows = cast(list[dict[str, Any]], inp.evaluations.to_dict(orient="records"))
    clean = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in rows]
    return replace(inp, evaluations=_frame(c.WAR_EVALUATIONS, [*clean, *extra]))


def _with_projection(inp: WarExportInput, role: Role, pid: int, **values: Any) -> WarExportInput:
    projections = dict(inp.projections)
    projections[role] = _set_rows(projections[role], pid, **values)
    return replace(inp, projections=projections)


def _bad_export_inputs() -> dict[str, tuple[Callable[[], WarExportInput], str]]:
    """build_war_web이 거부할 입력 → (만들기, 오류 메시지 조각)."""
    no_record, with_record = _export_input(None), _export_input(_record())
    test_row = _ev(2026, "bat", "war", "all", 9, 0.48766, 0.55, 0.57)
    past_only = no_record.evaluations
    summaries = {pid: s for pid, s in SUMMARIES.items() if pid != 4}
    cohorts = dict(no_record.cohorts)
    cohorts["bat"] = cohorts["bat"][cohorts["bat"]["player_id"] != 4].reset_index(drop=True)
    return {
        # D36 — 2026 행은 시험 기록으로만, 기록의 rows 그대로
        "2026-without-record": (lambda: _with_evaluations(no_record, [test_row]), "final_test"),
        "2026-unlike-record": (
            lambda: replace(
                _with_evaluations(no_record, [{**test_row, "mae": 0.5}]), final_test=_record()
            ),
            "rows",
        ),
        "2026-row-not-in-record": (
            lambda: _with_evaluations(
                with_record, [_ev(2026, "pit", "war", "all", 9, 0.3, 0.35, 0.33)]
            ),
            "rows",
        ),
        "record-without-2026-rows": (lambda: replace(with_record, evaluations=past_only), "rows"),
        "record-war-params": (
            lambda: replace(with_record, war_params_sha256="e" * 64),
            "war_params_sha256",
        ),
        "record-params": (
            lambda: replace(with_record, meta={**_sample_meta(), "params_sha256": "b" * 64}),
            "params_sha256",
        ),
        "record-malformed": (
            lambda: replace(with_record, final_test={**_record(), "commit": ""}),
            "형식",
        ),
        "evaluation-after-target": (
            lambda: _with_evaluations(
                no_record, [_ev(2027, "pit", "war", "all", 1, 0.1, 0.1, 0.1)]
            ),
            "이후",
        ),
        "ranking-target": (
            lambda: replace(
                no_record, rankings=_set_rows(no_record.rankings, 4, target_season=2026)
            ),
            "목표 시즌",
        ),
        "roles-unlike-projections": (
            lambda: replace(no_record, rankings=_set_rows(no_record.rankings, 4, roles="both")),
            "roles",
        ),
        "unknown-roles": (
            lambda: replace(no_record, rankings=_set_rows(no_record.rankings, 4, roles="dh")),
            "roles",
        ),
        "unranked-projection": (
            lambda: replace(
                no_record,
                rankings=no_record.rankings[no_record.rankings["player_id"] != 3].reset_index(
                    drop=True
                ),
            ),
            "순위 표에 없다",
        ),
        "missing-summary": (lambda: replace(no_record, player_summaries=summaries), "요약"),
        "missing-cohort": (lambda: replace(no_record, cohorts=cohorts), "대상 표"),
        "pitcher-slot": (lambda: _with_projection(no_record, "pit", 2, slot="bat"), "sp·rp"),
        "war-unlike-projection": (
            lambda: _with_projection(no_record, "bat", 4, war=0.3),
            "예측 표",
        ),
        "full-unlike-projection": (
            lambda: _with_projection(no_record, "pit", 660, war_full=4.0),
            "예측 표",
        ),
        "value-for-absent-role": (
            lambda: replace(
                no_record, rankings=_set_rows(no_record.rankings, 2, war_bat=0.1, war=0.7149)
            ),
            "예측이 없는데",
        ),
        "total-unlike-roles": (
            lambda: replace(no_record, rankings=_set_rows(no_record.rankings, 660, war=6.5)),
            "합",
        ),
        "war-seasons-duplicate": (
            lambda: replace(
                no_record,
                war_seasons=pd.concat(
                    [no_record.war_seasons, no_record.war_seasons.iloc[[1]]], ignore_index=True
                ),
            ),
            "WAR 시즌 표",
        ),
        # 결측 war는 history에 조용히 null로 나가면 안 된다 (계약상 결측 불가)
        "war-seasons-missing-war": (
            lambda: replace(
                no_record, war_seasons=_set_rows(no_record.war_seasons, 3, war=float("nan"))
            ),
            "결측",
        ),
    }


@pytest.mark.parametrize("case", sorted(_bad_export_inputs()))
def test_war_eval_build_war_web_refuses_inconsistent_input(case: str) -> None:
    make, message = _bad_export_inputs()[case]
    with pytest.raises(ValueError, match=message):
        war_eval.build_war_web(make())


def test_war_eval_write_war_csv_rejects_unknown_player(tmp_path: Path) -> None:
    inp = _export_input(None)
    players = inp.players[inp.players["player_id"] != 4].reset_index(drop=True)
    path = tmp_path / "projections_2027_war.csv"
    with pytest.raises(ValueError, match="players"):
        war_eval.write_war_csv(replace(inp, players=players), path)
    assert not path.exists()


def test_war_eval_write_war_csv_columns_and_values(tmp_path: Path) -> None:
    inp = _export_input(None)
    path = tmp_path / "projections_2027_war.csv"
    war_eval.write_war_csv(inp, path)
    df = pd.read_csv(path)
    assert list(df.columns) == [*c.WAR_RANKINGS.column_names, "name"]
    assert sorted(df["player_id"].tolist()) == [2, 3, 4, 660]
    names = dict(zip(df["player_id"].tolist(), df["name"].tolist(), strict=True))
    assert names == {660: "Two Way", 2: "Pure Pitcher", 4: "Bench Bat", 3: "Pure Batter"}
    got = {int(row["player_id"]): row for row in df.to_dict("records")}
    for record in inp.rankings.to_dict("records"):
        mine = got[int(record["player_id"])]
        assert mine["roles"] == record["roles"]
        assert int(mine["target_season"]) == 2027
        for col in (
            "war_bat",
            "war_pit",
            "war",
            "war_full_bat",
            "war_full_pit",
            "war_full",
            "prev_war",
        ):
            if pd.isna(record[col]):
                assert pd.isna(mine[col])
            else:
                assert float(mine[col]) == pytest.approx(float(record[col]))


def test_war_eval_write_war_csv_is_byte_stable(tmp_path: Path) -> None:
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    war_eval.write_war_csv(_export_input(None), first)
    war_eval.write_war_csv(_export_input(None), second)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()  # 비어 있지 않다
