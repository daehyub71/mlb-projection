"""내보내기(export.py) 테스트 (SPEC F11~F14·D14·D16·D21·D23·N1·N6, `contracts/web-data.schema.json`).

목표 시즌 Y = 2027 (입력 2022~2026). 정제 표·대상 표·예측 표·평가표는 이 파일에서 손으로 만든다.

선수 (작은 세계)
- 101 Alpha Catcher — 타자만. 타격 기록 2019~2026(8시즌) 중 기간 2022~2026(5시즌)만 history에 쓴다.
  생년월일 1999-07-15 → 2027-06-30 기준 27세(7월 생일 전). 정제 표 2026 나이 27 + 1 = 28로 대체하면 틀린다.
  포수 FRV 세부가 있다. 2021 수비 행(FIELDING_YEARS 밖)은 빠지고, 아웃 0인 포지션은 outs_by_pos에서 빠진다.
- 102 Beta Twoway — 타자·투수 둘 다 2027 대상(F12). 생년월일 1994-06-30 → 6월 30일 당일이 생일이라 33세.
  2025 투구는 아웃 0 → era·fip null, ip 0.
- 103 Gamma Pitcher — 투수만. 생년월일 없음 → 2025(input_last) 정제 표 나이 29 + (2027 − 2025) = 31.
  2026 미출전 → prev null. 2019 타격 행이 있어도 2027 타자 대상이 아니므로 bat null.
- 104 Delta Retired — players 표에만 있고 2027 대상이 아니다 → 파일 없음.

최종 모델: 타자 "extended", 투수 "weighted". group_error는 그 역할 최종 모델의 평가 행만 쓴다.
반올림은 `contracts.WEB_ROUND`의 키 이름 기준이다(group_error.mae 안의 키도 지표 이름 그대로 — ops 3자리 등).
"""

import copy
import dataclasses
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mlbproj import contracts as c
from mlbproj.contracts import Role
from mlbproj.export import ExportInput, build_web, dump_json, write_projection_csv, write_web
from mlbproj.schema import load_schema, validate
from mlbproj.tables import Tables

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "tests" / "fixtures" / "synthetic" / "web"
TARGET = 2027
SHA = "0123456789abcdef" * 4

# ---------------------------------------------------------------------------
# 도우미 — 계약 형식의 합성 표
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {"league": c.LEAGUE, "n_teams": 1}
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
    """행에 없는 열의 기본값. 결측 허용 열은 결측, 정수 0, 실수 0.0, 2020 출전량은 ×162/60."""
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
    if col.nullable:
        return None
    if col.kind == "int":
        return 0
    if col.kind == "float":
        return 0.0
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


def _row(pid: int, season: int, **values: Any) -> dict[str, Any]:
    return {"player_id": pid, "season": season, **values}


def _cohort_row(
    pid: int,
    role: Role,
    target: int,
    span: tuple[int, int],
    n_seasons: int,
    sample: int,
    sample_outs: int | None = None,
    window_first: int | None = None,
) -> dict[str, Any]:
    first = span[0] if window_first is None else window_first
    return {
        "player_id": pid,
        "role": role,
        "target_season": target,
        "window_first": first,
        "window_last": target - 1,
        "n_window": target - first,
        "input_first": span[0],
        "input_last": span[1],
        "n_seasons": n_seasons,
        "played_prev": span[1] == target - 1,
        "sample": sample,
        "sample_outs": sample_outs,
    }


def _ev(
    target: int, role: Role, model: str, metric: str, group: str, mae: float | None, **values: Any
) -> dict[str, Any]:
    return {
        "target_season": target,
        "role": role,
        "model": model,
        "metric": metric,
        "group": group,
        "n": 100,
        "mae": mae,
        "excluded": 0,
        **values,
    }


def _reference_dump(obj: Any) -> str:
    """모듈 docstring "직렬화"의 식 그대로 (기대값 계산용)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def _schema() -> dict[str, Any]:
    return load_schema(ROOT / c.WEB_SCHEMA_PATH)


def _def_name(path: str) -> str:
    return c.WEB_FILES.get(path, c.WEB_PLAYER_DEF)


def _assert_plain_json(value: Any, path: str = "$") -> None:
    """JSON에 그대로 쓸 수 있는 파이썬 값인지 (numpy·pandas 값, NaN·inf 금지)."""
    if value is None or type(value) in (str, bool, int):
        return
    if isinstance(value, float):
        assert math.isfinite(value), f"{path}: {value}"
        return
    if type(value) is list:
        for i, item in enumerate(value):
            _assert_plain_json(item, f"{path}[{i}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            assert type(key) is str, f"{path}: 키 {key!r}"
            _assert_plain_json(item, f"{path}.{key}")
        return
    raise AssertionError(f"{path}: JSON 값이 아니다 ({type(value).__name__}: {value!r})")


def _files_in(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# 작은 세계 (목표 2027)
# ---------------------------------------------------------------------------

META: dict[str, Any] = {
    "schema_version": c.WEB_SCHEMA_VERSION,
    "generated_at": "2026-11-20T09:00:00Z",
    "raw_collected_at": "2026-11-20T09:00:00Z",
    "params_version": "v1",
    "params_sha256": SHA,
    "target_season": TARGET,
    "input_seasons": [2022, 2023, 2024, 2025, 2026],
    "sources": [{"name": name, "url": url} for name, url in c.WEB_SOURCES],
    "disclaimer": c.WEB_DISCLAIMER,
    "counts": {"bat": 2, "pit": 2},
}

# 행 순서는 일부러 섞었다 (출력은 player_id·season 오름차순이어야 한다)
PLAYER_ROWS: list[dict[str, Any]] = [
    {
        "player_id": 104,
        "name": "Delta Retired",
        "birth_date": "1990-01-01",
        "primary_position": "1B",
        "bats": "R",
        "throws": "R",
        "active": False,
        "last_mlb_season": 2021,
    },
    {
        "player_id": 103,
        "name": "Gamma Pitcher",
        "birth_date": None,
        "primary_position": "P",
        "bats": None,
        "throws": "L",
        "active": False,
        "last_mlb_season": 2025,
    },
    {
        "player_id": 101,
        "name": "Alpha Catcher",
        "birth_date": "1999-07-15",
        "primary_position": "C",
        "bats": "R",
        "throws": "R",
        "active": True,
        "last_mlb_season": 2026,
    },
    {
        "player_id": 102,
        "name": "Beta Twoway",
        "birth_date": "1994-06-30",
        "primary_position": "TWP",
        "bats": "L",
        "throws": "R",
        "active": True,
        "last_mlb_season": 2026,
    },
]

# fmt: off
BAT_ROWS: list[dict[str, Any]] = [
    # 102 — 2025·2026 (n_seasons 2). 2025는 Statcast 결측
    _row(102, 2026, age=32, team_name="Team B", pa=450, hr=20, so=117, bb=40,
         obp=0.3333, slg=0.4777, ops=0.811, woba=0.345, xwoba=0.352,
         barrel_pct=10.06, hard_hit_pct=44.44, ev_avg=90.56),
    _row(102, 2025, age=31, team_name="Team B", pa=300, hr=12, so=90, bb=25,
         obp=0.30, slg=0.45, ops=0.75, woba=0.32),
    # 101 — 2026이 먼저, 기간 밖 2019~2021은 history에 나오면 안 된다
    _row(101, 2026, age=27, team_name="Team A", pa=655, hr=31, so=140, bb=67,
         obp=0.3621, slg=0.4994, ops=0.8614, woba=0.3684, xwoba=0.3712,
         barrel_pct=12.94, hard_hit_pct=48.12, ev_avg=91.62),
    _row(101, 2019, age=20, team_name="Team Old", pa=50, hr=1, so=15, bb=4, obp=0.2, slg=0.3, ops=0.5),
    _row(101, 2020, age=21, team_name="Team Old", pa=100, hr=3, so=25, bb=8),
    _row(101, 2021, age=22, team_name="Team Old", pa=300, hr=8, so=70, bb=20),
    _row(101, 2022, age=23, team_name="Team Old", pa=400, hr=10, so=100, bb=30,
         obp=0.3204, slg=0.4106, ops=0.731, woba=0.3156,
         barrel_pct=8.04, hard_hit_pct=40.16, ev_avg=89.04),
    _row(101, 2023, age=24, team_name="Team Old", pa=500, hr=15, so=110, bb=45,
         obp=0.33, slg=0.45, ops=0.78, woba=0.33, xwoba=0.335,
         barrel_pct=9.0, hard_hit_pct=42.0, ev_avg=90.0),
    _row(101, 2025, age=26, team_name="Team A", pa=620, hr=28, so=130, bb=50,
         obp=0.34, slg=0.5, ops=0.84, woba=0.36, xwoba=0.37,
         barrel_pct=12.0, hard_hit_pct=46.0, ev_avg=92.0),
    _row(101, 2024, age=25, team_name="Team A", pa=600, hr=25, so=120, bb=60,
         obp=0.351, slg=0.4888, ops=0.8398, woba=0.3611, xwoba=0.3666,
         barrel_pct=11.26, hard_hit_pct=45.54, ev_avg=91.27),
    # 103 — 오래된 타격 행. 2027 타자 대상이 아니다
    _row(103, 2019, age=23, team_name="Team C", pa=2, so=2),
    # 104 — 대상 아님
    _row(104, 2020, age=30, team_name="Team D", pa=200, hr=5, so=40, bb=15),
    _row(104, 2021, age=31, team_name="Team D", pa=150, hr=2, so=30, bb=10),
]

PIT_ROWS: list[dict[str, Any]] = [
    # 103 — 2021(기간 밖)·2022·2023·2025. 2024 미출전, 2026 미출전
    _row(103, 2025, age=29, team_name=None, g=40, gs=0, outs=150, bf=215, er=20, h=45,
         hr=6, bb=20, hbp=2, so=55, sv=10, xera=3.9, xwoba_against=0.312, whiff_pct=28.0),
    _row(103, 2021, age=25, team_name="Team C", g=30, gs=30, outs=450, bf=700, er=60, h=150,
         hr=18, bb=45, hbp=3, so=160),
    _row(103, 2023, age=27, team_name="Team C", g=28, gs=28, outs=481, bf=690, er=80, h=170,
         hr=25, bb=60, hbp=4, so=150),
    _row(103, 2022, age=26, team_name="Team C", g=31, gs=31, outs=540, bf=760, er=70, h=160,
         hr=20, bb=50, hbp=5, so=180, xera=3.62, xwoba_against=0.301, whiff_pct=25.04),
    # 102 — 2025는 아웃 0 (한 타자 이상 상대했지만 아웃을 못 잡았다)
    _row(102, 2026, age=32, team_name="Team B", g=20, gs=0, outs=60, bf=85, er=8, h=15,
         hr=2, bb=9, hbp=1, so=25, sv=3, xera=3.871, xwoba_against=0.2957, whiff_pct=27.33),
    _row(102, 2025, age=31, team_name="Team B", g=1, gs=0, outs=0, bf=3, er=2, h=2, hr=1, bb=1),
]
# fmt: on

# fmt: off
FIELDING_ROWS: list[dict[str, Any]] = [
    # 101 포수. 2026이 먼저(정렬 확인), 1B 300아웃, 나머지 포지션(예: LF)은 0 → 빠진다
    _row(101, 2026, frv_total=-1.26, frv_framing=-2.04, frv_blocking=0.5, frv_throwing=0.28,
         outs_2=1800, outs_3=300, outs_7=0),
    _row(101, 2021, frv_total=9.9, frv_framing=8.0, outs_2=1000),  # FIELDING_YEARS 밖
    _row(101, 2025, frv_total=5.24, frv_framing=4.06, frv_blocking=0.94, frv_throwing=0.26,
         outs_2=2400),
    # 102 야수 한 시즌 (1B·RF)
    _row(102, 2024, oaa=2.0, frv_total=1.04, frv_range=0.66, frv_arm=0.34, outs_3=600, outs_9=30),
    # 104 — 대상이 아니므로 어디에도 나오면 안 된다
    _row(104, 2022, oaa=-1.0, frv_total=-0.5, outs_3=900),
]
# fmt: on

# fmt: off
LEAGUE_BAT_ROWS: list[dict[str, Any]] = [
    {"season": s, "pa": 180_000, "obp": 0.31, "slg": round(ops - 0.31, 5), "ops": ops,
     "hr_per_pa": 0.03}
    for s, ops in ((2022, 0.706), (2023, 0.734), (2024, 0.711), (2025, 0.7111), (2026, 0.71849))
]
# fmt: on

# (시즌, era, k_bb_pct, fip_const). 2025 값은 Y−1이 아니라서 league_baseline에 쓰면 안 된다
_LEAGUE_PIT_VALUES = (
    (2021, 4.26, 14.0, 3.00),
    (2022, 3.97, 14.6, 3.10),
    (2023, 4.33, 13.9, 3.15),
    (2024, 4.08, 14.2, 3.20),
    (2025, 4.3049, 12.04, 3.05),
    (2026, 4.1523, 13.64, 3.12),
)
# fmt: off
LEAGUE_PIT_ROWS: list[dict[str, Any]] = [
    {"season": s, "bf": 185_000, "outs": 130_000, "er_per_out": era / 27, "era": era,
     "k_bb_pct": kbb, "fip_const": fip_const, "bf_per_out": 1.42}
    for s, era, kbb, fip_const in _LEAGUE_PIT_VALUES
]
# fmt: on

COHORT_ROWS: list[dict[str, Any]] = [
    # 목표 2025 (방해 행 — 2027 내보내기에 쓰이면 안 된다)
    _cohort_row(101, "bat", 2025, (2020, 2024), 5, 1900),
    _cohort_row(104, "bat", 2025, (2020, 2021), 2, 350),
    _cohort_row(103, "pit", 2025, (2021, 2023), 3, 2150, 1471),
    # 목표 2027
    _cohort_row(102, "pit", TARGET, (2025, 2026), 2, 88, 60),  # 3 + 85, 0 + 60
    _cohort_row(101, "bat", TARGET, (2022, 2026), 5, 2775),  # 400+500+600+620+655
    _cohort_row(103, "pit", TARGET, (2022, 2025), 3, 1665, 1171),  # 760+690+215, 540+481+150
    _cohort_row(102, "bat", TARGET, (2025, 2026), 2, 750),  # 300 + 450
]


def _bat_projection_rows(model: str) -> list[dict[str, Any]]:
    base = {"role": "bat", "target_season": TARGET, "model": model, "lg_ops": 0.71849}
    return [
        {
            **base,
            "player_id": 102,
            "input_first": 2025,
            "input_last": 2026,
            "n_seasons": 2,
            "sample_pa": 750,
            "pa": 402.96,
            "hr": 16.04,
            "obp": 0.32049,
            "slg": 0.46151,
            "ops": 0.782,
        },
        {
            **base,
            "player_id": 101,
            "input_first": 2022,
            "input_last": 2026,
            "n_seasons": 5,
            "sample_pa": 2775,
            "pa": 610.44,
            "hr": 27.06,
            "obp": 0.35123,
            "slg": 0.47249,
            "ops": 0.82372,
        },
    ]


def _pit_projection_rows(model: str) -> list[dict[str, Any]]:
    base = {
        "role": "pit",
        "target_season": TARGET,
        "model": model,
        "lg_era": 4.1523,
        "lg_k_bb_pct": 13.64,
    }
    return [
        {
            **base,
            "player_id": 103,
            "input_first": 2022,
            "input_last": 2025,
            "n_seasons": 3,
            "sample_bf": 1665,
            "sample_outs": 1171,
            "ip": 120.04,
            "era": 4.0051,
            "k_bb_pct": 14.24,
        },
        {
            **base,
            "player_id": 102,
            "input_first": 2025,
            "input_last": 2026,
            "n_seasons": 2,
            "sample_bf": 88,
            "sample_outs": 60,
            "ip": 35.27,
            "era": 3.9849,
            "k_bb_pct": 15.06,
        },
    ]


# group_error용 평가 행. 101 타자 = seasons_3_5, 102 타자 = seasons_1_2 (n_seasons 2 경계),
# 102 투수 = seasons_1_2, 103 투수 = seasons_3_5 (n_seasons 3 경계)
NAN = math.nan
GROUP_EVAL_ROWS: list[dict[str, Any]] = [
    # 타자 extended (최종 모델)
    _ev(2024, "bat", "extended", "ops", "seasons_3_5", 0.0712),
    _ev(2025, "bat", "extended", "ops", "seasons_3_5", 0.0631),
    _ev(2024, "bat", "extended", "pa", "seasons_3_5", 80.0),
    _ev(2025, "bat", "extended", "pa", "seasons_3_5", NAN),  # 결측은 평균에서 뺀다
    _ev(2024, "bat", "extended", "ops", "seasons_1_2", 0.0801),
    _ev(2025, "bat", "extended", "ops", "seasons_1_2", 0.0822),
    _ev(2024, "bat", "extended", "pa", "seasons_1_2", 100.04),
    _ev(2025, "bat", "extended", "pa", "seasons_1_2", 110.0),
    _ev(2024, "bat", "extended", "hr", "seasons_1_2", NAN),
    _ev(2025, "bat", "extended", "hr", "seasons_1_2", NAN),  # 전부 결측 → null
    # 방해 행: 다른 그룹·다른 모델
    _ev(2024, "bat", "extended", "ops", "all", 0.05),
    _ev(2024, "bat", "extended", "hr", "all", 3.0),
    _ev(2024, "bat", "extended", "ops", "prev_played", 0.04),
    _ev(2024, "bat", "weighted", "ops", "seasons_3_5", 0.09),
    _ev(2024, "bat", "weighted", "hr", "seasons_3_5", 9.9),
    _ev(2024, "bat", "league", "ops", "seasons_3_5", 0.2),
    # 투수 weighted (최종 모델)
    _ev(2024, "pit", "weighted", "era", "seasons_1_2", 0.95),
    _ev(2025, "pit", "weighted", "era", "seasons_1_2", 1.01),
    _ev(2024, "pit", "weighted", "ip", "seasons_1_2", 20.0),
    _ev(2025, "pit", "weighted", "ip", "seasons_1_2", 21.0),
    _ev(2024, "pit", "weighted", "k_bb_pct", "seasons_1_2", 3.84),
    _ev(2024, "pit", "weighted", "era", "seasons_3_5", 0.8),
    _ev(2025, "pit", "weighted", "era", "seasons_3_5", 0.9),
    _ev(2024, "pit", "weighted", "ip", "seasons_3_5", 15.0),
    _ev(2025, "pit", "weighted", "ip", "seasons_3_5", 17.0),
    # 방해 행
    _ev(2024, "pit", "weighted", "k_bb_pct", "all", 4.0),
    _ev(2024, "pit", "extended", "era", "seasons_1_2", 5.0),
    _ev(2024, "pit", "extended", "k_bb_pct", "seasons_3_5", 7.7),
    _ev(2025, "pit", "baseline", "k_bb_pct", "seasons_3_5", 8.8),
]


def _world(
    *,
    model: Mapping[Role, str] | None = None,
    evaluations: Sequence[Mapping[str, Any]] = GROUP_EVAL_ROWS,
    final_test: dict[str, Any] | None = None,
    pit_rows: Sequence[Mapping[str, Any]] = PIT_ROWS,
    bat_rows: Sequence[Mapping[str, Any]] = BAT_ROWS,
    league_bat_rows: Sequence[Mapping[str, Any]] = LEAGUE_BAT_ROWS,
    league_pit_rows: Sequence[Mapping[str, Any]] = LEAGUE_PIT_ROWS,
    fielding_rows: Sequence[Mapping[str, Any]] = FIELDING_ROWS,
) -> ExportInput:
    models: dict[Role, str] = {"bat": "extended", "pit": "weighted", **(model or {})}
    tables = Tables(
        seasons={
            "bat": _frame(c.BATTING_SEASONS, bat_rows),
            "pit": _frame(c.PITCHING_SEASONS, pit_rows),
        },
        cohorts=_frame(c.COHORTS, COHORT_ROWS),
        league={
            "bat": _frame(c.LEAGUE_BAT, league_bat_rows),
            "pit": _frame(c.LEAGUE_PIT, league_pit_rows),
        },
        debuts={},
        players=_frame(c.PLAYERS, PLAYER_ROWS),
        fielding=_frame(c.FIELDING_SEASONS, fielding_rows),
    )
    return ExportInput(
        tables=tables,
        projections={
            "bat": _frame(c.BAT_PROJECTIONS, _bat_projection_rows(models["bat"])),
            "pit": _frame(c.PIT_PROJECTIONS, _pit_projection_rows(models["pit"])),
        },
        model=models,
        evaluations=_frame(c.EVALUATIONS, evaluations),
        final_test=copy.deepcopy(final_test),
        meta=copy.deepcopy(META),
    )


# ---------------------------------------------------------------------------
# 기대값 (손 계산)
# ---------------------------------------------------------------------------

# 예측: pa 610.44 → 610.4, hr 27.06 → 27.1, obp .35123 → .351, slg .47249 → .472, ops .82372 → .824
PROJ_101: dict[str, Any] = {"pa": 610.4, "hr": 27.1, "obp": 0.351, "slg": 0.472, "ops": 0.824}
# pa 402.96 → 403.0, hr 16.04 → 16.0, obp .32049 → .32, slg .46151 → .462, ops .782
PROJ_102_BAT: dict[str, Any] = {"pa": 403.0, "hr": 16.0, "obp": 0.32, "slg": 0.462, "ops": 0.782}
# ip 35.27 → 35.3, era 3.9849 → 3.98, k_bb 15.06 → 15.1
PROJ_102_PIT: dict[str, Any] = {"ip": 35.3, "era": 3.98, "k_bb_pct": 15.1}
# ip 120.04 → 120.0, era 4.0051 → 4.01, k_bb 14.24 → 14.2
PROJ_103: dict[str, Any] = {"ip": 120.0, "era": 4.01, "k_bb_pct": 14.2}

# 리그 Y−1 = 2026 행: ops .71849 → .718, era 4.1523 → 4.15, k_bb 13.64 → 13.6 (2025 값이면 틀린다)
LG_BAT: dict[str, Any] = {"ops": 0.718}
LG_PIT: dict[str, Any] = {"era": 4.15, "k_bb_pct": 13.6}

WINDOW_101: dict[str, Any] = {"input_first": 2022, "input_last": 2026, "n_seasons": 5}
WINDOW_102: dict[str, Any] = {"input_first": 2025, "input_last": 2026, "n_seasons": 2}
WINDOW_103: dict[str, Any] = {"input_first": 2022, "input_last": 2025, "n_seasons": 3}

# prev (Y−1 = 2026 행)
PREV_101: dict[str, Any] = {"season": 2026, "pa": 655, "hr": 31, "ops": 0.861}  # ops .8614 → .861
PREV_102_BAT: dict[str, Any] = {"season": 2026, "pa": 450, "hr": 20, "ops": 0.811}
# ip = 60/3 = 20.0, era = 27·8/60 = 3.6, k_bb = 100·(25 − 9)/85 = 18.82 → 18.8
PREV_102_PIT: dict[str, Any] = {"season": 2026, "ip": 20.0, "era": 3.6, "k_bb_pct": 18.8}

# age: 101 = 2027 − 1999 − 1(7/15 생일 전) = 27, 102 = 2027 − 1994 = 33(6/30 당일은 생일 지남),
# 103 = 29 + (2027 − 2025) = 31. team_name = 그 역할 input_last 행 (103은 2025 결측 → null)
SUMMARY_101: dict[str, Any] = {
    "player_id": 101,
    "primary_position": "C",
    "name": "Alpha Catcher",
    "age": 27,
    "team_name": "Team A",
    "active": True,
    "last_mlb_season": 2026,
}
SUMMARY_102: dict[str, Any] = {
    "player_id": 102,
    "primary_position": "TWP",
    "name": "Beta Twoway",
    "age": 33,
    "team_name": "Team B",
    "active": True,
    "last_mlb_season": 2026,
}
SUMMARY_103: dict[str, Any] = {
    "player_id": 103,
    "primary_position": "P",
    "name": "Gamma Pitcher",
    "age": 31,
    "team_name": None,
    "active": False,
    "last_mlb_season": 2025,
}

ROW_101: dict[str, Any] = {
    "player": SUMMARY_101,
    "window": WINDOW_101,
    "sample_pa": 2775,
    "projection": PROJ_101,
    "prev": PREV_101,
}
ROW_102_BAT: dict[str, Any] = {
    "player": SUMMARY_102,
    "window": WINDOW_102,
    "sample_pa": 750,
    "projection": PROJ_102_BAT,
    "prev": PREV_102_BAT,
}
# sample_ip: 102 = 60/3 = 20.0, 103 = 1171/3 = 390.33 → 390.3
ROW_102_PIT: dict[str, Any] = {
    "player": SUMMARY_102,
    "window": WINDOW_102,
    "sample_bf": 88,
    "sample_ip": 20.0,
    "projection": PROJ_102_PIT,
    "prev": PREV_102_PIT,
}
ROW_103: dict[str, Any] = {
    "player": SUMMARY_103,
    "window": WINDOW_103,
    "sample_bf": 1665,
    "sample_ip": 390.3,
    "projection": PROJ_103,
    "prev": None,
}


def _bat_season(season: int, age: int, team: str, pa: int, hr: int, **rest: Any) -> dict[str, Any]:
    return {
        "season": season,
        "age": age,
        "team_name": team,
        "is_short_season": False,
        "pa": pa,
        "hr": hr,
        **rest,
    }


# fmt: off
HISTORY_101 = [
    # k = 100·100/400 = 25.0, bb = 100·30/400 = 7.5; obp .3204 → .32, slg .4106 → .411, woba .3156 → .316
    _bat_season(2022, 23, "Team Old", 400, 10, obp=0.32, slg=0.411, ops=0.731, k_pct=25.0,
                bb_pct=7.5, woba=0.316, xwoba=None, barrel_pct=8.0, hard_hit_pct=40.2,
                ev_avg=89.0),
    # k = 100·110/500 = 22.0, bb = 100·45/500 = 9.0
    _bat_season(2023, 24, "Team Old", 500, 15, obp=0.33, slg=0.45, ops=0.78, k_pct=22.0,
                bb_pct=9.0, woba=0.33, xwoba=0.335, barrel_pct=9.0, hard_hit_pct=42.0,
                ev_avg=90.0),
    # k = 100·120/600 = 20.0, bb = 100·60/600 = 10.0; slg .4888 → .489, ops .8398 → .84, xwoba .3666 → .367
    _bat_season(2024, 25, "Team A", 600, 25, obp=0.351, slg=0.489, ops=0.84, k_pct=20.0,
                bb_pct=10.0, woba=0.361, xwoba=0.367, barrel_pct=11.3, hard_hit_pct=45.5,
                ev_avg=91.3),
    # k = 100·130/620 = 20.97 → 21.0, bb = 100·50/620 = 8.06 → 8.1
    _bat_season(2025, 26, "Team A", 620, 28, obp=0.34, slg=0.5, ops=0.84, k_pct=21.0,
                bb_pct=8.1, woba=0.36, xwoba=0.37, barrel_pct=12.0, hard_hit_pct=46.0,
                ev_avg=92.0),
    # k = 100·140/655 = 21.37 → 21.4, bb = 100·67/655 = 10.23 → 10.2
    _bat_season(2026, 27, "Team A", 655, 31, obp=0.362, slg=0.499, ops=0.861, k_pct=21.4,
                bb_pct=10.2, woba=0.368, xwoba=0.371, barrel_pct=12.9, hard_hit_pct=48.1,
                ev_avg=91.6),
]

HISTORY_102_BAT = [
    # k = 100·90/300 = 30.0, bb = 100·25/300 = 8.33 → 8.3, Statcast 결측 → null
    _bat_season(2025, 31, "Team B", 300, 12, obp=0.3, slg=0.45, ops=0.75, k_pct=30.0,
                bb_pct=8.3, woba=0.32, xwoba=None, barrel_pct=None, hard_hit_pct=None,
                ev_avg=None),
    # k = 100·117/450 = 26.0, bb = 100·40/450 = 8.89 → 8.9
    _bat_season(2026, 32, "Team B", 450, 20, obp=0.333, slg=0.478, ops=0.811, k_pct=26.0,
                bb_pct=8.9, woba=0.345, xwoba=0.352, barrel_pct=10.1, hard_hit_pct=44.4,
                ev_avg=90.6),
]
# fmt: on


def _pit_season(
    season: int, age: int, team: str | None, g: int, gs: int, **rest: Any
) -> dict[str, Any]:
    return {
        "season": season,
        "age": age,
        "team_name": team,
        "is_short_season": False,
        "g": g,
        "gs": gs,
        **rest,
    }


# fmt: off
HISTORY_102_PIT = [
    # 아웃 0: ip 0.0, era·fip null. k = 100·0/3 = 0.0, bb = 100·1/3 = 33.3, k_bb = 0 − 33.33 = −33.3
    _pit_season(2025, 31, "Team B", 1, 0, ip=0.0, era=None, fip=None, k_pct=0.0, bb_pct=33.3,
                k_bb_pct=-33.3, xera=None, xwoba_against=None, whiff_pct=None),
    # ip 20.0, era = 27·8/60 = 3.6, fip = (13·2 + 3·(9+1) − 2·25)/20 + 3.12 = 6/20 + 3.12 = 3.42
    # k = 2500/85 = 29.41 → 29.4, bb = 900/85 = 10.59 → 10.6, k_bb = 18.82 → 18.8
    _pit_season(2026, 32, "Team B", 20, 0, ip=20.0, era=3.6, fip=3.42, k_pct=29.4, bb_pct=10.6,
                k_bb_pct=18.8, xera=3.87, xwoba_against=0.296, whiff_pct=27.3),
]

HISTORY_103 = [
    # ip = 540/3 = 180.0, era = 27·70/540 = 3.5, fip = (260 + 165 − 360)/180 + 3.10 = 3.461 → 3.46
    # k = 18000/760 = 23.68 → 23.7, bb = 5000/760 = 6.58 → 6.6, k_bb = 17.11 → 17.1
    _pit_season(2022, 26, "Team C", 31, 31, ip=180.0, era=3.5, fip=3.46, k_pct=23.7, bb_pct=6.6,
                k_bb_pct=17.1, xera=3.62, xwoba_against=0.301, whiff_pct=25.0),
    # ip = 481/3 = 160.33 → 160.3, era = 2160/481 = 4.4906 → 4.49,
    # fip = (325 + 192 − 300)/(481/3) + 3.15 = 1.3534 + 3.15 = 4.5034 → 4.5
    # k = 15000/690 = 21.74 → 21.7, bb = 6000/690 = 8.70 → 8.7, k_bb = 13.04 → 13.0
    _pit_season(2023, 27, "Team C", 28, 28, ip=160.3, era=4.49, fip=4.5, k_pct=21.7, bb_pct=8.7,
                k_bb_pct=13.0, xera=None, xwoba_against=None, whiff_pct=None),
    # ip = 150/3 = 50.0, era = 540/150 = 3.6, fip = (78 + 66 − 110)/50 + 3.05 = 0.68 + 3.05 = 3.73
    # k = 5500/215 = 25.58 → 25.6, bb = 2000/215 = 9.30 → 9.3, k_bb = 16.28 → 16.3
    _pit_season(2025, 29, None, 40, 0, ip=50.0, era=3.6, fip=3.73, k_pct=25.6, bb_pct=9.3,
                k_bb_pct=16.3, xera=3.9, xwoba_against=0.312, whiff_pct=28.0),
]
# fmt: on

_NO_FRV_PARTS = {"frv_range": None, "frv_arm": None, "frv_dp": None}
_NO_CATCHER_PARTS = {"frv_framing": None, "frv_blocking": None, "frv_throwing": None}
# fmt: off
FIELDING_101 = [
    # frv 5.24 → 5.2, framing 4.06 → 4.1, blocking .94 → .9, throwing .26 → .3. 포수만 → {"C"}
    {"season": 2025, "oaa": None, "frv_total": 5.2, **_NO_FRV_PARTS, "frv_framing": 4.1,
     "frv_blocking": 0.9, "frv_throwing": 0.3, "outs_by_pos": {"C": 2400}},
    # frv −1.26 → −1.3, framing −2.04 → −2.0, throwing .28 → .3. 아웃 0인 포지션은 없다
    {"season": 2026, "oaa": None, "frv_total": -1.3, **_NO_FRV_PARTS, "frv_framing": -2.0,
     "frv_blocking": 0.5, "frv_throwing": 0.3, "outs_by_pos": {"C": 1800, "1B": 300}},
]
# frv 1.04 → 1.0, range .66 → .7, arm .34 → .3
FIELDING_102 = [
    {"season": 2024, "oaa": 2.0, "frv_total": 1.0, "frv_range": 0.7, "frv_arm": 0.3,
     "frv_dp": None, **_NO_CATCHER_PARTS, "outs_by_pos": {"1B": 600, "RF": 30}},
]
# fmt: on

# group_error (최종 모델 행만, 목표 시즌 평균, 결측 제외, 지표 키 이름으로 반올림)
# 101 타자 seasons_3_5: ops (.0712 + .0631)/2 = .06715 → .067, pa 80.0 (2025 결측 제외), hr 행 없음 → null
GE_101: dict[str, Any] = {
    "group": "seasons_3_5",
    "mae": {"ops": 0.067, "pa": 80.0, "hr": None},
    "band80": {"ops": None},
}
# 102 타자 seasons_1_2: ops (.0801 + .0822)/2 = .08115 → .081, pa (100.04 + 110)/2 = 105.02 → 105.0, hr 전부 결측
GE_102_BAT: dict[str, Any] = {
    "group": "seasons_1_2",
    "mae": {"ops": 0.081, "pa": 105.0, "hr": None},
    "band80": {"ops": None},
}
# 102 투수 seasons_1_2: era (.95 + 1.01)/2 = .98, ip (20 + 21)/2 = 20.5, k_bb 3.84 → 3.8
GE_102_PIT: dict[str, Any] = {
    "group": "seasons_1_2",
    "mae": {"era": 0.98, "ip": 20.5, "k_bb_pct": 3.8},
    "band80": {"era": None, "k_bb_pct": None},
}
# 103 투수 seasons_3_5: era (.8 + .9)/2 = .85, ip (15 + 17)/2 = 16.0, k_bb 행 없음 → null
GE_103: dict[str, Any] = {
    "group": "seasons_3_5",
    "mae": {"era": 0.85, "ip": 16.0, "k_bb_pct": None},
    "band80": {"era": None, "k_bb_pct": None},
}

PLAYER_101: dict[str, Any] = {
    "meta": META,
    "player": {
        "player_id": 101,
        "name": "Alpha Catcher",
        "birth_date": "1999-07-15",
        "primary_position": "C",
        "bats": "R",
        "throws": "R",
        "active": True,
        "last_mlb_season": 2026,
    },
    "bat": {
        "window": WINDOW_101,
        "sample_pa": 2775,
        "projection": PROJ_101,
        "prev": PREV_101,
        "league_baseline": LG_BAT,
        "group_error": GE_101,
        "history": HISTORY_101,
    },
    "pit": None,
    "fielding": FIELDING_101,
}

PLAYER_102: dict[str, Any] = {
    "meta": META,
    "player": {
        "player_id": 102,
        "name": "Beta Twoway",
        "birth_date": "1994-06-30",
        "primary_position": "TWP",
        "bats": "L",
        "throws": "R",
        "active": True,
        "last_mlb_season": 2026,
    },
    "bat": {
        "window": WINDOW_102,
        "sample_pa": 750,
        "projection": PROJ_102_BAT,
        "prev": PREV_102_BAT,
        "league_baseline": LG_BAT,
        "group_error": GE_102_BAT,
        "history": HISTORY_102_BAT,
    },
    "pit": {
        "window": WINDOW_102,
        "sample_bf": 88,
        "sample_ip": 20.0,
        "projection": PROJ_102_PIT,
        "prev": PREV_102_PIT,
        "league_baseline": LG_PIT,
        "group_error": GE_102_PIT,
        "history": HISTORY_102_PIT,
    },
    "fielding": FIELDING_102,
}

PLAYER_103: dict[str, Any] = {
    "meta": META,
    "player": {
        "player_id": 103,
        "name": "Gamma Pitcher",
        "birth_date": None,
        "primary_position": "P",
        "bats": None,
        "throws": "L",
        "active": False,
        "last_mlb_season": 2025,
    },
    "bat": None,  # 2019 타격 행이 있어도 2027 타자 대상이 아니다 (F12)
    "pit": {
        "window": WINDOW_103,
        "sample_bf": 1665,
        "sample_ip": 390.3,
        "projection": PROJ_103,
        "prev": None,
        "league_baseline": LG_PIT,
        "group_error": GE_103,
        "history": HISTORY_103,
    },
    "fielding": [],
}

EXPECTED_PATHS = {
    "batters.json",
    "pitchers.json",
    "validation.json",
    "players/101.json",
    "players/102.json",
    "players/103.json",
}


# ---------------------------------------------------------------------------
# build_web — 예측표·선수 상세
# ---------------------------------------------------------------------------


def test_export_build_web_file_set() -> None:
    # 두 역할 대상의 합집합(101·102·103)마다 상세 한 파일. 대상이 아닌 104는 없다
    assert set(build_web(_world())) == EXPECTED_PATHS


def test_export_build_web_batters_table() -> None:
    files = build_web(_world())
    assert files["batters.json"] == {
        "meta": META,
        "role": "bat",
        "model": "extended",
        "league_baseline": LG_BAT,
        "bands": {"seasons_1_2": {"ops": None}, "seasons_3_5": {"ops": None}},  # 범위 표 없음
        "rows": [ROW_101, ROW_102_BAT],  # player_id 순
    }


def test_export_build_web_pitchers_table() -> None:
    files = build_web(_world())
    assert files["pitchers.json"] == {
        "meta": META,
        "role": "pit",
        "model": "weighted",
        "league_baseline": LG_PIT,
        "bands": {
            "seasons_1_2": {"era": None, "k_bb_pct": None},
            "seasons_3_5": {"era": None, "k_bb_pct": None},
        },
        "rows": [ROW_102_PIT, ROW_103],
    }


def test_export_build_web_player_batter_only() -> None:
    assert build_web(_world())["players/101.json"] == PLAYER_101


def test_export_build_web_player_two_way() -> None:
    assert build_web(_world())["players/102.json"] == PLAYER_102


def test_export_build_web_player_pitcher_only() -> None:
    assert build_web(_world())["players/103.json"] == PLAYER_103


def test_export_build_web_history_window_only() -> None:
    # 101 타격 기록은 8시즌이지만 history는 input_first(2022) ~ input_last(2026)만
    files = build_web(_world())
    assert [h["season"] for h in files["players/101.json"]["bat"]["history"]] == [
        2022,
        2023,
        2024,
        2025,
        2026,
    ]
    # 103 투구 2021(기간 밖)은 빠지고 미출전 2024는 행이 없다
    assert [h["season"] for h in files["players/103.json"]["pit"]["history"]] == [2022, 2023, 2025]


def test_export_build_web_age_fallback_missing_is_null() -> None:
    # 생년월일도 없고 input_last(2025) 행의 나이도 없으면 null
    pit_rows = [dict(r) for r in PIT_ROWS]
    for r in pit_rows:
        if r["player_id"] == 103 and r["season"] == 2025:
            r["age"] = None
    files = build_web(_world(pit_rows=pit_rows))
    row_103 = files["pitchers.json"]["rows"][1]
    assert row_103["player"]["player_id"] == 103
    assert row_103["player"]["age"] is None
    assert files["players/103.json"]["pit"]["history"][-1]["age"] is None


def test_export_build_web_ints_stay_ints() -> None:
    files = build_web(_world())
    bat_row = files["batters.json"]["rows"][0]
    pit_row = files["pitchers.json"]["rows"][0]
    detail = files["players/101.json"]
    two_way = files["players/102.json"]
    ints = [
        bat_row["player"]["player_id"],
        bat_row["player"]["age"],
        bat_row["player"]["last_mlb_season"],
        *bat_row["window"].values(),
        bat_row["sample_pa"],
        bat_row["prev"]["season"],
        bat_row["prev"]["pa"],  # WEB_ROUND에 "pa"가 있어도 정수 열은 int
        bat_row["prev"]["hr"],
        pit_row["sample_bf"],
        detail["bat"]["history"][0]["pa"],
        detail["bat"]["history"][0]["hr"],
        detail["bat"]["history"][0]["age"],
        *detail["fielding"][1]["outs_by_pos"].values(),
        two_way["pit"]["history"][1]["g"],
        two_way["pit"]["history"][1]["gs"],
    ]
    assert all(type(v) is int for v in ints), [type(v).__name__ for v in ints]
    assert type(detail["player"]["active"]) is bool
    assert type(detail["bat"]["history"][0]["is_short_season"]) is bool


def test_export_build_web_plain_json_values() -> None:
    # numpy·pandas 값, NaN·pd.NA가 남으면 안 된다 (결측은 None)
    for path, obj in build_web(_world()).items():
        _assert_plain_json(obj, path)


@pytest.mark.parametrize("path", sorted(EXPECTED_PATHS))
def test_export_build_web_matches_schema(path: str) -> None:
    files = build_web(_world())
    assert validate(files[path], _def_name(path), _schema()) == []


def test_export_build_web_with_final_test_matches_schema() -> None:
    files = build_web(_world(evaluations=VALIDATION_EVAL_ROWS, final_test=FINAL_TEST))
    schema = _schema()
    for path, obj in files.items():
        assert validate(obj, _def_name(path), schema) == [], path


def test_export_build_web_deterministic_bytes() -> None:
    first = build_web(_world())
    second = build_web(_world())
    assert sorted(first) == sorted(second)
    for path in first:
        assert dump_json(first[path]).encode("utf-8") == dump_json(second[path]).encode("utf-8")


def _reversed_input(inp: ExportInput) -> ExportInput:
    def rev(df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[::-1].reset_index(drop=True)

    t = inp.tables
    tables = Tables(
        seasons={role: rev(df) for role, df in t.seasons.items()},
        cohorts=rev(t.cohorts),
        league={role: rev(df) for role, df in t.league.items()},
        debuts=dict(t.debuts),
        players=rev(t.players),
        fielding=rev(t.fielding),
    )
    return dataclasses.replace(
        inp,
        tables=tables,
        projections={role: rev(df) for role, df in inp.projections.items()},
        evaluations=rev(inp.evaluations),
    )


def test_export_build_web_ignores_input_row_order() -> None:
    # 모든 출력 순서가 규칙으로 정해지므로 입력 행 순서를 뒤집어도 같은 결과 (N1)
    assert build_web(_reversed_input(_world())) == build_web(_world())


def test_export_build_web_ignores_rows_at_or_after_target(monkeypatch: pytest.MonkeyPatch) -> None:
    # N2: 목표 Y(2027) 행이 표에 섞여 있어도 결과가 같다. history·prev·나이·팀은 Y−1까지,
    # 리그 기준선은 Y−1 행, FIP 상수는 그 시즌 행, 수비도 Y−1까지 (FIELDING_YEARS에 2027을 넣어 시험)
    expected = build_web(_world())
    monkeypatch.setattr(c, "FIELDING_YEARS", (*c.FIELDING_YEARS, TARGET))
    # fmt: off
    future_bat = [
        *BAT_ROWS,
        _row(101, TARGET, age=99, team_name="Team Future", pa=700, hr=60, so=10, bb=200,
             obp=0.6, slg=0.9, ops=1.5),
    ]
    # 103은 2026 미출전 — 2027 행이 prev 자리를 차지하면 안 된다
    future_pit = [
        *PIT_ROWS,
        _row(103, TARGET, age=99, team_name="Team Future", g=70, gs=0, outs=300, bf=400, er=1,
             so=200, bb=1),
    ]
    league_bat = [
        *LEAGUE_BAT_ROWS,
        {"season": TARGET, "pa": 1, "obp": 0.5, "slg": 0.5, "ops": 1.0, "hr_per_pa": 0.5},
    ]
    league_pit = [
        *LEAGUE_PIT_ROWS,
        {"season": TARGET, "bf": 1, "outs": 1, "er_per_out": 1 / 3, "era": 9.0,
         "k_bb_pct": 50.0, "fip_const": 9.0, "bf_per_out": 1.0},
    ]
    fielding = [
        *FIELDING_ROWS,
        _row(101, TARGET, frv_total=99.0, outs_2=5000),
        _row(103, TARGET, oaa=9.0, outs_8=100),
    ]
    # fmt: on
    future = _world(
        bat_rows=future_bat,
        pit_rows=future_pit,
        league_bat_rows=league_bat,
        league_pit_rows=league_pit,
        fielding_rows=fielding,
    )
    assert build_web(future) == expected


def _with_pit_2026(**values: Any) -> list[dict[str, Any]]:
    """102의 2026 투구 행만 values로 바꾼 투구 표 행 (나머지 열은 0·결측)."""
    rows = [dict(r) for r in PIT_ROWS if not (r["player_id"] == 102 and r["season"] == 2026)]
    return [*rows, _row(102, 2026, age=32, team_name="Team B", **values)]


@pytest.mark.parametrize(
    ("values", "prev", "season"),
    [
        # 아웃 0, 상대 타자 2 (볼넷 둘·자책 1): ip 0.0, era·fip null,
        # k = 100·0/2 = 0.0, bb = 100·2/2 = 100.0, k_bb = 100·(0 − 2)/2 = −100.0
        (
            {"g": 1, "outs": 0, "bf": 2, "er": 1, "h": 0, "bb": 2},
            {"season": 2026, "ip": 0.0, "era": None, "k_bb_pct": -100.0},
            {
                "ip": 0.0,
                "era": None,
                "fip": None,
                "k_pct": 0.0,
                "bb_pct": 100.0,
                "k_bb_pct": -100.0,
            },
        ),
        # 견제사 한 개로 아웃 1, 상대 타자 0: ip 1/3 = 0.333 → 0.3, era 27·0/1 = 0.0,
        # fip = (0)/(1/3) + 3.12 = 3.12, k·bb·k_bb는 분모 0 → null
        (
            {"g": 1, "outs": 1, "bf": 0},
            {"season": 2026, "ip": 0.3, "era": 0.0, "k_bb_pct": None},
            {"ip": 0.3, "era": 0.0, "fip": 3.12, "k_pct": None, "bb_pct": None, "k_bb_pct": None},
        ),
    ],
)
def test_export_build_web_pitcher_zero_denominators_are_null(
    values: dict[str, Any], prev: dict[str, Any], season: dict[str, Any]
) -> None:
    files = build_web(_world(pit_rows=_with_pit_2026(**values)))
    row_102 = files["pitchers.json"]["rows"][0]
    assert row_102["player"]["player_id"] == 102
    assert row_102["prev"] == prev
    detail = files["players/102.json"]["pit"]
    assert detail["prev"] == prev  # 예측표 행과 같은 값
    last = detail["history"][-1]
    assert last["season"] == 2026
    assert {k: last[k] for k in season} == season
    schema = _schema()
    for path in ("pitchers.json", "players/102.json"):
        assert validate(files[path], _def_name(path), schema) == [], path


def test_export_build_web_negative_zero_becomes_zero() -> None:
    # frv_dp −0.04 → round 1자리 = −0.0 → 0.0 ("-0.0"이 화면에 나오지 않게)
    fielding = [dict(r) for r in FIELDING_ROWS]
    for r in fielding:
        if r["player_id"] == 102:
            r["frv_dp"] = -0.04
    files = build_web(_world(fielding_rows=fielding))
    value = files["players/102.json"]["fielding"][0]["frv_dp"]
    assert value == 0.0 and math.copysign(1.0, value) == 1.0
    for path, obj in files.items():
        assert ":-0.0," not in dump_json(obj) and ":-0.0}" not in dump_json(obj), path


def test_export_build_web_without_fielding_table() -> None:
    # Tables.fielding 기본값(빈 표)이면 모든 선수 fielding = []
    inp = _world()
    inp = dataclasses.replace(inp, tables=dataclasses.replace(inp.tables, fielding=pd.DataFrame()))
    files = build_web(inp)
    for pid in (101, 102, 103):
        assert files[f"players/{pid}.json"]["fielding"] == []


def test_export_build_web_empty_evaluations() -> None:
    # 평가 행이 하나도 없으면 검증 행은 [], group_error 값은 모두 null (스키마 통과)
    files = build_web(_world(evaluations=[]))
    assert files["validation.json"]["rows"] == []
    assert files["players/101.json"]["bat"]["group_error"] == {
        "group": "seasons_3_5",
        "mae": {"ops": None, "pa": None, "hr": None},
        "band80": {"ops": None},
    }
    assert files["players/103.json"]["pit"]["group_error"] == {
        "group": "seasons_3_5",
        "mae": {"era": None, "ip": None, "k_bb_pct": None},
        "band80": {"era": None, "k_bb_pct": None},
    }
    schema = _schema()
    for path, obj in files.items():
        assert validate(obj, _def_name(path), schema) == [], path


# ---------------------------------------------------------------------------
# build_web — group_error
# ---------------------------------------------------------------------------


def test_export_group_error_uses_final_model_rows() -> None:
    # 최종 모델을 바꾸면 그 모델 행만 쓴다 (타자 weighted, 투수 extended)
    files = build_web(_world(model={"bat": "weighted", "pit": "extended"}))
    assert files["batters.json"]["model"] == "weighted"
    assert files["pitchers.json"]["model"] == "extended"
    # 101 seasons_3_5 weighted: ops .09, hr 9.9, pa 행 없음
    assert files["players/101.json"]["bat"]["group_error"] == {
        "group": "seasons_3_5",
        "mae": {"ops": 0.09, "pa": None, "hr": 9.9},
        "band80": {"ops": None},
    }
    # 102 seasons_1_2 weighted 타자 행 없음
    assert files["players/102.json"]["bat"]["group_error"] == {
        "group": "seasons_1_2",
        "mae": {"ops": None, "pa": None, "hr": None},
        "band80": {"ops": None},
    }
    # 102 seasons_1_2 extended 투수: era 5.0만
    assert files["players/102.json"]["pit"]["group_error"] == {
        "group": "seasons_1_2",
        "mae": {"era": 5.0, "ip": None, "k_bb_pct": None},
        "band80": {"era": None, "k_bb_pct": None},
    }
    # 103 seasons_3_5 extended 투수: k_bb 7.7만 (baseline 8.8은 다른 모델)
    assert files["players/103.json"]["pit"]["group_error"] == {
        "group": "seasons_3_5",
        "mae": {"era": None, "ip": None, "k_bb_pct": 7.7},
        "band80": {"era": None, "k_bb_pct": None},
    }


def test_export_group_error_averages_every_target_season() -> None:
    # 최종 검증 뒤에는 평가표에 목표 2026 행도 들어온다 (합성 값, 기록과 함께). 목표 시즌 전부를 평균한다
    rows = [
        *GROUP_EVAL_ROWS,
        _ev(2026, "bat", "extended", "ops", "seasons_3_5", 0.0807),
        _ev(2026, "bat", "extended", "hr", "seasons_3_5", 2.96),
    ]
    files = build_web(_world(evaluations=rows, final_test=FINAL_TEST))
    # ops (.0712 + .0631 + .0807)/3 = .215/3 = .07167 → .072, pa 80.0, hr 2.96 → 3.0
    assert files["players/101.json"]["bat"]["group_error"] == {
        "group": "seasons_3_5",
        "mae": {"ops": 0.072, "pa": 80.0, "hr": 3.0},
        "band80": {"ops": None},
    }


# ---------------------------------------------------------------------------
# build_web — validation.json
# ---------------------------------------------------------------------------

# 입력 순서는 섞었다. 정렬 키 (target_season, role, model, metric, group)는 문자열 사전순
# fmt: off
VALIDATION_EVAL_ROWS: list[dict[str, Any]] = [
    _ev(2025, "pit", "weighted", "era", "all", 0.87412, n=300, mae_weighted=0.91236,
        baseline_mae_league=0.95, baseline_mae_prev=1.27449, excluded=120),
    _ev(2024, "bat", "weighted", "ops", "all", 0.071149, n=410, mae_weighted=NAN,
        baseline_mae_league=0.076849, baseline_mae_prev=0.096911, excluded=1100),
    _ev(2024, "bat", "baseline", "pa", "seasons_1_2", 80.12346, n=500,
        baseline_mae_prev=80.12346),
    _ev(2024, "bat", "weighted", "ops", "seasons_3_5", NAN, n=0, excluded=7),
    _ev(2024, "bat", "weighted", "ops", "prev_absent", 0.1, n=3, mae_weighted=0.12,
        baseline_mae_league=0.2, baseline_mae_prev=0.3, excluded=2),
    _ev(2024, "pit", "extended", "k_bb_pct", "all", 3.36214, n=250, mae_weighted=3.5,
        baseline_mae_league=4.08, baseline_mae_prev=4.32, excluded=90),
    _ev(2024, "bat", "extended", "hr", "all", 2.70004, n=1500, baseline_mae_prev=2.96),
]
# fmt: on


def _eval_row(
    target: int,
    role: str,
    model: str,
    metric: str,
    group: str,
    n: int,
    mae: float | None,
    mae_weighted: float | None,
    league: float | None,
    prev: float | None,
    excluded: int,
) -> dict[str, Any]:
    return {
        "target_season": target,
        "role": role,
        "model": model,
        "metric": metric,
        "group": group,
        "n": n,
        "mae": mae,
        "mae_weighted": mae_weighted,
        "baseline_mae_league": league,
        "baseline_mae_prev": prev,
        "excluded": excluded,
    }


# 반올림 4자리 (mae·mae_weighted·baseline_mae_*), NaN → null
EXPECTED_VALIDATION_ROWS = [
    # 80.12346 → 80.1235
    _eval_row(2024, "bat", "baseline", "pa", "seasons_1_2", 500, 80.1235, None, None, 80.1235, 0),
    # 2.70004 → 2.7
    _eval_row(2024, "bat", "extended", "hr", "all", 1500, 2.7, None, None, 2.96, 0),
    # .071149 → .0711, .076849 → .0768, .096911 → .0969
    _eval_row(2024, "bat", "weighted", "ops", "all", 410, 0.0711, None, 0.0768, 0.0969, 1100),
    _eval_row(2024, "bat", "weighted", "ops", "prev_absent", 3, 0.1, 0.12, 0.2, 0.3, 2),
    _eval_row(2024, "bat", "weighted", "ops", "seasons_3_5", 0, None, None, None, None, 7),
    # 3.36214 → 3.3621
    _eval_row(2024, "pit", "extended", "k_bb_pct", "all", 250, 3.3621, 3.5, 4.08, 4.32, 90),
    # .87412 → .8741, .91236 → .9124, 1.27449 → 1.2745
    _eval_row(2025, "pit", "weighted", "era", "all", 300, 0.8741, 0.9124, 0.95, 1.2745, 120),
]

FINAL_TEST: dict[str, Any] = {
    "target_season": 2026,
    "ran_at": "2026-12-01T03:00:00Z",
    "params_sha256": SHA,
    "reruns": [{"ran_at": "2026-12-02T03:00:00Z", "reason": "원본 재수집 뒤 다시 실행"}],
    "first_run": {"ran_at": "2026-12-01T03:00:00Z", "mae": {"ops": 0.0609, "era": 0.85}},
}


def test_export_build_web_validation_without_final_test() -> None:
    files = build_web(_world(evaluations=VALIDATION_EVAL_ROWS))
    assert files["validation.json"] == {
        "meta": META,
        "final_test": None,
        "rows": EXPECTED_VALIDATION_ROWS,
        "bands": [],
    }


def test_export_build_web_validation_with_final_test() -> None:
    # 최종 검증 뒤: final_test 객체는 그대로, 목표 2026 행(합성)은 정렬상 맨 뒤
    rows = [
        _ev(2026, "bat", "weighted", "ops", "all", 0.06549, n=400, excluded=1000),
        *VALIDATION_EVAL_ROWS,
    ]
    files = build_web(_world(evaluations=rows, final_test=FINAL_TEST))
    # .06549 → .0655
    row_2026 = _eval_row(2026, "bat", "weighted", "ops", "all", 400, 0.0655, None, None, None, 1000)
    assert files["validation.json"] == {
        "meta": META,
        "final_test": FINAL_TEST,
        "rows": [*EXPECTED_VALIDATION_ROWS, row_2026],
        "bands": [],
    }


# ---------------------------------------------------------------------------
# build_web — 오류
# ---------------------------------------------------------------------------


def _with_bat_projection(inp: ExportInput, extra: Mapping[str, Any]) -> ExportInput:
    rows = [*_bat_projection_rows(inp.model["bat"]), dict(extra)]
    projections = {**inp.projections, "bat": _frame(c.BAT_PROJECTIONS, rows)}
    return dataclasses.replace(inp, projections=projections)


_EXTRA_BAT = {
    "role": "bat",
    "target_season": TARGET,
    "model": "extended",
    "input_first": 2024,
    "input_last": 2026,
    "n_seasons": 2,
    "sample_pa": 300,
    "pa": 200.0,
    "hr": 5.0,
    "obp": 0.3,
    "slg": 0.4,
    "ops": 0.7,
    "lg_ops": 0.71849,
}


def test_export_build_web_rejects_player_outside_target_cohort() -> None:
    # 104는 목표 2025 대상 표에만 있다 (2027 대상 아님)
    inp = _with_bat_projection(_world(), {**_EXTRA_BAT, "player_id": 104})
    with pytest.raises(ValueError):
        build_web(inp)


def test_export_build_web_rejects_player_missing_from_players() -> None:
    inp = _with_bat_projection(_world(), {**_EXTRA_BAT, "player_id": 105})
    cohorts = _frame(
        c.COHORTS, [*COHORT_ROWS, _cohort_row(105, "bat", TARGET, (2024, 2026), 2, 300)]
    )
    inp = dataclasses.replace(inp, tables=dataclasses.replace(inp.tables, cohorts=cohorts))
    with pytest.raises(ValueError):
        build_web(inp)


def test_export_build_web_rejects_missing_league_row() -> None:
    # 리그 기준선은 Y−1(2026) 행. 없으면 다른 시즌 값으로 대신하지 않고 멈춘다
    league_pit = [r for r in LEAGUE_PIT_ROWS if r["season"] != TARGET - 1]
    with pytest.raises(ValueError, match="2026"):
        build_web(_world(league_pit_rows=league_pit))


def _bat_projections(inp: ExportInput, frame: pd.DataFrame) -> ExportInput:
    return dataclasses.replace(inp, projections={**inp.projections, "bat": frame})


def _bat_frame(**changes: Any) -> pd.DataFrame:
    rows = [{**r, **changes} for r in _bat_projection_rows("extended")]
    return _frame(c.BAT_PROJECTIONS, rows)


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        # 2026 예측을 2027로 공개하면 안 된다 (2026 규칙)
        (lambda: _bat_frame(target_season=2026), "target_season"),
        # 최종 모델은 extended인데 값은 weighted 행
        (lambda: _bat_frame(model="weighted"), "model"),
        (lambda: _bat_frame(role="pit"), "role"),
        # 같은 선수 두 행 (키 중복)
        (lambda: pd.concat([_bat_frame(), _bat_frame()], ignore_index=True), "중복"),
        # 계약 열이 빠졌다
        (lambda: _bat_frame().drop(columns=["lg_ops"]), "열 불일치"),
    ],
)
def test_export_build_web_rejects_projection_contract(
    frame: Callable[[], pd.DataFrame], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_web(_bat_projections(_world(), frame()))


def test_export_build_web_rejects_final_rows_without_record() -> None:
    # 2026 행은 최종 검증 기록(final_test)과 함께만 들어온다 (F10 한 번만, 2026 규칙)
    rows = [*GROUP_EVAL_ROWS, _ev(2026, "bat", "extended", "ops", "seasons_3_5", 0.08)]
    with pytest.raises(ValueError, match="final_test"):
        build_web(_world(evaluations=rows))


def test_export_build_web_rejects_evaluation_at_target() -> None:
    # 목표 Y(2027) 이후의 검증 행은 있을 수 없다 (실제값이 없다)
    rows = [*GROUP_EVAL_ROWS, _ev(TARGET, "bat", "extended", "ops", "all", 0.08)]
    with pytest.raises(ValueError, match=r"이후 시즌 행이 있다: \[2027\]$"):
        build_web(_world(evaluations=rows, final_test=FINAL_TEST))


def test_export_build_web_rejects_evaluation_contract() -> None:
    inp = _world()
    inp = dataclasses.replace(inp, evaluations=inp.evaluations.drop(columns=["excluded"]))
    with pytest.raises(ValueError, match="평가표"):
        build_web(inp)


# ---------------------------------------------------------------------------
# dump_json
# ---------------------------------------------------------------------------


def test_export_dump_json_exact_format() -> None:
    obj = {
        "b": 1,
        "a": [1.5, None, "한글 Zoë"],
        "ok": True,
        "neg": -0.25,
        "nested": {"z": 0, "y": "x"},
    }
    # 공백 없는 구분자, 한글 그대로(ensure_ascii False), 키 순서 유지, 끝에 줄바꿈
    expected = '{"b":1,"a":[1.5,null,"한글 Zoë"],"ok":true,"neg":-0.25,"nested":{"z":0,"y":"x"}}\n'
    assert dump_json(obj) == expected


def test_export_dump_json_deterministic() -> None:
    obj = json.loads((SAMPLES / "player_bat.json").read_text(encoding="utf-8"))
    text = dump_json(obj)
    assert text == dump_json(copy.deepcopy(obj))
    assert text == _reference_dump(obj)
    assert text.endswith("}\n")
    assert "\n" not in text[:-1]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_export_dump_json_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError):
        dump_json({"rows": [{"mae": bad}]})


# ---------------------------------------------------------------------------
# write_web
# ---------------------------------------------------------------------------


def _sample(name: str) -> Any:
    return json.loads((SAMPLES / name).read_text(encoding="utf-8"))


def _sample_bundle() -> dict[str, Any]:
    """기존 합성 샘플(tests/fixtures/synthetic/web)로 만든 묶음. 선수 1(타자)·2(투수)."""
    return {
        "validation.json": _sample("validation.json"),
        "players/2.json": _sample("player_pit.json"),
        "batters.json": _sample("batters.json"),
        "players/1.json": _sample("player_bat.json"),
        "pitchers.json": _sample("pitchers.json"),
    }


def test_export_write_web_accepts_synthetic_samples(tmp_path: Path) -> None:
    bundle = _sample_bundle()
    out_dir = tmp_path / "web"
    sizes = write_web(bundle, out_dir, _schema())
    # 반환은 경로 순 (pitchers.json < players/… < validation.json)
    assert list(sizes) == [
        "batters.json",
        "pitchers.json",
        "players/1.json",
        "players/2.json",
        "validation.json",
    ]
    for path, obj in bundle.items():
        written = (out_dir / path).read_bytes()
        assert written == _reference_dump(obj).encode("utf-8"), path
        assert sizes[path] == len(written), path
    assert _files_in(out_dir) == sorted(out_dir / p for p in bundle)


def test_export_write_web_build_web_roundtrip(tmp_path: Path) -> None:
    files = build_web(_world())
    out_dir = tmp_path / "web"
    sizes = write_web(files, out_dir, _schema())
    assert list(sizes) == sorted(files)
    for path, obj in files.items():
        text = (out_dir / path).read_text(encoding="utf-8")
        assert text == dump_json(obj), path
        assert json.loads(text) == obj, path
        assert sizes[path] == len(text.encode("utf-8")), path


@pytest.mark.parametrize(
    ("path", "breaks"),
    [
        ("batters.json", "name"),  # 빈 이름 (minLength 1)
        ("players/1.json", "extra"),  # 예측에 없는 키
    ],
)
def test_export_write_web_rejects_schema_violation(tmp_path: Path, path: str, breaks: str) -> None:
    bundle = _sample_bundle()
    if breaks == "name":
        bundle[path]["rows"][0]["player"]["name"] = ""
    else:
        bundle[path]["bat"]["projection"]["wrc_plus"] = 120
    out_dir = tmp_path / "web"
    stale = out_dir / "players" / "999.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=path):
        write_web(bundle, out_dir, _schema())
    # 아무것도 쓰지 않고, 지우지도 않는다
    assert _files_in(out_dir) == [stale]


def test_export_write_web_error_lists_first_violations(tmp_path: Path) -> None:
    # 위반 7건: 메시지에 파일 이름·스키마 정의·건수, 위반은 처음 5개만
    bundle = _sample_bundle()
    for i in range(7):
        bundle["batters.json"]["rows"][0]["player"][f"x{i}"] = i
    with pytest.raises(ValueError) as info:
        write_web(bundle, tmp_path / "web", _schema())
    message = str(info.value)
    assert message.startswith("batters.json: 스키마 BatTable 위반 7건")
    assert message.count("허용되지 않은 키") == 5
    assert _files_in(tmp_path / "web") == []


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_export_write_web_rejects_non_finite(tmp_path: Path, bad: float) -> None:
    # NaN·inf는 JSON에 들어가면 안 된다 — 스키마(number는 유한)에서 걸러 아무것도 쓰지 않는다
    bundle = _sample_bundle()
    bundle["pitchers.json"]["rows"][0]["projection"]["era"] = bad
    with pytest.raises(ValueError, match=r"^pitchers\.json: "):
        write_web(bundle, tmp_path / "web", _schema())
    assert _files_in(tmp_path / "web") == []


@pytest.mark.parametrize("unknown", ["notes.json", "sub/batters.json", "players/1.txt"])
def test_export_write_web_rejects_unknown_file_name(tmp_path: Path, unknown: str) -> None:
    bundle = _sample_bundle()
    bundle[unknown] = _sample("player_bat.json") if unknown.startswith("players/") else {"x": 1}
    out_dir = tmp_path / "web"
    with pytest.raises(ValueError):
        write_web(bundle, out_dir, _schema())
    assert _files_in(out_dir) == []


def _first_screen_limit() -> int:
    # 샘플 크기(직렬화 바이트): batters 864, pitchers 828 — 둘 중 큰 쪽
    bundle = _sample_bundle()
    return max(len(_reference_dump(bundle[p]).encode("utf-8")) for p in c.WEB_FIRST_SCREEN_FILES)


def test_export_write_web_first_screen_limit_inclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 한도는 호출 시점에 contracts에서 읽는다. 딱 한도 크기는 통과("넘으면" 거부)
    limit = _first_screen_limit()
    bundle = _sample_bundle()
    # 선수 상세·검증 파일은 한도보다 커도 된다 (첫 화면 파일만 검사, N6)
    assert len(_reference_dump(bundle["players/1.json"]).encode("utf-8")) > limit
    monkeypatch.setattr(c, "WEB_FIRST_SCREEN_MAX_BYTES", limit)
    sizes = write_web(bundle, tmp_path / "web", _schema())
    assert max(sizes[p] for p in c.WEB_FIRST_SCREEN_FILES) == limit


def test_export_write_web_rejects_oversized_first_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(c, "WEB_FIRST_SCREEN_MAX_BYTES", _first_screen_limit() - 1)
    out_dir = tmp_path / "web"
    with pytest.raises(ValueError):
        write_web(_sample_bundle(), out_dir, _schema())
    assert _files_in(out_dir) == []


def test_export_write_web_removes_stale_player_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "web"
    players = out_dir / "players"
    players.mkdir(parents=True)
    (players / "999.json").write_text('{"old":true}\n', encoding="utf-8")  # 빠진 선수 → 지운다
    (players / "1.json").write_text('{"old":true}\n', encoding="utf-8")  # 이번 묶음 → 덮어쓴다
    (players / "notes.txt").write_text("keep\n", encoding="utf-8")  # .json이 아니다 → 둔다
    (out_dir / "other.json").write_text("{}\n", encoding="utf-8")  # players/ 밖 → 둔다

    bundle = _sample_bundle()
    sizes = write_web(bundle, out_dir, _schema())

    assert not (players / "999.json").exists()
    assert (players / "notes.txt").read_text(encoding="utf-8") == "keep\n"
    assert (out_dir / "other.json").read_text(encoding="utf-8") == "{}\n"
    assert (players / "1.json").read_bytes() == _reference_dump(bundle["players/1.json"]).encode(
        "utf-8"
    )
    assert list(sizes) == sorted(bundle)


# ---------------------------------------------------------------------------
# write_projection_csv
# ---------------------------------------------------------------------------

# 예측 표 열 순서 + name, player_id 순, 실수는 pandas 기본 표기(반올림 없음)
EXPECTED_BAT_CSV = (
    "player_id,role,target_season,model,input_first,input_last,n_seasons,sample_pa,"
    "pa,hr,obp,slg,ops,lg_ops,name\n"
    "101,bat,2027,extended,2022,2026,5,2775,610.44,27.06,0.35123,0.47249,0.82372,0.71849,"
    "Alpha Catcher\n"
    "102,bat,2027,extended,2025,2026,2,750,402.96,16.04,0.32049,0.46151,0.782,0.71849,"
    "Beta Twoway\n"
)
EXPECTED_PIT_CSV = (
    "player_id,role,target_season,model,input_first,input_last,n_seasons,sample_bf,sample_outs,"
    "ip,era,k_bb_pct,lg_era,lg_k_bb_pct,name\n"
    "102,pit,2027,weighted,2025,2026,2,88,60,35.27,3.9849,15.06,4.1523,13.64,Beta Twoway\n"
    "103,pit,2027,weighted,2022,2025,3,1665,1171,120.04,4.0051,14.24,4.1523,13.64,"
    "Gamma Pitcher\n"
)


def test_export_write_projection_csv(tmp_path: Path) -> None:
    paths: dict[Role, Path] = {"bat": tmp_path / "bat.csv", "pit": tmp_path / "pit.csv"}
    result = write_projection_csv(_world(), paths)
    assert result == paths
    bat = paths["bat"].read_bytes()
    pit = paths["pit"].read_bytes()
    assert b"\r" not in bat and b"\r" not in pit
    assert bat.decode("utf-8") == EXPECTED_BAT_CSV
    assert pit.decode("utf-8") == EXPECTED_PIT_CSV


def test_export_write_projection_csv_creates_parent(tmp_path: Path) -> None:
    paths: dict[Role, Path] = {
        "bat": tmp_path / "out" / "a" / "bat.csv",
        "pit": tmp_path / "out" / "b" / "pit.csv",
    }
    write_projection_csv(_world(), paths)
    assert paths["bat"].read_text(encoding="utf-8") == EXPECTED_BAT_CSV
    assert paths["pit"].read_text(encoding="utf-8") == EXPECTED_PIT_CSV


def test_export_write_projection_csv_checks_every_role_first(tmp_path: Path) -> None:
    # 투수 예측 표에 players 표에 없는 105 → ValueError, 타자 CSV도 쓰지 않는다
    inp = _world()
    extra = {**_pit_projection_rows("weighted")[0], "player_id": 105}
    pit = _frame(c.PIT_PROJECTIONS, [*_pit_projection_rows("weighted"), extra])
    inp = dataclasses.replace(inp, projections={**inp.projections, "pit": pit})
    paths: dict[Role, Path] = {"bat": tmp_path / "bat.csv", "pit": tmp_path / "pit.csv"}
    with pytest.raises(ValueError, match="105"):
        write_projection_csv(inp, paths)
    assert _files_in(tmp_path) == []


def test_export_write_projection_csv_rejects_other_target(tmp_path: Path) -> None:
    # 2026 예측을 2027 검토용 CSV로 쓰면 안 된다
    inp = _bat_projections(_world(), _bat_frame(target_season=2026))
    paths: dict[Role, Path] = {"bat": tmp_path / "bat.csv", "pit": tmp_path / "pit.csv"}
    with pytest.raises(ValueError, match="target_season"):
        write_projection_csv(inp, paths)
    assert _files_in(tmp_path) == []


def test_export_write_projection_csv_columns(tmp_path: Path) -> None:
    paths: dict[Role, Path] = {"bat": tmp_path / "b.csv", "pit": tmp_path / "p.csv"}
    write_projection_csv(_world(), paths)
    specs: dict[Role, c.TableSpec] = {"bat": c.BAT_PROJECTIONS, "pit": c.PIT_PROJECTIONS}
    for role in c.ROLES:
        header = paths[role].read_text(encoding="utf-8").splitlines()[0]
        assert header.split(",") == [*specs[role].column_names, "name"]


# ---------------------------------------------------------------------------
# build_web — 과거 오차 80% 범위 (D29, 메인이 M4 통합 때 더함)
# ---------------------------------------------------------------------------


def _bands() -> pd.DataFrame:
    rows = [
        ("bat", "ops", "seasons_1_2", 267, 0.10712, "2024,2025,2026"),
        ("bat", "ops", "seasons_3_5", 734, 0.09694, "2024,2025,2026"),
        ("pit", "era", "seasons_1_2", 288, 1.4913, "2024,2025,2026"),
        ("pit", "era", "seasons_3_5", 656, 1.3638, "2024,2025,2026"),
        ("pit", "k_bb_pct", "seasons_1_2", 0, float("nan"), "2024,2025,2026"),
        ("pit", "k_bb_pct", "seasons_3_5", 656, 5.5381, "2024,2025,2026"),
    ]
    return pd.DataFrame(rows, columns=c.BANDS.column_names)


def test_export_build_web_bands() -> None:
    files = build_web(replace(_world(), bands=_bands()))
    # 표 머리: 그룹 → 지표 → ± (지표 키로 반올림: ops 3, era 2, k_bb_pct 1)
    assert files["batters.json"]["bands"] == {
        "seasons_1_2": {"ops": 0.107},
        "seasons_3_5": {"ops": 0.097},
    }
    assert files["pitchers.json"]["bands"] == {
        "seasons_1_2": {"era": 1.49, "k_bb_pct": None},  # n 0 → null
        "seasons_3_5": {"era": 1.36, "k_bb_pct": 5.5},
    }
    # 선수 상세: 그 선수 그룹의 범위 (101 타자 seasons_3_5, 102 투수 seasons_1_2)
    assert files["players/101.json"]["bat"]["group_error"]["band80"] == {"ops": 0.097}
    assert files["players/102.json"]["pit"]["group_error"]["band80"] == {
        "era": 1.49,
        "k_bb_pct": None,
    }
    # 검증 페이지: 표 순서 그대로, targets는 정수 목록
    assert files["validation.json"]["bands"][0] == {
        "role": "bat",
        "metric": "ops",
        "group": "seasons_1_2",
        "n": 267,
        "q80": 0.107,
        "targets": [2024, 2025, 2026],
    }
    assert len(files["validation.json"]["bands"]) == 6
    schema = _schema()
    for path, obj in files.items():
        assert validate(obj, _def_name(path), schema) == [], path


def test_export_build_web_rejects_bad_bands() -> None:
    with pytest.raises(ValueError, match="범위 표"):
        build_web(replace(_world(), bands=_bands().drop(columns=["targets"])))
