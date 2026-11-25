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

import json
import math
import re
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlbproj.contracts import (
    COHORTS,
    FINAL_TEST_PATH,
    FINAL_TEST_TARGET,
    FULL_TIME_UNITS,
    ROLES,
    TUNE_TARGETS,
    WAR_BAT_RUNS,
    WAR_EVALUATIONS,
    WAR_PROJECTIONS,
    WAR_RANKINGS,
    WAR_REGULAR_MIN,
    WAR_SEASONS,
    WAR_SOURCE,
    WAR_STAR_MIN,
    WAR_UNIT,
    WINDOW,
    Role,
    check_frame,
)
from mlbproj.export import _json_value
from mlbproj.final import (
    FinalTestError,
    GitRunner,
    _fits,
    _stamp_problem,
    _write_record,
    committed_head,
    first_run_path,
)
from mlbproj.models import war as war_model
from mlbproj.models.params import Params, load_params, params_sha256
from mlbproj.models.war import WarParams
from mlbproj.tables import Tables

_SHORT_CAREER_MAX = 2  # seasons_1_2 그룹의 n_seasons 상한 (evaluate·bands와 같다)
# 지표 → 그룹 (`WAR_GROUPS`를 지표별로 나눈 것, 모듈 docstring "검증")
_METRIC_GROUPS: dict[str, tuple[str, ...]] = {
    "war": ("all", "seasons_1_2", "seasons_3_5", "war2plus"),
    "war_full": ("regulars",),
}
_KIND_DTYPES = {"int": "int64", "float": "float64", "str": "str", "bool": "bool"}
_EVAL_KEYS = list(WAR_EVALUATIONS.keys)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECORD_KEYS = (
    "target_season",
    "ran_at",
    "params_sha256",
    "war_params_sha256",
    "commit",
    "reruns",
    "rows",
)
_RERUN_KEYS = ("ran_at", "reason", "commit")
_OUTS_PER_INNING = 3
_ROLES_OF: dict[str, tuple[Role, ...]] = {"bat": ("bat",), "pit": ("pit",), "both": ("bat", "pit")}
_PIT_SLOTS = frozenset({"sp", "rp"})
_RANKING_VALUES = (
    "war_bat",
    "war_pit",
    "war",
    "war_full_bat",
    "war_full_pit",
    "war_full",
    "prev_war",
)
_SHOW_MAX = 10  # 오류 메시지에 보여 줄 선수 수
_SAME_WAR_TOL = 1e-9  # 순위 표 값과 예측 표 값이 같은지 볼 때의 허용 오차 (복사·합 반올림만 허용)


class WarTestError(RuntimeError):
    """2026 WAR 시험을 돌릴 수 없을 때 (커밋 안 됨 · 값 불일치 · 이미 돌림 · 사유 문제)."""


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------


def actual_war(
    war_seasons: pd.DataFrame, seasons: pd.DataFrame, role: Role, season: int
) -> pd.DataFrame:
    """그 시즌 역할 실제 WAR과 출전. 열: player_id, war, units(`WAR_UNIT[role]`). 시즌 표에 행이 있는 선수만, player_id 순.

    출전은 시즌 표의 실제 값(타자 pa, 투수 outs)이다 — 2020 환산(pa_adj·outs_adj)은 쓰지 않는다.

    Raises:
        ValueError: 시즌 표의 역할 행에 맞는 WAR 행이 없을 때(정제 표 계약 위반, 어느 선수인지 적는다).
    """
    unit = WAR_UNIT[role]
    played = seasons.loc[seasons["season"] == season, ["player_id", unit]]
    mine = war_seasons.loc[
        (war_seasons["role"] == role) & (war_seasons["season"] == season), ["player_id", "war"]
    ]
    merged = played.merge(mine, on="player_id", how="left", validate="one_to_one")
    missing = merged["war"].isna()
    if bool(missing.any()):
        ids = sorted(int(pid) for pid in merged.loc[missing, "player_id"])
        raise ValueError(f"{season} {role} 시즌 행 {len(ids)}개에 WAR 행이 없다: {ids[:_SHOW_MAX]}")
    out = pd.DataFrame(
        {
            "player_id": merged["player_id"].astype("int64"),
            "war": merged["war"].astype("float64"),
            "units": merged[unit].astype("float64"),
        }
    )
    return out.sort_values("player_id", kind="mergesort").reset_index(drop=True)


def _evaluation_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """행 목록 → `WAR_EVALUATIONS` 표 (열 순서·종류 그대로, 결측 None → NaN)."""
    frame = pd.DataFrame(list(rows), columns=WAR_EVALUATIONS.column_names)
    return frame.astype({col.name: _KIND_DTYPES[col.kind] for col in WAR_EVALUATIONS.columns})


def _sorted_evaluations(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """평가표들을 이어 키 순(안정 정렬)으로. 비었으면 빈 `WAR_EVALUATIONS` 표."""
    if not frames:
        return _evaluation_frame([])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(_EVAL_KEYS, kind="mergesort").reset_index(drop=True)


def _require_projection(proj: pd.DataFrame, role: Role, target: int) -> None:
    """예측 표가 계약(`WAR_PROJECTIONS`)을 지키고 한 역할·한 목표·선수당 한 행인지 본다. 아니면 ValueError.

    계약 검사는 결측 예측(war·pt NaN)이 MAE를 조용히 NaN으로 만드는 것을 막는다.
    """
    problems = check_frame(proj, WAR_PROJECTIONS)
    if problems:
        raise ValueError(f"WAR 예측 표({role}, {target})가 계약을 어긴다: " + "; ".join(problems))
    for column, value in (("target_season", target), ("role", role)):
        wrong = sorted({str(v) for v in proj[column].tolist()} - {str(value)})
        if wrong:
            raise ValueError(
                f"WAR 예측 표의 {column}은 {value!r}만 있어야 하는데 {wrong[:_SHOW_MAX]}도 있다"
            )
    if bool(proj["player_id"].duplicated().any()):
        raise ValueError(f"WAR 예측 표({role}, {target})에 같은 선수가 두 번 있다")


def _mae_row(
    target: int,
    role: Role,
    metric: str,
    group: str,
    members: np.ndarray,
    pred: np.ndarray,
    actual: np.ndarray,
    prev: np.ndarray,
    league: np.ndarray,
) -> dict[str, Any]:
    """그룹 하나의 MAE 행. n = 0이면 세 MAE 모두 결측."""
    n = int(members.sum())

    def mae(values: np.ndarray) -> float | None:
        if n == 0:
            return None
        return float(np.abs(values[members] - actual[members]).mean())

    return {
        "target_season": int(target),
        "role": role,
        "metric": metric,
        "group": group,
        "n": n,
        "mae": mae(pred),
        "baseline_mae_prev": mae(prev),
        "baseline_mae_league": mae(league),
    }


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
        ValueError: proj의 목표·역할이 인자와 다르거나 proj가 계약을 어길 때, league_rate가 유한한 수가
            아닐 때, 시즌 표의 역할 행에 맞는 WAR 행이 없을 때(`actual_war`).
    """
    _require_projection(proj, role, target)
    if not math.isfinite(float(league_rate)):
        raise ValueError(
            f"리그 WAR 비율({role}, {target - 1})이 유한한 수가 아니다: {league_rate!r}"
        )
    now = actual_war(war_seasons, seasons, role, target).rename(
        columns={"war": "act_war", "units": "act_units"}
    )
    before = actual_war(war_seasons, seasons, role, target - 1).rename(
        columns={"war": "prev_war", "units": "prev_units"}
    )
    base = (
        proj[["player_id", "n_seasons", "pt", "war", "full_units", "war_full"]]
        .merge(now, on="player_id", how="left", validate="one_to_one")
        .merge(before, on="player_id", how="left", validate="one_to_one")
    )
    n_seasons = base["n_seasons"].to_numpy(dtype=np.int64)
    pt = base["pt"].to_numpy(dtype=float)
    full_units = base["full_units"].to_numpy(dtype=float)
    act_war = base["act_war"].fillna(0.0).to_numpy(dtype=float)  # 목표 시즌 미출전 = 0
    prev_war = base["prev_war"].fillna(0.0).to_numpy(dtype=float)  # 작년 미출전 = 0
    act_units = base["act_units"].fillna(0.0).to_numpy(dtype=float)
    prev_units = base["prev_units"].fillna(0.0).to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    # 예측 WAR — 대상 전원
    pred = base["war"].to_numpy(dtype=float)
    league = league_rate * pt
    members = {
        "all": np.ones(len(base), dtype=bool),
        "seasons_1_2": n_seasons <= _SHORT_CAREER_MAX,
        "seasons_3_5": n_seasons > _SHORT_CAREER_MAX,
        "war2plus": act_war >= WAR_STAR_MIN,
    }
    for group in _METRIC_GROUPS["war"]:
        rows.append(
            _mae_row(target, role, "war", group, members[group], pred, act_war, prev_war, league)
        )

    # 풀타임 환산 비율 — 목표 시즌 주전급이고 작년에도 출전한 선수
    regulars = (act_units >= WAR_REGULAR_MIN[role]) & (prev_units > 0)
    safe_act = np.where(regulars, act_units, 1.0)
    safe_prev = np.where(regulars, prev_units, 1.0)
    act_full = np.where(regulars, act_war / safe_act * full_units, 0.0)
    prev_full = np.where(regulars, prev_war / safe_prev * full_units, 0.0)
    pred_full = base["war_full"].to_numpy(dtype=float)
    league_full = league_rate * full_units
    for group in _METRIC_GROUPS["war_full"]:
        rows.append(
            _mae_row(
                target,
                role,
                "war_full",
                group,
                regulars,
                pred_full,
                act_full,
                prev_full,
                league_full,
            )
        )
    return _sorted_evaluations([_evaluation_frame(rows)])


# ---------------------------------------------------------------------------
# 과거 검증
# ---------------------------------------------------------------------------


def _evaluate_targets(
    tables: Tables,
    war_seasons: pd.DataFrame,
    params: Params,
    war_params: WarParams,
    targets: Sequence[int],
) -> pd.DataFrame:
    """targets × 두 역할: `models.war.war_projections` → 리그 비율(Y−1) → `evaluate_war`. 2026 가드 없음.

    Raises:
        ValueError: 합친 평가표가 `WAR_EVALUATIONS` 계약(열·종류·키 중복)을 어길 때.
    """
    frames: list[pd.DataFrame] = []
    for target in targets:
        for role in ROLES:
            seasons = tables.seasons[role]
            proj = war_model.war_projections(tables, war_seasons, params, war_params, role, target)
            league_rate = war_model.league_war_rate(war_seasons, seasons, role, target - 1)
            frames.append(evaluate_war(proj, war_seasons, seasons, role, target, league_rate))
    out = _sorted_evaluations(frames)
    problems = check_frame(out, WAR_EVALUATIONS)
    if problems:
        raise ValueError("WAR 평가표가 계약을 어긴다: " + "; ".join(problems))
    return out


def backtest_war(
    tables: Tables,
    war_seasons: pd.DataFrame,
    params: Params,
    war_params: WarParams,
    targets: Iterable[int] = TUNE_TARGETS,
) -> pd.DataFrame:
    """과거 검증표 (`WAR_EVALUATIONS`, 두 역할 × targets). 2026 이상은 읽기 전에 ValueError.

    같은 목표를 두 번 주면 키가 겹치므로 이것도 읽기 전에 ValueError다.
    """
    chosen = tuple(int(t) for t in targets)
    late = sorted({t for t in chosen if t >= FINAL_TEST_TARGET})
    if late:
        raise ValueError(
            f"목표 {late}는 과거 검증에 넣지 않는다 — {FINAL_TEST_TARGET} WAR 시험은 "
            "final-test-war로 한 번만 돌린다 (SPEC D36)"
        )
    repeated = sorted({t for t in chosen if chosen.count(t) > 1})
    if repeated:
        raise ValueError(f"과거 검증 목표 {repeated}가 두 번 이상 있다")
    return _evaluate_targets(tables, war_seasons, params, war_params, chosen)


# ---------------------------------------------------------------------------
# 2026 WAR 시험
# ---------------------------------------------------------------------------


def _json_rows(evaluations: pd.DataFrame) -> list[dict[str, Any]]:
    """평가표 → JSON 행 (`WAR_EVALUATIONS` 열 순, NaN → None, 정수는 파이썬 int)."""
    out: list[dict[str, Any]] = []
    for record in evaluations[WAR_EVALUATIONS.column_names].to_dict(orient="records"):
        row: dict[str, Any] = {}
        for col in WAR_EVALUATIONS.columns:
            value = record[col.name]
            if col.kind == "int":
                row[col.name] = int(value)
            elif col.kind == "float":
                row[col.name] = None if pd.isna(value) else float(value)
            else:
                row[col.name] = str(value)
        out.append(row)
    return out


def _record_problem(record: object) -> str | None:
    """2026 WAR 시험 기록의 형식 문제를 한국어로. 없으면 None (모듈 docstring 4의 모양)."""
    if not isinstance(record, dict) or set(record) != set(_RECORD_KEYS):
        keys = sorted(record) if isinstance(record, dict) else type(record).__name__
        return f"최상위 키가 {sorted(_RECORD_KEYS)}가 아니다: {keys}"
    if type(record["target_season"]) is not int or record["target_season"] != FINAL_TEST_TARGET:
        return f"target_season이 {FINAL_TEST_TARGET}이 아니다: {record['target_season']!r}"
    stamp = _stamp_problem(record["ran_at"])
    if stamp is not None:
        return f"ran_at — {stamp}"
    for key in ("params_sha256", "war_params_sha256"):
        value = record[key]
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            return f"{key}가 sha256(16진수 64자)이 아니다: {value!r}"
    if not isinstance(record["commit"], str) or not record["commit"].strip():
        return f"commit이 비었다: {record['commit']!r}"
    if not isinstance(record["reruns"], list):
        return "reruns가 목록이 아니다"
    for i, rerun in enumerate(record["reruns"]):
        if not isinstance(rerun, dict) or set(rerun) != set(_RERUN_KEYS):
            return f"reruns[{i}]의 키가 {list(_RERUN_KEYS)}가 아니다"
        empty = [k for k in _RERUN_KEYS if not isinstance(rerun[k], str) or not rerun[k].strip()]
        if empty:
            return f"reruns[{i}]의 {empty}가 비었거나 문자열이 아니다"
        stamp = _stamp_problem(rerun["ran_at"])
        if stamp is not None:
            return f"reruns[{i}].ran_at — {stamp}"
    if not isinstance(record["rows"], list):
        return "rows가 목록이 아니다"
    names = set(WAR_EVALUATIONS.column_names)
    for i, row in enumerate(record["rows"]):
        if not isinstance(row, dict) or set(row) != names:
            return f"rows[{i}]의 키가 {WAR_EVALUATIONS.name} 열과 다르다"
        bad = [col.name for col in WAR_EVALUATIONS.columns if not _fits(row[col.name], col)]
        if bad:
            return f"rows[{i}]의 값 종류가 계약과 다르다: {bad}"
        if row["target_season"] != FINAL_TEST_TARGET:
            return f"rows[{i}]의 목표 시즌이 {FINAL_TEST_TARGET}이 아니다: {row['target_season']}"
    return None


def _read_record(path: Path) -> dict[str, Any]:
    """기존 2026 WAR 시험 기록을 읽는다.

    Raises:
        WarTestError: 읽을 수 없거나 형식이 틀릴 때.
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WarTestError(f"2026 WAR 시험 기록을 읽지 못했다: {path} ({exc})") from exc
    problem = _record_problem(record)
    if problem is not None:
        raise WarTestError(f"2026 WAR 시험 기록의 형식이 틀렸다 ({problem}): {path}")
    out: dict[str, Any] = record
    return out


def _require_frozen_values(
    params: Params, war_params: WarParams, params_path: Path, war_params_path: Path
) -> tuple[str, str]:
    """넘긴 파라미터·WAR 파라미터가 커밋된 파일의 값과 같은지 확인하고 두 sha256을 돌려준다.

    Raises:
        WarTestError: 파일을 읽을 수 없거나 값이 다를 때.
    """
    given = (params_sha256(params), war_model.war_params_sha256(war_params))
    try:
        frozen = (
            params_sha256(load_params(params_path)),
            war_model.war_params_sha256(war_model.load_war_params(war_params_path)),
        )
    except (OSError, ValueError) as exc:
        raise WarTestError(f"커밋된 파라미터·WAR 파라미터를 읽지 못했다: {exc}") from exc
    for what, path, mine, theirs in (
        ("파라미터", params_path, given[0], frozen[0]),
        ("WAR 파라미터", war_params_path, given[1], frozen[1]),
    ):
        if mine != theirs:
            raise WarTestError(
                f"넘긴 {what}가 커밋된 파일과 다르다(sha256 {mine[:12]} ≠ {theirs[:12]}) — "
                f"2026 WAR 시험은 커밋된 파일 그대로 돈다: {path}"
            )
    return given


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

    out_path가 기존 여섯 지표 기록(`contracts.FINAL_TEST_PATH`와 그 첫 실행 보관본)의 자리면
    읽기 전에 거부한다 — 그 기록은 읽지도 쓰지도 않는다.

    Raises:
        WarTestError: 위 거부 조건.
        ValueError: now가 "YYYY-MM-DDTHH:MM:SSZ"가 아닐 때, 평가 결과가 기록 형식을 어길 때.
    """
    stamp = _stamp_problem(now)
    if stamp is not None:
        raise ValueError(f"실행 시각(now) — {stamp}")
    six_metrics = repo / FINAL_TEST_PATH
    if out_path.resolve() in {six_metrics.resolve(), first_run_path(six_metrics).resolve()}:
        raise WarTestError(
            f"기록 경로가 기존 여섯 지표의 2026 기록 자리다: {out_path} — "
            "WAR 시험 기록은 따로 둔다(`FINAL_TEST_WAR_PATH`, SPEC D36)"
        )
    try:
        head = committed_head([params_path, war_params_path], repo, git)
    except FinalTestError as exc:
        raise WarTestError(str(exc)) from exc
    sha_params, sha_war = _require_frozen_values(params, war_params, params_path, war_params_path)
    reason = (rerun_reason or "").strip()
    previous: dict[str, Any] | None = None
    if out_path.exists():
        if not reason:
            raise WarTestError(
                f"이미 한 번 돌렸다({out_path}) — {FINAL_TEST_TARGET} WAR 시험은 한 번만이다. "
                "꼭 다시 돌려야 하면 --rerun 사유를 준다"
            )
        previous = _read_record(out_path)
        for what, key, value in (
            ("파라미터", "params_sha256", sha_params),
            ("WAR 파라미터", "war_params_sha256", sha_war),
        ):
            if previous[key] != value:
                raise WarTestError(
                    f"첫 실행과 {what}가 다르다({key} {previous[key][:12]} → {value[:12]}) — "
                    f"{FINAL_TEST_TARGET} 결과를 본 뒤 바꾼 모델로 다시 재면 시험이 아니다 (SPEC D36)"
                )
    elif reason:
        raise WarTestError(
            f"다시 돌린다는 사유(--rerun)를 줬는데 기록이 없다: {out_path} — "
            "첫 실행이면 --rerun 없이 돌리고, 다시 돌리는 것이면 기록 경로(--out)를 확인한다"
        )

    evaluations = _evaluate_targets(tables, war_seasons, params, war_params, [FINAL_TEST_TARGET])
    rows = _json_rows(evaluations)
    if previous is None:
        ran_at, commit, reruns = now, head, []
    else:
        ran_at, commit = previous["ran_at"], previous["commit"]
        reruns = [*previous["reruns"], {"ran_at": now, "reason": reason, "commit": head}]
    record: dict[str, Any] = {
        "target_season": FINAL_TEST_TARGET,
        "ran_at": ran_at,
        "params_sha256": sha_params,
        "war_params_sha256": sha_war,
        "commit": commit,
        "reruns": reruns,
        "rows": rows,
    }
    # 다음 --rerun이 읽을 수 있는 기록만 쓴다 (예: 목표가 2026이 아닌 평가 행)
    problem = _record_problem(record)
    if problem is not None:
        raise ValueError(f"만든 2026 WAR 시험 기록이 형식을 어긴다 — 쓰지 않는다: {problem}")
    _write_record(record, out_path)
    return record


# ---------------------------------------------------------------------------
# 웹 war.json · 검토용 CSV
# ---------------------------------------------------------------------------


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


def _checked_rankings(inp: WarExportInput) -> tuple[int, pd.DataFrame]:
    """목표 시즌 Y(= meta target_season)와 계약·목표를 확인한 순위 표.

    Raises:
        ValueError: 순위 표가 `WAR_RANKINGS` 계약을 어기거나 목표 시즌이 Y가 아닐 때.
    """
    target = int(inp.meta["target_season"])
    rankings = inp.rankings
    problems = check_frame(rankings, WAR_RANKINGS)
    if problems:
        raise ValueError("WAR 순위 표가 계약을 어긴다: " + "; ".join(problems))
    wrong = sorted({int(t) for t in rankings["target_season"]} - {target})
    if wrong:
        raise ValueError(f"WAR 순위 표의 목표 시즌은 {target}만 있어야 하는데 {wrong}도 있다")
    return target, rankings


def _records_by_player(
    frame: pd.DataFrame, spec_name: str, role: Role, target: int
) -> dict[int, Mapping[Hashable, Any]]:
    """한 역할·목표 Y 표(예측·대상) → player_id → 행. 역할·목표가 다르거나 선수가 겹치면 ValueError."""
    for column, value in (("role", role), ("target_season", target)):
        wrong = sorted({str(v) for v in frame[column].tolist()} - {str(value)})
        if wrong:
            raise ValueError(
                f"{role} {spec_name}의 {column}은 {value!r}만 있어야 하는데 {wrong[:_SHOW_MAX]}도 있다"
            )
    out: dict[int, Mapping[Hashable, Any]] = {}
    for record in frame.to_dict(orient="records"):
        pid = int(record["player_id"])
        if pid in out:
            raise ValueError(f"{role} {spec_name}에 선수 {pid}가 두 번 있다")
        out[pid] = record
    return out


def _war_history(war_seasons: pd.DataFrame, seasons: range) -> dict[int, list[dict[str, Any]]]:
    """선수 → 입력 기간 시즌별 WAR(F22) — {season, war_bat, war_pit, war_pit_ra9}, 시즌 오름차순.

    war_pit은 FIP 기반(예측에 쓴다), war_pit_ra9는 실점 기반(참고, D40)이다.

    Raises:
        ValueError: 같은 선수·시즌·역할 행이 두 번 있거나 역할이 bat·pit이 아닐 때.
    """
    mine = war_seasons[war_seasons["season"].isin(list(seasons))]
    found: dict[int, dict[int, dict[str, Mapping[Hashable, Any]]]] = {}
    for record in mine[["player_id", "season", "role", "war", "war_ra9"]].to_dict(orient="records"):
        pid, season, role = int(record["player_id"]), int(record["season"]), str(record["role"])
        if role not in ROLES:
            raise ValueError(f"WAR 시즌 표에 모르는 역할 {role!r}이 있다 (선수 {pid}, {season})")
        slot = found.setdefault(pid, {}).setdefault(season, {})
        if role in slot:
            raise ValueError(f"WAR 시즌 표에 선수 {pid}의 {season} {role} 행이 두 번 있다")
        slot[role] = record

    def value(by_role: dict[str, Mapping[Hashable, Any]], role: Role, column: str, key: str) -> Any:
        record = by_role.get(role)
        if record is None or pd.isna(record[column]):
            return None
        return _json_value(float(record[column]), key)

    return {
        pid: [
            {
                "season": season,
                "war_bat": value(by_role, "bat", "war", "war_bat"),
                "war_pit": value(by_role, "pit", "war", "war_pit"),
                "war_pit_ra9": value(by_role, "pit", "war_ra9", "war_pit_ra9"),
            }
            for season, by_role in sorted(by_season.items())
        ]
        for pid, by_season in found.items()
    }


def _prev_bat_runs(
    war_seasons: pd.DataFrame, season: int
) -> tuple[dict[int, dict[str, Any]], float]:
    """Y−1 타자 WAR 구성 항목(점, F22·E29)과 그 시즌 리그 점/승 비율(Σ구성 ÷ ΣWAR).

    비율은 순위에 없는 선수까지 그 시즌 타자 행 전체로 낸다. 선수마다 구성 합이 WAR에 정확히
    비례하지 않으므로(E29) 화면은 이 리그 비율로만 점 → 승을 설명한다.

    Raises:
        ValueError: 그 시즌 타자 행에 구성 항목이 비었거나, 타자 행이 없거나 ΣWAR이 0 이하일 때.
    """
    columns = [column for _, column in WAR_BAT_RUNS]
    rows = war_seasons[(war_seasons["season"] == season) & (war_seasons["role"] == "bat")]
    holes = sorted({int(pid) for pid in rows.loc[rows[columns].isna().any(axis=1), "player_id"]})
    if holes:
        raise ValueError(f"{season} 타자 WAR 구성 항목이 빈 선수가 있다: {holes[:_SHOW_MAX]}")
    total_war = float(rows["war"].sum())
    if rows.empty or not total_war > 0:
        raise ValueError(f"{season} 타자 ΣWAR이 {total_war!r}라 점/승 비율을 낼 수 없다")
    runs_per_win = float(rows[columns].to_numpy().sum()) / total_war
    by_player = {
        int(record["player_id"]): {
            key: _json_value(float(record[column]), key) for key, column in WAR_BAT_RUNS
        }
        for record in rows.to_dict(orient="records")
    }
    return by_player, runs_per_win


def _final_test_summary(inp: WarExportInput) -> dict[str, Any] | None:
    """2026 WAR 시험 기록 → 스키마 WarFinalTest 모양 (commit·params_sha256·rows는 내보내지 않는다).

    WAR 예측은 기존 파라미터(가중치·출전량)에도 기대므로, 기록의 params_sha256도 meta의 것과 같아야 한다.

    Raises:
        ValueError: 기록 형식이 틀리거나, 기록의 WAR 파라미터·파라미터가 지금 내보내는 것과 다를 때.
    """
    record = inp.final_test
    if record is None:
        return None
    problem = _record_problem(record)
    if problem is not None:
        raise ValueError(f"2026 WAR 시험 기록의 형식이 틀렸다: {problem}")
    for key, value in (
        ("war_params_sha256", inp.war_params_sha256),
        ("params_sha256", inp.meta.get("params_sha256")),
    ):
        if record[key] != value:
            raise ValueError(
                f"2026 WAR 시험 기록의 {key}가 지금 내보내는 값과 다르다 — "
                f"2026을 잰 모델과 2027 예측 모델이 섞인다 ({str(record[key])[:12]} ≠ "
                f"{str(value)[:12]})"
            )
    return {
        "target_season": record["target_season"],
        "ran_at": record["ran_at"],
        "war_params_sha256": record["war_params_sha256"],
        "reruns": [
            {"ran_at": rerun["ran_at"], "reason": rerun["reason"]} for rerun in record["reruns"]
        ],
    }


def _eval_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """평가 JSON 행의 키 (`WAR_EVALUATIONS.keys` 순)."""
    return tuple(row[key] for key in _EVAL_KEYS)


def _web_evaluations(
    evaluations: pd.DataFrame, target: int, final_test: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    """평가표 → WarEvalRow 목록 (키 순, MAE는 `WEB_ROUND`).

    2026 행(`FINAL_TEST_TARGET`)은 2026 WAR 시험 기록에서만 온다(D36 "한 번만", `export`의 여섯 지표와
    같은 규칙). 기록이 없으면 2026 행이 있으면 안 되고, 있으면 2026 행이 기록의 rows와 같아야 한다.
    final_test는 `_final_test_summary`로 형식을 확인한 기록이다.

    Raises:
        ValueError: 계약 위반, 목표 Y 이후 시즌 행, 기록 없이 2026 행이 있을 때, 2026 행이 기록과 다를 때.
    """
    problems = check_frame(evaluations, WAR_EVALUATIONS)
    if problems:
        raise ValueError("WAR 평가표가 계약을 어긴다: " + "; ".join(problems))
    late = sorted({int(t) for t in evaluations["target_season"] if int(t) >= target})
    if late:
        raise ValueError(f"WAR 평가표에 목표 {target} 이후 시즌 행이 있다: {late}")
    ordered = evaluations.sort_values(_EVAL_KEYS, kind="mergesort")
    tested = _json_rows(ordered[ordered["target_season"] == FINAL_TEST_TARGET])
    if final_test is None:
        if tested:
            raise ValueError(
                f"WAR 평가표에 {FINAL_TEST_TARGET} 행 {len(tested)}개가 있는데 2026 WAR 시험 기록"
                f"(final_test)이 없다 — {FINAL_TEST_TARGET}은 final-test-war 기록으로만 공개한다 (SPEC D36)"
            )
    elif tested != sorted(final_test["rows"], key=_eval_key):
        raise ValueError(
            f"WAR 평가표의 {FINAL_TEST_TARGET} 행({len(tested)}개)이 2026 WAR 시험 기록의 rows"
            f"({len(final_test['rows'])}개)와 다르다 — {FINAL_TEST_TARGET} 결과는 기록 그대로 공개한다"
        )
    names = WAR_EVALUATIONS.column_names
    return [
        {name: _json_value(record[name], name) for name in names}
        for record in ordered[names].to_dict(orient="records")
    ]


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=_SAME_WAR_TOL)


def _require_same_war(
    pid: int,
    ranking: Mapping[Hashable, Any],
    proj: Mapping[Role, Mapping[int, Mapping[Hashable, Any]]],
) -> None:
    """순위 행의 역할별 WAR·풀타임 값이 예측 표와 같고, 합이 역할 값의 합인지 본다 (D34).

    한 행의 pa·ip(예측 표)와 war(순위 표)가 서로 다른 실행에서 오면 화면이 조용히 틀린다.

    Raises:
        ValueError: 예측이 있는 역할의 값이 다르거나 결측, 예측이 없는 역할의 값이 있거나, 합이 다를 때.
    """
    for total_key, proj_key in (("war", "war"), ("war_full", "war_full")):
        total = 0.0
        for role in ROLES:
            key = f"{total_key}_{role}"
            value = ranking[key]
            mine = proj[role].get(pid)
            if mine is None:
                if not pd.isna(value):
                    raise ValueError(
                        f"선수 {pid}의 {role} 예측이 없는데 순위 표 {key}가 {value!r}다"
                    )
                continue
            if pd.isna(value) or not _same(float(value), float(mine[proj_key])):
                raise ValueError(
                    f"선수 {pid}의 순위 표 {key}({value!r})가 {role} WAR 예측 표 {proj_key}"
                    f"({mine[proj_key]!r})와 다르다"
                )
            total += float(value)
        if pd.isna(ranking[total_key]) or not _same(float(ranking[total_key]), total):
            raise ValueError(
                f"선수 {pid}의 순위 표 {total_key}({ranking[total_key]!r})가 역할 값의 합({total!r})과 다르다"
            )


def build_war_web(inp: WarExportInput) -> dict[str, Any]:
    """war.json 내용(WarFile). rows는 rankings 순서 그대로.

    - pa = 타자 pt, ip = 투수 pt / 3(소수 이닝), slot = 투수 slot(투수 역할이 없으면 null).
    - group = 두 역할 n_seasons 중 큰 값으로 seasons_1_2(≤2) / seasons_3_5. played_prev = 어느 역할이든 Y−1 기록.
    - rows는 Y−1에 MLB 기록이 있는 선수(played_prev)만 싣는다(D42). 싣지 않는 선수도 계약 검사는 받는다.
    - history = 입력 기간(2022~2026) 시즌 중 그 선수 WAR 행이 있는 시즌, 오름차순
      {season, war_bat, war_pit, war_pit_ra9}(D40).
    - prev_bat_runs = Y−1 타자 구성 항목(점), runs_per_win = Y−1 리그 Σ구성 ÷ ΣWAR(E29).
    - final_test = 기록이 있으면 {"target_season", "ran_at", "war_params_sha256", "reruns": [{"ran_at", "reason"}]}.
    - full_time = {"pa": 600, "sp_ip": 180, "rp_ip": 65}, source = `WAR_SOURCE`.

    Raises:
        ValueError: 표 계약 위반, 목표 시즌 불일치, 순위의 역할·WAR 값과 예측·대상 표가 맞지 않을 때,
            선수 요약이 없을 때, 2026 시험 기록이 틀리거나 WAR 파라미터·파라미터가 다를 때,
            2026 평가 행이 시험 기록과 맞지 않을 때(기록 없이 2026 행이 있는 것 포함).
    """
    target, rankings = _checked_rankings(inp)
    # 기록 형식을 먼저 확인한다 (_web_evaluations가 기록의 rows를 읽는다)
    final_test = _final_test_summary(inp)
    evaluations = _web_evaluations(inp.evaluations, target, inp.final_test)
    problems = check_frame(inp.war_seasons, WAR_SEASONS)
    if problems:
        raise ValueError("WAR 시즌 표가 계약을 어긴다: " + "; ".join(problems))
    if set(inp.projections) != set(ROLES) or set(inp.cohorts) != set(ROLES):
        raise ValueError(f"예측·대상 표는 역할 {list(ROLES)}마다 하나다")
    proj: dict[Role, dict[int, Mapping[Hashable, Any]]] = {}
    cohort: dict[Role, dict[int, Mapping[Hashable, Any]]] = {}
    for role in ROLES:
        problems = check_frame(inp.projections[role], WAR_PROJECTIONS)
        problems += check_frame(inp.cohorts[role], COHORTS)
        if problems:
            raise ValueError(f"{role} WAR 예측·대상 표가 계약을 어긴다: " + "; ".join(problems))
        proj[role] = _records_by_player(inp.projections[role], "WAR 예측 표", role, target)
        cohort[role] = _records_by_player(inp.cohorts[role], "대상 표", role, target)
    ranked = [int(pid) for pid in rankings["player_id"]]
    unranked = sorted(set().union(*(set(proj[role]) for role in ROLES)) - set(ranked))
    if unranked:
        raise ValueError(f"WAR 예측 표의 선수가 순위 표에 없다: {unranked[:_SHOW_MAX]}")
    history = _war_history(inp.war_seasons, range(target - WINDOW, target))
    bat_runs, runs_per_win = _prev_bat_runs(inp.war_seasons, target - 1)

    rows: list[dict[str, Any]] = []
    for record in rankings.to_dict(orient="records"):
        pid = int(record["player_id"])
        roles = _ROLES_OF.get(str(record["roles"]))
        if roles is None:
            raise ValueError(f"선수 {pid}의 roles가 bat·pit·both가 아니다: {record['roles']!r}")
        projected = tuple(role for role in ROLES if pid in proj[role])
        if projected != roles:
            raise ValueError(
                f"선수 {pid}의 roles {record['roles']!r}와 WAR 예측 표의 역할 {list(projected)}이 다르다"
            )
        missing = [role for role in roles if pid not in cohort[role]]
        if missing:
            raise ValueError(f"선수 {pid}의 대상 표 행이 없다: {missing}")
        summary = inp.player_summaries.get(pid)
        if summary is None or summary.get("player_id") != pid:
            raise ValueError(f"선수 {pid}의 요약(PlayerSummary)이 없다")
        _require_same_war(pid, record, proj)
        bat = proj["bat"].get(pid)
        pit = proj["pit"].get(pid)
        slot: str | None = None
        if pit is not None:
            slot = str(pit["slot"])
            if slot not in _PIT_SLOTS:
                raise ValueError(f"선수 {pid}의 투수 자리가 sp·rp가 아니다: {slot!r}")
        n_seasons = max(int(cohort[role][pid]["n_seasons"]) for role in roles)
        if not any(bool(cohort[role][pid]["played_prev"]) for role in roles):
            continue  # D42 — 2026에 뛰지 않은 선수는 싣지 않는다
        rows.append(
            {
                "player": dict(summary),
                "roles": str(record["roles"]),
                "played_prev": True,
                "group": "seasons_1_2" if n_seasons <= _SHORT_CAREER_MAX else "seasons_3_5",
                "slot": slot,
                "pa": None if bat is None else _json_value(float(bat["pt"]), "pa"),
                "ip": (
                    None if pit is None else _json_value(float(pit["pt"]) / _OUTS_PER_INNING, "ip")
                ),
                **{key: _json_value(record[key], key) for key in _RANKING_VALUES},
                "prev_bat_runs": bat_runs.get(pid) if "bat" in roles else None,
                "history": history.get(pid, []),
            }
        )

    return {
        "meta": dict(inp.meta),
        "source": WAR_SOURCE,
        "war_params_sha256": inp.war_params_sha256,
        "full_time": {
            "pa": round(FULL_TIME_UNITS["bat"]),
            "sp_ip": round(FULL_TIME_UNITS["sp"] / _OUTS_PER_INNING),
            "rp_ip": round(FULL_TIME_UNITS["rp"] / _OUTS_PER_INNING),
        },
        "runs_per_win": {
            "season": target - 1,
            "value": _json_value(runs_per_win, "runs_per_win"),
        },
        "rows": rows,
        "evaluations": evaluations,
        "final_test": final_test,
    }


def write_war_csv(inp: WarExportInput, path: Path) -> None:
    """검토용 CSV(`WAR_PROJECTION_CSV`): rankings 열 + name, 같은 입력이면 같은 바이트.

    Raises:
        ValueError: 순위 표 계약·목표 시즌이 틀리거나, 순위의 선수가 players 표에 없을 때.
    """
    _, rankings = _checked_rankings(inp)
    names = inp.players[["player_id", "name"]]
    missing = sorted(set(rankings["player_id"].tolist()) - set(names["player_id"].tolist()))
    if missing:
        raise ValueError(
            f"WAR 순위의 선수 {len(missing)}명이 players 표에 없다: {missing[:_SHOW_MAX]}"
        )
    frame = rankings[WAR_RANKINGS.column_names].merge(
        names, on="player_id", how="left", validate="one_to_one"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
