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

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlbproj.contracts import (
    BAT_PLAYING_TIME,
    BATTING_SEASONS,
    COHORTS,
    FINAL_TEST_TARGET,
    FULL_TIME_UNITS,
    PIT_PLAYING_TIME,
    PITCHING_SEASONS,
    ROLES,
    STARTER_SHARE_MIN,
    TUNE_TARGETS,
    WAR_PROJECTIONS,
    WAR_RANKINGS,
    WAR_SEASONS,
    WAR_TUNE_GRID,
    WAR_UNIT,
    WINDOW,
    Role,
    TableSpec,
    check_frame,
)
from mlbproj.models.params import Params
from mlbproj.models.playing_time import predict_playing_time
from mlbproj.tables import Tables, cohort_for


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

# 역할별 정제 시즌 표·출전량 예측 표 계약
_SEASON_SPECS: dict[Role, TableSpec] = {"bat": BATTING_SEASONS, "pit": PITCHING_SEASONS}
_PT_SPECS: dict[Role, TableSpec] = {"bat": BAT_PLAYING_TIME, "pit": PIT_PLAYING_TIME}
# 투수 자리 (D35) — 타자는 늘 "bat"
_BAT_SLOT = "bat"
_PIT_SLOTS: frozenset[str] = frozenset({"sp", "rp"})
# config/war_v1.json 최상위 키
_PARAM_KEYS: frozenset[str] = frozenset({"regress", "version"})
# 조정 보고 표의 열 (D36)
_TUNE_COLUMNS: tuple[str, ...] = ("role", "regress", "target_season", "mae", "mean_mae", "chosen")
# 오류 메시지에 적는 선수 ID의 최대 개수
_SHOW_MAX = 5


# ---------------------------------------------------------------------------
# WarParams JSON (config/war_v1.json)
# ---------------------------------------------------------------------------


def _positive(value: Any, name: str) -> float:
    """양의 유한수(bool 아님)를 float로. 아니면 ValueError."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name}는 숫자여야 한다: {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name}는 양의 유한수여야 한다: {value!r}")
    return number


def _checked_regress(regress: Any, name: str = "regress") -> dict[Role, float]:
    """회귀 표본 사전을 확인해 역할 순서의 float 사전으로 돌려준다. 어기면 ValueError."""
    if not isinstance(regress, Mapping):
        raise ValueError(f"{name}는 역할 → R 객체여야 한다: {regress!r}")
    if set(regress) != set(ROLES):
        raise ValueError(f"{name} 역할은 {list(ROLES)}여야 한다: {sorted(map(str, regress))}")
    return {role: _positive(regress[role], f"{name}.{role}") for role in ROLES}


def _checked_war_params(p: WarParams) -> WarParams:
    """형식을 확인하고 값을 float로 맞춘 WarParams. 어기면 ValueError."""
    if not isinstance(p.version, str) or not p.version:
        raise ValueError(f"version은 비지 않은 문자열이어야 한다: {p.version!r}")
    return WarParams(version=p.version, regress=_checked_regress(p.regress))


def war_params_to_json(p: WarParams) -> str:
    """정규화된 JSON 문자열 (키 정렬·indent=2·끝 줄바꿈). 같은 값이면 같은 문자열 — sha256의 기준."""
    checked = _checked_war_params(p)
    doc = {"regress": dict(checked.regress), "version": checked.version}
    return json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """JSON 객체를 사전으로. 같은 키가 두 번이면 ValueError (json 기본은 마지막 값을 조용히 쓴다)."""
    doc: dict[str, Any] = {}
    for key, value in pairs:
        if key in doc:
            raise ValueError(f"WAR 파라미터 JSON에 같은 키가 두 번 있다: {key!r}")
        doc[key] = value
    return doc


def war_params_from_json(text: str) -> WarParams:
    """`war_params_to_json`의 역.

    Raises:
        ValueError: 키가 빠졌거나 남거나, 역할이 bat·pit 둘이 아니거나, R이 양의 유한수가 아닐 때.
    """
    # JSON 형식 오류(JSONDecodeError)도 ValueError다. 같은 키 두 번(어느 R인지 모호)도 막는다
    doc = json.loads(text, object_pairs_hook=_unique_keys)
    if not isinstance(doc, dict):
        raise ValueError(f"WAR 파라미터는 JSON 객체여야 한다: {type(doc).__name__}")
    if set(doc) != _PARAM_KEYS:
        raise ValueError(f"WAR 파라미터 최상위 키는 {sorted(_PARAM_KEYS)}여야 한다: {sorted(doc)}")
    return _checked_war_params(WarParams(version=doc["version"], regress=doc["regress"]))


def war_params_sha256(p: WarParams) -> str:
    """`war_params_to_json(p)`의 UTF-8 sha256 (소문자 16진 64자)."""
    return hashlib.sha256(war_params_to_json(p).encode("utf-8")).hexdigest()


def save_war_params(p: WarParams, path: Path) -> None:
    """파일로 쓴다(부모 디렉토리를 만든다)."""
    text = war_params_to_json(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_war_params(path: Path) -> WarParams:
    """파일에서 읽는다. 형식 오류는 ValueError."""
    return war_params_from_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 입력 확인 도우미
# ---------------------------------------------------------------------------


def _require(df: pd.DataFrame, spec: TableSpec, label: str) -> None:
    """표가 계약을 지키는지 확인한다. 어기면 ValueError."""
    problems = check_frame(df, spec)
    if problems:
        raise ValueError(f"{label}가 계약({spec.name})을 어긴다: " + "; ".join(problems))


def _show(ids: Iterable[int]) -> str:
    """오류 메시지용 선수 ID 목록 (앞 몇 명과 전체 수)."""
    values = sorted({int(i) for i in ids})
    more = f" 외 {len(values) - _SHOW_MAX}명" if len(values) > _SHOW_MAX else ""
    return f"{values[:_SHOW_MAX]}{more}"


def _cohort_target(cohort: pd.DataFrame, role: Role) -> int:
    """대상 표(한 역할·한 목표)를 확인하고 목표 시즌 Y를 돌려준다. 어기면 ValueError.

    기간이 Y−WINDOW ≤ window_first ≤ window_last = Y−1을 벗어난 대상도 막는다(N2).
    """
    _require(cohort, COHORTS, "대상 표")
    if cohort.empty:
        raise ValueError("대상 표가 비어 있어 목표 시즌을 정할 수 없다")
    roles = sorted({str(r) for r in cohort["role"]})
    if roles != [role]:
        raise ValueError(f"대상 표의 역할은 {role} 하나여야 한다: {roles}")
    targets = sorted({int(t) for t in cohort["target_season"]})
    if len(targets) != 1:
        raise ValueError(f"대상 표의 목표 시즌은 하나여야 한다: {targets}")
    target = targets[0]
    outside = (
        (cohort["window_last"] != target - 1)
        | (cohort["window_first"] < target - WINDOW)
        | (cohort["window_first"] > cohort["window_last"])
    )
    if bool(outside.any()):
        raise ValueError(
            f"기간이 목표 시즌 {target}의 {target - WINDOW}~{target - 1}을 벗어난 대상: "
            + _show(cohort.loc[outside, "player_id"])
        )
    return target


def _weight_of(target: int, weights: Sequence[float]) -> dict[int, float]:
    """시즌 → 가중치 (weights[0] = Y−1 … weights[WINDOW−1] = Y−WINDOW). 어기면 ValueError."""
    if len(weights) != WINDOW:
        raise ValueError(f"시즌 가중치는 {WINDOW}개여야 한다: {tuple(weights)}")
    values = [float(w) for w in weights]
    if any(not math.isfinite(w) or w < 0 for w in values):
        raise ValueError(f"시즌 가중치는 0 이상의 유한수여야 한다: {tuple(weights)}")
    return {target - 1 - k: w for k, w in enumerate(values)}


def _window_rows(
    cohort: pd.DataFrame, seasons: pd.DataFrame, target: int, columns: Sequence[str]
) -> pd.DataFrame:
    """대상마다 기간(window_first ~ window_last) 안 시즌 행. 열: player_id, season, *columns.

    season ≥ Y 행은 기간 밖이라 쓰지 않는다(N2 — 기간 검사와 별도로 한 번 더 막는다).
    입력 행 순서와 상관없이 같은 비트가 나오게 (player_id, season) 순으로 정렬한다.
    """
    past = seasons[(seasons["season"] < target) & (seasons["season"] >= target - WINDOW)]
    window = cohort[["player_id", "window_first", "window_last"]]
    rows = past[["player_id", "season", *columns]].merge(
        window, on="player_id", how="inner", validate="many_to_one"
    )
    keep = rows["season"].between(rows["window_first"], rows["window_last"])
    rows = rows.loc[keep, ["player_id", "season", *columns]]
    return rows.sort_values(["player_id", "season"], kind="mergesort").reset_index(drop=True)


def _weighted_sums(
    rows: pd.DataFrame,
    columns: Sequence[str],
    weight_of: Mapping[int, float],
    ids: pd.Index,
) -> pd.DataFrame:
    """선수마다 Σ w·열 (index = ids). 기간 안 행이 없는 선수는 0."""
    weight = rows["season"].map(weight_of).astype("float64")
    sums = (
        rows[list(columns)].astype("float64").mul(weight, axis=0).groupby(rows["player_id"]).sum()
    )
    return sums.reindex(ids, fill_value=0.0)


def _player_index(cohort: pd.DataFrame) -> pd.Index:
    return pd.Index(sorted(int(p) for p in cohort["player_id"]), name="player_id", dtype="int64")


def _role_weights(params: Params, role: Role) -> tuple[float, ...]:
    return params.bat.weights if role == "bat" else params.pit.weights


# ---------------------------------------------------------------------------
# 리그 비율 · WAR 비율 (F18) · 투수 자리 (D35)
# ---------------------------------------------------------------------------


def league_war_rate(
    war_seasons: pd.DataFrame, seasons: pd.DataFrame, role: Role, season: int
) -> float:
    """리그 WAR 비율 = 그 시즌 역할 WAR 합 / 출전 합 (F18의 lg).

    war_seasons(`WAR_SEASONS`)의 (role, season) 행 war 합을 seasons(그 역할 정제 시즌 표)의 같은 시즌
    `WAR_UNIT[role]` 합으로 나눈다.

    Raises:
        ValueError: 그 시즌 출전 합이 0이거나 행이 없을 때.
    """
    _require(war_seasons, WAR_SEASONS, "WAR 표")
    _require(seasons, _SEASON_SPECS[role], f"{role} 정제 표")
    wars = war_seasons[(war_seasons["role"] == role) & (war_seasons["season"] == season)]
    units = seasons.loc[seasons["season"] == season, WAR_UNIT[role]]
    if wars.empty or units.empty:
        raise ValueError(
            f"{season} {role} 리그 WAR 비율을 낼 행이 없다 (WAR {len(wars)}행, 시즌 {len(units)}행)"
        )
    # 행 순서와 상관없이 같은 비트가 나오게 키 순으로 더한다
    war_sum = float(wars.sort_values(["player_id", "league"], kind="mergesort")["war"].sum())
    unit_sum = float(units.astype("float64").sum())
    if unit_sum <= 0:
        raise ValueError(f"{season} {role} 리그 출전({WAR_UNIT[role]}) 합이 0이다")
    return war_sum / unit_sum


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
    target = _cohort_target(cohort, role)
    _require(seasons, _SEASON_SPECS[role], f"{role} 정제 표")
    _require(war_seasons, WAR_SEASONS, "WAR 표")
    weight_of = _weight_of(target, weights)
    r = _positive(regress, "regress")
    if not math.isfinite(league_rate):
        raise ValueError(f"리그 WAR 비율이 유한수가 아니다: {league_rate!r}")
    unit = WAR_UNIT[role]

    rows = _window_rows(cohort, seasons, target, [unit])
    # 같은 역할·기간 시즌의 WAR 행만 (다른 역할·목표 이후 행은 보지 않는다, N2)
    wars = war_seasons.loc[
        (war_seasons["role"] == role) & war_seasons["season"].isin(list(weight_of)),
        ["player_id", "season", "war"],
    ]
    rows = rows.merge(wars, on=["player_id", "season"], how="left", validate="one_to_one")
    missing = rows["war"].isna()
    if bool(missing.any()):
        pairs = [
            f"{int(p)}({int(s)})"
            for p, s in zip(
                rows.loc[missing, "player_id"], rows.loc[missing, "season"], strict=True
            )
        ]
        more = f" 외 {len(pairs) - _SHOW_MAX}행" if len(pairs) > _SHOW_MAX else ""
        raise ValueError(
            f"{target} {role} 기간 안 시즌 행에 맞는 WAR 행이 없다 — player_id(시즌): "
            f"{', '.join(pairs[:_SHOW_MAX])}{more}"
        )

    ids = _player_index(cohort)
    sums = _weighted_sums(rows, ["war", unit], weight_of, ids)
    rate = (sums["war"] + r * league_rate) / (sums[unit] + r)
    return pd.DataFrame(
        {
            "player_id": pd.Series(ids.to_numpy(), dtype="int64"),
            "war_rate": pd.Series(rate.to_numpy(), dtype="float64"),
        }
    )


def pitcher_slots(
    cohort: pd.DataFrame, pit_seasons: pd.DataFrame, weights: tuple[float, ...]
) -> pd.DataFrame:
    """투수 자리 (D35). 열: player_id, slot. 입력 기간 Σw·gs / Σw·g ≥ `STARTER_SHARE_MIN`이면 "sp", 아니면 "rp"
    (Σw·g = 0이면 "rp"). 가중치는 `war_rates`와 같은 규칙. player_id 순."""
    target = _cohort_target(cohort, "pit")
    _require(pit_seasons, PITCHING_SEASONS, "pit 정제 표")
    weight_of = _weight_of(target, weights)
    ids = _player_index(cohort)
    sums = _weighted_sums(
        _window_rows(cohort, pit_seasons, target, ["g", "gs"]), ["g", "gs"], weight_of, ids
    )
    games = sums["g"].to_numpy()
    starts = sums["gs"].to_numpy()
    # Σw·g = 0이면 나누지 않고 불펜
    share = np.divide(starts, games, out=np.zeros_like(starts), where=games > 0)
    starter = (games > 0) & (share >= STARTER_SHARE_MIN)
    return pd.DataFrame(
        {
            "player_id": pd.Series(ids.to_numpy(), dtype="int64"),
            "slot": pd.Series(np.where(starter, "sp", "rp"), dtype="str"),
        }
    )


# ---------------------------------------------------------------------------
# 예측 WAR · 풀타임 환산 (F19)
# ---------------------------------------------------------------------------


def _same_players(df: pd.DataFrame, label: str, expected: set[int]) -> None:
    """표의 선수 집합이 대상과 같고 중복이 없는지 확인한다. 어기면 ValueError."""
    ids = df["player_id"]
    if bool(ids.duplicated().any()):
        raise ValueError(f"{label}에 같은 선수가 두 번 있다: {_show(ids[ids.duplicated()])}")
    got = {int(p) for p in ids}
    if got != expected:
        raise ValueError(
            f"{label}의 선수 집합이 대상 표와 다르다 (없음 {_show(expected - got)}, "
            f"추가 {_show(got - expected)})"
        )


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
    target = _cohort_target(cohort, role)
    if role == "pit" and slots is None:
        raise ValueError("투수 예측에는 자리 표(slots — pitcher_slots 결과)가 필요하다")
    _require(playing_time, _PT_SPECS[role], f"{role} 출전량 표")
    expected = {int(p) for p in cohort["player_id"]}
    _same_players(rates, "WAR 비율 표", expected)
    _same_players(playing_time, "출전량 표", expected)
    late = sorted({int(t) for t in playing_time["target_season"]} - {target})
    if late:
        raise ValueError(f"출전량 표의 목표 시즌이 대상 표({target})와 다르다: {late}")

    pt_column = WAR_UNIT[role]  # 타자 pa, 투수 outs — 출전량 표의 열 이름과 같다
    base = cohort[["player_id", "n_seasons"]].sort_values("player_id", kind="mergesort")
    out = base.merge(rates[["player_id", "war_rate"]], on="player_id", validate="one_to_one")
    out = out.merge(
        playing_time[["player_id", pt_column]].rename(columns={pt_column: "pt"}),
        on="player_id",
        validate="one_to_one",
    )
    if role == "pit":
        assert slots is not None  # 위에서 막았다
        _same_players(slots, "자리 표", expected)
        out = out.merge(slots[["player_id", "slot"]], on="player_id", validate="one_to_one")
        unknown = sorted({str(s) for s in out["slot"]} - _PIT_SLOTS)
        if unknown:
            raise ValueError(f"투수 자리는 {sorted(_PIT_SLOTS)} 중 하나여야 한다: {unknown}")
        slot = [str(s) for s in out["slot"]]
    else:
        slot = [_BAT_SLOT] * len(out)

    war_rate = out["war_rate"].astype("float64").to_numpy()
    pt = out["pt"].astype("float64").to_numpy()
    full_units = np.array([FULL_TIME_UNITS[s] for s in slot], dtype="float64")
    result = pd.DataFrame(
        {
            "player_id": pd.Series(out["player_id"].to_numpy(), dtype="int64"),
            "role": pd.Series([role] * len(out), dtype="str"),
            "target_season": pd.Series([target] * len(out), dtype="int64"),
            "n_seasons": pd.Series(out["n_seasons"].to_numpy(), dtype="int64"),
            "slot": pd.Series(slot, dtype="str"),
            "war_rate": pd.Series(war_rate, dtype="float64"),
            "pt": pd.Series(pt, dtype="float64"),
            "war": pd.Series(war_rate * pt, dtype="float64"),
            "full_units": pd.Series(full_units, dtype="float64"),
            "war_full": pd.Series(war_rate * full_units, dtype="float64"),
        }
    )[WAR_PROJECTIONS.column_names]
    _require(result, WAR_PROJECTIONS, "WAR 예측 결과")
    return result


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
    regress = _checked_war_params(war_params).regress[role]
    weights = _role_weights(params, role)
    cohort = cohort_for(tables, role, target)
    seasons = tables.seasons[role]
    league_rate = league_war_rate(war_seasons, seasons, role, target - 1)
    rates = war_rates(cohort, seasons, war_seasons, role, weights, regress, league_rate)
    slots = pitcher_slots(cohort, seasons, weights) if role == "pit" else None
    if playing_time is None:
        playing_time = predict_playing_time(cohort, seasons, role, tables.debuts)
    return project_war(cohort, rates, playing_time, role, slots)


# ---------------------------------------------------------------------------
# 투타 합산 순위 (D34)
# ---------------------------------------------------------------------------


def _role_columns(proj: pd.DataFrame | None, ids: pd.Index) -> tuple[pd.Series, pd.Series]:
    """한 역할 예측의 (war, war_full)을 ids에 맞춘다. 예측이 없는 선수는 결측."""
    if proj is None:
        empty = pd.Series(np.nan, index=ids, dtype="float64")
        return empty, empty.copy()
    by_pid = proj.set_index("player_id")
    return (
        by_pid["war"].astype("float64").reindex(ids),
        by_pid["war_full"].astype("float64").reindex(ids),
    )


def _check_projection(proj: pd.DataFrame, role: Role, target: int) -> set[int]:
    """`WAR_PROJECTIONS` 한 역할 표를 확인하고 선수 집합을 돌려준다. 어기면 ValueError."""
    _require(proj, WAR_PROJECTIONS, f"{role} WAR 예측 표")
    roles = sorted({str(r) for r in proj["role"]} - {role})
    if roles:
        raise ValueError(f"{role} WAR 예측 표에 다른 역할이 있다: {roles}")
    targets = sorted({int(t) for t in proj["target_season"]} - {target})
    if targets:
        raise ValueError(f"{role} WAR 예측 표의 목표 시즌이 {target}과 다르다: {targets}")
    return {int(p) for p in proj["player_id"]}


def combine_roles(
    bat: pd.DataFrame | None, pit: pd.DataFrame | None, war_seasons: pd.DataFrame, target: int
) -> pd.DataFrame:
    """두 역할 예측(`WAR_PROJECTIONS`, 같은 목표)을 투타 합산 순위 표 `WAR_RANKINGS`로 (D34).

    roles = "both"(두 역할 다) / "bat" / "pit". war = war_bat + war_pit(결측 0), war_full도 같다.
    prev_war = war_seasons에서 목표 Y−1의 그 선수 행(두 역할) war 합, 행이 없으면 결측.
    정렬: war 내림차순, 같으면 player_id 오름차순.

    Raises:
        ValueError: 두 입력의 목표 시즌이 target과 다를 때. 두 입력이 모두 None일 때.
    """
    if bat is None and pit is None:
        raise ValueError("타자·투수 WAR 예측이 모두 없다")
    bat_ids = set() if bat is None else _check_projection(bat, "bat", target)
    pit_ids = set() if pit is None else _check_projection(pit, "pit", target)
    _require(war_seasons, WAR_SEASONS, "WAR 표")

    ids = pd.Index(sorted(bat_ids | pit_ids), name="player_id", dtype="int64")
    war_bat, war_full_bat = _role_columns(bat, ids)
    war_pit, war_full_pit = _role_columns(pit, ids)
    has_bat = ids.isin(list(bat_ids))
    has_pit = ids.isin(list(pit_ids))
    roles = np.where(has_bat & has_pit, "both", np.where(has_bat, "bat", "pit"))

    # Y−1 실제 WAR 투타 합 — 그 시즌 행만 (목표 시즌 이후는 보지 않는다, N2)
    prev_rows = war_seasons[war_seasons["season"] == target - 1].sort_values(
        ["player_id", "role", "league"], kind="mergesort"
    )
    prev_war = prev_rows.groupby("player_id")["war"].sum().astype("float64").reindex(ids)

    out = pd.DataFrame(
        {
            "player_id": pd.Series(ids.to_numpy(), dtype="int64"),
            "target_season": pd.Series([target] * len(ids), dtype="int64"),
            "roles": pd.Series(roles, dtype="str"),
            "war_bat": war_bat.to_numpy(),
            "war_pit": war_pit.to_numpy(),
            "war": (war_bat.fillna(0.0) + war_pit.fillna(0.0)).to_numpy(),
            "war_full_bat": war_full_bat.to_numpy(),
            "war_full_pit": war_full_pit.to_numpy(),
            "war_full": (war_full_bat.fillna(0.0) + war_full_pit.fillna(0.0)).to_numpy(),
            "prev_war": prev_war.to_numpy(),
        }
    )[WAR_RANKINGS.column_names]
    out = out.sort_values(
        ["war", "player_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    _require(out, WAR_RANKINGS, "WAR 순위 결과")
    return out


# ---------------------------------------------------------------------------
# WAR 회귀 표본 조정 (D36)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TuneCase:
    """한 역할·한 목표의 조정 입력 (R 후보 사이에 재사용한다)."""

    target: int
    cohort: pd.DataFrame
    seasons: pd.DataFrame
    league_rate: float
    slots: pd.DataFrame | None
    playing_time: pd.DataFrame
    actual: pd.Series  # 목표 시즌 실제 WAR (index = 대상 player_id, 역할 행 없으면 0)


def _tune_case(
    tables: Tables, war_seasons: pd.DataFrame, role: Role, target: int, weights: tuple[float, ...]
) -> _TuneCase:
    """`war_projections`와 같은 순서로 R과 무관한 입력을 한 번만 만든다 (출전량 예측 포함)."""
    cohort = cohort_for(tables, role, target)
    seasons = tables.seasons[role]
    league_rate = league_war_rate(war_seasons, seasons, role, target - 1)
    slots = pitcher_slots(cohort, seasons, weights) if role == "pit" else None
    playing_time = predict_playing_time(cohort, seasons, role, tables.debuts)
    rows = war_seasons[(war_seasons["role"] == role) & (war_seasons["season"] == target)]
    actual = rows.groupby("player_id")["war"].sum().astype("float64")
    actual = actual.reindex(_player_index(cohort), fill_value=0.0)
    return _TuneCase(target, cohort, seasons, league_rate, slots, playing_time, actual)


def _tune_mae(
    case: _TuneCase,
    war_seasons: pd.DataFrame,
    role: Role,
    weights: tuple[float, ...],
    regress: float,
) -> float:
    """후보 R의 예측 WAR MAE (대상 전원)."""
    rates = war_rates(
        case.cohort, case.seasons, war_seasons, role, weights, regress, case.league_rate
    )
    proj = project_war(case.cohort, rates, case.playing_time, role, case.slots)
    predicted = proj.set_index("player_id")["war"].reindex(case.actual.index)
    return float((predicted - case.actual).abs().mean())


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
    # 데이터를 읽기 전에 목표부터 본다 — 생성기여도 한 번만 돈다
    chosen_targets = sorted({int(t) for t in targets})
    late = [t for t in chosen_targets if t >= FINAL_TEST_TARGET]
    if late:
        raise ValueError(
            f"WAR 조정에는 {list(TUNE_TARGETS)}만 쓴다 — {FINAL_TEST_TARGET} 이후는 "
            f"final-test-war로 한 번만 (SPEC D36): {late}"
        )
    if not chosen_targets:
        raise ValueError("조정할 목표 시즌이 없다")
    if set(grid) != set(ROLES):
        raise ValueError(f"grid 역할은 {list(ROLES)}여야 한다: {sorted(map(str, grid))}")
    candidates: dict[Role, list[float]] = {}
    for role in ROLES:
        values = sorted({_positive(r, f"grid.{role}") for r in grid[role]})
        if not values:
            raise ValueError(f"grid.{role} 후보가 없다")
        candidates[role] = values

    records: list[dict[str, Any]] = []
    chosen: dict[Role, float] = {}
    for role in ROLES:
        weights = _role_weights(params, role)
        cases = [_tune_case(tables, war_seasons, role, t, weights) for t in chosen_targets]
        maes = {
            r: [_tune_mae(case, war_seasons, role, weights, r) for case in cases]
            for r in candidates[role]
        }
        means = {r: sum(values) / len(values) for r, values in maes.items()}
        best = candidates[role][0]
        for r in candidates[role]:  # 오름차순 — 같으면 먼저 나온(작은) R
            if means[r] < means[best]:
                best = r
        chosen[role] = best
        for r in candidates[role]:
            for case, mae in zip(cases, maes[r], strict=True):
                records.append(
                    {
                        "role": role,
                        "regress": r,
                        "target_season": case.target,
                        "mae": mae,
                        "mean_mae": means[r],
                        "chosen": r == best,
                    }
                )

    report = pd.DataFrame(
        {
            "role": pd.Series([rec["role"] for rec in records], dtype="str"),
            "regress": pd.Series([rec["regress"] for rec in records], dtype="float64"),
            "target_season": pd.Series([rec["target_season"] for rec in records], dtype="int64"),
            "mae": pd.Series([rec["mae"] for rec in records], dtype="float64"),
            "mean_mae": pd.Series([rec["mean_mae"] for rec in records], dtype="float64"),
            "chosen": pd.Series([rec["chosen"] for rec in records], dtype="bool"),
        }
    )[list(_TUNE_COLUMNS)]
    return WarParams(version="v1", regress=chosen), report
