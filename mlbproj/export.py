"""웹 JSON·검토용 CSV 내보내기 (SPEC F11·F13~F15·D14·D16·D21·D23·N1·N6). 레인 B가 구현한다.

웹과의 계약은 `contracts/web-data.schema.json` 하나다. 파일: batters.json(BatTable), pitchers.json(PitTable),
validation.json(ValidationReport), players/{player_id}.json(PlayerDetail). 여기서는 모델 계산을 하지 않는다 —
입력 표의 값을 모양에 맞추고, 계약에 적힌 간단한 식(비율·FIP·평균)과 반올림만 한다.

공통 규칙
- 목표 시즌 Y = meta["target_season"](2027). Y−1 = 마지막 입력 시즌.
- 반올림: 키 이름이 `contracts.WEB_ROUND`에 있으면 그 자릿수로 `round()`. 정수 열은 int 그대로.
  결측(NaN·None)은 null. NaN·inf가 JSON에 들어가면 안 된다.
- 행 순서: player_id 오름차순. 검증 행은 (target_season, role, model, metric, group) 순.
- 직렬화(`dump_json`): `json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\\n"`.
  같은 입력이면 같은 바이트(N1).

예측표 행 (BatRow·PitRow, 그 역할의 Y 대상 전원 = inp.projections[role]의 선수)
- player(PlayerSummary): name·primary_position·active·last_mlb_season은 players 표. age = `age.season_age(birth_date, Y)`,
  생년월일이 없으면 그 역할 정제 표의 input_last 행 `age` + (Y − input_last), 그것도 없으면 null.
  team_name = 그 역할 정제 표의 input_last 행 team_name.
- window: 대상 표(cohorts, Y)의 input_first·input_last·n_seasons.
- 타자: sample_pa = 대상 표 sample. projection = 예측 표의 pa·hr·obp·slg·ops.
  prev = Y−1 타격 행이 있으면 {season, pa, hr, ops}, 없으면 null.
- 투수: sample_bf = 대상 표 sample, sample_ip = sample_outs / 3. projection = ip·era·k_bb_pct.
  prev = Y−1 투구 행이 있으면 {season, ip = outs/3, era = 27·er/outs, k_bb_pct = 100·(so − bb)/bf}
  (분모 0이면 그 값 null), 없으면 null.
- league_baseline: 타자 {"ops": 리그 Y−1 ops}, 투수 {"era": 리그 Y−1 era, "k_bb_pct": 리그 Y−1 k_bb_pct}
  (tables.league).
- 표 객체: {meta, role, model = inp.model[role], league_baseline, rows}.

선수 상세 (PlayerDetail, 두 역할 Y 대상의 합집합마다 한 파일)
- player: players 표의 player_id·name·birth_date·primary_position·bats·throws·active·last_mlb_season.
- bat / pit: 그 역할 Y 대상이 아니면 null(F12). 대상이면 BatDetail/PitDetail —
  window·sample·projection·prev·league_baseline은 예측표 행과 같은 값.
  history: 그 역할 정제 표에서 input_first ~ input_last 안의 행(최대 5개, season 오름차순).
    타자 BatSeason: k_pct = 100·so/pa, bb_pct = 100·bb/pa (pa 0이면 null), 나머지는 표 값.
    투수 PitSeason: ip = outs/3, era = 27·er/outs, fip = (13·hr + 3·(bb + hbp) − 2·so) / (outs/3) + 그 시즌
    리그 fip_const, k_pct = 100·so/bf, bb_pct = 100·bb/bf, k_bb_pct = k_pct − bb_pct (분모 0이면 null).
  group_error: group = n_seasons ≤ 2이면 "seasons_1_2", 아니면 "seasons_3_5".
    mae[지표] = evaluations에서 (role, model = inp.model[role], metric, group) 행의 mae를 목표 시즌들에 걸쳐
    단순 평균한 값(결측은 빼고 평균, 행이 없으면 null). 지표 = `BAT_METRICS`/`PIT_METRICS`.
- fielding: 그 선수의 수비 표 행(`FIELDING_YEARS` 안, 최근 5개, season 오름차순). outs_by_pos =
  {`POSITION_ABBR[p]`: outs_p} 중 outs_p > 0인 포지션만. 수비 행이 없으면 [].

검증 (ValidationReport): {meta, final_test = inp.final_test, rows = evaluations 전부(EvalRow),
bands = inp.bands 행(BandRow, 표 순서 그대로, q80은 지표 키로 반올림, targets는 정수 목록)}.

과거 오차 80% 범위 (D29, inp.bands = `contracts.BANDS`. 비면 값은 모두 null)
- 예측표 객체의 bands = {그룹: {지표: q80}} (`BAND_GROUPS` × `BAND_METRICS[role]`).
- 선수 상세 group_error.band80 = 그 선수 그룹의 {지표: q80}.

쓰기 (`write_web`)
- 파일마다 스키마 정의(`WEB_FILES`, 선수 상세는 `WEB_PLAYER_DEF`)로 `schema.validate` 검사. 위반이 있으면
  아무것도 쓰지 않고 ValueError(파일 이름과 처음 몇 개 위반).
- `WEB_FIRST_SCREEN_FILES`가 각각 `WEB_FIRST_SCREEN_MAX_BYTES`를 넘으면 쓰기 전에 ValueError(N6).
- out_dir/players/ 안에 이번 묶음에 없는 .json 파일은 지운다(은퇴 등으로 빠진 선수). 다른 파일은 건드리지 않는다.

검토용 CSV (`write_projection_csv`): 역할마다 예측 표 열(`BAT_PROJECTIONS`·`PIT_PROJECTIONS` 순) 뒤에
name 열을 붙여 `PROJECTION_CSV[role]`에 쓴다. player_id 순, 줄 끝 "\\n", 실수는 pandas 기본 표기.
"""

import json
import math
import re
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlbproj import contracts as c
from mlbproj.age import season_age
from mlbproj.bands import empty_bands
from mlbproj.contracts import PROJECTION_CSV, Role
from mlbproj.schema import validate
from mlbproj.tables import Tables

_OUTS_PER_GAME = 27  # ERA = 9이닝(27아웃)당 자책점
_OUTS_PER_INNING = 3
_PCT = 100.0
_SHORT_CAREER_MAX = 2  # seasons_1_2 그룹의 n_seasons 상한 (evaluate와 같은 규칙, D13)
_GROUP_SHORT = "seasons_1_2"
_GROUP_LONG = "seasons_3_5"
_SHOW_MAX = 5  # 오류 메시지에 적는 선수 ID·스키마 위반의 최대 개수

_PLAYERS_DIR = "players"  # 선수 상세 하위 디렉토리 (`contracts.web_player_path`와 같다)
_PLAYER_FILE = re.compile(rf"{_PLAYERS_DIR}/[1-9][0-9]*\.json")
_VALIDATION_FILE = "validation.json"
_TABLE_FILES: dict[Role, str] = {"bat": "batters.json", "pit": "pitchers.json"}
# 검증 행 순서. EVALUATIONS 키에 role을 더한 것이다 (문자열 사전순)
_EVAL_ORDER: list[str] = ["target_season", "role", "model", "metric", "group"]

_PROJECTION_SPECS: dict[Role, c.TableSpec] = {"bat": c.BAT_PROJECTIONS, "pit": c.PIT_PROJECTIONS}
_METRICS: dict[Role, tuple[str, ...]] = {"bat": c.BAT_METRICS, "pit": c.PIT_METRICS}
# 리그 평균 기준선 = 그 역할의 비율 지표 (타자 ops, 투수 era·k_bb_pct)
_BASELINE_KEYS: dict[Role, tuple[str, ...]] = {
    role: tuple(m for m in _METRICS[role] if m in c.RATE_METRICS) for role in c.ROLES
}

# 웹 JSON 객체의 키 (스키마 정의의 필수 키 순서)
_SUMMARY_KEYS = (
    "player_id",
    "name",
    "primary_position",
    "age",
    "team_name",
    "active",
    "last_mlb_season",
)
_WINDOW_KEYS = ("input_first", "input_last", "n_seasons")
_SAMPLE_KEYS: dict[Role, tuple[str, ...]] = {
    "bat": ("sample_pa",),
    "pit": ("sample_bf", "sample_ip"),
}
_PROJECTION_KEYS: dict[Role, tuple[str, ...]] = {
    "bat": ("pa", "hr", "obp", "slg", "ops"),
    "pit": ("ip", "era", "k_bb_pct"),
}
_PREV_KEYS: dict[Role, tuple[str, ...]] = {
    "bat": ("season", "pa", "hr", "ops"),
    "pit": ("season", "ip", "era", "k_bb_pct"),
}
_HISTORY_KEYS: dict[Role, tuple[str, ...]] = {
    "bat": (
        "season",
        "age",
        "team_name",
        "is_short_season",
        "pa",
        "hr",
        "obp",
        "slg",
        "ops",
        "k_pct",
        "bb_pct",
        "woba",
        "xwoba",
        "barrel_pct",
        "hard_hit_pct",
        "ev_avg",
    ),
    "pit": (
        "season",
        "age",
        "team_name",
        "is_short_season",
        "g",
        "gs",
        "ip",
        "era",
        "fip",
        "k_pct",
        "bb_pct",
        "k_bb_pct",
        "xera",
        "xwoba_against",
        "whiff_pct",
    ),
}
_FIELDING_KEYS = (
    "season",
    "oaa",
    "frv_total",
    "frv_range",
    "frv_arm",
    "frv_dp",
    "frv_framing",
    "frv_blocking",
    "frv_throwing",
)


def _empty_bands() -> pd.DataFrame:
    """빈 범위 표 (`contracts.BANDS` 열과 형식)."""
    return empty_bands()


@dataclass(frozen=True)
class ExportInput:
    """내보내기 입력 한 묶음.

    Attributes:
        tables: 정제 표 묶음 (players·seasons·fielding·league·cohorts. cohorts에 목표 Y 행이 있어야 한다).
        projections: 역할 → 목표 Y의 최종 모델 예측 표 (`BAT_PROJECTIONS`·`PIT_PROJECTIONS`, 그 모델 행만).
        model: 역할 → 최종 모델 이름 ("extended" 또는 "weighted").
        evaluations: 과거 검증 행 (`EVALUATIONS`, 2024·2025와 최종 검증 뒤 2026).
        final_test: ValidationReport.final_test 객체. 최종 검증 전이면 None.
        meta: Meta 객체 전체 (schema_version 포함, 스키마 그대로).
        bands: 과거 오차 80% 범위 (`contracts.BANDS`, D29). 비면 범위 값은 모두 null.
    """

    tables: Tables
    projections: dict[Role, pd.DataFrame]
    model: dict[Role, str]
    evaluations: pd.DataFrame
    final_test: dict[str, Any] | None
    meta: dict[str, Any]
    bands: pd.DataFrame = field(default_factory=_empty_bands)


# ---------------------------------------------------------------------------
# 값 변환
# ---------------------------------------------------------------------------


def _json_value(value: Any, key: str) -> Any:
    """표 값 하나를 JSON 값으로. 결측 → None, 정수 → int, 실수는 키가 WEB_ROUND에 있으면 반올림.

    반올림 결과가 −0.0이면 0.0으로 바꾼다 (FRV −0.04 → "-0.0"이 화면에 나오지 않게. 값은 같다).
    inf는 그대로 두고 쓰기 단계(스키마 검사·`dump_json`)에서 거부한다.
    """
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        if math.isnan(number):
            return None
        digits = c.WEB_ROUND.get(key)
        rounded = number if digits is None else round(number, digits)
        return 0.0 if rounded == 0 else rounded
    if value is None or value is pd.NA:
        return None
    if isinstance(value, str):
        return value
    raise TypeError(f"{key}: JSON 값으로 바꿀 수 없다 ({type(value).__name__}: {value!r})")


def _pick(record: Mapping[Hashable, Any], keys: Iterable[str]) -> dict[str, Any]:
    """레코드에서 keys만 골라 JSON 값 객체로 만든다 (키 순서 = keys 순서)."""
    return {key: _json_value(record[key], key) for key in keys}


def _by_player(df: pd.DataFrame, keys: Iterable[str]) -> dict[int, list[dict[str, Any]]]:
    """선수 ID → 그 선수 행들의 JSON 객체 목록 (표의 행 순서 그대로)."""
    names = tuple(keys)
    out: dict[int, list[dict[str, Any]]] = {}
    for record in df.to_dict("records"):
        out.setdefault(int(record["player_id"]), []).append(_pick(record, names))
    return out


# ---------------------------------------------------------------------------
# 예측표 행·선수 상세 (역할마다)
# ---------------------------------------------------------------------------


def _require_ids(ids: pd.Series, known: pd.Series, what: str) -> None:
    """예측 표의 선수가 모두 known에 있는지 확인한다. 없으면 ValueError."""
    missing = sorted(set(ids.tolist()) - set(known.tolist()))
    if missing:
        raise ValueError(f"예측 표의 선수 {len(missing)}명이 {what}에 없다: {missing[:_SHOW_MAX]}")


def _checked_projections(inp: ExportInput, role: Role, target: int) -> pd.DataFrame:
    """예측 표가 ExportInput 계약(그 역할·목표 Y·최종 모델 행만)을 지키는지 확인하고 player_id 순으로 돌려준다.

    다른 목표 시즌(예: 2026)의 예측이 2027로 공개되거나, 모델 이름과 값이 어긋나는 것을 막는다.

    Raises:
        ValueError: 표 계약(`BAT_PROJECTIONS`·`PIT_PROJECTIONS`) 위반, 또는 role·target_season·model이
            (role, Y, inp.model[role])이 아닌 행이 있을 때.
    """
    projections = inp.projections[role]
    problems = c.check_frame(projections, _PROJECTION_SPECS[role])
    if problems:
        raise ValueError(f"{role} 예측 표가 계약을 어긴다: " + "; ".join(problems))
    expected: dict[str, Any] = {"role": role, "target_season": target, "model": inp.model[role]}
    for column, value in expected.items():
        wrong = sorted(set(projections[column].tolist()) - {value})
        if wrong:
            raise ValueError(
                f"{role} 예측 표의 {column}은 {value!r}만 있어야 하는데 {wrong[:_SHOW_MAX]}도 있다"
            )
    return projections.sort_values("player_id", kind="mergesort").reset_index(drop=True)


def _checked_evaluations(inp: ExportInput, target: int) -> pd.DataFrame:
    """평가표를 확인하고 (target_season, role, model, metric, group) 순으로 정렬한다.

    2026 행(`FINAL_TEST_TARGET`)은 최종 검증 기록(final_test)이 있을 때만 받는다 — 한 번만 돌리는
    최종 검증(F10)을 거치지 않은 2026 결과를 공개하지 않는다.

    Raises:
        ValueError: 표 계약(`EVALUATIONS`) 위반, 목표 Y 이후 행, final_test 없이 2026 행이 있을 때.
    """
    evaluations = inp.evaluations
    problems = c.check_frame(evaluations, c.EVALUATIONS)
    if problems:
        raise ValueError("평가표가 계약을 어긴다: " + "; ".join(problems))
    seasons = evaluations["target_season"]
    late = sorted(set(seasons[seasons >= target].tolist()))
    if late:
        raise ValueError(f"평가표에 목표 {target} 이후 시즌 행이 있다: {late}")
    if inp.final_test is None and bool((seasons == c.FINAL_TEST_TARGET).any()):
        raise ValueError(
            f"평가표에 {c.FINAL_TEST_TARGET} 행이 있는데 최종 검증 기록(final_test)이 없다"
        )
    # 평균·행 순서가 입력 행 순서에 기대지 않게 정렬한다 (N1)
    return evaluations.sort_values(_EVAL_ORDER, kind="mergesort").reset_index(drop=True)


def _role_base(inp: ExportInput, role: Role, target: int) -> pd.DataFrame:
    """예측 표 + 대상 표(기간·표본) + 선수 표 + input_last 행(나이·팀). player_id 순, 선수당 한 행.

    Raises:
        ValueError: 예측 표가 계약을 어기거나, 예측 표의 선수가 목표 Y 대상 표나 players 표에 없을 때.
    """
    tables = inp.tables
    projections = _checked_projections(inp, role, target)
    cohorts = tables.cohorts
    cohort = cohorts[(cohorts["target_season"] == target) & (cohorts["role"] == role)]
    _require_ids(projections["player_id"], cohort["player_id"], f"{target} {role} 대상 표")
    _require_ids(projections["player_id"], tables.players["player_id"], "players 표")

    last = tables.seasons[role][["player_id", "season", "age", "team_name"]].rename(
        columns={"season": "input_last", "age": "last_age"}
    )
    base = (
        projections[["player_id", *_PROJECTION_KEYS[role]]]
        .merge(
            cohort[["player_id", *_WINDOW_KEYS, "sample", "sample_outs"]],
            on="player_id",
            how="left",
            validate="one_to_one",
        )
        .merge(tables.players, on="player_id", how="left", validate="one_to_one")
        .merge(last, on=["player_id", "input_last"], how="left", validate="one_to_one")
    )
    # 나이: 생년월일 기준 Y 시즌 나이, 없으면 input_last 행 나이 + (Y − input_last), 그것도 없으면 결측
    born = [season_age(b if isinstance(b, str) else None, target) for b in base["birth_date"]]
    fallback = base["last_age"] + (target - base["input_last"])
    base["age"] = pd.Series(born, index=base.index, dtype="Int64").fillna(fallback)
    if role == "bat":
        base["sample_pa"] = base["sample"]
    else:
        base["sample_bf"] = base["sample"]
        base["sample_ip"] = base["sample_outs"].astype("float64") / _OUTS_PER_INNING
    return base


def _season_stats(role: Role, seasons: pd.DataFrame, league: pd.DataFrame) -> pd.DataFrame:
    """정제 시즌 표 + 표시용 계산 열 (모듈 docstring history·prev). 분모가 0이면 그 값은 결측."""
    if role == "bat":
        pa = seasons["pa"].where(seasons["pa"] > 0)
        return seasons.assign(k_pct=_PCT * seasons["so"] / pa, bb_pct=_PCT * seasons["bb"] / pa)
    outs = seasons["outs"].where(seasons["outs"] > 0)
    bf = seasons["bf"].where(seasons["bf"] > 0)
    fip_const = seasons["season"].map(league.set_index("season")["fip_const"])
    fip_core = 13 * seasons["hr"] + 3 * (seasons["bb"] + seasons["hbp"]) - 2 * seasons["so"]
    return seasons.assign(
        ip=seasons["outs"] / _OUTS_PER_INNING,
        era=_OUTS_PER_GAME * seasons["er"] / outs,
        fip=fip_core / (outs / _OUTS_PER_INNING) + fip_const,
        k_pct=_PCT * seasons["so"] / bf,
        bb_pct=_PCT * seasons["bb"] / bf,
        # = k_pct − bb_pct. prev와 같은 식이라 같은 시즌이면 같은 값이 나온다
        k_bb_pct=_PCT * (seasons["so"] - seasons["bb"]) / bf,
    )


def _league_baseline(league: pd.DataFrame, role: Role, season: int) -> dict[str, Any]:
    """리그 평균 기준선 (Y−1 시즌 행의 비율 지표)."""
    rows = league[league["season"] == season].to_dict("records")
    if len(rows) != 1:
        raise ValueError(f"{role} 리그 평균에 {season} 시즌 행이 하나가 아니다 ({len(rows)}개)")
    return _pick(rows[0], _BASELINE_KEYS[role])


def _band_values(bands: pd.DataFrame, role: Role, group: str) -> dict[str, Any]:
    """그룹 하나의 80% 범위 (지표 → ± 값, `BAND_METRICS[role]` 순, 없으면 null). 지표 키로 반올림."""
    mine = bands[(bands["role"] == role) & (bands["group"] == group)]
    values = dict(zip(mine["metric"], mine["q80"], strict=True))
    return {m: _json_value(values.get(m, math.nan), m) for m in c.BAND_METRICS[role]}


def _checked_bands(inp: ExportInput) -> pd.DataFrame:
    """범위 표가 비었거나 계약을 지키는지 확인한다."""
    if inp.bands.empty:
        return _empty_bands()
    problems = c.check_frame(inp.bands, c.BANDS)
    if problems:
        raise ValueError("범위 표 계약 위반: " + "; ".join(problems))
    return inp.bands


def _group_error(
    means: pd.Series, bands: pd.DataFrame, role: Role, n_seasons: int
) -> dict[str, Any]:
    """선수가 속한 그룹의 과거 검증 MAE와 80% 범위 (GroupError). 지표 키 이름으로 반올림한다."""
    group = _GROUP_SHORT if n_seasons <= _SHORT_CAREER_MAX else _GROUP_LONG
    mae = {m: _json_value(means.get((m, group), math.nan), m) for m in _METRICS[role]}
    return {"group": group, "mae": mae, "band80": _band_values(bands, role, group)}


def _role_entries(
    inp: ExportInput, role: Role, target: int, evaluations: pd.DataFrame
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """한 역할의 예측표 객체(BatTable·PitTable)와 선수 ID → 상세 블록(BatDetail·PitDetail)."""
    tables = inp.tables
    model = inp.model[role]
    base = _role_base(inp, role, target)
    stats = _season_stats(role, tables.seasons[role], tables.league[role])

    prev = {
        pid: rows[0]
        for pid, rows in _by_player(stats[stats["season"] == target - 1], _PREV_KEYS[role]).items()
    }
    spans = stats.merge(base[["player_id", "input_first", "input_last"]], on="player_id")
    in_window = (spans["season"] >= spans["input_first"]) & (spans["season"] <= spans["input_last"])
    history = _by_player(
        spans[in_window].sort_values(["player_id", "season"], kind="mergesort"),
        _HISTORY_KEYS[role],
    )
    baseline = _league_baseline(tables.league[role], role, target - 1)
    # (지표, 그룹) → 목표 시즌들에 걸친 mae 단순 평균 (결측은 빼고, 전부 결측이면 결측)
    mine = evaluations[(evaluations["role"] == role) & (evaluations["model"] == model)]
    means = mine.groupby(["metric", "group"])["mae"].mean()
    bands = _checked_bands(inp)

    rows: list[dict[str, Any]] = []
    details: dict[int, dict[str, Any]] = {}
    for record in base.to_dict("records"):
        pid = int(record["player_id"])
        core = {
            "window": _pick(record, _WINDOW_KEYS),
            **_pick(record, _SAMPLE_KEYS[role]),
            "projection": _pick(record, _PROJECTION_KEYS[role]),
            "prev": prev.get(pid),
        }
        rows.append({"player": _pick(record, _SUMMARY_KEYS), **core})
        details[pid] = {
            **core,
            "league_baseline": baseline,
            "group_error": _group_error(means, bands, role, int(record["n_seasons"])),
            "history": history.get(pid, []),
        }
    table = {
        "meta": inp.meta,
        "role": role,
        "model": model,
        "league_baseline": baseline,
        "bands": {group: _band_values(bands, role, group) for group in c.BAND_GROUPS},
        "rows": rows,
    }
    return table, details


def _fielding(
    fielding: pd.DataFrame, ids: Iterable[int], target: int
) -> dict[int, list[dict[str, Any]]]:
    """선수 ID → FieldingSeason 목록 (`FIELDING_YEARS` 안 최근 `WINDOW`개, season 순).

    목표 Y 이후 시즌은 넣지 않는다 — 다른 블록처럼 Y−1까지만 보여준다(N2). 2027에서는 FIELDING_YEARS가
    모두 2026 이하라 결과가 같다.
    """
    if fielding.empty:
        return {}
    keep = (
        fielding["season"].isin(c.FIELDING_YEARS)
        & (fielding["season"] < target)
        & fielding["player_id"].isin(list(ids))
    )
    rows = (
        fielding[keep]
        .sort_values(["player_id", "season"], kind="mergesort")
        .groupby("player_id")
        .tail(c.WINDOW)
    )
    out: dict[int, list[dict[str, Any]]] = {}
    for record in rows.to_dict("records"):
        entry = _pick(record, _FIELDING_KEYS)
        outs = {c.POSITION_ABBR[p]: int(record[f"outs_{p}"]) for p in c.FRV_OUT_POSITIONS}
        entry["outs_by_pos"] = {pos: n for pos, n in outs.items() if n > 0}
        out.setdefault(int(record["player_id"]), []).append(entry)
    return out


def build_web(inp: ExportInput) -> dict[str, Any]:
    """웹 JSON 묶음. 상대 경로("batters.json", "players/660271.json" …) → JSON 객체.

    Raises:
        ValueError: 예측 표의 선수가 목표 Y 대상 표에 없거나, players 표에 없을 때.
    """
    target = int(inp.meta["target_season"])
    evaluations = _checked_evaluations(inp, target)
    files: dict[str, Any] = {}
    details: dict[Role, dict[int, dict[str, Any]]] = {}
    for role in c.ROLES:
        files[_TABLE_FILES[role]], details[role] = _role_entries(inp, role, target, evaluations)

    ids = sorted(set(details["bat"]) | set(details["pit"]))
    players = inp.tables.players
    people = {
        int(record["player_id"]): _pick(record, c.PLAYERS.column_names)
        for record in players[players["player_id"].isin(ids)].to_dict("records")
    }
    fielding = _fielding(inp.tables.fielding, ids, target)
    for pid in ids:
        files[f"{_PLAYERS_DIR}/{pid}.json"] = {
            "meta": inp.meta,
            "player": people[pid],
            "bat": details["bat"].get(pid),
            "pit": details["pit"].get(pid),
            "fielding": fielding.get(pid, []),
        }
    files[_VALIDATION_FILE] = {
        "meta": inp.meta,
        "final_test": inp.final_test,
        "rows": [
            _pick(record, c.EVALUATIONS.column_names) for record in evaluations.to_dict("records")
        ],
        "bands": [
            {
                "role": str(record["role"]),
                "metric": str(record["metric"]),
                "group": str(record["group"]),
                "n": int(record["n"]),
                "q80": _json_value(record["q80"], str(record["metric"])),
                "targets": [int(t) for t in str(record["targets"]).split(",") if t],
            }
            for record in _checked_bands(inp).to_dict("records")
        ],
    }
    return dict(sorted(files.items()))


def dump_json(obj: Any) -> str:
    """결정적 직렬화 (모듈 docstring "직렬화")."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def _def_name(path: str) -> str:
    """웹 파일 상대 경로 → 스키마 정의 이름. 모르는 이름이면 ValueError."""
    if path in c.WEB_FILES:
        return c.WEB_FILES[path]
    if _PLAYER_FILE.fullmatch(path):
        return c.WEB_PLAYER_DEF
    raise ValueError(f"모르는 웹 파일 이름: {path!r}")


def write_web(files: Mapping[str, Any], out_dir: Path, schema: dict[str, Any]) -> dict[str, int]:
    """스키마 검사·크기 검사 뒤 파일을 쓴다. 반환: 상대 경로 → 바이트 수 (경로 순).

    Raises:
        ValueError: 스키마 위반, 첫 화면 크기 초과, 모르는 파일 이름일 때 (아무것도 쓰지 않는다).
    """
    # 검사를 모두 마친 뒤에만 디스크를 건드린다
    encoded: dict[str, bytes] = {}
    for path in sorted(files):
        def_name = _def_name(path)
        errors = validate(files[path], def_name, schema)
        if errors:
            raise ValueError(
                f"{path}: 스키마 {def_name} 위반 {len(errors)}건 — {errors[:_SHOW_MAX]}"
            )
        encoded[path] = dump_json(files[path]).encode("utf-8")
    limit = c.WEB_FIRST_SCREEN_MAX_BYTES  # 호출 시점의 계약 값
    for path in c.WEB_FIRST_SCREEN_FILES:
        if path in encoded and len(encoded[path]) > limit:
            raise ValueError(
                f"{path}: {len(encoded[path]):,}바이트 — 첫 화면 한도 {limit:,}바이트를 넘는다 (N6)"
            )

    players_dir = out_dir / _PLAYERS_DIR
    players_dir.mkdir(parents=True, exist_ok=True)
    for old in sorted(players_dir.glob("*.json")):
        if old.is_file() and f"{_PLAYERS_DIR}/{old.name}" not in encoded:
            old.unlink()
    for path, data in encoded.items():
        (out_dir / path).write_bytes(data)
    return {path: len(data) for path, data in encoded.items()}


def write_projection_csv(
    inp: ExportInput, paths: Mapping[Role, Path] = PROJECTION_CSV
) -> dict[Role, Path]:
    """검토용 2027 예측 CSV를 쓴다 (모듈 docstring "검토용 CSV")."""
    target = int(inp.meta["target_season"])
    names = inp.tables.players[["player_id", "name"]]
    frames: dict[Role, pd.DataFrame] = {}
    # 두 역할을 모두 확인한 뒤에만 쓴다 (한 역할만 바뀐 CSV가 남지 않게)
    for role in c.ROLES:
        projections = _checked_projections(inp, role, target)
        _require_ids(projections["player_id"], names["player_id"], "players 표")
        frames[role] = projections.merge(names, on="player_id", how="left", validate="one_to_one")
    for role, frame in frames.items():
        paths[role].parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(paths[role], index=False, lineterminator="\n")
    return {role: paths[role] for role in c.ROLES}
