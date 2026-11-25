"""최종 검증(2026, 한 번)·2027 최종 예측·내보내기 입력 (SPEC F10·F11·N1, PLAN §2-3·§2-4). 레인 C가 구현한다.

최종 모델 (역할마다)
- 채택표에서 그 역할 지표(`EXT_ROLE_METRICS[role]`) 중 하나라도 True면 "extended", 아니면 "weighted".

평가 (`evaluate_target`, 가드 없음 — 가드는 명령 쪽)
- 역할마다 모델 "league"·"baseline"(`backtest.baseline_projections`), "weighted"
  (`models.extended.weighted_projection_for`), 그리고 최종 모델이 "extended"면 "extended"
  (`models.extended.extended_projection`)를 합쳐 `evaluate.evaluate`로 평가한다.
- 결과: 두 역할을 이은 `EVALUATIONS` 표, 키 순(안정 정렬).

최종 검증 (`run_final_test`) — PLAN §2-3 "파라미터 고정 절차"
1. `committed_head([params_path, adoption_path])`: 두 파일이 git에 추적되고 작업 트리·스테이징이 HEAD와
   같아야 한다. 아니면 FinalTestError(무엇이 문제인지 한국어로). 통과하면 HEAD 커밋 해시.
2. out_path가 이미 있고 rerun_reason이 없으면 FinalTestError("이미 한 번 돌렸다 … --rerun 사유").
3. `evaluate_target(…, FINAL_TEST_TARGET)`.
4. 기록(JSON, `export.dump_json`과 같은 직렬화 규칙이되 사람이 읽게 indent=1):
   {"target_season": 2026, "ran_at": 첫 실행 시각, "params_sha256", "adoption_sha256", "commit",
    "reruns": [{"ran_at", "reason", "commit"}…], "rows": [EVALUATIONS 행…(NaN → null)]}.
   다시 돌리면 ran_at은 첫 기록 값을 그대로 두고 reruns 끝에 이번 실행을 붙이고 rows를 새 결과로 바꾼다.
   반환은 이 기록(dict).

2027 예측 (`final_projections`): 역할마다 최종 모델의 목표 Y 예측 표 (그 모델 행만).

내보내기 입력 (`build_export_input`)
- evaluations = backtest_csv(`EVALUATIONS`, 2024·2025) 행 + 최종 검증 기록이 있으면 그 rows(2026).
- final_test = 기록이 있으면 {"target_season", "ran_at", "params_sha256", "reruns": [{"ran_at", "reason"}]}
  (스키마 ValidationReport.final_test 모양), 없으면 None.
- meta: schema_version = `WEB_SCHEMA_VERSION`, raw_collected_at = `raw_collected_at(raw_dir)`,
  generated_at = raw_collected_at(같은 원본이면 같은 바이트 — N1), params_version·params_sha256,
  target_season = Y, input_seasons = [Y−5 … Y−1], sources = `WEB_SOURCES`({name, url}), disclaimer = `WEB_DISCLAIMER`,
  counts = 역할 → 예측 표 행 수(예측 대상 수).

원본 수집 시각 (`raw_collected_at`): 원본 매니페스트(`collect.raw.load_manifest`) 항목의 받은 시각 중
가장 늦은 값, UTC "YYYY-MM-DDTHH:MM:SSZ". 매니페스트가 비었으면 ValueError.
"""

import json
import math
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mlbproj.backtest import baseline_projections
from mlbproj.collect.raw import load_manifest
from mlbproj.contracts import (
    EVALUATIONS,
    EXT_ROLE_METRICS,
    FINAL_TEST_TARGET,
    MANIFEST_NAME,
    ROLES,
    TARGET_SEASON,
    WEB_DISCLAIMER,
    WEB_ROUND,
    WEB_SCHEMA_VERSION,
    WEB_SOURCES,
    WINDOW,
    Column,
    ModelName,
    Role,
    check_frame,
)
from mlbproj.evaluate import evaluate
from mlbproj.export import ExportInput
from mlbproj.interim import read_table
from mlbproj.models import extended
from mlbproj.models.params import (
    Adoption,
    Params,
    adoption_sha256,
    load_adoption,
    load_params,
    params_sha256,
    validate_adoption,
)
from mlbproj.tables import Tables, cohort_for

# git 인자 목록 → 표준 출력. 실패하면 예외 (테스트는 가짜를 넣는다)
GitRunner = Callable[[Sequence[str]], str]

_STAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"  # UTC 시각 표기 (매니페스트·최종 검증 기록)
# strptime은 "2026-1-5T1:2:3Z"처럼 한 자리도 받으므로 자릿수는 정규식으로 따로 본다
_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECORD_INDENT = 1  # 최종 검증 기록은 사람이 읽는다
_RECORD_KEYS = frozenset(
    {"target_season", "ran_at", "params_sha256", "adoption_sha256", "commit", "reruns", "rows"}
)
_RERUN_KEYS = ("ran_at", "reason", "commit")
_KIND_DTYPES = {"int": "int64", "float": "float64", "str": "str", "bool": "bool"}
# git status가 알려 주는 커밋과 다른 점
_UNTRACKED = "추적되지 않는다"
_STAGED = "스테이징에 커밋하지 않은 변경이 있다"
_WORKTREE = "작업 트리에 커밋하지 않은 변경이 있다"
_STATUS_ORDER = (_UNTRACKED, _STAGED, _WORKTREE)


class FinalTestError(RuntimeError):
    """최종 검증을 돌릴 수 없는 상태 (커밋 안 됨, 이미 돌림 등)."""


def final_model_name(adoption: Adoption, role: Role) -> ModelName:
    """역할의 최종 모델 이름 (모듈 docstring "최종 모델").

    Raises:
        ValueError: 채택표가 형식을 어길 때 (`validate_adoption`).
    """
    validate_adoption(adoption)
    if any(adoption.adopt[metric] for metric in EXT_ROLE_METRICS[role]):
        return "extended"
    return "weighted"


# ---------------------------------------------------------------------------
# 커밋 확인
# ---------------------------------------------------------------------------


def _system_git(repo: Path) -> GitRunner:
    """`git -C repo …`를 실제로 부르는 실행기.

    상태 확인이 색인 파일을 고치지 않게 잠금을 끄고, 경로의 `*`·`?`·`:`을 패턴이 아니라 글자 그대로
    보게 한다(다른 파일이 같이 잡히지 않게).
    """

    def run(args: Sequence[str]) -> str:
        done = subprocess.run(
            ["git", "--no-optional-locks", "--literal-pathspecs", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return done.stdout

    return run


def _call_git(git: GitRunner, args: Sequence[str]) -> str:
    """git을 부른다. 실패(종료 코드·실행 파일 없음)는 FinalTestError로 바꾼다."""
    shown = " ".join(["git", *args])
    try:
        return git(args)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
        detail = stderr or f"종료 코드 {exc.returncode}"
        raise FinalTestError(f"git 확인에 실패했다 ({shown}): {detail}") from exc
    except OSError as exc:
        raise FinalTestError(f"git을 실행할 수 없다 ({shown}): {exc}") from exc


def _repo_relative(path: Path, repo: Path) -> str:
    """repo 기준 상대 경로(posix). 저장소 밖이면 FinalTestError."""
    try:
        rel = Path(os.path.relpath(path.resolve(), repo.resolve()))
    except ValueError as exc:  # 윈도우에서 드라이브가 다를 때
        raise FinalTestError(f"저장소 밖의 파일이다: {path} (저장소 {repo})") from exc
    if rel.parts[:1] == ("..",):
        raise FinalTestError(f"저장소 밖의 파일이다: {path} (저장소 {repo})")
    return rel.as_posix()


def _status_problems(porcelain: str) -> list[str]:
    """`git status --porcelain` 출력(XY 경로)에서 커밋과 다른 점을 한국어로 (`_STATUS_ORDER` 순)."""
    found: set[str] = set()
    for line in porcelain.splitlines():
        if line.startswith("??"):
            found.add(_UNTRACKED)
            continue
        index, worktree = line[:1], line[1:2]
        if index.strip():
            found.add(_STAGED)
        if worktree.strip():
            found.add(_WORKTREE)
    return [problem for problem in _STATUS_ORDER if problem in found]


def committed_head(paths: Sequence[Path], repo: Path, git: GitRunner | None = None) -> str:
    """paths가 모두 커밋된 그대로면 HEAD 커밋 해시. git=None이면 `git -C repo …`를 실제로 부른다.

    파일마다 ① 색인에 있는지(`ls-files`) ② 스테이징·작업 트리가 HEAD와 같은지(`status --porcelain`)를
    본다. 다른 파일의 변경은 따지지 않는다.

    Raises:
        FinalTestError: 추적되지 않거나, 작업 트리·스테이징에 변경이 있거나, git을 부를 수 없을 때.
    """
    runner = git if git is not None else _system_git(repo)
    head = _call_git(runner, ["rev-parse", "--verify", "HEAD"]).strip()
    if not head:
        raise FinalTestError(f"HEAD 커밋을 찾지 못했다 (저장소 {repo})")
    for path in paths:
        rel = _repo_relative(path, repo)
        if not _call_git(runner, ["ls-files", "--", rel]).strip():
            raise FinalTestError(f"git이 추적하지 않는 파일이다 — 커밋한 뒤 돌린다: {path}")
        problems = _status_problems(_call_git(runner, ["status", "--porcelain", "--", rel]))
        if problems:
            raise FinalTestError(
                f"커밋된 그대로가 아니다({', '.join(problems)}) — 커밋한 뒤 돌린다: {path}"
            )
    return head


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------


def evaluate_target(
    tables: Tables, params: Params, adoption: Adoption, target: int
) -> pd.DataFrame:
    """목표 시즌 하나의 평가표 (모듈 docstring "평가")."""
    frames: list[pd.DataFrame] = []
    for role in ROLES:
        projections = [
            baseline_projections(tables, role, target),
            extended.weighted_projection_for(tables, role, target, params),
        ]
        if final_model_name(adoption, role) == "extended":
            projections.append(extended.extended_projection(tables, role, target, params, adoption))
        frames.append(
            evaluate(
                pd.concat(projections, ignore_index=True),
                tables.seasons[role],
                cohort_for(tables, role, target),
                role,
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(list(EVALUATIONS.keys), kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 최종 검증 기록
# ---------------------------------------------------------------------------


def _stamp_problem(value: object) -> str | None:
    """UTC 시각 표기("YYYY-MM-DDTHH:MM:SSZ", 있는 날짜·시각)가 아니면 그 이유, 맞으면 None."""
    if not isinstance(value, str) or _STAMP.fullmatch(value) is None:
        return f'UTC "YYYY-MM-DDTHH:MM:SSZ" 형식이 아니다: {value!r}'
    try:
        datetime.strptime(value, _STAMP_FORMAT)
    except ValueError:  # 13월·25시처럼 없는 날짜·시각
        return f"없는 날짜·시각이다: {value!r}"
    return None


def _json_rows(evaluations: pd.DataFrame) -> list[dict[str, Any]]:
    """평가표 → JSON 행 (`EVALUATIONS` 열 순, NaN → None, 정수는 파이썬 int)."""
    rows = evaluations[EVALUATIONS.column_names].to_dict(orient="records")
    return [
        {str(k): None if isinstance(v, float) and math.isnan(v) else v for k, v in row.items()}
        for row in rows
    ]


def _dtype(col: Column) -> str:
    """계약 열의 pandas dtype (결측 허용 정수는 Int64)."""
    if col.kind == "int" and col.nullable:
        return "Int64"
    return _KIND_DTYPES[col.kind]


def _fits(value: object, col: Column) -> bool:
    """JSON 값 하나가 계약 열의 종류에 맞는지. bool은 정수로, 3.5는 정수로 치지 않는다."""
    if value is None:
        return col.nullable
    if col.kind == "int":
        return type(value) is int
    if col.kind == "float":
        # JSON의 NaN·Infinity는 json.loads가 받아 주므로 여기서 거른다
        return isinstance(value, int | float) and type(value) is not bool and math.isfinite(value)
    if col.kind == "str":
        return type(value) is str
    return type(value) is bool


def _evaluation_frame(rows: Sequence[object]) -> pd.DataFrame:
    """기록의 JSON 행 → `EVALUATIONS` 표. 모두 null인 실수 열도 float64가 된다.

    astype은 3.5를 3으로, true를 1로 조용히 바꾸므로 그 전에 행마다 키와 값의 종류를 확인한다.

    Raises:
        ValueError: 행의 키가 `EVALUATIONS` 열과 다르거나, 값의 종류·결측이 계약과 다를 때.
    """
    names = EVALUATIONS.column_names
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(names):
            keys = sorted(row) if isinstance(row, dict) else type(row).__name__
            raise ValueError(f"rows[{i}]의 키가 {EVALUATIONS.name} 열과 다르다: {keys}")
        bad = [
            f"{col.name}={row[col.name]!r}"
            for col in EVALUATIONS.columns
            if not _fits(row[col.name], col)
        ]
        if bad:
            raise ValueError(f"rows[{i}]의 값이 계약 종류와 다르다: {', '.join(bad)}")
    frame = pd.DataFrame(list(rows), columns=names)
    return frame.astype({col.name: _dtype(col) for col in EVALUATIONS.columns})


def _record_problem(record: object) -> str | None:
    """기록(rows 제외)의 형식 문제를 한국어로. 없으면 None (`run_final_test`가 쓰는 모양)."""
    if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
        keys = sorted(record) if isinstance(record, dict) else type(record).__name__
        return f"최상위 키가 {sorted(_RECORD_KEYS)}가 아니다: {keys}"
    if type(record["target_season"]) is not int or record["target_season"] != FINAL_TEST_TARGET:
        return f"target_season이 {FINAL_TEST_TARGET}이 아니다: {record['target_season']!r}"
    stamp = _stamp_problem(record["ran_at"])
    if stamp is not None:
        return f"ran_at — {stamp}"
    for key in ("params_sha256", "adoption_sha256"):
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
    return None


def _read_record(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """최종 검증 기록과 그 rows 표(`EVALUATIONS`, 목표 2026 행만)를 읽는다.

    Raises:
        ValueError: 읽을 수 없거나 형식이 틀릴 때 (무엇이 틀렸는지 적는다).
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"최종 검증 기록을 읽지 못했다: {path} ({exc})") from exc
    problem = _record_problem(record)
    if problem is None:
        try:
            rows = _evaluation_frame(record["rows"])
        except ValueError as exc:
            problem = str(exc)
        else:
            found = check_frame(rows, EVALUATIONS)
            late = sorted({int(t) for t in rows["target_season"]} - {FINAL_TEST_TARGET})
            if found:
                problem = "; ".join(found)
            elif late:
                problem = f"rows에 {FINAL_TEST_TARGET}이 아닌 목표 시즌 {late}이 있다"
    if problem is not None:
        raise ValueError(f"최종 검증 기록의 형식이 틀렸다 ({problem}): {path}")
    return record, rows


def _write_record(record: Mapping[str, Any], path: Path) -> None:
    """기록을 사람이 읽는 JSON으로 쓴다 (NaN 금지, 한국어 그대로, 원자적 교체)."""
    text = json.dumps(record, ensure_ascii=False, indent=_RECORD_INDENT, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _require_frozen_values(
    params: Params, adoption: Adoption, params_path: Path, adoption_path: Path
) -> tuple[str, str]:
    """넘긴 파라미터·채택표가 커밋된 파일의 값과 같은지 확인하고 두 sha256을 돌려준다.

    커밋 확인은 파일만 본다. 메모리의 다른 값(예: 기본값 v0)으로 돌면 "커밋된 파일로만 돈다"
    (`models.params`)가 깨지므로 값도 맞춰 본다.

    Raises:
        FinalTestError: 파일을 읽을 수 없거나 값이 다를 때.
    """
    given = (params_sha256(params), adoption_sha256(adoption))
    try:
        frozen = (
            params_sha256(load_params(params_path)),
            adoption_sha256(load_adoption(adoption_path)),
        )
    except (OSError, ValueError) as exc:
        raise FinalTestError(f"커밋된 파라미터·채택표를 읽지 못했다: {exc}") from exc
    for what, path, mine, theirs in (
        ("파라미터", params_path, given[0], frozen[0]),
        ("채택표", adoption_path, given[1], frozen[1]),
    ):
        if mine != theirs:
            raise FinalTestError(
                f"넘긴 {what}가 커밋된 파일과 다르다(sha256 {mine[:12]} ≠ {theirs[:12]}) — "
                f"최종 검증은 커밋된 파일 그대로 돈다: {path}"
            )
    return given


def run_final_test(
    tables: Tables,
    params: Params,
    adoption: Adoption,
    *,
    repo: Path,
    params_path: Path,
    adoption_path: Path,
    out_path: Path,
    now: str,
    rerun_reason: str | None = None,
    git: GitRunner | None = None,
) -> dict[str, Any]:
    """2026 최종 검증을 한 번 돌리고 기록한다 (모듈 docstring "최종 검증").

    모듈 docstring 순서에 더해 두 가지를 막는다 (SPEC D10·F10 — 모델은 2026 전에 고정한다).
    - 넘긴 params·adoption이 커밋된 파일(params_path·adoption_path)의 값과 다르면 돌지 않는다.
    - 다시 돌릴 때 params_sha256·adoption_sha256이 첫 기록과 다르면 돌지 않는다 — 2026을 본 뒤 바꾼
      모델로 다시 재면 최종 검증이 아니다. 그래서 기록의 두 sha256은 모든 실행에 같다.
    다시 돌릴 때 ran_at·commit은 첫 실행 값을 두고, 이번 커밋은 reruns 항목에 남는다.
    기록이 없는데 rerun_reason을 주면 돌지 않는다 — 사유를 남길 곳이 없고, out_path를 잘못 준 것일 수 있다.

    Args:
        now: 실행 시각 (UTC "YYYY-MM-DDTHH:MM:SSZ").
        rerun_reason: 다시 돌리는 사유 (빈 문자열은 사유 없음으로 본다).

    Raises:
        FinalTestError: 커밋 확인 실패, 넘긴 값이 커밋된 파일과 다를 때, 이미 돌렸는데 사유가 없을 때,
            기록이 없는데 사유를 줬을 때, 기존 기록이 형식을 어기거나 첫 실행과 모델이 다를 때.
        ValueError: now가 "YYYY-MM-DDTHH:MM:SSZ" 형식이 아닐 때.
    """
    stamp = _stamp_problem(now)
    if stamp is not None:
        raise ValueError(f"실행 시각(now) — {stamp}")
    head = committed_head([params_path, adoption_path], repo, git)
    sha_params, sha_adoption = _require_frozen_values(params, adoption, params_path, adoption_path)
    reason = (rerun_reason or "").strip()
    previous: dict[str, Any] | None = None
    if out_path.exists():
        if not reason:
            raise FinalTestError(
                f"이미 한 번 돌렸다({out_path}) — 2026 최종 검증은 한 번만이다. "
                "꼭 다시 돌려야 하면 --rerun 사유를 준다"
            )
        try:
            previous, _ = _read_record(out_path)
        except ValueError as exc:
            raise FinalTestError(str(exc)) from exc
        for what, key, value in (
            ("파라미터", "params_sha256", sha_params),
            ("채택표", "adoption_sha256", sha_adoption),
        ):
            if previous[key] != value:
                raise FinalTestError(
                    f"첫 실행과 {what}가 다르다({key} {previous[key][:12]} → {value[:12]}) — "
                    "2026 결과를 본 뒤 바꾼 모델로 다시 재면 최종 검증이 아니다 (SPEC D10·F10)"
                )
    elif reason:
        raise FinalTestError(
            f"다시 돌린다는 사유(--rerun)를 줬는데 기록이 없다: {out_path} — "
            "첫 실행이면 --rerun 없이 돌리고, 다시 돌리는 것이면 기록 경로(--out)를 확인한다"
        )

    rows = _json_rows(evaluate_target(tables, params, adoption, FINAL_TEST_TARGET))
    if previous is None:
        ran_at, commit, reruns = now, head, []
    else:
        ran_at, commit = previous["ran_at"], previous["commit"]
        reruns = [*previous["reruns"], {"ran_at": now, "reason": reason, "commit": head}]
    record: dict[str, Any] = {
        "target_season": FINAL_TEST_TARGET,
        "ran_at": ran_at,
        "params_sha256": sha_params,
        "adoption_sha256": sha_adoption,
        "commit": commit,
        "reruns": reruns,
        "rows": rows,
    }
    _write_record(record, out_path)
    return record


# ---------------------------------------------------------------------------
# 2027 예측·내보내기 입력
# ---------------------------------------------------------------------------


def final_projections(
    tables: Tables, params: Params, adoption: Adoption, target: int = TARGET_SEASON
) -> dict[Role, pd.DataFrame]:
    """역할마다 최종 모델의 목표 Y 예측 표."""
    out: dict[Role, pd.DataFrame] = {}
    for role in ROLES:
        model = final_model_name(adoption, role)
        if model == "extended":
            frame = extended.extended_projection(tables, role, target, params, adoption)
        else:
            frame = extended.weighted_projection_for(tables, role, target, params)
        out[role] = frame[frame["model"] == model].reset_index(drop=True)
    return out


def raw_collected_at(raw_dir: Path) -> str:
    """원본 수집 시각 (모듈 docstring "원본 수집 시각").

    Raises:
        ValueError: 매니페스트가 없거나 비었을 때, 받은 시각의 형식이 틀릴 때.
    """
    manifest = load_manifest(raw_dir)
    if not manifest:
        raise ValueError(
            f"원본 매니페스트가 비었다 (collect를 먼저 돌린다): {raw_dir / MANIFEST_NAME}"
        )
    for key in sorted(manifest):
        problem = _stamp_problem(manifest[key].fetched_at)
        if problem is not None:
            raise ValueError(f"원본 매니페스트 {key}의 받은 시각(fetched_at) — {problem}")
    # 고정 자릿수 UTC 표기라 문자열 최댓값이 가장 늦은 시각이다
    return max(entry.fetched_at for entry in manifest.values())


def _projection_target(projections: Mapping[Role, pd.DataFrame]) -> int:
    """예측 표들의 목표 시즌 Y (두 역할이 같은 Y 하나여야 한다)."""
    if set(projections) != set(ROLES):
        raise ValueError(f"예측 표는 역할 {list(ROLES)}마다 하나: {sorted(projections)}")
    targets = sorted({int(t) for role in ROLES for t in projections[role]["target_season"]})
    if len(targets) != 1:
        raise ValueError(f"예측 표의 목표 시즌이 하나가 아니다: {targets}")
    return targets[0]


def first_run_path(final_test_path: Path) -> Path:
    """다시 돌린 최종 검증의 첫 실행 보관본 경로 (`final_test_2026.json` → `final_test_2026_run1.json`)."""
    return final_test_path.with_name(f"{final_test_path.stem}_run1.json")


def _first_run(
    record: Mapping[str, Any], final_test_path: Path, adoption: Adoption
) -> dict[str, Any] | None:
    """첫 실행 요약 (SPEC D31) — 다시 돌린 기록이고 보관본이 있으면 {ran_at, mae: 지표 → 최종 모델 "all" mae}.

    역할마다 최종 모델(`final_model_name`)의 행만 쓴다. 다시 돌리지 않았거나 보관본이 없으면 None.
    """
    path = first_run_path(final_test_path)
    if not record["reruns"] or not path.exists():
        return None
    first, rows = _read_record(path)
    final = {role: final_model_name(adoption, role) for role in ROLES}
    mine = rows[(rows["group"] == "all") & (rows["model"] == rows["role"].map(final))]
    mae = {
        str(metric): (None if pd.isna(value) else round(float(value), WEB_ROUND["mae"]))
        for metric, value in zip(mine["metric"], mine["mae"], strict=True)
    }
    return {"ran_at": first["ran_at"], "mae": mae}


def _final_test_summary(
    record: Mapping[str, Any], first_run: dict[str, Any] | None
) -> dict[str, Any]:
    """검사한 기록 → 스키마 ValidationReport.final_test 모양 (commit·adoption_sha256은 내보내지 않는다)."""
    return {
        "target_season": record["target_season"],
        "ran_at": record["ran_at"],
        "params_sha256": record["params_sha256"],
        "reruns": [
            {"ran_at": rerun["ran_at"], "reason": rerun["reason"]} for rerun in record["reruns"]
        ],
        "first_run": first_run,
    }


def build_export_input(
    tables: Tables,
    params: Params,
    adoption: Adoption,
    projections: dict[Role, pd.DataFrame],
    *,
    raw_dir: Path,
    backtest_csv: Path,
    final_test_path: Path,
) -> ExportInput:
    """내보내기 입력 한 묶음 (모듈 docstring "내보내기 입력").

    Y는 예측 표의 목표 시즌이다.

    Raises:
        ValueError: 예측 표의 역할·목표 시즌·모델이 맞지 않거나, 과거 검증표에 최종 검증 시즌 이후
            행이 있거나, 최종 검증 기록이 형식을 어기거나, 합친 평가표가 계약을 어길 때.
    """
    target = _projection_target(projections)
    model: dict[Role, str] = {role: final_model_name(adoption, role) for role in ROLES}
    for role in ROLES:
        found = sorted(set(projections[role]["model"]))
        if found != [model[role]]:
            raise ValueError(f"{role} 예측 표의 모델 {found}이 최종 모델 {model[role]}과 다르다")

    evaluations = read_table(EVALUATIONS, backtest_csv)
    late = sorted({int(t) for t in evaluations["target_season"] if t >= FINAL_TEST_TARGET})
    if late:
        raise ValueError(
            f"과거 검증표 {backtest_csv}에 {late} 행이 있다 — 2026은 최종 검증 기록에서만 온다"
        )
    final_test: dict[str, Any] | None = None
    if final_test_path.exists():
        record, rows = _read_record(final_test_path)  # 목표 2026 행만, 계약 종류 그대로
        # 2026을 잰 모델과 2027을 내보내는 모델이 같아야 검증 페이지와 예측표가 짝이 맞는다
        expected = {
            "params_sha256": params_sha256(params),
            "adoption_sha256": adoption_sha256(adoption),
        }
        differ = [key for key, value in expected.items() if record.get(key) != value]
        if differ:
            raise ValueError(
                f"최종 검증 기록 {final_test_path}의 {differ}가 지금 내보내는 파라미터·채택표와 다르다 — "
                "2026을 잰 모델과 2027 예측 모델이 섞인다"
            )
        evaluations = pd.concat([evaluations, rows], ignore_index=True)
        final_test = _final_test_summary(record, _first_run(record, final_test_path, adoption))
    evaluations = evaluations.sort_values(list(EVALUATIONS.keys), kind="mergesort").reset_index(
        drop=True
    )
    problems = check_frame(evaluations, EVALUATIONS)
    if problems:
        raise ValueError("평가표 계약 위반: " + "; ".join(problems))

    collected = raw_collected_at(raw_dir)
    meta: dict[str, Any] = {
        "schema_version": WEB_SCHEMA_VERSION,
        "generated_at": collected,  # 실행 시각이 아니라 원본 시각 — 같은 원본이면 같은 바이트 (N1)
        "raw_collected_at": collected,
        "params_version": params.version,
        "params_sha256": params_sha256(params),
        "target_season": target,
        "input_seasons": list(range(target - WINDOW, target)),
        "sources": [{"name": name, "url": url} for name, url in WEB_SOURCES],
        "disclaimer": WEB_DISCLAIMER,
        # 예측 대상 수 — 첫 화면 칩이 투수 표를 받기 전에도 쓴다 (SA-1, N6 지연 로딩)
        "counts": {role: len(projections[role]) for role in ROLES},
    }
    return ExportInput(
        tables=tables,
        projections=dict(projections),
        model=model,
        evaluations=evaluations,
        final_test=final_test,
        meta=meta,
    )
