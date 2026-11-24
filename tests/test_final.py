"""최종 검증·2027 최종 예측·내보내기 입력(final.py) 테스트 (SPEC F9~F11·N1·N2, PLAN §2-3).

2026 규칙: 이 파일은 손으로 만든 합성 표만 쓴다. 실데이터(`data/`)를 읽지 않으며, 목표 2026 계산은
합성 세계에서만 한다(실제 최종 검증은 메인 세션이 한 번만 돌린다).

바꿔 끼우는 곳 — final.py는 다른 레인을 **모듈 속성으로** 불러야 한다
- `mlbproj.models.extended.weighted_projection_for`·`extended_projection` (레인 A가 아직 미구현일 수 있다).
  final.py는 `from mlbproj.models import extended` 뒤 `extended.extended_projection(...)`처럼 부른다.
- `mlbproj.final.evaluate_target` — run_final_test는 모듈 전역 이름으로 부른다(여기서 가짜로 바꾼다).
- 내보내기 묶음은 `mlbproj.export.ExportInput`(모듈 속성)이다.
기준선("league"·"baseline", `backtest.baseline_projections`)과 평가(`evaluate.evaluate`)는 진짜를 쓴다.

합성 세계 (목표 2025, `_world`)
- 타자 1: 2022 PA 500, 2023 PA 520, 2024 PA 600·HR 30, 2025 PA 550·HR 25 (n_seasons 3)
- 타자 2: 2024 PA 300·HR 6, 2025 PA 250·HR 5 (n_seasons 1)
- 투수 11: 2023 아웃 450, 2024 아웃 300(100 IP), 2025 아웃 270(90 IP) (n_seasons 2)
- 투수 12: 2024 아웃 150(50 IP), 2025 아웃 135(45 IP) (n_seasons 1)
기준 모델 예측 = 2024 값: 타자 PA 600·300, HR 600×(30/600)=30·300×(6/300)=6, 투수 IP 100·50.
"""

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlbproj import backtest, export, final
from mlbproj import contracts as c
from mlbproj.collect.raw import save_raw
from mlbproj.contracts import Role
from mlbproj.final import FinalTestError
from mlbproj.models import extended
from mlbproj.models.league import league_batting, league_pitching
from mlbproj.models.params import (
    DEFAULT_PARAMS,
    NO_ADOPTION,
    Adoption,
    Params,
    adoption_sha256,
    params_sha256,
    save_adoption,
    save_params,
)
from mlbproj.schema import load_schema, validate
from mlbproj.tables import Tables

ROOT = Path(__file__).resolve().parents[1]
TARGET = 2025  # evaluate_target 시험의 목표 (가드 없음 — 아무 목표나 받는다)
PARAMS: Params = replace(DEFAULT_PARAMS, version="v1")
HEAD = "0123456789abcdef0123456789abcdef01234567"
HEAD2 = "89abcdef0123456789abcdef0123456789abcdef"
NOW1 = "2026-11-25T01:02:03Z"
NOW2 = "2026-11-26T04:05:06Z"
NOW3 = "2026-11-27T07:08:09Z"
RECORD_KEYS = {
    "target_season",
    "ran_at",
    "params_sha256",
    "adoption_sha256",
    "commit",
    "reruns",
    "rows",
}
_HANGUL = re.compile("[가-힣]")
_KEYS = list(c.EVALUATIONS.keys)


def _adoption(*metrics: str) -> Adoption:
    """metrics만 True인 채택표."""
    return Adoption(version="v1", adopt={m: m in metrics for m in c.EXT_METRICS})


ADOPTION = _adoption("ops")  # 타자 최종 모델 = extended, 투수 = weighted

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


def _bat_season(pid: int, season: int, pa: int, hr: int, ops: float) -> dict[str, Any]:
    # PA = AB + BB (HBP·SF·SH 0). 안타는 AB의 1/4, 2루타는 안타의 1/5
    ab = pa - 50
    h = ab // 4
    obp = round(ops * 0.45, 3)
    return {
        "player_id": pid,
        "season": season,
        "pa": pa,
        "ab": ab,
        "h": h,
        "b2": h // 5,
        "hr": hr,
        "bb": pa - ab,
        "so": pa // 5,
        "obp": obp,
        "slg": round(ops - obp, 3),
        "ops": ops,
    }


def _pit_season(
    pid: int, season: int, outs: int, bf: int, er: int, so: int, bb: int, hr: int, g: int, gs: int
) -> dict[str, Any]:
    return {
        "player_id": pid,
        "season": season,
        "outs": outs,
        "bf": bf,
        "er": er,
        "so": so,
        "bb": bb,
        "hr": hr,
        "g": g,
        "gs": gs,
        "h": bf // 4,
    }


def _cohort_row(
    pid: int, role: Role, window_first: int, n_seasons: int, sample: int, sample_outs: int | None
) -> dict[str, Any]:
    return {
        "player_id": pid,
        "role": role,
        "target_season": TARGET,
        "window_first": window_first,
        "window_last": TARGET - 1,
        "n_window": TARGET - window_first,
        "input_first": window_first,
        "input_last": TARGET - 1,
        "n_seasons": n_seasons,
        "played_prev": True,
        "sample": sample,
        "sample_outs": sample_outs,
    }


def _world() -> Tables:
    """목표 2025의 작은 합성 세계 (모듈 docstring)."""
    batting = _frame(
        c.BATTING_SEASONS,
        [
            _bat_season(1, 2022, 500, 20, 0.750),
            _bat_season(1, 2023, 520, 22, 0.760),
            _bat_season(1, 2024, 600, 30, 0.800),
            _bat_season(1, 2025, 550, 25, 0.820),
            _bat_season(2, 2024, 300, 6, 0.700),
            _bat_season(2, 2025, 250, 5, 0.680),
        ],
    )
    pitching = _frame(
        c.PITCHING_SEASONS,
        [
            _pit_season(11, 2023, 450, 620, 50, 120, 40, 12, 25, 25),
            _pit_season(11, 2024, 300, 410, 40, 90, 30, 10, 20, 20),
            _pit_season(11, 2025, 270, 370, 30, 80, 25, 8, 18, 18),
            _pit_season(12, 2024, 150, 210, 20, 50, 20, 5, 50, 0),
            _pit_season(12, 2025, 135, 190, 15, 45, 15, 4, 45, 0),
        ],
    )
    cohorts = _frame(
        c.COHORTS,
        [
            _cohort_row(1, "bat", 2022, 3, 1620, None),  # 500 + 520 + 600
            _cohort_row(2, "bat", 2024, 1, 300, None),
            _cohort_row(11, "pit", 2023, 2, 1030, 750),  # BF 620 + 410, 아웃 450 + 300
            _cohort_row(12, "pit", 2024, 1, 210, 150),
        ],
    )
    return Tables(
        seasons={"bat": batting, "pit": pitching},
        cohorts=cohorts,
        league={"bat": league_batting(batting), "pit": league_pitching(pitching)},
        debuts={},
    )


def _projection(role: Role, target: int, model: str, ids: Sequence[int]) -> pd.DataFrame:
    """예측 표 하나 (`BAT_PROJECTIONS`·`PIT_PROJECTIONS`). 값은 선수 ID로 구분되게 넣는다."""
    rows: list[dict[str, Any]] = []
    for pid in ids:
        base: dict[str, Any] = {
            "player_id": pid,
            "role": role,
            "target_season": target,
            "model": model,
            "input_first": target - 3,
            "input_last": target - 1,
            "n_seasons": 3,
        }
        if role == "bat":
            base.update(
                {
                    "sample_pa": 1500 + pid,
                    "pa": 500.0 + pid,
                    "hr": 20.0 + pid,
                    "obp": 0.330,
                    "slg": 0.420,
                    "ops": 0.750,
                    "lg_ops": 0.720,
                }
            )
        else:
            base.update(
                {
                    "sample_bf": 1000 + pid,
                    "sample_outs": 700 + pid,
                    "ip": 100.0 + pid,
                    "era": 4.0,
                    "k_bb_pct": 14.0,
                    "lg_era": 4.2,
                    "lg_k_bb_pct": 13.5,
                }
            )
        rows.append(base)
    spec = c.BAT_PROJECTIONS if role == "bat" else c.PIT_PROJECTIONS
    return _frame(spec, rows)


def _evaluations(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """평가표 (`EVALUATIONS`). 빠진 MAE 열은 결측, excluded 기본 0."""
    return _frame(c.EVALUATIONS, rows)


def _eval_row(
    target: int,
    role: str,
    model: str,
    metric: str,
    n: int,
    mae: float | None,
    mae_weighted: float | None,
    baseline_mae_league: float | None,
    baseline_mae_prev: float | None,
    excluded: int,
    group: str = "all",
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
        "baseline_mae_league": baseline_mae_league,
        "baseline_mae_prev": baseline_mae_prev,
        "excluded": excluded,
    }


def _no_nan(text: str) -> Any:
    """NaN·Infinity가 있으면 실패하는 JSON 읽기."""

    def reject(token: str) -> Any:
        raise AssertionError(f"JSON에 {token}이 들어 있다")

    return json.loads(text, parse_constant=reject)


# ---------------------------------------------------------------------------
# git 가짜
# ---------------------------------------------------------------------------

_SUPPORTED_GIT = (
    "rev-parse HEAD | --verify HEAD | --show-toplevel",
    "ls-files [--error-unmatch] [-o|--others] [-m|--modified] [-s|--stage] [--] paths",
    "status --porcelain[=v1] | -s | --short [-z] [-b] [-uno|--untracked-files=…] [--] [paths]",
    "diff [--cached|--staged] [--quiet|--exit-code] [--name-only|--name-status] [HEAD] [--] [paths]",
    "diff-index [--cached] [--quiet] [--name-only|--name-status] HEAD [--] [paths]",
    "cat-file -e|-t HEAD:path",
    "ls-tree [--name-only] [-r] HEAD [--] [paths]",
)
_IN_HEAD = {"clean", "modified", "staged"}
_IN_INDEX = {"clean", "modified", "staged", "added"}
_PORCELAIN = {"modified": " M", "staged": "M ", "added": "A ", "untracked": "??"}
_GLOBAL_FLAGS = {"--no-pager", "--no-optional-locks", "--literal-pathspecs", "-P"}


class FakeGit:
    """`final.GitRunner` 가짜 — 파일 상태표로 git 출력을 흉내 낸다.

    상태: "clean"(HEAD·스테이징·작업 트리 같음), "modified"(작업 트리만 바뀜), "staged"(스테이징이
    HEAD와 다름), "added"(스테이징에만 있고 HEAD에 없음), "untracked"(추적 안 됨). 상태표에 없는 경로는
    "untracked"로 본다. 인자 앞의 `git`·`-C <dir>`·`-c k=v`는 무시한다. 경로는 절대 경로(repo 아래)나
    repo 기준 상대 경로 모두 받는다. 지원하는 명령은 `_SUPPORTED_GIT`이고, 모르는 명령은 `unsupported`에
    남긴 뒤 AssertionError를 낸다. 실제 git처럼 `--error-unmatch`·`--quiet`·`--exit-code`·`cat-file -e`
    실패는 CalledProcessError다.
    """

    def __init__(
        self,
        repo: Path,
        states: Mapping[str, str],
        *,
        head: str = HEAD,
        fail: BaseException | None = None,
        log: list[str] | None = None,
    ) -> None:
        self.repo = repo
        self.states = dict(states)
        self.head = head
        self.fail = fail
        self.log = log
        self.calls: list[list[str]] = []
        self.unsupported: list[list[str]] = []

    def __call__(self, args: Sequence[str]) -> str:
        argv = [str(a) for a in args]
        self.calls.append(argv)
        if self.log is not None:
            self.log.append("git")
        if self.fail is not None:
            raise self.fail
        argv = self._strip_globals(argv)
        if not argv:
            return self._unsupported(argv)
        command, rest = argv[0], argv[1:]
        if "--" in rest:
            cut = rest.index("--")
            before, after = rest[:cut], rest[cut + 1 :]
        else:
            before, after = rest, []
        opts = [a for a in before if a.startswith("-")]
        positional = [a for a in before if not a.startswith("-")] + after
        handler = {
            "rev-parse": self._rev_parse,
            "ls-files": self._ls_files,
            "status": self._status,
            "diff": self._diff,
            "diff-index": self._diff,
            "cat-file": self._cat_file,
            "ls-tree": self._ls_tree,
        }.get(command)
        if handler is None:
            return self._unsupported(argv)
        return handler(argv, command, opts, positional)

    # -- 도우미 --

    def _strip_globals(self, argv: list[str]) -> list[str]:
        if argv[:1] == ["git"]:
            argv = argv[1:]
        while argv:
            if argv[0] in ("-C", "-c"):
                argv = argv[2:]
            elif argv[0] in _GLOBAL_FLAGS or argv[0].startswith(("--git-dir=", "--work-tree=")):
                argv = argv[1:]
            else:
                break
        return argv

    def _unsupported(self, argv: list[str]) -> str:
        self.unsupported.append(argv)
        raise AssertionError(f"FakeGit가 모르는 git 명령: {argv} — 지원: {_SUPPORTED_GIT}")

    def _fail(self, argv: list[str], code: int, stderr: str = "") -> subprocess.CalledProcessError:
        return subprocess.CalledProcessError(code, ["git", *argv], output="", stderr=stderr)

    def _rel(self, arg: str) -> str:
        path = Path(arg)
        if path.is_absolute():
            for base in (self.repo, self.repo.resolve()):
                for candidate in (path, path.resolve()):
                    try:
                        return candidate.relative_to(base).as_posix()
                    except ValueError:
                        continue
        return path.as_posix()

    def _state(self, rel: str) -> str:
        return self.states.get(rel, "untracked")

    def _select(self, paths: Sequence[str]) -> list[str]:
        return [self._rel(p) for p in paths] if paths else sorted(self.states)

    def _is_rev(self, arg: str) -> bool:
        return arg in ("HEAD", "@", self.head)

    @staticmethod
    def _lines(items: Sequence[str], sep: str = "\n") -> str:
        return "".join(f"{item}{sep}" for item in items)

    # -- 명령 --

    def _rev_parse(self, argv: list[str], _cmd: str, opts: list[str], pos: list[str]) -> str:
        if "--show-toplevel" in opts:
            return f"{self.repo}\n"
        if any(self._is_rev(a.removesuffix("^{commit}")) for a in pos):
            return f"{self.head}\n"
        return self._unsupported(argv)

    def _ls_files(self, argv: list[str], _cmd: str, opts: list[str], pos: list[str]) -> str:
        selected = self._select(pos)
        if "--error-unmatch" in opts:
            missing = [p for p in selected if self._state(p) not in _IN_INDEX]
            if missing:
                raise self._fail(
                    argv,
                    1,
                    f"error: pathspec '{missing[0]}' did not match any file(s) known to git",
                )
        if "-o" in opts or "--others" in opts:
            out = [p for p in selected if self._state(p) == "untracked"]
        elif "-m" in opts or "--modified" in opts:
            out = [p for p in selected if self._state(p) == "modified"]
        else:
            out = [p for p in selected if self._state(p) in _IN_INDEX]
        if "-s" in opts or "--stage" in opts:
            out = [f"100644 {'0' * 40} 0\t{p}" for p in out]
        return self._lines(out)

    def _status(self, argv: list[str], _cmd: str, opts: list[str], pos: list[str]) -> str:
        short = {"--porcelain", "--porcelain=v1", "-s", "--short"}
        if not short.intersection(opts):
            return self._unsupported(argv)
        hide_untracked = "-uno" in opts or "--untracked-files=no" in opts
        lines = []
        for path in self._select(pos):
            code = _PORCELAIN.get(self._state(path))
            if code is None or (code == "??" and hide_untracked):
                continue
            lines.append(f"{code} {path}")
        if "-b" in opts or "--branch" in opts:
            lines.insert(0, "## main")
        return self._lines(lines, "\0" if "-z" in opts else "\n")

    def _diff(self, argv: list[str], cmd: str, opts: list[str], pos: list[str]) -> str:
        cached = "--cached" in opts or "--staged" in opts
        revs = [a for a in pos if self._is_rev(a)]
        paths = [a for a in pos if not self._is_rev(a)]
        if cmd == "diff-index" and not revs:
            return self._unsupported(argv)
        if cached:  # 스테이징 ↔ HEAD
            changed_states = {"staged", "added"}
        elif revs:  # 작업 트리 ↔ HEAD
            changed_states = {"modified", "staged", "added"}
        else:  # 작업 트리 ↔ 스테이징
            changed_states = {"modified"}
        changed = [p for p in self._select(paths) if self._state(p) in changed_states]
        if changed and ("--quiet" in opts or "--exit-code" in opts):
            raise self._fail(argv, 1)
        if "--quiet" in opts:
            return ""
        if "--name-only" in opts:
            return self._lines(changed)
        if "--name-status" in opts:
            return self._lines(
                [f"{'A' if self._state(p) == 'added' else 'M'}\t{p}" for p in changed]
            )
        if cmd == "diff-index":
            return self._lines([f":100644 100644 {'0' * 40} {'1' * 40} M\t{p}" for p in changed])
        return self._lines([f"diff --git a/{p} b/{p}" for p in changed])

    def _cat_file(self, argv: list[str], _cmd: str, opts: list[str], pos: list[str]) -> str:
        if not ("-e" in opts or "-t" in opts) or len(pos) != 1 or ":" not in pos[0]:
            return self._unsupported(argv)
        rev, path = pos[0].split(":", 1)
        known = _IN_INDEX if rev == "" else _IN_HEAD if self._is_rev(rev) else None
        if known is None:
            return self._unsupported(argv)
        if self._state(self._rel(path)) not in known:
            raise self._fail(argv, 1 if "-e" in opts else 128)
        return "" if "-e" in opts else "blob\n"

    def _ls_tree(self, argv: list[str], _cmd: str, opts: list[str], pos: list[str]) -> str:
        revs = [a for a in pos if self._is_rev(a)]
        if not revs:
            return self._unsupported(argv)
        paths = [a for a in pos if not self._is_rev(a)]
        out = [p for p in self._select(paths) if self._state(p) in _IN_HEAD]
        if "--name-only" in opts or "--name-status" in opts:
            return self._lines(out)
        return self._lines([f"100644 blob {'0' * 40}\t{p}" for p in out])


PARAMS_REL = "config/params_v1.json"
ADOPTION_REL = "config/extended_v1.json"
CLEAN = {PARAMS_REL: "clean", ADOPTION_REL: "clean"}


def _repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    """저장소 자리와 파라미터·채택표 파일 (git 가짜와 함께 쓴다)."""
    repo = tmp_path / "repo"
    params_path = repo / PARAMS_REL
    adoption_path = repo / ADOPTION_REL
    save_params(PARAMS, params_path)
    save_adoption(ADOPTION, adoption_path)
    return repo, params_path, adoption_path


def _head(paths: Sequence[Path], repo: Path, git: FakeGit) -> str:
    """committed_head를 부르고, 가짜가 모르는 git 명령을 썼으면 그 명령을 보여 주며 실패한다."""
    try:
        return final.committed_head(paths, repo, git)
    except FinalTestError:
        if git.unsupported:
            pytest.fail(f"FakeGit가 모르는 git 명령을 썼다: {git.unsupported}")
        raise


# ---------------------------------------------------------------------------
# 최종 모델 이름
# ---------------------------------------------------------------------------


def test_final_final_model_name_none_adopted_is_weighted() -> None:
    for role in c.ROLES:
        assert final.final_model_name(NO_ADOPTION, role) == "weighted"


@pytest.mark.parametrize("metric", c.EXT_METRICS)
def test_final_final_model_name_follows_role_metrics(metric: str) -> None:
    # 지표 하나만 채택하면 그 지표의 역할만 extended, 다른 역할은 weighted
    adoption = _adoption(metric)
    owner = next(role for role in c.ROLES if metric in c.EXT_ROLE_METRICS[role])
    for role in c.ROLES:
        expected = "extended" if role == owner else "weighted"
        assert final.final_model_name(adoption, role) == expected


def test_final_final_model_name_all_adopted_is_extended() -> None:
    adoption = _adoption(*c.EXT_METRICS)
    assert [final.final_model_name(adoption, role) for role in c.ROLES] == ["extended"] * 2


# ---------------------------------------------------------------------------
# 커밋 확인 (git 가짜)
# ---------------------------------------------------------------------------


def test_final_committed_head_clean_returns_head(tmp_path: Path) -> None:
    repo, params_path, adoption_path = _repo(tmp_path)
    git = FakeGit(repo, CLEAN)
    assert _head([params_path, adoption_path], repo, git) == HEAD  # 줄바꿈 없이
    assert git.calls, "git 가짜를 한 번도 부르지 않았다 (git 인자를 써야 한다)"


def test_final_committed_head_ignores_unrelated_changes(tmp_path: Path) -> None:
    # 두 파일만 본다 — 다른 파일이 바뀌었거나 추적되지 않아도 통과한다
    repo, params_path, adoption_path = _repo(tmp_path)
    states = {**CLEAN, "README.md": "modified", "notes.txt": "untracked", "web/x.ts": "staged"}
    git = FakeGit(repo, states)
    assert _head([params_path, adoption_path], repo, git) == HEAD


@pytest.mark.parametrize(
    ("rel", "state"),
    [
        (PARAMS_REL, "untracked"),  # 커밋한 적 없음
        (ADOPTION_REL, "untracked"),
        (ADOPTION_REL, "modified"),  # 작업 트리만 바뀜
        (PARAMS_REL, "staged"),  # git add 뒤 커밋 안 함
        (ADOPTION_REL, "added"),  # 새 파일을 스테이징만 함
    ],
)
def test_final_committed_head_rejects_uncommitted(tmp_path: Path, rel: str, state: str) -> None:
    repo, params_path, adoption_path = _repo(tmp_path)
    git = FakeGit(repo, {**CLEAN, rel: state})
    with pytest.raises(FinalTestError) as info:
        final.committed_head([params_path, adoption_path], repo, git)
    assert git.unsupported == [], f"FakeGit가 모르는 git 명령: {git.unsupported}"
    message = str(info.value)
    assert _HANGUL.search(message), message
    assert Path(rel).name in message  # 어느 파일이 문제인지 적는다


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(
            128, ["git", "rev-parse", "HEAD"], stderr="fatal: not a git repository"
        ),
        FileNotFoundError(2, "No such file or directory", "git"),
    ],
    ids=["git-error", "git-missing"],
)
def test_final_committed_head_rejects_git_failure(tmp_path: Path, error: BaseException) -> None:
    repo, params_path, adoption_path = _repo(tmp_path)
    git = FakeGit(repo, CLEAN, fail=error)
    with pytest.raises(FinalTestError) as info:
        final.committed_head([params_path, adoption_path], repo, git)
    assert _HANGUL.search(str(info.value)), str(info.value)


def _real_git(repo: Path, *args: str) -> str:
    """테스트용 저장소에 git을 부른다 (전역 훅·서명 설정을 끈다)."""
    done = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="git이 없다")
def test_final_committed_head_real_repository(tmp_path: Path) -> None:
    """git=None이면 실제 git을 부른다 — 버리는 저장소로 확인한다."""
    repo, params_path, adoption_path = _repo(tmp_path)
    _real_git(repo, "init", "-q")
    _real_git(repo, "add", PARAMS_REL)
    _real_git(repo, "commit", "-q", "--no-verify", "-m", "params")
    paths = [params_path, adoption_path]

    # 채택표가 아직 추적되지 않는다
    with pytest.raises(FinalTestError):
        final.committed_head(paths, repo)

    _real_git(repo, "add", ADOPTION_REL)
    _real_git(repo, "commit", "-q", "--no-verify", "-m", "adoption")
    head = _real_git(repo, "rev-parse", "HEAD").strip()
    assert final.committed_head(paths, repo) == head

    # 커밋 뒤 작업 트리에서 바꾸면 거부한다
    adoption_path.write_text(adoption_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(FinalTestError) as info:
        final.committed_head(paths, repo)
    assert _HANGUL.search(str(info.value)), str(info.value)


# ---------------------------------------------------------------------------
# 평가 (evaluate_target) — 기준선·평가는 진짜, 가중·확장은 가짜
# ---------------------------------------------------------------------------


class ModelFakes:
    """가중·확장 예측 가짜. 기준 모델 예측을 이름만 바꾸고 출전량을 옮긴다.

    - weighted: 타자 PA − 20 (580·280), 투수 IP − 5 (95·45)
    - extended: 타자 PA − 50 (550·250), 투수 IP × 0.9 (90·45)
    HR·비율 열은 기준 모델 값 그대로다.
    """

    def __init__(self) -> None:
        self.weighted_calls: list[tuple[Role, int, Params]] = []
        self.extended_calls: list[tuple[Role, int, Params, Adoption]] = []

    @staticmethod
    def _from_baseline(tables: Tables, role: Role, target: int, model: str) -> pd.DataFrame:
        base = backtest.baseline_projections(tables, role, target)
        out = base[base["model"] == "baseline"].copy()
        out["model"] = model
        return out.reset_index(drop=True)

    def weighted(self, tables: Tables, role: Role, target: int, params: Params) -> pd.DataFrame:
        self.weighted_calls.append((role, target, params))
        out = self._from_baseline(tables, role, target, "weighted")
        if role == "bat":
            out["pa"] = out["pa"] - 20.0
        else:
            out["ip"] = out["ip"] - 5.0
        return out

    def extended(
        self, tables: Tables, role: Role, target: int, params: Params, adoption: Adoption
    ) -> pd.DataFrame:
        self.extended_calls.append((role, target, params, adoption))
        out = self._from_baseline(tables, role, target, "extended")
        if role == "bat":
            out["pa"] = out["pa"] - 50.0
        else:
            out["ip"] = out["ip"] * 0.9
        return out

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(extended, "weighted_projection_for", self.weighted)
        monkeypatch.setattr(extended, "extended_projection", self.extended)


def _mae(out: pd.DataFrame, role: str, model: str, metric: str, group: str = "all") -> float:
    rows = out[
        (out["role"] == role)
        & (out["model"] == model)
        & (out["metric"] == metric)
        & (out["group"] == group)
    ]
    assert len(rows) == 1, (role, model, metric, group)
    return float(rows["mae"].iloc[0])


_ROLE_METRICS = {"bat": {"ops", "pa", "hr"}, "pit": {"era", "ip", "k_bb_pct"}}
_LEAGUE_METRICS = {"bat": {"ops"}, "pit": {"era", "k_bb_pct"}}


@pytest.mark.parametrize(
    ("adopted", "extended_roles"),
    [((), []), (("ops",), ["bat"]), (("ip",), ["pit"]), (("pa", "era"), ["bat", "pit"])],
)
def test_final_evaluate_target_models_per_role(
    monkeypatch: pytest.MonkeyPatch, adopted: tuple[str, ...], extended_roles: list[str]
) -> None:
    fakes = ModelFakes()
    fakes.install(monkeypatch)
    tables = _world()
    adoption = _adoption(*adopted)
    out = final.evaluate_target(tables, PARAMS, adoption, TARGET)

    assert c.check_frame(out, c.EVALUATIONS) == []
    assert set(out["target_season"]) == {TARGET}
    for role in c.ROLES:
        models = {"league", "baseline", "weighted"} | (
            {"extended"} if role in extended_roles else set()
        )
        mine = out[out["role"] == role]
        assert set(mine["model"]) == models, role
        for model in models:
            rows = mine[mine["model"] == model]
            metrics = _LEAGUE_METRICS[role] if model == "league" else _ROLE_METRICS[role]
            assert set(rows["metric"]) == metrics, (role, model)
            assert len(rows) == len(metrics) * len(c.GROUPS), (role, model)
    # 가중은 두 역할 모두, 확장은 최종 모델이 extended인 역할만 (같은 목표·파라미터·채택표로)
    assert sorted(r for r, _, _ in fakes.weighted_calls) == ["bat", "pit"]
    assert all(t == TARGET and p == PARAMS for _, t, p in fakes.weighted_calls)
    assert sorted(r for r, *_ in fakes.extended_calls) == extended_roles
    assert all(t == TARGET and p == PARAMS and a == adoption for _, t, p, a in fakes.extended_calls)


def test_final_evaluate_target_hand_computed_mae(monkeypatch: pytest.MonkeyPatch) -> None:
    ModelFakes().install(monkeypatch)
    out = final.evaluate_target(_world(), PARAMS, _adoption("ops", "ip"), TARGET)
    # 타자 PA (실제 550·250): baseline 600·300 → |50|·|50| → 50, weighted 580·280 → 30,
    # extended 550·250 → 0
    assert _mae(out, "bat", "baseline", "pa") == pytest.approx(50.0)
    assert _mae(out, "bat", "weighted", "pa") == pytest.approx(30.0)
    assert _mae(out, "bat", "extended", "pa") == pytest.approx(0.0)
    # 타자 HR (실제 25·5): 기준 30·6 → |5|·|1| → 3 (가짜는 HR을 바꾸지 않는다)
    assert _mae(out, "bat", "baseline", "hr") == pytest.approx(3.0)
    assert _mae(out, "bat", "weighted", "hr") == pytest.approx(3.0)
    # 그룹: 타자 2만 seasons_1_2 (n_seasons 1) → weighted PA |280 − 250| = 30
    assert _mae(out, "bat", "weighted", "pa", "seasons_1_2") == pytest.approx(30.0)
    # 투수 IP (실제 90·45): baseline 100·50 → |10|·|5| → 7.5, weighted 95·45 → |5|·|0| → 2.5,
    # extended 90·45 → 0
    assert _mae(out, "pit", "baseline", "ip") == pytest.approx(7.5)
    assert _mae(out, "pit", "weighted", "ip") == pytest.approx(2.5)
    assert _mae(out, "pit", "extended", "ip") == pytest.approx(0.0)
    # 기준선 열: weighted PA 행의 baseline_mae_prev = baseline 모델 PA MAE 50
    row = out[(out["model"] == "weighted") & (out["metric"] == "pa") & (out["group"] == "all")]
    assert float(row["baseline_mae_prev"].iloc[0]) == pytest.approx(50.0)


def test_final_evaluate_target_sorted_by_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    ModelFakes().install(monkeypatch)
    out = final.evaluate_target(_world(), PARAMS, ADOPTION, TARGET)
    # 두 역할을 이은 뒤 키 순(안정 정렬), 색인 0…n−1
    expected = out.sort_values(_KEYS, kind="mergesort").reset_index(drop=True)
    assert_frame_equal(out, expected)
    assert list(out.index) == list(range(len(out)))
    # 타자 50행(league 5 + baseline·weighted·extended 15씩) + 투수 40행(league 10 + 15·15)
    assert len(out) == 90


# ---------------------------------------------------------------------------
# 최종 검증 (run_final_test) — evaluate_target은 가짜
# ---------------------------------------------------------------------------

ROWS_A = [
    _eval_row(2026, "bat", "baseline", "pa", 2, 50.0, None, None, 50.0, 0),
    _eval_row(2026, "bat", "weighted", "ops", 1, 0.0625, 0.05, 0.075, 0.1, 1),
]
ROWS_B = [
    _eval_row(2026, "pit", "baseline", "ip", 2, 7.5, None, None, 7.5, 0),
    _eval_row(2026, "pit", "weighted", "era", 1, 0.5, 0.625, 0.75, 0.875, 1),
    _eval_row(2026, "pit", "weighted", "ip", 2, 2.5, None, None, 7.5, 0),
]


class FakeEvaluate:
    """evaluate_target 가짜 — 정해 둔 평가표를 돌려주고 부른 인자를 남긴다."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], log: list[str] | None = None) -> None:
        self.frame = _evaluations(rows)
        self.log = log
        self.calls: list[tuple[Tables, Params, Adoption, int]] = []

    def __call__(
        self, tables: Tables, params: Params, adoption: Adoption, target: int
    ) -> pd.DataFrame:
        self.calls.append((tables, params, adoption, target))
        if self.log is not None:
            self.log.append("evaluate")
        return self.frame.copy()


def _run(
    tables: Tables,
    repo: Path,
    params_path: Path,
    adoption_path: Path,
    out_path: Path,
    now: str,
    git: FakeGit,
    reason: str | None = None,
) -> dict[str, Any]:
    return final.run_final_test(
        tables,
        PARAMS,
        ADOPTION,
        repo=repo,
        params_path=params_path,
        adoption_path=adoption_path,
        out_path=out_path,
        now=now,
        rerun_reason=reason,
        git=git,
    )


def test_final_run_final_test_writes_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, params_path, adoption_path = _repo(tmp_path)
    log: list[str] = []
    git = FakeGit(repo, CLEAN, log=log)
    evaluate = FakeEvaluate(ROWS_A, log)
    monkeypatch.setattr(final, "evaluate_target", evaluate)
    tables = _world()
    out_path = tmp_path / "out" / "final_test_2026.json"  # 상위 디렉토리가 아직 없다

    record = _run(tables, repo, params_path, adoption_path, out_path, NOW1, git)

    # 커밋 확인이 평가보다 먼저, 평가는 한 번 — 목표 2026, 받은 표·파라미터·채택표 그대로
    assert "evaluate" in log and "git" in log
    assert log.index("evaluate") > max(i for i, name in enumerate(log) if name == "git")
    assert len(evaluate.calls) == 1
    got_tables, got_params, got_adoption, target = evaluate.calls[0]
    assert got_tables is tables
    assert got_params == PARAMS and got_adoption == ADOPTION
    assert target == c.FINAL_TEST_TARGET == 2026

    expected = {
        "target_season": 2026,
        "ran_at": NOW1,
        "params_sha256": params_sha256(PARAMS),
        "adoption_sha256": adoption_sha256(ADOPTION),
        "commit": HEAD,
        "reruns": [],
        "rows": ROWS_A,  # 결측은 None (JSON null)
    }
    assert set(record) == RECORD_KEYS
    assert record == expected
    text = out_path.read_text(encoding="utf-8")
    assert _no_nan(text) == expected
    assert text.startswith('{\n "') and text.endswith("}\n")  # 사람이 읽는 indent=1
    for row in _no_nan(text)["rows"]:
        assert set(row) == set(c.EVALUATIONS.column_names)
        assert type(row["n"]) is int and type(row["excluded"]) is int
        assert type(row["target_season"]) is int


@pytest.mark.parametrize(
    ("states", "fail"),
    [
        ({**CLEAN, PARAMS_REL: "untracked"}, None),
        ({**CLEAN, ADOPTION_REL: "modified"}, None),
        (CLEAN, subprocess.CalledProcessError(128, ["git"], stderr="fatal: not a git repository")),
    ],
    ids=["params-untracked", "adoption-modified", "git-error"],
)
def test_final_run_final_test_checks_commit_before_evaluating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    states: dict[str, str],
    fail: BaseException | None,
) -> None:
    repo, params_path, adoption_path = _repo(tmp_path)
    evaluate = FakeEvaluate(ROWS_A)
    monkeypatch.setattr(final, "evaluate_target", evaluate)
    git = FakeGit(repo, states, fail=fail)
    out_path = tmp_path / "out" / "final_test_2026.json"
    with pytest.raises(FinalTestError) as info:
        _run(_world(), repo, params_path, adoption_path, out_path, NOW1, git)
    assert git.unsupported == [], f"FakeGit가 모르는 git 명령: {git.unsupported}"
    assert evaluate.calls == []  # 커밋 확인이 실패하면 평가하지 않는다
    assert not out_path.exists()
    assert "--rerun" not in str(info.value)  # "이미 돌렸다"가 아니라 커밋 문제


@pytest.mark.parametrize("reason", [None, ""], ids=["no-reason", "empty-reason"])
def test_final_run_final_test_refuses_second_run_without_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str | None
) -> None:
    repo, params_path, adoption_path = _repo(tmp_path)
    evaluate = FakeEvaluate(ROWS_A)
    monkeypatch.setattr(final, "evaluate_target", evaluate)
    out_path = tmp_path / "out" / "final_test_2026.json"
    _run(_world(), repo, params_path, adoption_path, out_path, NOW1, FakeGit(repo, CLEAN))
    before = out_path.read_bytes()

    monkeypatch.setattr(final, "evaluate_target", FakeEvaluate(ROWS_B))
    with pytest.raises(FinalTestError) as info:
        _run(
            _world(), repo, params_path, adoption_path, out_path, NOW2, FakeGit(repo, CLEAN), reason
        )
    message = str(info.value)
    assert "이미" in message and "--rerun" in message
    assert out_path.read_bytes() == before  # 기록은 바이트까지 그대로
    assert len(evaluate.calls) == 1


def test_final_run_final_test_rerun_keeps_first_ran_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, params_path, adoption_path = _repo(tmp_path)
    out_path = tmp_path / "out" / "final_test_2026.json"
    monkeypatch.setattr(final, "evaluate_target", FakeEvaluate(ROWS_A))
    _run(_world(), repo, params_path, adoption_path, out_path, NOW1, FakeGit(repo, CLEAN))

    reason = "원본 재수집 뒤 다시 확인"
    monkeypatch.setattr(final, "evaluate_target", FakeEvaluate(ROWS_B))
    git2 = FakeGit(repo, CLEAN, head=HEAD2)
    record = _run(_world(), repo, params_path, adoption_path, out_path, NOW2, git2, reason)

    assert set(record) == RECORD_KEYS
    assert record["target_season"] == 2026
    assert record["ran_at"] == NOW1  # 첫 실행 시각을 그대로 둔다
    assert record["reruns"] == [{"ran_at": NOW2, "reason": reason, "commit": HEAD2}]
    assert record["rows"] == ROWS_B  # 새 결과로 바꾼다
    assert record["params_sha256"] == params_sha256(PARAMS)
    assert record["adoption_sha256"] == adoption_sha256(ADOPTION)
    raw = out_path.read_bytes()
    assert _no_nan(raw.decode("utf-8")) == record
    assert reason.encode("utf-8") in raw  # ensure_ascii=False — 한국어가 그대로

    # 한 번 더 — reruns 끝에 붙는다
    monkeypatch.setattr(final, "evaluate_target", FakeEvaluate(ROWS_A))
    again = _run(
        _world(),
        repo,
        params_path,
        adoption_path,
        out_path,
        NOW3,
        FakeGit(repo, CLEAN),
        "둘째 사유",
    )
    assert again["ran_at"] == NOW1
    assert again["reruns"] == [
        {"ran_at": NOW2, "reason": reason, "commit": HEAD2},
        {"ran_at": NOW3, "reason": "둘째 사유", "commit": HEAD},
    ]
    assert again["rows"] == ROWS_A
    assert _no_nan(out_path.read_text(encoding="utf-8")) == again


# ---------------------------------------------------------------------------
# 2027 예측 (final_projections)
# ---------------------------------------------------------------------------


_IDS: dict[Role, tuple[int, ...]] = {"bat": (1, 2), "pit": (11, 12)}


class ProjectionFakes:
    """가중·확장 예측 가짜 — 목표마다 역할별 고정 표를 돌려준다."""

    def __init__(self) -> None:
        self.weighted_calls: list[tuple[Role, int]] = []
        self.extended_calls: list[tuple[Role, int, Adoption]] = []

    def weighted(self, tables: Tables, role: Role, target: int, params: Params) -> pd.DataFrame:
        self.weighted_calls.append((role, target))
        return _projection(role, target, "weighted", _IDS[role])

    def extended(
        self, tables: Tables, role: Role, target: int, params: Params, adoption: Adoption
    ) -> pd.DataFrame:
        self.extended_calls.append((role, target, adoption))
        return _projection(role, target, "extended", _IDS[role])

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(extended, "weighted_projection_for", self.weighted)
        monkeypatch.setattr(extended, "extended_projection", self.extended)


def test_final_final_projections_picks_final_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = ProjectionFakes()
    fakes.install(monkeypatch)
    adoption = _adoption("hr")  # 타자 extended, 투수 weighted
    out = final.final_projections(_world(), PARAMS, adoption)  # 기본 목표 = TARGET_SEASON

    assert set(out) == set(c.ROLES)
    assert_frame_equal(
        out["bat"].reset_index(drop=True), _projection("bat", c.TARGET_SEASON, "extended", (1, 2))
    )
    assert_frame_equal(
        out["pit"].reset_index(drop=True),
        _projection("pit", c.TARGET_SEASON, "weighted", (11, 12)),
    )
    for role, frame in out.items():
        assert set(frame["model"]) == {final.final_model_name(adoption, role)}
        assert set(frame["target_season"]) == {c.TARGET_SEASON} == {2027}
    # 확장 예측은 타자만, 목표 2027·받은 채택표로
    assert fakes.extended_calls == [("bat", 2027, adoption)]
    assert ("pit", 2027) in fakes.weighted_calls


def test_final_final_projections_passes_target(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = ProjectionFakes()
    fakes.install(monkeypatch)
    out = final.final_projections(_world(), PARAMS, NO_ADOPTION, target=TARGET)
    assert fakes.extended_calls == []  # 채택 없음 → 두 역할 모두 weighted
    assert sorted(fakes.weighted_calls) == [("bat", TARGET), ("pit", TARGET)]
    for role in c.ROLES:
        assert set(out[role]["model"]) == {"weighted"}
        assert set(out[role]["target_season"]) == {TARGET}


# ---------------------------------------------------------------------------
# 원본 수집 시각 (raw_collected_at)
# ---------------------------------------------------------------------------


def _clock(moment: datetime) -> Callable[[], datetime]:
    """늘 같은 시각을 돌려주는 시계 (save_raw의 now)."""

    def now() -> datetime:
        return moment

    return now


def _manifest(raw_dir: Path, times: Sequence[datetime]) -> None:
    """원본 파일마다 받은 시각을 정해 저장한다 (collect.raw.save_raw → manifest.json)."""
    for i, moment in enumerate(times):
        save_raw(
            raw_dir,
            Path("statsapi") / f"file_{i}.json",
            f"https://example.invalid/{i}",
            b"{}",
            now=_clock(moment),
        )


def test_final_raw_collected_at_takes_latest_fetch(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _manifest(
        raw_dir,
        [
            datetime(2026, 11, 20, 3, 4, 5, tzinfo=UTC),
            datetime(2026, 11, 23, 12, 0, 0, tzinfo=UTC),  # 가장 늦다
            datetime(2026, 11, 21, 23, 59, 59, tzinfo=UTC),
        ],
    )
    assert final.raw_collected_at(raw_dir) == "2026-11-23T12:00:00Z"


@pytest.mark.parametrize("content", [None, "{}\n"], ids=["no-manifest", "empty-manifest"])
def test_final_raw_collected_at_rejects_empty_manifest(tmp_path: Path, content: str | None) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    if content is not None:
        (raw_dir / c.MANIFEST_NAME).write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        final.raw_collected_at(raw_dir)


# ---------------------------------------------------------------------------
# 내보내기 입력 (build_export_input)
# ---------------------------------------------------------------------------

COLLECTED_AT = "2026-11-23T12:00:00Z"
# 과거 검증표 (2024·2025). 값은 CSV 왕복에도 정확한 이진 분수로 고른다
BACKTEST_ROWS = [
    _eval_row(2025, "pit", "weighted", "era", 300, 0.875, 0.8125, 0.9375, 1.25, 50),
    _eval_row(2024, "bat", "weighted", "ops", 400, 0.0625, 0.0703125, 0.078125, 0.09375, 100),
    _eval_row(2024, "bat", "weighted", "pa", 900, 76.25, None, None, 77.75, 0),
]
# 최종 검증 기록의 rows — 출전량 지표뿐이라 mae_weighted·baseline_mae_league가 모두 null이다
FINAL_ROWS = [
    _eval_row(2026, "bat", "weighted", "pa", 950, 75.5, None, None, 76.5, 0),
    _eval_row(2026, "pit", "weighted", "ip", 800, 18.25, None, None, 18.5, 0),
]
EXPECTED_META = {
    "schema_version": c.WEB_SCHEMA_VERSION,
    "generated_at": COLLECTED_AT,  # 같은 원본이면 같은 바이트 (N1) — 실행 시각이 아니다
    "raw_collected_at": COLLECTED_AT,
    "params_version": "v1",
    "params_sha256": params_sha256(PARAMS),
    "target_season": 2027,
    "input_seasons": [2022, 2023, 2024, 2025, 2026],  # Y−5 … Y−1
    "sources": [
        {"name": "MLB Stats API", "url": "https://statsapi.mlb.com"},
        {"name": "Baseball Savant", "url": "https://baseballsavant.mlb.com"},
    ],
    "disclaimer": c.WEB_DISCLAIMER,
}


def _final_record() -> dict[str, Any]:
    """run_final_test가 남기는 모양의 기록 (손으로 쓴다)."""
    return {
        "target_season": 2026,
        "ran_at": NOW1,
        "params_sha256": params_sha256(PARAMS),
        "adoption_sha256": adoption_sha256(ADOPTION),
        "commit": HEAD,
        "reruns": [{"ran_at": NOW2, "reason": "원본 정정", "commit": HEAD2}],
        "rows": FINAL_ROWS,
    }


def _export_paths(tmp_path: Path, *, with_record: bool) -> tuple[Path, Path, Path]:
    """원본 매니페스트·과거 검증 CSV·(있으면) 최종 검증 기록을 만든다."""
    raw_dir = tmp_path / "raw"
    _manifest(
        raw_dir,
        [datetime(2026, 11, 20, 3, 4, 5, tzinfo=UTC), datetime(2026, 11, 23, 12, 0, tzinfo=UTC)],
    )
    backtest_csv = tmp_path / "out" / "backtest.csv"
    backtest_csv.parent.mkdir(parents=True)
    # backtest 명령과 같은 방식으로 쓴다
    _evaluations(BACKTEST_ROWS).to_csv(backtest_csv, index=False, lineterminator="\n")
    final_test_path = tmp_path / "out" / "final_test_2026.json"
    if with_record:
        text = json.dumps(_final_record(), ensure_ascii=False, indent=1) + "\n"
        final_test_path.write_text(text, encoding="utf-8")
    return raw_dir, backtest_csv, final_test_path


def _projections_2027() -> dict[Role, pd.DataFrame]:
    return {
        "bat": _projection("bat", 2027, "extended", (1, 2)),
        "pit": _projection("pit", 2027, "weighted", (11, 12)),
    }


def _build(tmp_path: Path, tables: Tables, *, with_record: bool) -> export.ExportInput:
    raw_dir, backtest_csv, final_test_path = _export_paths(tmp_path, with_record=with_record)
    return final.build_export_input(
        tables,
        PARAMS,
        ADOPTION,
        _projections_2027(),
        raw_dir=raw_dir,
        backtest_csv=backtest_csv,
        final_test_path=final_test_path,
    )


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(_KEYS, kind="mergesort").reset_index(drop=True)


def _check_schema(inp: export.ExportInput) -> None:
    schema = load_schema(ROOT / c.WEB_SCHEMA_PATH)
    assert validate(inp.meta, "Meta", schema) == []
    report: dict[str, Any] = {"meta": inp.meta, "final_test": inp.final_test, "rows": []}
    assert validate(report, "ValidationReport", schema) == []


def test_final_build_export_input_without_final_test(tmp_path: Path) -> None:
    tables = _world()
    inp = _build(tmp_path, tables, with_record=False)

    assert isinstance(inp, export.ExportInput)
    assert inp.tables is tables
    assert set(inp.projections) == set(c.ROLES)
    for role, frame in _projections_2027().items():
        assert_frame_equal(inp.projections[role], frame)
    assert inp.model == {"bat": "extended", "pit": "weighted"}  # 채택표 ops → 타자만 extended
    assert inp.final_test is None
    assert c.check_frame(inp.evaluations, c.EVALUATIONS) == []
    assert_frame_equal(
        _sorted(inp.evaluations), _sorted(_evaluations(BACKTEST_ROWS)), check_dtype=False
    )
    assert inp.meta == EXPECTED_META
    _check_schema(inp)


def test_final_build_export_input_with_final_test(tmp_path: Path) -> None:
    inp = _build(tmp_path, _world(), with_record=True)

    # 2024·2025 과거 검증 + 2026 최종 검증 rows
    assert c.check_frame(inp.evaluations, c.EVALUATIONS) == []
    expected = _evaluations([*BACKTEST_ROWS, *FINAL_ROWS])
    assert_frame_equal(_sorted(inp.evaluations), _sorted(expected), check_dtype=False)
    assert sorted(set(inp.evaluations["target_season"])) == [2024, 2025, 2026]
    # 스키마 ValidationReport.final_test 모양 — adoption_sha256·commit은 웹에 내보내지 않는다
    assert inp.final_test == {
        "target_season": 2026,
        "ran_at": NOW1,
        "params_sha256": params_sha256(PARAMS),
        "reruns": [{"ran_at": NOW2, "reason": "원본 정정"}],
    }
    assert inp.meta == EXPECTED_META
    _check_schema(inp)


def test_final_build_export_input_is_reproducible(tmp_path: Path) -> None:
    # N1: 같은 입력이면 같은 묶음 (실행 시각이 섞이지 않는다)
    tables = _world()
    raw_dir, backtest_csv, final_test_path = _export_paths(tmp_path, with_record=True)

    def build() -> export.ExportInput:
        return final.build_export_input(
            tables,
            PARAMS,
            ADOPTION,
            _projections_2027(),
            raw_dir=raw_dir,
            backtest_csv=backtest_csv,
            final_test_path=final_test_path,
        )

    first, second = build(), build()
    assert first.tables is second.tables
    assert first.model == second.model
    assert first.final_test == second.final_test
    assert first.meta == second.meta
    assert set(first.projections) == set(second.projections)
    for role in first.projections:
        assert_frame_equal(first.projections[role], second.projections[role])
    assert_frame_equal(first.evaluations, second.evaluations)


# ---------------------------------------------------------------------------
# 리뷰에서 더한 시험 — 2026 규칙 가드, 누설(N2), 재현성(N1), 기록 형식
# ---------------------------------------------------------------------------


def _run_with(
    tmp_path: Path,
    params: Params,
    adoption: Adoption,
    *,
    now: str = NOW1,
    reason: str | None = None,
    git: FakeGit | None = None,
    out_name: str = "final_test_2026.json",
) -> dict[str, Any]:
    """파일에는 PARAMS·ADOPTION이 커밋돼 있고, 메모리 값은 params·adoption인 실행."""
    repo = tmp_path / "repo"
    return final.run_final_test(
        _world(),
        params,
        adoption,
        repo=repo,
        params_path=repo / PARAMS_REL,
        adoption_path=repo / ADOPTION_REL,
        out_path=tmp_path / "out" / out_name,
        now=now,
        rerun_reason=reason,
        git=git if git is not None else FakeGit(repo, CLEAN),
    )


@pytest.mark.parametrize(
    "now",
    [
        "2026-11-25 01:02:03",  # T 없음
        "2026-1-5T1:2:3Z",  # strptime은 받지만 자릿수가 틀리다
        "2026-13-01T00:00:00Z",  # 13월
        "2026-11-25T01:02:03+00:00",  # Z가 아니다
    ],
)
def test_final_run_final_test_rejects_bad_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, now: str
) -> None:
    _repo(tmp_path)
    evaluate = FakeEvaluate(ROWS_A)
    monkeypatch.setattr(final, "evaluate_target", evaluate)
    with pytest.raises(ValueError) as info:
        _run_with(tmp_path, PARAMS, ADOPTION, now=now)
    assert _HANGUL.search(str(info.value)) and now in str(info.value), str(info.value)
    assert evaluate.calls == []
    assert not (tmp_path / "out" / "final_test_2026.json").exists()


@pytest.mark.parametrize(
    ("params", "adoption", "named"),
    [
        (DEFAULT_PARAMS, ADOPTION, PARAMS_REL),  # 메모리는 기본값 v0, 파일은 v1
        (PARAMS, NO_ADOPTION, ADOPTION_REL),  # 메모리는 채택 없음, 파일은 ops 채택
    ],
    ids=["params-differ", "adoption-differ"],
)
def test_final_run_final_test_requires_committed_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, params: Params, adoption: Adoption, named: str
) -> None:
    # 파일은 커밋된 그대로(clean)여도, 넘긴 값이 그 파일과 다르면 돌지 않는다
    _repo(tmp_path)
    evaluate = FakeEvaluate(ROWS_A)
    monkeypatch.setattr(final, "evaluate_target", evaluate)
    with pytest.raises(FinalTestError) as info:
        _run_with(tmp_path, params, adoption)
    message = str(info.value)
    assert _HANGUL.search(message) and Path(named).name in message, message
    assert evaluate.calls == []
    assert not (tmp_path / "out" / "final_test_2026.json").exists()


def test_final_run_final_test_reason_without_record_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 사유를 남길 기록이 없다 — 기록 경로를 잘못 준 것일 수 있으니 새 "첫 실행"으로 돌지 않는다
    _repo(tmp_path)
    evaluate = FakeEvaluate(ROWS_A)
    monkeypatch.setattr(final, "evaluate_target", evaluate)
    with pytest.raises(FinalTestError) as info:
        _run_with(tmp_path, PARAMS, ADOPTION, reason="원본 정정")
    message = str(info.value)
    assert _HANGUL.search(message) and "final_test_2026.json" in message, message
    assert evaluate.calls == []
    assert not (tmp_path / "out" / "final_test_2026.json").exists()


@pytest.mark.parametrize(
    ("changed_params", "changed_adoption", "what"),
    [
        (replace(PARAMS, version="v2"), ADOPTION, "파라미터"),
        (PARAMS, _adoption("ops", "hr"), "채택표"),
    ],
    ids=["params-changed", "adoption-changed"],
)
def test_final_run_final_test_rerun_refuses_changed_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_params: Params,
    changed_adoption: Adoption,
    what: str,
) -> None:
    """2026을 본 뒤 파라미터·채택표를 바꿔 (새로 커밋해) 다시 돌리면 거부한다 (SPEC D10·F10)."""
    repo, params_path, adoption_path = _repo(tmp_path)
    evaluate = FakeEvaluate(ROWS_A)
    monkeypatch.setattr(final, "evaluate_target", evaluate)
    _run_with(tmp_path, PARAMS, ADOPTION)
    out_path = tmp_path / "out" / "final_test_2026.json"
    before = out_path.read_bytes()

    # 바꾼 값을 파일에 쓰고 커밋했다고 치자 (git 가짜는 clean, 새 HEAD)
    save_params(changed_params, params_path)
    save_adoption(changed_adoption, adoption_path)
    with pytest.raises(FinalTestError) as info:
        _run_with(
            tmp_path,
            changed_params,
            changed_adoption,
            now=NOW2,
            reason="모델을 고쳤다",
            git=FakeGit(repo, CLEAN, head=HEAD2),
        )
    message = str(info.value)
    assert "첫 실행" in message and what in message, message  # 무엇이 바뀌었는지 적는다
    assert out_path.read_bytes() == before
    assert len(evaluate.calls) == 1  # 두 번째는 평가하지 않는다


@pytest.mark.parametrize(
    "content",
    ["not json\n", '{"target_season": 2026}\n', "[]\n"],
    ids=["not-json", "missing-keys", "not-object"],
)
def test_final_run_final_test_rerun_rejects_malformed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    _repo(tmp_path)
    evaluate = FakeEvaluate(ROWS_A)
    monkeypatch.setattr(final, "evaluate_target", evaluate)
    out_path = tmp_path / "out" / "final_test_2026.json"
    out_path.parent.mkdir(parents=True)
    out_path.write_text(content, encoding="utf-8")
    with pytest.raises(FinalTestError) as info:
        _run_with(tmp_path, PARAMS, ADOPTION, now=NOW2, reason="다시")
    assert _HANGUL.search(str(info.value)), str(info.value)
    assert out_path.read_text(encoding="utf-8") == content
    assert evaluate.calls == []


def test_final_run_final_test_record_is_byte_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # N1: 같은 입력·같은 시각이면 기록은 바이트까지 같다
    _repo(tmp_path)
    monkeypatch.setattr(final, "evaluate_target", FakeEvaluate(ROWS_B))
    _run_with(tmp_path, PARAMS, ADOPTION, out_name="a.json")
    _run_with(tmp_path, PARAMS, ADOPTION, out_name="b.json")
    first = (tmp_path / "out" / "a.json").read_bytes()
    assert first == (tmp_path / "out" / "b.json").read_bytes()
    assert not list((tmp_path / "out").glob("*.tmp"))  # 임시 파일이 남지 않는다


def test_final_committed_head_rejects_path_outside_repo(tmp_path: Path) -> None:
    repo, params_path, _ = _repo(tmp_path)
    outside = tmp_path / "elsewhere" / "extended_v1.json"
    save_adoption(ADOPTION, outside)
    git = FakeGit(repo, CLEAN)
    with pytest.raises(FinalTestError) as info:
        final.committed_head([params_path, outside], repo, git)
    message = str(info.value)
    assert _HANGUL.search(message) and "elsewhere" in message, message


def _poisoned(tables: Tables) -> Tables:
    """목표 2025 세계에 2026 행(엉뚱한 값)과 2026에만 나오는 선수를 더한다."""
    batting = pd.concat(
        [
            tables.seasons["bat"],
            _frame(
                c.BATTING_SEASONS,
                [
                    _bat_season(1, 2026, 700, 70, 1.500),
                    _bat_season(2, 2026, 60, 0, 0.100),
                    _bat_season(3, 2026, 650, 45, 1.200),  # 2026 데뷔
                ],
            ),
        ],
        ignore_index=True,
    )
    pitching = pd.concat(
        [
            tables.seasons["pit"],
            _frame(
                c.PITCHING_SEASONS,
                [
                    _pit_season(11, 2026, 600, 700, 5, 300, 1, 0, 30, 30),
                    _pit_season(13, 2026, 450, 600, 90, 50, 80, 30, 25, 25),
                ],
            ),
        ],
        ignore_index=True,
    )
    return Tables(
        seasons={"bat": batting, "pit": pitching},
        cohorts=tables.cohorts,
        league={"bat": league_batting(batting), "pit": league_pitching(pitching)},
        debuts={**tables.debuts, 3: 2026, 13: 2026},
    )


def test_final_evaluate_target_ignores_later_seasons(monkeypatch: pytest.MonkeyPatch) -> None:
    """N2: 목표 2025 평가는 2026 행(독)이 있어도 바이트까지 같다 (기준선·평가는 진짜)."""
    ModelFakes().install(monkeypatch)
    clean = final.evaluate_target(_world(), PARAMS, ADOPTION, TARGET)
    poisoned = final.evaluate_target(_poisoned(_world()), PARAMS, ADOPTION, TARGET)
    assert_frame_equal(clean, poisoned)


# -- 내보내기 입력: 잘못된 입력은 조용히 넘어가지 않는다 --


def _build_with(
    tmp_path: Path, projections: Mapping[Role, pd.DataFrame], *, with_record: bool = False
) -> export.ExportInput:
    raw_dir, backtest_csv, final_test_path = _export_paths(tmp_path, with_record=with_record)
    return final.build_export_input(
        _world(),
        PARAMS,
        ADOPTION,
        dict(projections),
        raw_dir=raw_dir,
        backtest_csv=backtest_csv,
        final_test_path=final_test_path,
    )


@pytest.mark.parametrize(
    "case",
    ["missing-role", "targets-differ", "not-final-model", "two-models"],
)
def test_final_build_export_input_rejects_bad_projections(tmp_path: Path, case: str) -> None:
    projections: dict[Role, pd.DataFrame] = _projections_2027()
    if case == "missing-role":
        del projections["pit"]
    elif case == "targets-differ":  # 투수만 2026 예측 — 2026이 2027로 공개되면 안 된다
        projections["pit"] = _projection("pit", 2026, "weighted", (11, 12))
    elif case == "not-final-model":  # 채택표는 타자 extended인데 weighted 예측
        projections["bat"] = _projection("bat", 2027, "weighted", (1, 2))
    else:
        projections["bat"] = pd.concat(
            [projections["bat"], _projection("bat", 2027, "weighted", (1, 2))], ignore_index=True
        )
    with pytest.raises(ValueError) as info:
        _build_with(tmp_path, projections)
    assert _HANGUL.search(str(info.value)), str(info.value)


def test_final_build_export_input_rejects_2026_in_backtest(tmp_path: Path) -> None:
    raw_dir, backtest_csv, final_test_path = _export_paths(tmp_path, with_record=False)
    leaked = _evaluations([*BACKTEST_ROWS, FINAL_ROWS[0]])  # 2026 행이 과거 검증표에 섞였다
    leaked.to_csv(backtest_csv, index=False, lineterminator="\n")
    with pytest.raises(ValueError) as info:
        final.build_export_input(
            _world(),
            PARAMS,
            ADOPTION,
            _projections_2027(),
            raw_dir=raw_dir,
            backtest_csv=backtest_csv,
            final_test_path=final_test_path,
        )
    assert "2026" in str(info.value) and _HANGUL.search(str(info.value))


def _broken_records() -> dict[str, dict[str, Any]]:
    """한 곳씩 틀린 최종 검증 기록들 (이름 → 기록)."""
    good = _final_record()
    row = dict(FINAL_ROWS[0])
    rerun = dict(good["reruns"][0])
    return {
        "n-float": {**good, "rows": [{**row, "n": 950.5}]},  # astype이면 950으로 잘린다
        "n-bool": {**good, "rows": [{**row, "n": True}]},  # astype이면 1이 된다
        "row-extra-key": {**good, "rows": [{**row, "note": "x"}]},  # 조용히 버려진다
        "row-missing-key": {**good, "rows": [{k: v for k, v in row.items() if k != "group"}]},
        "row-null-role": {**good, "rows": [{**row, "role": None}]},
        "row-nan-mae": {**good, "rows": [{**row, "mae": float("nan")}]},
        "row-target-2025": {**good, "rows": [{**row, "target_season": 2025}]},
        "target-2025": {**good, "target_season": 2025},
        "extra-key": {**good, "note": "x"},
        "bad-ran-at": {**good, "ran_at": "2026-11-25"},
        "bad-sha": {**good, "params_sha256": "abc"},
        "rerun-no-reason": {
            **good,
            "reruns": [{k: v for k, v in rerun.items() if k != "reason"}],
        },
        "rerun-empty-reason": {**good, "reruns": [{**rerun, "reason": " "}]},
    }


@pytest.mark.parametrize("name", sorted(_broken_records()))
def test_final_build_export_input_rejects_malformed_record(tmp_path: Path, name: str) -> None:
    raw_dir, backtest_csv, final_test_path = _export_paths(tmp_path, with_record=False)
    # NaN은 json.dumps 기본값(allow_nan=True)으로 NaN 토큰이 된다 — 손으로 고친 기록을 흉내 낸다
    text = json.dumps(_broken_records()[name], ensure_ascii=False, indent=1) + "\n"
    final_test_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as info:
        final.build_export_input(
            _world(),
            PARAMS,
            ADOPTION,
            _projections_2027(),
            raw_dir=raw_dir,
            backtest_csv=backtest_csv,
            final_test_path=final_test_path,
        )
    message = str(info.value)
    assert "최종 검증 기록" in message and final_test_path.name in message, message


def test_final_run_final_test_record_feeds_build_export_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_final_test가 쓴 기록을 build_export_input이 그대로 받는다 (두 함수 사이 형식 계약)."""
    _repo(tmp_path)
    monkeypatch.setattr(final, "evaluate_target", FakeEvaluate(ROWS_B))
    _run_with(tmp_path, PARAMS, ADOPTION)
    raw_dir, backtest_csv, _ = _export_paths(tmp_path / "export", with_record=False)
    inp = final.build_export_input(
        _world(),
        PARAMS,
        ADOPTION,
        _projections_2027(),
        raw_dir=raw_dir,
        backtest_csv=backtest_csv,
        final_test_path=tmp_path / "out" / "final_test_2026.json",
    )
    expected = _evaluations([*BACKTEST_ROWS, *ROWS_B])
    assert_frame_equal(_sorted(inp.evaluations), _sorted(expected), check_dtype=False)
    assert inp.final_test == {
        "target_season": 2026,
        "ran_at": NOW1,
        "params_sha256": params_sha256(PARAMS),
        "reruns": [],
    }
    _check_schema(inp)


@pytest.mark.parametrize("mismatch", ["params", "adoption"])
def test_final_build_export_input_rejects_record_from_other_model(
    tmp_path: Path, mismatch: str
) -> None:
    # 2026 기록을 잰 모델과 2027을 내보내는 모델이 다르면 웹에 섞어 내보내지 않는다 (M4 검증 지적)
    raw_dir, backtest_csv, final_test_path = _export_paths(tmp_path, with_record=True)
    params = replace(PARAMS, version="v9") if mismatch == "params" else PARAMS
    adoption = (
        Adoption(ADOPTION.version, {**ADOPTION.adopt, "era": not ADOPTION.adopt["era"]})
        if mismatch == "adoption"
        else ADOPTION
    )
    projections = _projections_2027()
    if mismatch == "adoption" and final.final_model_name(adoption, "pit") == "extended":
        projections["pit"] = projections["pit"].assign(model="extended")
    with pytest.raises(ValueError, match="최종 검증 기록"):
        final.build_export_input(
            _world(),
            params,
            adoption,
            projections,
            raw_dir=raw_dir,
            backtest_csv=backtest_csv,
            final_test_path=final_test_path,
        )
