"""M4 명령(final-test·export·refresh) 테스트 (SPEC F10·F11·D25·N1, PLAN §2-3·§2-4).

단계 함수를 기록하는 가짜(`Steps`)로 바꿔 인자 기본값·단계 순서·멈춤·종료 코드·오류 메시지를 본다.

2026 규칙: final-test는 가짜 run_final_test로만 돈다. 모든 명령은 tmp_path로 옮긴 작업 디렉토리
(`workspace` 고정 장치)에서 돌아, 상대 경로 기본값(data/…, config/…, web/…)도 tmp 안이다. 저장소의
data·config·web에는 아무것도 쓰지 않는다. 파라미터·채택표·웹 스키마는 작업 디렉토리 안의 진짜 파일이다.

바꿔 끼우는 이름 — cli는 단계를 이 이름들로 부른다. 모듈 쪽 이름을 바꾸고, cli가 같은 이름을 직접
가져왔을 때에 대비해 cli 모듈의 같은 이름도 바꾼다(`Steps.install`).
- cli 자체: `run_collect`, `run_build`
- `load_tables` (`mlbproj.tables`·`mlbproj.backtest`·cli)
- `mlbproj.validate.validate_seasons`·`write_report`
- `mlbproj.final.run_final_test`·`evaluate_target`·`final_projections`·`build_export_input`
- `mlbproj.export.build_web`·`write_web`·`write_projection_csv`
"""

import json
import re
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mlbproj import backtest, cli, export, final
from mlbproj import contracts as c
from mlbproj import tables as tables_module
from mlbproj import validate as validation
from mlbproj.collect.raw import CollectOutcome, Getter, save_raw
from mlbproj.contracts import Role
from mlbproj.export import ExportInput
from mlbproj.interim import read_table
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
from mlbproj.schema import load_schema
from mlbproj.tables import Tables
from mlbproj.validate import (
    CHECK_NAMES,
    DEFAULT_LIMITS,
    CheckResult,
    LeagueLimits,
    SeasonReport,
    ValidationError,
)

ROOT = Path(__file__).resolve().parents[1]
PARAMS: Params = replace(DEFAULT_PARAMS, version="v1")
ADOPTION = Adoption(version="v1", adopt={m: m == "ops" for m in c.EXT_METRICS})
KEYS = list(c.EVALUATIONS.keys)
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_HANGUL = re.compile("[가-힣]")
_STAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


# ---------------------------------------------------------------------------
# 도우미 — 합성 표
# ---------------------------------------------------------------------------


def _dtype(col: c.Column) -> str:
    if col.kind == "int":
        return "Int64" if col.nullable else "int64"
    if col.kind == "float":
        return "float64"
    if col.kind == "bool":
        return "bool"
    return "str"


def _frame(spec: c.TableSpec, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """계약 열을 모두 가진 표. 행에 없는 열은 결측(None)이다."""
    df = pd.DataFrame(
        {
            col.name: pd.Series([row.get(col.name) for row in rows], dtype=_dtype(col))
            for col in spec.columns
        }
    )
    assert c.check_frame(df, spec) == []
    return df


def _eval_frame(target: int) -> pd.DataFrame:
    """목표 하나의 평가표 — 일부러 키 순이 아니다(weighted가 baseline보다 앞)."""
    base = {"target_season": target, "role": "bat", "group": "all"}
    return _frame(
        c.EVALUATIONS,
        [
            {
                **base,
                "model": "weighted",
                "metric": "pa",
                "n": 10,
                "mae": 30.5,
                "baseline_mae_prev": 40.25,
                "excluded": 0,
            },
            {
                **base,
                "model": "baseline",
                "metric": "pa",
                "n": 10,
                "mae": 40.25,
                "baseline_mae_prev": 40.25,
                "excluded": 0,
            },
            {
                **base,
                "model": "baseline",
                "metric": "ops",
                "n": 8,
                "mae": 0.0625,
                "mae_weighted": 0.0703125,
                "baseline_mae_league": 0.078125,
                "baseline_mae_prev": 0.0625,
                "excluded": 2,
            },
        ],
    )


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def _projection(role: Role) -> pd.DataFrame:
    """2027 예측 표 한 장 (선수 두 명)."""
    rows: list[dict[str, Any]] = []
    for pid in (1, 2) if role == "bat" else (11, 12):
        row: dict[str, Any] = {
            "player_id": pid,
            "role": role,
            "target_season": c.TARGET_SEASON,
            "model": "extended" if role == "bat" else "weighted",
            "input_first": 2022,
            "input_last": 2026,
            "n_seasons": 5,
        }
        if role == "bat":
            row.update(sample_pa=2500, pa=600.0 + pid, hr=25.0, obp=0.33, slg=0.42, ops=0.75)
            row.update(lg_ops=0.72)
        else:
            row.update(sample_bf=3000, sample_outs=2100, ip=150.0 + pid, era=3.9, k_bb_pct=15.0)
            row.update(lg_era=4.1, lg_k_bb_pct=13.4)
        rows.append(row)
    return _frame(c.BAT_PROJECTIONS if role == "bat" else c.PIT_PROJECTIONS, rows)


def _report(season: int, *, passed: bool) -> SeasonReport:
    """검증 보고서 한 시즌. passed=False면 첫 검사(F2-1)만 실패한다."""
    checks = tuple(
        CheckResult(name, passed or i > 0, "통과" if passed or i > 0 else "파일 없음")
        for i, name in enumerate(CHECK_NAMES)
    )
    return SeasonReport(season=season, checks=checks, merged_duplicates=(), inputs={})


def _eval_row(model: str, group: str, mae: float) -> dict[str, Any]:
    return {
        "target_season": 2026,
        "role": "bat",
        "model": model,
        "metric": "ops",
        "group": group,
        "n": 5,
        "mae": mae,
        "mae_weighted": 0.3,
        "baseline_mae_league": 0.4,
        "baseline_mae_prev": 0.5,
        "excluded": 1,
    }


# run_final_test 가짜가 돌려주는 기록. "all" 그룹만 요약에 나와야 한다(seasons_1_2의 0.875는 안 나온다)
FINAL_RECORD: dict[str, Any] = {
    "target_season": 2026,
    "ran_at": "2026-11-25T01:02:03Z",
    "params_sha256": params_sha256(PARAMS),
    "adoption_sha256": adoption_sha256(ADOPTION),  # 내보내기 안전장치가 대조한다
    "commit": "c" * 40,
    "reruns": [],
    "rows": [
        _eval_row("baseline", "all", 0.5),
        _eval_row("weighted", "all", 0.25),
        _eval_row("weighted", "seasons_1_2", 0.875),
    ],
}


def _clock(moment: datetime) -> Callable[[], datetime]:
    """늘 같은 시각을 돌려주는 시계 (save_raw의 now)."""

    def now() -> datetime:
        return moment

    return now


class NoNetwork:
    """수집 가짜에 넘기는 요청 객체 — 부르면 실패한다."""

    def get(self, url: str) -> bytes:
        raise AssertionError(f"요청하면 안 된다: {url}")


# ---------------------------------------------------------------------------
# 단계 가짜
# ---------------------------------------------------------------------------


class Steps:
    """단계 함수 가짜 묶음. 부른 순서와 인자를 `calls`에 남긴다."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tables = Tables(seasons={}, cohorts=pd.DataFrame(), league={}, debuts={})
        self.projections: dict[Role, pd.DataFrame] = {r: _projection(r) for r in c.ROLES}
        self.inp = ExportInput(
            tables=self.tables,
            projections=self.projections,
            model={"bat": "extended", "pit": "weighted"},
            evaluations=_eval_frame(2024),
            final_test=None,
            meta={"target_season": c.TARGET_SEASON},
        )
        self.files: dict[str, Any] = {
            "batters.json": {"rows": []},
            "pitchers.json": {"rows": []},
            "validation.json": {"rows": []},
            "players/1.json": {},
            "players/11.json": {},
        }
        self.sizes = {
            "batters.json": 123456,
            "pitchers.json": 65432,
            "players/1.json": 700,
            "players/11.json": 800,
            "validation.json": 9000,
        }
        self.failing_season: int | None = None
        self.build_error: Exception | None = None
        self.final_test_error: Exception | None = None
        self.write_web_error: Exception | None = None

    # -- 기록 조회 --

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def every(self, name: str) -> list[dict[str, Any]]:
        return [args for n, args in self.calls if n == name]

    def first(self, name: str) -> dict[str, Any]:
        found = self.every(name)
        assert found, f"{name}을 부르지 않았다 (부른 순서: {self.names()})"
        return found[0]

    def _log(self, name: str, **args: Any) -> None:
        self.calls.append((name, args))

    # -- 가짜 (진짜와 같은 이름·인자) --

    def run_collect(
        self, getter: Getter, raw_dir: Path, seasons: Sequence[int], *, force: bool
    ) -> list[CollectOutcome]:
        self._log("run_collect", getter=getter, raw_dir=raw_dir, seasons=list(seasons), force=force)
        return []

    def validate_seasons(
        self, raw_dir: Path, seasons: Iterable[int], *, limits: LeagueLimits = DEFAULT_LIMITS
    ) -> list[SeasonReport]:
        ordered = sorted(set(seasons))
        self._log("validate_seasons", raw_dir=raw_dir, seasons=ordered, limits=limits)
        return [_report(s, passed=s != self.failing_season) for s in ordered]

    def write_report(self, reports: Iterable[SeasonReport], path: Path) -> None:
        self._log("write_report", reports=list(reports), path=path)

    def run_build(self, raw_dir: Path, report: Path, out_dir: Path) -> dict[str, int]:
        self._log("run_build", raw_dir=raw_dir, report=report, out_dir=out_dir)
        if self.build_error is not None:
            raise self.build_error
        return {"cohorts": 4}

    def load_tables(self, interim_dir: Path = c.INTERIM_DIR, raw_dir: Path = c.RAW_DIR) -> Tables:
        self._log("load_tables", interim_dir=interim_dir, raw_dir=raw_dir)
        return self.tables

    def evaluate_target(
        self, tables: Tables, params: Params, adoption: Adoption, target: int
    ) -> pd.DataFrame:
        self._log("evaluate_target", tables=tables, params=params, adoption=adoption, target=target)
        return _eval_frame(target)

    def run_final_test(
        self,
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
        git: Any = None,
    ) -> dict[str, Any]:
        self._log(
            "run_final_test",
            tables=tables,
            params=params,
            adoption=adoption,
            repo=repo,
            params_path=params_path,
            adoption_path=adoption_path,
            out_path=out_path,
            now=now,
            rerun_reason=rerun_reason,
            git=git,
        )
        if self.final_test_error is not None:
            raise self.final_test_error
        return FINAL_RECORD

    def final_projections(
        self, tables: Tables, params: Params, adoption: Adoption, target: int = c.TARGET_SEASON
    ) -> dict[Role, pd.DataFrame]:
        self._log(
            "final_projections", tables=tables, params=params, adoption=adoption, target=target
        )
        return self.projections

    def build_export_input(
        self,
        tables: Tables,
        params: Params,
        adoption: Adoption,
        projections: dict[Role, pd.DataFrame],
        *,
        raw_dir: Path,
        backtest_csv: Path,
        final_test_path: Path,
    ) -> ExportInput:
        self._log(
            "build_export_input",
            tables=tables,
            params=params,
            adoption=adoption,
            projections=projections,
            raw_dir=raw_dir,
            backtest_csv=backtest_csv,
            final_test_path=final_test_path,
        )
        return self.inp

    def build_web(self, inp: ExportInput) -> dict[str, Any]:
        self._log("build_web", inp=inp)
        return self.files

    def write_web(
        self, files: Mapping[str, Any], out_dir: Path, schema: dict[str, Any]
    ) -> dict[str, int]:
        self._log("write_web", files=files, out_dir=out_dir, schema=schema)
        if self.write_web_error is not None:
            raise self.write_web_error
        return self.sizes

    def write_projection_csv(
        self, inp: ExportInput, paths: Mapping[Role, Path] = c.PROJECTION_CSV
    ) -> dict[Role, Path]:
        self._log("write_projection_csv", inp=inp, paths=dict(paths))
        return dict(paths)

    def install(self, monkeypatch: pytest.MonkeyPatch, *, real: Iterable[str] = ()) -> None:
        """가짜를 끼운다. real에 적은 이름은 진짜를 그대로 둔다."""
        keep = set(real)
        places: list[tuple[Any, str]] = [
            (cli, "run_collect"),
            (cli, "run_build"),
            (tables_module, "load_tables"),
            (backtest, "load_tables"),
            (validation, "validate_seasons"),
            (validation, "write_report"),
            (final, "run_final_test"),
            (final, "evaluate_target"),
            (final, "final_projections"),
            (final, "build_export_input"),
            (export, "build_web"),
            (export, "write_web"),
            (export, "write_projection_csv"),
        ]
        for module, name in places:
            if name in keep:
                continue
            fake = getattr(self, name)
            monkeypatch.setattr(module, name, fake)
            monkeypatch.setattr(cli, name, fake, raising=False)


def _replace(monkeypatch: pytest.MonkeyPatch, module: Any, name: str, fake: Any) -> None:
    """모듈과 cli의 같은 이름을 함께 바꾼다."""
    monkeypatch.setattr(module, name, fake)
    monkeypatch.setattr(cli, name, fake, raising=False)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """tmp_path로 옮긴 작업 디렉토리 — 파라미터·채택표·웹 스키마가 계약 경로에 있다."""
    monkeypatch.chdir(tmp_path)
    schema = tmp_path / c.WEB_SCHEMA_PATH
    schema.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / c.WEB_SCHEMA_PATH, schema)
    save_params(PARAMS, tmp_path / c.PARAMS_PATH)
    save_adoption(ADOPTION, tmp_path / c.ADOPTION_PATH)
    assert Path.cwd().resolve() == tmp_path.resolve()
    return tmp_path


@pytest.fixture
def steps(monkeypatch: pytest.MonkeyPatch) -> Steps:
    found = Steps()
    found.install(monkeypatch)
    return found


def _before(names: list[str], *order: str) -> None:
    """order의 단계가 이 순서로 (각각 처음 부른 때 기준) 불렸는지."""
    missing = [name for name in order if name not in names]
    assert not missing, f"부르지 않은 단계 {missing} (부른 순서: {names})"
    positions = [names.index(name) for name in order]
    assert positions == sorted(positions), f"순서가 다르다: {names}"


# ---------------------------------------------------------------------------
# final-test
# ---------------------------------------------------------------------------


def test_cli_m4_final_test_defaults(
    workspace: Path, steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    before = datetime.now(UTC).strftime(_STAMP_FORMAT)
    code = cli.main(["final-test", "--target", "2026"])
    after = datetime.now(UTC).strftime(_STAMP_FORMAT)
    out = capsys.readouterr().out

    assert code == 0
    assert steps.first("load_tables") == {"interim_dir": c.INTERIM_DIR, "raw_dir": c.RAW_DIR}
    run = steps.first("run_final_test")
    assert len(steps.every("run_final_test")) == 1
    assert run["tables"] is steps.tables
    assert run["params"] == PARAMS and run["adoption"] == ADOPTION
    assert run["repo"] == Path(".")
    assert run["params_path"] == c.PARAMS_PATH
    assert run["adoption_path"] == c.ADOPTION_PATH
    assert run["out_path"] == c.FINAL_TEST_PATH
    assert run["rerun_reason"] is None
    assert run["git"] is None  # 실제 git (가짜는 테스트만)
    assert STAMP.fullmatch(run["now"]), run["now"]
    assert before <= run["now"] <= after  # 지금 UTC 시각
    # 평가는 run_final_test 안에서만 — 명령이 2026을 따로 평가하지 않는다
    assert "evaluate_target" not in steps.names()
    # "all" 그룹 요약: weighted 0.25·baseline 0.5는 나오고 seasons_1_2의 0.875는 안 나온다
    assert "0.25" in out and "0.5" in out
    assert "0.875" not in out


def test_cli_m4_final_test_options(workspace: Path, steps: Steps) -> None:
    params_path = workspace / "frozen" / "p.json"
    adoption_path = workspace / "frozen" / "a.json"
    save_params(PARAMS, params_path)
    save_adoption(ADOPTION, adoption_path)
    code = cli.main(
        [
            "final-test",
            "--target",
            "2026",
            "--rerun",
            "원본 정정 뒤 다시 확인",
            "--interim-dir",
            str(workspace / "i"),
            "--raw-dir",
            str(workspace / "r"),
            "--params",
            str(params_path),
            "--adoption",
            str(adoption_path),
            "--out",
            str(workspace / "o" / "ft.json"),
            "--repo",
            str(workspace),
        ]
    )
    assert code == 0
    assert steps.first("load_tables") == {
        "interim_dir": workspace / "i",
        "raw_dir": workspace / "r",
    }
    run = steps.first("run_final_test")
    assert run["rerun_reason"] == "원본 정정 뒤 다시 확인"
    assert run["params_path"] == params_path
    assert run["adoption_path"] == adoption_path
    assert run["out_path"] == workspace / "o" / "ft.json"
    assert run["repo"] == workspace


@pytest.mark.parametrize("target", ["2025", "2027"])
def test_cli_m4_final_test_target_is_2026_only(
    workspace: Path, steps: Steps, capsys: pytest.CaptureFixture[str], target: str
) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["final-test", "--target", target])
    assert info.value.code == 2  # argparse choices
    assert "--target" in capsys.readouterr().err
    assert "run_final_test" not in steps.names()


def test_cli_m4_final_test_reports_final_test_error(
    workspace: Path, steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    message = "이미 한 번 돌렸다 — 다시 돌리려면 --rerun 사유를 준다"
    steps.final_test_error = final.FinalTestError(message)
    code = cli.main(["final-test", "--target", "2026"])
    assert code == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("missing", [c.PARAMS_PATH, c.ADOPTION_PATH], ids=["params", "adoption"])
def test_cli_m4_final_test_requires_frozen_files(
    workspace: Path, steps: Steps, capsys: pytest.CaptureFixture[str], missing: Path
) -> None:
    # 최종 검증은 기본값(v0)·채택 없음으로 대신 돌지 않는다
    (workspace / missing).unlink()
    code = cli.main(["final-test", "--target", "2026"])
    err = capsys.readouterr().err
    assert code == 1
    assert missing.name in err and _HANGUL.search(err), err
    assert "run_final_test" not in steps.names()


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_cli_m4_export_defaults(
    workspace: Path, steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["export"])
    out = capsys.readouterr().out

    assert code == 0
    names = steps.names()
    _before(
        names, "load_tables", "final_projections", "build_export_input", "build_web", "write_web"
    )
    _before(names, "build_export_input", "write_projection_csv")
    assert steps.first("load_tables") == {"interim_dir": c.INTERIM_DIR, "raw_dir": c.RAW_DIR}
    projections = steps.first("final_projections")
    assert projections["tables"] is steps.tables
    assert projections["params"] == PARAMS and projections["adoption"] == ADOPTION
    assert projections["target"] == c.TARGET_SEASON
    inp = steps.first("build_export_input")
    assert inp["projections"] is steps.projections
    assert inp["params"] == PARAMS and inp["adoption"] == ADOPTION
    assert inp["raw_dir"] == c.RAW_DIR
    assert inp["backtest_csv"] == c.BACKTEST_CSV
    assert inp["final_test_path"] == c.FINAL_TEST_PATH
    assert steps.first("build_web")["inp"] is steps.inp
    web = steps.first("write_web")
    assert web["files"] is steps.files
    assert web["out_dir"] == c.WEB_DATA_DIR
    assert web["schema"] == load_schema(ROOT / c.WEB_SCHEMA_PATH)
    csv = steps.first("write_projection_csv")
    assert csv["inp"] is steps.inp
    assert csv["paths"] == c.PROJECTION_CSV
    # 첫 화면 파일 크기 (바이트, 천 단위 쉼표는 있어도 된다)
    printed = out.replace(",", "")
    assert "123456" in printed and "65432" in printed
    assert "run_final_test" not in names


def test_cli_m4_export_options(workspace: Path, steps: Steps) -> None:
    params_path = workspace / "frozen" / "p.json"
    adoption_path = workspace / "frozen" / "a.json"
    save_params(PARAMS, params_path)
    save_adoption(NO_ADOPTION, adoption_path)
    code = cli.main(
        [
            "export",
            "--interim-dir",
            str(workspace / "i"),
            "--raw-dir",
            str(workspace / "r"),
            "--params",
            str(params_path),
            "--adoption",
            str(adoption_path),
            "--backtest-csv",
            str(workspace / "o" / "bt.csv"),
            "--final-test",
            str(workspace / "o" / "ft.json"),
            "--web-dir",
            str(workspace / "w"),
        ]
    )
    assert code == 0
    assert steps.first("load_tables") == {
        "interim_dir": workspace / "i",
        "raw_dir": workspace / "r",
    }
    assert steps.first("final_projections")["adoption"] == NO_ADOPTION
    inp = steps.first("build_export_input")
    assert inp["raw_dir"] == workspace / "r"
    assert inp["backtest_csv"] == workspace / "o" / "bt.csv"
    assert inp["final_test_path"] == workspace / "o" / "ft.json"
    assert steps.first("write_web")["out_dir"] == workspace / "w"


def test_cli_m4_export_missing_adoption_uses_no_adoption(
    workspace: Path, steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    (workspace / c.ADOPTION_PATH).unlink()
    code = cli.main(["export"])
    captured = capsys.readouterr()
    assert code == 0
    assert steps.first("final_projections")["adoption"] == NO_ADOPTION
    assert steps.first("build_export_input")["adoption"] == NO_ADOPTION
    # 경고에 빠진 파일을 적는다 (표준 출력이든 표준 오류든)
    assert c.ADOPTION_PATH.name in captured.out + captured.err


def test_cli_m4_export_reports_write_error(
    workspace: Path, steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    steps.write_web_error = ValueError("batters.json: 첫 화면 1MB 초과")
    code = cli.main(["export"])
    assert code == 1
    assert "batters.json: 첫 화면 1MB 초과" in capsys.readouterr().err


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_cli_m4_export_is_byte_stable(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N1: 같은 원본·같은 가짜면 export를 두 번 돌려도 같은 바이트 (build_export_input은 진짜)."""
    steps = Steps()
    steps.install(monkeypatch, real={"build_export_input"})

    def build_files(inp: ExportInput) -> dict[str, Any]:
        # 입력 묶음을 그대로 옮긴 결정적 파일들
        rows = _sorted(inp.evaluations).to_csv(index=False, lineterminator="\n")
        return {
            "validation.json": {"meta": inp.meta, "final_test": inp.final_test, "rows": rows},
            "batters.json": {"meta": inp.meta, "model": inp.model["bat"]},
            "pitchers.json": {"meta": inp.meta, "model": inp.model["pit"]},
        }

    def write_files(
        files: Mapping[str, Any], out_dir: Path, schema: dict[str, Any]
    ) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for rel in sorted(files):
            data = (json.dumps(files[rel], ensure_ascii=False, sort_keys=True) + "\n").encode()
            path = out_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            sizes[rel] = len(data)
        return sizes

    def write_csvs(
        inp: ExportInput, paths: Mapping[Role, Path] = c.PROJECTION_CSV
    ) -> dict[Role, Path]:
        for role, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                inp.projections[role].to_csv(index=False, lineterminator="\n"), encoding="utf-8"
            )
        return dict(paths)

    _replace(monkeypatch, export, "build_web", build_files)
    _replace(monkeypatch, export, "write_web", write_files)
    _replace(monkeypatch, export, "write_projection_csv", write_csvs)

    # 원본 매니페스트(가장 늦은 받은 시각 2026-11-23T12:00:00Z)·과거 검증표·최종 검증 기록
    for i, moment in enumerate(
        [datetime(2026, 11, 20, 3, 4, 5, tzinfo=UTC), datetime(2026, 11, 23, 12, tzinfo=UTC)]
    ):
        save_raw(
            workspace / c.RAW_DIR,
            Path(f"statsapi/f{i}.json"),
            f"https://example.invalid/{i}",
            b"{}",
            now=_clock(moment),
        )
    backtest_csv = workspace / c.BACKTEST_CSV
    backtest_csv.parent.mkdir(parents=True, exist_ok=True)
    _eval_frame(2024).to_csv(backtest_csv, index=False, lineterminator="\n")
    record = {**FINAL_RECORD, "rows": FINAL_RECORD["rows"][:2]}
    (workspace / c.FINAL_TEST_PATH).write_text(
        json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    shots: list[dict[str, bytes]] = []
    for web in ("w1", "w2"):
        assert cli.main(["export", "--web-dir", str(workspace / web)]) == 0
        shot = _snapshot(workspace / web)
        for role in c.ROLES:
            shot[f"csv/{role}"] = (workspace / c.PROJECTION_CSV[role]).read_bytes()
        shots.append(shot)
    assert shots[0] == shots[1]
    meta = json.loads(shots[0]["validation.json"])["meta"]
    assert meta["generated_at"] == meta["raw_collected_at"] == "2026-11-23T12:00:00Z"


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


def _expected_backtest_csv() -> bytes:
    """TUNE_TARGETS 평가표를 이어 키 순으로 정렬한 CSV (EVALUATIONS 열, 줄 끝 "\\n")."""
    frames = [_eval_frame(t) for t in c.TUNE_TARGETS]
    merged = _sorted(pd.concat(frames, ignore_index=True))
    return merged.to_csv(index=False, lineterminator="\n").encode("utf-8")


def test_cli_m4_refresh_runs_steps_in_order(workspace: Path, steps: Steps) -> None:
    raw, interim, out, web = (workspace / name for name in ("r", "i", "o", "w"))
    out.mkdir()
    record = out / c.FINAL_TEST_PATH.name
    record.write_text('{"kept": true}\n', encoding="utf-8")
    before = record.read_bytes()
    getter = NoNetwork()

    code = cli.main(
        [
            "refresh",
            "--raw-dir",
            str(raw),
            "--interim-dir",
            str(interim),
            "--out-dir",
            str(out),
            "--web-dir",
            str(web),
        ],
        getter=getter,
    )
    assert code == 0
    names = steps.names()
    _before(
        names,
        "run_collect",
        "validate_seasons",
        "write_report",
        "run_build",
        "load_tables",
        "evaluate_target",
        "final_projections",
        "build_export_input",
        "build_web",
        "write_web",
    )
    _before(names, "build_export_input", "write_projection_csv")
    assert names.index("load_tables") > names.index("run_build")  # 새로 만든 정제 표를 읽는다
    # 최종 검증은 refresh가 돌리지 않는다 — 2026 평가도 하지 않는다
    assert "run_final_test" not in names

    collect = steps.first("run_collect")
    assert collect["getter"] is getter
    assert collect["raw_dir"] == raw
    assert collect["seasons"] == list(c.SEASONS)
    assert collect["force"] is False  # 기본은 캐시를 쓴다
    validate = steps.first("validate_seasons")
    assert validate["raw_dir"] == raw and validate["seasons"] == list(c.SEASONS)
    build = steps.first("run_build")
    assert build["raw_dir"] == raw and build["out_dir"] == interim
    assert build["report"] == steps.first("write_report")["path"]  # 방금 쓴 검증 보고서
    assert steps.first("load_tables") == {"interim_dir": interim, "raw_dir": raw}

    evaluations = steps.every("evaluate_target")
    assert [e["target"] for e in evaluations] == list(c.TUNE_TARGETS) == [2024, 2025]
    for e in evaluations:
        assert e["tables"] is steps.tables
        assert e["params"] == PARAMS and e["adoption"] == ADOPTION  # 고정 파일 그대로

    # 과거 검증표: EVALUATIONS 열, 키 순, 줄 끝 "\n"
    backtest_csv = out / c.BACKTEST_CSV.name
    assert backtest_csv.read_bytes() == _expected_backtest_csv()
    assert c.check_frame(read_table(c.EVALUATIONS, backtest_csv), c.EVALUATIONS) == []

    inp = steps.first("build_export_input")
    assert inp["raw_dir"] == raw
    assert inp["backtest_csv"] == backtest_csv
    assert inp["final_test_path"] == record  # 있으면 그 기록을 쓴다
    assert inp["params"] == PARAMS and inp["adoption"] == ADOPTION
    assert steps.first("write_web")["out_dir"] == web
    assert steps.first("write_projection_csv")["paths"] == {
        role: out / c.PROJECTION_CSV[role].name for role in c.ROLES
    }
    assert record.read_bytes() == before  # 최종 검증 기록은 건드리지 않는다


def test_cli_m4_refresh_defaults(workspace: Path, steps: Steps) -> None:
    code = cli.main(["refresh"], getter=NoNetwork())
    assert code == 0
    assert steps.first("run_collect")["raw_dir"] == c.RAW_DIR
    assert steps.first("run_build")["out_dir"] == c.INTERIM_DIR
    assert steps.first("load_tables") == {"interim_dir": c.INTERIM_DIR, "raw_dir": c.RAW_DIR}
    # 상대 경로 기본값은 작업 디렉토리(tmp) 안이다
    assert (workspace / c.BACKTEST_CSV).read_bytes() == _expected_backtest_csv()
    inp = steps.first("build_export_input")
    assert inp["raw_dir"] == c.RAW_DIR
    assert inp["backtest_csv"] == c.BACKTEST_CSV
    assert inp["final_test_path"] == c.FINAL_TEST_PATH
    assert steps.first("write_web")["out_dir"] == c.WEB_DATA_DIR
    assert steps.first("write_projection_csv")["paths"] == c.PROJECTION_CSV
    # 기록이 없어도 최종 검증을 돌리지 않는다
    assert not (workspace / c.FINAL_TEST_PATH).exists()
    assert "run_final_test" not in steps.names()
    assert c.FINAL_TEST_TARGET not in [e["target"] for e in steps.every("evaluate_target")]


def test_cli_m4_refresh_force_refetches(workspace: Path, steps: Steps) -> None:
    code = cli.main(["refresh", "--force"], getter=NoNetwork())
    assert code == 0
    assert steps.first("run_collect")["force"] is True


def test_cli_m4_refresh_stops_on_validation_failure(workspace: Path, steps: Steps) -> None:
    steps.failing_season = 2026
    code = cli.main(["refresh", "--out-dir", str(workspace / "o")], getter=NoNetwork())
    assert code == 1
    names = steps.names()
    assert "run_collect" in names and "validate_seasons" in names
    later = {
        "run_build",
        "load_tables",
        "evaluate_target",
        "run_final_test",
        "final_projections",
        "build_export_input",
        "build_web",
        "write_web",
        "write_projection_csv",
    }
    assert later.isdisjoint(names), names
    assert not (workspace / "o" / c.BACKTEST_CSV.name).exists()


# ---------------------------------------------------------------------------
# 리뷰에서 더한 시험 — 멈춤·경고·두 보고서 파일
# ---------------------------------------------------------------------------


def test_cli_m4_export_requires_params(
    workspace: Path, steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    # 내보내기도 기본값(v0)으로 대신 돌지 않는다
    (workspace / c.PARAMS_PATH).unlink()
    code = cli.main(["export"])
    err = capsys.readouterr().err
    assert code == 1
    assert c.PARAMS_PATH.name in err and _HANGUL.search(err), err
    assert "final_projections" not in steps.names()


def test_cli_m4_refresh_requires_params_before_collecting(
    workspace: Path, steps: Steps, capsys: pytest.CaptureFixture[str]
) -> None:
    (workspace / c.PARAMS_PATH).unlink()
    code = cli.main(["refresh"], getter=NoNetwork())
    err = capsys.readouterr().err
    assert code == 1
    assert c.PARAMS_PATH.name in err and _HANGUL.search(err), err
    assert steps.names() == []  # 수집도 하지 않는다


@pytest.mark.parametrize(
    "error",
    [ValidationError("검증 뒤 원본이 바뀌었다"), ValueError("players: 열 불일치")],
    ids=["validation", "contract"],
)
def test_cli_m4_refresh_stops_on_build_failure(
    workspace: Path, steps: Steps, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    steps.build_error = error
    code = cli.main(["refresh", "--out-dir", str(workspace / "o")], getter=NoNetwork())
    err = capsys.readouterr().err
    assert code == 1
    assert str(error) in err and _HANGUL.search(err), err
    names = steps.names()
    assert "run_build" in names
    assert {"load_tables", "evaluate_target", "build_export_input", "write_web"}.isdisjoint(names)
    assert not (workspace / "o" / c.BACKTEST_CSV.name).exists()


def test_cli_m4_refresh_backtest_json_matches_csv(workspace: Path, steps: Steps) -> None:
    out = workspace / "o"
    assert cli.main(["refresh", "--out-dir", str(out)], getter=NoNetwork()) == 0
    csv_rows = read_table(c.EVALUATIONS, out / c.BACKTEST_CSV.name)
    payload = json.loads((out / "backtest.json").read_text(encoding="utf-8"))
    # backtest 명령과 같은 모양
    assert set(payload) == {"params_version", "params_sha256", "targets", "rows"}
    assert payload["params_version"] == "v1"
    assert payload["params_sha256"] == params_sha256(PARAMS)
    assert payload["targets"] == [2024, 2025]
    # 행 순서·값이 CSV와 같다 (NaN ↔ null)
    json_rows = pd.DataFrame(payload["rows"], columns=c.EVALUATIONS.column_names)
    pd.testing.assert_frame_equal(
        json_rows.astype({"mae_weighted": "float64", "baseline_mae_league": "float64"}),
        csv_rows,
        check_dtype=False,
    )


def _eval_rows(cells: Iterable[tuple[int, str, str]]) -> pd.DataFrame:
    """(목표, 역할, 모델)마다 "all" 그룹 한 행씩인 평가표."""
    metric = {"bat": "pa", "pit": "ip"}
    return _frame(
        c.EVALUATIONS,
        [
            {
                "target_season": t,
                "role": role,
                "model": model,
                "metric": metric[role],
                "group": "all",
                "n": 3,
                "mae": 1.5,
                "excluded": 0,
            }
            for t, role, model in cells
        ],
    )


_FULL_CELLS = [
    (t, r, m) for t in (2024, 2025) for r, m in (("bat", "extended"), ("pit", "weighted"))
]


@pytest.mark.parametrize(
    ("cells", "final_test", "warned"),
    [
        (_FULL_CELLS, None, []),
        # M3 backtest.csv처럼 extended 행이 없다 → 타자만 경고
        ([(t, r, "weighted") for t, r, _ in _FULL_CELLS], None, ["bat"]),
        # 최종 검증 기록이 있으면 2026 행도 있어야 한다
        (_FULL_CELLS, {"target_season": 2026}, ["bat", "pit"]),
    ],
    ids=["complete", "no-extended-rows", "no-2026-rows"],
)
def test_cli_m4_export_warns_when_final_model_rows_missing(
    workspace: Path,
    steps: Steps,
    capsys: pytest.CaptureFixture[str],
    cells: list[tuple[int, str, str]],
    final_test: dict[str, Any] | None,
    warned: list[str],
) -> None:
    steps.inp = replace(steps.inp, evaluations=_eval_rows(cells), final_test=final_test)
    code = cli.main(["export"])
    err = capsys.readouterr().err
    assert code == 0  # 경고만 하고 내보내기는 한다
    lines = [line for line in err.splitlines() if "최종 모델" in line]
    assert [r for r in c.ROLES if any(f" {r} " in line for line in lines)] == warned, err
    if final_test is not None:
        assert all("2026" in line for line in lines), err
    for line in lines:
        assert "refresh" in line and _HANGUL.search(line), line
