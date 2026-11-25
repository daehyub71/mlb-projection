"""v0.5 WAR 명령 — SPEC F1 ⑤·F2 ⑦·F3·F18~F22·D33~D36, PLAN v0.3 §2-4. 메인 세션이 쓴다.

기존 명령(cli.py)과 기록은 건드리지 않고 WAR 단계만 붙인다.
    collect-war → validate-war → build-war → tune-war → (war_v1.json 커밋) → final-test-war(2026 한 번)
    → export-war(2027 · war.json · 검토용 CSV). backtest-war는 2024·2025 과거 검증표만 다시 쓴다.

WAR의 출전량은 기존 출전량 모델(`predict_playing_time`)이다. 이것이 화면의 PA·IP와 같으려면 확장 모델
채택표가 출전량 보정을 쓰지 않아야 한다(E23: PA·IP 끔) — 켜져 있으면 WAR 명령은 멈춘다.
war.json은 `export.write_web`을 쓰지 않고 따로 쓴다 — write_web은 넘긴 목록에 없는 선수 상세 파일을 지운다.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mlbproj import contracts as c
from mlbproj import export, final
from mlbproj.clean.war import build_war_seasons
from mlbproj.collect import statsapi
from mlbproj.collect.http import Fetcher, FetchError
from mlbproj.collect.raw import CollectOutcome, Getter
from mlbproj.interim import read_table, write_table
from mlbproj.models.params import Adoption, Params, load_adoption, load_params
from mlbproj.models.war import (
    WarParams,
    combine_roles,
    load_war_params,
    save_war_params,
    tune_war,
    war_params_sha256,
    war_projections,
)
from mlbproj.schema import load_schema, validate
from mlbproj.tables import Tables, load_tables
from mlbproj.validate import ValidationError
from mlbproj.validate_war import assert_war_validated, validate_war_seasons, write_war_report
from mlbproj.war_eval import (
    WarExportInput,
    WarTestError,
    backtest_war,
    build_war_web,
    run_final_test_war,
    write_war_csv,
)

WAR_COMMANDS: tuple[str, ...] = (
    "collect-war",
    "validate-war",
    "build-war",
    "tune-war",
    "backtest-war",
    "final-test-war",
    "export-war",
)
WAR_WEB_FILE = "war.json"


def _utc_stamp() -> str:
    """지금 UTC 시각 ("YYYY-MM-DDTHH:MM:SSZ")."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def require_playing_time_unadopted(adoption: Adoption) -> None:
    """WAR 출전량 = 기존 출전량 모델이 되려면 채택표가 PA·IP 보정을 쓰지 않아야 한다 (F19).

    Raises:
        ValueError: 채택표가 pa나 ip를 켰을 때.
    """
    on = [m for m in ("pa", "ip") if adoption.adopt.get(m, False)]
    if on:
        raise ValueError(
            f"채택표가 출전량 보정 {on}을 켰다 — WAR 출전량(predict_playing_time)이 화면의 PA·IP와 "
            "달라진다. models.war의 출전량 입력을 최종 모델 출전량으로 바꾼 뒤 돌린다"
        )


def run_collect_war(
    getter: Getter, raw_dir: Path, seasons: Sequence[int], *, force: bool
) -> list[CollectOutcome]:
    """세이버메트릭스 원본을 시즌 × 역할마다 확보한다 (F1 ⑤)."""
    return [
        statsapi.collect_saber(getter, raw_dir, season, role, force=force)
        for season in seasons
        for role in c.ROLES
    ]


def run_build_war(raw_dir: Path, report: Path, interim_dir: Path) -> int:
    """검사를 통과한 세이버메트릭스로 `war_seasons`를 만든다 (F3). 반환: 행 수.

    Raises:
        ValidationError: 원본이 F2 ⑦을 통과하지 않았거나 그 뒤에 바뀌었을 때.
    """
    assert_war_validated(raw_dir, report, c.SEASONS)
    batting = read_table(c.BATTING_SEASONS, c.interim_path(c.BATTING_SEASONS, interim_dir))
    pitching = read_table(c.PITCHING_SEASONS, c.interim_path(c.PITCHING_SEASONS, interim_dir))
    war = build_war_seasons(raw_dir, c.SEASONS, batting, pitching)
    write_table(war, c.WAR_SEASONS, c.interim_path(c.WAR_SEASONS, interim_dir))
    return len(war)


def _load(interim_dir: Path, raw_dir: Path) -> tuple[Tables, pd.DataFrame]:
    tables = load_tables(interim_dir, raw_dir)
    war = read_table(c.WAR_SEASONS, c.interim_path(c.WAR_SEASONS, interim_dir))
    return tables, war


def _require_file(path: Path, what: str) -> Path:
    if not path.exists():
        raise ValueError(f"{what} 파일이 없다: {path}")
    return path


def _require_base(params_path: Path, adoption_path: Path) -> tuple[Params, Adoption]:
    params = load_params(_require_file(params_path, "고정 파라미터"))
    adoption = load_adoption(_require_file(adoption_path, "확장 모델 채택표"))
    require_playing_time_unadopted(adoption)
    return params, adoption


def _print_evaluations(evaluations: pd.DataFrame) -> None:
    cols = ["target_season", "role", "metric", "group", "n", "mae", "baseline_mae_prev"]
    cols.append("baseline_mae_league")
    view = evaluations[evaluations["group"].isin(["all", "war2plus", "regulars"])][cols]
    print(view.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def write_war_json(content: dict[str, Any], web_dir: Path) -> int:
    """war.json을 스키마(WarFile)로 검사한 뒤 그 파일 하나만 쓴다. 반환: 바이트 수.

    `export.write_web`을 쓰지 않는다 — write_web은 넘긴 목록에 없는 `players/*.json`을 지운다.

    Raises:
        ValueError: 스키마 위반(아무것도 쓰지 않는다).
    """
    errors = validate(content, c.WEB_FILES[WAR_WEB_FILE], load_schema(c.WEB_SCHEMA_PATH))
    if errors:
        raise ValueError(f"{WAR_WEB_FILE}: 스키마 WarFile 위반 {len(errors)}건 — {errors[:5]}")
    data = export.dump_json(content).encode("utf-8")
    web_dir.mkdir(parents=True, exist_ok=True)
    (web_dir / WAR_WEB_FILE).write_bytes(data)
    return len(data)


def _tune_war_command(args: argparse.Namespace) -> int:
    try:
        params, _ = _require_base(args.params, args.adoption)
        tables, war = _load(args.interim_dir, args.raw_dir)
        chosen, report = tune_war(tables, war, params, args.targets)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.out_dir / c.TUNE_WAR_CSV.name, index=False)
        evaluations = backtest_war(tables, war, params, chosen, args.targets)
        write_table(evaluations, c.WAR_EVALUATIONS, args.out_dir / c.BACKTEST_WAR_CSV.name)
    except (ValueError, FileNotFoundError) as exc:
        print(f"WAR 조정 중단: {exc}", file=sys.stderr)
        return 1
    save_war_params(chosen, args.war_params_out)
    print(
        f"고른 회귀 표본 {chosen.regress} → {args.war_params_out} (sha256 {war_params_sha256(chosen)})"
    )
    print(f"후보 결과 → {args.out_dir / c.TUNE_WAR_CSV.name}")
    _print_evaluations(evaluations)
    print(f"과거 검증 → {args.out_dir / c.BACKTEST_WAR_CSV.name}")
    return 0


def _backtest_war_command(args: argparse.Namespace) -> int:
    try:
        params, _ = _require_base(args.params, args.adoption)
        war_params = load_war_params(_require_file(args.war_params, "WAR 파라미터"))
        tables, war = _load(args.interim_dir, args.raw_dir)
        evaluations = backtest_war(tables, war, params, war_params, args.targets)
        write_table(evaluations, c.WAR_EVALUATIONS, args.out_dir / c.BACKTEST_WAR_CSV.name)
    except (ValueError, FileNotFoundError) as exc:
        print(f"WAR 과거 검증 중단: {exc}", file=sys.stderr)
        return 1
    _print_evaluations(evaluations)
    print(f"→ {args.out_dir / c.BACKTEST_WAR_CSV.name}")
    return 0


def _final_test_war_command(args: argparse.Namespace) -> int:
    try:
        params, _ = _require_base(args.params, args.adoption)
        war_params = load_war_params(_require_file(args.war_params, "WAR 파라미터"))
        tables, war = _load(args.interim_dir, args.raw_dir)
        record = run_final_test_war(
            tables,
            war,
            params,
            war_params,
            repo=args.repo,
            params_path=args.params,
            war_params_path=args.war_params,
            out_path=args.out,
            now=_utc_stamp(),
            rerun_reason=args.rerun,
        )
    except (WarTestError, ValueError, FileNotFoundError) as exc:
        print(f"2026 WAR 시험 중단: {exc}", file=sys.stderr)
        return 1
    print(
        f"2026 WAR 시험 — 첫 실행 {record['ran_at']}, 다시 돌림 {len(record['reruns'])}번, "
        f"WAR 파라미터 sha256 {record['war_params_sha256'][:12]}, 커밋 {record['commit'][:12]}"
    )
    _print_evaluations(pd.DataFrame(record["rows"]))
    print(f"→ {args.out}")
    return 0


def _read_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _export_war_command(args: argparse.Namespace) -> int:
    try:
        params, adoption = _require_base(args.params, args.adoption)
        war_params = load_war_params(_require_file(args.war_params, "WAR 파라미터"))
        tables, war = _load(args.interim_dir, args.raw_dir)
        target = c.TARGET_SEASON
        # 메타·선수 요약은 기존 내보내기와 같은 값을 쓴다 (같은 고정 파라미터에서)
        projections = final.final_projections(tables, params, adoption, target)
        inp6 = final.build_export_input(
            tables,
            params,
            adoption,
            projections,
            raw_dir=args.raw_dir,
            backtest_csv=args.backtest_csv,
            final_test_path=args.final_test,
        )
        files6 = export.build_web(inp6)
        meta = files6["batters.json"]["meta"]
        summaries = {
            int(row["player"]["player_id"]): row["player"]
            for name in ("batters.json", "pitchers.json")
            for row in files6[name]["rows"]
        }
        proj = {
            role: war_projections(tables, war, params, war_params, role, target) for role in c.ROLES
        }
        rankings = combine_roles(proj["bat"], proj["pit"], war, target)
        backtest = read_table(
            c.WAR_EVALUATIONS, _require_file(args.backtest_war_csv, "WAR 과거 검증표")
        )
        record = _read_record(args.final_test_war)
        evaluations: pd.DataFrame = backtest
        if record is not None:
            rows: list[dict[str, Any]] = record["rows"]
            evaluations = pd.concat(
                [backtest, pd.DataFrame(rows)[c.WAR_EVALUATIONS.column_names]],
                ignore_index=True,
            )
        inp = WarExportInput(
            meta=meta,
            rankings=rankings,
            projections=proj,
            cohorts={
                role: tables.cohorts[
                    (tables.cohorts["role"] == role) & (tables.cohorts["target_season"] == target)
                ].reset_index(drop=True)
                for role in c.ROLES
            },
            players=tables.players,
            player_summaries=summaries,
            war_seasons=war,
            evaluations=evaluations,
            final_test=record,
            war_params_sha256=war_params_sha256(war_params),
        )
        web = build_war_web(inp)
        size = write_war_json(web, args.web_dir)
        write_war_csv(inp, args.out_dir / c.WAR_PROJECTION_CSV.name)
    except (ValueError, FileNotFoundError, WarTestError) as exc:
        print(f"WAR 내보내기 중단: {exc}", file=sys.stderr)
        return 1
    state = "기록 있음" if record is not None else "아직 안 돌림 (final_test = null)"
    print(f"2026 WAR 시험: {state}")
    shown = len(web["rows"])
    print(
        f"{WAR_WEB_FILE}: {size:,} bytes, 2026 출전 {shown:,}명 (순위 {len(rankings):,}명, D42)"
        f" → {args.web_dir}"
    )
    print(f"검토용 CSV → {args.out_dir / c.WAR_PROJECTION_CSV.name}")
    return 0


def add_war_parsers(
    sub: "argparse._SubParsersAction[argparse.ArgumentParser]",
    parse_seasons: Callable[[str], list[int]],
) -> None:
    """WAR 명령 7개를 등록한다."""
    seasons_type = parse_seasons  # cli.parse_seasons (순환 import를 피해 넘겨받는다)
    all_seasons = f"{c.SEASON_FIRST}-{c.SEASON_LAST}"
    tune_default = ",".join(str(t) for t in c.TUNE_TARGETS)

    def base(p: argparse.ArgumentParser) -> None:
        p.add_argument("--interim-dir", type=Path, default=c.INTERIM_DIR)
        p.add_argument("--raw-dir", type=Path, default=c.RAW_DIR)
        p.add_argument("--params", type=Path, default=c.PARAMS_PATH)
        p.add_argument("--adoption", type=Path, default=c.ADOPTION_PATH)

    p = sub.add_parser("collect-war", help="(v0.5) 세이버메트릭스 WAR 원본 수집 (F1 ⑤)")
    p.add_argument("--seasons", type=seasons_type, default=seasons_type(all_seasons))
    p.add_argument("--raw-dir", type=Path, default=c.RAW_DIR)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("validate-war", help="(v0.5) 세이버메트릭스 완전성 검사 (F2 ⑦)")
    p.add_argument("--seasons", type=seasons_type, default=seasons_type(all_seasons))
    p.add_argument("--raw-dir", type=Path, default=c.RAW_DIR)
    p.add_argument("--report", type=Path, default=c.WAR_VALIDATION_REPORT_PATH)

    p = sub.add_parser("build-war", help="(v0.5) WAR 정제 표 war_seasons (F3) — 검사 통과 원본만")
    p.add_argument("--raw-dir", type=Path, default=c.RAW_DIR)
    p.add_argument("--report", type=Path, default=c.WAR_VALIDATION_REPORT_PATH)
    p.add_argument("--interim-dir", type=Path, default=c.INTERIM_DIR)

    p = sub.add_parser(
        "tune-war", help="(v0.5) WAR 회귀 표본 조정 — 2024·2025 → config/war_v1.json"
    )
    base(p)
    p.add_argument("--targets", type=seasons_type, default=seasons_type(tune_default))
    p.add_argument("--out-dir", type=Path, default=c.OUT_DIR)
    p.add_argument("--war-params-out", type=Path, default=c.WAR_PARAMS_PATH)

    p = sub.add_parser("backtest-war", help="(v0.5) WAR 과거 검증표 — 2024·2025만")
    base(p)
    p.add_argument("--targets", type=seasons_type, default=seasons_type(tune_default))
    p.add_argument("--war-params", type=Path, default=c.WAR_PARAMS_PATH)
    p.add_argument("--out-dir", type=Path, default=c.OUT_DIR)

    p = sub.add_parser(
        "final-test-war", help="(v0.5) 2026 WAR 시험 — 커밋된 war_v1.json으로 한 번만 (D36)"
    )
    base(p)
    p.add_argument("--target", type=int, choices=[c.FINAL_TEST_TARGET], required=True)
    p.add_argument(
        "--rerun", metavar="사유", help="이미 돌렸을 때만: 다시 돌리는 사유 (기록에 남는다)"
    )
    p.add_argument("--war-params", type=Path, default=c.WAR_PARAMS_PATH)
    p.add_argument("--out", type=Path, default=c.FINAL_TEST_WAR_PATH)
    p.add_argument("--repo", type=Path, default=Path("."))

    p = sub.add_parser("export-war", help="(v0.5) 2027 WAR 순위 → web war.json + 검토용 CSV")
    base(p)
    p.add_argument("--war-params", type=Path, default=c.WAR_PARAMS_PATH)
    p.add_argument("--backtest-csv", type=Path, default=c.BACKTEST_CSV)
    p.add_argument("--final-test", type=Path, default=c.FINAL_TEST_PATH)
    p.add_argument("--backtest-war-csv", type=Path, default=c.BACKTEST_WAR_CSV)
    p.add_argument("--final-test-war", type=Path, default=c.FINAL_TEST_WAR_PATH)
    p.add_argument("--out-dir", type=Path, default=c.OUT_DIR)
    p.add_argument("--web-dir", type=Path, default=c.WEB_DATA_DIR)


def run_war_command(args: argparse.Namespace, getter: Getter | None) -> int:
    """WAR 명령을 실행한다. 반환: 종료 코드."""
    if args.command == "collect-war":
        fetcher = getter if getter is not None else Fetcher()
        try:
            outcomes = run_collect_war(fetcher, args.raw_dir, args.seasons, force=args.force)
        except (FetchError, ValueError) as exc:
            print(f"WAR 수집 실패: {exc}", file=sys.stderr)
            return 1
        fetched = sum(o.fetched for o in outcomes)
        print(f"세이버메트릭스 {len(outcomes)}개 확보 (새로 받음 {fetched}) → {args.raw_dir}")
        return 0
    if args.command == "validate-war":
        reports = validate_war_seasons(args.raw_dir, args.seasons)
        write_war_report(reports, args.report)
        for report in reports:
            print(f"[{report.season}] {'통과' if report.passed else '실패'}")
            for check in report.checks:
                print(f"  {'✓' if check.passed else '✗'} {check.name}: {check.detail}")
        failed = [r.season for r in reports if not r.passed]
        print(
            f"보고서: {args.report}" + (f" — 실패 시즌 {failed}" if failed else " — 전 시즌 통과")
        )
        return 1 if failed else 0
    if args.command == "build-war":
        try:
            rows = run_build_war(args.raw_dir, args.report, args.interim_dir)
        except (ValidationError, ValueError) as exc:
            print(f"WAR 정제 중단: {exc}", file=sys.stderr)
            return 1
        print(f"{c.WAR_SEASONS.name}: {rows:,}행 → {args.interim_dir}")
        return 0
    if args.command == "tune-war":
        return _tune_war_command(args)
    if args.command == "backtest-war":
        return _backtest_war_command(args)
    if args.command == "final-test-war":
        return _final_test_war_command(args)
    if args.command == "export-war":
        return _export_war_command(args)
    raise ValueError(f"WAR 명령이 아니다: {args.command}")


__all__ = ["WAR_COMMANDS", "WarParams", "add_war_parsers", "run_war_command"]
