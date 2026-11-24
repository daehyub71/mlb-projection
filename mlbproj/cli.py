"""명령줄 진입점 (PLAN §2-4). 명령은 마일스톤마다 붙인다."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from mlbproj import __version__
from mlbproj import contracts as c
from mlbproj.backtest import load_tables, run_backtest, summary
from mlbproj.clean.batting import build_batting
from mlbproj.clean.fielding import build_fielding
from mlbproj.clean.pitching import build_pitching
from mlbproj.clean.players import build_players, people_records
from mlbproj.cohort import build_cohort
from mlbproj.collect import savant, statsapi
from mlbproj.collect.http import Fetcher, FetchError
from mlbproj.collect.raw import CollectOutcome, Getter
from mlbproj.interim import write_table
from mlbproj.models.params import DEFAULT_PARAMS, Params, load_params, params_sha256, save_params
from mlbproj.tune import tune
from mlbproj.validate import (
    DEFAULT_LIMITS,
    ValidationError,
    assert_validated,
    validate_seasons,
    write_report,
)

# 정제·대상 표를 만드는 목표 시즌: 과거 검증(2024~2026)과 최종 예측(2027)
BUILD_TARGETS: tuple[int, ...] = (*c.TUNE_TARGETS, c.FINAL_TEST_TARGET, c.TARGET_SEASON)
TUNE_DEFAULT: str = ",".join(str(t) for t in c.TUNE_TARGETS)


def parse_seasons(text: str) -> list[int]:
    """시즌 인자를 목록으로 바꾼다. `2019-2026`(범위) 또는 `2024,2025`(나열).

    Args:
        text: 인자 문자열.

    Returns:
        오름차순 시즌 목록 (중복 없음).

    Raises:
        argparse.ArgumentTypeError: 형식이 틀렸거나 수집 범위(contracts.SEASONS) 밖일 때.
    """
    try:
        if "-" in text:
            first, last = (int(part) for part in text.split("-", 1))
            seasons = list(range(first, last + 1))
        else:
            seasons = [int(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"시즌 형식이 틀렸다: {text!r} (예: 2019-2026, 2024,2025)"
        ) from exc
    outside = sorted(set(seasons) - set(c.SEASONS))
    if not seasons or outside:
        raise argparse.ArgumentTypeError(
            f"시즌은 {c.SEASON_FIRST}~{c.SEASON_LAST} 안에서 고른다 (범위 밖: {outside})"
        )
    return sorted(set(seasons))


def run_collect(
    getter: Getter, raw_dir: Path, seasons: Sequence[int], *, force: bool
) -> list[CollectOutcome]:
    """원본을 모두 확보한다 (F1). 순서: API 시즌 기록 → Savant 리더보드·수비 → 선수 정보.

    Args:
        getter: 요청 객체.
        raw_dir: 원본 디렉토리.
        seasons: 시즌들.
        force: 캐시가 있어도 다시 받는다.

    Returns:
        확보 결과 목록 (받은 것과 캐시를 쓴 것 모두).
    """
    outcomes: list[CollectOutcome] = []
    for season in seasons:
        for role in c.ROLES:
            outcomes.append(
                statsapi.collect_season_stats(getter, raw_dir, season, role, force=force)
            )
    for season in seasons:
        for role in c.ROLES:
            outcomes.append(savant.collect_leaderboard(getter, raw_dir, season, role, force=force))
        if season in c.FIELDING_YEARS:
            outcomes.append(savant.collect_oaa(getter, raw_dir, season, force=force))
            outcomes.append(savant.collect_frv(getter, raw_dir, season, force=force))
    player_ids = statsapi.stats_player_ids(raw_dir, seasons)
    outcomes.extend(statsapi.collect_people(getter, raw_dir, player_ids, force=force))
    return outcomes


def run_build(raw_dir: Path, report: Path, out_dir: Path) -> dict[str, int]:
    """검증을 통과한 원본으로 정제 표와 대상 표를 만든다 (F3~F5).

    Args:
        raw_dir: 원본 디렉토리.
        report: 검증 보고서 경로 (`assert_validated`로 확인한다).
        out_dir: 정제 표를 쓸 디렉토리.

    Returns:
        표 이름 → 행 수.

    Raises:
        ValidationError: 원본이 검증을 통과하지 않았거나 검증 뒤에 바뀌었을 때.
    """
    assert_validated(raw_dir, report, c.SEASONS)
    batting = build_batting(raw_dir)
    pitching = build_pitching(raw_dir)
    tables = {
        c.PLAYERS: build_players(raw_dir),
        c.BATTING_SEASONS: batting,
        c.PITCHING_SEASONS: pitching,
        c.FIELDING_SEASONS: build_fielding(raw_dir),
    }
    seasons_by_role = {"bat": batting, "pit": pitching}
    # 정제 표는 2019부터라 그 전 기록은 MLB 데뷔 연도로 보완한다 (F5, E21)
    debuts = {
        int(str(p["id"])): int(str(p["mlbDebutDate"])[:4])
        for p in people_records(raw_dir)
        if p.get("mlbDebutDate")
    }
    cohorts = [
        build_cohort(seasons_by_role[role], role, target, debut_seasons=debuts)
        for target in BUILD_TARGETS
        for role in c.ROLES
    ]
    tables[c.COHORTS] = pd.concat(cohorts, ignore_index=True)
    for spec, df in tables.items():
        write_table(df, spec, c.interim_path(spec, out_dir))
    return {spec.name: len(df) for spec, df in tables.items()}


def reject_final_target(targets: Sequence[int]) -> None:
    """과거 검증·조정 목표에 최종 검증 시즌(2026) 이후가 있으면 멈춘다 (SPEC F10).

    Raises:
        ValueError: 그런 목표가 있을 때.
    """
    late = [t for t in targets if t >= c.FINAL_TEST_TARGET]
    if late:
        raise ValueError(
            f"{late}은 과거 검증·조정에 쓰지 않는다 — 최종 검증은 final-test로 한 번만 (SPEC F10)"
        )


def run_backtest_command(
    params: Params, targets: Sequence[int], interim_dir: Path, raw_dir: Path, out_dir: Path
) -> pd.DataFrame:
    """과거 검증표를 `out_dir/backtest.csv`·`backtest.json`으로 쓰고 평가표를 돌려준다."""
    reject_final_target(targets)
    evaluations = run_backtest(load_tables(interim_dir, raw_dir), params, targets)
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluations.to_csv(out_dir / "backtest.csv", index=False, lineterminator="\n")
    payload = {
        "params_version": params.version,
        "params_sha256": params_sha256(params),
        "targets": list(targets),
        "rows": json.loads(evaluations.to_json(orient="records")),
    }
    (out_dir / "backtest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return evaluations


def run_tune_command(
    targets: Sequence[int], interim_dir: Path, raw_dir: Path, out_dir: Path, params_out: Path
) -> Params:
    """격자 탐색 결과를 `out_dir/tune_report.csv`에, 고른 파라미터를 params_out에 쓴다."""
    reject_final_target(targets)
    chosen, report = tune(load_tables(interim_dir, raw_dir), targets)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "tune_report.csv", index=False, lineterminator="\n")
    save_params(chosen, params_out)
    return chosen


def build_parser() -> argparse.ArgumentParser:
    """명령줄 파서를 만든다.

    Returns:
        하위 명령을 담은 파서.
    """
    parser = argparse.ArgumentParser(
        prog="mlbproj",
        description="재미로 보는 MLB 선수 성적 예측 — 수집 → 검증 → 정제 → 예측 → 내보내기",
    )
    parser.add_argument("--version", action="version", version=f"mlbproj {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    all_seasons = f"{c.SEASON_FIRST}-{c.SEASON_LAST}"

    collect = sub.add_parser("collect", help="원본 수집 (F1) — MLB Stats API, Baseball Savant")
    collect.add_argument("--seasons", type=parse_seasons, default=parse_seasons(all_seasons))
    collect.add_argument("--raw-dir", type=Path, default=c.RAW_DIR)
    collect.add_argument("--force", action="store_true", help="캐시가 있어도 다시 받는다")

    validate = sub.add_parser("validate", help="수집 완전성 검증 (F2)")
    validate.add_argument("--seasons", type=parse_seasons, default=parse_seasons(all_seasons))
    validate.add_argument("--raw-dir", type=Path, default=c.RAW_DIR)
    validate.add_argument("--report", type=Path, default=c.VALIDATION_REPORT_PATH)

    build = sub.add_parser("build", help="정제 표·예측 대상 만들기 (F3~F5) — 검증 통과 원본만")
    build.add_argument("--raw-dir", type=Path, default=c.RAW_DIR)
    build.add_argument("--report", type=Path, default=c.VALIDATION_REPORT_PATH)
    build.add_argument("--out-dir", type=Path, default=c.INTERIM_DIR)

    backtest = sub.add_parser("backtest", help="과거 검증표 (F10) — 2024·2025만, 2026은 final-test")
    backtest.add_argument("--targets", type=parse_seasons, default=parse_seasons(TUNE_DEFAULT))
    backtest.add_argument(
        "--params", type=Path, default=c.PARAMS_PATH, help="없으면 기본값(v0)으로 돈다"
    )
    backtest.add_argument("--interim-dir", type=Path, default=c.INTERIM_DIR)
    backtest.add_argument("--raw-dir", type=Path, default=c.RAW_DIR)
    backtest.add_argument("--out-dir", type=Path, default=c.OUT_DIR)

    tune_cmd = sub.add_parser("tune", help="파라미터 조정 (D10) — 2024·2025 격자 탐색")
    tune_cmd.add_argument("--targets", type=parse_seasons, default=parse_seasons(TUNE_DEFAULT))
    tune_cmd.add_argument("--interim-dir", type=Path, default=c.INTERIM_DIR)
    tune_cmd.add_argument("--raw-dir", type=Path, default=c.RAW_DIR)
    tune_cmd.add_argument("--out-dir", type=Path, default=c.OUT_DIR)
    tune_cmd.add_argument("--params-out", type=Path, default=c.PARAMS_PATH)
    return parser


def main(argv: list[str] | None = None, *, getter: Getter | None = None) -> int:
    """명령을 실행한다.

    Args:
        argv: 인자 목록. None이면 sys.argv를 쓴다.
        getter: 수집에 쓸 요청 객체 (테스트용). None이면 실제 Fetcher.

    Returns:
        종료 코드. 실패하면 1.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "collect":
        fetcher = getter if getter is not None else Fetcher()
        try:
            outcomes = run_collect(fetcher, args.raw_dir, args.seasons, force=args.force)
        except (FetchError, ValueError) as exc:
            print(f"수집 실패: {exc}", file=sys.stderr)
            return 1
        fetched = sum(o.fetched for o in outcomes)
        size = sum(o.entry.bytes for o in outcomes)
        print(
            f"원본 {len(outcomes)}개 확보 (새로 받음 {fetched}, 캐시 {len(outcomes) - fetched}), {size:,} bytes → {args.raw_dir}"
        )
        return 0

    if args.command == "validate":
        reports = validate_seasons(args.raw_dir, args.seasons, limits=DEFAULT_LIMITS)
        write_report(reports, args.report)
        for report in reports:
            mark = "통과" if report.passed else "실패"
            print(f"[{report.season}] {mark}")
            for check in report.checks:
                print(f"  {'✓' if check.passed else '✗'} {check.name}: {check.detail}")
            for merged in report.merged_duplicates:
                print(f"  · 같은 행 합침: {merged}")
        failed = [r.season for r in reports if not r.passed]
        print(
            f"보고서: {args.report}" + (f" — 실패 시즌 {failed}" if failed else " — 전 시즌 통과")
        )
        return 1 if failed else 0

    if args.command == "build":
        try:
            counts = run_build(args.raw_dir, args.report, args.out_dir)
        except ValidationError as exc:
            print(f"정제 중단 — 검증을 먼저 통과해야 한다: {exc}", file=sys.stderr)
            return 1
        for name, rows in counts.items():
            print(f"{name}: {rows:,}행")
        print(f"→ {args.out_dir}")
        return 0

    if args.command == "backtest":
        params = load_params(args.params) if args.params.exists() else DEFAULT_PARAMS
        try:
            evaluations = run_backtest_command(
                params, args.targets, args.interim_dir, args.raw_dir, args.out_dir
            )
        except ValueError as exc:
            print(f"과거 검증 중단: {exc}", file=sys.stderr)
            return 1
        print(f"파라미터 {params.version} (sha256 {params_sha256(params)[:12]})")
        print(summary(evaluations).to_string())
        print(f"→ {args.out_dir / 'backtest.csv'}, {args.out_dir / 'backtest.json'}")
        return 0

    if args.command == "tune":
        try:
            chosen = run_tune_command(
                args.targets, args.interim_dir, args.raw_dir, args.out_dir, args.params_out
            )
        except ValueError as exc:
            print(f"조정 중단: {exc}", file=sys.stderr)
            return 1
        print(f"고른 파라미터 → {args.params_out} (sha256 {params_sha256(chosen)})")
        print(f"후보 결과 → {args.out_dir / 'tune_report.csv'}")
        return 0

    parser.error(f"알 수 없는 명령: {args.command}")
    return 2
