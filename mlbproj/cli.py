"""명령줄 진입점 (PLAN §2-4). 명령은 마일스톤마다 붙인다."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mlbproj import __version__
from mlbproj import contracts as c
from mlbproj.collect import savant, statsapi
from mlbproj.collect.http import Fetcher, FetchError
from mlbproj.collect.raw import CollectOutcome, Getter
from mlbproj.validate import DEFAULT_LIMITS, validate_seasons, write_report


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

    parser.error(f"알 수 없는 명령: {args.command}")
    return 2
