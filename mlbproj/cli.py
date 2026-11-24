"""명령줄 진입점 (PLAN §2-4). 명령은 마일스톤마다 붙인다."""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from mlbproj import __version__, export, final
from mlbproj import contracts as c
from mlbproj.backtest import load_tables, summary
from mlbproj.clean.batting import build_batting
from mlbproj.clean.fielding import build_fielding
from mlbproj.clean.pitching import build_pitching
from mlbproj.clean.players import build_players, people_records
from mlbproj.cohort import build_cohort
from mlbproj.collect import savant, statsapi
from mlbproj.collect.http import Fetcher, FetchError
from mlbproj.collect.raw import CollectOutcome, Getter
from mlbproj.interim import write_table
from mlbproj.models.params import (
    DEFAULT_PARAMS,
    NO_ADOPTION,
    Adoption,
    Params,
    adoption_sha256,
    load_adoption,
    load_params,
    params_sha256,
    save_adoption,
    save_params,
)
from mlbproj.schema import load_schema
from mlbproj.tables import Tables
from mlbproj.tune import extended_scorer, select_adoption, tune
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
    params: Params,
    adoption: Adoption,
    targets: Sequence[int],
    interim_dir: Path,
    raw_dir: Path,
    out_dir: Path,
) -> pd.DataFrame:
    """과거 검증표를 `out_dir/backtest.csv`·`backtest.json`으로 쓰고 평가표를 돌려준다.

    모델: league·baseline·weighted, 그리고 채택표로 최종 모델이 extended인 역할은 extended
    (`final.evaluate_target`, refresh와 같은 경로).
    """
    reject_final_target(targets)
    tables = load_tables(interim_dir, raw_dir)
    evaluations = pd.concat(
        [final.evaluate_target(tables, params, adoption, t) for t in targets], ignore_index=True
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_backtest(evaluations, params, targets, out_dir / "backtest.csv")
    return evaluations.sort_values(list(c.EVALUATIONS.keys), kind="mergesort").reset_index(
        drop=True
    )


def run_adopt_command(
    params: Params,
    targets: Sequence[int],
    interim_dir: Path,
    raw_dir: Path,
    out: Path,
    report: Path,
) -> Adoption:
    """확장 모델 채택표(F9)를 2024·2025로 정해 out에, 지표별 결과를 report CSV에 쓴다."""
    reject_final_target(targets)
    tables = load_tables(interim_dir, raw_dir)
    adoption, table = select_adoption(extended_scorer(tables, params, targets), targets)
    report.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(report, index=False, lineterminator="\n")
    save_adoption(adoption, out)
    return adoption


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


def _utc_stamp() -> str:
    """지금 UTC 시각 ("YYYY-MM-DDTHH:MM:SSZ")."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_params(path: Path) -> Params:
    """고정 파라미터를 읽는다. 없으면 ValueError — 기본값(v0)으로 대신 돌지 않는다."""
    if not path.exists():
        raise ValueError(f"고정 파라미터 파일이 없다: {path} — tune으로 만들고 커밋한 뒤 돌린다")
    return load_params(path)


def _require_adoption(path: Path) -> Adoption:
    """채택표를 읽는다. 없으면 ValueError — 채택 없음(NO_ADOPTION)으로 대신 돌지 않는다."""
    if not path.exists():
        raise ValueError(
            f"확장 모델 채택표 파일이 없다: {path} — 2024·2025로 정해 커밋한 뒤 돌린다"
        )
    return load_adoption(path)


def _adoption_or_none(path: Path) -> Adoption:
    """채택표를 읽는다. 없으면 경고하고 채택 없음(두 역할 모두 weighted)으로 둔다."""
    if path.exists():
        return load_adoption(path)
    print(
        f"경고: 채택표 {path}가 없어 채택 없음({NO_ADOPTION.version})으로 한다 — "
        "두 역할 모두 weighted",
        file=sys.stderr,
    )
    return NO_ADOPTION


def _write_backtest(
    evaluations: pd.DataFrame, params: Params, targets: Sequence[int], csv_path: Path
) -> None:
    """과거 검증표를 csv_path(계약 확인·키 순)와 같은 곳의 backtest.json으로 쓴다 (backtest 명령과 같은 모양).

    JSON 행도 CSV와 같은 키 순이라 두 파일의 행 순서가 언제나 같다.
    """
    write_table(evaluations, c.EVALUATIONS, csv_path)
    evaluations = evaluations.sort_values(list(c.EVALUATIONS.keys), kind="mergesort")
    payload = {
        "params_version": params.version,
        "params_sha256": params_sha256(params),
        "targets": list(targets),
        "rows": json.loads(evaluations.to_json(orient="records")),
    }
    csv_path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def _missing_model_rows(inp: export.ExportInput) -> list[str]:
    """역할마다 최종 모델의 검증 행이 빠진 목표 시즌을 경고 문장으로 (없으면 빈 목록).

    M3 `backtest` 명령은 "extended" 행을 만들지 않는다. 최종 모델이 extended인데 그 행이 없으면
    선수 상세의 그룹 오차(F14)와 검증 페이지가 조용히 비므로 알린다. 목표 = 2024·2025, 최종 검증
    기록이 있으면 2026도.
    """
    targets = [*c.TUNE_TARGETS, *([c.FINAL_TEST_TARGET] if inp.final_test is not None else [])]
    rows = inp.evaluations
    warnings: list[str] = []
    for role in c.ROLES:
        model = inp.model[role]
        mine = rows[(rows["role"] == role) & (rows["model"] == model)]
        missing = [t for t in targets if not bool((mine["target_season"] == t).any())]
        if missing:
            warnings.append(
                f"경고: 검증표에 {role} 최종 모델 {model}의 {missing} 행이 없다 — 선수 상세의 "
                "그룹 오차가 null로 나간다. refresh로 과거 검증표를 다시 만든다"
            )
    return warnings


def _export(
    tables: Tables,
    params: Params,
    adoption: Adoption,
    *,
    raw_dir: Path,
    backtest_csv: Path,
    final_test_path: Path,
    web_dir: Path,
    csv_paths: Mapping[c.Role, Path],
) -> None:
    """2027 최종 예측 → 내보내기 입력 → 웹 JSON·검토용 CSV (F11). 최종 검증은 돌리지 않는다.

    Raises:
        ValueError: 입력이 계약을 어기거나 웹 JSON이 스키마·크기 검사를 통과하지 못할 때.
    """
    projections = final.final_projections(tables, params, adoption, c.TARGET_SEASON)
    inp = final.build_export_input(
        tables,
        params,
        adoption,
        projections,
        raw_dir=raw_dir,
        backtest_csv=backtest_csv,
        final_test_path=final_test_path,
    )
    for warning in _missing_model_rows(inp):
        print(warning, file=sys.stderr)
    files = export.build_web(inp)
    sizes = export.write_web(files, web_dir, load_schema(c.WEB_SCHEMA_PATH))
    written = export.write_projection_csv(inp, csv_paths)

    print(
        f"최종 모델: 타자 {inp.model['bat']}, 투수 {inp.model['pit']} "
        f"(파라미터 {params.version}, 채택표 {adoption.version})"
    )
    state = "기록 있음" if inp.final_test is not None else "아직 안 돌림 (final_test = null)"
    print(f"2026 최종 검증: {state} — {final_test_path}")
    for name in c.WEB_FIRST_SCREEN_FILES:
        print(f"{name}: {sizes.get(name, 0):,} bytes (한도 {c.WEB_FIRST_SCREEN_MAX_BYTES:,})")
    players = sum(1 for rel in sizes if rel.startswith("players/"))
    print(
        f"웹 JSON {len(sizes):,}개 (선수 상세 {players:,}개), "
        f"합계 {sum(sizes.values()):,} bytes → {web_dir}"
    )
    print("검토용 CSV → " + ", ".join(str(path) for path in written.values()))


def _final_test_command(args: argparse.Namespace) -> int:
    """final-test: 커밋된 파라미터·채택표로 2026 최종 검증을 한 번 돌린다 (F10)."""
    try:
        params = _require_params(args.params)
        adoption = _require_adoption(args.adoption)
        tables = load_tables(args.interim_dir, args.raw_dir)
        record = final.run_final_test(
            tables,
            params,
            adoption,
            repo=args.repo,
            params_path=args.params,
            adoption_path=args.adoption,
            out_path=args.out,
            now=_utc_stamp(),
            rerun_reason=args.rerun,
        )
    except (final.FinalTestError, ValueError, FileNotFoundError) as exc:
        print(f"최종 검증 중단: {exc}", file=sys.stderr)
        return 1
    rows = pd.DataFrame(record["rows"]).astype({"mae": "float64"})
    print(
        f"최종 검증 {record['target_season']} — 첫 실행 {record['ran_at']}, "
        f"다시 돌림 {len(record['reruns'])}번"
    )
    print(
        f"파라미터 sha256 {record['params_sha256'][:12]}, "
        f"채택표 sha256 {record['adoption_sha256'][:12]}, 커밋 {record['commit'][:12]}"
    )
    print(summary(rows).to_string())  # "all" 그룹 MAE
    print(f"→ {args.out}")
    return 0


def _export_command(args: argparse.Namespace) -> int:
    """export: 2027 최종 예측 CSV와 웹 JSON을 쓴다 (F11)."""
    try:
        params = _require_params(args.params)
        adoption = _adoption_or_none(args.adoption)
        tables = load_tables(args.interim_dir, args.raw_dir)
        _export(
            tables,
            params,
            adoption,
            raw_dir=args.raw_dir,
            backtest_csv=args.backtest_csv,
            final_test_path=args.final_test,
            web_dir=args.web_dir,
            csv_paths=c.PROJECTION_CSV,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"내보내기 중단: {exc}", file=sys.stderr)
        return 1
    return 0


def _refresh_command(args: argparse.Namespace, getter: Getter | None) -> int:
    """refresh: 수집 → 검증 → 정제 → 과거 검증(2024·2025) → 내보내기 (D25 수동 갱신).

    파라미터·채택표는 고정 파일(`PARAMS_PATH`·`ADOPTION_PATH`)을 쓴다. 최종 검증은 돌리지 않고,
    out_dir에 기록이 있으면 내보내기에 그대로 싣는다.
    """
    try:
        params = _require_params(c.PARAMS_PATH)
        adoption = _adoption_or_none(c.ADOPTION_PATH)
    except ValueError as exc:
        print(f"갱신 중단: {exc}", file=sys.stderr)
        return 1

    fetcher = getter if getter is not None else Fetcher()
    try:
        outcomes = run_collect(fetcher, args.raw_dir, c.SEASONS, force=args.force)
    except (FetchError, ValueError) as exc:
        print(f"갱신 중단 — 수집 실패: {exc}", file=sys.stderr)
        return 1
    fetched = sum(o.fetched for o in outcomes)
    print(f"수집: 원본 {len(outcomes)}개 (새로 받음 {fetched}, 캐시 {len(outcomes) - fetched})")

    report_path = args.interim_dir / c.VALIDATION_REPORT_PATH.name
    reports = validate_seasons(args.raw_dir, c.SEASONS, limits=DEFAULT_LIMITS)
    write_report(reports, report_path)
    failed = [r for r in reports if not r.passed]
    if failed:
        detail = "; ".join(
            f"{r.season}({', '.join(check.name for check in r.checks if not check.passed)})"
            for r in failed
        )
        print(f"갱신 중단 — 검증 실패: {detail} (보고서 {report_path})", file=sys.stderr)
        return 1
    print(f"검증: {len(reports)}시즌 통과 → {report_path}")

    try:
        counts = run_build(args.raw_dir, report_path, args.interim_dir)
    except (ValidationError, ValueError) as exc:  # ValueError: 정제 표가 계약을 어길 때
        print(f"갱신 중단 — 정제 실패: {exc}", file=sys.stderr)
        return 1
    print(f"정제: 표 {len(counts)}개 → {args.interim_dir}")

    backtest_csv = args.out_dir / c.BACKTEST_CSV.name
    try:
        tables = load_tables(args.interim_dir, args.raw_dir)
        evaluations = pd.concat(
            [final.evaluate_target(tables, params, adoption, t) for t in c.TUNE_TARGETS],
            ignore_index=True,
        )
        _write_backtest(evaluations, params, c.TUNE_TARGETS, backtest_csv)
        print(f"과거 검증 {list(c.TUNE_TARGETS)} → {backtest_csv}")
        print(summary(evaluations).to_string())
        _export(
            tables,
            params,
            adoption,
            raw_dir=args.raw_dir,
            backtest_csv=backtest_csv,
            final_test_path=args.out_dir / c.FINAL_TEST_PATH.name,
            web_dir=args.web_dir,
            csv_paths={role: args.out_dir / c.PROJECTION_CSV[role].name for role in c.ROLES},
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"갱신 중단: {exc}", file=sys.stderr)
        return 1
    return 0


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
    backtest.add_argument(
        "--adoption", type=Path, default=c.ADOPTION_PATH, help="없으면 채택 없음(extended 행 없음)"
    )

    adopt = sub.add_parser(
        "adopt", help="확장 모델 채택표 (F9) — 2024·2025, 지표마다 MAE가 줄 때만"
    )
    adopt.add_argument("--targets", type=parse_seasons, default=parse_seasons(TUNE_DEFAULT))
    adopt.add_argument("--params", type=Path, default=c.PARAMS_PATH)
    adopt.add_argument("--interim-dir", type=Path, default=c.INTERIM_DIR)
    adopt.add_argument("--raw-dir", type=Path, default=c.RAW_DIR)
    adopt.add_argument("--out", type=Path, default=c.ADOPTION_PATH)
    adopt.add_argument("--report", type=Path, default=c.OUT_DIR / "adoption_report.csv")

    tune_cmd = sub.add_parser("tune", help="파라미터 조정 (D10) — 2024·2025 격자 탐색")
    tune_cmd.add_argument("--targets", type=parse_seasons, default=parse_seasons(TUNE_DEFAULT))
    tune_cmd.add_argument("--interim-dir", type=Path, default=c.INTERIM_DIR)
    tune_cmd.add_argument("--raw-dir", type=Path, default=c.RAW_DIR)
    tune_cmd.add_argument("--out-dir", type=Path, default=c.OUT_DIR)
    tune_cmd.add_argument("--params-out", type=Path, default=c.PARAMS_PATH)

    final_cmd = sub.add_parser(
        "final-test", help="2026 최종 검증 (F10) — 커밋된 파라미터·채택표로 한 번만"
    )
    final_cmd.add_argument(
        "--target",
        type=int,
        choices=[c.FINAL_TEST_TARGET],
        required=True,
        help="최종 검증 시즌 (2026만 받는다)",
    )
    final_cmd.add_argument(
        "--rerun", metavar="사유", help="이미 돌렸을 때만: 다시 돌리는 사유 (기록에 남는다)"
    )
    final_cmd.add_argument("--interim-dir", type=Path, default=c.INTERIM_DIR)
    final_cmd.add_argument("--raw-dir", type=Path, default=c.RAW_DIR)
    final_cmd.add_argument("--params", type=Path, default=c.PARAMS_PATH, help="커밋된 파라미터")
    final_cmd.add_argument(
        "--adoption", type=Path, default=c.ADOPTION_PATH, help="커밋된 확장 모델 채택표"
    )
    final_cmd.add_argument("--out", type=Path, default=c.FINAL_TEST_PATH)
    final_cmd.add_argument("--repo", type=Path, default=Path("."), help="커밋을 확인할 git 저장소")

    export_cmd = sub.add_parser(
        "export", help="2027 최종 예측 CSV + 웹 JSON (F11) — 최종 검증은 돌리지 않는다"
    )
    export_cmd.add_argument("--interim-dir", type=Path, default=c.INTERIM_DIR)
    export_cmd.add_argument("--raw-dir", type=Path, default=c.RAW_DIR)
    export_cmd.add_argument("--params", type=Path, default=c.PARAMS_PATH)
    export_cmd.add_argument(
        "--adoption", type=Path, default=c.ADOPTION_PATH, help="없으면 채택 없음(weighted)"
    )
    export_cmd.add_argument("--backtest-csv", type=Path, default=c.BACKTEST_CSV)
    export_cmd.add_argument(
        "--final-test", type=Path, default=c.FINAL_TEST_PATH, help="있으면 2026 행을 싣는다"
    )
    export_cmd.add_argument("--web-dir", type=Path, default=c.WEB_DATA_DIR)

    refresh = sub.add_parser(
        "refresh", help="수동 갱신 (D25) — 수집 → 검증 → 정제 → 과거 검증 → 내보내기"
    )
    refresh.add_argument("--raw-dir", type=Path, default=c.RAW_DIR)
    refresh.add_argument("--interim-dir", type=Path, default=c.INTERIM_DIR)
    refresh.add_argument("--out-dir", type=Path, default=c.OUT_DIR)
    refresh.add_argument("--web-dir", type=Path, default=c.WEB_DATA_DIR)
    refresh.add_argument("--force", action="store_true", help="캐시가 있어도 다시 받는다")
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
        adoption = load_adoption(args.adoption) if args.adoption.exists() else NO_ADOPTION
        try:
            evaluations = run_backtest_command(
                params, adoption, args.targets, args.interim_dir, args.raw_dir, args.out_dir
            )
        except ValueError as exc:
            print(f"과거 검증 중단: {exc}", file=sys.stderr)
            return 1
        print(
            f"파라미터 {params.version} (sha256 {params_sha256(params)[:12]}), "
            f"채택표 {adoption.version} (sha256 {adoption_sha256(adoption)[:12]})"
        )
        print(summary(evaluations).to_string())
        print(f"→ {args.out_dir / 'backtest.csv'}, {args.out_dir / 'backtest.json'}")
        return 0

    if args.command == "adopt":
        try:
            chosen_adoption = run_adopt_command(
                _require_params(args.params),
                args.targets,
                args.interim_dir,
                args.raw_dir,
                args.out,
                args.report,
            )
        except ValueError as exc:
            print(f"채택 중단: {exc}", file=sys.stderr)
            return 1
        on = [m for m in c.EXT_METRICS if chosen_adoption.adopt[m]]
        print(f"채택: {on or '없음'} → {args.out} (sha256 {adoption_sha256(chosen_adoption)})")
        print(f"지표별 결과 → {args.report}")
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

    if args.command == "final-test":
        return _final_test_command(args)
    if args.command == "export":
        return _export_command(args)
    if args.command == "refresh":
        return _refresh_command(args, getter)

    parser.error(f"알 수 없는 명령: {args.command}")
    return 2
