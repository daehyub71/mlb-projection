"""세이버메트릭스(WAR) 수집 완전성 검사 — SPEC v0.5 F2 ⑦, E26. 레인 A가 구현한다.

기존 시즌 검증(`validate.py`, F2 ①~⑥)과 보고서는 건드리지 않는다. 시즌마다 두 역할 원본을 함께 보고
`contracts.WAR_CHECK_NAMES` 순서의 검사 4개를 낸다(이름은 contracts 주석 참고).

- saber_complete: `raw_saber_path(season, "bat"/"pit")` 두 파일이 있고, JSON이며 `stats[0].splits` 목록,
  `totalSplits` = 행 수, 모든 행에 `SABER_SPLIT_FIELDS`와 `stat`의 `SABER_STAT_FIELDS[role]` 키가 있다.
  (M8 리뷰에서 더함) 원본 전체 행에 같은 `player.id`가 두 번 없고, 모든 행의 `season` 값이 str(시즌)이다.
- saber_war_present: 모든 행의 `stat.war`가 유한한 숫자다(bool 아님). 결측·문자열이면 실패(어느 선수인지 적는다).
- saber_covers_roles: 같은 시즌의 시즌 기록 원본(`raw_statsapi_path`)에서 역할 행(D8 — 타자: 포지션이
  `BATTER_EXCLUDED_POSITIONS`가 아니고 타석이 1 이상인 타격 행, 투수: 포지션이 `PITCHER_POSITIONS`인 투구 행)의 선수 ID가
  모두 그 역할 세이버메트릭스에 있다. 세이버메트릭스에만 있는 ID는 실패가 아니다(투수 타격·야수 등판, E26).
- saber_league_sum: 역할마다 원본 전체 행의 war 합이 `league_war[role] × season_games(season) / 162`에서
  `tolerance` 안이다(테스트는 잘라 낸 픽스처에 맞는 기준값을 넘긴다).

파일이 없거나 형식이 틀려도 예외 대신 실패한 검사로 담는다. 통과 detail에는 핵심 수치를 적는다
(예: "타격 570.0 · 투구 430.0 (기준 570.0 · 430.0, ±1.0)").
"""

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeGuard

from mlbproj.collect import raw
from mlbproj.contracts import (
    BATTER_EXCLUDED_POSITIONS,
    FULL_SEASON_GAMES,
    LEAGUE_WAR,
    LEAGUE_WAR_TOLERANCE,
    PITCHER_POSITIONS,
    ROLES,
    SABER_SPLIT_FIELDS,
    SABER_STAT_FIELDS,
    WAR_CHECK_NAMES,
    Role,
    raw_saber_path,
    raw_statsapi_path,
    season_games,
)
from mlbproj.validate import CheckResult, ValidationError

_COMPLETE, _WAR_PRESENT, _COVERS_ROLES, _LEAGUE_SUM = WAR_CHECK_NAMES
_ROLE_LABEL: dict[Role, str] = {"bat": "타격", "pit": "투구"}
_SHOW_MAX = 5  # detail에 적는 ID·행의 최대 개수
_UNCHECKED = "원본을 읽지 못해 검사하지 못했다 (saber_complete 참고)"
# 보고서 JSON의 시즌 항목에 올 수 있는 키 (`passed`는 사람이 읽기 위한 값)
_ENTRY_KEYS = frozenset({"season", "checks", "inputs", "passed"})


@dataclass(frozen=True)
class WarSeasonReport:
    """한 시즌의 세이버메트릭스 검사 보고서.

    Attributes:
        season: 시즌.
        checks: `WAR_CHECK_NAMES` 순서의 결과 4개.
        inputs: 검사에 쓴 원본 상대 경로(posix) → sha256 (세이버메트릭스 두 파일 + 시즌 기록 두 파일).
    """

    season: int
    checks: tuple[CheckResult, ...]
    inputs: dict[str, str]

    @property
    def passed(self) -> bool:
        """검사 4개가 모두 있고 모두 통과했는지."""
        names = tuple(c.name for c in self.checks)
        return names == WAR_CHECK_NAMES and all(c.passed for c in self.checks)


# ---------------------------------------------------------------------------
# 원본 읽기 — 파일이 없거나 형식이 틀리면 예외 대신 문제 설명을 돌려준다
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Splits:
    """읽어 낸 API 원본 (세이버메트릭스 또는 시즌 기록). `total`은 `totalSplits` 값 그대로다."""

    rel: Path
    splits: list[dict[str, Any]]
    total: Any


def _as_int(value: Any) -> int | None:
    """JSON 정수면 그 값, 아니면 None (bool은 정수로 보지 않는다)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _is_finite_number(value: Any) -> TypeGuard[int | float]:
    """유한한 JSON 숫자인지 (bool·문자열·NaN·무한대는 아니다)."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _player_id(split: dict[str, Any]) -> int | None:
    player = split.get("player")
    return _as_int(player.get("id")) if isinstance(player, dict) else None


def _position(split: dict[str, Any]) -> str | None:
    position = split.get("position")
    abbr = position.get("abbreviation") if isinstance(position, dict) else None
    return abbr if isinstance(abbr, str) else None


def _stat(split: dict[str, Any]) -> dict[str, Any] | None:
    stat = split.get("stat")
    return stat if isinstance(stat, dict) else None


def _row_label(split: dict[str, Any], index: int) -> str:
    """detail에 적는 행 이름: 선수 ID, 없으면 splits 위치."""
    pid = _player_id(split)
    return str(pid) if pid is not None else f"splits[{index}]"


def _shown(items: Sequence[str], sep: str = ", ") -> str:
    """앞의 `_SHOW_MAX`개만 이어 적고, 더 있으면 `…`를 붙인다."""
    more = f"{sep}…" if len(items) > _SHOW_MAX else ""
    return sep.join(items[:_SHOW_MAX]) + more


def _head(ids: Iterable[int]) -> str:
    """ID 앞의 몇 개를 `[1, 2, …]`로 적는다."""
    return f"[{_shown([str(i) for i in sorted(ids)])}]"


def _load(raw_dir: Path, rel: Path) -> tuple[_Splits | None, str | None]:
    """API 원본을 읽는다. 구조가 깨져 행을 꺼낼 수 없으면 (None, 사유)."""
    where = rel.as_posix()
    if not (raw_dir / rel).is_file():
        return None, f"{where}: 파일이 없다"
    try:
        doc = raw.read_json(raw_dir, rel)
    except (OSError, ValueError) as exc:  # JSONDecodeError·UnicodeDecodeError는 ValueError
        return None, f"{where}: JSON을 읽지 못했다 ({exc})"
    stats = doc.get("stats") if isinstance(doc, dict) else None
    # 수집 검사(check_stats_body)와 같이 stats는 원소 1개짜리 목록이어야 한다
    if not isinstance(stats, list) or len(stats) != 1 or not isinstance(stats[0], dict):
        return None, f"{where}: stats가 객체 하나짜리 목록이 아니다"
    splits = stats[0].get("splits")
    if not isinstance(splits, list) or not all(isinstance(s, dict) for s in splits):
        return None, f"{where}: stats[0].splits가 객체 목록이 아니다"
    return _Splits(rel, splits, stats[0].get("totalSplits")), None


def _field_problems(split: dict[str, Any], role: Role) -> list[str]:
    """세이버메트릭스 한 행에서 빠졌거나 쓸 수 없는 필수 필드 (짧은 설명)."""
    found = [f"{f} 없음" for f in SABER_SPLIT_FIELDS if f not in split]
    stat = split.get("stat")
    if isinstance(stat, dict):
        found += [f"stat.{f} 없음" for f in SABER_STAT_FIELDS[role] if f not in stat]
    elif "stat" in split:
        found.append("stat 객체가 아님")
    # 역할 행 대조(saber_covers_roles)가 쓰는 하위 필드
    if "player" in split and _player_id(split) is None:
        found.append("player.id 없음 또는 정수가 아님")
    return found


# ---------------------------------------------------------------------------
# 검사 4개
# ---------------------------------------------------------------------------


def _result(name: str, problems: list[str], ok_detail: str) -> CheckResult:
    return CheckResult(name, not problems, "; ".join(problems) if problems else ok_detail)


def _identity_problems(doc: _Splits, season: int) -> list[str]:
    """선수 ID 중복과, split의 season 값이 파일 시즌과 다른 행.

    `clean.war.build_war_seasons`는 "F2 ⑦을 통과했다면 같은 선수가 한 원본에 두 번 없다"를 전제로 한다.
    또 리그 합(570 · 430)은 시즌마다 같아서 다른 시즌의 응답이 이 자리에 저장돼도 합 검사로는 잡히지 않는다.
    """
    where = doc.rel.as_posix()
    problems: list[str] = []
    counts = Counter(pid for split in doc.splits if (pid := _player_id(split)) is not None)
    dups = [pid for pid, n in counts.items() if n > 1]
    if dups:
        problems.append(f"{where}: 같은 선수 ID가 두 번 이상 있다 {_head(dups)}")
    # 필드가 없는 행은 _field_problems가 적는다. API는 시즌을 문자열("2025")로 준다
    wrong = [
        f"{_row_label(split, index)}({split['season']!r})"
        for index, split in enumerate(doc.splits)
        if "season" in split and split["season"] != str(season)
    ]
    if wrong:
        problems.append(f"{where}: season이 {season}이 아닌 행 {len(wrong)}개 ({_shown(wrong)})")
    return problems


def _check_complete(
    saber: dict[Role, _Splits | None], load_problems: list[str], season: int
) -> CheckResult:
    """saber_complete: 두 원본의 형식 · totalSplits = 행 수 · 필수 필드 · 선수 ID 중복 없음 · 시즌 값."""
    problems = list(load_problems)
    for role, doc in saber.items():
        if doc is None:
            continue
        where = doc.rel.as_posix()
        if _as_int(doc.total) != len(doc.splits):
            problems.append(
                f"{where}: totalSplits {doc.total!r} ≠ splits {len(doc.splits)}행 (응답 불완전)"
            )
        counts = Counter(label for split in doc.splits for label in _field_problems(split, role))
        problems += [f"{where}: {label} ({n}행)" for label, n in counts.items()]
        problems += _identity_problems(doc, season)
    counts_text = " · ".join(
        f"{_ROLE_LABEL[role]} {len(doc.splits)}행" for role, doc in saber.items() if doc is not None
    )
    return _result(
        _COMPLETE,
        problems,
        f"{counts_text} — totalSplits 일치, 필수 필드 모두 있음, 선수 ID 중복 없음",
    )


def _bad_war_rows(doc: _Splits) -> list[str]:
    """war가 유한한 숫자가 아닌 행 (`"646240(None)"` 같은 설명)."""
    bad: list[str] = []
    for index, split in enumerate(doc.splits):
        stat = _stat(split)
        if stat is None or "war" not in stat:
            bad.append(f"{_row_label(split, index)}(war 없음)")
        elif not _is_finite_number(stat["war"]):
            bad.append(f"{_row_label(split, index)}({stat['war']!r})")
    return bad


def _check_war_present(saber: dict[Role, _Splits | None]) -> CheckResult:
    """saber_war_present: 원본의 모든 행(역할 행이 아닌 행 포함)의 war가 유한한 숫자."""
    problems: list[str] = []
    for role in ROLES:
        doc = saber[role]
        if doc is None:
            problems.append(f"{_ROLE_LABEL[role]} 세이버메트릭스: {_UNCHECKED}")
            continue
        bad = _bad_war_rows(doc)
        if bad:
            problems.append(
                f"{doc.rel.as_posix()}: war가 유한한 숫자가 아닌 행 {len(bad)}개 ({_shown(bad)})"
            )
    rows = sum(len(doc.splits) for doc in saber.values() if doc is not None)
    return _result(_WAR_PRESENT, problems, f"war 결측 0 — {rows}행 모두 유한한 숫자")


def _is_role_row(split: dict[str, Any], role: Role) -> bool | None:
    """시즌 기록 한 행이 역할 행(D8)인지. 판정에 쓰는 필드가 없거나 틀리면 None.

    타자는 contracts의 D8 주석대로 포지션이 P가 아니고 **타석이 1 이상**인 행이다
    (`clean.batting.is_batter_split`과 같다 — 정제 표의 모든 행이 WAR을 갖는다는 보증이 이 검사다).
    """
    position = _position(split)
    if position is None:
        return None
    if role == "pit":
        return position in PITCHER_POSITIONS
    stat = _stat(split)
    pa = _as_int(stat.get("plateAppearances")) if stat is not None else None
    if pa is None:
        return None
    return position not in BATTER_EXCLUDED_POSITIONS and pa > 0


def _role_ids(doc: _Splits, role: Role) -> tuple[set[int], list[str]]:
    """시즌 기록 원본의 역할 행 선수 ID와, 역할을 판정하지 못한 행의 문제 설명."""
    ids: set[int] = set()
    unknown: list[str] = []
    for index, split in enumerate(doc.splits):
        pid = _player_id(split)
        is_role = _is_role_row(split, role)
        if pid is None or is_role is None:
            unknown.append(_row_label(split, index))
        elif is_role:
            ids.add(pid)
    problems = (
        [f"{doc.rel.as_posix()}: 역할을 판정하지 못한 행 {len(unknown)}개 ({_shown(unknown)})"]
        if unknown
        else []
    )
    return ids, problems


def _check_covers_roles(
    saber: dict[Role, _Splits | None],
    records: dict[Role, _Splits | None],
    record_problems: dict[Role, str | None],
) -> CheckResult:
    """saber_covers_roles: 시즌 기록의 역할 행(D8)이 모두 그 역할 세이버메트릭스에 있다."""
    problems: list[str] = []
    found: list[str] = []
    for role in ROLES:
        label = _ROLE_LABEL[role]
        doc, record = saber[role], records[role]
        if record is None:
            problems.append(f"시즌 기록 원본을 읽지 못했다 — {record_problems[role]}")
        if doc is None:
            problems.append(f"{label} 세이버메트릭스: {_UNCHECKED}")
        if doc is None or record is None:
            continue
        role_ids, unknown = _role_ids(record, role)
        problems += unknown
        saber_ids = {pid for s in doc.splits if (pid := _player_id(s)) is not None}
        missing = role_ids - saber_ids
        if missing:
            problems.append(
                f"{doc.rel.as_posix()}: 시즌 기록의 {label} 역할 행 {len(missing)}명이 없다 "
                f"{_head(missing)}"
            )
        found.append(
            f"{label} 역할 행 {len(role_ids)}명 모두 있음 "
            f"(세이버메트릭스에만 있는 행 {len(saber_ids - role_ids)}명)"
        )
    return _result(_COVERS_ROLES, problems, " · ".join(found))


def _check_league_sum(
    saber: dict[Role, _Splits | None],
    season: int,
    league_war: Mapping[Role, float],
    tolerance: float,
) -> CheckResult:
    """saber_league_sum: 역할마다 원본 전체 행의 war 합 = league_war × 경기 수/162 ± tolerance."""
    ratio = season_games(season) / FULL_SEASON_GAMES
    problems: list[str] = []
    totals: dict[Role, float] = {}
    for role in ROLES:
        label = _ROLE_LABEL[role]
        doc = saber[role]
        if doc is None:
            problems.append(f"{label} 세이버메트릭스: {_UNCHECKED}")
            continue
        if _bad_war_rows(doc):
            problems.append(
                f"{label} war가 유한한 숫자가 아닌 행이 있어 합을 내지 못했다 (saber_war_present 참고)"
            )
            continue
        # 역할 규칙(D8)으로 거르기 전, 원본 전체 행의 합이다 (E26)
        total = math.fsum(float(split["stat"]["war"]) for split in doc.splits)
        expected = league_war[role] * ratio
        totals[role] = total
        if abs(total - expected) > tolerance:
            problems.append(
                f"{label} war 합 {total:.2f} — 기준 {expected:.2f} ± {tolerance} 밖 "
                f"(차이 {total - expected:+.2f}, 경기 {season_games(season)}/{FULL_SEASON_GAMES})"
            )
    got = " · ".join(f"{_ROLE_LABEL[role]} {totals[role]:.1f}" for role in ROLES if role in totals)
    want = " · ".join(f"{league_war[role] * ratio:.1f}" for role in ROLES)
    return _result(_LEAGUE_SUM, problems, f"{got} (기준 {want}, ±{tolerance})")


def _season_rels(season: int) -> list[Path]:
    """시즌 검사에 쓰는 원본 네 개: 세이버메트릭스 두 파일 + 시즌 기록 두 파일."""
    return [
        *(raw_saber_path(season, role) for role in ROLES),
        *(raw_statsapi_path(season, role) for role in ROLES),
    ]


def validate_war_season(
    raw_dir: Path,
    season: int,
    *,
    league_war: Mapping[Role, float] = LEAGUE_WAR,
    tolerance: float = LEAGUE_WAR_TOLERANCE,
) -> WarSeasonReport:
    """한 시즌의 세이버메트릭스 원본을 검사한다 (모듈 docstring)."""
    # 읽기 전에 sha256을 남긴다. 없는 파일은 해당 검사가 실패로 적는다
    inputs = {
        rel.as_posix(): raw.file_sha256(raw_dir / rel)
        for rel in _season_rels(season)
        if (raw_dir / rel).is_file()
    }
    saber: dict[Role, _Splits | None] = {}
    load_problems: list[str] = []
    records: dict[Role, _Splits | None] = {}
    record_problems: dict[Role, str | None] = {}
    for role in ROLES:
        saber[role], problem = _load(raw_dir, raw_saber_path(season, role))
        if problem is not None:
            load_problems.append(problem)
        records[role], record_problems[role] = _load(raw_dir, raw_statsapi_path(season, role))

    checks = (
        _check_complete(saber, load_problems, season),
        _check_war_present(saber),
        _check_covers_roles(saber, records, record_problems),
        _check_league_sum(saber, season, league_war, tolerance),
    )
    return WarSeasonReport(season, checks, inputs)


def validate_war_seasons(
    raw_dir: Path,
    seasons: Iterable[int],
    *,
    league_war: Mapping[Role, float] = LEAGUE_WAR,
    tolerance: float = LEAGUE_WAR_TOLERANCE,
) -> list[WarSeasonReport]:
    """여러 시즌을 검사한다. 시즌 오름차순, 중복 시즌은 한 번."""
    return [
        validate_war_season(raw_dir, season, league_war=league_war, tolerance=tolerance)
        for season in sorted(set(seasons))
    ]


# ---------------------------------------------------------------------------
# 보고서 파일
# ---------------------------------------------------------------------------


def _check_unique_seasons(seasons: Iterable[int]) -> None:
    """같은 시즌이 두 번 있으면 ValueError (어느 쪽이 맞는지 정할 수 없다)."""
    dups = sorted(season for season, n in Counter(seasons).items() if n > 1)
    if dups:
        raise ValueError(f"WAR 검증 보고서에 시즌이 중복된다: {', '.join(map(str, dups))}")


def write_war_report(reports: Iterable[WarSeasonReport], path: Path) -> None:
    """보고서를 JSON으로 쓴다 (사람이 읽게 indent, 같은 입력이면 같은 바이트 — N1)."""
    ordered = sorted(reports, key=lambda r: r.season)
    _check_unique_seasons(r.season for r in ordered)
    # `passed`는 사람이 읽기 위한 값이다. 읽을 때는 checks로 다시 계산한다
    doc = [{**asdict(r), "passed": r.passed} for r in ordered]
    text = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))  # 운영체제 줄바꿈 변환 없이


def _check_from_doc(doc: Any) -> CheckResult:
    """보고서 JSON의 검사 하나. 형식이 틀리면 ValueError."""
    if not isinstance(doc, dict):
        raise ValueError(f"검사 결과가 객체가 아니다: {doc!r}")
    name, passed, detail = doc.get("name"), doc.get("passed"), doc.get("detail")
    if not (isinstance(name, str) and isinstance(passed, bool) and isinstance(detail, str)):
        raise ValueError(f"검사 결과 형식이 틀렸다: {doc!r}")
    return CheckResult(name, passed, detail)


def _report_from_doc(doc: Any) -> WarSeasonReport:
    """보고서 JSON의 시즌 하나. 형식이 틀리면 ValueError (게이트가 엉터리 값을 통과로 읽지 않게)."""
    if not isinstance(doc, dict):
        raise ValueError(f"시즌 보고서가 객체가 아니다: {doc!r}")
    # 다른 보고서(F2 검증 보고서의 merged_duplicates 등)를 WAR 보고서로 읽지 않는다
    extra = sorted(set(doc) - _ENTRY_KEYS)
    if extra:
        raise ValueError(f"WAR 검증 보고서에 없는 키가 있다: {extra}")
    season = _as_int(doc.get("season"))
    if season is None:
        raise ValueError(f"season이 정수가 아니다: {doc.get('season')!r}")
    checks, inputs = doc.get("checks"), doc.get("inputs")
    if not isinstance(checks, list):
        raise ValueError(f"{season} checks가 목록이 아니다")
    if not (isinstance(inputs, dict) and all(isinstance(v, str) for v in inputs.values())):
        raise ValueError(f"{season} inputs가 경로 → sha256 사전이 아니다")
    return WarSeasonReport(
        season=season,
        checks=tuple(_check_from_doc(c) for c in checks),
        inputs={str(rel): sha for rel, sha in inputs.items()},
    )


def read_war_report(path: Path) -> list[WarSeasonReport]:
    """`write_war_report`가 쓴 보고서를 읽는다.

    Raises:
        ValueError: 형식이 틀렸을 때.
    """
    # 파일을 읽지 못하면 OSError, JSON이 아니거나(UnicodeDecodeError 포함) 형식이 틀렸으면 ValueError
    doc = json.loads(path.read_bytes().decode("utf-8"))
    if not isinstance(doc, list):
        raise ValueError(f"WAR 검증 보고서는 시즌 목록이어야 한다: {path}")
    reports = sorted((_report_from_doc(item) for item in doc), key=lambda r: r.season)
    _check_unique_seasons(r.season for r in reports)
    return reports


def _current_sha256(raw_dir: Path, rel: str) -> str | None:
    path = raw_dir / rel
    return raw.file_sha256(path) if path.is_file() else None


def _failed_label(report: WarSeasonReport) -> str:
    """실패한 시즌을 `2025(saber_league_sum)` 형식으로."""
    if tuple(c.name for c in report.checks) != WAR_CHECK_NAMES:
        return f"{report.season}(검사 {len(WAR_CHECK_NAMES)}개가 순서대로 있지 않다)"
    return f"{report.season}({', '.join(c.name for c in report.checks if not c.passed)})"


def assert_war_validated(raw_dir: Path, report_path: Path, seasons: Iterable[int]) -> None:
    """WAR 정제 전 확인: 보고서가 있고, 요청 시즌이 모두 통과했고, 원본(inputs)이 그 뒤로 바뀌지 않았다.

    Raises:
        mlbproj.validate.ValidationError: 보고서 없음 / 시즌 누락 / 실패한 시즌 / sha256 불일치. 사유를 적는다.
    """
    wanted = sorted(set(seasons))
    if not report_path.is_file():
        raise ValidationError(
            f"WAR 검증 보고서가 없다: {report_path} — 세이버메트릭스 검증(F2 ⑦)을 먼저 실행한다"
        )
    try:
        reports = {r.season: r for r in read_war_report(report_path)}
    except (OSError, ValueError) as exc:  # JSONDecodeError·UnicodeDecodeError도 ValueError
        raise ValidationError(f"WAR 검증 보고서를 읽지 못했다: {report_path} ({exc})") from exc

    reasons: list[str] = []
    absent = [s for s in wanted if s not in reports]
    if absent:
        reasons.append(f"보고서에 없는 시즌: {', '.join(map(str, absent))}")
    checked = [reports[s] for s in wanted if s in reports]
    failed = [r for r in checked if not r.passed]
    if failed:
        reasons.append(f"검증을 통과하지 못한 시즌: {', '.join(_failed_label(r) for r in failed)}")
    # 검사에 쓰는 원본 네 개가 모두 sha256으로 기록돼 있어야 "그 뒤로 바뀌지 않았다"를 대조할 수 있다
    unrecorded = sorted(
        {
            rel.as_posix()
            for r in checked
            for rel in _season_rels(r.season)
            if rel.as_posix() not in r.inputs
        }
    )
    if unrecorded:
        reasons.append(f"보고서에 sha256이 없는 원본: {', '.join(unrecorded)}")
    expected = {(rel, sha) for r in checked for rel, sha in r.inputs.items()}
    changed = sorted({rel for rel, sha in expected if _current_sha256(raw_dir, rel) != sha})
    if changed:
        reasons.append(f"검증 뒤 바뀌거나 사라진 원본: {', '.join(changed)}")
    if reasons:
        raise ValidationError("; ".join(reasons))
