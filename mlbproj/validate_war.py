"""세이버메트릭스(WAR) 수집 완전성 검사 — SPEC v0.5 F2 ⑦, E26. 레인 A가 구현한다.

기존 시즌 검증(`validate.py`, F2 ①~⑥)과 보고서는 건드리지 않는다. 시즌마다 두 역할 원본을 함께 보고
`contracts.WAR_CHECK_NAMES` 순서의 검사 4개를 낸다(이름은 contracts 주석 참고).

- saber_complete: `raw_saber_path(season, "bat"/"pit")` 두 파일이 있고, JSON이며 `stats[0].splits` 목록,
  `totalSplits` = 행 수, 모든 행에 `SABER_SPLIT_FIELDS`와 `stat`의 `SABER_STAT_FIELDS[role]` 키가 있다.
- saber_war_present: 모든 행의 `stat.war`가 유한한 숫자다(bool 아님). 결측·문자열이면 실패(어느 선수인지 적는다).
- saber_covers_roles: 같은 시즌의 시즌 기록 원본(`raw_statsapi_path`)에서 역할 행(D8 — 타자: 포지션이
  `BATTER_EXCLUDED_POSITIONS`가 아닌 타격 행, 투수: 포지션이 `PITCHER_POSITIONS`인 투구 행)의 선수 ID가
  모두 그 역할 세이버메트릭스에 있다. 세이버메트릭스에만 있는 ID는 실패가 아니다(투수 타격·야수 등판, E26).
- saber_league_sum: 역할마다 원본 전체 행의 war 합이 `league_war[role] × season_games(season) / 162`에서
  `tolerance` 안이다(테스트는 잘라 낸 픽스처에 맞는 기준값을 넘긴다).

파일이 없거나 형식이 틀려도 예외 대신 실패한 검사로 담는다. 통과 detail에는 핵심 수치를 적는다
(예: "타격 570.0 · 투구 430.0 (기준 570.0 · 430.0, ±1.0)").
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from mlbproj.contracts import LEAGUE_WAR, LEAGUE_WAR_TOLERANCE, WAR_CHECK_NAMES, Role
from mlbproj.validate import CheckResult


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


def validate_war_season(
    raw_dir: Path,
    season: int,
    *,
    league_war: Mapping[Role, float] = LEAGUE_WAR,
    tolerance: float = LEAGUE_WAR_TOLERANCE,
) -> WarSeasonReport:
    """한 시즌의 세이버메트릭스 원본을 검사한다 (모듈 docstring)."""
    raise NotImplementedError


def validate_war_seasons(
    raw_dir: Path,
    seasons: Iterable[int],
    *,
    league_war: Mapping[Role, float] = LEAGUE_WAR,
    tolerance: float = LEAGUE_WAR_TOLERANCE,
) -> list[WarSeasonReport]:
    """여러 시즌을 검사한다. 시즌 오름차순, 중복 시즌은 한 번."""
    raise NotImplementedError


def write_war_report(reports: Iterable[WarSeasonReport], path: Path) -> None:
    """보고서를 JSON으로 쓴다 (사람이 읽게 indent, 같은 입력이면 같은 바이트 — N1)."""
    raise NotImplementedError


def read_war_report(path: Path) -> list[WarSeasonReport]:
    """`write_war_report`가 쓴 보고서를 읽는다.

    Raises:
        ValueError: 형식이 틀렸을 때.
    """
    raise NotImplementedError


def assert_war_validated(raw_dir: Path, report_path: Path, seasons: Iterable[int]) -> None:
    """WAR 정제 전 확인: 보고서가 있고, 요청 시즌이 모두 통과했고, 원본(inputs)이 그 뒤로 바뀌지 않았다.

    Raises:
        mlbproj.validate.ValidationError: 보고서 없음 / 시즌 누락 / 실패한 시즌 / sha256 불일치. 사유를 적는다.
    """
    raise NotImplementedError
