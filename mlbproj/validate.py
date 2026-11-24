"""수집 완전성 검증 (SPEC F2). 레인 C가 구현한다.

원본 디렉토리의 한 시즌을 검사해 보고서를 만든다. 원본은 `collect.raw.read_json`·`read_csv`로만
읽는다(수집 모듈 statsapi·savant는 import하지 않는다 — 다른 레인이다). 검사 결과가 하나라도 실패면
그 시즌은 통과가 아니고, 이후 단계(M2 정제)는 `assert_validated`로 막힌다.

검사 (이름은 `CHECK_NAMES` 순서, 결과도 이 순서로 담는다)
- F2-1 필수 필드: API 타격·투구 원본(splits의 `STATSAPI_SPLIT_FIELDS`, stat의 `STATSAPI_*_STAT_FIELDS`),
  응답 완전성(`totalSplits` == splits 길이), 투구 `inningsPitched` 표기(172.1 = 172×3+1)와 `outs`가
  모든 행에서 같은지, Savant 타자·투수 CSV의 `SAVANT_*_COLUMNS`, 수비 시즌(`FIELDING_YEARS`)이면
  OAA·FRV CSV의 열. 파일이 없으면 이 검사가 실패한다(예외를 내지 않는다).
- F2-2 ID 중복: API 타격·투구 각각 선수 ID가 한 번씩. Savant 타자·투수·OAA·FRV(ID 열 `id`)도 한 번씩.
  **모든 값이 같은 중복 행은 하나로 합친 것으로 보고 통과**시키되 `merged_duplicates`에
  `"savant/2019_pitcher.csv: 519381 ×2"` 형식으로 남긴다(E19). 값이 다른 중복은 실패.
- F2-3 타자 ID 일치: API 타격 중 `plateAppearances` > 0인 선수 ID 집합 == Savant 타자 ID 집합.
  실패하면 양쪽에만 있는 ID 수와 앞의 몇 개를 detail에 적는다.
- F2-4 투수 포함 관계(E2): Savant 투수 ID ⊆ API 투구 ID. 그리고 API 투구 중 Savant에 없는 선수 가운데
  포지션이 `PITCHER_POSITIONS`인 선수는 한 명당 `battersFaced` ≤ `limits.max_missing_pitcher_bf`이고,
  그들의 BF 합 ≤ 리그 BF 합 × `limits.max_missing_pitcher_share`. 포지션이 그 밖(등판한 야수)이면 제한 없음.
- F2-5 리그 합계: 리그 PA 합(API 타격 `plateAppearances` 합)이 범위 안, 리그 아웃 합(API 투구 `outs` 합)이
  범위 안, 그리고 `limits.pa_bf_identity`면 **PA 합 == BF 합(API 투구 `battersFaced` 합)** 이 정확히 같다. 범위는 시즌이
  `SHORT_SEASON`이면 `limits.*_short`, 아니면 `limits.*_full`.
- F2-6 선수 정보: 이 시즌 API 타격·투구에 나온 모든 선수 ID가 원본 디렉토리의 `statsapi/people_*.json`
  (모든 묶음을 합친 `people`)에 있고, 그 항목에 `PEOPLE_FIELDS`가 모두 있다.

보고서의 `inputs`는 검사에 쓴 원본 파일(상대 경로 posix) → sha256이다. `assert_validated`는 지금 파일의
sha256과 비교해, 검증 뒤에 원본이 바뀌었으면 막는다.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

CHECK_NAMES: tuple[str, ...] = (
    "F2-1 필수 필드",
    "F2-2 ID 중복",
    "F2-3 타자 ID 일치",
    "F2-4 투수 포함 관계",
    "F2-5 리그 합계",
    "F2-6 선수 정보",
)


@dataclass(frozen=True)
class LeagueLimits:
    """F2-4·F2-5의 기준값. 기본값은 2019~2026 실측(SPEC E2·E6·E20)에 여유를 둔 것이다.

    Attributes:
        pa_full: 정규 시즌 리그 PA 합 범위 (포함).
        pa_short: 2020(60경기) 리그 PA 합 범위.
        outs_full: 정규 시즌 리그 아웃 합 범위.
        outs_short: 2020 리그 아웃 합 범위.
        max_missing_pitcher_bf: Savant에 없는 실제 투수 한 명의 BF 상한.
        max_missing_pitcher_share: Savant에 없는 실제 투수 BF 합 / 리그 BF 합 상한.
        pa_bf_identity: 리그 PA 합 == BF 합을 검사할지. 실데이터는 항상 True. 선수 일부만 잘라 낸
            픽스처에서는 성립하지 않으므로 테스트가 False로 끈다.
    """

    pa_full: tuple[int, int] = (175_000, 195_000)
    pa_short: tuple[int, int] = (60_000, 72_000)
    outs_full: tuple[int, int] = (124_000, 135_000)
    outs_short: tuple[int, int] = (43_000, 50_000)
    max_missing_pitcher_bf: int = 30
    max_missing_pitcher_share: float = 0.005
    pa_bf_identity: bool = True


DEFAULT_LIMITS = LeagueLimits()


@dataclass(frozen=True)
class CheckResult:
    """검사 하나의 결과.

    Attributes:
        name: `CHECK_NAMES` 중 하나.
        passed: 통과 여부.
        detail: 사람이 읽는 설명. 실패면 무엇이 어긋났는지(파일·ID·수치), 통과면 핵심 수치
            (예: "PA 183,849 = BF 183,849, 아웃 129,239").
    """

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SeasonReport:
    """한 시즌의 검증 보고서.

    Attributes:
        season: 시즌.
        checks: `CHECK_NAMES` 순서의 결과 6개.
        merged_duplicates: 하나로 합친 같은 값 중복 행 기록(F2-2).
        inputs: 검사에 쓴 원본 상대 경로(posix) → sha256.
    """

    season: int
    checks: tuple[CheckResult, ...]
    merged_duplicates: tuple[str, ...]
    inputs: dict[str, str]

    @property
    def passed(self) -> bool:
        """모든 검사를 통과했는지."""
        raise NotImplementedError


class ValidationError(Exception):
    """검증을 통과하지 않은 원본으로 다음 단계를 시작하려 할 때."""


def validate_season(
    raw_dir: Path, season: int, *, limits: LeagueLimits = DEFAULT_LIMITS
) -> SeasonReport:
    """한 시즌의 원본을 검사한다 (모듈 docstring의 F2-1~F2-6).

    Args:
        raw_dir: 원본 디렉토리.
        season: 시즌.
        limits: 기준값 (테스트는 잘라 낸 픽스처에 맞는 값을 넣는다).

    Returns:
        보고서. 파일이 없거나 형식이 틀려도 예외 대신 실패한 검사로 담는다.
    """
    raise NotImplementedError


def validate_seasons(
    raw_dir: Path, seasons: Iterable[int], *, limits: LeagueLimits = DEFAULT_LIMITS
) -> list[SeasonReport]:
    """여러 시즌을 시즌 오름차순으로 검사한다."""
    raise NotImplementedError


def write_report(reports: Iterable[SeasonReport], path: Path) -> None:
    """보고서를 JSON으로 쓴다 (시즌 오름차순, 키 정렬, UTF-8, 같은 입력이면 같은 바이트)."""
    raise NotImplementedError


def read_report(path: Path) -> list[SeasonReport]:
    """`write_report`가 쓴 JSON을 읽는다 (왕복하면 같은 보고서)."""
    raise NotImplementedError


def assert_validated(raw_dir: Path, report_path: Path, seasons: Iterable[int]) -> None:
    """다음 단계 시작 전 확인: 보고서가 있고, 요청 시즌이 모두 통과했고, 원본이 그 뒤로 바뀌지 않았다.

    Raises:
        ValidationError: 보고서 없음 / 시즌 누락 / 실패한 시즌 / 원본 sha256 불일치. 메시지에 사유를 적는다.
    """
    raise NotImplementedError
