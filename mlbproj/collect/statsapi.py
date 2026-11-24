"""MLB Stats API 수집 — 시즌 기록·선수 정보 (SPEC F1·E1). 레인 A가 구현한다.

저장은 `collect.raw.fetch_to_raw`만 쓴다. URL과 경로는 `contracts`의 함수만 쓴다.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from mlbproj.collect.raw import CollectOutcome, Getter
from mlbproj.contracts import Role


def check_stats_body(body: bytes) -> None:
    """시즌 기록 응답이 저장할 만한지 검사한다 (`fetch_to_raw`의 check로 쓴다).

    JSON이고, `stats`가 원소 1개짜리 목록이며, 그 안의 `splits`가 목록이고,
    `totalSplits`가 `splits` 길이와 같아야 한다(같지 않으면 limit에 잘린 응답이다).

    Raises:
        ValueError: 위 조건 중 하나라도 어긋날 때. 메시지에 어떤 조건인지 적는다.
    """
    raise NotImplementedError


def stats_splits(raw_dir: Path, season: int, role: Role) -> list[dict[str, Any]]:
    """저장된 시즌 기록 원본의 splits를 읽는다.

    Raises:
        FileNotFoundError: 원본이 없을 때.
    """
    raise NotImplementedError


def collect_season_stats(
    getter: Getter, raw_dir: Path, season: int, role: Role, *, force: bool = False
) -> CollectOutcome:
    """시즌 기록 하나를 받는다 (`contracts.statsapi_stats_url` → `contracts.raw_statsapi_path`).

    캐시가 있으면 요청하지 않는다(force면 다시 받는다). 응답은 `check_stats_body`로 검사한 뒤 저장한다.
    """
    raise NotImplementedError


def stats_player_ids(raw_dir: Path, seasons: Iterable[int]) -> list[int]:
    """주어진 시즌들의 타격·투구 원본에 나온 선수 ID를 중복 없이 오름차순으로 돌려준다.

    Raises:
        FileNotFoundError: 시즌 원본이 하나라도 없을 때.
    """
    raise NotImplementedError


def people_batches(player_ids: Sequence[int]) -> list[list[int]]:
    """선수 ID를 중복 제거·오름차순 정렬한 뒤 `PEOPLE_BATCH_SIZE`개씩 나눈다. 빈 입력은 빈 목록."""
    raise NotImplementedError


def check_people_body(body: bytes, expected_ids: Sequence[int]) -> None:
    """선수 정보 응답 검사: JSON이고 `people`가 목록이며, 요청한 ID가 모두 들어 있어야 한다.

    Raises:
        ValueError: 형식이 틀리거나 빠진 ID가 있을 때 (빠진 ID를 메시지에 적는다).
    """
    raise NotImplementedError


def collect_people(
    getter: Getter, raw_dir: Path, player_ids: Sequence[int], *, force: bool = False
) -> list[CollectOutcome]:
    """선수 정보를 묶음별로 받는다 (`people_batches` → `contracts.raw_people_path(i)`).

    묶음 i의 URL은 `contracts.statsapi_people_url(batch)`이고, 묶음마다 캐시를 쓴다.
    선수 목록이 줄어 묶음 수가 줄었으면, 남은 `people_{n}.json`(n ≥ 묶음 수)을
    `collect.raw.remove_raw`로 지운다.

    Returns:
        묶음 순서대로의 결과.
    """
    raise NotImplementedError
