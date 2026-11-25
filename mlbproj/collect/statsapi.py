"""MLB Stats API 수집 — 시즌 기록·선수 정보 (SPEC F1·E1). 레인 A가 구현한다.

저장은 `collect.raw.fetch_to_raw`만 쓴다. URL과 경로는 `contracts`의 함수만 쓴다.
"""

import json
from collections.abc import Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, TypeGuard

from mlbproj.collect.raw import (
    CollectOutcome,
    Getter,
    fetch_to_raw,
    load_manifest,
    read_json,
    remove_raw,
)
from mlbproj.contracts import (
    PEOPLE_BATCH_SIZE,
    ROLES,
    Role,
    raw_people_path,
    raw_saber_path,
    raw_statsapi_path,
    statsapi_people_url,
    statsapi_saber_url,
    statsapi_stats_url,
)


def _parse_json(body: bytes, what: str) -> Any:
    """UTF-8 JSON 응답을 읽는다. 아니면 ValueError (UnicodeDecodeError·JSONDecodeError 포함).

    BOM이 붙은 JSON도 거부한다. 저장한 원본을 읽는 `raw.read_json`도 BOM을 읽지 못하므로,
    검사를 통과한 응답은 나중에 반드시 다시 읽힌다.
    """
    try:
        return json.loads(body.decode("utf-8"))
    except ValueError as err:
        raise ValueError(f"{what} 응답이 UTF-8 JSON이 아니다: {err}") from err


def _is_int(value: object) -> TypeGuard[int]:
    """JSON 정수인지 (bool은 int의 하위 클래스지만 정수로 치지 않는다)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _describe(value: Any) -> str:
    """오류 메시지용으로 값의 모양을 짧게 적는다."""
    if value is None:
        return "없음"
    if isinstance(value, list):
        return f"목록, 원소 {len(value)}개"
    return type(value).__name__


def check_stats_body(body: bytes) -> None:
    """시즌 기록 응답이 저장할 만한지 검사한다 (`fetch_to_raw`의 check로 쓴다).

    JSON이고, `stats`가 원소 1개짜리 목록이며, 그 안의 `splits`가 목록이고,
    `totalSplits`가 `splits` 길이와 같아야 한다(같지 않으면 limit에 잘린 응답이다).

    Raises:
        ValueError: 위 조건 중 하나라도 어긋날 때. 메시지에 어떤 조건인지 적는다.
    """
    doc = _parse_json(body, "시즌 기록")
    if not isinstance(doc, dict):
        raise ValueError(f"시즌 기록 응답의 최상위가 객체가 아니다 ({_describe(doc)})")
    stats = doc.get("stats")
    if not isinstance(stats, list) or len(stats) != 1:
        raise ValueError(f"stats가 원소 1개짜리 목록이 아니다 ({_describe(stats)})")
    block = stats[0]
    if not isinstance(block, dict):
        raise ValueError(f"stats[0]이 객체가 아니어서 splits를 읽을 수 없다 ({_describe(block)})")
    splits = block.get("splits")
    if not isinstance(splits, list):
        raise ValueError(f"stats[0].splits가 목록이 아니다 ({_describe(splits)})")
    total = block.get("totalSplits")
    if not _is_int(total):
        raise ValueError(f"stats[0].totalSplits가 없거나 정수가 아니다 ({_describe(total)})")
    if total != len(splits):
        raise ValueError(
            f"stats[0].totalSplits({total})가 splits 행 수({len(splits)})와 다르다"
            " — limit에 잘린 응답일 수 있다"
        )


def stats_splits(raw_dir: Path, season: int, role: Role) -> list[dict[str, Any]]:
    """저장된 시즌 기록 원본의 splits를 읽는다.

    Raises:
        FileNotFoundError: 원본이 없을 때.
    """
    doc = read_json(raw_dir, raw_statsapi_path(season, role))
    splits: list[dict[str, Any]] = doc["stats"][0]["splits"]
    return splits


def collect_season_stats(
    getter: Getter, raw_dir: Path, season: int, role: Role, *, force: bool = False
) -> CollectOutcome:
    """시즌 기록 하나를 받는다 (`contracts.statsapi_stats_url` → `contracts.raw_statsapi_path`).

    캐시가 있으면 요청하지 않는다(force면 다시 받는다). 응답은 `check_stats_body`로 검사한 뒤 저장한다.
    """
    return fetch_to_raw(
        getter,
        raw_dir,
        raw_statsapi_path(season, role),
        statsapi_stats_url(season, role),
        check=check_stats_body,
        force=force,
    )


def stats_player_ids(raw_dir: Path, seasons: Iterable[int]) -> list[int]:
    """주어진 시즌들의 타격·투구 원본에 나온 선수 ID를 중복 없이 오름차순으로 돌려준다.

    Raises:
        FileNotFoundError: 시즌 원본이 하나라도 없을 때.
        ValueError: 원본의 split에 정수 `player.id`가 없을 때 (어느 파일의 몇 번째 행인지 적는다).
    """
    ids: set[int] = set()
    for season in seasons:
        for role in ROLES:
            rel = raw_statsapi_path(season, role)
            for index, split in enumerate(stats_splits(raw_dir, season, role)):
                ids.add(_split_player_id(split, f"{rel.as_posix()} splits[{index}]"))
    return sorted(ids)


def _split_player_id(split: Any, where: str) -> int:
    """split의 `player.id`를 꺼낸다.

    `check_stats_body`는 행 모양까지 보지 않고(필드 검사는 F2 ①), 선수 정보 요청은 검증보다
    먼저 돈다. 그래서 KeyError 대신 어느 행인지 적은 ValueError로 멈춘다.
    """
    player = split.get("player") if isinstance(split, dict) else None
    player_id = player.get("id") if isinstance(player, dict) else None
    if not _is_int(player_id):
        raise ValueError(f"{where}에 정수 player.id가 없다 ({_describe(player_id)})")
    return player_id


def people_batches(player_ids: Sequence[int]) -> list[list[int]]:
    """선수 ID를 중복 제거·오름차순 정렬한 뒤 `PEOPLE_BATCH_SIZE`개씩 나눈다. 빈 입력은 빈 목록."""
    unique = sorted(set(player_ids))
    return [
        unique[start : start + PEOPLE_BATCH_SIZE]
        for start in range(0, len(unique), PEOPLE_BATCH_SIZE)
    ]


def check_people_body(body: bytes, expected_ids: Sequence[int]) -> None:
    """선수 정보 응답 검사: JSON이고 `people`가 목록이며, 요청한 ID가 모두 들어 있어야 한다.

    Raises:
        ValueError: 형식이 틀리거나 빠진 ID가 있을 때 (빠진 ID를 메시지에 적는다).
    """
    doc = _parse_json(body, "선수 정보")
    if not isinstance(doc, dict):
        raise ValueError(f"선수 정보 응답의 최상위가 객체가 아니다 ({_describe(doc)})")
    people = doc.get("people")
    if not isinstance(people, list):
        raise ValueError(f"선수 정보 응답의 people가 목록이 아니다 ({_describe(people)})")
    found: set[int] = set()
    for index, person in enumerate(people):
        if not isinstance(person, dict):
            raise ValueError(f"people[{index}]가 객체가 아니다 ({_describe(person)})")
        person_id = person.get("id")
        if not _is_int(person_id):
            raise ValueError(f"people[{index}]에 정수 id가 없다 ({_describe(person_id)})")
        found.add(person_id)
    missing = sorted(set(expected_ids) - found)
    if missing:
        listed = ", ".join(str(i) for i in missing)
        raise ValueError(f"선수 정보 응답에 요청한 ID {len(missing)}명이 빠졌다: {listed}")


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
    batches = people_batches(player_ids)
    outcomes = [
        fetch_to_raw(
            getter,
            raw_dir,
            raw_people_path(index),
            statsapi_people_url(batch),
            check=partial(check_people_body, expected_ids=batch),
            force=force,
        )
        for index, batch in enumerate(batches)
    ]
    # 모든 묶음을 받은 뒤에만 지운다 (중간에 실패하면 이전 원본을 그대로 둔다)
    for rel in _people_paths(raw_dir, first=len(batches)):
        remove_raw(raw_dir, rel)
    return outcomes


def _people_paths(raw_dir: Path, *, first: int) -> list[Path]:
    """디스크나 매니페스트에 있는 선수 정보 원본 중 묶음 번호가 first 이상인 것의 상대 경로.

    번호 사이가 비어 있어도 찾는다. 이름이 `contracts.raw_people_path(n)`과 정확히 같은
    것만 고르므로 시즌 기록 등 다른 원본은 건드리지 않는다.
    """
    directory = raw_people_path(0).parent
    candidates = {Path(key) for key in load_manifest(raw_dir)}
    if (raw_dir / directory).is_dir():
        candidates.update(p.relative_to(raw_dir) for p in (raw_dir / directory).iterdir())
    stale: list[Path] = []
    for rel in candidates:
        number = rel.stem.rpartition("_")[2]
        if number.isdecimal() and int(number) >= first and raw_people_path(int(number)) == rel:
            stale.append(rel)
    return sorted(stale)


# ---------------------------------------------------------------------------
# v0.5 세이버메트릭스(WAR) — SPEC F1 ⑤·E26. 레인 A가 구현한다.
# ---------------------------------------------------------------------------


def saber_splits(raw_dir: Path, season: int, role: Role) -> list[dict[str, Any]]:
    """저장된 세이버메트릭스 원본(`contracts.raw_saber_path`)의 splits를 읽는다.

    Raises:
        FileNotFoundError: 원본이 없을 때.
    """
    doc = read_json(raw_dir, raw_saber_path(season, role))
    splits: list[dict[str, Any]] = doc["stats"][0]["splits"]
    return splits


def collect_saber(
    getter: Getter, raw_dir: Path, season: int, role: Role, *, force: bool = False
) -> CollectOutcome:
    """세이버메트릭스 하나를 받는다 (`contracts.statsapi_saber_url` → `contracts.raw_saber_path`).

    `collect_season_stats`와 같은 규칙이다 — 캐시가 있으면 요청하지 않고(force면 다시), 응답은
    `check_stats_body`(형식·totalSplits = 행 수)로 검사한 뒤 저장한다.
    """
    return fetch_to_raw(
        getter,
        raw_dir,
        raw_saber_path(season, role),
        statsapi_saber_url(season, role),
        check=check_stats_body,
        force=force,
    )
