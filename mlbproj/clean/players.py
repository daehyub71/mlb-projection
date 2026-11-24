"""선수 표 (SPEC §6 `players`, D23). 레인 A가 구현한다.

원본은 `collect.raw.read_json`과 `collect.statsapi.stats_splits`로만 읽는다.
"""

import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from mlbproj.collect.raw import read_json
from mlbproj.collect.statsapi import stats_splits
from mlbproj.contracts import PLAYERS, ROLES, SEASONS, Column, check_frame, raw_people_path

# 생년월일 형식 (contracts: ISO 날짜)
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def people_records(raw_dir: Path) -> list[dict[str, object]]:
    """선수 정보 원본 묶음을 모두 읽어 `people` 항목을 이어 붙인다.

    묶음은 `contracts.raw_people_path(i)`를 i = 0부터 파일이 없을 때까지 읽는다(이름이 다른
    파일은 읽지 않는다 — TASKS 미해소 이슈 ③). 같은 ID가 여러 묶음에 있으면 값이 완전히 같을 때만
    하나로 두고, 다르면 ValueError.

    Raises:
        FileNotFoundError: 묶음이 하나도 없을 때.
        ValueError: 묶음끼리 같은 선수의 값이 다를 때.
    """
    records: list[dict[str, object]] = []
    # 선수 ID → (먼저 읽은 항목, 그 묶음 경로). 값 비교는 중첩 필드까지 사전 == 로 한다
    seen: dict[object, tuple[dict[str, object], Path]] = {}
    batch = 0
    while (raw_dir / (rel := raw_people_path(batch))).is_file():
        people: list[dict[str, object]] = read_json(raw_dir, rel)["people"]
        for person in people:
            person_id = person["id"]
            first = seen.get(person_id)
            if first is None:
                seen[person_id] = (person, rel)
                records.append(person)
            elif first[0] != person:
                raise ValueError(
                    f"{rel.as_posix()}: 선수 {person_id}의 값이 {first[1].as_posix()}에서"
                    " 먼저 읽은 값과 다르다"
                )
        batch += 1
    if batch == 0:
        raise FileNotFoundError(
            f"선수 정보 원본이 없다: {(raw_dir / raw_people_path(0)).as_posix()}"
        )
    return records


def _sub_field(person: Mapping[str, object], key: str, field: str) -> object:
    """`person[key][field]`를 꺼낸다. 어느 쪽이든 없으면 None (결측)."""
    value = person.get(key)
    return value.get(field) if isinstance(value, dict) else None


def _player_row(person: Mapping[str, object], last_season: int) -> dict[str, object]:
    """선수 정보 한 명을 `PLAYERS` 열 순서의 행으로 바꾼다.

    결측을 허용하지 않는 active는 bool이 아니면 멈춘다(None을 bool로 바꾸면 조용히 False가 된다).
    생년월일은 ISO 날짜(YYYY-MM-DD)가 아니면 멈춘다.
    """
    person_id = person["id"]
    active = person.get("active")
    if not isinstance(active, bool):
        raise ValueError(f"선수 {person_id}의 active가 참·거짓이 아니다 ({active!r})")
    birth_date = person.get("birthDate")
    if birth_date is not None and not (
        isinstance(birth_date, str) and _ISO_DATE.fullmatch(birth_date)
    ):
        raise ValueError(f"선수 {person_id}의 birthDate가 YYYY-MM-DD가 아니다 ({birth_date!r})")
    return {
        "player_id": person_id,
        "name": person["fullName"],
        "birth_date": birth_date,
        "primary_position": _sub_field(person, "primaryPosition", "abbreviation"),
        "bats": _sub_field(person, "batSide", "code"),
        "throws": _sub_field(person, "pitchHand", "code"),
        "active": active,
        "last_mlb_season": last_season,
    }


def _dtype(col: Column) -> str:
    """계약의 열 → pandas 형식 (`interim.read_table`이 읽어 들이는 형식과 같다)."""
    if col.kind == "int":
        return "Int64" if col.nullable else "int64"
    if col.kind == "float":
        return "float64"
    if col.kind == "bool":
        return "boolean" if col.nullable else "bool"
    # 문자열 열: 모두 결측이어도 object가 아니라 문자열 형식(결측은 NaN)으로 둔다
    return "str"


def build_players(raw_dir: Path) -> pd.DataFrame:
    """선수 표를 만든다 (`contracts.PLAYERS`, `check_frame` 통과, player_id 오름차순).

    - 대상: `contracts.SEASONS` 모든 시즌의 API 타격·투구 원본에 **행이 하나라도 있는** 선수 전원
      (타석·상대 타자가 0이어도 행이 있으면 MLB 기록이다).
    - name = fullName, birth_date = birthDate(YYYY-MM-DD, 없으면 결측), primary_position =
      primaryPosition.abbreviation, bats = batSide.code, throws = pitchHand.code (없으면 결측).
    - active = 선수 정보의 active 그대로(bool). 은퇴 판정이 아니다 — 표시용(D23, E16).
    - last_mlb_season = 그 선수가 타격 또는 투구 원본에 행으로 나온 마지막 시즌.

    Raises:
        ValueError: 원본에 나온 선수가 선수 정보에 없을 때 (빠진 ID를 적는다).
    """
    # 역할을 가리지 않고 행이 있으면 기록이다 (타석·상대 타자 0인 행 포함).
    # last_mlb_season은 모든 시즌을 본 값이라 표시용이다 — 과거 목표 시즌의 모델 입력으로 쓰면 누설(N2)
    last_seasons: dict[int, int] = {}
    for season in SEASONS:
        for role in ROLES:
            for split in stats_splits(raw_dir, season, role):
                player_id: int = split["player"]["id"]
                last_seasons[player_id] = max(season, last_seasons.get(player_id, season))

    people = {person["id"]: person for person in people_records(raw_dir)}
    missing = sorted(set(last_seasons) - set(people))
    if missing:
        listed = ", ".join(str(i) for i in missing)
        raise ValueError(f"시즌 원본에 나온 선수 {len(missing)}명이 선수 정보에 없다: {listed}")

    rows = [_player_row(people[pid], last_seasons[pid]) for pid in sorted(last_seasons)]
    df = pd.DataFrame(rows, columns=PLAYERS.column_names)
    df = df.astype({col.name: _dtype(col) for col in PLAYERS.columns})
    problems = check_frame(df, PLAYERS)
    if problems:
        raise ValueError("; ".join(problems))
    return df
