"""선수 표(clean/players.py, SPEC §6 `players`·D23·E16) 테스트.

기준 상태는 `tests/fixtures/raw`(실제 응답에서 14명만 잘라 낸 원본, SOURCE.md)다.
기댓값은 구현을 돌려서가 아니라 픽스처 JSON을 직접 읽어 만든다.
원본을 바꾸는 테스트는 tmp_path에 복사본을 만들어 고친다(픽스처 자체는 건드리지 않는다).
"""

import copy
import json
import re
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mlbproj.clean.players import build_players, people_records
from mlbproj.contracts import (
    PLAYERS,
    ROLES,
    SEASONS,
    Role,
    check_frame,
    raw_people_path,
    raw_statsapi_path,
)
from mlbproj.interim import read_table, write_table

FIXTURE_RAW = Path(__file__).parent / "fixtures" / "raw"

CABRERA = 408234  # 2023 이후 미출전, active=false
WAINWRIGHT = 425794  # 2023 이후 미출전, active=false
VENDITTE = 519381  # 마지막 출전 2020, 2019 타격 행 0 PA
COLE = 543037
NOLA = 605400
IGLESIAS = 628452  # 2026 타격 행 0 PA + 2026 투구 행
OHTANI = 660271  # 투타 겸업, primaryPosition TWP
MCGONIGLE = 805808  # 2026 데뷔
SKENES = 694973  # 투수만, 2024 데뷔 (타격 행 없음)
FICTIONAL = 999001  # 픽스처에 없는 가짜 ID (테스트에서만 만든다)


# ---------------------------------------------------------------------------
# 픽스처 읽기·고치기 도우미
# ---------------------------------------------------------------------------


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _people(raw_dir: Path, batch: int) -> list[dict[str, Any]]:
    people: list[dict[str, Any]] = _load(raw_dir / raw_people_path(batch))["people"]
    return people


def _all_people(raw_dir: Path) -> list[dict[str, Any]]:
    """픽스처의 선수 정보 두 묶음(000·001)을 순서대로 이어 붙인다."""
    return _people(raw_dir, 0) + _people(raw_dir, 1)


def _person(raw_dir: Path, player_id: int) -> dict[str, Any]:
    for person in _all_people(raw_dir):
        if person["id"] == player_id:
            return person
    raise AssertionError(f"픽스처에 {player_id}가 없다")


def _write_people_file(raw_dir: Path, name: str, people: list[dict[str, Any]]) -> None:
    """선수 정보 원본과 같은 모양({"copyright", "people"})으로 statsapi/ 아래에 쓴다."""
    doc = _load(raw_dir / raw_people_path(0))
    doc["people"] = people
    _dump(raw_dir / raw_people_path(0).parent / name, doc)


def _edit_people(
    raw_dir: Path, batch: int, edit: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
) -> None:
    path = raw_dir / raw_people_path(batch)
    doc = _load(path)
    doc["people"] = edit(doc["people"])
    _dump(path, doc)


def _splits(raw_dir: Path, season: int, role: Role) -> list[dict[str, Any]]:
    splits: list[dict[str, Any]] = _load(raw_dir / raw_statsapi_path(season, role))["stats"][0][
        "splits"
    ]
    return splits


def _drop_splits(raw_dir: Path, season: int, role: Role, player_id: int) -> None:
    """시즌 원본에서 한 선수의 행을 지우고 totalSplits를 맞춘다."""
    path = raw_dir / raw_statsapi_path(season, role)
    doc = _load(path)
    block = doc["stats"][0]
    before = len(block["splits"])
    block["splits"] = [s for s in block["splits"] if s["player"]["id"] != player_id]
    assert len(block["splits"]) == before - 1, "지울 행이 정확히 하나여야 한다"
    block["totalSplits"] = len(block["splits"])
    _dump(path, doc)


def _stats_seasons(raw_dir: Path) -> dict[int, set[int]]:
    """선수 ID → 타격·투구 원본에 행으로 나온 시즌들 (타석·상대 타자 0인 행 포함)."""
    seasons: dict[int, set[int]] = {}
    for season in SEASONS:
        for role in ROLES:
            for split in _splits(raw_dir, season, role):
                seasons.setdefault(split["player"]["id"], set()).add(season)
    return seasons


def _pid(record: Mapping[str, object]) -> int:
    value = record["id"]
    assert isinstance(value, int)
    return value


def _rows(df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """player_id → 행 (값은 파이썬 기본형)."""
    records = [{str(k): v for k, v in row.items()} for row in df.to_dict("records")]
    return {int(row["player_id"]): row for row in records}


@pytest.fixture
def raw_copy(tmp_path: Path) -> Path:
    """고쳐 쓸 원본 복사본."""
    dst = tmp_path / "raw"
    shutil.copytree(FIXTURE_RAW, dst)
    return dst


# ---------------------------------------------------------------------------
# people_records
# ---------------------------------------------------------------------------


def test_clean_players_people_records_concatenates_batches_in_order() -> None:
    expected = _all_people(FIXTURE_RAW)
    assert len(expected) == 14

    records = people_records(FIXTURE_RAW)

    # 묶음 000 → 001 순서로 people 항목을 그대로 이어 붙인다
    assert records == expected
    assert [_pid(r) for r in records] == [p["id"] for p in expected]


def test_clean_players_people_records_stops_at_first_missing_index(raw_copy: Path) -> None:
    # people_002가 없으므로 people_003은 읽지 않는다. 읽으면 값이 다른 중복으로 실패하거나 15명이 된다
    conflicting = copy.deepcopy(_person(raw_copy, CABRERA))
    conflicting["fullName"] = "Not Miguel Cabrera"
    extra = copy.deepcopy(_person(raw_copy, CABRERA))
    extra["id"] = FICTIONAL
    _write_people_file(raw_copy, raw_people_path(3).name, [conflicting, extra])

    records = people_records(raw_copy)

    assert records == _all_people(FIXTURE_RAW)
    assert FICTIONAL not in {_pid(r) for r in records}


@pytest.mark.parametrize("name", ["people_old.json", "people_2.json", "people_002.json.tmp"])
def test_clean_players_people_records_ignores_other_file_names(raw_copy: Path, name: str) -> None:
    # 이름이 contracts.raw_people_path(i)와 다른 파일은 읽지 않는다 (TASKS 미해소 이슈 ③)
    assert raw_people_path(2).name != name
    conflicting = copy.deepcopy(_person(raw_copy, NOLA))
    conflicting["fullName"] = "Not Aaron Nola"
    extra = copy.deepcopy(_person(raw_copy, NOLA))
    extra["id"] = FICTIONAL
    _write_people_file(raw_copy, name, [conflicting, extra])

    records = people_records(raw_copy)

    assert records == _all_people(FIXTURE_RAW)


def test_clean_players_people_records_identical_duplicate_kept_once(raw_copy: Path) -> None:
    # 같은 ID가 여러 묶음에 값까지 똑같이 있으면 하나로 둔다
    duplicates = [
        copy.deepcopy(_person(raw_copy, CABRERA)),
        copy.deepcopy(_person(raw_copy, NOLA)),
    ]
    _write_people_file(raw_copy, raw_people_path(2).name, duplicates)

    records = people_records(raw_copy)

    ids = [_pid(r) for r in records]
    assert len(records) == 14
    assert len(set(ids)) == 14
    expected = _all_people(FIXTURE_RAW)
    assert sorted(records, key=_pid) == sorted(expected, key=lambda p: int(p["id"]))


def _change_full_name(person: dict[str, Any]) -> None:
    person["fullName"] = person["fullName"] + " Jr."


def _change_position(person: dict[str, Any]) -> None:
    person["primaryPosition"]["abbreviation"] = "SS"


def _change_active(person: dict[str, Any]) -> None:
    person["active"] = not person["active"]


def _drop_birth_date(person: dict[str, Any]) -> None:
    del person["birthDate"]


@pytest.mark.parametrize(
    "change",
    [_change_full_name, _change_position, _change_active, _drop_birth_date],
    ids=["fullName", "nested primaryPosition", "active", "missing birthDate"],
)
def test_clean_players_people_records_conflicting_duplicate_raises(
    raw_copy: Path, change: Callable[[dict[str, Any]], None]
) -> None:
    changed = copy.deepcopy(_person(raw_copy, CABRERA))
    change(changed)
    _write_people_file(raw_copy, raw_people_path(2).name, [changed])

    with pytest.raises(ValueError):
        people_records(raw_copy)


def test_clean_players_people_records_conflict_between_existing_batches_raises(
    raw_copy: Path,
) -> None:
    # 기존 두 묶음 사이에서도 같다: 001에 값이 다른 Cabrera를 넣는다
    changed = copy.deepcopy(_person(raw_copy, CABRERA))
    changed["birthDate"] = "1983-04-19"
    _edit_people(raw_copy, 1, lambda people: [*people, changed])

    with pytest.raises(ValueError):
        people_records(raw_copy)


def test_clean_players_people_records_no_batch_raises(tmp_path: Path) -> None:
    empty = tmp_path / "raw"
    (empty / raw_people_path(0).parent).mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        people_records(empty)


def test_clean_players_people_records_missing_first_batch_raises(raw_copy: Path) -> None:
    # people_000이 없으면 i = 0에서 멈춘다 → 읽은 묶음이 없다 (people_001이 있어도)
    (raw_copy / raw_people_path(0)).unlink()
    assert (raw_copy / raw_people_path(1)).is_file()

    with pytest.raises(FileNotFoundError):
        people_records(raw_copy)


# ---------------------------------------------------------------------------
# build_players
# ---------------------------------------------------------------------------


def test_clean_players_build_players_contract_and_order() -> None:
    expected_ids = sorted(p["id"] for p in _all_people(FIXTURE_RAW))
    assert expected_ids == sorted(_stats_seasons(FIXTURE_RAW))
    assert len(expected_ids) == 14

    df = build_players(FIXTURE_RAW)

    assert check_frame(df, PLAYERS) == []
    assert len(df) == 14
    # player_id 오름차순
    assert [int(v) for v in df["player_id"]] == expected_ids


def test_clean_players_build_players_values_match_people() -> None:
    df = build_players(FIXTURE_RAW)

    rows = _rows(df)
    for person in _all_people(FIXTURE_RAW):
        row = rows[person["id"]]
        assert row["name"] == person["fullName"]
        assert row["birth_date"] == person["birthDate"]
        assert row["primary_position"] == person["primaryPosition"]["abbreviation"]
        assert row["bats"] == person["batSide"]["code"]
        assert row["throws"] == person["pitchHand"]["code"]
        # active는 선수 정보 그대로 (은퇴 판정이 아니다, D23·E16)
        assert bool(row["active"]) is person["active"]


def test_clean_players_build_players_last_mlb_season_from_stats() -> None:
    expected = {pid: max(seasons) for pid, seasons in _stats_seasons(FIXTURE_RAW).items()}

    df = build_players(FIXTURE_RAW)

    assert {pid: int(row["last_mlb_season"]) for pid, row in _rows(df).items()} == expected


def test_clean_players_build_players_known_cases() -> None:
    df = build_players(FIXTURE_RAW)
    rows = _rows(df)

    # 2023 이후 미출전(은퇴) — active=false, 마지막 시즌 2023
    assert bool(rows[CABRERA]["active"]) is False
    assert int(rows[CABRERA]["last_mlb_season"]) == 2023
    assert bool(rows[WAINWRIGHT]["active"]) is False
    assert int(rows[WAINWRIGHT]["last_mlb_season"]) == 2023
    # 마지막 출전 2020 (2019 타격 행은 0 PA지만 기록이다)
    assert int(rows[VENDITTE]["last_mlb_season"]) == 2020
    # 2026 데뷔
    assert int(rows[MCGONIGLE]["last_mlb_season"]) == 2026
    assert bool(rows[MCGONIGLE]["active"]) is True
    # 2025 결장 후 2026 복귀
    assert int(rows[COLE]["last_mlb_season"]) == 2026
    # 투타 겸업
    assert rows[OHTANI]["primary_position"] == "TWP"
    assert rows[OHTANI]["bats"] == "L"
    assert rows[OHTANI]["throws"] == "R"
    assert rows[OHTANI]["name"] == "Shohei Ohtani"


def test_clean_players_build_players_birth_date_iso_format() -> None:
    df = build_players(FIXTURE_RAW)

    assert df["birth_date"].notna().all()
    for value in df["birth_date"]:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)), value
    assert _rows(df)[CABRERA]["birth_date"] == "1983-04-18"


def test_clean_players_build_players_zero_pa_row_is_a_record(raw_copy: Path) -> None:
    # Venditte의 투구 행(2019·2020)을 모두 지우면 남는 것은 2019 타격 행(0 PA) 하나뿐이다.
    # 타석이 0이어도 행이 있으면 MLB 기록이다 → 대상에 남고 마지막 시즌은 2019
    venditte_2019 = [s for s in _splits(raw_copy, 2019, "bat") if s["player"]["id"] == VENDITTE]
    assert venditte_2019[0]["stat"]["plateAppearances"] == 0
    _drop_splits(raw_copy, 2019, "pit", VENDITTE)
    _drop_splits(raw_copy, 2020, "pit", VENDITTE)
    # Iglesias의 2026 투구 행을 지워도 2026 타격 행(0 PA)이 남는다 → 마지막 시즌 2026
    iglesias_2026 = [s for s in _splits(raw_copy, 2026, "bat") if s["player"]["id"] == IGLESIAS]
    assert iglesias_2026[0]["stat"]["plateAppearances"] == 0
    _drop_splits(raw_copy, 2026, "pit", IGLESIAS)

    df = build_players(raw_copy)

    rows = _rows(df)
    assert len(df) == 14
    assert int(rows[VENDITTE]["last_mlb_season"]) == 2019
    assert int(rows[IGLESIAS]["last_mlb_season"]) == 2026
    assert check_frame(df, PLAYERS) == []


@pytest.mark.parametrize(
    ("field", "column"),
    [
        ("birthDate", "birth_date"),
        ("primaryPosition", "primary_position"),
        ("batSide", "bats"),
        ("pitchHand", "throws"),
    ],
)
def test_clean_players_build_players_missing_optional_field_is_null(
    raw_copy: Path, field: str, column: str
) -> None:
    def drop(people: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for person in people:
            if person["id"] == MCGONIGLE:
                del person[field]
        return people

    _edit_people(raw_copy, 0, drop)

    df = build_players(raw_copy)

    assert check_frame(df, PLAYERS) == []
    rows = _rows(df)
    assert pd.isna(rows[MCGONIGLE][column])
    # 다른 선수는 그대로
    assert not pd.isna(rows[OHTANI][column])


def test_clean_players_build_players_missing_person_raises(raw_copy: Path) -> None:
    _edit_people(raw_copy, 1, lambda people: [p for p in people if p["id"] != VENDITTE])

    with pytest.raises(ValueError, match=str(VENDITTE)):
        build_players(raw_copy)


def test_clean_players_build_players_missing_people_all_named(raw_copy: Path) -> None:
    _edit_people(raw_copy, 0, lambda people: [p for p in people if p["id"] != MCGONIGLE])
    _edit_people(raw_copy, 1, lambda people: [p for p in people if p["id"] != VENDITTE])

    with pytest.raises(ValueError) as excinfo:
        build_players(raw_copy)

    message = str(excinfo.value)
    assert str(VENDITTE) in message
    assert str(MCGONIGLE) in message


def test_clean_players_build_players_ignores_people_without_stats(raw_copy: Path) -> None:
    # 대상은 시즌 원본에 행이 있는 선수다. 선수 정보에만 있는 사람은 표에 넣지 않는다
    extra = copy.deepcopy(_person(raw_copy, CABRERA))
    extra["id"] = FICTIONAL
    extra["fullName"] = "Fictional Player"
    _write_people_file(raw_copy, raw_people_path(2).name, [extra])

    df = build_players(raw_copy)

    assert len(df) == 14
    assert FICTIONAL not in {int(v) for v in df["player_id"]}


def test_clean_players_people_records_conflict_message_names_id_and_files(raw_copy: Path) -> None:
    changed = copy.deepcopy(_person(raw_copy, CABRERA))
    changed["fullName"] = "Not Miguel Cabrera"
    _write_people_file(raw_copy, raw_people_path(2).name, [changed])

    with pytest.raises(ValueError) as excinfo:
        people_records(raw_copy)

    # 어느 선수가 어느 두 묶음에서 어긋났는지 적는다
    message = str(excinfo.value)
    assert str(CABRERA) in message
    assert raw_people_path(2).as_posix() in message
    assert raw_people_path(0).as_posix() in message


def test_clean_players_build_players_zero_bf_pitching_row_is_a_record(raw_copy: Path) -> None:
    # Skenes는 투구 행만 있다. 2025·2026 행을 지우고 2024 행의 상대 타자를 0으로 만들어도
    # 행이 있으면 MLB 기록이다 → 대상에 남고 마지막 시즌은 2024
    assert all(s["player"]["id"] != SKENES for x in SEASONS for s in _splits(raw_copy, x, "bat"))
    _drop_splits(raw_copy, 2025, "pit", SKENES)
    _drop_splits(raw_copy, 2026, "pit", SKENES)
    path = raw_copy / raw_statsapi_path(2024, "pit")
    doc = _load(path)
    for split in doc["stats"][0]["splits"]:
        if split["player"]["id"] == SKENES:
            split["stat"]["battersFaced"] = 0
    _dump(path, doc)

    df = build_players(raw_copy)

    assert int(_rows(df)[SKENES]["last_mlb_season"]) == 2024
    assert len(df) == 14


@pytest.mark.parametrize("active", [None, "true", 1, 0], ids=["null", "string", "one", "zero"])
def test_clean_players_build_players_active_not_bool_raises(raw_copy: Path, active: Any) -> None:
    # active는 결측을 허용하지 않는다. None·문자열·숫자를 bool로 바꾸면 조용히 값이 정해진다
    def set_active(people: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for person in people:
            if person["id"] == CABRERA:
                person["active"] = active
        return people

    _edit_people(raw_copy, 0, set_active)

    with pytest.raises(ValueError, match=str(CABRERA)):
        build_players(raw_copy)


def test_clean_players_build_players_active_missing_raises(raw_copy: Path) -> None:
    def drop_active(people: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for person in people:
            if person["id"] == CABRERA:
                del person["active"]
        return people

    _edit_people(raw_copy, 0, drop_active)

    with pytest.raises(ValueError, match=str(CABRERA)):
        build_players(raw_copy)


@pytest.mark.parametrize("birth_date", ["04/18/1983", "1983-4-18", 19830418, ""])
def test_clean_players_build_players_birth_date_not_iso_raises(
    raw_copy: Path, birth_date: Any
) -> None:
    def set_birth_date(people: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for person in people:
            if person["id"] == CABRERA:
                person["birthDate"] = birth_date
        return people

    _edit_people(raw_copy, 0, set_birth_date)

    with pytest.raises(ValueError, match=str(CABRERA)):
        build_players(raw_copy)


def test_clean_players_build_players_dtypes() -> None:
    df = build_players(FIXTURE_RAW)

    # 정수가 실수로, 참·거짓이 object로 바뀌지 않는다
    assert df["player_id"].dtype == "int64"
    assert df["last_mlb_season"].dtype == "int64"
    assert df["active"].dtype == "bool"
    for column in ("name", "birth_date", "primary_position", "bats", "throws"):
        assert pd.api.types.is_string_dtype(df[column].dtype), column
        assert not pd.api.types.is_object_dtype(df[column].dtype), column


def test_clean_players_build_players_roundtrip_through_interim(tmp_path: Path) -> None:
    df = build_players(FIXTURE_RAW)
    path = tmp_path / "players.csv"

    write_table(df, PLAYERS, path)

    # 정제 표는 이미 키 순서다 — 쓰고 읽어도 값·형식이 그대로
    pd.testing.assert_frame_equal(read_table(PLAYERS, path), df)


def test_clean_players_build_players_all_missing_column_keeps_string_dtype(
    raw_copy: Path, tmp_path: Path
) -> None:
    def drop_birth_dates(people: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for person in people:
            del person["birthDate"]
        return people

    _edit_people(raw_copy, 0, drop_birth_dates)
    _edit_people(raw_copy, 1, drop_birth_dates)

    df = build_players(raw_copy)

    assert df["birth_date"].isna().all()
    # 모두 결측이어도 object(None)가 아니라 읽어 들인 표와 같은 문자열 형식이다
    assert not pd.api.types.is_object_dtype(df["birth_date"].dtype)
    path = tmp_path / "players.csv"
    write_table(df, PLAYERS, path)
    pd.testing.assert_frame_equal(read_table(PLAYERS, path), df)


def test_clean_players_build_players_independent_of_raw_order(raw_copy: Path) -> None:
    # 원본 안의 행 순서(시즌 split·선수 정보 항목)가 바뀌어도 같은 표가 나온다
    expected = build_players(FIXTURE_RAW)
    for season in SEASONS:
        for role in ROLES:
            path = raw_copy / raw_statsapi_path(season, role)
            doc = _load(path)
            doc["stats"][0]["splits"].reverse()
            _dump(path, doc)
    _edit_people(raw_copy, 0, lambda people: people[::-1])
    _edit_people(raw_copy, 1, lambda people: people[::-1])

    pd.testing.assert_frame_equal(build_players(raw_copy), expected)


@pytest.mark.parametrize(("season", "role"), [(2019, "bat"), (2026, "pit")])
def test_clean_players_build_players_missing_season_file_raises(
    raw_copy: Path, season: int, role: Role
) -> None:
    # 시즌 원본이 없으면 "그 시즌 미출전"으로 넘기지 않고 멈춘다 (수집 누락과 미출전을 섞지 않는다, F2)
    (raw_copy / raw_statsapi_path(season, role)).unlink()

    with pytest.raises(FileNotFoundError):
        build_players(raw_copy)
