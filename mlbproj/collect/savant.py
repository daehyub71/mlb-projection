"""Baseball Savant 수집 — 리더보드·OAA·FRV CSV (SPEC F1·E2~E4·E18). 레인 B가 구현한다.

저장은 `collect.raw.fetch_to_raw`만 쓴다. URL·경로·필수 열은 `contracts`만 쓴다.
원본은 받은 그대로 저장하고(SPEC D6), 연도 채우기 같은 손질은 `load_savant`에서만 한다.
"""

import csv
import io
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from mlbproj.collect.raw import CollectOutcome, Getter, fetch_to_raw, read_csv
from mlbproj.contracts import (
    FIELDING_YEARS,
    SAVANT_BATTER_COLUMNS,
    SAVANT_FRV_COLUMNS,
    SAVANT_OAA_COLUMNS,
    SAVANT_PITCHER_COLUMNS,
    Role,
    SavantKind,
    raw_savant_path,
    savant_frv_url,
    savant_leaderboard_url,
    savant_oaa_url,
)

# 종류별 필수 열 (F2 ① 검사 대상)
_REQUIRED_COLUMNS: dict[SavantKind, tuple[str, ...]] = {
    "batter": SAVANT_BATTER_COLUMNS,
    "pitcher": SAVANT_PITCHER_COLUMNS,
    "oaa": SAVANT_OAA_COLUMNS,
    "frv": SAVANT_FRV_COLUMNS,
}

# 리더보드 역할 → 원본 종류
_LEADERBOARD_KIND: dict[Role, SavantKind] = {"bat": "batter", "pit": "pitcher"}

# 수비 리더보드. 원본 year가 비어 있거나(OAA, E4) 없다(FRV, E18)
_FIELDING_KINDS: frozenset[SavantKind] = frozenset({"oaa", "frv"})

# FRV 원본의 선수 ID 열 이름 (E18). 읽을 때 `player_id`로 바꾼다
_FRV_ID_COLUMN = "id"

# 오류 메시지에 적는 값·행의 최대 개수와 머리줄 앞부분 길이
_SHOW_MAX = 5
_HEADER_PREVIEW_CHARS = 60


def required_columns(kind: SavantKind) -> tuple[str, ...]:
    """종류별 필수 열 (`contracts.SAVANT_*_COLUMNS`)."""
    return _REQUIRED_COLUMNS[kind]


def check_csv_body(body: bytes, kind: SavantKind) -> None:
    """Savant 응답이 CSV인지 검사한다 (`fetch_to_raw`의 check로 쓴다).

    UTF-8(BOM 허용)로 읽히고, 머리줄에 `required_columns(kind)`가 모두 있고, 데이터 행이
    1개 이상이어야 한다. Savant는 오류일 때도 200으로 HTML을 돌려줄 수 있다.

    Raises:
        ValueError: 위 조건이 어긋날 때. 빠진 열 이름을 메시지에 적는다.
    """
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Savant {kind} 응답이 UTF-8이 아니다: {exc}") from exc
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as exc:
        raise ValueError(f"Savant {kind} 응답을 CSV로 읽을 수 없다: {exc}") from exc
    if not rows:
        raise ValueError(f"Savant {kind} 응답이 비어 있다")
    header = rows[0]
    # 열 이름은 부분 문자열이 아니라 CSV 필드 단위로 정확히 맞춘다
    missing = [col for col in required_columns(kind) if col not in header]
    if missing:
        # HTML 오류 페이지처럼 CSV가 아닌 응답을 알아볼 수 있게 머리줄 앞부분을 붙인다
        preview = ",".join(header)[:_HEADER_PREVIEW_CHARS]
        raise ValueError(
            f"Savant {kind} 응답에 필수 열이 없다: {missing} (머리줄 앞부분: {preview!r})"
        )
    # 빈 줄(모든 필드가 공백)은 데이터 행으로 세지 않는다
    data = [(n, row) for n, row in enumerate(rows[1:], start=1) if any(f.strip() for f in row)]
    if not data:
        raise ValueError(f"Savant {kind} 응답에 데이터 행이 없다")
    # 필드 수가 머리줄과 다른 행은 중간에 잘린 응답이다. pandas는 모자란 필드를 빈 값으로 채워
    # 조용히 읽으므로 저장 전에 막는다
    ragged = [n for n, row in data if len(row) != len(header)]
    if ragged:
        raise ValueError(
            f"Savant {kind} 응답에 필드 수가 머리줄({len(header)}개)과 다른 데이터 행이 있다"
            f"(잘린 응답일 수 있다). 데이터 행 번호: {ragged[:_SHOW_MAX]}"
        )


def _collect(
    getter: Getter, raw_dir: Path, season: int, kind: SavantKind, url: str, *, force: bool
) -> CollectOutcome:
    """원본 하나를 검사해서 저장한다. 캐시가 있으면 요청하지 않는다."""

    def check(body: bytes) -> None:
        check_csv_body(body, kind)

    return fetch_to_raw(
        getter, raw_dir, raw_savant_path(season, kind), url, check=check, force=force
    )


def collect_leaderboard(
    getter: Getter, raw_dir: Path, season: int, role: Role, *, force: bool = False
) -> CollectOutcome:
    """타자(`bat` → kind `batter`) 또는 투수(`pit` → `pitcher`) 리더보드를 받는다.

    URL은 `contracts.savant_leaderboard_url`, 경로는 `contracts.raw_savant_path(season, kind)`.
    캐시가 있으면 요청하지 않는다.
    """
    kind = _LEADERBOARD_KIND[role]
    return _collect(
        getter, raw_dir, season, kind, savant_leaderboard_url(season, role), force=force
    )


def _check_fielding_season(season: int, kind: SavantKind) -> None:
    """수비 리더보드는 `FIELDING_YEARS`만 받는다. 요청·저장 전에 검사한다."""
    if season not in FIELDING_YEARS:
        raise ValueError(
            f"Savant {kind}는 {FIELDING_YEARS[0]}~{FIELDING_YEARS[-1]} 시즌만 받는다: {season}"
        )


def collect_oaa(
    getter: Getter, raw_dir: Path, season: int, *, force: bool = False
) -> CollectOutcome:
    """OAA 리더보드를 받는다 (`contracts.savant_oaa_url`, kind `oaa`).

    Raises:
        ValueError: season이 `contracts.FIELDING_YEARS`에 없을 때 (요청하지 않는다).
    """
    _check_fielding_season(season, "oaa")
    return _collect(getter, raw_dir, season, "oaa", savant_oaa_url(season), force=force)


def collect_frv(
    getter: Getter, raw_dir: Path, season: int, *, force: bool = False
) -> CollectOutcome:
    """Fielding Run Value 리더보드를 받는다 (`contracts.savant_frv_url`, kind `frv`).

    Raises:
        ValueError: season이 `contracts.FIELDING_YEARS`에 없을 때 (요청하지 않는다).
    """
    _check_fielding_season(season, "frv")
    return _collect(getter, raw_dir, season, "frv", savant_frv_url(season), force=force)


def _shown(values: Iterable[Any]) -> list[str]:
    """오류 메시지에 적을 값 목록. 빈 값은 `(빈 값)`, 정수로 읽힌 실수는 정수로 적는다."""

    def show(value: Any) -> str:
        if pd.isna(value):
            return "(빈 값)"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    return sorted({show(v) for v in values})[:_SHOW_MAX]


def _check_year(df: pd.DataFrame, season: int, name: str, *, allow_empty: bool) -> None:
    """원본 `year`가 요청 시즌과 다른 행이 있으면 ValueError. allow_empty면 빈 값은 넘긴다."""
    years = pd.to_numeric(df["year"], errors="coerce")
    wrong = years != season
    if allow_empty:
        wrong &= df["year"].notna()
    if bool(wrong.any()):
        found = _shown(df.loc[wrong, "year"])
        raise ValueError(f"{name}: 요청 시즌은 {season}인데 year가 다른 행이 있다: {found}")


def _cast_player_ids(df: pd.DataFrame, name: str) -> None:
    """`player_id`를 int64로 바꾼다. 빈 값·숫자가 아닌 값·소수가 있으면 ValueError."""
    ids = pd.to_numeric(df["player_id"], errors="coerce")
    # 빈 값·숫자가 아닌 값(NaN)·무한대·소수는 1로 나눈 나머지가 0이 아니다
    bad = (ids % 1) != 0
    if bool(bad.any()):
        found = _shown(df.loc[bad, "player_id"])
        raise ValueError(f"{name}: 정수가 아닌 선수 ID가 있다: {found}")
    df["player_id"] = ids.astype("int64")


def load_savant(raw_dir: Path, season: int, kind: SavantKind) -> pd.DataFrame:
    """저장된 Savant CSV를 읽는다.

    - BOM을 떼고 읽는다(`collect.raw.read_csv`).
    - 선수 ID 열은 `player_id`로 맞춘다(FRV는 원본이 `id`). 정수형이다.
    - `year` 열을 요청 시즌으로 채운다: OAA는 원본 `year`가 비어 있고(E4), FRV는 열이 없다.
      타자·투수 리더보드는 `year`가 있는데, 요청 시즌과 다른 값이 있으면 ValueError.
    - 다른 열은 이름·값을 바꾸지 않는다(정제는 M2).

    Raises:
        FileNotFoundError: 원본이 없을 때.
        ValueError: 리더보드의 year가 요청 시즌과 다를 때.
    """
    rel = raw_savant_path(season, kind)
    name = rel.as_posix()
    if not (raw_dir / rel).is_file():
        raise FileNotFoundError(f"Savant 원본이 없다: {(raw_dir / rel).as_posix()}")
    df = read_csv(raw_dir, rel)
    # 형식이 바뀐 원본은 KeyError 대신 빠진 열 이름으로 멈춘다
    missing = [col for col in required_columns(kind) if col not in df.columns]
    if missing:
        raise ValueError(f"{name}: 필수 열이 없다: {missing}")

    if kind == "frv":
        if "player_id" in df.columns:
            raise ValueError(f"{name}: FRV 원본에 `id`와 `player_id`가 함께 있다 (형식 변경)")
        df = df.rename(columns={_FRV_ID_COLUMN: "player_id"})
    if "year" in df.columns:
        # 타자·투수는 year가 모두 요청 시즌이어야 한다. 수비는 빈 값(E4)만 채우고,
        # 값이 들어 있으면 요청 시즌과 같아야 한다 — 다른 해 원본에 시즌을 덮어쓰지 않는다
        _check_year(df, season, name, allow_empty=kind in _FIELDING_KINDS)

    _cast_player_ids(df, name)
    # 기존 year 열은 제자리에서 바꾸고, FRV는 끝에 붙인다
    df["year"] = pd.Series(season, index=df.index, dtype="int64")
    return df
