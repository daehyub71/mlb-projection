"""Baseball Savant 수집 — 리더보드·OAA·FRV CSV (SPEC F1·E2~E4·E18). 레인 B가 구현한다.

저장은 `collect.raw.fetch_to_raw`만 쓴다. URL·경로·필수 열은 `contracts`만 쓴다.
원본은 받은 그대로 저장하고(SPEC D6), 연도 채우기 같은 손질은 `load_savant`에서만 한다.
"""

from pathlib import Path

import pandas as pd

from mlbproj.collect.raw import CollectOutcome, Getter
from mlbproj.contracts import Role, SavantKind


def required_columns(kind: SavantKind) -> tuple[str, ...]:
    """종류별 필수 열 (`contracts.SAVANT_*_COLUMNS`)."""
    raise NotImplementedError


def check_csv_body(body: bytes, kind: SavantKind) -> None:
    """Savant 응답이 CSV인지 검사한다 (`fetch_to_raw`의 check로 쓴다).

    UTF-8(BOM 허용)로 읽히고, 머리줄에 `required_columns(kind)`가 모두 있고, 데이터 행이
    1개 이상이어야 한다. Savant는 오류일 때도 200으로 HTML을 돌려줄 수 있다.

    Raises:
        ValueError: 위 조건이 어긋날 때. 빠진 열 이름을 메시지에 적는다.
    """
    raise NotImplementedError


def collect_leaderboard(
    getter: Getter, raw_dir: Path, season: int, role: Role, *, force: bool = False
) -> CollectOutcome:
    """타자(`bat` → kind `batter`) 또는 투수(`pit` → `pitcher`) 리더보드를 받는다.

    URL은 `contracts.savant_leaderboard_url`, 경로는 `contracts.raw_savant_path(season, kind)`.
    캐시가 있으면 요청하지 않는다.
    """
    raise NotImplementedError


def collect_oaa(
    getter: Getter, raw_dir: Path, season: int, *, force: bool = False
) -> CollectOutcome:
    """OAA 리더보드를 받는다 (`contracts.savant_oaa_url`, kind `oaa`).

    Raises:
        ValueError: season이 `contracts.FIELDING_YEARS`에 없을 때 (요청하지 않는다).
    """
    raise NotImplementedError


def collect_frv(
    getter: Getter, raw_dir: Path, season: int, *, force: bool = False
) -> CollectOutcome:
    """Fielding Run Value 리더보드를 받는다 (`contracts.savant_frv_url`, kind `frv`).

    Raises:
        ValueError: season이 `contracts.FIELDING_YEARS`에 없을 때 (요청하지 않는다).
    """
    raise NotImplementedError


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
    raise NotImplementedError
