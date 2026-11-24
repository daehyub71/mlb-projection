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

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeGuard

import pandas as pd

from mlbproj.collect import raw
from mlbproj.contracts import (
    FIELDING_YEARS,
    PEOPLE_FIELDS,
    PITCHER_POSITIONS,
    ROLES,
    SAVANT_BATTER_COLUMNS,
    SAVANT_FRV_COLUMNS,
    SAVANT_OAA_COLUMNS,
    SAVANT_PITCHER_COLUMNS,
    SHORT_SEASON,
    STATSAPI_HITTING_STAT_FIELDS,
    STATSAPI_PITCHING_STAT_FIELDS,
    STATSAPI_SPLIT_FIELDS,
    Role,
    SavantKind,
    raw_people_path,
    raw_savant_path,
    raw_statsapi_path,
)

CHECK_NAMES: tuple[str, ...] = (
    "F2-1 필수 필드",
    "F2-2 ID 중복",
    "F2-3 타자 ID 일치",
    "F2-4 투수 포함 관계",
    "F2-5 리그 합계",
    "F2-6 선수 정보",
)
_F2_1, _F2_2, _F2_3, _F2_4, _F2_5, _F2_6 = CHECK_NAMES


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
        # 검사가 빠진 보고서가 통과로 읽히지 않도록 6개가 모두 있어야 한다
        names = tuple(c.name for c in self.checks)
        return names == CHECK_NAMES and all(c.passed for c in self.checks)


class ValidationError(Exception):
    """검증을 통과하지 않은 원본으로 다음 단계를 시작하려 할 때."""


# ---------------------------------------------------------------------------
# 원본 읽기 — 파일이 없거나 형식이 틀리면 예외 대신 F2-1 문제로 돌려준다
# ---------------------------------------------------------------------------

_API_STAT_FIELDS: dict[Role, tuple[str, ...]] = {
    "bat": STATSAPI_HITTING_STAT_FIELDS,
    "pit": STATSAPI_PITCHING_STAT_FIELDS,
}
# F2-3~F2-5가 더하거나 비교하는 stat 필드. 정수가 아니면 0으로 셈하지 않고 F2-1에서 실패시킨다
_API_INT_STAT_FIELDS: dict[Role, tuple[str, ...]] = {
    "bat": ("plateAppearances",),
    "pit": ("battersFaced", "outs"),
}
_SAVANT_COLUMNS: dict[SavantKind, tuple[str, ...]] = {
    "batter": SAVANT_BATTER_COLUMNS,
    "pitcher": SAVANT_PITCHER_COLUMNS,
    "oaa": SAVANT_OAA_COLUMNS,
    "frv": SAVANT_FRV_COLUMNS,
}
# 선수 ID 열. FRV만 `id`다(E18)
_SAVANT_ID_COLUMN: dict[SavantKind, str] = {
    "batter": "player_id",
    "pitcher": "player_id",
    "oaa": "player_id",
    "frv": "id",
}
_PEOPLE_DIR = raw_people_path(0).parent
_PEOPLE_GLOB = "people_*.json"  # raw_people_path(batch)의 모든 묶음

# `172.1` 이닝 표기 (정수부.아웃 0~2)
_INNINGS_PATTERN = re.compile(r"(\d+)\.([012])", re.ASCII)

_SHOW_MAX = 5  # detail에 적는 ID·행의 최대 개수
_UNCHECKED = "원본을 읽지 못해 검사하지 못했다 (F2-1 참고)"


@dataclass(frozen=True)
class _Api:
    """읽어 낸 API 시즌 기록 원본."""

    rel: Path
    splits: list[dict[str, Any]]


@dataclass(frozen=True)
class _Savant:
    """읽어 낸 Savant CSV 원본. `ids`는 행 순서대로의 선수 ID다."""

    rel: Path
    frame: pd.DataFrame
    ids: list[int]


def _as_int(value: Any) -> int | None:
    """JSON 정수면 그 값, 아니면 None (bool은 정수로 보지 않는다)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _player_id(split: dict[str, Any]) -> int | None:
    player = split.get("player")
    return _as_int(player.get("id")) if isinstance(player, dict) else None


def _position(split: dict[str, Any]) -> str | None:
    position = split.get("position")
    abbr = position.get("abbreviation") if isinstance(position, dict) else None
    return abbr if isinstance(abbr, str) else None


def _stat(split: dict[str, Any], field: str) -> int | None:
    stat = split.get("stat")
    return _as_int(stat.get(field)) if isinstance(stat, dict) else None


def _shown(items: Sequence[str], sep: str = "; ") -> str:
    """앞의 `_SHOW_MAX`개만 이어 적고, 더 있으면 `…`를 붙인다."""
    more = f"{sep}…" if len(items) > _SHOW_MAX else ""
    return sep.join(items[:_SHOW_MAX]) + more


def _head(ids: Iterable[int]) -> str:
    """ID 앞의 몇 개를 `[1, 2, …]`로 적는다."""
    return f"[{_shown([str(i) for i in sorted(ids)], ', ')}]"


def _savant_kinds(season: int) -> tuple[SavantKind, ...]:
    """시즌에 검사하는 Savant 원본 종류. 수비 시즌이면 OAA·FRV도."""
    basic: tuple[SavantKind, ...] = ("batter", "pitcher")
    fielding: tuple[SavantKind, ...] = ("oaa", "frv")
    return basic + fielding if season in FIELDING_YEARS else basic


def _season_rels(season: int) -> list[Path]:
    """시즌 검사에 쓰는 시즌별 원본(선수 정보 묶음 제외)의 상대 경로."""
    api = [raw_statsapi_path(season, role) for role in ROLES]
    return api + [raw_savant_path(season, kind) for kind in _savant_kinds(season)]


def _outs_from_innings(value: Any) -> int | None:
    """`172.1` 이닝 표기를 아웃 수(172×3+1)로 바꾼다. 표기가 틀리면 None."""
    match = _INNINGS_PATTERN.fullmatch(value) if isinstance(value, str) else None
    return int(match[1]) * 3 + int(match[2]) if match else None


def _field_problems(split: dict[str, Any], role: Role) -> list[str]:
    """split 한 행에서 빠졌거나 쓸 수 없는 필수 필드 (`"stat.age 없음"` 같은 짧은 설명)."""
    found = [f"{f} 없음" for f in STATSAPI_SPLIT_FIELDS if f not in split]
    stat = split.get("stat")
    if isinstance(stat, dict):
        found += [f"stat.{f} 없음" for f in _API_STAT_FIELDS[role] if f not in stat]
        found += [
            f"stat.{f} 정수가 아님"
            for f in _API_INT_STAT_FIELDS[role]
            if f in stat and _as_int(stat[f]) is None
        ]
    elif "stat" in split:
        found.append("stat 객체가 아님")
    # 다른 검사가 쓰는 하위 필드
    if "player" in split and _player_id(split) is None:
        found.append("player.id 없음 또는 정수가 아님")
    if "position" in split and _position(split) is None:
        found.append("position.abbreviation 없음")
    return found


def _innings_problems(where: str, splits: list[dict[str, Any]]) -> list[str]:
    """`inningsPitched` 표기와 `outs`가 어긋난 행 (둘 중 하나가 없는 행은 필드 검사가 잡는다)."""
    bad: list[str] = []
    for split in splits:
        stat = split.get("stat")
        if not isinstance(stat, dict) or "inningsPitched" not in stat or "outs" not in stat:
            continue
        expected = _outs_from_innings(stat["inningsPitched"])
        if expected is None or expected != _as_int(stat["outs"]):
            bad.append(f"{_player_id(split)}: {stat['inningsPitched']} ≠ 아웃 {stat['outs']}")
    return [f"{where}: inningsPitched와 outs 불일치 {len(bad)}행 ({_shown(bad)})"] if bad else []


def _load_api(raw_dir: Path, rel: Path, role: Role) -> tuple[_Api | None, list[str]]:
    """API 원본을 읽고 F2-1 문제를 모은다. 구조가 깨져 쓸 수 없으면 None."""
    where = rel.as_posix()
    if not (raw_dir / rel).is_file():
        return None, [f"{where}: 파일이 없다"]
    try:
        doc = raw.read_json(raw_dir, rel)
    except (OSError, ValueError) as exc:  # JSONDecodeError·UnicodeDecodeError는 ValueError
        return None, [f"{where}: JSON을 읽지 못했다 ({exc})"]
    stats = doc.get("stats") if isinstance(doc, dict) else None
    block = stats[0] if isinstance(stats, list) and stats else None
    if not isinstance(block, dict):
        return None, [f"{where}: stats[0]이 없다"]
    splits = block.get("splits")
    if not isinstance(splits, list) or not all(isinstance(s, dict) for s in splits):
        return None, [f"{where}: splits가 객체 목록이 아니다"]

    problems: list[str] = []
    if not splits:
        problems.append(f"{where}: splits가 비어 있다")
    total = block.get("totalSplits")
    if _as_int(total) != len(splits):
        problems.append(f"{where}: totalSplits {total} ≠ splits {len(splits)}행 (응답 불완전)")
    counts = Counter(label for split in splits for label in _field_problems(split, role))
    problems += [f"{where}: {label} ({n}행)" for label, n in counts.items()]
    if role == "pit":
        problems += _innings_problems(where, splits)
    return _Api(rel, splits), problems


def _int_ids(column: "pd.Series[Any]") -> list[int] | None:
    """ID 열을 정수 목록으로. 비었거나 정수가 아닌 값이 있으면 None."""
    values = pd.to_numeric(column, errors="coerce")
    if bool(values.isna().any()) or bool((values % 1 != 0).any()):
        return None
    return [int(v) for v in values]


def _load_savant(raw_dir: Path, rel: Path, kind: SavantKind) -> tuple[_Savant | None, list[str]]:
    """Savant CSV를 읽고 F2-1 문제를 모은다. 읽지 못했거나 ID 열을 쓸 수 없으면 None."""
    where = rel.as_posix()
    if not (raw_dir / rel).is_file():
        return None, [f"{where}: 파일이 없다"]
    try:
        frame = raw.read_csv(raw_dir, rel)
    except (OSError, ValueError) as exc:  # EmptyDataError·ParserError도 ValueError
        return None, [f"{where}: CSV를 읽지 못했다 ({exc})"]

    problems: list[str] = []
    absent = [c for c in _SAVANT_COLUMNS[kind] if c not in frame.columns]
    if absent:
        problems.append(f"{where}: 열 없음 {absent}")
    if frame.empty:
        problems.append(f"{where}: 데이터 행이 없다")
    id_column = _SAVANT_ID_COLUMN[kind]
    if id_column not in frame.columns:
        return None, problems
    ids = _int_ids(frame[id_column])
    if ids is None:
        problems.append(f"{where}: {id_column}에 정수가 아닌 값이 있다")
        return None, problems
    return _Savant(rel, frame, ids), problems


def _people_files(raw_dir: Path) -> list[Path]:
    """원본 디렉토리의 선수 정보 묶음 (상대 경로, 이름 순)."""
    # 이름 규칙(people_000)에 딱 맞지 않는 파일도 읽고 sha256을 남긴다. 다음 단계가 무엇을 읽든
    # 검증되지 않은 선수 정보가 끼지 않게 하고, 값이 엇갈리면 F2-6이 실패한다
    found = (raw_dir / _PEOPLE_DIR).glob(_PEOPLE_GLOB)
    return sorted(_PEOPLE_DIR / p.name for p in found if p.is_file())


def _load_people(raw_dir: Path, rels: list[Path]) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """모든 묶음의 `people`을 ID → 항목으로 합친다. `id`가 없는 항목은 없는 것으로 본다.

    같은 ID가 여러 번 나오면 값이 모두 같을 때만 하나로 본다. 값이 다르면(지워지지 않은 옛 묶음 등)
    어느 쪽을 쓸지 정할 수 없으므로 문제로 돌려준다.
    """
    people: dict[int, dict[str, Any]] = {}
    conflicts: set[int] = set()
    problems: list[str] = []
    if not rels:
        problems.append(f"선수 정보 원본({(_PEOPLE_DIR / _PEOPLE_GLOB).as_posix()})이 없다")
    for rel in rels:
        try:
            doc = raw.read_json(raw_dir, rel)
        except (OSError, ValueError) as exc:
            problems.append(f"{rel.as_posix()}: JSON을 읽지 못했다 ({exc})")
            continue
        entries = doc.get("people") if isinstance(doc, dict) else None
        if not isinstance(entries, list):
            problems.append(f"{rel.as_posix()}: people 목록이 없다")
            continue
        for entry in entries:
            if not isinstance(entry, dict) or (pid := _as_int(entry.get("id"))) is None:
                continue
            if pid in people and people[pid] != entry:
                conflicts.add(pid)
            people.setdefault(pid, entry)
    if conflicts:
        problems.append(f"묶음마다 값이 다른 선수 정보 {len(conflicts)}명 {_head(conflicts)}")
    return people, problems


# ---------------------------------------------------------------------------
# 검사 F2-1 ~ F2-6
# ---------------------------------------------------------------------------


def _result(name: str, problems: list[str], ok_detail: str) -> CheckResult:
    return CheckResult(name, not problems, "; ".join(problems) if problems else ok_detail)


def _unchecked(name: str) -> CheckResult:
    return CheckResult(name, False, _UNCHECKED)


def _row_summary(api: dict[Role, _Api | None], savant: dict[SavantKind, _Savant | None]) -> str:
    counts = [f"{a.rel.as_posix()} {len(a.splits)}행" for a in api.values() if a is not None]
    counts += [f"{s.rel.as_posix()} {len(s.frame)}행" for s in savant.values() if s is not None]
    return "필수 필드·열 모두 있음 — " + ", ".join(counts)


def _check_duplicates(
    apis: list[_Api], savants: list[_Savant]
) -> tuple[CheckResult, tuple[str, ...]]:
    """F2-2. (결과, 하나로 합친 같은 값 중복 기록)을 돌려준다."""
    problems: list[str] = []
    merged: list[str] = []
    for api in apis:
        counts = Counter(pid for s in api.splits if (pid := _player_id(s)) is not None)
        dups = [pid for pid, n in counts.items() if n > 1]
        if dups:
            problems.append(f"{api.rel.as_posix()}: 선수 ID 중복 {len(dups)}명 {_head(dups)}")
    for sv in savants:
        conflicts: list[int] = []
        for pid, n in sorted(Counter(sv.ids).items()):
            if n < 2:
                continue
            rows = sv.frame.iloc[[i for i, v in enumerate(sv.ids) if v == pid]]
            # drop_duplicates는 빈 칸(NaN)끼리도 같은 값으로 본다 (OAA year, E4)
            if len(rows.drop_duplicates()) == 1:
                merged.append(f"{sv.rel.as_posix()}: {pid} ×{n}")
            else:
                conflicts.append(pid)
        if conflicts:
            problems.append(
                f"{sv.rel.as_posix()}: 값이 다른 ID 중복 {len(conflicts)}명 {_head(conflicts)}"
            )
    ok = "ID 중복 없음" + (f" (같은 값 중복 {len(merged)}건은 합쳤다)" if merged else "")
    return _result(_F2_2, problems, ok), tuple(merged)


def _check_batters(bat: _Api, savant: _Savant) -> CheckResult:
    """F2-3. API PA>0 타자 == Savant 타자."""
    api_ids = {
        pid
        for s in bat.splits
        if (pid := _player_id(s)) is not None and (_stat(s, "plateAppearances") or 0) > 0
    }
    savant_ids = set(savant.ids)
    if api_ids == savant_ids:
        return CheckResult(
            _F2_3, True, f"API PA>0 타자 {len(api_ids)}명 = Savant 타자 {len(savant_ids)}명"
        )
    only_api, only_savant = api_ids - savant_ids, savant_ids - api_ids
    return CheckResult(
        _F2_3,
        False,
        f"API PA>0 타자에만 {len(only_api)}명 {_head(only_api)}, "
        f"Savant 타자에만 {len(only_savant)}명 {_head(only_savant)}",
    )


def _check_pitchers(pit: _Api, savant: _Savant, limits: LeagueLimits) -> CheckResult:
    """F2-4. Savant 투수 ⊆ API 투구, 빠진 실제 투수는 BF 상한 안."""
    savant_ids = set(savant.ids)
    api_ids = {pid for s in pit.splits if (pid := _player_id(s)) is not None}
    league_bf = sum(_stat(s, "battersFaced") or 0 for s in pit.splits)
    # Savant에 없는 실제 투수(포지션 P·TWP). 등판한 야수는 제한이 없다
    missing = {
        pid: _stat(s, "battersFaced") or 0
        for s in pit.splits
        if (pid := _player_id(s)) is not None
        and pid not in savant_ids
        and _position(s) in PITCHER_POSITIONS
    }
    missing_bf = sum(missing.values())

    problems: list[str] = []
    extra = savant_ids - api_ids
    if extra:
        problems.append(f"Savant 투수에만 {len(extra)}명 {_head(extra)}")
    over = {pid: bf for pid, bf in missing.items() if bf > limits.max_missing_pitcher_bf}
    if over:
        shown = _shown([f"{pid}({over[pid]} BF)" for pid in sorted(over)], ", ")
        problems.append(
            f"Savant에 없는 실제 투수 {len(over)}명이 한 명 상한 "
            f"{limits.max_missing_pitcher_bf} BF를 넘는다: {shown}"
        )
    if missing_bf > league_bf * limits.max_missing_pitcher_share:
        problems.append(
            f"Savant에 없는 실제 투수 BF 합 {missing_bf:,} > 리그 BF {league_bf:,} × "
            f"{limits.max_missing_pitcher_share}"
        )
    share = missing_bf / league_bf if league_bf else 0.0
    ok = (
        f"Savant 투수 {len(savant_ids)}명 ⊆ API 투구 {len(api_ids)}명, "
        f"Savant에 없는 실제 투수 {len(missing)}명 BF {missing_bf:,} (리그 BF {league_bf:,}의 {share:.2%})"
    )
    return _result(_F2_4, problems, ok)


def _check_totals(bat: _Api, pit: _Api, season: int, limits: LeagueLimits) -> CheckResult:
    """F2-5. 리그 PA·아웃 합 범위와 PA 합 == BF 합."""
    pa = sum(_stat(s, "plateAppearances") or 0 for s in bat.splits)
    outs = sum(_stat(s, "outs") or 0 for s in pit.splits)
    bf = sum(_stat(s, "battersFaced") or 0 for s in pit.splits)
    short = season == SHORT_SEASON
    ranges = {
        "PA": (pa, limits.pa_short if short else limits.pa_full),
        "아웃": (outs, limits.outs_short if short else limits.outs_full),
    }
    problems = [
        f"리그 {label} 합 {total:,}이 범위 {lo:,}~{hi:,} 밖"
        for label, (total, (lo, hi)) in ranges.items()
        if not lo <= total <= hi
    ]
    if limits.pa_bf_identity and pa != bf:
        problems.append(f"리그 PA 합 {pa:,} ≠ BF 합 {bf:,}")
    relation = "=" if pa == bf else "≠"
    identity = "" if limits.pa_bf_identity else " (PA = BF 검사 끔)"
    return _result(_F2_5, problems, f"PA {pa:,} {relation} BF {bf:,}, 아웃 {outs:,}{identity}")


def _check_people(
    bat: _Api,
    pit: _Api,
    people: dict[int, dict[str, Any]],
    file_problems: list[str],
    n_files: int,
) -> CheckResult:
    """F2-6. 이 시즌 모든 선수의 선수 정보에 `PEOPLE_FIELDS`가 있다. 묶음 파일 문제도 실패다."""
    season_ids = sorted(
        {pid for api in (bat, pit) for s in api.splits if (pid := _player_id(s)) is not None}
    )
    missing = [pid for pid in season_ids if pid not in people]
    lacking = {
        pid: absent
        for pid in season_ids
        if pid in people and (absent := [f for f in PEOPLE_FIELDS if f not in people[pid]])
    }
    problems = list(file_problems)
    if missing:
        problems.append(f"선수 정보 없음 {len(missing)}명 {_head(missing)}")
    if lacking:
        shown = _shown([f"{pid}: {', '.join(fields)}" for pid, fields in lacking.items()])
        problems.append(f"필드가 빠진 선수 정보 {len(lacking)}명 ({shown})")
    return _result(_F2_6, problems, f"선수 {len(season_ids)}명 모두 있음 (묶음 {n_files}개)")


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
    people_rels = _people_files(raw_dir)
    # 읽기 전에 sha256을 남긴다. 없는 파일은 F2-1이 실패로 적는다
    inputs = {
        rel.as_posix(): raw.file_sha256(raw_dir / rel)
        for rel in [*_season_rels(season), *people_rels]
        if (raw_dir / rel).is_file()
    }

    field_problems: list[str] = []
    api: dict[Role, _Api | None] = {}
    for role in ROLES:
        api[role], problems = _load_api(raw_dir, raw_statsapi_path(season, role), role)
        field_problems += problems
    savant: dict[SavantKind, _Savant | None] = {}
    for kind in _savant_kinds(season):
        savant[kind], problems = _load_savant(raw_dir, raw_savant_path(season, kind), kind)
        field_problems += problems
    people, people_problems = _load_people(raw_dir, people_rels)

    # 읽지 못한 원본이 필요한 검사는 실패로 둔다 (사유는 F2-1에 있다)
    bat, pit = api["bat"], api["pit"]
    sv_bat, sv_pit = savant["batter"], savant["pitcher"]
    apis = [a for a in api.values() if a is not None]
    savants = [s for s in savant.values() if s is not None]
    merged: tuple[str, ...] = ()
    f2_2, f2_3, f2_4, f2_5 = (_unchecked(name) for name in CHECK_NAMES[1:5])
    # 선수 정보 묶음 자체의 문제는 F2-1에 없으므로 검사하지 못했어도 남긴다
    f2_6 = CheckResult(_F2_6, False, "; ".join([_UNCHECKED, *people_problems]))
    if len(apis) == len(api) and len(savants) == len(savant):
        f2_2, merged = _check_duplicates(apis, savants)
    if bat is not None and sv_bat is not None:
        f2_3 = _check_batters(bat, sv_bat)
    if pit is not None and sv_pit is not None:
        f2_4 = _check_pitchers(pit, sv_pit, limits)
    if bat is not None and pit is not None:
        f2_5 = _check_totals(bat, pit, season, limits)
        f2_6 = _check_people(bat, pit, people, people_problems, len(people_rels))
    f2_1 = _result(_F2_1, field_problems, _row_summary(api, savant))
    checks = (f2_1, f2_2, f2_3, f2_4, f2_5, f2_6)
    return SeasonReport(season, checks, merged, inputs)


def validate_seasons(
    raw_dir: Path, seasons: Iterable[int], *, limits: LeagueLimits = DEFAULT_LIMITS
) -> list[SeasonReport]:
    """여러 시즌을 시즌 오름차순으로 검사한다."""
    return [validate_season(raw_dir, season, limits=limits) for season in sorted(set(seasons))]


# ---------------------------------------------------------------------------
# 보고서 파일
# ---------------------------------------------------------------------------


def _check_unique_seasons(seasons: Iterable[int]) -> None:
    """같은 시즌이 두 번 있으면 ValueError (어느 쪽이 맞는지 정할 수 없다)."""
    dups = sorted(season for season, n in Counter(seasons).items() if n > 1)
    if dups:
        raise ValueError(f"검증 보고서에 시즌이 중복된다: {', '.join(map(str, dups))}")


def write_report(reports: Iterable[SeasonReport], path: Path) -> None:
    """보고서를 JSON으로 쓴다 (시즌 오름차순, 키 정렬, UTF-8, 같은 입력이면 같은 바이트)."""
    ordered = sorted(reports, key=lambda r: r.season)
    _check_unique_seasons(r.season for r in ordered)
    # `passed`는 사람이 읽기 위한 값이다. 읽을 때는 checks로 다시 계산한다
    doc = [{**asdict(r), "passed": r.passed} for r in ordered]
    text = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))  # 운영체제 줄바꿈 변환 없이


def _is_str_list(value: Any) -> TypeGuard[list[str]]:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _check_from_doc(doc: Any) -> CheckResult:
    """보고서 JSON의 검사 하나. 형식이 틀리면 ValueError."""
    if not isinstance(doc, dict):
        raise ValueError(f"검사 결과가 객체가 아니다: {doc!r}")
    name, passed, detail = doc.get("name"), doc.get("passed"), doc.get("detail")
    if not (isinstance(name, str) and isinstance(passed, bool) and isinstance(detail, str)):
        raise ValueError(f"검사 결과 형식이 틀렸다: {doc!r}")
    return CheckResult(name, passed, detail)


def _report_from_doc(doc: Any) -> SeasonReport:
    """보고서 JSON의 시즌 하나. 형식이 틀리면 ValueError (게이트가 엉터리 값을 통과로 읽지 않게)."""
    if not isinstance(doc, dict):
        raise ValueError(f"시즌 보고서가 객체가 아니다: {doc!r}")
    checks, merged, inputs = doc.get("checks"), doc.get("merged_duplicates"), doc.get("inputs")
    season = _as_int(doc.get("season"))
    if season is None:
        raise ValueError(f"season이 정수가 아니다: {doc.get('season')!r}")
    if not isinstance(checks, list):
        raise ValueError(f"{season} checks가 목록이 아니다")
    if not _is_str_list(merged):
        raise ValueError(f"{season} merged_duplicates가 문자열 목록이 아니다")
    if not (isinstance(inputs, dict) and _is_str_list([*inputs.values()])):
        raise ValueError(f"{season} inputs가 경로 → sha256 사전이 아니다")
    return SeasonReport(
        season=season,
        checks=tuple(_check_from_doc(c) for c in checks),
        merged_duplicates=tuple(merged),
        inputs={str(rel): sha for rel, sha in inputs.items()},
    )


def read_report(path: Path) -> list[SeasonReport]:
    """`write_report`가 쓴 JSON을 읽는다 (왕복하면 같은 보고서)."""
    # 파일을 읽지 못하면 OSError, JSON이 아니거나 형식이 틀렸거나 시즌이 중복되면 ValueError
    doc = json.loads(path.read_bytes().decode("utf-8"))
    if not isinstance(doc, list):
        raise ValueError(f"검증 보고서는 시즌 목록이어야 한다: {path}")
    reports = sorted((_report_from_doc(item) for item in doc), key=lambda r: r.season)
    _check_unique_seasons(r.season for r in reports)
    return reports


def _current_sha256(raw_dir: Path, rel: str) -> str | None:
    path = raw_dir / rel
    return raw.file_sha256(path) if path.is_file() else None


def assert_validated(raw_dir: Path, report_path: Path, seasons: Iterable[int]) -> None:
    """다음 단계 시작 전 확인: 보고서가 있고, 요청 시즌이 모두 통과했고, 원본이 그 뒤로 바뀌지 않았다.

    Raises:
        ValidationError: 보고서 없음 / 시즌 누락 / 실패한 시즌 / 원본 sha256 불일치. 메시지에 사유를 적는다.
    """
    # 읽지 못하는 보고서, sha256이 기록되지 않은 원본(검증 뒤 생긴 선수 정보 묶음 등)도 막는다
    wanted = sorted(set(seasons))
    if not report_path.is_file():
        raise ValidationError(f"검증 보고서가 없다: {report_path} — 수집 검증(F2)을 먼저 실행한다")
    try:
        reports = {r.season: r for r in read_report(report_path)}
    except (OSError, ValueError) as exc:  # JSONDecodeError·UnicodeDecodeError도 ValueError
        raise ValidationError(f"검증 보고서를 읽지 못했다: {report_path} ({exc})") from exc

    reasons: list[str] = []
    absent = [s for s in wanted if s not in reports]
    if absent:
        reasons.append(f"보고서에 없는 시즌: {', '.join(map(str, absent))}")
    failed = [reports[s] for s in wanted if s in reports and not reports[s].passed]
    if failed:
        listed = ", ".join(
            f"{r.season}({', '.join(c.name for c in r.checks if not c.passed) or '검사 누락'})"
            for r in failed
        )
        reasons.append(f"검증을 통과하지 못한 시즌: {listed}")
    checked = [reports[s] for s in wanted if s in reports]
    # 시즌 원본과 지금 있는 선수 정보 묶음이 모두 sha256으로 기록돼 있어야 대조할 수 있다.
    # 검증 뒤에 새로 생긴 묶음도 다음 단계의 입력이 바뀐 것이다
    people_now = [rel.as_posix() for rel in _people_files(raw_dir)]
    unrecorded = sorted(
        {
            rel
            for r in checked
            for rel in [*(p.as_posix() for p in _season_rels(r.season)), *people_now]
            if rel not in r.inputs
        }
    )
    if unrecorded:
        reasons.append(
            f"보고서에 sha256이 없는 원본(검증 뒤 생겼거나 검증하지 않음): {', '.join(unrecorded)}"
        )
    # 여러 시즌이 함께 쓰는 선수 정보 묶음은 시즌마다 기록된 sha256을 모두 대조한다
    expected = {(rel, sha) for r in checked for rel, sha in r.inputs.items()}
    current = {rel: _current_sha256(raw_dir, rel) for rel, _ in expected}
    changed = sorted({rel for rel, sha in expected if current[rel] != sha})
    if changed:
        reasons.append(f"검증 뒤 바뀌거나 사라진 원본: {', '.join(changed)}")
    if reasons:
        raise ValidationError("; ".join(reasons))
