"""파이프라인 모듈과 웹이 함께 쓰는 계약 (PLAN §2-2).

열 이름·단위·경로·요청 URL을 한곳에 고정한다. 메인 세션만 고치고, 구현 에이전트는 읽기만 한다.

단위 규칙
- `*_pct` 열은 백분율(0~100)이다. 예: K% 22.5, K−BB% 12.3
- `obp`·`slg`·`ops`·`woba`·`xwoba`는 소수(0.350)다.
- 투수 출전량은 정제 표에서 아웃 수(`outs`), 예측·웹에서 소수 이닝(`ip` = 아웃 / 3)이다.
  화면의 `172.1` 표기는 웹에서 바꾼다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd
from pandas.api import types as ptypes

# ---------------------------------------------------------------------------
# 시즌·대상 (SPEC D3·D9·D10·D13·D21)
# ---------------------------------------------------------------------------

LEAGUE: Final = "MLB"
SEASON_FIRST: Final = 2019
SEASON_LAST: Final = 2026
SEASONS: Final[tuple[int, ...]] = tuple(range(SEASON_FIRST, SEASON_LAST + 1))
FIELDING_FIRST: Final = 2022
FIELDING_YEARS: Final[tuple[int, ...]] = tuple(range(FIELDING_FIRST, SEASON_LAST + 1))

TARGET_SEASON: Final = 2027
TUNE_TARGETS: Final[tuple[int, ...]] = (2024, 2025)
FINAL_TEST_TARGET: Final = 2026
WINDOW: Final = 5  # 목표 Y의 입력 = Y−5 ~ Y−1 (F4·F5)

SHORT_SEASON: Final = 2020
SHORT_SEASON_SCALE: Final = 162 / 60  # 출전량 입력만 환산 (D9)

RATE_MIN_PA: Final = 200  # 비율 지표 대표 집단 (D13)
RATE_MIN_OUTS: Final = 150  # 50 IP

Role = Literal["bat", "pit"]
ROLES: Final[tuple[Role, ...]] = ("bat", "pit")

# 역할 판정 (D8, 2026-11-24 실측으로 확정)
# - 타자: API 타격 기록 중 포지션이 BATTER_EXCLUDED_POSITIONS가 아니고 타석이 1 이상인 행.
#   포지션 X(시즌당 13~59행)·DH도 타자다. 투수의 타격(포지션 P)은 뺀다. 타석 0인 행(대주자 등)은 기록으로 치지 않는다.
# - 투수: API 투구 기록 중 포지션이 PITCHER_POSITIONS인 행. 등판한 야수는 뺀다. 오타니의 투구 포지션은 TWP다.
PITCHER_POSITIONS: Final = frozenset({"P", "TWP"})
BATTER_EXCLUDED_POSITIONS: Final = frozenset({"P"})

ModelName = Literal["league", "baseline", "weighted", "extended"]
MODELS: Final[tuple[ModelName, ...]] = ("league", "baseline", "weighted", "extended")

BAT_METRICS: Final[tuple[str, ...]] = ("ops", "pa", "hr")
PIT_METRICS: Final[tuple[str, ...]] = ("era", "ip", "k_bb_pct")
RATE_METRICS: Final = frozenset({"ops", "era", "k_bb_pct"})

# 평가 그룹 (D13, PLAN R8)
GROUPS: Final[tuple[str, ...]] = (
    "all",
    "seasons_1_2",
    "seasons_3_5",
    "prev_played",
    "prev_absent",
)

# ---------------------------------------------------------------------------
# 저장 위치 (SPEC D6, PLAN §2-1) — 저장소 루트 기준 상대 경로
# ---------------------------------------------------------------------------

RAW_DIR: Final = Path("data/raw")
INTERIM_DIR: Final = Path("data/interim")
OUT_DIR: Final = Path("data/out")
WEB_DATA_DIR: Final = Path("web/public/data")
PARAMS_PATH: Final = Path("config/params_v1.json")
WEB_SCHEMA_PATH: Final = Path("contracts/web-data.schema.json")
# F2 검증 결과. M2 이후 단계는 이 보고서가 통과·최신인지 확인한 뒤에만 진행한다
VALIDATION_REPORT_PATH: Final = INTERIM_DIR / "validation_report.json"
MANIFEST_NAME: Final = "manifest.json"  # 원본 디렉토리 안. 파일별 URL·받은 시각·sha256 (F1·N1)

SavantKind = Literal["batter", "pitcher", "oaa", "frv"]

PEOPLE_BATCH_SIZE: Final = 100
MIN_INTERVAL_SECONDS: Final = 1.0  # 요청 사이 최소 간격 (N3)

# 아래 원본 경로는 모두 **원본 디렉토리 기준 상대 경로**다. 실제 위치는 `RAW_DIR / 경로`,
# 테스트에서는 `tests/fixtures/raw / 경로`다. 매니페스트 키도 이 상대 경로(posix)다.


def raw_statsapi_path(season: int, role: Role) -> Path:
    """API 시즌 기록 원본의 상대 경로를 돌려준다."""
    return Path("statsapi") / f"{season}_{role}.json"


def raw_people_path(batch: int) -> Path:
    """API 선수 정보 원본의 상대 경로를 돌려준다. batch는 0부터 센다."""
    return Path("statsapi") / f"people_{batch:03d}.json"


def raw_savant_path(season: int, kind: SavantKind) -> Path:
    """Savant CSV 원본의 상대 경로를 돌려준다."""
    return Path("savant") / f"{season}_{kind}.csv"


# ---------------------------------------------------------------------------
# 요청 URL (SPEC E1·E2·E4·E18) — 연도마다 같은 열을 받는다 (F1)
# ---------------------------------------------------------------------------

STATSAPI_BASE: Final = "https://statsapi.mlb.com/api/v1"
SAVANT_BASE: Final = "https://baseballsavant.mlb.com"
USER_AGENT: Final = "mlbproj/0.1 (+non-commercial fan projection)"  # 개인정보 없음 (N3)

SAVANT_BATTER_SELECTIONS: Final[tuple[str, ...]] = (
    "player_age",
    "pa",
    "home_run",
    "k_percent",
    "bb_percent",
    "woba",
    "xwoba",
    "exit_velocity_avg",
    "launch_angle_avg",
    "hard_hit_percent",
    "barrel_batted_rate",
)
# p_gs는 받아도 비어 있다(E3). GS는 API에서 받는다.
SAVANT_PITCHER_SELECTIONS: Final[tuple[str, ...]] = (
    "player_age",
    "pa",
    "p_formatted_ip",
    "k_percent",
    "bb_percent",
    "woba",
    "xwoba",
    "whiff_percent",
    "hard_hit_percent",
    "barrel_batted_rate",
    "xera",
)


def statsapi_stats_url(season: int, role: Role) -> str:
    """API 시즌 기록 URL. 선수당 1행, 이적 선수는 합산 1행이다(E1)."""
    group = "hitting" if role == "bat" else "pitching"
    return (
        f"{STATSAPI_BASE}/stats?stats=season&group={group}&season={season}"
        "&sportId=1&playerPool=ALL&limit=3000"
    )


def statsapi_people_url(player_ids: Sequence[int]) -> str:
    """API 선수 정보 URL. ID는 PEOPLE_BATCH_SIZE개 이하로 나눠 부른다."""
    if not player_ids:
        raise ValueError("player_ids가 비어 있다")
    if len(player_ids) > PEOPLE_BATCH_SIZE:
        raise ValueError(f"한 번에 {PEOPLE_BATCH_SIZE}명까지 부른다")
    return f"{STATSAPI_BASE}/people?personIds={','.join(str(i) for i in player_ids)}"


def savant_leaderboard_url(season: int, role: Role) -> str:
    """Savant Custom Leaderboard CSV URL (`min=1`, E2)."""
    kind = "batter" if role == "bat" else "pitcher"
    selections = SAVANT_BATTER_SELECTIONS if role == "bat" else SAVANT_PITCHER_SELECTIONS
    return (
        f"{SAVANT_BASE}/leaderboard/custom?year={season}&type={kind}&min=1"
        f"&selections={','.join(selections)}&csv=true"
    )


def savant_oaa_url(season: int) -> str:
    """Savant Outs Above Average CSV URL. 받은 CSV의 `year` 열은 비어 있다(E4)."""
    return (
        f"{SAVANT_BASE}/leaderboard/outs_above_average?type=Fielder&startYear={season}"
        f"&endYear={season}&split=no&team=&range=year&min=1&pos=&roles=&viz=hide&csv=true"
    )


def savant_frv_url(season: int) -> str:
    """Savant Fielding Run Value CSV URL (E18)."""
    return (
        f"{SAVANT_BASE}/leaderboard/fielding-run-value?gameType=Regular"
        f"&seasonStart={season}&seasonEnd={season}&type=fielder&position="
        "&minInnings=0&minResults=1&csv=true"
    )


# ---------------------------------------------------------------------------
# 원본 필수 필드 (F2 ① 검사 대상)
# ---------------------------------------------------------------------------

# API split의 최상위 필드. player.id·position.abbreviation을 쓴다.
STATSAPI_SPLIT_FIELDS: Final[tuple[str, ...]] = (
    "season",
    "player",
    "position",
    "team",
    "numTeams",
    "stat",
)
STATSAPI_HITTING_STAT_FIELDS: Final[tuple[str, ...]] = (
    "age",
    "plateAppearances",
    "atBats",
    "hits",
    "doubles",
    "triples",
    "homeRuns",
    "baseOnBalls",
    "intentionalWalks",
    "hitByPitch",
    "sacFlies",
    "sacBunts",
    "strikeOuts",
    "obp",
    "slg",
    "ops",
)
# `outs`가 직접 있다. `inningsPitched`(172.1 표기)와 서로 맞는지 검사한다.
STATSAPI_PITCHING_STAT_FIELDS: Final[tuple[str, ...]] = (
    "age",
    "gamesPitched",
    "gamesStarted",
    "inningsPitched",
    "outs",
    "battersFaced",
    "earnedRuns",
    "hits",
    "homeRuns",
    "baseOnBalls",
    "hitByPitch",
    "strikeOuts",
    "saves",
)
PEOPLE_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "fullName",
    "birthDate",
    "primaryPosition",
    "batSide",
    "pitchHand",
    "active",
)

SAVANT_ID_COLUMNS: Final[tuple[str, ...]] = ("last_name, first_name", "player_id", "year")
SAVANT_BATTER_COLUMNS: Final = SAVANT_ID_COLUMNS + SAVANT_BATTER_SELECTIONS
SAVANT_PITCHER_COLUMNS: Final = SAVANT_ID_COLUMNS + SAVANT_PITCHER_SELECTIONS
SAVANT_OAA_COLUMNS: Final[tuple[str, ...]] = (
    "player_id",
    "year",
    "primary_pos_formatted",
    "outs_above_average",
)
FRV_OUT_POSITIONS: Final[tuple[int, ...]] = tuple(range(2, 10))  # 2=C … 9=RF
SAVANT_FRV_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "name",
    "total_runs",
    "range_runs",
    "arm_runs",
    "dp_runs",
    "framing_runs",
    "blocking_runs",
    "throwing_runs",
    *(f"outs_{p}" for p in FRV_OUT_POSITIONS),
)
POSITION_ABBR: Final[dict[int, str]] = {
    2: "C",
    3: "1B",
    4: "2B",
    5: "3B",
    6: "SS",
    7: "LF",
    8: "CF",
    9: "RF",
}

# ---------------------------------------------------------------------------
# 정제·산출 표 (SPEC §6)
# ---------------------------------------------------------------------------

Kind = Literal["int", "float", "str", "bool"]


@dataclass(frozen=True)
class Column:
    """표의 열 하나.

    Attributes:
        name: 열 이름.
        kind: 값 종류. 결측이 있는 정수 열은 pandas `Int64`를 쓴다.
        nullable: 결측을 허용하는지.
    """

    name: str
    kind: Kind
    nullable: bool = False


@dataclass(frozen=True)
class TableSpec:
    """표 정의. 열 순서도 계약이다.

    Attributes:
        name: 표 이름 (`data/interim/{name}.csv`).
        keys: 행을 하나로 정하는 열들.
        columns: 열 정의 (순서 고정).
    """

    name: str
    keys: tuple[str, ...]
    columns: tuple[Column, ...]

    @property
    def column_names(self) -> list[str]:
        """열 이름 목록."""
        return [c.name for c in self.columns]


def _cols(kind: Kind, names: Iterable[str], nullable: bool = False) -> tuple[Column, ...]:
    return tuple(Column(n, kind, nullable) for n in names)


_SEASON_KEYS = (Column("player_id", "int"), Column("season", "int"), Column("league", "str"))

PLAYERS = TableSpec(
    name="players",
    keys=("player_id",),
    columns=(
        Column("player_id", "int"),
        Column("name", "str"),
        Column("birth_date", "str", nullable=True),  # ISO 날짜
        Column("primary_position", "str", nullable=True),
        Column("bats", "str", nullable=True),
        Column("throws", "str", nullable=True),
        # 아래 두 열은 2026까지 모든 시즌을 본 값이다 — 화면 표시용(D23)이고, 과거 목표(2024~2026)의
        # 모델·평가 입력으로 쓰면 미래 정보가 샌다(N2). 모델은 cohort·정제 시즌 표만 쓴다.
        Column("active", "bool"),  # 표시용 (D23, E16)
        Column("last_mlb_season", "int"),  # 역할 무관 마지막 MLB 기록 시즌
    ),
)

BATTING_SEASONS = TableSpec(
    name="batting_seasons",
    keys=("player_id", "season", "league"),
    columns=(
        *_SEASON_KEYS,
        Column("age", "int", nullable=True),
        Column("team_name", "str", nullable=True),  # 이적 시 API가 준 팀 그대로
        Column("n_teams", "int"),
        *_cols("int", ("pa", "ab", "h", "b2", "b3", "hr", "bb", "ibb", "hbp", "sf", "sh", "so")),
        *_cols("float", ("obp", "slg", "ops"), nullable=True),
        Column("pa_adj", "float"),  # 2020은 ×162/60 (D9)
        Column("is_short_season", "bool"),
        *_cols(
            "float",
            ("woba", "xwoba", "ev_avg", "la_avg", "hard_hit_pct", "barrel_pct"),
            nullable=True,
        ),
    ),
)

PITCHING_SEASONS = TableSpec(
    name="pitching_seasons",
    keys=("player_id", "season", "league"),
    columns=(
        *_SEASON_KEYS,
        Column("age", "int", nullable=True),
        Column("team_name", "str", nullable=True),
        Column("n_teams", "int"),
        *_cols("int", ("g", "gs", "outs", "bf", "er", "h", "hr", "bb", "hbp", "so", "sv")),
        Column("outs_adj", "float"),  # 2020은 ×162/60 (D9)
        Column("is_short_season", "bool"),
        *_cols(
            "float",
            ("xwoba_against", "whiff_pct", "hard_hit_pct", "barrel_pct", "xera"),
            nullable=True,
        ),
    ),
)

FIELDING_SEASONS = TableSpec(
    name="fielding_seasons",
    keys=("player_id", "season", "league"),
    columns=(
        *_SEASON_KEYS,
        Column("oaa", "float", nullable=True),
        *_cols(
            "float",
            (
                "frv_total",
                "frv_range",
                "frv_arm",
                "frv_dp",
                "frv_framing",
                "frv_blocking",
                "frv_throwing",
            ),
            nullable=True,
        ),
        *_cols("int", tuple(f"outs_{p}" for p in FRV_OUT_POSITIONS)),
    ),
)

_PROJECTION_KEYS = (
    Column("player_id", "int"),
    Column("role", "str"),
    Column("target_season", "int"),
    Column("model", "str"),
    Column("input_first", "int"),
    Column("input_last", "int"),
    Column("n_seasons", "int"),
)

BAT_PROJECTIONS = TableSpec(
    name="projections_bat",
    keys=("player_id", "role", "target_season", "model"),
    columns=(
        *_PROJECTION_KEYS,
        Column("sample_pa", "int"),
        *_cols("float", ("pa", "hr", "obp", "slg", "ops", "lg_ops")),
    ),
)

PIT_PROJECTIONS = TableSpec(
    name="projections_pit",
    keys=("player_id", "role", "target_season", "model"),
    columns=(
        *_PROJECTION_KEYS,
        Column("sample_bf", "int"),
        Column("sample_outs", "int"),
        *_cols("float", ("ip", "era", "k_bb_pct", "lg_era", "lg_k_bb_pct")),
    ),
)

# ---------------------------------------------------------------------------
# 모델 사이의 표 (M3, SPEC F6~F8·D10~D12)
# ---------------------------------------------------------------------------

# 타자 성분 (F7): 타석당 비율로 가중·회귀한다. h1 = h − b2 − b3 − hr (단타).
BAT_COMPONENTS: Final[tuple[str, ...]] = ("h1", "b2", "b3", "hr", "bb", "hbp", "sf", "ab")
# 투수 성분 (F7): 상대 타자당 비율. 자책점은 아웃당(er_per_out)으로 따로 둔다.
PIT_COMPONENTS: Final[tuple[str, ...]] = ("so", "bb", "hbp", "hr")
# 회귀 표본 키 (params의 regress): 타자 성분 + 투수 성분 + "er"
ERA_METHODS: Final[tuple[str, ...]] = (
    "er",
    "fip",
)  # D11 후보 (a) 가중 ER/아웃 (b) FIP식 + 리그 보정. xERA는 M4


def _rates(prefix: str, names: Iterable[str]) -> tuple[Column, ...]:
    return tuple(Column(f"{prefix}{n}", "float") for n in names)


# 시즌별 리그 평균 (역할 표 전체 합으로 계산). 비율 열은 r_<성분> = 성분 합 / PA(또는 BF) 합.
LEAGUE_BAT = TableSpec(
    name="league_bat",
    keys=("season",),
    columns=(
        Column("season", "int"),
        Column("pa", "int"),
        *_rates("r_", BAT_COMPONENTS),
        *_cols("float", ("obp", "slg", "ops", "hr_per_pa")),
    ),
)
LEAGUE_PIT = TableSpec(
    name="league_pit",
    keys=("season",),
    columns=(
        Column("season", "int"),
        Column("bf", "int"),
        Column("outs", "int"),
        *_rates("r_", PIT_COMPONENTS),
        # era = 27 × er_per_out, k_bb_pct = 100 × (r_so − r_bb)
        # fip_const = era − (13·HR + 3·(BB+HBP) − 2·K) / IP  (리그 합 기준, IP = outs / 3)
        *_cols("float", ("er_per_out", "era", "k_bb_pct", "fip_const", "bf_per_out")),
    ),
)

# 모델의 비율 예측 (목표 시즌 하나, 역할 하나, 모델 하나). 결측 없음 — 기록이 없으면 리그 평균.
BAT_RATES = TableSpec(
    name="rates_bat",
    keys=("player_id",),
    columns=(
        Column("player_id", "int"),
        Column("target_season", "int"),
        *_rates("r_", BAT_COMPONENTS),
        *_cols("float", ("obp", "slg", "ops", "hr_per_pa", "lg_ops")),
    ),
)
PIT_RATES = TableSpec(
    name="rates_pit",
    keys=("player_id",),
    columns=(
        Column("player_id", "int"),
        Column("target_season", "int"),
        *_rates("r_", PIT_COMPONENTS),
        *_cols("float", ("er_per_out", "era", "k_bb_pct", "lg_era", "lg_k_bb_pct")),
    ),
)

# 모델의 출전량 예측. 타자 pa, 투수 outs (0 이상 실수).
BAT_PLAYING_TIME = TableSpec(
    name="playing_time_bat",
    keys=("player_id",),
    columns=(Column("player_id", "int"), Column("target_season", "int"), Column("pa", "float")),
)
PIT_PLAYING_TIME = TableSpec(
    name="playing_time_pit",
    keys=("player_id",),
    columns=(Column("player_id", "int"), Column("target_season", "int"), Column("outs", "float")),
)

# 예측 대상과 사용 기간 (F4·F5). 목표 시즌 Y, 역할 r마다 한 표.
COHORTS = TableSpec(
    name="cohorts",
    keys=("player_id", "role", "target_season"),
    columns=(
        Column("player_id", "int"),
        Column("role", "str"),
        Column("target_season", "int"),
        Column("window_first", "int"),  # max(Y−5, 그 역할의 첫 기록 시즌)
        Column("window_last", "int"),  # Y−1
        Column("n_window", "int"),  # window_last − window_first + 1 (미출전 시즌 포함)
        Column("input_first", "int"),  # 기간 안 첫 기록 시즌
        Column("input_last", "int"),  # 기간 안 마지막 기록 시즌
        Column("n_seasons", "int"),  # 기간 안 기록이 있는 시즌 수 (1~5, D13 그룹 기준)
        Column("played_prev", "bool"),  # Y−1에 기록이 있는지 (PLAN R8 그룹 기준)
        Column("sample", "int"),  # 기간 누적 타석(타자) 또는 상대 타자(투수)
        Column("sample_outs", "int", nullable=True),  # 투수만: 기간 누적 아웃
    ),
)

EVALUATIONS = TableSpec(
    name="evaluations",
    keys=("target_season", "model", "metric", "group"),
    columns=(
        Column("target_season", "int"),
        Column("role", "str"),
        Column("model", "str"),
        Column("metric", "str"),
        Column("group", "str"),
        Column("n", "int"),
        Column("mae", "float", nullable=True),
        Column("mae_weighted", "float", nullable=True),  # 비율 지표만
        Column("baseline_mae_league", "float", nullable=True),
        Column("baseline_mae_prev", "float", nullable=True),
        Column("excluded", "int"),
    ),
)

TABLES: Final[tuple[TableSpec, ...]] = (
    PLAYERS,
    BATTING_SEASONS,
    PITCHING_SEASONS,
    FIELDING_SEASONS,
    COHORTS,
    LEAGUE_BAT,
    LEAGUE_PIT,
    BAT_RATES,
    PIT_RATES,
    BAT_PLAYING_TIME,
    PIT_PLAYING_TIME,
    BAT_PROJECTIONS,
    PIT_PROJECTIONS,
    EVALUATIONS,
)

# ---------------------------------------------------------------------------
# 웹 JSON (contracts/web-data.schema.json의 $defs 이름과 짝)
# ---------------------------------------------------------------------------

WEB_SCHEMA_VERSION: Final = "1"
WEB_FILES: Final[dict[str, str]] = {
    "batters.json": "BatTable",
    "pitchers.json": "PitTable",
    "validation.json": "ValidationReport",
}
WEB_PLAYER_DEF: Final = "PlayerDetail"  # players/{player_id}.json
WEB_FIRST_SCREEN_MAX_BYTES: Final = 1_000_000  # N6


def interim_path(spec: TableSpec, base: Path = INTERIM_DIR) -> Path:
    """정제 표 CSV 경로 (`data/interim/{name}.csv`)."""
    return base / f"{spec.name}.csv"


def web_player_path(player_id: int) -> Path:
    """선수 상세 JSON 경로를 돌려준다."""
    return WEB_DATA_DIR / "players" / f"{player_id}.json"


# ---------------------------------------------------------------------------
# 표 검사
# ---------------------------------------------------------------------------


def _is_str_dtype(dtype: Any) -> bool:
    return bool(ptypes.is_string_dtype(dtype) or ptypes.is_object_dtype(dtype))


_KIND_CHECKS: dict[str, Callable[[Any], bool]] = {
    "int": ptypes.is_integer_dtype,
    "float": ptypes.is_float_dtype,
    "bool": ptypes.is_bool_dtype,
    "str": _is_str_dtype,
}


def check_frame(df: pd.DataFrame, spec: TableSpec) -> list[str]:
    """표가 계약을 지키는지 검사한다.

    Args:
        df: 검사할 표.
        spec: 계약.

    Returns:
        위반 사항 목록. 비어 있으면 통과다.
    """
    problems: list[str] = []
    if list(df.columns) != spec.column_names:
        missing = [c for c in spec.column_names if c not in df.columns]
        extra = [c for c in df.columns if c not in spec.column_names]
        problems.append(f"{spec.name}: 열 불일치 (없음 {missing}, 추가 {extra}, 순서 포함)")
        return problems
    for col in spec.columns:
        series = df[col.name]
        if not _KIND_CHECKS[col.kind](series.dtype):
            problems.append(f"{spec.name}.{col.name}: {col.kind}가 아니다 ({series.dtype})")
        if not col.nullable and bool(series.isna().any()):
            problems.append(f"{spec.name}.{col.name}: 결측 허용 안 됨")
    if bool(df.duplicated(list(spec.keys)).any()):
        problems.append(f"{spec.name}: 키 {spec.keys} 중복")
    return problems
