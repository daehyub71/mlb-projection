"""수비 시즌 표 (SPEC D21·E4·E18, §6 `fielding_seasons`). 레인 C가 구현한다.

과거 표시 전용이다 — 예측에 쓰지 않고, FRV와 OAA를 더하지 않는다(D21).
원본은 `collect.savant.load_savant`로만 읽는다.
"""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from mlbproj.collect.savant import load_savant
from mlbproj.contracts import (
    FIELDING_SEASONS,
    FIELDING_YEARS,
    FRV_OUT_POSITIONS,
    LEAGUE,
    SavantKind,
    raw_savant_path,
)

# Savant FRV CSV 열 → 정제 열
FRV_COLUMNS: Mapping[str, str] = {
    "total_runs": "frv_total",
    "range_runs": "frv_range",
    "arm_runs": "frv_arm",
    "dp_runs": "frv_dp",
    "framing_runs": "frv_framing",
    "blocking_runs": "frv_blocking",
    "throwing_runs": "frv_throwing",
}

# 포지션별 아웃 수 열 (원본·정제 이름이 같다)
_OUTS_COLUMNS: tuple[str, ...] = tuple(f"outs_{p}" for p in FRV_OUT_POSITIONS)
# Savant OAA CSV 열 → 정제 열 (fielding_runs_prevented가 아니다)
_OAA_COLUMNS: Mapping[str, str] = {"outs_above_average": "oaa"}
# 오류 메시지에 적는 값의 최대 개수
_SHOW_MAX = 5


def _load_unique(raw_dir: Path, season: int, kind: SavantKind) -> pd.DataFrame:
    """원본을 읽고 같은 선수의 중복 행을 정리한다.

    원본의 **모든 열** 값이 같은 중복(결측끼리도 같다고 본다)은 하나로 합치고, 한 열이라도 다르면
    ValueError (E19). 쓰지 않는 열(이름·팀)까지 비교한다 — validate F2-2와 같은 규칙이다.
    """
    df = load_savant(raw_dir, season, kind).drop_duplicates()
    conflicts = df["player_id"].duplicated()
    if bool(conflicts.any()):
        ids = sorted({int(i) for i in df.loc[conflicts, "player_id"]})
        name = raw_savant_path(season, kind).as_posix()
        raise ValueError(f"{name}: 값이 다른 같은 선수 행이 있다: {ids[:_SHOW_MAX]}")
    return df


def _floats(series: pd.Series, where: str) -> pd.Series:
    """실수로 바꾼다. 빈 칸은 결측으로 두고, 숫자가 아닌 글자는 결측으로 숨기지 않고 ValueError."""
    values = pd.to_numeric(series, errors="coerce")
    bad = values.isna() & series.notna()
    if bool(bad.any()):
        found = sorted({str(v) for v in series[bad]})[:_SHOW_MAX]
        raise ValueError(f"{where}: {series.name} 열에 숫자가 아닌 값이 있다: {found}")
    return values.astype("float64")


def _outs(series: pd.Series, where: str) -> pd.Series:
    """아웃 수를 정수로 바꾼다. 빈 칸은 0, 숫자가 아니거나 소수면 ValueError (버림으로 숨기지 않는다)."""
    values = _floats(series, where).fillna(0)
    # 무한대는 1로 나눈 나머지가 NaN이라 여기서 함께 걸린다
    fractional = values % 1 != 0
    if bool(fractional.any()):
        found = sorted({str(v) for v in series[fractional]})[:_SHOW_MAX]
        raise ValueError(f"{where}: {series.name} 열에 정수가 아닌 아웃 수가 있다: {found}")
    return values.astype("int64")


def _season_frame(raw_dir: Path, season: int) -> pd.DataFrame:
    """한 시즌의 OAA·FRV를 선수 ID로 바깥 결합한다 (값은 더하지 않는다). 열은 계약 순서다."""
    frv = _load_unique(raw_dir, season, "frv")
    oaa = _load_unique(raw_dir, season, "oaa")
    # 형 변환은 파일별로 해서 오류 메시지에 어느 원본인지 적는다
    frv_name = raw_savant_path(season, "frv").as_posix()
    oaa_name = raw_savant_path(season, "oaa").as_posix()
    frv_part = pd.DataFrame(
        {
            "player_id": frv["player_id"],
            **{clean: _floats(frv[src], frv_name) for src, clean in FRV_COLUMNS.items()},
            **{col: _outs(frv[col], frv_name) for col in _OUTS_COLUMNS},
        }
    )
    oaa_part = pd.DataFrame(
        {
            "player_id": oaa["player_id"],
            **{clean: _floats(oaa[src], oaa_name) for src, clean in _OAA_COLUMNS.items()},
        }
    )
    merged = oaa_part.merge(frv_part, on="player_id", how="outer", validate="one_to_one")
    # OAA에만 있는 선수는 FRV 아웃 수가 없다 → 0. 결합으로 실수가 된 열을 정수로 되돌린다
    for col in _OUTS_COLUMNS:
        merged[col] = merged[col].fillna(0).astype("int64")
    merged["season"] = season
    merged["league"] = LEAGUE
    return merged[FIELDING_SEASONS.column_names]


def build_fielding(raw_dir: Path) -> pd.DataFrame:
    """수비 시즌 표를 만든다 (`contracts.FIELDING_SEASONS`, `check_frame` 통과, season·player_id 순).

    - 시즌: `contracts.FIELDING_YEARS`. league = "MLB".
    - FRV(`load_savant(..., "frv")`): `FRV_COLUMNS`대로 바꾸고, `outs_2`~`outs_9`는 정수(빈 칸은 0).
    - OAA(`load_savant(..., "oaa")`): oaa = outs_above_average.
    - (player_id, season)으로 **바깥 결합**한다. OAA에만 있는 선수는 FRV 열이 결측·아웃 수 0,
      FRV에만 있는 선수는 oaa가 결측.
    - 같은 (player_id, season)이 한 파일에 두 번 있으면 값이 같을 때만 하나로, 다르면 ValueError.
    """
    df = pd.concat([_season_frame(raw_dir, season) for season in FIELDING_YEARS], ignore_index=True)
    return df.sort_values(["season", "player_id"], kind="mergesort").reset_index(drop=True)
