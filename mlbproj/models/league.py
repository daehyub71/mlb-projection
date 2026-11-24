"""시즌별 리그 평균 (SPEC F6·F7·D11, `contracts.LEAGUE_BAT`·`LEAGUE_PIT`). 레인 A가 구현한다.

입력은 정제 표(`BATTING_SEASONS`·`PITCHING_SEASONS`)다. 리그 평균은 그 역할 표의 **모든 행 합**으로
계산한다(선수 평균의 평균이 아니다). 목표 시즌 Y의 모델은 Y−1 행만 쓴다 — 한 시즌의 행은 그 시즌
기록만으로 정해지므로 미래가 섞이지 않는다(N2).

공식 (합은 그 시즌 모든 행)
- 타자: h1 = h − b2 − b3 − hr. r_<성분> = Σ성분 / Σpa (성분 = `contracts.BAT_COMPONENTS`).
  obp = (Σh + Σbb + Σhbp) / (Σab + Σbb + Σhbp + Σsf), slg = (Σh1 + 2Σb2 + 3Σb3 + 4Σhr) / Σab,
  ops = obp + slg, hr_per_pa = Σhr / Σpa.
- 투수: r_<성분> = Σ성분 / Σbf (`contracts.PIT_COMPONENTS`), er_per_out = Σer / Σouts,
  era = 27 × er_per_out, k_bb_pct = 100 × (r_so − r_bb),
  fip_const = era − (13Σhr + 3(Σbb + Σhbp) − 2Σso) / (Σouts / 3), bf_per_out = Σbf / Σouts.
"""

import pandas as pd

from mlbproj.contracts import (
    BAT_COMPONENTS,
    BATTING_SEASONS,
    LEAGUE_BAT,
    LEAGUE_PIT,
    PIT_COMPONENTS,
    PITCHING_SEASONS,
    TableSpec,
    check_frame,
)

# 공식이 읽는 개수 열 (h1은 h에서 만든다). 2020도 환산 전 실제 표본이다(D9 — 환산은 출전량에만).
_BAT_COUNTS: tuple[str, ...] = ("pa", "h", "b2", "b3", "hr", "bb", "hbp", "sf", "ab")
_PIT_COUNTS: tuple[str, ...] = ("bf", "outs", "er", *PIT_COMPONENTS)

_OUTS_PER_GAME = 27  # ERA = 9이닝(27아웃)당 자책점
_OUTS_PER_INNING = 3


def _require(df: pd.DataFrame, spec: TableSpec, what: str) -> None:
    """표가 계약을 지키는지 확인한다. 어기면 ValueError (메시지 앞에 what을 붙인다)."""
    problems = check_frame(df, spec)
    if problems:
        raise ValueError(f"{what} 계약 위반: " + "; ".join(problems))


def _positive(values: pd.Series) -> pd.Series:
    """분모로 쓸 값. 0 이하(또는 결측)는 결측으로 바꿔 나눈 결과가 결측이 되게 한다."""
    return values.where(values > 0)


def _bat_rates(counts: pd.DataFrame) -> pd.DataFrame:
    """타자 개수 합 → r_*·obp·slg·ops·hr_per_pa (모듈 docstring 공식). 분모가 0인 칸은 결측.

    한 시즌 리그 합에도, 한 선수·한 시즌 행에도 쓴다(기준 모델). counts에는 `_BAT_COUNTS` 열이 있어야 한다.
    """
    comps = counts.assign(h1=counts["h"] - counts["b2"] - counts["b3"] - counts["hr"])
    pa = _positive(comps["pa"])
    ab = _positive(comps["ab"])
    out = pd.DataFrame({f"r_{name}": comps[name] / pa for name in BAT_COMPONENTS})
    on_base = comps["h"] + comps["bb"] + comps["hbp"]
    out["obp"] = on_base / _positive(comps["ab"] + comps["bb"] + comps["hbp"] + comps["sf"])
    bases = comps["h1"] + 2 * comps["b2"] + 3 * comps["b3"] + 4 * comps["hr"]
    out["slg"] = bases / ab
    out["ops"] = out["obp"] + out["slg"]
    out["hr_per_pa"] = comps["hr"] / pa
    return out


def _pit_rates(counts: pd.DataFrame) -> pd.DataFrame:
    """투수 개수 합 → r_*·er_per_out·era·k_bb_pct (모듈 docstring 공식). 분모가 0인 칸은 결측.

    counts에는 `_PIT_COUNTS` 열이 있어야 한다.
    """
    bf = _positive(counts["bf"])
    out = pd.DataFrame({f"r_{name}": counts[name] / bf for name in PIT_COMPONENTS})
    out["er_per_out"] = counts["er"] / _positive(counts["outs"])
    out["era"] = _OUTS_PER_GAME * out["er_per_out"]
    out["k_bb_pct"] = 100 * (out["r_so"] - out["r_bb"])
    return out


def _season_sums(seasons: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """시즌별 개수 합 (season 색인, 오름차순)."""
    return seasons.groupby("season", sort=True)[list(columns)].sum()


def _reject_zero(sums: pd.DataFrame, denominators: tuple[str, ...], role: str) -> None:
    """어떤 시즌의 분모 합이 0이면 ValueError (그 시즌의 리그 평균을 정할 수 없다).

    메시지에는 시즌마다 합이 0인 분모 열을 적는다. 예: "2022(ab)".
    """
    zero = sums[list(denominators)] <= 0
    bad = [
        f"{int(season)}({'·'.join(name for name in denominators if bool(zero.at[season, name]))})"
        for season in zero.index[zero.any(axis=1)].tolist()
    ]
    if bad:
        raise ValueError(f"{role} 리그 평균을 낼 수 없다 — 분모 합이 0인 시즌: {', '.join(bad)}")


def league_batting(batting: pd.DataFrame) -> pd.DataFrame:
    """시즌별 타자 리그 평균 (`contracts.LEAGUE_BAT`, `check_frame` 통과, season 순).

    Args:
        batting: `contracts.BATTING_SEASONS` 표.

    Raises:
        ValueError: 입력이 계약을 어길 때, 또는 어떤 시즌의 Σpa·Σab가 0일 때.
    """
    _require(batting, BATTING_SEASONS, "타자 정제 표")
    sums = _season_sums(batting, _BAT_COUNTS)
    _reject_zero(sums, ("pa", "ab"), "타자")
    out = pd.concat([sums[["pa"]], _bat_rates(sums)], axis=1).reset_index()
    return out[LEAGUE_BAT.column_names]


def league_pitching(pitching: pd.DataFrame) -> pd.DataFrame:
    """시즌별 투수 리그 평균 (`contracts.LEAGUE_PIT`, `check_frame` 통과, season 순).

    Args:
        pitching: `contracts.PITCHING_SEASONS` 표.

    Raises:
        ValueError: 입력이 계약을 어길 때, 또는 어떤 시즌의 Σbf·Σouts가 0일 때.
    """
    _require(pitching, PITCHING_SEASONS, "투수 정제 표")
    sums = _season_sums(pitching, _PIT_COUNTS)
    _reject_zero(sums, ("bf", "outs"), "투수")
    out = pd.concat([sums[["bf", "outs"]], _pit_rates(sums)], axis=1)
    fip_core = 13 * sums["hr"] + 3 * (sums["bb"] + sums["hbp"]) - 2 * sums["so"]
    out["fip_const"] = out["era"] - fip_core / (sums["outs"] / _OUTS_PER_INNING)
    out["bf_per_out"] = sums["bf"] / sums["outs"]
    return out.reset_index()[LEAGUE_PIT.column_names]


def league_row(league: pd.DataFrame, season: int) -> pd.Series:
    """리그 평균 표에서 한 시즌 행을 꺼낸다.

    Raises:
        KeyError: 그 시즌이 없을 때 (메시지에 시즌을 적는다).
    """
    rows = league[league["season"] == season]
    if rows.empty:
        raise KeyError(f"리그 평균 표에 {season} 시즌이 없다")
    return rows.iloc[0]
