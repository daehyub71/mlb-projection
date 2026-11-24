"""가중 + 리그 평균 회귀 비율 모델 (SPEC F7·D9·D10·D11). 레인 B가 구현한다.

입력은 목표 시즌 Y의 대상 표(`COHORTS`), 같은 역할 정제 표, 리그 평균 표(`LEAGUE_BAT`/`LEAGUE_PIT`),
파라미터(`models.params.Params`)다. **Y 이후 행·리그 평균은 쓰지 않는다**(N2). 리그 평균은 Y−1 행만 쓴다.

규칙
- 대상마다 기간(window_first ~ window_last, 최대 5시즌) 안의 행만 쓴다. 시즌 Y−k의 가중치는
  `params.<role>.weights[k−1]` (k = 1…5). 행이 없는 시즌은 더하지 않는다.
- 비율 지표는 **실제 표본**(pa·bf·outs, 환산 전)으로 가중한다(D9 — 2020 환산은 출전량에만).
- 성분 x의 회귀 비율 = (Σ w·x + R_x · lg_x) / (Σ w·분모 + R_x). R_x = `params.<role>.regress[x]`,
  lg_x = Y−1 리그의 r_x. 타자 분모는 pa, 투수 성분 분모는 bf.
- 타자: 회귀한 r_h1…r_ab로 obp = (r_h1+r_b2+r_b3+r_hr+r_bb+r_hbp)/(r_ab+r_bb+r_hbp+r_sf),
  slg = (r_h1+2r_b2+3r_b3+4r_hr)/r_ab, ops = obp + slg, hr_per_pa = r_hr. lg_ops = Y−1 리그 ops.
- 투수: er_per_out = (Σw·er + R_er·lg_er_per_out)/(Σw·outs + R_er). k_bb_pct = 100(r_so − r_bb).
  era는 `params.era_method`로 고른다(D11):
  - "er": era = 27 × er_per_out.
  - "fip": 선수 bf_per_out = (Σw·bf + R_er·lg_bf_per_out)/(Σw·outs + R_er),
    era = (13r_hr + 3(r_bb + r_hbp) − 2r_so) × bf_per_out × 3 + Y−1 리그 fip_const.
  lg_era·lg_k_bb_pct = Y−1 리그 값.
- 결과의 선수 집합은 대상 표와 같고 player_id 순, 결측 없음. target_season = Y.
"""

import pandas as pd

from mlbproj.contracts import (
    BAT_COMPONENTS,
    BAT_RATES,
    BATTING_SEASONS,
    COHORTS,
    LEAGUE_BAT,
    LEAGUE_PIT,
    PIT_COMPONENTS,
    PIT_RATES,
    PITCHING_SEASONS,
    WINDOW,
    Role,
    TableSpec,
    check_frame,
)
from mlbproj.models.params import Params, validate_params

# 역할별 입력·결과 계약
_SEASON_SPECS: dict[Role, TableSpec] = {"bat": BATTING_SEASONS, "pit": PITCHING_SEASONS}
_LEAGUE_SPECS: dict[Role, TableSpec] = {"bat": LEAGUE_BAT, "pit": LEAGUE_PIT}
_RATE_SPECS: dict[Role, TableSpec] = {"bat": BAT_RATES, "pit": PIT_RATES}
# 가중해 더하는 횟수 열 — 실제 표본이다(환산값 pa_adj·outs_adj가 아니다, D9).
# 타자 h1(단타)은 정제 표에 없어서 h − b2 − b3 − hr로 만든다.
_SUM_COLUMNS: dict[Role, tuple[str, ...]] = {
    "bat": ("pa", *BAT_COMPONENTS),
    "pit": ("bf", "outs", "er", *PIT_COMPONENTS),
}
# 오류 메시지에 적는 선수 ID의 최대 개수
_SHOW_MAX = 5


def _check_inputs(
    cohort: pd.DataFrame,
    seasons: pd.DataFrame,
    league: pd.DataFrame,
    role: Role,
    params: Params,
) -> int:
    """파라미터·입력 표를 확인하고 목표 시즌 Y를 돌려준다. 어기면 ValueError."""
    validate_params(params)
    checks = (
        ("대상 표", cohort, COHORTS),
        ("정제 표", seasons, _SEASON_SPECS[role]),
        ("리그 평균 표", league, _LEAGUE_SPECS[role]),
    )
    for label, df, spec in checks:
        problems = check_frame(df, spec)
        if problems:
            raise ValueError(
                f"{label}가 {role} 입력 계약({spec.name})을 어긴다: " + "; ".join(problems)
            )
    roles = sorted(set(cohort["role"]))
    if roles != [role]:
        raise ValueError(f"대상 표의 역할은 {role} 하나여야 한다: {roles}")
    targets = sorted({int(t) for t in cohort["target_season"]})
    if len(targets) != 1:
        raise ValueError(f"대상 표의 목표 시즌은 하나여야 한다: {targets}")
    target = targets[0]
    _check_window(cohort, target)
    return target


def _check_window(cohort: pd.DataFrame, target: int) -> None:
    """대상마다 기간이 Y−WINDOW ≤ window_first ≤ window_last = Y−1인지 확인한다. 어기면 ValueError.

    `COHORTS` 계약의 기간 규칙이다. 손으로 고친 기간(예: window_last = Y)을 조용히 넘기지 않는다(N2).
    """
    outside = (
        (cohort["window_last"] != target - 1)
        | (cohort["window_first"] < target - WINDOW)
        | (cohort["window_first"] > cohort["window_last"])
    )
    if bool(outside.any()):
        ids = sorted(int(i) for i in cohort.loc[outside, "player_id"])
        raise ValueError(
            f"기간이 목표 시즌 {target}의 {target - WINDOW}~{target - 1}을 벗어난 대상이 "
            f"{len(ids)}명 있다: {ids[:_SHOW_MAX]}"
        )


def _league_prev(league: pd.DataFrame, season: int) -> dict[str, float]:
    """리그 평균 표의 한 시즌 행 (가까운 시즌으로 대신하지 않는다). 없으면 KeyError."""
    rows = league[league["season"] == season]
    if rows.empty:
        raise KeyError(f"리그 평균 표에 {season} 시즌(목표 시즌의 Y−1) 행이 없다")
    return {str(name): float(value) for name, value in rows.iloc[0].items()}


def _weighted_sums(
    cohort: pd.DataFrame,
    seasons: pd.DataFrame,
    role: Role,
    target: int,
    weights: tuple[float, ...],
) -> pd.DataFrame:
    """대상마다 기간 안 행의 Σ w·횟수 (index = 대상 표의 player_id, 오름차순).

    시즌 Y−k의 가중치는 weights[k−1]이다. Y−WINDOW ~ Y−1 밖(Y 이후 포함, N2)이거나
    대상의 기간 밖인 행은 쓰지 않는다. 기간 안 행이 없는 대상은 0이다.
    더하기 전에 행을 키 순서로 정렬한다 — 정수가 아닌 가중치의 합은 더하는 순서에 따라 끝자리가
    달라지므로, 입력 행 순서와 상관없이 같은 비트가 나오게 한다.
    """
    # Y−1 … Y−WINDOW (길이는 validate_params가 보장한다)
    weight_of = {target - k: w for k, w in zip(range(1, WINDOW + 1), weights, strict=True)}
    window = cohort[["player_id", "window_first", "window_last"]]
    rows = seasons.merge(window, on="player_id", how="inner", validate="many_to_one")
    # N2: 가중치 표에 있는 시즌(Y−WINDOW ~ Y−1)만 — 기간 검사와 별도로 한 번 더 막는다
    keep = rows["season"].isin(list(weight_of)) & rows["season"].between(
        rows["window_first"], rows["window_last"]
    )
    rows = rows[keep].sort_values(list(_SEASON_SPECS[role].keys), kind="mergesort")
    if role == "bat":
        rows = rows.assign(h1=rows["h"] - rows["b2"] - rows["b3"] - rows["hr"])
    columns = list(_SUM_COLUMNS[role])
    weight = rows["season"].map(weight_of).astype("float64")
    sums = rows[columns].astype("float64").mul(weight, axis=0).groupby(rows["player_id"]).sum()
    ids = pd.Index(sorted(int(i) for i in cohort["player_id"]), name="player_id")
    return sums.reindex(ids, fill_value=0.0)


def _ratio(num: pd.Series, den: pd.Series, fallback: float | pd.Series) -> pd.Series:
    """num / den. 분모가 0인 곳은 fallback(Y−1 리그 값)으로 둔다 — 결측·무한을 내지 않는다.

    분모가 0이 되는 것은 R = 0이면서 가중 표본(Σw·pa·bf·outs)도 0이거나, 파생 분모(r_ab, obp 분모)가
    0일 때뿐이다. R > 0이면 표본 0의 회귀 비율은 공식대로 리그 값이므로 0/0은 그 극한을 R = 0까지 잇는다.
    아웃 0인데 자책점이 있는 x/0(무한 ERA)도 리그 값으로 둔다 — 비율 표는 "기록이 없으면 리그 평균"이다.
    """
    return (num / den).where(den > 0, fallback)


def _regressed(sums: pd.DataFrame, x: str, den: str, regress: float, lg: float) -> pd.Series:
    """성분 x의 회귀 비율 (Σw·x + R·lg) / (Σw·분모 + R)."""
    return _ratio(sums[x] + regress * lg, sums[den] + regress, lg)


def _bat_rates(sums: pd.DataFrame, lg: dict[str, float], params: Params) -> dict[str, pd.Series]:
    """타자 성분 비율(타석당)과 obp·slg·ops·hr_per_pa·lg_ops."""
    regress = params.bat.regress
    r = {x: _regressed(sums, x, "pa", regress[x], lg[f"r_{x}"]) for x in BAT_COMPONENTS}
    obp = _ratio(
        r["h1"] + r["b2"] + r["b3"] + r["hr"] + r["bb"] + r["hbp"],
        r["ab"] + r["bb"] + r["hbp"] + r["sf"],
        lg["obp"],
    )
    slg = _ratio(r["h1"] + 2 * r["b2"] + 3 * r["b3"] + 4 * r["hr"], r["ab"], lg["slg"])
    return {
        **{f"r_{x}": r[x] for x in BAT_COMPONENTS},
        "obp": obp,
        "slg": slg,
        "ops": obp + slg,
        "hr_per_pa": r["hr"],
        "lg_ops": pd.Series(lg["ops"], index=sums.index),
    }


def _pit_rates(sums: pd.DataFrame, lg: dict[str, float], params: Params) -> dict[str, pd.Series]:
    """투수 성분 비율(상대 타자당)과 er_per_out·era·k_bb_pct·lg_era·lg_k_bb_pct."""
    regress = params.pit.regress
    r = {x: _regressed(sums, x, "bf", regress[x], lg[f"r_{x}"]) for x in PIT_COMPONENTS}
    r_er = regress["er"]
    er_per_out = _regressed(sums, "er", "outs", r_er, lg["er_per_out"])
    if params.era_method == "er":
        era = 27 * er_per_out
    else:
        # FIP식 (D11): 상대 타자당 비율을 아웃당으로 바꾸는 bf_per_out도 R_er로 회귀한다
        bf_per_out = _regressed(sums, "bf", "outs", r_er, lg["bf_per_out"])
        core = 13 * r["hr"] + 3 * (r["bb"] + r["hbp"]) - 2 * r["so"]
        era = core * bf_per_out * 3 + lg["fip_const"]
    return {
        **{f"r_{x}": r[x] for x in PIT_COMPONENTS},
        "er_per_out": er_per_out,
        "era": era,
        "k_bb_pct": 100 * (r["so"] - r["bb"]),
        "lg_era": pd.Series(lg["era"], index=sums.index),
        "lg_k_bb_pct": pd.Series(lg["k_bb_pct"], index=sums.index),
    }


def weighted_rates(
    cohort: pd.DataFrame,
    seasons: pd.DataFrame,
    league: pd.DataFrame,
    role: Role,
    params: Params,
) -> pd.DataFrame:
    """가중·회귀 비율. 타자 `contracts.BAT_RATES`, 투수 `contracts.PIT_RATES` 형식 (`check_frame` 통과).

    Raises:
        ValueError: 입력 표가 계약을 어기거나, 대상 표의 역할·목표 시즌이 하나가 아니거나, 파라미터가 틀릴 때.
        KeyError: 리그 평균에 Y−1이 없을 때.
    """
    target = _check_inputs(cohort, seasons, league, role, params)
    # N2: 리그 평균은 Y−1 행만 쓴다
    lg = _league_prev(league, target - 1)
    rate = params.bat if role == "bat" else params.pit
    sums = _weighted_sums(cohort, seasons, role, target, rate.weights)
    values = _bat_rates(sums, lg, params) if role == "bat" else _pit_rates(sums, lg, params)
    out = pd.DataFrame(values, index=sums.index).reset_index()
    out["target_season"] = target
    return out[_RATE_SPECS[role].column_names]
