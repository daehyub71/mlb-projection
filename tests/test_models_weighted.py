"""가중 + 리그 평균 회귀 비율 모델(models/weighted.py) 테스트 (SPEC F7·D9·D10·D11·N2).

입력은 모두 이 파일에서 만든 작은 합성 표다.
- 정제 표: `_seasons`가 `BATTING_SEASONS`·`PITCHING_SEASONS`의 모든 열을 채운다. 시즌별 obp·slg·ops와
  Statcast 같은 나머지 실수 열은 엉뚱한 값(`DECOY`)이라, 모델이 성분 대신 이 열을 쓰면 기대값과 어긋난다.
- 리그 평균 표: `_league`가 시즌 합계(`*_LEAGUE_TOTALS`)로 `LEAGUE_BAT`·`LEAGUE_PIT` 행을 만든다
  (리그 모듈은 쓰지 않는다). 합계에서 만들어 표 안의 값끼리 맞는다(fip_const 포함).
- 대상 표: `cohort.build_cohort`(M2 완료)로 만든다.

기대값은 손으로 계산해 각 테스트 안에 산식을 적었다. 규칙(weighted.py docstring):
시즌 Y−k의 가중치 = weights[k−1], 성분 x의 회귀 비율 = (Σw·x + R_x·lg_x) / (Σw·분모 + R_x),
분모는 실제 표본(타자 pa, 투수 bf, 자책점은 outs — 2020 환산값이 아니다, D9).
"""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlbproj import contracts as c
from mlbproj.cohort import build_cohort
from mlbproj.contracts import Role
from mlbproj.models.params import Params, RateParams, validate_params
from mlbproj.models.weighted import weighted_rates

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

TARGET = 2025  # 기본 목표 시즌 Y — 기간 2020~2024, 리그 평균은 2024 행
WEIGHTS: tuple[float, ...] = (5.0, 4.0, 3.0, 2.0, 1.0)  # [Y−1, Y−2, Y−3, Y−4, Y−5]
# 정수가 아닌 가중치 — 곱이 정확하지 않아서 더하는 순서가 결과 비트를 바꿀 수 있다
FRACTIONAL_WEIGHTS: tuple[float, ...] = (0.91, 0.737, 0.333, 0.217, 0.131)
# 성분마다 다른 회귀 표본 — 성분끼리 R이 섞이면 기대값과 어긋난다
BAT_R: dict[str, float] = {
    "h1": 100.0,
    "b2": 200.0,
    "b3": 300.0,
    "hr": 400.0,
    "bb": 150.0,
    "hbp": 50.0,
    "sf": 250.0,
    "ab": 500.0,
}
PIT_R: dict[str, float] = {"so": 100.0, "bb": 200.0, "hbp": 300.0, "hr": 400.0, "er": 150.0}
HUGE_R = 1e12  # 리그 평균으로 사실상 완전히 회귀

DECOY = 9.0  # 모델이 쓰면 안 되는 실수 열의 값
POISON = 99_999  # 목표 시즌 이후 행에 넣는 "독" 값 (N2)
NEW_PLAYER = 999  # 목표 시즌 이후에만 기록이 있는 선수
ERA_METHODS: tuple[str, ...] = ("er", "fip")

SPECS: dict[Role, c.TableSpec] = {"bat": c.BATTING_SEASONS, "pit": c.PITCHING_SEASONS}
LEAGUE_SPECS: dict[Role, c.TableSpec] = {"bat": c.LEAGUE_BAT, "pit": c.LEAGUE_PIT}
RATE_SPECS: dict[Role, c.TableSpec] = {"bat": c.BAT_RATES, "pit": c.PIT_RATES}

# ---------------------------------------------------------------------------
# 합성 선수 기록
# ---------------------------------------------------------------------------

_BAT_FIELDS = ("player_id", "season", "pa", "ab", "h", "b2", "b3", "hr", "bb", "hbp", "sf")
# h1 = h − b2 − b3 − hr. pa = ab + bb + hbp + sf (sh 0). Y = 2025 기준 가중치를 적었다
_BAT_TABLE: tuple[tuple[int, ...], ...] = (
    # pid, season, pa, ab, h, b2, b3, hr, bb, hbp, sf
    (11, 2019, 700, 600, 200, 50, 10, 50, 80, 10, 10),  # Y−6: 기간(2020~) 밖 — 더하면 안 된다
    (11, 2021, 300, 270, 66, 12, 1, 8, 25, 3, 2),  # Y−4 w=2, h1 45
    (11, 2023, 500, 450, 120, 25, 2, 18, 40, 5, 5),  # Y−2 w=4, h1 75 (2022·2020 결장)
    (11, 2024, 600, 540, 150, 30, 3, 27, 50, 6, 4),  # Y−1 w=5, h1 90
    (12, 2023, 200, 184, 40, 6, 0, 2, 12, 2, 2),  # 2023 데뷔, Y−2 w=4, h1 32
    (12, 2024, 400, 360, 100, 20, 2, 16, 30, 5, 5),  # Y−1 w=5, h1 62
    (13, 2020, 100, 92, 20, 4, 0, 1, 6, 1, 1),  # Y−5 w=1, 2020 (pa_adj 270), h1 15
    (13, 2022, 500, 450, 125, 25, 5, 25, 40, 5, 5),  # Y−3 w=3, 이후 미출전, h1 70
)
_PIT_FIELDS = ("player_id", "season", "bf", "outs", "er", "so", "bb", "hbp", "hr")
_PIT_TABLE: tuple[tuple[int, ...], ...] = (
    # pid, season, bf, outs, er, so, bb, hbp, hr
    (21, 2019, 800, 500, 150, 50, 100, 10, 40),  # Y−6: 기간 밖
    (21, 2020, 250, 180, 25, 60, 20, 2, 8),  # Y−5 w=1, 2020 (outs_adj 486)
    (21, 2023, 600, 420, 66, 150, 45, 5, 18),  # Y−2 w=4 (2022·2021 결장)
    (21, 2024, 700, 510, 70, 180, 50, 6, 20),  # Y−1 w=5
    (22, 2022, 100, 60, 20, 15, 15, 3, 5),  # Y−3 w=3
    (22, 2024, 300, 200, 30, 90, 20, 2, 10),  # Y−1 w=5
)


def _rows(fields: Sequence[str], table: Sequence[Sequence[int]]) -> list[dict[str, int]]:
    return [dict(zip(fields, values, strict=True)) for values in table]


BAT_ROWS = _rows(_BAT_FIELDS, _BAT_TABLE)
PIT_ROWS = _rows(_PIT_FIELDS, _PIT_TABLE)

# 리그 시즌 합계. 시즌마다 값이 달라서 Y−1이 아닌 행을 쓰면 기대값과 어긋난다
_BAT_LEAGUE_FIELDS = ("pa", "h1", "b2", "b3", "hr", "bb", "hbp", "sf", "ab")
BAT_LEAGUE_TOTALS: dict[int, tuple[int, ...]] = {
    2020: (36_000, 4_860, 1_692, 144, 1_188, 3_240, 432, 288, 31_896),
    2021: (96_000, 13_920, 4_224, 384, 2_784, 8_352, 1_056, 768, 85_440),
    2022: (99_000, 13_860, 4_455, 396, 2_970, 8_415, 1_089, 792, 88_110),
    2023: (98_000, 14_210, 4_116, 490, 2_744, 8_820, 980, 784, 86_816),
    # 2024: r_h1 .14, r_b2 .045, r_b3 .004, r_hr .03, r_bb .085, r_hbp .011, r_sf .008, r_ab .89
    2024: (100_000, 14_000, 4_500, 400, 3_000, 8_500, 1_100, 800, 89_000),
}
_PIT_LEAGUE_FIELDS = ("bf", "outs", "er", "so", "bb", "hbp", "hr")
PIT_LEAGUE_TOTALS: dict[int, tuple[int, ...]] = {
    2020: (36_000, 27_000, 4_800, 8_280, 3_240, 396, 1_188),
    2021: (96_000, 72_000, 12_000, 21_120, 8_160, 1_056, 2_880),
    2022: (99_000, 74_250, 11_880, 22_770, 7_920, 1_089, 2_772),
    2023: (98_000, 73_500, 13_230, 21_560, 8_820, 980, 3_430),
    # 2024: r_so .225, r_bb .085, r_hbp .011, r_hr .03, er_per_out 12500/75000 = 1/6, era 4.5,
    # k_bb_pct 14, bf_per_out 4/3, fip_const = 4.5 − (13·3000 + 3·9600 − 2·22500)/25000 = 3.588
    2024: (100_000, 75_000, 12_500, 22_500, 8_500, 1_100, 3_000),
}
# 목표 시즌 이후 리그 행에 넣는 "독" 합계 (N2)
POISON_LEAGUE_TOTALS: dict[Role, tuple[int, ...]] = {
    "bat": (1_000, 900, 900, 900, 900, 900, 900, 900, 10),
    "pit": (1_000, 10, 900, 1, 900, 900, 900),
}

# ---------------------------------------------------------------------------
# 도우미 — 계약 형식의 합성 표
# ---------------------------------------------------------------------------


def _dtype(col: c.Column) -> str:
    if col.kind == "int":
        return "Int64" if col.nullable else "int64"
    if col.kind == "float":
        return "float64"
    return col.kind  # "bool" · "str"


def _frame(spec: c.TableSpec, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """계약의 모든 열을 계약 순서·형식으로 담은 표. 계약을 지키는지 바로 확인한다."""
    df = pd.DataFrame(
        {
            col.name: pd.Series([row[col.name] for row in rows], dtype=_dtype(col))
            for col in spec.columns
        }
    )
    assert c.check_frame(df, spec) == []
    return df


_SEASON_DEFAULTS: dict[Role, dict[str, Any]] = {
    "bat": {"league": c.LEAGUE, "age": 27, "team_name": "TST", "n_teams": 1, "ibb": 0, "so": 100},
    "pit": {
        "league": c.LEAGUE,
        "age": 28,
        "team_name": "TST",
        "n_teams": 1,
        "g": 30,
        "gs": 10,
        "h": 50,
        "sv": 0,
    },
}


def _season_row(role: Role, row: Mapping[str, int]) -> dict[str, Any]:
    """정제 표 한 행. 2020은 출전량 열(pa_adj·outs_adj)만 162/60배다(D9)."""
    season = int(row["season"])
    scale = c.SHORT_SEASON_SCALE if season == c.SHORT_SEASON else 1.0
    out: dict[str, Any] = {
        **_SEASON_DEFAULTS[role],
        **row,
        "is_short_season": season == c.SHORT_SEASON,
    }
    if role == "bat":
        out.setdefault("sh", out["pa"] - out["ab"] - out["bb"] - out["hbp"] - out["sf"])
        out["pa_adj"] = out["pa"] * scale
    else:
        out["outs_adj"] = out["outs"] * scale
    for col in SPECS[role].columns:
        if col.name not in out:
            assert col.kind == "float", col.name
            out[col.name] = DECOY  # 시즌 obp·slg·ops, Statcast — 모델이 쓰면 안 된다
    return out


def _seasons(role: Role, rows: Sequence[Mapping[str, int]] | None = None) -> pd.DataFrame:
    """합성 정제 표. rows가 없으면 그 역할의 기본 선수 기록."""
    if rows is None:
        rows = BAT_ROWS if role == "bat" else PIT_ROWS
    return _frame(SPECS[role], [_season_row(role, row) for row in rows])


def _league_row(role: Role, season: int, totals: Sequence[int]) -> dict[str, Any]:
    """시즌 합계 → 리그 평균 행 (contracts의 LEAGUE_BAT·LEAGUE_PIT 주석 공식)."""
    if role == "bat":
        t = dict(zip(_BAT_LEAGUE_FIELDS, totals, strict=True))
        h = t["h1"] + t["b2"] + t["b3"] + t["hr"]
        row: dict[str, Any] = {"season": season, "pa": t["pa"]}
        row.update({f"r_{x}": t[x] / t["pa"] for x in c.BAT_COMPONENTS})
        row["obp"] = (h + t["bb"] + t["hbp"]) / (t["ab"] + t["bb"] + t["hbp"] + t["sf"])
        row["slg"] = (t["h1"] + 2 * t["b2"] + 3 * t["b3"] + 4 * t["hr"]) / t["ab"]
        row["ops"] = row["obp"] + row["slg"]
        row["hr_per_pa"] = t["hr"] / t["pa"]
        return row
    t = dict(zip(_PIT_LEAGUE_FIELDS, totals, strict=True))
    row = {"season": season, "bf": t["bf"], "outs": t["outs"]}
    row.update({f"r_{x}": t[x] / t["bf"] for x in c.PIT_COMPONENTS})
    row["er_per_out"] = t["er"] / t["outs"]
    row["era"] = 27 * row["er_per_out"]
    row["k_bb_pct"] = 100 * (row["r_so"] - row["r_bb"])
    fip_core = 13 * t["hr"] + 3 * (t["bb"] + t["hbp"]) - 2 * t["so"]
    row["fip_const"] = row["era"] - fip_core / (t["outs"] / 3)
    row["bf_per_out"] = t["bf"] / t["outs"]
    return row


def _league(role: Role, totals: Mapping[int, Sequence[int]] | None = None) -> pd.DataFrame:
    """합성 리그 평균 표. totals가 없으면 2020~2024 기본 합계."""
    if totals is None:
        totals = BAT_LEAGUE_TOTALS if role == "bat" else PIT_LEAGUE_TOTALS
    rows = [_league_row(role, season, t) for season, t in totals.items()]
    return _frame(LEAGUE_SPECS[role], rows)


def _cohort(role: Role, seasons: pd.DataFrame, target: int = TARGET) -> pd.DataFrame:
    cohort = build_cohort(seasons, role, target)
    assert c.check_frame(cohort, c.COHORTS) == []
    return cohort


def _params(
    *,
    weights: tuple[float, ...] = WEIGHTS,
    bat_r: Mapping[str, float] = BAT_R,
    pit_r: Mapping[str, float] = PIT_R,
    era_method: str = "er",
    same_r: float | None = None,
) -> Params:
    """시험용 파라미터. same_r을 주면 두 역할의 모든 회귀 표본을 그 값으로 둔다."""
    if same_r is not None:
        bat_r = dict.fromkeys(c.BAT_COMPONENTS, same_r)
        pit_r = dict.fromkeys((*c.PIT_COMPONENTS, "er"), same_r)
    params = Params(
        version="test",
        bat=RateParams(weights=weights, regress=dict(bat_r)),
        pit=RateParams(weights=weights, regress=dict(pit_r)),
        era_method=era_method,
    )
    validate_params(params)
    return params


def _run(
    role: Role,
    params: Params,
    *,
    rows: Sequence[Mapping[str, int]] | None = None,
    target: int = TARGET,
) -> dict[int, dict[str, Any]]:
    """기본 리그 표로 모델을 돌리고 {player_id: 결과 행}으로 돌려준다."""
    seasons = _seasons(role, rows)
    out = weighted_rates(_cohort(role, seasons, target), seasons, _league(role), role, params)
    assert c.check_frame(out, RATE_SPECS[role]) == []
    return _records(out, "player_id")


def _records(df: pd.DataFrame, key: str) -> dict[int, dict[str, Any]]:
    """표 → {key 열 값: 행 dict}."""
    return {
        int(rec[key]): {str(name): value for name, value in rec.items()}
        for rec in df.to_dict("records")
    }


def _league_at(role: Role, season: int) -> dict[str, Any]:
    """기본 리그 표의 한 시즌 행."""
    return _records(_league(role), "season")[season]


def _pick(row: Mapping[str, Any], names: Sequence[str]) -> dict[str, float]:
    return {name: float(row[name]) for name in names}


def _bat_derived(r: Mapping[str, float]) -> dict[str, float]:
    """docstring 공식: 회귀한 성분 비율 → obp·slg·ops·hr_per_pa."""
    obp = (r["r_h1"] + r["r_b2"] + r["r_b3"] + r["r_hr"] + r["r_bb"] + r["r_hbp"]) / (
        r["r_ab"] + r["r_bb"] + r["r_hbp"] + r["r_sf"]
    )
    slg = (r["r_h1"] + 2 * r["r_b2"] + 3 * r["r_b3"] + 4 * r["r_hr"]) / r["r_ab"]
    return {"obp": obp, "slg": slg, "ops": obp + slg, "hr_per_pa": r["r_hr"]}


def _fip_era(r: Mapping[str, float], bf_per_out: float, fip_const: float) -> float:
    """docstring 공식: (13r_hr + 3(r_bb + r_hbp) − 2r_so) × bf_per_out × 3 + fip_const."""
    core = 13 * r["r_hr"] + 3 * (r["r_bb"] + r["r_hbp"]) - 2 * r["r_so"]
    return core * bf_per_out * 3 + fip_const


BAT_RATE_COLUMNS = [f"r_{x}" for x in c.BAT_COMPONENTS]
PIT_RATE_COLUMNS = [f"r_{x}" for x in c.PIT_COMPONENTS]
LG_2024_FIP_CONST = 4.5 - (13 * 3000 + 3 * (8500 + 1100) - 2 * 22500) / (75000 / 3)  # 3.588


# ---------------------------------------------------------------------------
# 입력 도우미 확인
# ---------------------------------------------------------------------------


def test_models_weighted_helpers_build_expected_league_2024() -> None:
    # 아래 손 계산이 쓰는 2024 리그 값이 합계에서 정확히 나오는지 먼저 확인한다
    bat = _league_at("bat", 2024)
    assert _pick(bat, BAT_RATE_COLUMNS) == pytest.approx(
        {
            "r_h1": 0.14,
            "r_b2": 0.045,
            "r_b3": 0.004,
            "r_hr": 0.03,
            "r_bb": 0.085,
            "r_hbp": 0.011,
            "r_sf": 0.008,
            "r_ab": 0.89,
        },
        rel=1e-12,
    )
    # obp = (14000+4500+400+3000+8500+1100)/(89000+8500+1100+800) = 31500/99400
    # slg = (14000 + 2·4500 + 3·400 + 4·3000)/89000 = 36200/89000
    assert float(bat["obp"]) == pytest.approx(31500 / 99400, rel=1e-12)
    assert float(bat["ops"]) == pytest.approx(31500 / 99400 + 36200 / 89000, rel=1e-12)
    pit = _league_at("pit", 2024)
    assert _pick(pit, [*PIT_RATE_COLUMNS, "er_per_out", "era", "k_bb_pct", "bf_per_out"]) == (
        pytest.approx(
            {
                "r_so": 0.225,
                "r_bb": 0.085,
                "r_hbp": 0.011,
                "r_hr": 0.03,
                "er_per_out": 1 / 6,
                "era": 4.5,
                "k_bb_pct": 14.0,
                "bf_per_out": 4 / 3,
            },
            rel=1e-12,
        )
    )
    assert float(pit["fip_const"]) == pytest.approx(3.588, rel=1e-12)
    assert LG_2024_FIP_CONST == pytest.approx(3.588, rel=1e-12)


# ---------------------------------------------------------------------------
# 타자 손 계산
# ---------------------------------------------------------------------------


def test_models_weighted_bat_hand_computed() -> None:
    row = _run("bat", _params())[11]
    # 선수 11 (Y = 2025, 기간 2020~2024): 2024(Y−1, w=5) · 2023(Y−2, w=4) · 2021(Y−4, w=2).
    # 2022·2020은 결장이라 더하지 않고, 2019 행은 기간 밖이라 더하지 않는다.
    # Σw·pa  = 5·600 + 4·500 + 2·300 = 5600
    # Σw·h1  = 5·90  + 4·75  + 2·45  = 840
    # Σw·b2  = 5·30  + 4·25  + 2·12  = 274
    # Σw·b3  = 5·3   + 4·2   + 2·1   = 25
    # Σw·hr  = 5·27  + 4·18  + 2·8   = 223
    # Σw·bb  = 5·50  + 4·40  + 2·25  = 460
    # Σw·hbp = 5·6   + 4·5   + 2·3   = 56
    # Σw·sf  = 5·4   + 4·5   + 2·2   = 44
    # Σw·ab  = 5·540 + 4·450 + 2·270 = 5040
    # r_x = (Σw·x + R_x·lg_x) / (Σw·pa + R_x), lg_x = 2024 리그
    expected = {
        "r_h1": (840 + 100 * 0.14) / (5600 + 100),  # 854/5700 ≈ .149825
        "r_b2": (274 + 200 * 0.045) / (5600 + 200),  # 283/5800 ≈ .048793
        "r_b3": (25 + 300 * 0.004) / (5600 + 300),  # 26.2/5900 ≈ .004441
        "r_hr": (223 + 400 * 0.03) / (5600 + 400),  # 235/6000 ≈ .039167
        "r_bb": (460 + 150 * 0.085) / (5600 + 150),  # 472.75/5750 ≈ .082217
        "r_hbp": (56 + 50 * 0.011) / (5600 + 50),  # 56.55/5650 ≈ .010009
        "r_sf": (44 + 250 * 0.008) / (5600 + 250),  # 46/5850 ≈ .007863
        "r_ab": (5040 + 500 * 0.89) / (5600 + 500),  # 5485/6100 ≈ .899180
    }
    assert _pick(row, BAT_RATE_COLUMNS) == pytest.approx(expected, rel=1e-12)
    # obp = (r_h1+r_b2+r_b3+r_hr+r_bb+r_hbp)/(r_ab+r_bb+r_hbp+r_sf) ≈ .3402
    # slg = (r_h1 + 2r_b2 + 3r_b3 + 4r_hr)/r_ab ≈ .4650, hr_per_pa = r_hr
    derived = _bat_derived(expected)
    assert _pick(row, ["obp", "slg", "ops", "hr_per_pa"]) == pytest.approx(derived, rel=1e-12)
    assert float(row["ops"]) == pytest.approx(float(row["obp"]) + float(row["slg"]), rel=1e-12)
    assert float(row["hr_per_pa"]) == float(row["r_hr"])
    # lg_ops = 2024 리그 ops = 31500/99400 + 36200/89000 (2023 행이 아니다)
    assert float(row["lg_ops"]) == pytest.approx(31500 / 99400 + 36200 / 89000, rel=1e-12)
    assert int(row["target_season"]) == TARGET


# ---------------------------------------------------------------------------
# 투수 손 계산
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("era_method", ERA_METHODS)
def test_models_weighted_pit_hand_computed(era_method: str) -> None:
    row = _run("pit", _params(era_method=era_method))[21]
    # 선수 21 (Y = 2025): 2024(Y−1, w=5) · 2023(Y−2, w=4) · 2020(Y−5, w=1). 2019는 기간 밖.
    # 2020은 실제 표본(outs 180, bf 250)으로 가중한다 — outs_adj 486이 아니다(D9).
    # Σw·bf   = 5·700 + 4·600 + 1·250 = 6150
    # Σw·outs = 5·510 + 4·420 + 1·180 = 4410
    # Σw·er   = 5·70  + 4·66  + 1·25  = 639
    # Σw·so   = 5·180 + 4·150 + 1·60  = 1560
    # Σw·bb   = 5·50  + 4·45  + 1·20  = 450
    # Σw·hbp  = 5·6   + 4·5   + 1·2   = 52
    # Σw·hr   = 5·20  + 4·18  + 1·8   = 180
    expected = {
        "r_so": (1560 + 100 * 0.225) / (6150 + 100),  # 1582.5/6250 = .2532
        "r_bb": (450 + 200 * 0.085) / (6150 + 200),  # 467/6350 ≈ .073543
        "r_hbp": (52 + 300 * 0.011) / (6150 + 300),  # 55.3/6450 ≈ .008574
        "r_hr": (180 + 400 * 0.03) / (6150 + 400),  # 192/6550 ≈ .029313
    }
    assert _pick(row, PIT_RATE_COLUMNS) == pytest.approx(expected, rel=1e-12)
    # er_per_out = (Σw·er + R_er·lg_er_per_out)/(Σw·outs + R_er) = (639 + 150/6)/(4410 + 150)
    er_per_out = (639 + 150 * (1 / 6)) / (4410 + 150)  # 664/4560 ≈ .145614
    assert float(row["er_per_out"]) == pytest.approx(er_per_out, rel=1e-12)
    # k_bb_pct = 100(r_so − r_bb) ≈ 17.9657
    assert float(row["k_bb_pct"]) == pytest.approx(
        100 * (expected["r_so"] - expected["r_bb"]), rel=1e-12
    )
    if era_method == "er":
        era = 27 * er_per_out  # 17928/4560 ≈ 3.9316
    else:
        # 선수 bf_per_out = (Σw·bf + R_er·lg_bf_per_out)/(Σw·outs + R_er) = (6150 + 150·4/3)/4560
        bf_per_out = (6150 + 150 * (4 / 3)) / (4410 + 150)  # 6350/4560 ≈ 1.392544
        era = _fip_era(expected, bf_per_out, LG_2024_FIP_CONST)  # ≈ 3.9795
    assert float(row["era"]) == pytest.approx(era, rel=1e-12)
    # lg_era·lg_k_bb_pct = 2024 리그 값 (era_method와 무관)
    assert float(row["lg_era"]) == pytest.approx(4.5, rel=1e-12)
    assert float(row["lg_k_bb_pct"]) == pytest.approx(14.0, rel=1e-12)
    assert int(row["target_season"]) == TARGET


# ---------------------------------------------------------------------------
# 가중치 배정·회귀 강도
# ---------------------------------------------------------------------------


def test_models_weighted_weights_follow_recency_from_target() -> None:
    # R = 0이면 r_x = Σw·x / Σw·분모 — 가중치 배정만 본다. 가중치는 Y와의 거리로 정한다
    # (기간 첫 시즌·마지막 출전 시즌 기준이 아니다).
    bat = _run("bat", _params(same_r=0.0))
    # 선수 12 (2023 데뷔): 2024 → w=5, 2023 → w=4. Σw·pa = 5·400 + 4·200 = 2800
    # (거꾸로 2023 → 5, 2024 → 4로 매기면 r_hr = 74/2600 ≈ .0285)
    assert _pick(bat[12], ["r_h1", "r_hr", "r_bb", "r_ab"]) == pytest.approx(
        {
            "r_h1": (5 * 62 + 4 * 32) / 2800,  # 438/2800
            "r_hr": (5 * 16 + 4 * 2) / 2800,  # 88/2800 ≈ .0314
            "r_bb": (5 * 30 + 4 * 12) / 2800,  # 198/2800
            "r_ab": (5 * 360 + 4 * 184) / 2800,  # 2536/2800
        },
        rel=1e-12,
    )
    # 선수 13 (2022가 마지막): 2022 → Y−3 w=3, 2020 → Y−5 w=1. Σw·pa = 3·500 + 1·100 = 1600
    # (마지막 출전을 Y−1로 당겨 2022 → 5, 2020 → 3으로 매기면 r_hr = 128/2800 ≈ .0457)
    assert _pick(bat[13], ["r_h1", "r_hr", "r_bb", "r_ab"]) == pytest.approx(
        {
            "r_h1": (3 * 70 + 1 * 15) / 1600,  # 225/1600
            "r_hr": (3 * 25 + 1 * 1) / 1600,  # 76/1600 = .0475
            "r_bb": (3 * 40 + 1 * 6) / 1600,  # 126/1600
            "r_ab": (3 * 450 + 1 * 92) / 1600,  # 1442/1600
        },
        rel=1e-12,
    )
    pit = _run("pit", _params(same_r=0.0))[22]
    # 선수 22: 2024 → w=5, 2022 → Y−3 w=3. Σw·bf = 5·300 + 3·100 = 1800, Σw·outs = 5·200 + 3·60 = 1180
    expected = {
        "r_so": (5 * 90 + 3 * 15) / 1800,  # 495/1800 = .275
        "r_bb": (5 * 20 + 3 * 15) / 1800,  # 145/1800
        "r_hbp": (5 * 2 + 3 * 3) / 1800,  # 19/1800
        "r_hr": (5 * 10 + 3 * 5) / 1800,  # 65/1800
    }
    assert _pick(pit, PIT_RATE_COLUMNS) == pytest.approx(expected, rel=1e-12)
    assert float(pit["er_per_out"]) == pytest.approx((5 * 30 + 3 * 20) / 1180, rel=1e-12)
    # k_bb_pct = 100·(495 − 145)/1800 ≈ 19.444
    assert float(pit["k_bb_pct"]) == pytest.approx(100 * 350 / 1800, rel=1e-12)


def test_models_weighted_zero_regress_is_pure_weighted_rate() -> None:
    # R = 0: 리그 평균이 섞이지 않는다. 선수 11의 합(위 손 계산)을 Σw·pa = 5600으로 나눈 값
    bat = _run("bat", _params(same_r=0.0))[11]
    expected = {
        "r_h1": 840 / 5600,  # .15
        "r_b2": 274 / 5600,
        "r_b3": 25 / 5600,
        "r_hr": 223 / 5600,
        "r_bb": 460 / 5600,
        "r_hbp": 56 / 5600,
        "r_sf": 44 / 5600,
        "r_ab": 5040 / 5600,  # .9
    }
    assert _pick(bat, BAT_RATE_COLUMNS) == pytest.approx(expected, rel=1e-12)
    # obp = (840+274+25+223+460+56)/(5040+460+56+44) = 1878/5600 ≈ .3354
    # slg = (840 + 2·274 + 3·25 + 4·223)/5040 = 2355/5040 ≈ .4673
    assert _pick(bat, ["obp", "slg", "ops", "hr_per_pa"]) == pytest.approx(
        {
            "obp": 1878 / 5600,
            "slg": 2355 / 5040,
            "ops": 1878 / 5600 + 2355 / 5040,
            "hr_per_pa": 223 / 5600,
        },
        rel=1e-12,
    )
    # 투수 21: Σw·bf 6150, Σw·outs 4410, Σw·er 639 (위 손 계산)
    for era_method in ERA_METHODS:
        pit = _run("pit", _params(era_method=era_method, same_r=0.0))[21]
        assert _pick(pit, PIT_RATE_COLUMNS) == pytest.approx(
            {"r_so": 1560 / 6150, "r_bb": 450 / 6150, "r_hbp": 52 / 6150, "r_hr": 180 / 6150},
            rel=1e-12,
        )
        assert float(pit["er_per_out"]) == pytest.approx(639 / 4410, rel=1e-12)
        # "er": 27·639/4410 ≈ 3.9122
        # "fip": (13·180 + 3·(450+52) − 2·1560)/6150 × (6150/4410) × 3 + 3.588
        #      = 726·3/4410 + 3.588 ≈ 4.0819 (가중 IP 1470의 교과서 FIP)
        era = 27 * 639 / 4410 if era_method == "er" else 726 * 3 / 4410 + LG_2024_FIP_CONST
        assert float(pit["era"]) == pytest.approx(era, rel=1e-12), era_method


def test_models_weighted_huge_regress_gives_league_rates() -> None:
    # R이 아주 크면 모든 비율이 Y−1(2024) 리그 값으로 간다
    lg_bat = _league_at("bat", 2024)
    bat = _run("bat", _params(same_r=HUGE_R))
    for pid in (11, 12, 13):
        row = bat[pid]
        names = [*BAT_RATE_COLUMNS, "obp", "slg", "ops", "hr_per_pa"]
        assert _pick(row, names) == pytest.approx(_pick(lg_bat, names), rel=1e-6), pid
        assert float(row["lg_ops"]) == pytest.approx(float(lg_bat["ops"]), rel=1e-12)
    for era_method in ERA_METHODS:
        pit = _run("pit", _params(era_method=era_method, same_r=HUGE_R))
        for pid in (21, 22):
            row = pit[pid]
            # 합계로 만든 리그 표는 서로 맞으므로 FIP식도 리그 era 4.5로 간다
            assert _pick(row, [*PIT_RATE_COLUMNS, "er_per_out", "era", "k_bb_pct"]) == (
                pytest.approx(
                    {
                        "r_so": 0.225,
                        "r_bb": 0.085,
                        "r_hbp": 0.011,
                        "r_hr": 0.03,
                        "er_per_out": 1 / 6,
                        "era": 4.5,
                        "k_bb_pct": 14.0,
                    },
                    rel=1e-6,
                )
            ), (era_method, pid)


def test_models_weighted_zero_weighted_sample_gets_league() -> None:
    # Y−5 가중치를 0으로 두면 2020에만 뛴 선수(15·25)의 가중 표본은 0이다
    # → r_x = (0 + R_x·lg_x)/(0 + R_x) = lg_x (2024 리그 값 그대로)
    weights = (5.0, 4.0, 3.0, 2.0, 0.0)
    extra_bat = _rows(_BAT_FIELDS, [(15, 2020, 300, 270, 90, 20, 2, 15, 25, 3, 2)])
    lg_bat = _league_at("bat", 2024)
    bat = _run("bat", _params(weights=weights), rows=[*BAT_ROWS, *extra_bat])[15]
    names = [*BAT_RATE_COLUMNS, "obp", "slg", "ops", "hr_per_pa", "lg_ops"]
    expected = {**_pick(lg_bat, BAT_RATE_COLUMNS), **_bat_derived(_pick(lg_bat, BAT_RATE_COLUMNS))}
    expected["lg_ops"] = float(lg_bat["ops"])
    assert _pick(bat, names) == pytest.approx(expected, rel=1e-12)
    # 리그 표가 서로 맞으므로 리그 비율로 계산한 obp·ops는 리그 obp·ops와 같다
    assert float(bat["obp"]) == pytest.approx(float(lg_bat["obp"]), rel=1e-12)
    assert float(bat["ops"]) == pytest.approx(float(lg_bat["ops"]), rel=1e-12)

    extra_pit = _rows(_PIT_FIELDS, [(25, 2020, 200, 150, 10, 70, 10, 1, 2)])
    for era_method in ERA_METHODS:
        params = _params(weights=weights, era_method=era_method)
        pit = _run("pit", params, rows=[*PIT_ROWS, *extra_pit])[25]
        # er_per_out = R_er·(1/6)/R_er, bf_per_out = R_er·(4/3)/R_er → era = 4.5 (두 방식 모두)
        assert _pick(pit, [*PIT_RATE_COLUMNS, "er_per_out", "era", "k_bb_pct"]) == (
            pytest.approx(
                {
                    "r_so": 0.225,
                    "r_bb": 0.085,
                    "r_hbp": 0.011,
                    "r_hr": 0.03,
                    "er_per_out": 1 / 6,
                    "era": 4.5,
                    "k_bb_pct": 14.0,
                },
                rel=1e-12,
            )
        ), era_method


# ---------------------------------------------------------------------------
# 2020 단축 시즌 (D9)
# ---------------------------------------------------------------------------


def test_models_weighted_short_season_uses_actual_sample() -> None:
    # Y = 2022: 2021(Y−1, w=5) · 2020(Y−2, w=4). 2020은 실제 표본으로 가중한다(D9).
    target = 2022
    bat_rows = _rows(
        _BAT_FIELDS,
        [
            (31, 2020, 200, 180, 54, 10, 1, 9, 16, 2, 2),  # pa_adj = 540, h1 34
            (31, 2021, 600, 540, 150, 30, 3, 21, 50, 6, 4),  # h1 96
        ],
    )
    bat = _run("bat", _params(same_r=0.0), rows=bat_rows, target=target)[31]
    # Σw·pa = 5·600 + 4·200 = 3800 (pa_adj로 하면 5·600 + 4·540 = 5160 → r_hr 141/5160)
    assert _pick(bat, ["r_h1", "r_hr", "r_ab"]) == pytest.approx(
        {
            "r_h1": (5 * 96 + 4 * 34) / 3800,  # 616/3800
            "r_hr": (5 * 21 + 4 * 9) / 3800,  # 141/3800 ≈ .0371
            "r_ab": (5 * 540 + 4 * 180) / 3800,  # 3420/3800 = .9
        },
        rel=1e-12,
    )
    # lg_ops = Y−1 = 2021 리그 ops
    lg_2021 = _league_at("bat", 2021)
    assert float(bat["lg_ops"]) == pytest.approx(float(lg_2021["ops"]), rel=1e-12)
    assert int(bat["target_season"]) == target

    pit_rows = _rows(
        _PIT_FIELDS,
        [
            (41, 2020, 200, 150, 20, 50, 15, 2, 6),  # outs_adj = 405
            (41, 2021, 600, 450, 50, 150, 40, 4, 15),
        ],
    )
    pit = _run("pit", _params(same_r=0.0), rows=pit_rows, target=target)[41]
    # Σw·outs = 5·450 + 4·150 = 2850 (outs_adj로 하면 2250 + 4·405 = 3870)
    # Σw·er = 5·50 + 4·20 = 330 → er_per_out = 330/2850, era = 27·330/2850 ≈ 3.1263
    assert float(pit["er_per_out"]) == pytest.approx(330 / 2850, rel=1e-12)
    assert float(pit["era"]) == pytest.approx(27 * 330 / 2850, rel=1e-12)
    # Σw·bf = 5·600 + 4·200 = 3800, Σw·so = 5·150 + 4·50 = 950 → r_so = .25
    assert float(pit["r_so"]) == pytest.approx(950 / 3800, rel=1e-12)
    lg_2021_pit = _league_at("pit", 2021)
    assert float(pit["lg_era"]) == pytest.approx(float(lg_2021_pit["era"]), rel=1e-12)


# ---------------------------------------------------------------------------
# 기간 밖 행
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", c.ROLES)
def test_models_weighted_ignores_rows_before_window(role: Role) -> None:
    # 2019(Y−6) 행이 있든 없든 결과가 같다 — 같은 대상 표를 넣는다
    rows = BAT_ROWS if role == "bat" else PIT_ROWS
    seasons = _seasons(role, rows)
    cohort = _cohort(role, seasons)
    league = _league(role)
    params = _params(era_method="fip")
    with_old = weighted_rates(cohort, seasons, league, role, params)
    trimmed = _seasons(role, [r for r in rows if r["season"] != 2019])
    without_old = weighted_rates(cohort, trimmed, league, role, params)
    assert_frame_equal(with_old.reset_index(drop=True), without_old.reset_index(drop=True))


def test_models_weighted_uses_only_cohort_window() -> None:
    # 대상 표의 기간을 2022~2024로 좁히면 2021 행이 정제 표에 있어도 쓰지 않는다
    seasons = _seasons("bat")
    cohort = _cohort("bat", seasons)
    narrowed = cohort["player_id"] == 11
    cohort.loc[narrowed, ["window_first", "n_window", "input_first", "n_seasons", "sample"]] = [
        2022,
        3,
        2023,
        2,
        1100,
    ]
    assert c.check_frame(cohort, c.COHORTS) == []
    league = _league("bat")
    params = _params()
    out = weighted_rates(cohort, seasons, league, "bat", params)
    row = _records(out, "player_id")[11]
    # 2024(w=5) · 2023(w=4)만: Σw·pa = 5000, Σw·h1 = 450 + 300 = 750, Σw·hr = 135 + 72 = 207
    assert float(row["r_h1"]) == pytest.approx((750 + 100 * 0.14) / (5000 + 100), rel=1e-12)
    assert float(row["r_hr"]) == pytest.approx((207 + 400 * 0.03) / (5000 + 400), rel=1e-12)
    # 2021 행을 지운 정제 표와 결과가 같다
    trimmed = _seasons("bat", [r for r in BAT_ROWS if (r["player_id"], r["season"]) != (11, 2021)])
    assert_frame_equal(
        out.reset_index(drop=True),
        weighted_rates(cohort, trimmed, league, "bat", params).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 결과 형식
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", c.ROLES)
def test_models_weighted_output_contract(role: Role) -> None:
    seasons = _seasons(role)
    cohort = _cohort(role, seasons)
    shuffled = cohort.iloc[::-1].reset_index(drop=True)  # 입력 순서를 뒤집는다
    out = weighted_rates(shuffled, seasons, _league(role), role, _params(era_method="fip"))
    assert c.check_frame(out, RATE_SPECS[role]) == []
    expected_ids = [11, 12, 13] if role == "bat" else [21, 22]
    assert out["player_id"].tolist() == expected_ids  # 대상 표와 같은 선수, player_id 순
    assert set(out["player_id"]) == set(cohort["player_id"])
    assert out["target_season"].tolist() == [TARGET] * len(expected_ids)
    assert not bool(out.isna().any().any())


def test_models_weighted_zero_denominator_stays_finite() -> None:
    # R = 0이고 분모가 0이면 0/0·x/0이 된다. 결과는 결측·무한 없이 Y−1 리그 값으로 둔다
    # (BAT_RATES·PIT_RATES: "결측 없음 — 기록이 없으면 리그 평균").
    weights = (5.0, 4.0, 3.0, 2.0, 0.0)  # Y−5 = 2020 가중치 0
    zero_r = _params(weights=weights, same_r=0.0)
    lg_bat = _league_at("bat", 2024)
    # 타자 15: 2020에만 뛰었다 → Σw·pa = 0, R = 0 → 모든 성분이 2024 리그 값
    extra_bat = _rows(_BAT_FIELDS, [(15, 2020, 300, 270, 90, 20, 2, 15, 25, 3, 2)])
    seasons = _seasons("bat", [*BAT_ROWS, *extra_bat])
    out = weighted_rates(_cohort("bat", seasons), seasons, _league("bat"), "bat", zero_r)
    assert c.check_frame(out, c.BAT_RATES) == []
    assert bool(out.drop(columns=["player_id", "target_season"]).map(_finite).all().all())
    row = _records(out, "player_id")[15]
    assert _pick(row, BAT_RATE_COLUMNS) == _pick(lg_bat, BAT_RATE_COLUMNS)
    # 리그 표가 서로 맞아서 리그 비율로 계산한 obp·slg는 리그 obp·slg다
    assert _pick(row, ["obp", "slg", "ops"]) == pytest.approx(
        _pick(lg_bat, ["obp", "slg", "ops"]), rel=1e-12
    )

    # 투수 26: 2024에 3타자 상대(볼넷 2, 홈런 1, 자책 2) 아웃 0 → Σw·outs = 0, R_er = 0
    # er_per_out = 10/0 이 아니라 2024 리그 값 1/6, era("er") = 4.5. 성분 비율은 bf로 정상 계산
    extra_pit = _rows(_PIT_FIELDS, [(26, 2024, 3, 0, 2, 0, 2, 0, 1)])
    pit_seasons = _seasons("pit", [*PIT_ROWS, *extra_pit])
    pit_cohort = _cohort("pit", pit_seasons)
    for era_method in ERA_METHODS:
        params = dataclasses.replace(zero_r, era_method=era_method)
        out = weighted_rates(pit_cohort, pit_seasons, _league("pit"), "pit", params)
        assert c.check_frame(out, c.PIT_RATES) == []
        assert bool(out.drop(columns=["player_id", "target_season"]).map(_finite).all().all())
        row = _records(out, "player_id")[26]
        # Σw·bf = 15, Σw·bb = 10, Σw·hr = 5 → r_bb = 2/3, r_hr = 1/3
        assert _pick(row, PIT_RATE_COLUMNS) == pytest.approx(
            {"r_so": 0.0, "r_bb": 2 / 3, "r_hbp": 0.0, "r_hr": 1 / 3}, rel=1e-12
        )
        assert float(row["er_per_out"]) == pytest.approx(1 / 6, rel=1e-12)
        if era_method == "er":
            assert float(row["era"]) == pytest.approx(4.5, rel=1e-12)


def _finite(value: Any) -> bool:
    return bool(pd.notna(value)) and abs(float(value)) != float("inf")


# ---------------------------------------------------------------------------
# 누설 방지 (N2)
# ---------------------------------------------------------------------------


def _poison_rows(role: Role, pids: Sequence[int], seasons: Sequence[int]) -> list[dict[str, int]]:
    """모든 출전량·성분 열을 POISON으로 채운 행."""
    skip = {"player_id", "season", "age", "n_teams"}
    counts = [col.name for col in SPECS[role].columns if col.kind == "int" and col.name not in skip]
    return [
        {"player_id": pid, "season": season, **dict.fromkeys(counts, POISON)}
        for pid in pids
        for season in seasons
    ]


@pytest.mark.parametrize("weights", [WEIGHTS, FRACTIONAL_WEIGHTS], ids=["int", "frac"])
@pytest.mark.parametrize("era_method", ERA_METHODS)
@pytest.mark.parametrize("target", [2023, TARGET])
@pytest.mark.parametrize("role", c.ROLES)
def test_models_weighted_no_leakage(
    role: Role, target: int, era_method: str, weights: tuple[float, ...]
) -> None:
    rows = BAT_ROWS if role == "bat" else PIT_ROWS
    totals = BAT_LEAGUE_TOTALS if role == "bat" else PIT_LEAGUE_TOTALS
    clean_rows = [r for r in rows if r["season"] < target]
    clean = _seasons(role, clean_rows)
    clean_league = _league(role, {s: t for s, t in totals.items() if s < target})
    cohort = _cohort(role, clean, target)
    params = _params(weights=weights, era_method=era_method)
    expected = weighted_rates(cohort, clean, clean_league, role, params)

    # 독: 모든 선수(+ 새 선수)의 target~2026 행, target 이후 리그 행. 앞에 둬서 입력 순서도 섞는다
    after = list(range(target, c.SEASON_LAST + 1))
    pids = sorted({r["player_id"] for r in rows} | {NEW_PLAYER})
    poisoned = pd.concat(
        [_seasons(role, _poison_rows(role, pids, after)), clean], ignore_index=True
    )
    assert c.check_frame(poisoned, SPECS[role]) == []
    poisoned_league = pd.concat(
        [
            _league(role, dict.fromkeys([*after, c.TARGET_SEASON], POISON_LEAGUE_TOTALS[role])),
            clean_league,
        ],
        ignore_index=True,
    )
    assert c.check_frame(poisoned_league, LEAGUE_SPECS[role]) == []
    got = weighted_rates(cohort, poisoned, poisoned_league, role, params)

    assert NEW_PLAYER not in set(got["player_id"])
    # 비트까지 같아야 한다 — 독 행이 걸러지고 남은 행의 합이 그대로여야 한다
    assert_frame_equal(
        got.reset_index(drop=True), expected.reset_index(drop=True), check_exact=True
    )


@pytest.mark.parametrize("role", c.ROLES)
def test_models_weighted_no_leakage_with_cohort_from_full_table(role: Role) -> None:
    # 파이프라인처럼 대상 표를 2026까지 있는 전체 정제 표로 만든다 (build_cohort는 Y 이후 행을 무시한다).
    # 정제 표에 Y 이후 독 행이 있어도 Y 이전 행만 있을 때와 결과가 비트까지 같다.
    rows = BAT_ROWS if role == "bat" else PIT_ROWS
    clean = _seasons(role, [r for r in rows if r["season"] < TARGET])
    pids = sorted({r["player_id"] for r in rows} | {NEW_PLAYER})
    poison = _seasons(role, _poison_rows(role, pids, range(TARGET, c.SEASON_LAST + 1)))
    full = pd.concat([clean, poison], ignore_index=True)
    league = _league(role)
    params = _params(weights=FRACTIONAL_WEIGHTS, era_method="fip")
    got = weighted_rates(_cohort(role, full), full, league, role, params)
    expected = weighted_rates(_cohort(role, clean), clean, league, role, params)
    assert_frame_equal(got, expected, check_exact=True)


@pytest.mark.parametrize("era_method", ERA_METHODS)
@pytest.mark.parametrize("role", c.ROLES)
def test_models_weighted_row_order_does_not_change_bits(role: Role, era_method: str) -> None:
    # 정제 표의 행 순서를 바꿔도 결과가 비트까지 같다 (정수가 아닌 가중치는 더하는 순서에 민감하다).
    # 2019 행(기간 밖)도 섞어 둔다.
    rows = BAT_ROWS if role == "bat" else PIT_ROWS
    seasons = _seasons(role, rows)
    cohort = _cohort(role, seasons)
    league = _league(role)
    params = _params(weights=FRACTIONAL_WEIGHTS, era_method=era_method)
    expected = weighted_rates(cohort, seasons, league, role, params)
    for seed in range(20):
        shuffled = seasons.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        got = weighted_rates(cohort, shuffled, league, role, params)
        assert_frame_equal(got, expected, check_exact=True, obj=f"seed {seed}")


# ---------------------------------------------------------------------------
# 오류
# ---------------------------------------------------------------------------

_BASE = _params()
INVALID_PARAMS: dict[str, Params] = {
    "bat_weights_len4": dataclasses.replace(
        _BASE, bat=RateParams(weights=(5.0, 4.0, 3.0, 2.0), regress=dict(BAT_R))
    ),
    "bat_weights_all_zero": dataclasses.replace(
        _BASE, bat=RateParams(weights=(0.0,) * 5, regress=dict(BAT_R))
    ),
    "bat_regress_missing_key": dataclasses.replace(
        _BASE,
        bat=RateParams(weights=WEIGHTS, regress={k: v for k, v in BAT_R.items() if k != "sf"}),
    ),
    "pit_regress_missing_er": dataclasses.replace(
        _BASE,
        pit=RateParams(weights=WEIGHTS, regress={k: v for k, v in PIT_R.items() if k != "er"}),
    ),
    "pit_regress_negative": dataclasses.replace(
        _BASE, pit=RateParams(weights=WEIGHTS, regress={**PIT_R, "so": -1.0})
    ),
    "pit_weights_negative": dataclasses.replace(
        _BASE, pit=RateParams(weights=(5.0, 4.0, -3.0, 2.0, 1.0), regress=dict(PIT_R))
    ),
    "era_method_unknown": dataclasses.replace(_BASE, era_method="xera"),
    "empty_version": dataclasses.replace(_BASE, version=""),
}


@pytest.mark.parametrize("role", c.ROLES)
@pytest.mark.parametrize("case", sorted(INVALID_PARAMS))
def test_models_weighted_invalid_params_raise(case: str, role: Role) -> None:
    params = INVALID_PARAMS[case]
    with pytest.raises(ValueError):
        validate_params(params)  # 전제: 형식 검사에서 걸리는 파라미터
    seasons = _seasons(role)
    with pytest.raises(ValueError):
        weighted_rates(_cohort(role, seasons), seasons, _league(role), role, params)


def test_models_weighted_role_mismatch_raises() -> None:
    bat, pit = _seasons("bat"), _seasons("pit")
    bat_cohort, pit_cohort = _cohort("bat", bat), _cohort("pit", pit)
    bat_league, pit_league = _league("bat"), _league("pit")
    params = _params()
    # 대상 표의 역할이 인자와 다르다
    with pytest.raises(ValueError):
        weighted_rates(pit_cohort, bat, bat_league, "bat", params)
    with pytest.raises(ValueError):
        weighted_rates(bat_cohort, pit, pit_league, "pit", params)
    # 대상 표에 두 역할이 섞였다
    mixed = pd.concat([bat_cohort, pit_cohort], ignore_index=True)
    assert c.check_frame(mixed, c.COHORTS) == []
    with pytest.raises(ValueError):
        weighted_rates(mixed, bat, bat_league, "bat", params)
    # 정제 표·리그 표가 다른 역할의 계약이다
    with pytest.raises(ValueError):
        weighted_rates(bat_cohort, pit, bat_league, "bat", params)
    with pytest.raises(ValueError):
        weighted_rates(bat_cohort, bat, pit_league, "bat", params)
    with pytest.raises(ValueError):
        weighted_rates(pit_cohort, pit, bat_league, "pit", params)


@pytest.mark.parametrize("role", c.ROLES)
def test_models_weighted_empty_cohort_raises(role: Role) -> None:
    # 빈 대상 표에는 역할·목표 시즌이 하나도 없다 → "하나가 아니다"
    seasons = _seasons(role)
    empty = _cohort(role, seasons).iloc[0:0]
    assert c.check_frame(empty, c.COHORTS) == []
    with pytest.raises(ValueError):
        weighted_rates(empty, seasons, _league(role), role, _params())


# 대상 표의 기간을 손으로 바꾼 경우 — 계약(COHORTS: window_last = Y−1, window_first ≥ Y−WINDOW) 위반
BAD_WINDOWS: dict[str, dict[str, int]] = {
    "last_is_target": {"window_last": TARGET, "n_window": 6},  # Y 행이 기간 안에 든다 (N2)
    "last_before_prev": {"window_last": TARGET - 2, "n_window": 4},
    "first_before_window": {"window_first": TARGET - c.WINDOW - 1, "n_window": 6},  # Y−6
    "first_after_last": {"window_first": TARGET, "n_window": 0},
}


@pytest.mark.parametrize("role", c.ROLES)
@pytest.mark.parametrize("case", sorted(BAD_WINDOWS))
def test_models_weighted_cohort_window_outside_range_raises(case: str, role: Role) -> None:
    seasons = _seasons(role)
    cohort = _cohort(role, seasons)
    first = cohort.index[0]
    for column, value in BAD_WINDOWS[case].items():
        cohort.loc[first, column] = value
    assert c.check_frame(cohort, c.COHORTS) == []
    with pytest.raises(ValueError):
        weighted_rates(cohort, seasons, _league(role), role, _params())


@pytest.mark.parametrize("role", c.ROLES)
def test_models_weighted_mixed_targets_raise(role: Role) -> None:
    seasons = _seasons(role)
    mixed = pd.concat(
        [_cohort(role, seasons, 2024), _cohort(role, seasons, TARGET)], ignore_index=True
    )
    assert c.check_frame(mixed, c.COHORTS) == []
    with pytest.raises(ValueError):
        weighted_rates(mixed, seasons, _league(role), role, _params())


@pytest.mark.parametrize("role", c.ROLES)
def test_models_weighted_contract_violation_raises(role: Role) -> None:
    seasons = _seasons(role)
    cohort = _cohort(role, seasons)
    league = _league(role)
    params = _params()
    with pytest.raises(ValueError):
        weighted_rates(cohort.drop(columns=["n_seasons"]), seasons, league, role, params)
    count = "pa" if role == "bat" else "bf"
    with pytest.raises(ValueError):
        weighted_rates(cohort, seasons.astype({count: "float64"}), league, role, params)
    with pytest.raises(ValueError):
        weighted_rates(cohort, seasons, league.drop(columns=[league.columns[-1]]), role, params)


@pytest.mark.parametrize("role", c.ROLES)
def test_models_weighted_missing_prev_league_raises_key_error(role: Role) -> None:
    # 리그 표에 2023·2025는 있지만 Y−1 = 2024가 없다 — 가까운 시즌으로 대신하지 않는다
    totals = BAT_LEAGUE_TOTALS if role == "bat" else PIT_LEAGUE_TOTALS
    league = _league(role, {2023: totals[2023], 2025: totals[2024]})
    seasons = _seasons(role)
    with pytest.raises(KeyError):
        weighted_rates(_cohort(role, seasons), seasons, league, role, _params())
