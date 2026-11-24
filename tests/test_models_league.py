"""리그 평균(models/league.py) 테스트 (SPEC F6·F7·D9·D11·N2, `contracts.LEAGUE_BAT`·`LEAGUE_PIT`).

입력은 이 파일에서 만든 작은 합성 정제 표다. `_frame`이 계약의 모든 열을 유효한 값으로 채운다.
리그 평균은 그 시즌 **모든 행의 합**으로 계산한다 — 선수 비율의 평균이 아니고, 행의 obp·slg·ops 열도
쓰지 않는다(그래서 합성 행의 obp·slg·ops에는 일부러 엉뚱한 값을 넣었다). 기대값은 아래 주석에 손으로
계산해 적었다.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlbproj import contracts as c
from mlbproj.models.league import league_batting, league_pitching, league_row

# ---------------------------------------------------------------------------
# 도우미 — 계약 형식의 합성 표
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {"league": c.LEAGUE, "team_name": "TST", "age": 27, "n_teams": 1}
_ADJUSTED = {"pa_adj": "pa", "outs_adj": "outs"}


def _dtype(col: c.Column) -> str:
    if col.kind == "int":
        return "Int64" if col.nullable else "int64"
    if col.kind == "float":
        return "float64"
    if col.kind == "bool":
        return "bool"
    return "str"


def _value(col: c.Column, row: Mapping[str, Any]) -> Any:
    """행에 없는 열의 기본값. 정수 0, 결측 허용 열은 결측, 2020 출전량은 ×162/60."""
    name = col.name
    if name in row:
        return row[name]
    if name in _DEFAULTS:
        return _DEFAULTS[name]
    if name == "is_short_season":
        return int(row["season"]) == c.SHORT_SEASON
    if name in _ADJUSTED:
        scale = c.SHORT_SEASON_SCALE if int(row["season"]) == c.SHORT_SEASON else 1.0
        return float(row.get(_ADJUSTED[name], 0)) * scale
    if col.kind == "int" and not col.nullable:
        return 0
    if col.nullable:
        return None
    raise AssertionError(f"채울 수 없는 열: {name}")


def _frame(spec: c.TableSpec, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """계약의 모든 열을 채운 표 (`check_frame` 통과를 확인한다)."""
    df = pd.DataFrame(
        {
            col.name: pd.Series([_value(col, row) for row in rows], dtype=_dtype(col))
            for col in spec.columns
        }
    )
    assert c.check_frame(df, spec) == []
    return df


def _bat(
    pid: int,
    season: int,
    pa: int,
    ab: int,
    h: int,
    b2: int,
    b3: int,
    hr: int,
    bb: int,
    hbp: int,
    sf: int,
    sh: int,
    **extra: Any,
) -> dict[str, Any]:
    """타자 한 시즌 행. obp·slg·ops 열에는 엉뚱한 값을 넣는다 (리그 평균은 이 열을 쓰면 안 된다)."""
    assert ab + bb + hbp + sf + sh == pa, "합성 행의 타석 항등식"
    row: dict[str, Any] = {
        "player_id": pid,
        "season": season,
        "pa": pa,
        "ab": ab,
        "h": h,
        "b2": b2,
        "b3": b3,
        "hr": hr,
        "bb": bb,
        "hbp": hbp,
        "sf": sf,
        "sh": sh,
        "obp": 0.999,
        "slg": 0.999,
        "ops": 1.998,
    }
    row.update(extra)
    return row


def _pit(
    pid: int,
    season: int,
    outs: int,
    bf: int,
    er: int,
    hr: int,
    bb: int,
    hbp: int,
    so: int,
    **extra: Any,
) -> dict[str, Any]:
    """투수 한 시즌 행."""
    row: dict[str, Any] = {
        "player_id": pid,
        "season": season,
        "outs": outs,
        "bf": bf,
        "er": er,
        "hr": hr,
        "bb": bb,
        "hbp": hbp,
        "so": so,
        "g": 10,
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# 합성 시즌 표와 손 계산 기대값
# ---------------------------------------------------------------------------

# 입력 순서를 섞어 둔다 (결과는 season 순이어야 한다)
BAT_ROWS: list[dict[str, Any]] = [
    # 2024: 세 명
    _bat(1, 2024, pa=200, ab=170, h=50, b2=10, b3=2, hr=8, bb=20, hbp=5, sf=3, sh=2),
    # 2023: 두 명
    _bat(1, 2023, pa=100, ab=80, h=24, b2=5, b3=1, hr=3, bb=12, hbp=3, sf=2, sh=3),
    _bat(3, 2024, pa=40, ab=36, h=8, b2=1, b3=1, hr=0, bb=3, hbp=1, sf=0, sh=0, obp=None),
    # 2020 (단축 시즌): pa_adj = 60 × 162/60 = 162 이지만 리그 평균은 실제 pa(60)로 낸다 (D9)
    _bat(5, 2020, pa=60, ab=50, h=15, b2=3, b3=0, hr=2, bb=8, hbp=1, sf=1, sh=0, pa_adj=162.0),
    _bat(2, 2023, pa=50, ab=45, h=10, b2=2, b3=0, hr=2, bb=4, hbp=0, sf=1, sh=0, ops=None),
    _bat(4, 2024, pa=10, ab=9, h=2, b2=0, b3=0, hr=1, bb=1, hbp=0, sf=0, sh=0),
]

# 2020: Σpa 60, Σab 50, Σh 15, Σb2 3, Σb3 0, Σhr 2, Σbb 8, Σhbp 1, Σsf 1
#   h1 = 15 − 3 − 0 − 2 = 10
#   obp = (15 + 8 + 1) / (50 + 8 + 1 + 1) = 24/60 = 0.4
#   slg = (10 + 2·3 + 3·0 + 4·2) / 50 = 24/50 = 0.48
# 2023: Σpa 100+50 = 150, Σab 80+45 = 125, Σh 24+10 = 34, Σb2 5+2 = 7, Σb3 1+0 = 1, Σhr 3+2 = 5,
#       Σbb 12+4 = 16, Σhbp 3+0 = 3, Σsf 2+1 = 3
#   h1 = 34 − 7 − 1 − 5 = 21
#   obp = (34 + 16 + 3) / (125 + 16 + 3 + 3) = 53/147
#   slg = (21 + 2·7 + 3·1 + 4·5) / 125 = 58/125
# 2024: Σpa 200+40+10 = 250, Σab 170+36+9 = 215, Σh 50+8+2 = 60, Σb2 10+1+0 = 11,
#       Σb3 2+1+0 = 3, Σhr 8+0+1 = 9, Σbb 20+3+1 = 24, Σhbp 5+1+0 = 6, Σsf 3+0+0 = 3
#   h1 = 60 − 11 − 3 − 9 = 37
#   obp = (60 + 24 + 6) / (215 + 24 + 6 + 3) = 90/248
#   slg = (37 + 2·11 + 3·3 + 4·9) / 215 = 104/215
EXPECTED_BAT: dict[int, dict[str, float]] = {
    2020: {
        "pa": 60,
        "r_h1": 10 / 60,
        "r_b2": 3 / 60,
        "r_b3": 0 / 60,
        "r_hr": 2 / 60,
        "r_bb": 8 / 60,
        "r_hbp": 1 / 60,
        "r_sf": 1 / 60,
        "r_ab": 50 / 60,
        "obp": 0.4,
        "slg": 0.48,
        "ops": 0.88,
        "hr_per_pa": 2 / 60,
    },
    2023: {
        "pa": 150,
        "r_h1": 21 / 150,
        "r_b2": 7 / 150,
        "r_b3": 1 / 150,
        "r_hr": 5 / 150,
        "r_bb": 16 / 150,
        "r_hbp": 3 / 150,
        "r_sf": 3 / 150,
        "r_ab": 125 / 150,
        "obp": 53 / 147,
        "slg": 58 / 125,
        "ops": 53 / 147 + 58 / 125,
        "hr_per_pa": 5 / 150,
    },
    2024: {
        "pa": 250,
        "r_h1": 37 / 250,
        "r_b2": 11 / 250,
        "r_b3": 3 / 250,
        "r_hr": 9 / 250,
        "r_bb": 24 / 250,
        "r_hbp": 6 / 250,
        "r_sf": 3 / 250,
        "r_ab": 215 / 250,
        "obp": 90 / 248,
        "slg": 104 / 215,
        "ops": 90 / 248 + 104 / 215,
        "hr_per_pa": 9 / 250,
    },
}

PIT_ROWS: list[dict[str, Any]] = [
    _pit(11, 2024, outs=270, bf=360, er=30, hr=9, bb=25, hbp=2, so=90),
    _pit(11, 2023, outs=300, bf=420, er=40, hr=12, bb=35, hbp=5, so=100),
    _pit(13, 2024, outs=90, bf=130, er=14, hr=3, bb=12, hbp=1, so=25),
    # 아웃 0인 투수 행도 합에 들어간다
    _pit(14, 2024, outs=0, bf=3, er=2, hr=1, bb=1, hbp=0, so=0),
    # 2020: outs_adj = 90 × 162/60 = 243 이지만 리그 평균은 실제 outs(90)로 낸다 (D9)
    _pit(15, 2020, outs=90, bf=120, er=10, hr=3, bb=9, hbp=1, so=30, outs_adj=243.0),
    _pit(12, 2023, outs=150, bf=210, er=25, hr=8, bb=20, hbp=3, so=40),
]

# 2020: Σbf 120, Σouts 90, Σer 10, Σhr 3, Σbb 9, Σhbp 1, Σso 30
#   era = 27 × 10/90 = 3.0, k_bb_pct = 100 × (30 − 9)/120 = 17.5
#   fip_const = 3.0 − (13·3 + 3·(9+1) − 2·30) / (90/3) = 3.0 − 9/30 = 2.7
# 2023: Σbf 420+210 = 630, Σouts 300+150 = 450, Σer 40+25 = 65, Σhr 12+8 = 20,
#       Σbb 35+20 = 55, Σhbp 5+3 = 8, Σso 100+40 = 140
#   era = 27 × 65/450 = 3.9, k_bb_pct = 100 × (140 − 55)/630 = 8500/630
#   fip_const = 3.9 − (13·20 + 3·(55+8) − 2·140) / (450/3) = 3.9 − 169/150
#   bf_per_out = 630/450 = 1.4
# 2024: Σbf 360+130+3 = 493, Σouts 270+90+0 = 360, Σer 30+14+2 = 46, Σhr 9+3+1 = 13,
#       Σbb 25+12+1 = 38, Σhbp 2+1+0 = 3, Σso 90+25+0 = 115
#   era = 27 × 46/360 = 3.45, k_bb_pct = 100 × (115 − 38)/493 = 7700/493
#   fip_const = 3.45 − (13·13 + 3·(38+3) − 2·115) / (360/3) = 3.45 − 62/120
EXPECTED_PIT: dict[int, dict[str, float]] = {
    2020: {
        "bf": 120,
        "outs": 90,
        "r_so": 30 / 120,
        "r_bb": 9 / 120,
        "r_hbp": 1 / 120,
        "r_hr": 3 / 120,
        "er_per_out": 10 / 90,
        "era": 3.0,
        "k_bb_pct": 17.5,
        "fip_const": 2.7,
        "bf_per_out": 120 / 90,
    },
    2023: {
        "bf": 630,
        "outs": 450,
        "r_so": 140 / 630,
        "r_bb": 55 / 630,
        "r_hbp": 8 / 630,
        "r_hr": 20 / 630,
        "er_per_out": 65 / 450,
        "era": 3.9,
        "k_bb_pct": 8500 / 630,
        "fip_const": 3.9 - 169 / 150,
        "bf_per_out": 1.4,
    },
    2024: {
        "bf": 493,
        "outs": 360,
        "r_so": 115 / 493,
        "r_bb": 38 / 493,
        "r_hbp": 3 / 493,
        "r_hr": 13 / 493,
        "er_per_out": 46 / 360,
        "era": 3.45,
        "k_bb_pct": 7700 / 493,
        "fip_const": 3.45 - 62 / 120,
        "bf_per_out": 493 / 360,
    },
}


def _assert_matches(out: pd.DataFrame, expected: Mapping[int, Mapping[str, float]]) -> None:
    assert out["season"].tolist() == sorted(expected)
    for season, values in expected.items():
        row = out[out["season"] == season]
        assert len(row) == 1, f"{season} 행은 하나"
        for name, value in values.items():
            assert row[name].iloc[0] == pytest.approx(value, rel=1e-12), f"{season} {name}"


# ---------------------------------------------------------------------------
# 공식
# ---------------------------------------------------------------------------


def test_models_league_batting_sums_per_season() -> None:
    out = league_batting(_frame(c.BATTING_SEASONS, BAT_ROWS))
    assert c.check_frame(out, c.LEAGUE_BAT) == []
    _assert_matches(out, EXPECTED_BAT)
    # pa는 정수 합 (2020도 환산 전 실제 타석)
    assert out["pa"].tolist() == [60, 150, 250]


def test_models_league_pitching_sums_per_season() -> None:
    out = league_pitching(_frame(c.PITCHING_SEASONS, PIT_ROWS))
    assert c.check_frame(out, c.LEAGUE_PIT) == []
    _assert_matches(out, EXPECTED_PIT)
    # bf·outs는 정수 합 (2020도 환산 전 실제 아웃)
    assert out["bf"].tolist() == [120, 630, 493]
    assert out["outs"].tolist() == [90, 450, 360]


def test_models_league_batting_single_row_season_equals_player() -> None:
    # 한 명뿐인 시즌의 리그 평균은 그 선수 기록 그대로다 (평균의 평균이 아님을 반대로 확인)
    row = _bat(9, 2022, pa=10, ab=8, h=4, b2=1, b3=0, hr=1, bb=2, hbp=0, sf=0, sh=0)
    out = league_batting(_frame(c.BATTING_SEASONS, [row]))
    # h1 = 4 − 1 − 0 − 1 = 2, obp = (4+2+0)/(8+2+0+0) = 0.6, slg = (2 + 2·1 + 4·1)/8 = 1.0
    assert out["season"].tolist() == [2022]
    assert out["r_h1"].iloc[0] == pytest.approx(0.2)
    assert out["obp"].iloc[0] == pytest.approx(0.6)
    assert out["slg"].iloc[0] == pytest.approx(1.0)
    assert out["ops"].iloc[0] == pytest.approx(1.6)


# ---------------------------------------------------------------------------
# 오류
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_rows",
    [
        # 2022 시즌 Σpa = 0 (Σab도 0)
        [_bat(7, 2022, pa=0, ab=0, h=0, b2=0, b3=0, hr=0, bb=0, hbp=0, sf=0, sh=0)],
        # 2022 시즌 Σpa = 3 > 0 이지만 Σab = 0 (볼넷·희생번트만)
        [
            _bat(7, 2022, pa=2, ab=0, h=0, b2=0, b3=0, hr=0, bb=2, hbp=0, sf=0, sh=0),
            _bat(8, 2022, pa=1, ab=0, h=0, b2=0, b3=0, hr=0, bb=0, hbp=0, sf=0, sh=1),
        ],
    ],
    ids=["pa_zero", "ab_zero"],
)
def test_models_league_batting_zero_denominator_raises(bad_rows: list[dict[str, Any]]) -> None:
    # 다른 시즌(2023·2024)이 정상이어도 한 시즌의 분모가 0이면 오류다. 메시지에 그 시즌과 분모를 적는다
    with pytest.raises(ValueError, match=r"2022\((pa·)?ab\)"):
        league_batting(_frame(c.BATTING_SEASONS, BAT_ROWS + bad_rows))


@pytest.mark.parametrize(
    "bad_rows",
    [
        # 2022 시즌 Σbf = 0 (Σouts도 0)
        [_pit(17, 2022, outs=0, bf=0, er=0, hr=0, bb=0, hbp=0, so=0)],
        # 2022 시즌 Σbf = 3 > 0 이지만 Σouts = 0
        [_pit(17, 2022, outs=0, bf=3, er=3, hr=0, bb=3, hbp=0, so=0)],
    ],
    ids=["bf_zero", "outs_zero"],
)
def test_models_league_pitching_zero_denominator_raises(bad_rows: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError, match=r"2022\((bf·)?outs\)"):
        league_pitching(_frame(c.PITCHING_SEASONS, PIT_ROWS + bad_rows))


def test_models_league_rejects_contract_violation() -> None:
    bat = _frame(c.BATTING_SEASONS, BAT_ROWS)
    pit = _frame(c.PITCHING_SEASONS, PIT_ROWS)
    # 열이 빠짐 (메시지에 어느 표인지 적는다)
    with pytest.raises(ValueError, match="타자 정제 표 계약 위반"):
        league_batting(bat.drop(columns=["hbp"]))
    with pytest.raises(ValueError):
        league_pitching(pit.drop(columns=["er"]))
    # 다른 역할의 표
    with pytest.raises(ValueError):
        league_batting(pit)
    with pytest.raises(ValueError):
        league_pitching(bat)
    # 타입 위반 (정수 열이 실수)
    with pytest.raises(ValueError):
        league_batting(bat.astype({"pa": "float64"}))
    # 키 중복 (같은 선수·시즌 두 행)
    with pytest.raises(ValueError):
        league_pitching(pd.concat([pit, pit.iloc[[0]]], ignore_index=True))


# ---------------------------------------------------------------------------
# league_row
# ---------------------------------------------------------------------------


def _league_bat_frame(seasons: Sequence[int]) -> pd.DataFrame:
    """손으로 만든 LEAGUE_BAT 표 — 시즌마다 값이 다르다 (league_batting과 따로 시험한다)."""
    rows: list[dict[str, Any]] = []
    for season in seasons:
        k = season - 2000
        row: dict[str, Any] = {"season": season, "pa": 1000 * k}
        for i, comp in enumerate(c.BAT_COMPONENTS):
            row[f"r_{comp}"] = 0.01 * (i + 1) + 0.0001 * k
        row.update({"obp": 0.3 + 0.001 * k, "slg": 0.4, "ops": 0.7 + 0.001 * k, "hr_per_pa": 0.03})
        rows.append(row)
    return _frame(c.LEAGUE_BAT, rows)


def _league_pit_frame(seasons: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season in seasons:
        k = season - 2000
        row: dict[str, Any] = {"season": season, "bf": 1000 * k, "outs": 700 * k}
        for i, comp in enumerate(c.PIT_COMPONENTS):
            row[f"r_{comp}"] = 0.02 * (i + 1)
        row.update(
            {
                "er_per_out": 0.15,
                "era": 4.0 + 0.01 * k,
                "k_bb_pct": 15.0,
                "fip_const": 3.1,
                "bf_per_out": 1.43,
            }
        )
        rows.append(row)
    return _frame(c.LEAGUE_PIT, rows)


def test_models_league_row_returns_season() -> None:
    bat = _league_bat_frame([2022, 2023, 2024])
    row = league_row(bat, 2023)
    assert isinstance(row, pd.Series)
    # 2023: k = 23 → pa 23000, ops 0.7 + 0.023 = 0.723
    assert int(row["season"]) == 2023
    assert int(row["pa"]) == 23_000
    assert float(row["ops"]) == pytest.approx(0.723)

    pit = _league_pit_frame([2024, 2023])
    prow = league_row(pit, 2024)
    # 2024: k = 24 → bf 24000, era 4.0 + 0.24 = 4.24
    assert int(prow["season"]) == 2024
    assert int(prow["bf"]) == 24_000
    assert float(prow["era"]) == pytest.approx(4.24)


def test_models_league_row_missing_season_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="2030"):
        league_row(_league_bat_frame([2022, 2023]), 2030)
    with pytest.raises(KeyError, match="2021"):
        league_row(_league_pit_frame([2022]), 2021)


# ---------------------------------------------------------------------------
# 누설 방지 (N2) — 한 시즌의 행은 그 시즌 기록만으로 정해진다
# ---------------------------------------------------------------------------

POISON = 99_999


def test_models_league_future_rows_do_not_change_past_seasons() -> None:
    # 2025·2026 행(기존 선수의 엄청난 기록, 새 선수)을 섞어도 2020·2023·2024 행은 그대로다
    bat_future = [
        _bat(1, 2025, pa=POISON, ab=POISON, h=POISON, b2=0, b3=0, hr=0, bb=0, hbp=0, sf=0, sh=0),
        _bat(99, 2026, pa=5, ab=1, h=1, b2=0, b3=0, hr=1, bb=4, hbp=0, sf=0, sh=0),
    ]
    bat_clean = league_batting(_frame(c.BATTING_SEASONS, BAT_ROWS))
    bat_all = league_batting(_frame(c.BATTING_SEASONS, bat_future + BAT_ROWS))
    assert bat_all["season"].tolist() == [2020, 2023, 2024, 2025, 2026]
    past = bat_all[bat_all["season"] < 2025].reset_index(drop=True)
    assert_frame_equal(past, bat_clean.reset_index(drop=True))

    pit_future = [
        _pit(11, 2025, outs=POISON, bf=POISON, er=POISON, hr=0, bb=0, hbp=0, so=0),
        _pit(99, 2026, outs=3, bf=5, er=4, hr=2, bb=1, hbp=1, so=0),
    ]
    pit_clean = league_pitching(_frame(c.PITCHING_SEASONS, PIT_ROWS))
    pit_all = league_pitching(_frame(c.PITCHING_SEASONS, PIT_ROWS + pit_future))
    assert pit_all["season"].tolist() == [2020, 2023, 2024, 2025, 2026]
    past_pit = pit_all[pit_all["season"] < 2025].reset_index(drop=True)
    assert_frame_equal(past_pit, pit_clean.reset_index(drop=True))


def test_models_league_row_order_does_not_matter() -> None:
    # 같은 행을 거꾸로 넣어도 결과가 정확히 같다 (합은 순서와 무관, 결과는 season 순)
    bat = _frame(c.BATTING_SEASONS, BAT_ROWS)
    pit = _frame(c.PITCHING_SEASONS, PIT_ROWS)
    assert_frame_equal(league_batting(bat), league_batting(bat.iloc[::-1].reset_index(drop=True)))
    assert_frame_equal(league_pitching(pit), league_pitching(pit.iloc[::-1].reset_index(drop=True)))


def test_models_league_ignores_adjusted_playing_time() -> None:
    # pa_adj·outs_adj를 엉뚱하게 바꿔도 리그 평균은 그대로다 (D9 — 환산은 출전량 모델에만)
    bat = _frame(c.BATTING_SEASONS, BAT_ROWS)
    pit = _frame(c.PITCHING_SEASONS, PIT_ROWS)
    assert_frame_equal(league_batting(bat), league_batting(bat.assign(pa_adj=12345.0)))
    assert_frame_equal(league_pitching(pit), league_pitching(pit.assign(outs_adj=12345.0)))
