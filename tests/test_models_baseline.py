"""기준 모델(models/baseline.py) 테스트 — 직전 시즌·리그 평균 (SPEC F6·D9·D13·N2).

목표 시즌 Y = 2025 (직전 시즌 Y−1 = 2024). 대상 표·정제 표·리그 평균 표는 모두 이 파일에서 손으로
만든다(league 모듈은 쓰지 않는다). 리그 평균 표는 `_lg_bat`·`_lg_pit`이 리그 합에서 일관되게 만든다.
직전 시즌 비율은 그 행의 **개수 열**로 계산해야 하므로, 정제 표 행의 obp·slg·ops 열에는 엉뚱한 값을 넣었다.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlbproj import contracts as c
from mlbproj.contracts import Role
from mlbproj.models.baseline import baseline_playing_time, baseline_rates, league_model_rates

TARGET = 2025
PREV = TARGET - 1
POISON = 99_999

# ---------------------------------------------------------------------------
# 도우미 — 계약 형식의 합성 표
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {"league": c.LEAGUE, "team_name": "TST", "age": 27, "n_teams": 1}
_ADJUSTED = {"pa_adj": "pa", "outs_adj": "outs"}
SEASON_SPECS: dict[Role, c.TableSpec] = {"bat": c.BATTING_SEASONS, "pit": c.PITCHING_SEASONS}
LEAGUE_SPECS: dict[Role, c.TableSpec] = {"bat": c.LEAGUE_BAT, "pit": c.LEAGUE_PIT}
RATE_SPECS: dict[Role, c.TableSpec] = {"bat": c.BAT_RATES, "pit": c.PIT_RATES}
PT_SPECS: dict[Role, c.TableSpec] = {"bat": c.BAT_PLAYING_TIME, "pit": c.PIT_PLAYING_TIME}


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
    ab: int = 0,
    h: int = 0,
    b2: int = 0,
    b3: int = 0,
    hr: int = 0,
    bb: int = 0,
    hbp: int = 0,
    sf: int = 0,
    sh: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    """타자 한 시즌 행. obp·slg·ops 열은 엉뚱한 값 (기준 모델은 개수 열로 계산해야 한다)."""
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
        "obp": 0.111,
        "slg": 0.222,
        "ops": 0.333,
    }
    row.update(extra)
    return row


def _pit(
    pid: int,
    season: int,
    outs: int,
    bf: int,
    er: int = 0,
    hr: int = 0,
    bb: int = 0,
    hbp: int = 0,
    so: int = 0,
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


def _lg_bat(
    pa: int, h1: int, b2: int, b3: int, hr: int, bb: int, hbp: int, sf: int, ab: int
) -> dict[str, float]:
    """리그 합에서 LEAGUE_BAT 한 행의 값 (league 모듈 docstring 공식, 값끼리 일관되게)."""
    h = h1 + b2 + b3 + hr
    obp = (h + bb + hbp) / (ab + bb + hbp + sf)
    slg = (h1 + 2 * b2 + 3 * b3 + 4 * hr) / ab
    counts = {"h1": h1, "b2": b2, "b3": b3, "hr": hr, "bb": bb, "hbp": hbp, "sf": sf, "ab": ab}
    out: dict[str, float] = {"pa": pa}
    out.update({f"r_{k}": v / pa for k, v in counts.items()})
    out.update({"obp": obp, "slg": slg, "ops": obp + slg, "hr_per_pa": hr / pa})
    return out


def _lg_pit(bf: int, outs: int, so: int, bb: int, hbp: int, hr: int, er: int) -> dict[str, float]:
    """리그 합에서 LEAGUE_PIT 한 행의 값."""
    era = 27 * er / outs
    return {
        "bf": bf,
        "outs": outs,
        "r_so": so / bf,
        "r_bb": bb / bf,
        "r_hbp": hbp / bf,
        "r_hr": hr / bf,
        "er_per_out": er / outs,
        "era": era,
        "k_bb_pct": 100 * (so - bb) / bf,
        "fip_const": era - (13 * hr + 3 * (bb + hbp) - 2 * so) / (outs / 3),
        "bf_per_out": bf / outs,
    }


# 리그 평균. 2024(Y−1)를 써야 하고, 2023(Y−2)·2025 이후(미래)를 쓰면 값이 달라진다.
# 2024 타자: r_h1 0.15, r_b2 0.045, r_b3 0.005, r_hr 0.03, r_bb 0.08, r_hbp 0.01, r_sf 0.008, r_ab 0.89
#   obp = (230 + 80 + 10)/(890 + 80 + 10 + 8) = 320/988, slg = (150 + 90 + 15 + 120)/890 = 375/890
LG_BAT: dict[int, dict[str, float]] = {
    2020: _lg_bat(600, 85, 25, 2, 18, 50, 6, 4, 530),
    2023: _lg_bat(1000, 140, 40, 4, 25, 75, 9, 7, 900),
    2024: _lg_bat(1000, 150, 45, 5, 30, 80, 10, 8, 890),
}
LG_BAT_FUTURE: dict[int, dict[str, float]] = {
    2025: _lg_bat(1000, 300, 90, 20, 80, 150, 30, 10, 700),
    2026: _lg_bat(500, 10, 1, 0, 1, 100, 50, 0, 340),
}
# 2024 투수: r_so 0.23, r_bb 0.08, r_hbp 0.01, r_hr 0.03, er_per_out 105/700 = 0.15,
#   era = 27 × 0.15 = 4.05, k_bb_pct = 100 × (0.23 − 0.08) = 15.0
LG_PIT: dict[int, dict[str, float]] = {
    2020: _lg_pit(400, 290, 90, 30, 4, 12, 40),
    2023: _lg_pit(1000, 710, 220, 85, 12, 32, 110),
    2024: _lg_pit(1000, 700, 230, 80, 10, 30, 105),
}
LG_PIT_FUTURE: dict[int, dict[str, float]] = {
    2025: _lg_pit(1000, 690, 300, 50, 8, 10, 60),
    2026: _lg_pit(800, 400, 50, 200, 30, 60, 300),
}


def _league(role: Role, values: Mapping[int, Mapping[str, float]]) -> pd.DataFrame:
    rows = [{"season": season, **v} for season, v in values.items()]
    return _frame(LEAGUE_SPECS[role], rows)


def _cohort(
    role: Role, target: int, players: Sequence[tuple[int, bool]], *, targets: Sequence[int] = ()
) -> pd.DataFrame:
    """대상 표. players = (player_id, played_prev). targets를 주면 행마다 그 목표 시즌을 쓴다."""
    rows: list[dict[str, Any]] = []
    for i, (pid, played_prev) in enumerate(players):
        y = targets[i] if targets else target
        rows.append(
            {
                "player_id": pid,
                "role": role,
                "target_season": y,
                "window_first": y - c.WINDOW,
                "window_last": y - 1,
                "n_window": c.WINDOW,
                "input_first": y - c.WINDOW,
                "input_last": y - 1 if played_prev else y - 2,
                "n_seasons": 2,
                "played_prev": played_prev,
                "sample": 500,
                "sample_outs": 300 if role == "pit" else None,
            }
        )
    return _frame(c.COHORTS, rows)


# ---------------------------------------------------------------------------
# 합성 선수 (Y = 2025)
# ---------------------------------------------------------------------------

# 타자
#  1: 2023·2024 출전 → 2024 행의 개수로 계산
#  2: 2022·2023만 출전 (2024 미출전) → 2024 리그 평균
#  3: 2024 행이 희생번트 1타석뿐 (pa 1, ab 0) → r_*는 자기 값(모두 0), obp·slg·ops는 분모 0이라 리그
#  4: 2024 행의 pa 0 → 모든 비율이 리그
#  50: 2024 행이 있지만 대상이 아님 → 결과에 없다
BAT_ROWS: list[dict[str, Any]] = [
    _bat(1, 2023, pa=400, ab=360, h=90, b2=20, b3=2, hr=10, bb=30, hbp=4, sf=3, sh=3),
    _bat(1, 2024, pa=500, ab=440, h=120, b2=25, b3=3, hr=20, bb=45, hbp=5, sf=5, sh=5),
    _bat(2, 2022, pa=300, ab=270, h=70, b2=15, b3=1, hr=8, bb=25, hbp=2, sf=2, sh=1),
    _bat(2, 2023, pa=200, ab=180, h=50, b2=10, b3=1, hr=6, bb=15, hbp=2, sf=2, sh=1),
    _bat(3, 2023, pa=100, ab=90, h=25, b2=5, b3=1, hr=3, bb=8, hbp=1, sf=1, sh=0),
    _bat(3, 2024, pa=1, sh=1, obp=None, slg=None, ops=None),
    _bat(4, 2023, pa=50, ab=45, h=10, b2=2, b3=0, hr=1, bb=4, hbp=1, sf=0, sh=0),
    _bat(4, 2024, pa=0, obp=None, slg=None, ops=None),
    _bat(50, 2024, pa=600, ab=550, h=170, b2=30, b3=5, hr=40, bb=40, hbp=5, sf=5, sh=0),
]
BAT_COHORT_PLAYERS: list[tuple[int, bool]] = [(3, True), (1, True), (4, True), (2, False)]

# 선수 1의 2024 행: pa 500, ab 440, h 120, b2 25, b3 3, hr 20, bb 45, hbp 5, sf 5
#   h1 = 120 − 25 − 3 − 20 = 72
#   obp = (120 + 45 + 5)/(440 + 45 + 5 + 5) = 170/495
#   slg = (72 + 2·25 + 3·3 + 4·20)/440 = 211/440
P1_RATES: dict[str, float] = {
    "r_h1": 72 / 500,
    "r_b2": 25 / 500,
    "r_b3": 3 / 500,
    "r_hr": 20 / 500,
    "r_bb": 45 / 500,
    "r_hbp": 5 / 500,
    "r_sf": 5 / 500,
    "r_ab": 440 / 500,
    "obp": 170 / 495,
    "slg": 211 / 440,
    "ops": 170 / 495 + 211 / 440,
    "hr_per_pa": 20 / 500,
}

# 투수
#  11: 2024 출전 (outs 180, bf 250, er 20, hr 6, bb 20, hbp 2, so 60)
#      r_so 60/250 = 0.24, r_bb 0.08, r_hbp 0.008, r_hr 0.024, er_per_out 20/180,
#      era = 27 × 20/180 = 3.0, k_bb_pct = 100 × (0.24 − 0.08) = 16.0
#  12: 2023만 출전 → 2024 리그 평균
#  13: 2024 행의 outs 0, bf 4 (er 3, hr 1, bb 2) → r_*는 자기 값, er_per_out·era는 리그
#      r_so 0, r_bb 2/4 = 0.5, r_hbp 0, r_hr 1/4 = 0.25, k_bb_pct = 100 × (0 − 0.5) = −50
#  14: 2024 행의 bf 0·outs 0 → 모든 비율이 리그
#  60: 대상이 아님
PIT_ROWS: list[dict[str, Any]] = [
    _pit(11, 2023, outs=400, bf=560, er=60, hr=15, bb=50, hbp=5, so=120),
    _pit(11, 2024, outs=180, bf=250, er=20, hr=6, bb=20, hbp=2, so=60),
    _pit(12, 2023, outs=150, bf=210, er=25, hr=8, bb=20, hbp=3, so=40),
    _pit(13, 2023, outs=60, bf=90, er=12, hr=3, bb=10, hbp=1, so=15),
    _pit(13, 2024, outs=0, bf=4, er=3, hr=1, bb=2),
    _pit(14, 2024, outs=0, bf=0),
    _pit(60, 2024, outs=600, bf=800, er=50, hr=20, bb=40, hbp=5, so=250),
]
PIT_COHORT_PLAYERS: list[tuple[int, bool]] = [(14, True), (12, False), (11, True), (13, True)]

P11_RATES: dict[str, float] = {
    "r_so": 0.24,
    "r_bb": 0.08,
    "r_hbp": 0.008,
    "r_hr": 0.024,
    "er_per_out": 20 / 180,
    "era": 3.0,
    "k_bb_pct": 16.0,
}
P13_RATES: dict[str, float] = {
    "r_so": 0.0,
    "r_bb": 0.5,
    "r_hbp": 0.0,
    "r_hr": 0.25,
    "er_per_out": LG_PIT[PREV]["er_per_out"],  # outs 0 → 리그 0.15
    "era": LG_PIT[PREV]["era"],  # 27 × 0.15 = 4.05
    "k_bb_pct": -50.0,
}

BAT_RATE_COLUMNS = [f"r_{x}" for x in c.BAT_COMPONENTS] + ["obp", "slg", "ops", "hr_per_pa"]
PIT_RATE_COLUMNS = [f"r_{x}" for x in c.PIT_COMPONENTS] + ["er_per_out", "era", "k_bb_pct"]


def _league_rates(role: Role, season: int) -> dict[str, float]:
    """리그 평균 한 시즌을 비율 표의 값 열로 옮긴 것."""
    if role == "bat":
        lg = LG_BAT[season]
        return {name: lg[name] for name in BAT_RATE_COLUMNS}
    lgp = LG_PIT[season]
    return {name: lgp[name] for name in PIT_RATE_COLUMNS}


def _inputs(role: Role) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(대상 표, 정제 표, 리그 평균) — 모두 Y 이전 행만."""
    if role == "bat":
        return (
            _cohort("bat", TARGET, BAT_COHORT_PLAYERS),
            _frame(c.BATTING_SEASONS, BAT_ROWS),
            _league("bat", LG_BAT),
        )
    return (
        _cohort("pit", TARGET, PIT_COHORT_PLAYERS),
        _frame(c.PITCHING_SEASONS, PIT_ROWS),
        _league("pit", LG_PIT),
    )


def _poisoned_inputs(role: Role) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """`_inputs`에 Y 이후(season ≥ 2025) "독" 행을 앞에 섞은 것 (N2).

    정제 표: 대상 전원의 2025·2026 행(엄청난 기록) + 2025에만 기록이 있는 새 선수 999.
    리그 평균: 2025·2026 행 (값이 2024와 전혀 다르다).
    """
    cohort, _, _ = _inputs(role)
    players = BAT_COHORT_PLAYERS if role == "bat" else PIT_COHORT_PLAYERS
    ids = [pid for pid, _ in players] + [999]
    if role == "bat":
        poison = [
            _bat(pid, season, pa=POISON, ab=POISON - 1, h=POISON - 1, hr=POISON - 1, bb=1)
            for pid in ids
            for season in (TARGET, TARGET + 1)
        ]
        seasons = _frame(c.BATTING_SEASONS, poison + BAT_ROWS)
        league = _league("bat", {**LG_BAT_FUTURE, **LG_BAT})
    else:
        poison = [
            _pit(pid, season, outs=POISON, bf=POISON, er=POISON, hr=POISON, so=1)
            for pid in ids
            for season in (TARGET, TARGET + 1)
        ]
        seasons = _frame(c.PITCHING_SEASONS, poison + PIT_ROWS)
        league = _league("pit", {**LG_PIT_FUTURE, **LG_PIT})
    return cohort, seasons, league


def _by_player(df: pd.DataFrame) -> dict[int, dict[Any, Any]]:
    return {int(rec["player_id"]): rec for rec in df.to_dict("records")}


def _assert_values(actual: Mapping[Any, Any], expected: Mapping[str, float], who: str) -> None:
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value, rel=1e-12, abs=1e-15), f"{who} {name}"


def _assert_rates_shape(out: pd.DataFrame, role: Role, ids: list[int]) -> None:
    assert c.check_frame(out, RATE_SPECS[role]) == []
    assert out["player_id"].tolist() == ids
    assert out["target_season"].tolist() == [TARGET] * len(ids)
    assert not bool(out.isna().any().any()), "결과에 결측이 없다"


# ---------------------------------------------------------------------------
# baseline_rates — 타자
# ---------------------------------------------------------------------------


def test_models_baseline_rates_bat_prev_season_counts() -> None:
    out = baseline_rates(*_inputs("bat"), "bat")
    _assert_rates_shape(out, "bat", [1, 2, 3, 4])
    rows = _by_player(out)
    # 선수 1: 2024 행의 개수로 계산 (2023 행·obp/slg/ops 열은 쓰지 않는다)
    _assert_values(rows[1], P1_RATES, "선수 1")
    # lg_ops = 2024 리그 ops (전원)
    for pid in (1, 2, 3, 4):
        assert rows[pid]["lg_ops"] == pytest.approx(LG_BAT[PREV]["ops"])


def test_models_baseline_rates_bat_absent_prev_uses_league() -> None:
    rows = _by_player(baseline_rates(*_inputs("bat"), "bat"))
    # 선수 2: 2024 미출전 → 2024 리그 평균 (2023 리그·2023 자기 기록이 아니다)
    _assert_values(rows[2], _league_rates("bat", PREV), "선수 2")
    assert rows[2]["ops"] != pytest.approx(LG_BAT[2023]["ops"])


def test_models_baseline_rates_bat_zero_denominators_fall_back() -> None:
    rows = _by_player(baseline_rates(*_inputs("bat"), "bat"))
    lg = LG_BAT[PREV]
    # 선수 3: pa 1(희생번트), ab 0 → 성분 비율은 자기 값 0/1 = 0
    for comp in c.BAT_COMPONENTS:
        assert rows[3][f"r_{comp}"] == 0.0, f"선수 3 r_{comp}"
    assert rows[3]["hr_per_pa"] == 0.0
    #   obp 분모 ab+bb+hbp+sf = 0, slg 분모 ab = 0 → 리그 obp·slg, ops = 리그 obp + 리그 slg = 리그 ops
    _assert_values(rows[3], {"obp": lg["obp"], "slg": lg["slg"], "ops": lg["ops"]}, "선수 3")
    # 선수 4: pa 0 → 모든 비율이 2024 리그 평균
    _assert_values(rows[4], _league_rates("bat", PREV), "선수 4")


# ---------------------------------------------------------------------------
# baseline_rates — 투수
# ---------------------------------------------------------------------------


def test_models_baseline_rates_pit_prev_season_counts() -> None:
    out = baseline_rates(*_inputs("pit"), "pit")
    _assert_rates_shape(out, "pit", [11, 12, 13, 14])
    rows = _by_player(out)
    _assert_values(rows[11], P11_RATES, "선수 11")
    # lg_era·lg_k_bb_pct = 2024 리그 (전원): 4.05, 15.0
    for pid in (11, 12, 13, 14):
        assert rows[pid]["lg_era"] == pytest.approx(4.05)
        assert rows[pid]["lg_k_bb_pct"] == pytest.approx(15.0)


def test_models_baseline_rates_pit_absent_and_zero_denominators() -> None:
    rows = _by_player(baseline_rates(*_inputs("pit"), "pit"))
    # 선수 12: 2024 미출전 → 2024 리그 평균
    _assert_values(rows[12], _league_rates("pit", PREV), "선수 12")
    # 선수 13: outs 0 → er_per_out·era만 리그, bf 4로 r_*·k_bb_pct는 자기 값
    _assert_values(rows[13], P13_RATES, "선수 13")
    # 선수 14: bf 0·outs 0 → 전부 리그
    _assert_values(rows[14], _league_rates("pit", PREV), "선수 14")


# ---------------------------------------------------------------------------
# league_model_rates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["bat", "pit"])
def test_models_baseline_league_model_rates_gives_league_prev(role: Role) -> None:
    cohort, _, league = _inputs(role)
    out = league_model_rates(cohort, league, role)
    ids = sorted(pid for pid, _ in (BAT_COHORT_PLAYERS if role == "bat" else PIT_COHORT_PLAYERS))
    _assert_rates_shape(out, role, ids)
    expected = _league_rates(role, PREV)
    if role == "bat":
        expected["lg_ops"] = LG_BAT[PREV]["ops"]
    else:
        expected["lg_era"] = LG_PIT[PREV]["era"]
        expected["lg_k_bb_pct"] = LG_PIT[PREV]["k_bb_pct"]
    for pid, rec in _by_player(out).items():
        # 직전 시즌에 뛰었든 안 뛰었든 전원 2024 리그 평균
        _assert_values(rec, expected, f"선수 {pid}")


# ---------------------------------------------------------------------------
# baseline_playing_time
# ---------------------------------------------------------------------------


def test_models_baseline_playing_time_prev_season() -> None:
    cohort, seasons, _ = _inputs("bat")
    out = baseline_playing_time(cohort, seasons, "bat")
    assert c.check_frame(out, c.BAT_PLAYING_TIME) == []
    assert out["player_id"].tolist() == [1, 2, 3, 4]
    assert out["target_season"].tolist() == [TARGET] * 4
    # 1: 2024 pa_adj 500, 2: 2024 행 없음 → 0, 3: 1, 4: 0
    assert out["pa"].tolist() == [500.0, 0.0, 1.0, 0.0]

    pcohort, pseasons, _ = _inputs("pit")
    pout = baseline_playing_time(pcohort, pseasons, "pit")
    assert c.check_frame(pout, c.PIT_PLAYING_TIME) == []
    assert pout["player_id"].tolist() == [11, 12, 13, 14]
    # 11: 2024 outs_adj 180, 12: 행 없음 → 0, 13: 0, 14: 0
    assert pout["outs"].tolist() == [180.0, 0.0, 0.0, 0.0]


def test_models_baseline_playing_time_short_season_uses_adjusted() -> None:
    # 목표 2021 → Y−1 = 2020(60경기). 출전량은 환산값 pa_adj·outs_adj를 쓴다 (D9)
    bat_rows = [
        _bat(7, 2020, pa=60, ab=55, h=15, bb=5, pa_adj=162.0),  # 60 × 162/60 = 162
        _bat(8, 2019, pa=300, ab=280, h=80, bb=20),  # 2020 미출전 → 0
    ]
    out = baseline_playing_time(
        _cohort("bat", 2021, [(8, False), (7, True)]), _frame(c.BATTING_SEASONS, bat_rows), "bat"
    )
    assert out["player_id"].tolist() == [7, 8]
    assert out["target_season"].tolist() == [2021, 2021]
    assert out["pa"].tolist() == pytest.approx([162.0, 0.0])

    pit_rows = [_pit(17, 2020, outs=90, bf=120, outs_adj=243.0)]  # 90 × 162/60 = 243
    pout = baseline_playing_time(
        _cohort("pit", 2021, [(17, True)]), _frame(c.PITCHING_SEASONS, pit_rows), "pit"
    )
    assert pout["outs"].tolist() == pytest.approx([243.0])


def test_models_baseline_rates_short_season_uses_actual_counts() -> None:
    # 목표 2021: 비율은 2020 실제 표본으로 계산한다 (환산은 출전량에만, D9)
    # 선수 7의 2020 행: pa 60, ab 50, h 15, b2 3, hr 2, bb 8, hbp 1, sf 1
    #   h1 = 15 − 3 − 0 − 2 = 10, obp = (15 + 8 + 1)/(50 + 8 + 1 + 1) = 0.4,
    #   slg = (10 + 2·3 + 4·2)/50 = 0.48, hr_per_pa = 2/60
    row = _bat(7, 2020, pa=60, ab=50, h=15, b2=3, hr=2, bb=8, hbp=1, sf=1, pa_adj=162.0)
    out = baseline_rates(
        _cohort("bat", 2021, [(7, True)]),
        _frame(c.BATTING_SEASONS, [row]),
        _league("bat", LG_BAT),
        "bat",
    )
    rec = _by_player(out)[7]
    _assert_values(
        rec,
        {"r_h1": 10 / 60, "obp": 0.4, "slg": 0.48, "ops": 0.88, "hr_per_pa": 2 / 60},
        "선수 7",
    )
    assert rec["lg_ops"] == pytest.approx(LG_BAT[2020]["ops"])
    assert int(rec["target_season"]) == 2021


# ---------------------------------------------------------------------------
# 누설 방지 (N2)
# ---------------------------------------------------------------------------

ModelCall = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, Role], pd.DataFrame]
CALLS: dict[str, ModelCall] = {
    "baseline_rates": lambda co, se, lg, r: baseline_rates(co, se, lg, r),
    "league_model_rates": lambda co, se, lg, r: league_model_rates(co, lg, r),
    "baseline_playing_time": lambda co, se, lg, r: baseline_playing_time(co, se, r),
}


@pytest.mark.parametrize("role", ["bat", "pit"])
@pytest.mark.parametrize("name", list(CALLS))
def test_models_baseline_no_leakage_from_target_season_or_later(name: str, role: Role) -> None:
    call = CALLS[name]
    clean = call(*_inputs(role), role)
    poisoned = call(*_poisoned_inputs(role), role)
    assert_frame_equal(clean.reset_index(drop=True), poisoned.reset_index(drop=True))


# ---------------------------------------------------------------------------
# 오류
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(CALLS))
def test_models_baseline_rejects_wrong_role(name: str) -> None:
    call = CALLS[name]
    # 타자 대상 표를 투수로 부른다 (정제 표·리그 평균은 투수 것)
    bat_cohort, _, _ = _inputs("bat")
    _, pit_seasons, pit_league = _inputs("pit")
    with pytest.raises(ValueError):
        call(bat_cohort, pit_seasons, pit_league, "pit")


@pytest.mark.parametrize("name", list(CALLS))
def test_models_baseline_rejects_mixed_target(name: str) -> None:
    call = CALLS[name]
    _, seasons, league = _inputs("bat")
    mixed = _cohort(
        "bat", TARGET, [(1, True), (2, False)], targets=[TARGET, TARGET + 1]
    )  # 2025와 2026이 섞임
    with pytest.raises(ValueError):
        call(mixed, seasons, league, "bat")


def test_models_baseline_rejects_contract_violation() -> None:
    cohort, seasons, league = _inputs("bat")
    _, pit_seasons, pit_league = _inputs("pit")
    # 정제 표가 다른 역할의 계약
    with pytest.raises(ValueError):
        baseline_rates(cohort, pit_seasons, league, "bat")
    with pytest.raises(ValueError):
        baseline_playing_time(cohort, pit_seasons, "bat")
    # 리그 평균이 다른 역할의 계약
    with pytest.raises(ValueError):
        baseline_rates(cohort, seasons, pit_league, "bat")
    # 대상 표 열이 빠짐
    with pytest.raises(ValueError):
        baseline_rates(cohort.drop(columns=["played_prev"]), seasons, league, "bat")


def test_models_baseline_rates_missing_prev_league_raises_keyerror() -> None:
    cohort, seasons, _ = _inputs("bat")
    # 리그 평균에 2024(Y−1)가 없다 (2023만)
    league = _league("bat", {2023: LG_BAT[2023]})
    with pytest.raises(KeyError, match="2024"):
        baseline_rates(cohort, seasons, league, "bat")
    pcohort, pseasons, _ = _inputs("pit")
    with pytest.raises(KeyError, match="2024"):
        baseline_rates(pcohort, pseasons, _league("pit", {2023: LG_PIT[2023]}), "pit")


# ---------------------------------------------------------------------------
# 분모가 한쪽만 0인 경우 · 중복 행 · 행 순서
# ---------------------------------------------------------------------------


def test_models_baseline_rates_bat_obp_own_slg_league() -> None:
    # 선수 5의 2024 행: pa 5, ab 0, bb 4, hbp 1 (볼넷·사구만)
    #   r_bb = 4/5 = 0.8, r_hbp = 1/5 = 0.2, 나머지 r_* = 0, hr_per_pa = 0
    #   obp = (0 + 4 + 1)/(0 + 4 + 1 + 0) = 1.0 (자기 값), slg 분모 ab = 0 → 리그 slg 375/890
    #   ops = obp + slg = 1.0 + 375/890 (한쪽만 리그로 채워도 ops = obp + slg)
    row = _bat(5, PREV, pa=5, bb=4, hbp=1)
    out = baseline_rates(
        _cohort("bat", TARGET, [(5, True)]),
        _frame(c.BATTING_SEASONS, [row]),
        _league("bat", LG_BAT),
        "bat",
    )
    rec = _by_player(out)[5]
    expected = {f"r_{comp}": 0.0 for comp in c.BAT_COMPONENTS}
    expected.update({"r_bb": 0.8, "r_hbp": 0.2, "hr_per_pa": 0.0})
    expected.update({"obp": 1.0, "slg": 375 / 890, "ops": 1.0 + 375 / 890})
    _assert_values(rec, expected, "선수 5")
    assert rec["ops"] == pytest.approx(rec["obp"] + rec["slg"], rel=1e-15)


def test_models_baseline_rates_pit_outs_without_batters_faced() -> None:
    # 선수 16의 2024 행: 견제 아웃 1개만 (outs 1, bf 0, er 0)
    #   bf 0 → r_*·k_bb_pct는 2024 리그 (0.23, 0.08, 0.01, 0.03, 15.0)
    #   outs 1 → er_per_out = 0/1 = 0, era = 27 × 0 = 0 (자기 값)
    row = _pit(16, PREV, outs=1, bf=0)
    out = baseline_rates(
        _cohort("pit", TARGET, [(16, True)]),
        _frame(c.PITCHING_SEASONS, [row]),
        _league("pit", LG_PIT),
        "pit",
    )
    expected = {"r_so": 0.23, "r_bb": 0.08, "r_hbp": 0.01, "r_hr": 0.03, "k_bb_pct": 15.0}
    expected.update({"er_per_out": 0.0, "era": 0.0})
    _assert_values(_by_player(out)[16], expected, "선수 16")


@pytest.mark.parametrize("role", ["bat", "pit"])
@pytest.mark.parametrize("name", ["baseline_rates", "baseline_playing_time"])
def test_models_baseline_rejects_two_prev_rows_for_one_player(name: str, role: Role) -> None:
    # 한 선수의 2024 행이 리그 둘로 두 행이면 계약(키에 league 포함)은 통과하지만 직전 시즌이 모호하다.
    # v0.1은 MLB만 다루므로(D1) 조용히 하나를 고르지 않고 오류를 낸다 — 메시지에 시즌·선수를 적는다
    cohort, seasons, league = _inputs(role)
    pid = 1 if role == "bat" else 11
    extra = seasons[(seasons["player_id"] == pid) & (seasons["season"] == PREV)].assign(
        league="KBO"
    )
    doubled = pd.concat([seasons, extra], ignore_index=True)
    assert c.check_frame(doubled, SEASON_SPECS[role]) == []
    with pytest.raises(ValueError, match=rf"{PREV} 시즌.*\[{pid}\]"):
        CALLS[name](cohort, doubled, league, role)


@pytest.mark.parametrize("role", ["bat", "pit"])
@pytest.mark.parametrize("name", list(CALLS))
def test_models_baseline_row_order_does_not_matter(name: str, role: Role) -> None:
    # 대상 표·정제 표·리그 평균의 행 순서를 뒤집어도 결과가 정확히 같다 (player_id 순)
    cohort, seasons, league = _inputs(role)
    flipped = (_flipped(cohort), _flipped(seasons), _flipped(league))
    assert_frame_equal(CALLS[name](cohort, seasons, league, role), CALLS[name](*flipped, role))


def _flipped(df: pd.DataFrame) -> pd.DataFrame:
    """행 순서를 뒤집은 표 (색인은 0부터 다시)."""
    return df.iloc[::-1].reset_index(drop=True)
