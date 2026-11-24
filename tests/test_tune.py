"""파라미터 조정(tune.py) 테스트 — 격자와 선택 규칙 (SPEC D10, PLAN §2-3)."""

import pandas as pd
import pytest

from mlbproj import contracts as c
from mlbproj.models.params import validate_params
from mlbproj.tune import WEIGHT_CANDIDATES, best_row, best_row_within, candidate_params


def test_tune_candidates_are_valid_params() -> None:
    combos = list(candidate_params())
    assert combos, "격자가 비었다"
    for params in combos:
        validate_params(params)
    # 같은 파라미터가 두 번 나오지 않는다
    assert len({repr(p) for p in combos}) == len(combos)
    for weights in WEIGHT_CANDIDATES:
        assert len(weights) == c.WINDOW


def test_tune_best_row_takes_lowest_mean_then_first() -> None:
    report = pd.DataFrame(
        {
            "candidate": ["a", "b", "c"],
            "mae_2024": [0.080, 0.070, 0.071],
            "mae_2025": [0.070, 0.072, 0.071],
        }
    )
    assert (
        best_row(report, ["mae_2024", "mae_2025"])["candidate"] == "c"
    )  # 평균 .071 < .0715 < .075
    tie = pd.DataFrame({"candidate": ["x", "y"], "m": [1.0, 1.0]})
    assert best_row(tie, ["m"])["candidate"] == "x"  # 같으면 먼저 나온 후보


def test_tune_best_row_rejects_empty() -> None:
    with pytest.raises(ValueError):
        best_row(pd.DataFrame({"candidate": [], "m": []}), ["m"])


def _guarded() -> pd.DataFrame:
    # 홈런 회귀 표본 후보: hr_* = HR MAE(목표), ops_* = OPS MAE(지킬 지표)
    return pd.DataFrame(
        {
            "candidate": ["r150", "r1200", "r2400", "r4800"],
            "hr_2024": [2.60, 2.66, 2.70, 2.76],
            "hr_2025": [2.78, 2.82, 2.87, 2.93],
            "ops_2024": [0.0754, 0.0721, 0.07108, 0.07100],
            "ops_2025": [0.0657, 0.0631, 0.06262, 0.06276],
        }
    )


def test_tune_best_row_within_keeps_guard_metric() -> None:
    # OPS 평균이 기준(.06688) 이하인 r2400·r4800 중 HR 평균이 작은 r2400
    row = best_row_within(
        _guarded(), ["hr_2024", "hr_2025"], ["ops_2024", "ops_2025"], limit=0.06688
    )
    assert row["candidate"] == "r2400"


def test_tune_best_row_within_limit_is_inclusive() -> None:
    # 기준과 같은 값은 통과한다 — 1단계에서 고른 값 자체는 늘 후보로 남는다
    report = _guarded()
    limit = float(report.loc[3, ["ops_2024", "ops_2025"]].mean())
    row = best_row_within(report.iloc[[3]], ["hr_2024", "hr_2025"], ["ops_2024", "ops_2025"], limit)
    assert row["candidate"] == "r4800"


def test_tune_best_row_within_rejects_when_none_pass() -> None:
    with pytest.raises(ValueError, match="기준"):
        best_row_within(_guarded(), ["hr_2024"], ["ops_2024"], limit=0.01)
