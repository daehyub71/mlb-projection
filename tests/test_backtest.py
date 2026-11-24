"""과거 검증·조정 명령 테스트 — 2026 차단과 빈 대상 (SPEC F10, PLAN §2-3).

2026 최종 검증은 파라미터를 커밋한 뒤 final-test로 한 번만 한다. 과거 검증(backtest)과 조정(tune)은
2026을 목표로 받으면 데이터를 읽기 전에 멈춰야 한다.
"""

from pathlib import Path

import pandas as pd
import pytest

from mlbproj import contracts as c
from mlbproj.backtest import Tables, cohort_for, run_backtest
from mlbproj.cli import main
from mlbproj.models.params import DEFAULT_PARAMS
from mlbproj.tune import tune


def _empty_tables() -> Tables:
    cohorts = pd.DataFrame({"target_season": pd.Series(dtype="int64"), "role": [], "player_id": []})
    return Tables(seasons={}, cohorts=cohorts, league={}, debuts={})


def test_backtest_rejects_final_test_target() -> None:
    with pytest.raises(ValueError, match="final-test"):
        run_backtest(_empty_tables(), DEFAULT_PARAMS, [c.FINAL_TEST_TARGET])


def test_backtest_tune_rejects_final_test_target() -> None:
    with pytest.raises(ValueError, match="2026"):
        tune(_empty_tables(), targets=(2024, c.FINAL_TEST_TARGET))


def test_backtest_cohort_for_missing_target() -> None:
    with pytest.raises(ValueError, match="대상 표가 없다"):
        cohort_for(_empty_tables(), "bat", 2024)


@pytest.mark.parametrize("command", ["backtest", "tune"])
def test_backtest_cli_refuses_2026(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([command, "--targets", "2024,2026", "--out-dir", str(tmp_path)])
    assert code == 1
    assert "final-test" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())  # 아무것도 쓰지 않는다
