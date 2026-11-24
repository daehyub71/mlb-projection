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


def test_backtest_cli_adopt_refuses_2026(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "extended_v1.json"
    code = main(
        ["adopt", "--targets", "2025,2026", "--out", str(out), "--report", str(tmp_path / "r.csv")]
    )
    assert code == 1
    assert "2026" in capsys.readouterr().err
    assert not out.exists()


def test_backtest_cli_adopt_writes_adoption_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import mlbproj.cli as cli
    from mlbproj.models.params import Adoption, load_adoption, save_params

    chosen = Adoption("v1", {m: m in ("hr", "era") for m in c.EXT_METRICS})
    report = pd.DataFrame({"role": ["bat"], "metric": ["hr"], "adopted": [True]})
    monkeypatch.setattr(cli, "load_tables", lambda *a, **k: _empty_tables())
    monkeypatch.setattr(cli, "extended_scorer", lambda tables, params, targets: "scorer")
    monkeypatch.setattr(cli, "select_adoption", lambda score, targets: (chosen, report))
    params_path = tmp_path / "params.json"
    save_params(DEFAULT_PARAMS, params_path)
    out, rep = tmp_path / "config" / "extended_v1.json", tmp_path / "out" / "adoption.csv"
    code = main(["adopt", "--params", str(params_path), "--out", str(out), "--report", str(rep)])
    assert code == 0, capsys.readouterr().err
    assert load_adoption(out) == chosen
    assert pd.read_csv(rep)["metric"].tolist() == ["hr"]
    assert "sha256" in capsys.readouterr().out


def test_backtest_cli_uses_adoption_for_extended_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlbproj.cli as cli
    from mlbproj.models.params import Adoption, save_adoption

    seen: list[tuple[int, Adoption]] = []

    def fake_eval(tables: Tables, params: object, adoption: Adoption, target: int) -> pd.DataFrame:
        seen.append((target, adoption))
        return pd.DataFrame(
            {
                "target_season": [target],
                "role": ["bat"],
                "model": ["extended"],
                "metric": ["ops"],
                "group": ["all"],
                "n": [10],
                "mae": [0.07],
                "mae_weighted": [0.071],
                "baseline_mae_league": [0.08],
                "baseline_mae_prev": [0.09],
                "excluded": [0],
            }
        )

    monkeypatch.setattr(cli, "load_tables", lambda *a, **k: _empty_tables())
    monkeypatch.setattr(cli.final, "evaluate_target", fake_eval)
    adoption = Adoption("v1", {m: m == "ops" for m in c.EXT_METRICS})
    adoption_path = tmp_path / "extended_v1.json"
    save_adoption(adoption, adoption_path)
    code = main(
        [
            "backtest",
            "--targets",
            "2024,2025",
            "--adoption",
            str(adoption_path),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert [t for t, _ in seen] == [2024, 2025]
    assert all(a == adoption for _, a in seen)
    written = pd.read_csv(tmp_path / "backtest.csv")
    assert set(written["model"]) == {"extended"}
