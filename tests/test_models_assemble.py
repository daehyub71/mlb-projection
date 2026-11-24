"""모델 결과 결합(models/assemble.py) 테스트 — 비율 × 출전량 → 예측 표 (F8: HR = PA × HR/PA)."""

import pandas as pd
import pytest

from mlbproj import contracts as c
from mlbproj.models.assemble import assemble_bat, assemble_pit


def _cohort(role: str, ids: list[int]) -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame(
        {
            "player_id": ids,
            "role": [role] * n,
            "target_season": [2025] * n,
            "window_first": [2020] * n,
            "window_last": [2024] * n,
            "n_window": [5] * n,
            "input_first": [2021] * n,
            "input_last": [2024] * n,
            "n_seasons": [4] * n,
            "played_prev": [True] * n,
            "sample": [1000 + i for i in range(n)],
            "sample_outs": pd.array(
                [300 + i for i in range(n)] if role == "pit" else [None] * n, dtype="Int64"
            ),
        }
    )


def _bat_rates(ids: list[int]) -> pd.DataFrame:
    df = pd.DataFrame({"player_id": ids, "target_season": [2025] * len(ids)})
    for comp in c.BAT_COMPONENTS:
        df[f"r_{comp}"] = 0.1
    df["obp"] = 0.330
    df["slg"] = 0.420
    df["ops"] = 0.750
    df["hr_per_pa"] = [0.04, 0.02][: len(ids)]
    df["lg_ops"] = 0.720
    return df


def _pit_rates(ids: list[int]) -> pd.DataFrame:
    df = pd.DataFrame({"player_id": ids, "target_season": [2025] * len(ids)})
    for comp in c.PIT_COMPONENTS:
        df[f"r_{comp}"] = 0.05
    df["er_per_out"] = 0.15
    df["era"] = 4.05
    df["k_bb_pct"] = 14.0
    df["lg_era"] = 4.20
    df["lg_k_bb_pct"] = 13.5
    return df


def test_models_assemble_bat() -> None:
    ids = [2, 1]
    pt = pd.DataFrame({"player_id": [1, 2], "target_season": [2025, 2025], "pa": [600.0, 0.0]})
    out = assemble_bat(_cohort("bat", ids), _bat_rates(ids), pt, "weighted")
    assert c.check_frame(out, c.BAT_PROJECTIONS) == []
    assert out["player_id"].tolist() == [1, 2]
    row = out.set_index("player_id").loc[1]
    assert row["pa"] == 600.0 and row["hr"] == pytest.approx(600.0 * 0.02)
    assert row["model"] == "weighted" and row["role"] == "bat" and row["sample_pa"] == 1001
    assert out.set_index("player_id").loc[2, "hr"] == 0.0


def test_models_assemble_pit() -> None:
    ids = [7]
    pt = pd.DataFrame({"player_id": [7], "target_season": [2025], "outs": [517.0]})
    out = assemble_pit(_cohort("pit", ids), _pit_rates(ids), pt, "baseline")
    assert c.check_frame(out, c.PIT_PROJECTIONS) == []
    row = out.iloc[0]
    assert row["ip"] == pytest.approx(517 / 3)
    assert (row["sample_bf"], row["sample_outs"]) == (1000, 300)
    assert (row["era"], row["k_bb_pct"], row["lg_era"]) == (4.05, 14.0, 4.20)


def test_models_assemble_rejects_mismatched_players() -> None:
    pt = pd.DataFrame({"player_id": [1], "target_season": [2025], "pa": [600.0]})
    with pytest.raises(ValueError, match="선수"):
        assemble_bat(_cohort("bat", [1, 2]), _bat_rates([1, 2]), pt, "weighted")


def test_models_assemble_rejects_other_target_or_role() -> None:
    pt = pd.DataFrame({"player_id": [1], "target_season": [2026], "pa": [600.0]})
    with pytest.raises(ValueError, match="목표 시즌"):
        assemble_bat(_cohort("bat", [1]), _bat_rates([1]), pt, "weighted")
    pt_ok = pd.DataFrame({"player_id": [1], "target_season": [2025], "outs": [10.0]})
    with pytest.raises(ValueError, match="역할"):
        assemble_pit(_cohort("bat", [1]), _pit_rates([1]), pt_ok, "weighted")


def test_models_assemble_rejects_unknown_model() -> None:
    pt = pd.DataFrame({"player_id": [1], "target_season": [2025], "pa": [600.0]})
    with pytest.raises(ValueError, match="모델"):
        assemble_bat(_cohort("bat", [1]), _bat_rates([1]), pt, "magic")  # type: ignore[arg-type]
