"""과거 오차 80% 범위 테스트 (SPEC D29·D24·D13, contracts.BANDS)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from mlbproj import contracts as c
from mlbproj.bands import band_targets, error_bands, player_errors

_DEFAULTS: dict[str, Any] = {"int": 0, "float": 0.0, "str": "", "bool": False}


def _seasons(spec: c.TableSpec, rows: list[dict[str, Any]]) -> pd.DataFrame:
    """계약의 모든 열을 기본값으로 채운 정제 시즌 표."""
    full = [
        {col.name: row.get(col.name, _DEFAULTS[col.kind]) for col in spec.columns} for row in rows
    ]
    df = pd.DataFrame(full, columns=spec.column_names)
    for col in spec.columns:
        if col.kind == "int":
            df[col.name] = df[col.name].astype("int64")
        elif col.kind == "float":
            df[col.name] = df[col.name].astype("float64")
        elif col.kind == "bool":
            df[col.name] = df[col.name].astype("bool")
    return df


def _bat_row(pid: int, season: int, pa: int, ops: float) -> dict[str, Any]:
    return {
        "player_id": pid,
        "season": season,
        "league": "MLB",
        "pa": pa,
        "ab": pa,
        "ops": ops,
        "pa_adj": float(pa),
    }


def _pit_row(
    pid: int, season: int, outs: int, bf: int, er: int, so: int, bb: int
) -> dict[str, Any]:
    return {
        "player_id": pid,
        "season": season,
        "league": "MLB",
        "outs": outs,
        "bf": bf,
        "er": er,
        "so": so,
        "bb": bb,
        "outs_adj": float(outs),
    }


def test_bands_player_errors_rate_populations_and_groups() -> None:
    seasons = {
        "bat": _seasons(
            c.BATTING_SEASONS,
            [
                _bat_row(1, 2025, 500, 0.800),  # 대표 집단 (200 PA 이상)
                _bat_row(2, 2025, 199, 0.900),  # 199 PA — 빠진다
                _bat_row(3, 2025, 200, 0.600),  # 경계 포함
            ],
        ),
        "pit": _seasons(
            c.PITCHING_SEASONS,
            [
                _pit_row(11, 2025, 150, 200, 20, 50, 10),  # 50 IP 경계 포함: era 3.6, k_bb 20.0
                _pit_row(12, 2025, 149, 200, 30, 40, 20),  # 빠진다
            ],
        ),
    }
    projections = {
        "bat": pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4],
                "n_seasons": [5, 2, 1, 3],
                "ops": [0.750, 0.700, 0.650, 0.7],
            }
        ),
        "pit": pd.DataFrame(
            {
                "player_id": [11, 12],
                "n_seasons": [2, 4],
                "era": [4.1, 3.0],
                "k_bb_pct": [15.5, 10.0],
            }
        ),
    }
    out = player_errors(projections, seasons, 2025)
    assert list(out.columns) == [
        "target_season",
        "role",
        "metric",
        "group",
        "player_id",
        "abs_error",
    ]
    got = {
        (r.role, r.metric, r.player_id): (r.group, round(r.abs_error, 6)) for r in out.itertuples()
    }
    assert got == {
        ("bat", "ops", 1): ("seasons_3_5", 0.05),  # |.750 − .800|
        ("bat", "ops", 3): ("seasons_1_2", 0.05),  # |.650 − .600|
        ("pit", "era", 11): ("seasons_1_2", 0.5),  # 27·20/150 = 3.6, |4.1 − 3.6|
        ("pit", "k_bb_pct", 11): ("seasons_1_2", 4.5),  # 100·(50 − 10)/200 = 20, |15.5 − 20|
    }
    assert set(out["target_season"]) == {2025}


def test_bands_error_bands_quantile_per_group() -> None:
    errors = pd.DataFrame(
        {
            "target_season": [2024] * 5 + [2025] * 5,
            "role": ["bat"] * 10,
            "metric": ["ops"] * 10,
            "group": ["seasons_3_5"] * 10,
            "player_id": range(10),
            "abs_error": [0.01 * i for i in range(10)],
        }
    )
    bands = error_bands(errors)
    assert c.check_frame(bands, c.BANDS) == []
    # 역할·지표·그룹 전부 (값이 없으면 n 0·q80 결측)
    expected_keys = [
        (role, metric, group)
        for role in c.ROLES
        for metric in c.BAND_METRICS[role]
        for group in c.BAND_GROUPS
    ]
    assert list(zip(bands.role, bands.metric, bands.group, strict=True)) == expected_keys
    row = bands[(bands.metric == "ops") & (bands.group == "seasons_3_5")].iloc[0]
    assert row.n == 10
    assert row.q80 == pytest.approx(float(np.quantile([0.01 * i for i in range(10)], 0.8)))  # 0.072
    assert row.targets == "2024,2025"
    empty = bands[(bands.metric == "ops") & (bands.group == "seasons_1_2")].iloc[0]
    assert empty.n == 0 and np.isnan(empty.q80)


def test_bands_band_targets_include_2026_only_after_final_test() -> None:
    assert band_targets(final_test_done=False) == (2024, 2025)
    assert band_targets(final_test_done=True) == (2024, 2025, c.FINAL_TEST_TARGET)
