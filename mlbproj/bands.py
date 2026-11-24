"""과거 오차 80% 범위 (SPEC D29·D24·D13). 화면의 비율 지표 옆 "± 값"이다.

최종 모델의 선수별 절대 오차 |예측 − 실제|를 목표 시즌 2024·2025(최종 검증 뒤에는 2026도)에 걸쳐 모으고,
역할·지표·그룹(사용 시즌 1~2 / 3~5)마다 `BAND_QUANTILE` 분위수를 낸다. 집단은 평가(D13)와 같다 —
OPS는 실제 PA ≥ `RATE_MIN_PA`, ERA·K−BB%는 실제 아웃 ≥ `RATE_MIN_OUTS`, 실제값이 결측이면 뺀다.
모델은 바꾸지 않는다(2026 결과는 보고용으로만 쓴다).
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from mlbproj import contracts as c
from mlbproj.evaluate import actual_values

_ERROR_COLUMNS = ["target_season", "role", "metric", "group", "player_id", "abs_error"]
_SHORT_CAREER_MAX = 2  # seasons_1_2 그룹의 n_seasons 상한 (evaluate와 같다)


def empty_bands() -> pd.DataFrame:
    """행이 없는 범위 표 (`contracts.BANDS` 열과 형식)."""
    dtypes = {"str": "str", "int": "int64", "float": "float64"}
    return pd.DataFrame({col.name: pd.Series(dtype=dtypes[col.kind]) for col in c.BANDS.columns})


def band_targets(final_test_done: bool) -> tuple[int, ...]:
    """범위에 쓰는 목표 시즌. 2026은 최종 검증 기록이 있을 때만 (F10)."""
    return (*c.TUNE_TARGETS, *((c.FINAL_TEST_TARGET,) if final_test_done else ()))


def player_errors(
    projections: Mapping[c.Role, pd.DataFrame], seasons: Mapping[c.Role, pd.DataFrame], target: int
) -> pd.DataFrame:
    """목표 시즌 하나의 선수별 절대 오차 (비율 지표, 대표 집단만).

    Args:
        projections: 역할 → 그 목표의 최종 모델 예측 표 (player_id·n_seasons·지표 열을 쓴다).
        seasons: 역할 → 정제 시즌 표.
        target: 목표 시즌.

    Returns:
        열: target_season, role, metric, group, player_id, abs_error. (role, metric, player_id) 순.
    """
    parts: list[pd.DataFrame] = []
    for role in c.ROLES:
        actual = actual_values(seasons[role], role, target).set_index("player_id")
        if role == "bat":
            members = actual.index[actual["pa"] >= c.RATE_MIN_PA]
        else:
            members = actual.index[actual["outs"] >= c.RATE_MIN_OUTS]
        proj = projections[role].set_index("player_id")
        ids = proj.index.intersection(members).sort_values()
        group = np.where(
            proj.loc[ids, "n_seasons"] <= _SHORT_CAREER_MAX, "seasons_1_2", "seasons_3_5"
        )
        for metric in c.BAND_METRICS[role]:
            error = (proj.loc[ids, metric] - actual.loc[ids, metric]).abs()
            keep = error.notna().to_numpy()
            parts.append(
                pd.DataFrame(
                    {
                        "target_season": target,
                        "role": role,
                        "metric": metric,
                        "group": group[keep],
                        "player_id": ids[keep],
                        "abs_error": error.to_numpy()[keep],
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)[_ERROR_COLUMNS]


def error_bands(errors: pd.DataFrame) -> pd.DataFrame:
    """선수별 절대 오차 → 범위 표 (`contracts.BANDS`, 역할·지표·그룹 계약 순, 빈 칸은 n 0·q80 결측)."""
    targets = ",".join(str(int(t)) for t in sorted(set(errors["target_season"])))
    rows: list[dict[str, object]] = []
    for role in c.ROLES:
        for metric in c.BAND_METRICS[role]:
            for group in c.BAND_GROUPS:
                values = errors.loc[
                    (errors["role"] == role)
                    & (errors["metric"] == metric)
                    & (errors["group"] == group),
                    "abs_error",
                ].to_numpy(dtype="float64")
                q80 = float(np.quantile(values, c.BAND_QUANTILE)) if len(values) else np.nan
                rows.append(
                    {
                        "role": role,
                        "metric": metric,
                        "group": group,
                        "n": len(values),
                        "q80": q80,
                        "targets": targets,
                    }
                )
    out = pd.DataFrame(rows, columns=c.BANDS.column_names)
    out["n"] = out["n"].astype("int64")
    out["q80"] = out["q80"].astype("float64")
    return out


def bands_for(
    projections_by_target: Mapping[int, Mapping[c.Role, pd.DataFrame]],
    seasons: Mapping[c.Role, pd.DataFrame],
    targets: Sequence[int],
) -> pd.DataFrame:
    """목표 시즌들의 최종 모델 예측으로 범위 표를 만든다."""
    errors = pd.concat(
        [player_errors(projections_by_target[t], seasons, t) for t in targets], ignore_index=True
    )
    return error_bands(errors)
