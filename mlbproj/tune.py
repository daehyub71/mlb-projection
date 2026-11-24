"""파라미터 조정 (SPEC D10·D11, PLAN §2-3) — 목표 2024·2025만 쓴다.

격자 탐색으로 고른다. 후보 수를 작게 둔다(과적합을 줄이려고 두 해 평균 MAE로 고른다).

- 타자: 시즌 가중치 × 회귀 표본 R(모든 성분 같은 값) → OPS MAE(대표 집단, "all").
- 타자 홈런: 위 값을 고정하고 HR 성분의 R만 따로 → HR MAE. 단 OPS 평균 MAE가 위 단계보다
  나빠지지 않는 후보 중에서만 고른다(2026-11-24 사용자 결정 — OPS가 통과 기준 지표, SPEC §수용 기준 2).
- 투수 성분: 시즌 가중치 × R(K·BB·HBP·HR 같은 값) → K−BB% MAE.
- 투수 ERA: 위에서 고른 가중치·R 위에서 ERA 방식("er"/"fip") × R_er → ERA MAE.

출전량 모델(F8)은 계수를 학습 기간에서 맞추므로 여기서 조정하지 않는다.
"""

from collections.abc import Iterator, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from mlbproj import contracts as c
from mlbproj.backtest import Tables, baseline_projections, cohort_for, weighted_projection
from mlbproj.evaluate import evaluate
from mlbproj.models.params import DEFAULT_PARAMS, Params, RateParams
from mlbproj.models.playing_time import predict_playing_time

# 시즌 가중치 후보 [Y−1 … Y−5]
WEIGHT_CANDIDATES: tuple[tuple[float, ...], ...] = (
    (5.0, 4.0, 3.0, 2.0, 1.0),  # 원안 (5시즌 선형)
    (5.0, 4.0, 3.0, 0.0, 0.0),  # 3시즌 마르셀식
    (8.0, 4.0, 2.0, 1.0, 0.5),  # 최근 시즌을 더 크게 (기하)
    (1.0, 1.0, 1.0, 1.0, 1.0),  # 같은 무게
)
REGRESS_CANDIDATES: tuple[float, ...] = (300.0, 600.0, 1200.0, 2400.0, 4800.0, 9600.0)
REGRESS_ER_CANDIDATES: tuple[float, ...] = (150.0, 600.0, 2400.0, 9600.0)
TUNE_TARGETS: tuple[int, ...] = c.TUNE_TARGETS


def _rate(
    weights: tuple[float, ...],
    regress: float,
    keys: Sequence[str],
    er: float | None = None,
    hr: float | None = None,
) -> RateParams:
    values = {k: regress for k in keys}
    if er is not None:
        values["er"] = er
    if hr is not None:
        values["hr"] = hr
    return RateParams(weights=weights, regress=values)


def candidate_params() -> Iterator[Params]:
    """타자 격자(가중치 × R) 후보. 투수는 기본값 그대로 두고 `tune`이 따로 돌린다."""
    for weights in WEIGHT_CANDIDATES:
        for regress in REGRESS_CANDIDATES:
            yield replace(DEFAULT_PARAMS, bat=_rate(weights, regress, c.BAT_COMPONENTS))


def best_row(report: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """MAE 열들의 평균이 가장 작은 행. 같으면 먼저 나온 행.

    Raises:
        ValueError: 표가 비었을 때.
    """
    if report.empty:
        raise ValueError("조정 결과가 비었다")
    mean = report[list(columns)].mean(axis=1).to_numpy()
    return report.iloc[int(np.argmin(mean))]  # argmin은 같은 값이면 첫 위치


def best_row_within(
    report: pd.DataFrame, columns: Sequence[str], guard: Sequence[str], limit: float
) -> pd.Series:
    """guard 열 평균이 limit 이하(같으면 통과)인 행 중 columns 평균이 가장 작은 행.

    Raises:
        ValueError: 기준을 지키는 행이 없을 때.
    """
    passed = report[report[list(guard)].mean(axis=1) <= limit]
    if passed.empty:
        raise ValueError(f"기준(평균 {limit})을 지키는 후보가 없다")
    return best_row(passed, columns)


def _maes(
    tables: Tables,
    role: c.Role,
    target: int,
    base: pd.DataFrame,
    params: Params,
    pt: pd.DataFrame,
) -> dict[str, float]:
    """가중 모델의 지표별 MAE (대표 집단 "all")."""
    weighted = weighted_projection(tables, role, target, params, playing_time=pt)
    ev = evaluate(
        pd.concat([base, weighted], ignore_index=True),
        tables.seasons[role],
        cohort_for(tables, role, target),
        role,
    )
    rows = ev[(ev["model"] == "weighted") & (ev["group"] == "all")]
    return {str(m): float(v) for m, v in zip(rows["metric"], rows["mae"], strict=True)}


def tune(tables: Tables, targets: Sequence[int] = TUNE_TARGETS) -> tuple[Params, pd.DataFrame]:
    """격자 탐색으로 파라미터를 고른다.

    Returns:
        (고른 파라미터(version "v1"), 모든 후보의 결과 표 — 열: stage, metric, weights, regress,
        regress_hr, regress_er, era_method, guard, mae_<목표>…, guard_<목표>…).

    Raises:
        ValueError: 목표에 최종 검증 시즌(2026) 이후가 들어 있을 때.
    """
    if any(t >= c.FINAL_TEST_TARGET for t in targets):
        raise ValueError("조정에는 2024·2025만 쓴다 — 2026은 최종 검증 전용 (SPEC F10)")
    cache: dict[tuple[c.Role, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for role in c.ROLES:
        for target in targets:
            cohort = cohort_for(tables, role, target)
            pt = predict_playing_time(cohort, tables.seasons[role], role, tables.debuts)
            cache[(role, target)] = (baseline_projections(tables, role, target), pt)

    cols = [f"mae_{t}" for t in targets]
    records: list[dict[str, object]] = []

    def run(
        stage: str,
        role: c.Role,
        metric: str,
        params: Params,
        weights: tuple[float, ...],
        regress: float,
        er: float | None,
        method: str | None,
        hr: float | None = None,
        guard: str | None = None,
    ) -> None:
        rec: dict[str, object] = {
            "stage": stage,
            "metric": metric,
            "weights": str(weights),
            "regress": regress,
            "regress_hr": hr,
            "regress_er": er,
            "era_method": method,
            "guard": guard,
        }
        for t in targets:
            base, pt = cache[(role, t)]
            maes = _maes(tables, role, t, base, params, pt)
            rec[f"mae_{t}"] = maes[metric]
            rec[f"guard_{t}"] = maes[guard] if guard else None
        records.append(rec)

    # 1) 타자 — OPS
    for w in WEIGHT_CANDIDATES:
        for r in REGRESS_CANDIDATES:
            run(
                "bat_ops",
                "bat",
                "ops",
                replace(DEFAULT_PARAMS, bat=_rate(w, r, c.BAT_COMPONENTS)),
                w,
                r,
                None,
                None,
            )
    report = pd.DataFrame(records)
    stage1 = report[report["stage"] == "bat_ops"]
    best_bat = best_row(stage1, cols)
    bat_w = tuple(float(x) for x in str(best_bat["weights"]).strip("()").split(","))
    bat_r = float(best_bat["regress"])

    # 1-2) 타자 홈런 — HR 성분 R만 바꾼다, OPS 평균 MAE는 1단계 최선 이하로 지킨다
    for r_hr in REGRESS_CANDIDATES:
        params = replace(DEFAULT_PARAMS, bat=_rate(bat_w, bat_r, c.BAT_COMPONENTS, hr=r_hr))
        run("bat_hr", "bat", "hr", params, bat_w, bat_r, None, None, hr=r_hr, guard="ops")
    report = pd.DataFrame(records)
    ops_limit = float(stage1[cols].mean(axis=1).min())
    guard_cols = [f"guard_{t}" for t in targets]
    best_hr = best_row_within(report[report["stage"] == "bat_hr"], cols, guard_cols, ops_limit)

    # 2) 투수 성분 — K−BB% (ER 설정은 기본값, K−BB%에는 영향 없음)
    er0 = DEFAULT_PARAMS.pit.regress["er"]
    for w in WEIGHT_CANDIDATES:
        for r in REGRESS_CANDIDATES:
            params = replace(DEFAULT_PARAMS, pit=_rate(w, r, c.PIT_COMPONENTS, er0))
            run("pit_kbb", "pit", "k_bb_pct", params, w, r, er0, DEFAULT_PARAMS.era_method)
    report = pd.DataFrame(records)
    best_pit = best_row(report[report["stage"] == "pit_kbb"], cols)
    pit_w = tuple(float(x) for x in str(best_pit["weights"]).strip("()").split(","))
    pit_r = float(best_pit["regress"])

    # 3) 투수 ERA — 방식 × R_er
    for method in c.ERA_METHODS:
        for er in REGRESS_ER_CANDIDATES:
            params = replace(
                DEFAULT_PARAMS, pit=_rate(pit_w, pit_r, c.PIT_COMPONENTS, er), era_method=method
            )
            run("pit_era", "pit", "era", params, pit_w, pit_r, er, method)
    report = pd.DataFrame(records)
    best_era = best_row(report[report["stage"] == "pit_era"], cols)

    chosen = Params(
        version="v1",
        bat=_rate(bat_w, bat_r, c.BAT_COMPONENTS, hr=float(best_hr["regress_hr"])),
        pit=_rate(pit_w, pit_r, c.PIT_COMPONENTS, float(best_era["regress_er"])),
        era_method=str(best_era["era_method"]),
    )
    return chosen, report
