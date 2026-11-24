"""출전량 모델 (SPEC F8·D9·D12·N2). 레인 C가 구현한다.

교과서 마르셀 공식을 쓰지 않고(E11), 선형 회귀 계수를 **목표 시즌 Y보다 앞선 목표들**로 적합한다.
numpy 최소제곱(`numpy.linalg.lstsq`, pandas에 포함)만 쓴다.

특성 (대상 표의 한 선수, 목표 시즌 T 기준 — 같은 규칙으로 학습·예측에 쓴다)
- 타자: pt1 = T−1의 pa_adj, pt2 = T−2의 pa_adj (행이 없으면 0, D9 환산값), gap = (T−1) − input_last,
  n_seasons(대상 표), 상수항 1.
- 투수: pt1·pt2 = outs_adj, gap, n_seasons, gs_share = 기간 안 Σgs / Σg (Σg = 0이면 0), 상수항 1.
- 특성 열 순서는 `FEATURES[role]`.

학습 (목표 Y를 예측할 때)
- 학습 목표 T = max(SEASON_FIRST + 2, …) ~ Y−1 (T−2가 데이터 안에 있어야 한다: T ≥ 2021).
- 각 T마다 `cohort.build_cohort(seasons, role, T, debut_seasons)`로 대상을 만들고, 정답은 T의 pa_adj
  (투수 outs_adj), 행이 없으면 0.
- 모든 T의 행을 합쳐 최소제곱으로 계수를 구한다. **season ≥ Y 행은 특성·정답 어디에도 쓰지 않는다**(N2).
- 예측 = 특성 · 계수, 0 미만은 0으로 자른다.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from mlbproj.cohort import build_cohort, expand_window
from mlbproj.contracts import (
    BAT_PLAYING_TIME,
    BATTING_SEASONS,
    COHORTS,
    PIT_PLAYING_TIME,
    PITCHING_SEASONS,
    SEASON_FIRST,
    Role,
    TableSpec,
    check_frame,
)

FEATURES: dict[Role, tuple[str, ...]] = {
    "bat": ("pt1", "pt2", "gap", "n_seasons", "const"),
    "pit": ("pt1", "pt2", "gap", "n_seasons", "gs_share", "const"),
}

# 특성(pt1·pt2)과 정답에 쓰는 출전량 열 — 2020은 ×162/60 환산값 (D9)
_PT_ADJ: dict[Role, str] = {"bat": "pa_adj", "pit": "outs_adj"}
# 예측 표의 출전량 열과 계약
_PT_COLUMN: dict[Role, str] = {"bat": "pa", "pit": "outs"}
_PT_SPECS: dict[Role, TableSpec] = {"bat": BAT_PLAYING_TIME, "pit": PIT_PLAYING_TIME}
# 역할별 정제 표 계약
_SEASON_SPECS: dict[Role, TableSpec] = {"bat": BATTING_SEASONS, "pit": PITCHING_SEASONS}
# 가장 먼 특성 시즌 (pt2 = T−2). 학습 목표는 T−2가 데이터 안에 있는 시즌부터다
_MAX_LAG = 2
# 학습 표의 앞쪽 정수 열
_TRAIN_LEAD: tuple[str, ...] = ("player_id", "train_target")


def _check_seasons(seasons: pd.DataFrame, role: Role) -> None:
    """정제 표가 그 역할의 계약을 지키는지 확인한다. 어기면 ValueError."""
    problems = check_frame(seasons, _SEASON_SPECS[role])
    if problems:
        raise ValueError(f"{role} 정제 표가 계약을 어긴다: " + "; ".join(problems))


def _target_season(cohort: pd.DataFrame, role: Role) -> int:
    """대상 표의 목표 시즌을 돌려준다. 계약 위반, 역할 불일치, 목표 시즌이 정확히 하나가 아니면 ValueError."""
    problems = check_frame(cohort, COHORTS)
    if problems:
        raise ValueError("대상 표가 계약을 어긴다: " + "; ".join(problems))
    if cohort.empty:
        raise ValueError("대상 표가 비어 있어 목표 시즌을 정할 수 없다")
    roles = sorted(set(cohort["role"]))
    if roles != [role]:
        raise ValueError(f"대상 표의 역할이 {role} 하나가 아니다: {roles}")
    targets = sorted({int(t) for t in cohort["target_season"]})
    if len(targets) != 1:
        raise ValueError(f"대상 표의 목표 시즌이 하나가 아니다: {targets}")
    return targets[0]


def _empty_training(role: Role) -> pd.DataFrame:
    """열·형식만 있는 빈 학습 표."""
    columns: dict[str, pd.Series] = {name: pd.Series(dtype="int64") for name in _TRAIN_LEAD}
    for name in (*FEATURES[role], "label"):
        columns[name] = pd.Series(dtype="float64")
    return pd.DataFrame(columns)


def playing_time_features(cohort: pd.DataFrame, seasons: pd.DataFrame, role: Role) -> pd.DataFrame:
    """대상 표의 선수마다 특성을 만든다. 열: player_id, 그다음 `FEATURES[role]` 순서 (모두 실수, 결측 없음).

    Raises:
        ValueError: 입력 표가 계약을 어기거나 대상 표의 역할·목표 시즌이 하나가 아닐 때.
    """
    target = _target_season(cohort, role)
    _check_seasons(seasons, role)
    # N2: 목표 시즌 이후 행은 펼치기 전에 잘라 낸다. expand_window는 (선수, 시즌) 일대일 병합이라
    # 목표 이후 시즌에 같은 선수·시즌이 다른 리그로 한 번 더 있으면 오류를 내 결과가 달라진다
    past = seasons[seasons["season"] < target]
    # 대상마다 기간(window_first ~ T−1)의 시즌을 한 행씩. 미출전 시즌의 출전량(pa_adj·outs_adj·g·gs)은 0이고,
    # 기간 밖 행은 들어오지 않는다. 대상 표의 기간(Y−WINDOW ~ Y−1)도 여기서 확인된다
    rows = expand_window(cohort, past, role)
    base = cohort.set_index("player_id").sort_index()

    # T−1·T−2 출전량. T−2가 기간 밖이면(그 역할의 첫 기록이 T−1) 열·값이 없어 0이다
    lagged = (
        rows.pivot(index="player_id", columns="season", values=_PT_ADJ[role])
        .reindex(index=base.index, columns=[target - 1, target - _MAX_LAG])
        .fillna(0.0)
    )
    values: dict[str, pd.Series] = {
        "pt1": lagged[target - 1],
        "pt2": lagged[target - _MAX_LAG],
        "gap": (target - 1 - base["input_last"]).astype("float64"),
        "n_seasons": base["n_seasons"].astype("float64"),
        "const": pd.Series(1.0, index=base.index),
    }
    if role == "pit":
        # 기간 안 Σgs / Σg — 미출전 시즌은 0으로 채워져 합에 영향이 없다. Σg = 0이면 0
        sums = rows.groupby("player_id")[["g", "gs"]].sum().astype("float64")
        games = sums["g"].where(sums["g"] > 0)
        values["gs_share"] = (sums["gs"] / games).fillna(0.0).reindex(base.index)

    # player_id 순, 특성 열은 FEATURES 순서
    out = pd.DataFrame(values, index=base.index)[list(FEATURES[role])]
    return out.rename_axis("player_id").reset_index()


def training_frame(
    seasons: pd.DataFrame,
    role: Role,
    target_season: int,
    debut_seasons: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """목표 Y의 학습 표: 열 = player_id, train_target, `FEATURES[role]`, label. 학습 목표가 없으면 빈 표."""
    _check_seasons(seasons, role)
    # N2: 목표 시즌 Y 이후 행은 특성·정답 어디에도 쓰지 않는다 — 처음에 잘라 낸다
    past = seasons[seasons["season"] < target_season]
    if past.empty:
        return _empty_training(role)
    # T−2가 데이터 안에 있어야 한다: T ≥ max(SEASON_FIRST, 데이터 첫 시즌) + 2
    first_target = max(SEASON_FIRST, int(past["season"].min())) + _MAX_LAG

    parts: list[pd.DataFrame] = []
    for train_target in range(first_target, target_season):
        cohort = build_cohort(past, role, train_target, debut_seasons)
        if cohort.empty:  # 기간 안에 기록이 하나도 없는 목표 (데이터 공백)
            continue
        part = playing_time_features(cohort, past, role)
        # 정답 = T의 환산 출전량, 그 시즌 행이 없으면 0 (T < Y라 목표 시즌 이후 행이 아니다)
        actual = past.loc[past["season"] == train_target].set_index("player_id")[_PT_ADJ[role]]
        part["label"] = actual.reindex(part["player_id"]).fillna(0.0).to_numpy()
        part.insert(1, "train_target", train_target)
        parts.append(part)
    if not parts:
        return _empty_training(role)
    return pd.concat(parts, ignore_index=True)


def fit_coefficients(train: pd.DataFrame, role: Role) -> np.ndarray:
    """최소제곱 계수 (길이 = len(FEATURES[role])).

    Raises:
        ValueError: 학습 행이 특성 수보다 적을 때.
    """
    names = list(FEATURES[role])
    if len(train) < len(names):
        raise ValueError(f"학습 행 {len(train)}개가 특성 {len(names)}개보다 적다")
    x = train[names].to_numpy(dtype="float64")
    y = train["label"].to_numpy(dtype="float64")
    # 결측·무한대가 있으면 lstsq가 알아보기 힘든 SVD 오류를 내므로 먼저 막는다
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("학습 표의 특성·정답에 결측이나 무한대 값이 있다")
    coef: np.ndarray = np.linalg.lstsq(x, y, rcond=None)[0]
    return coef


def predict_playing_time(
    cohort: pd.DataFrame,
    seasons: pd.DataFrame,
    role: Role,
    debut_seasons: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """대상 표의 출전량 예측. 타자 `contracts.BAT_PLAYING_TIME`(pa), 투수 `PIT_PLAYING_TIME`(outs).

    학습(`training_frame` → `fit_coefficients`)과 예측을 한 번에 한다. 결과는 player_id 순, 0 이상.

    Raises:
        ValueError: 입력 표가 계약을 어기거나 학습 행이 모자랄 때.
    """
    # 입력 표를 먼저 확인한다 (특성은 player_id 순)
    features = playing_time_features(cohort, seasons, role)
    target = _target_season(cohort, role)
    coef = fit_coefficients(training_frame(seasons, role, target, debut_seasons), role)
    raw = features[list(FEATURES[role])].to_numpy(dtype="float64") @ coef
    out = pd.DataFrame(
        {
            "player_id": features["player_id"],
            "target_season": target,
            _PT_COLUMN[role]: np.maximum(raw, 0.0),  # 0 미만은 0으로 자른다
        }
    )
    return out[_PT_SPECS[role].column_names]
