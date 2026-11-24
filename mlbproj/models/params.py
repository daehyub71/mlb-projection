"""모델 파라미터 (SPEC D10·D11, PLAN §2-3).

2024·2025 과거 검증으로 조정한 뒤 `config/params_v1.json`으로 고정한다. 2026 최종 검증은 커밋된
파라미터 파일로만 돈다. 직렬화는 키 정렬·고정 형식이라 같은 값이면 언제나 같은 sha256이 나온다.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlbproj.contracts import BAT_COMPONENTS, ERA_METHODS, PIT_COMPONENTS, WINDOW

_PIT_REGRESS_KEYS: tuple[str, ...] = (*PIT_COMPONENTS, "er")


@dataclass(frozen=True)
class RateParams:
    """한 역할의 비율 모델 파라미터 (F7).

    Attributes:
        weights: 시즌 가중치. [Y−1, Y−2, …, Y−5] 순서, 길이 `WINDOW`. 0 이상이고 합이 0보다 크다.
        regress: 성분 → 리그 평균 회귀 표본 크기(0 이상). 타자는 타석, 투수 성분은 상대 타자, "er"은 아웃.
    """

    weights: tuple[float, ...]
    regress: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Params:
    """모델 파라미터 전체.

    Attributes:
        version: 파라미터 이름 (예: "v1").
        bat: 타자 비율 파라미터 (회귀 키 = `BAT_COMPONENTS`).
        pit: 투수 비율 파라미터 (회귀 키 = `PIT_COMPONENTS` + "er").
        era_method: ERA 후보 (`ERA_METHODS`, D11).
    """

    version: str
    bat: RateParams
    pit: RateParams
    era_method: str


# 조정 전 출발점: 5·4·3·2·1 가중(원안), 회귀 표본은 임시값
DEFAULT_PARAMS = Params(
    version="v0",
    bat=RateParams(weights=(5.0, 4.0, 3.0, 2.0, 1.0), regress={k: 200.0 for k in BAT_COMPONENTS}),
    pit=RateParams(
        weights=(5.0, 4.0, 3.0, 2.0, 1.0),
        regress={**{k: 200.0 for k in PIT_COMPONENTS}, "er": 150.0},
    ),
    era_method="er",
)


def _check_rate(name: str, rate: RateParams, keys: tuple[str, ...]) -> None:
    if len(rate.weights) != WINDOW:
        raise ValueError(f"{name}.weights는 {WINDOW}개여야 한다 ({len(rate.weights)}개)")
    if any(not math.isfinite(w) or w < 0 for w in rate.weights) or sum(rate.weights) <= 0:
        raise ValueError(f"{name}.weights는 0 이상이고 합이 0보다 커야 한다: {rate.weights}")
    if set(rate.regress) != set(keys):
        raise ValueError(f"{name}.regress 키가 {sorted(keys)}와 다르다: {sorted(rate.regress)}")
    bad = {k: v for k, v in rate.regress.items() if not math.isfinite(v) or v < 0}
    if bad:
        raise ValueError(f"{name}.regress는 0 이상이어야 한다: {bad}")


def validate_params(params: Params) -> None:
    """파라미터가 형식을 지키는지 확인한다.

    Raises:
        ValueError: 버전이 비었거나, 가중치·회귀 표본·ERA 방식이 규칙을 어길 때.
    """
    if not params.version:
        raise ValueError("version이 비었다")
    _check_rate("bat", params.bat, BAT_COMPONENTS)
    _check_rate("pit", params.pit, _PIT_REGRESS_KEYS)
    if params.era_method not in ERA_METHODS:
        raise ValueError(f"era_method는 {ERA_METHODS} 중 하나: {params.era_method!r}")


def _rate_doc(rate: RateParams) -> dict[str, Any]:
    return {
        "regress": {k: float(rate.regress[k]) for k in sorted(rate.regress)},
        "weights": [float(w) for w in rate.weights],
    }


def to_json(params: Params) -> str:
    """키를 정렬한 JSON 문자열 (같은 값이면 같은 문자열)."""
    validate_params(params)
    doc = {
        "bat": _rate_doc(params.bat),
        "era_method": params.era_method,
        "pit": _rate_doc(params.pit),
        "version": params.version,
    }
    return json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def _rate_from(doc: Any, name: str) -> RateParams:
    if not isinstance(doc, dict) or set(doc) != {"regress", "weights"}:
        raise ValueError(f"{name}는 regress·weights만 가져야 한다")
    return RateParams(
        weights=tuple(float(w) for w in doc["weights"]),
        regress={str(k): float(v) for k, v in doc["regress"].items()},
    )


def from_json(text: str) -> Params:
    """`to_json` 결과를 읽는다.

    Raises:
        ValueError: 모르는 키가 있거나 형식이 틀릴 때.
    """
    doc = json.loads(text)
    if not isinstance(doc, dict) or set(doc) != {"bat", "era_method", "pit", "version"}:
        raise ValueError(
            f"파라미터 최상위 키가 다르다: {sorted(doc) if isinstance(doc, dict) else doc!r}"
        )
    params = Params(
        version=str(doc["version"]),
        bat=_rate_from(doc["bat"], "bat"),
        pit=_rate_from(doc["pit"], "pit"),
        era_method=str(doc["era_method"]),
    )
    validate_params(params)
    return params


def params_sha256(params: Params) -> str:
    """파라미터 직렬화의 sha256 (TASKS·산출물 meta에 남긴다, N1)."""
    return hashlib.sha256(to_json(params).encode("utf-8")).hexdigest()


def save_params(params: Params, path: Path) -> None:
    """파라미터를 파일로 쓴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(params), encoding="utf-8")


def load_params(path: Path) -> Params:
    """파라미터 파일을 읽는다."""
    return from_json(path.read_text(encoding="utf-8"))
