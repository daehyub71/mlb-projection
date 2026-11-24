"""모델 파라미터(models/params.py) 테스트 — 고정·해시·형식 검사 (SPEC D10, PLAN §2-3)."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mlbproj import contracts as c
from mlbproj.models.params import (
    DEFAULT_PARAMS,
    Params,
    RateParams,
    from_json,
    load_params,
    params_sha256,
    save_params,
    to_json,
    validate_params,
)


def test_models_params_default_is_valid_and_complete() -> None:
    validate_params(DEFAULT_PARAMS)
    assert len(DEFAULT_PARAMS.bat.weights) == c.WINDOW
    assert set(DEFAULT_PARAMS.bat.regress) == set(c.BAT_COMPONENTS)
    assert set(DEFAULT_PARAMS.pit.regress) == {*c.PIT_COMPONENTS, "er"}
    assert DEFAULT_PARAMS.era_method in c.ERA_METHODS


def test_models_params_json_round_trip_and_stable_hash(tmp_path: Path) -> None:
    text = to_json(DEFAULT_PARAMS)
    assert from_json(text) == DEFAULT_PARAMS
    assert to_json(from_json(text)) == text
    doc = json.loads(text)
    assert list(doc) == sorted(doc)  # 키 정렬
    path = tmp_path / "p.json"
    save_params(DEFAULT_PARAMS, path)
    assert load_params(path) == DEFAULT_PARAMS
    assert params_sha256(DEFAULT_PARAMS) == params_sha256(load_params(path))
    assert len(params_sha256(DEFAULT_PARAMS)) == 64


def test_models_params_hash_changes_with_any_value() -> None:
    changed = replace(
        DEFAULT_PARAMS, bat=replace(DEFAULT_PARAMS.bat, weights=(5.0, 4.0, 3.0, 2.0, 0.5))
    )
    assert params_sha256(changed) != params_sha256(DEFAULT_PARAMS)


@pytest.mark.parametrize(
    "bad",
    [
        replace(DEFAULT_PARAMS, era_method="xera"),
        replace(
            DEFAULT_PARAMS,
            bat=RateParams(weights=(5.0, 4.0, 3.0), regress=dict(DEFAULT_PARAMS.bat.regress)),
        ),
        replace(
            DEFAULT_PARAMS,
            bat=RateParams(
                weights=(5.0, 4.0, 3.0, 2.0, -1.0), regress=dict(DEFAULT_PARAMS.bat.regress)
            ),
        ),
        replace(
            DEFAULT_PARAMS,
            bat=RateParams(
                weights=(0.0, 0.0, 0.0, 0.0, 0.0), regress=dict(DEFAULT_PARAMS.bat.regress)
            ),
        ),
        replace(
            DEFAULT_PARAMS,
            bat=RateParams(weights=DEFAULT_PARAMS.bat.weights, regress={"h1": 100.0}),
        ),
        replace(
            DEFAULT_PARAMS,
            pit=RateParams(
                weights=DEFAULT_PARAMS.pit.weights,
                regress={**DEFAULT_PARAMS.pit.regress, "er": -5.0},
            ),
        ),
        replace(DEFAULT_PARAMS, version=""),
    ],
)
def test_models_params_validate_rejects(bad: Params) -> None:
    with pytest.raises(ValueError):
        validate_params(bad)


def test_models_params_from_json_rejects_unknown_keys() -> None:
    doc = json.loads(to_json(DEFAULT_PARAMS))
    doc["surprise"] = 1
    with pytest.raises(ValueError):
        from_json(json.dumps(doc))
