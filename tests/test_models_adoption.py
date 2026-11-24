"""확장 모델 채택표 테스트 — 고정·해시·형식 검사 (SPEC F9, PLAN §2-3)."""

from pathlib import Path

import pytest

from mlbproj import contracts as c
from mlbproj.models.params import (
    NO_ADOPTION,
    Adoption,
    adoption_from_json,
    adoption_sha256,
    adoption_to_json,
    load_adoption,
    save_adoption,
    validate_adoption,
)


def _some() -> Adoption:
    adopt = {m: False for m in c.EXT_METRICS}
    adopt["ops"] = True
    adopt["ip"] = True
    return Adoption(version="v1", adopt=adopt)


def test_models_adoption_none_is_valid_and_complete() -> None:
    validate_adoption(NO_ADOPTION)
    assert set(NO_ADOPTION.adopt) == set(c.EXT_METRICS)
    assert not any(NO_ADOPTION.adopt.values())


def test_models_adoption_round_trip_and_stable_hash(tmp_path: Path) -> None:
    path = tmp_path / "config" / "extended_v1.json"
    save_adoption(_some(), path)
    loaded = load_adoption(path)
    assert loaded == _some()
    assert adoption_sha256(loaded) == adoption_sha256(_some())
    # 키 정렬 — 지표 순서가 달라도 같은 문자열
    reordered = Adoption("v1", dict(reversed(list(_some().adopt.items()))))
    assert adoption_to_json(reordered) == adoption_to_json(_some())
    assert path.read_text(encoding="utf-8") == adoption_to_json(_some())


def test_models_adoption_hash_changes_with_any_flag() -> None:
    base = adoption_sha256(_some())
    for metric in c.EXT_METRICS:
        flipped = dict(_some().adopt)
        flipped[metric] = not flipped[metric]
        assert adoption_sha256(Adoption("v1", flipped)) != base


@pytest.mark.parametrize(
    "bad",
    [
        Adoption("", {m: False for m in c.EXT_METRICS}),
        Adoption("v1", {m: False for m in c.EXT_METRICS if m != "ops"}),  # 빠진 지표
        Adoption("v1", {**{m: False for m in c.EXT_METRICS}, "obp": True}),  # 모르는 지표
        Adoption("v1", {**{m: False for m in c.EXT_METRICS}, "ops": 1}),  # type: ignore[dict-item]
    ],
)
def test_models_adoption_validate_rejects(bad: Adoption) -> None:
    with pytest.raises(ValueError):
        validate_adoption(bad)


def test_models_adoption_from_json_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        adoption_from_json('{"adopt": {}, "version": "v1", "extra": 1}')
