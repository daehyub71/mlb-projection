"""웹 JSON 계약 검사기와 스키마 파일 테스트 (PLAN §5 계약 층)."""

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from mlbproj import contracts
from mlbproj.schema import load_schema, validate

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "tests" / "fixtures" / "synthetic" / "web"

MINI: dict[str, Any] = {
    "$defs": {
        "Item": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "label"],
            "properties": {
                "id": {"type": "integer", "minimum": 1},
                "label": {"type": "string", "minLength": 1, "pattern": "^[a-z]+$"},
                "score": {"type": ["number", "null"], "maximum": 10},
                "kind": {"enum": ["a", "b"]},
                "version": {"const": "1"},
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
                "child": {
                    "type": ["object", "null"],
                    "additionalProperties": False,
                    "required": ["x"],
                    "properties": {"x": {"type": "integer"}},
                },
                "either": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
                "extra": {"type": "object", "additionalProperties": {"type": "number"}},
            },
        },
        "Ref": {"$ref": "#/$defs/Item"},
    }
}


def _load(name: str) -> Any:
    return json.loads((SAMPLES / name).read_text(encoding="utf-8"))


def test_schema_valid_minimal_passes() -> None:
    assert validate({"id": 1, "label": "ok"}, "Item", MINI) == []


@pytest.mark.parametrize(
    ("instance", "fragment"),
    [
        ({"id": "1", "label": "ok"}, "$.id"),
        ({"id": True, "label": "ok"}, "$.id"),  # bool은 정수가 아니다
        ({"id": 1.0, "label": "ok"}, "$.id"),  # 정수 열은 int로 써야 한다
        ({"id": 0, "label": "ok"}, "$.id"),
        ({"label": "ok"}, "id"),
        ({"id": 1, "label": "ok", "nope": 1}, "nope"),
        ({"id": 1, "label": ""}, "$.label"),
        ({"id": 1, "label": "ABC"}, "$.label"),
        ({"id": 1, "label": "ok", "score": 11}, "$.score"),
        ({"id": 1, "label": "ok", "score": math.nan}, "$.score"),
        ({"id": 1, "label": "ok", "score": math.inf}, "$.score"),
        ({"id": 1, "label": "ok", "kind": "c"}, "$.kind"),
        ({"id": 1, "label": "ok", "version": "2"}, "$.version"),
        ({"id": 1, "label": "ok", "tags": ["a", "b", "c"]}, "$.tags"),
        ({"id": 1, "label": "ok", "tags": ["a", 1]}, "$.tags[1]"),
        ({"id": 1, "label": "ok", "child": {}}, "x"),
        ({"id": 1, "label": "ok", "either": 1.5}, "$.either"),
        ({"id": 1, "label": "ok", "extra": {"k": "v"}}, "$.extra.k"),
    ],
)
def test_schema_invalid_reports_path(instance: dict[str, Any], fragment: str) -> None:
    errors = validate(instance, "Item", MINI)
    assert errors, f"통과하면 안 된다: {instance}"
    assert any(fragment in e for e in errors), errors


def test_schema_nullable_object_skips_object_keywords() -> None:
    instance = {"id": 1, "label": "ok", "child": None, "score": None}
    assert validate(instance, "Item", MINI) == []


def test_schema_ref_resolves() -> None:
    assert validate({"id": 1, "label": "ok"}, "Ref", MINI) == []
    assert validate({"id": 1}, "Ref", MINI) != []


def test_schema_unknown_def_raises() -> None:
    with pytest.raises(KeyError):
        validate({}, "Missing", MINI)


def test_schema_unsupported_keyword_raises() -> None:
    schema = {"$defs": {"X": {"type": "string", "format": "date"}}}
    with pytest.raises(ValueError, match="format"):
        validate("2026-01-01", "X", schema)


def test_schema_file_refs_and_defs_exist() -> None:
    schema = load_schema(ROOT / contracts.WEB_SCHEMA_PATH)
    defs = schema["$defs"]
    for def_name in [*contracts.WEB_FILES.values(), contracts.WEB_PLAYER_DEF]:
        assert def_name in defs
    text = json.dumps(schema)
    for ref in {part.split('"')[0] for part in text.split('"$ref": "#/$defs/')[1:]}:
        assert ref in defs, ref
    assert defs["Meta"]["properties"]["schema_version"]["const"] == contracts.WEB_SCHEMA_VERSION


@pytest.mark.parametrize(
    ("sample", "def_name"),
    [
        ("batters.json", "BatTable"),
        ("pitchers.json", "PitTable"),
        ("validation.json", "ValidationReport"),
        ("player_bat.json", "PlayerDetail"),
        ("player_pit.json", "PlayerDetail"),
    ],
)
def test_schema_synthetic_samples_pass(sample: str, def_name: str) -> None:
    schema = load_schema(ROOT / contracts.WEB_SCHEMA_PATH)
    assert validate(_load(sample), def_name, schema) == []


def test_schema_rejects_pitcher_row_in_batter_table() -> None:
    schema = load_schema(ROOT / contracts.WEB_SCHEMA_PATH)
    doc = _load("batters.json")
    doc["rows"].append(_load("pitchers.json")["rows"][0])
    assert validate(doc, "BatTable", schema) != []


def test_schema_rejects_bad_detail() -> None:
    schema = load_schema(ROOT / contracts.WEB_SCHEMA_PATH)
    base = _load("player_bat.json")

    extra = copy.deepcopy(base)
    extra["bat"]["projection"]["wrc_plus"] = 120
    assert validate(extra, "PlayerDetail", schema) != []

    six = copy.deepcopy(base)
    six["bat"]["history"] = six["bat"]["history"] * 3
    assert validate(six, "PlayerDetail", schema) != []

    bad_pos = copy.deepcopy(base)
    bad_pos["fielding"][0]["outs_by_pos"]["DH"] = 10
    assert validate(bad_pos, "PlayerDetail", schema) != []

    no_meta = copy.deepcopy(base)
    del no_meta["meta"]["params_sha256"]
    assert validate(no_meta, "PlayerDetail", schema) != []
