"""웹 JSON 계약 검사기.

`contracts/web-data.schema.json`이 쓰는 JSON Schema 키워드만 표준 라이브러리로 검사한다.
지원하지 않는 키워드를 만나면 조용히 넘어가지 않고 오류를 낸다 — 스키마에 새 키워드를 쓰면
여기에 구현을 더해야 한다는 신호다.
"""

import json
import math
import re
from pathlib import Path
from typing import Any

# 검사에 영향이 없는 설명용 키워드
_ANNOTATIONS = frozenset({"$schema", "$id", "title", "description", "$defs"})
_SUPPORTED = frozenset(
    {
        "$ref",
        "type",
        "const",
        "enum",
        "pattern",
        "minLength",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "items",
        "properties",
        "required",
        "additionalProperties",
        "oneOf",
    }
)


def load_schema(path: Path) -> dict[str, Any]:
    """스키마 파일을 읽는다.

    Args:
        path: 스키마 JSON 경로.

    Returns:
        스키마 객체.
    """
    schema: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return schema


def validate(instance: Any, def_name: str, schema: dict[str, Any]) -> list[str]:
    """문서가 스키마의 정의 하나를 따르는지 검사한다.

    Args:
        instance: 검사할 값 (JSON에서 읽은 파이썬 값).
        def_name: `$defs` 안의 정의 이름.
        schema: `load_schema` 결과.

    Returns:
        `"경로: 사유"` 형식의 오류 목록. 비어 있으면 통과다.

    Raises:
        KeyError: 정의가 없을 때.
        ValueError: 지원하지 않는 키워드가 있을 때.
    """
    node = schema["$defs"][def_name]
    errors: list[str] = []
    _check(instance, node, schema, "$", errors)
    return errors


def _type_ok(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return (
            isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
        )
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    raise ValueError(f"지원하지 않는 type: {type_name}")


def _resolve(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    prefix = "#/$defs/"
    if not ref.startswith(prefix):
        raise ValueError(f"지원하지 않는 $ref: {ref}")
    resolved: dict[str, Any] = root["$defs"][ref[len(prefix) :]]
    return resolved


def _check(
    value: Any, node: dict[str, Any], root: dict[str, Any], path: str, errors: list[str]
) -> None:
    unknown = set(node) - _SUPPORTED - _ANNOTATIONS
    if unknown:
        raise ValueError(f"{path}: 지원하지 않는 스키마 키워드 {sorted(unknown)}")

    if "$ref" in node:
        _check(value, _resolve(node["$ref"], root), root, path, errors)

    if "type" in node:
        types = node["type"] if isinstance(node["type"], list) else [node["type"]]
        if not any(_type_ok(value, t) for t in types):
            errors.append(f"{path}: {types} 형이어야 한다 (값 {value!r})")
            return

    if "const" in node and value != node["const"]:
        errors.append(f"{path}: {node['const']!r}이어야 한다")
    if "enum" in node and value not in node["enum"]:
        errors.append(f"{path}: {node['enum']} 중 하나여야 한다 (값 {value!r})")

    if "oneOf" in node:
        matches = 0
        for option in node["oneOf"]:
            sub: list[str] = []
            _check(value, option, root, path, sub)
            matches += not sub
        if matches != 1:
            errors.append(f"{path}: oneOf 중 정확히 하나와 맞아야 한다 ({matches}개 일치)")

    if isinstance(value, str):
        if "minLength" in node and len(value) < node["minLength"]:
            errors.append(f"{path}: 길이 {node['minLength']} 이상이어야 한다")
        if "pattern" in node and not re.search(node["pattern"], value):
            errors.append(f"{path}: 형식 {node['pattern']}과 맞지 않는다")

    if _type_ok(value, "number"):
        if "minimum" in node and value < node["minimum"]:
            errors.append(f"{path}: {node['minimum']} 이상이어야 한다 (값 {value})")
        if "maximum" in node and value > node["maximum"]:
            errors.append(f"{path}: {node['maximum']} 이하여야 한다 (값 {value})")

    if isinstance(value, list):
        if "minItems" in node and len(value) < node["minItems"]:
            errors.append(f"{path}: 항목이 {node['minItems']}개 이상이어야 한다")
        if "maxItems" in node and len(value) > node["maxItems"]:
            errors.append(f"{path}: 항목이 {node['maxItems']}개 이하여야 한다")
        if "items" in node:
            for i, item in enumerate(value):
                _check(item, node["items"], root, f"{path}[{i}]", errors)

    if isinstance(value, dict):
        for key in node.get("required", []):
            if key not in value:
                errors.append(f"{path}: 필수 키 {key} 없음")
        props: dict[str, Any] = node.get("properties", {})
        extra = node.get("additionalProperties", True)
        for key, item in value.items():
            if key in props:
                _check(item, props[key], root, f"{path}.{key}", errors)
            elif extra is False:
                errors.append(f"{path}: 허용되지 않은 키 {key}")
            elif isinstance(extra, dict):
                _check(item, extra, root, f"{path}.{key}", errors)
