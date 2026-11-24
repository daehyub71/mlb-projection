"""원본 저장과 매니페스트 (SPEC F1·D6·N1).

원본은 받은 바이트 그대로 저장한다. 매니페스트(`<raw_dir>/manifest.json`)에는 파일마다
요청 URL·받은 시각(UTC)·sha256·크기를 남긴다. 키는 원본 디렉토리 기준 상대 경로(posix)다.
두 수집 모듈(statsapi·savant)이 이 모듈만 거쳐 저장한다.
"""

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from mlbproj.contracts import MANIFEST_NAME


class Getter(Protocol):
    """URL을 받아 응답 바이트를 돌려주는 것 (collect.http.Fetcher)."""

    def get(self, url: str) -> bytes:
        """URL의 응답 바이트를 돌려준다."""
        ...


@dataclass(frozen=True)
class ManifestEntry:
    """매니페스트의 파일 하나.

    Attributes:
        url: 요청 URL.
        fetched_at: 받은 시각 (UTC, `YYYY-MM-DDTHH:MM:SSZ`).
        sha256: 파일 내용의 sha256.
        bytes: 파일 크기.
    """

    url: str
    fetched_at: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class CollectOutcome:
    """원본 하나를 확보한 결과.

    Attributes:
        path: 원본 디렉토리 기준 상대 경로.
        fetched: 새로 받았으면 True, 캐시를 썼으면 False.
        entry: 매니페스트 항목.
    """

    path: Path
    fetched: bool
    entry: ManifestEntry


def utc_now() -> datetime:
    """현재 UTC 시각."""
    return datetime.now(UTC)


def file_sha256(path: Path) -> str:
    """파일 내용의 sha256을 돌려준다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_file(raw_dir: Path) -> Path:
    return raw_dir / MANIFEST_NAME


def _key(rel: Path) -> str:
    return rel.as_posix()


def load_manifest(raw_dir: Path) -> dict[str, ManifestEntry]:
    """매니페스트를 읽는다. 없으면 빈 사전.

    Args:
        raw_dir: 원본 디렉토리.

    Returns:
        상대 경로(posix) → 항목.
    """
    path = _manifest_file(raw_dir)
    if not path.exists():
        return {}
    doc: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return {key: ManifestEntry(**value) for key, value in doc.items()}


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_manifest(raw_dir: Path, manifest: dict[str, ManifestEntry]) -> None:
    doc = {key: asdict(manifest[key]) for key in sorted(manifest)}
    text = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
    _write_atomic(_manifest_file(raw_dir), text.encode("utf-8"))


def save_raw(
    raw_dir: Path,
    rel: Path,
    url: str,
    body: bytes,
    *,
    now: Callable[[], datetime] = utc_now,
) -> ManifestEntry:
    """원본을 저장하고 매니페스트를 갱신한다.

    Args:
        raw_dir: 원본 디렉토리.
        rel: 원본 디렉토리 기준 상대 경로.
        url: 요청 URL.
        body: 응답 바이트 (그대로 저장한다).
        now: 현재 시각 (테스트용).

    Returns:
        새 매니페스트 항목.
    """
    _write_atomic(raw_dir / rel, body)
    entry = ManifestEntry(
        url=url,
        fetched_at=now().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sha256=hashlib.sha256(body).hexdigest(),
        bytes=len(body),
    )
    manifest = load_manifest(raw_dir)
    manifest[_key(rel)] = entry
    _write_manifest(raw_dir, manifest)
    return entry


def is_cached(raw_dir: Path, rel: Path, url: str) -> bool:
    """같은 요청으로 받은 원본이 손상 없이 있는지.

    Args:
        raw_dir: 원본 디렉토리.
        rel: 상대 경로.
        url: 이번 요청 URL. 매니페스트의 URL과 다르면 캐시가 아니다.

    Returns:
        파일이 있고, 매니페스트 URL이 같고, sha256이 맞으면 True.
    """
    path = raw_dir / rel
    entry = load_manifest(raw_dir).get(_key(rel))
    if not path.exists() or entry is None or entry.url != url:
        return False
    return file_sha256(path) == entry.sha256


def fetch_to_raw(
    getter: Getter,
    raw_dir: Path,
    rel: Path,
    url: str,
    *,
    check: Callable[[bytes], None] | None = None,
    force: bool = False,
    now: Callable[[], datetime] = utc_now,
) -> CollectOutcome:
    """캐시가 없으면 받아서 검사한 뒤 저장한다.

    Args:
        getter: 요청 객체.
        raw_dir: 원본 디렉토리.
        rel: 상대 경로.
        url: 요청 URL.
        check: 저장 전 응답 검사. 형식이 틀리면 ValueError를 낸다. 그때는 아무것도 저장하지 않는다.
        force: 캐시가 있어도 다시 받는다.
        now: 현재 시각 (테스트용).

    Returns:
        확보 결과.
    """
    if not force and is_cached(raw_dir, rel, url):
        return CollectOutcome(path=rel, fetched=False, entry=load_manifest(raw_dir)[_key(rel)])
    body = getter.get(url)
    if check is not None:
        check(body)
    entry = save_raw(raw_dir, rel, url, body, now=now)
    return CollectOutcome(path=rel, fetched=True, entry=entry)


def remove_raw(raw_dir: Path, rel: Path) -> None:
    """원본 파일과 매니페스트 항목을 지운다. 없으면 아무것도 하지 않는다."""
    (raw_dir / rel).unlink(missing_ok=True)
    manifest = load_manifest(raw_dir)
    if manifest.pop(_key(rel), None) is not None:
        _write_manifest(raw_dir, manifest)


def read_json(raw_dir: Path, rel: Path) -> Any:
    """JSON 원본을 읽는다."""
    return json.loads((raw_dir / rel).read_text(encoding="utf-8"))


def read_csv(raw_dir: Path, rel: Path) -> pd.DataFrame:
    """Savant CSV 원본을 읽는다. 첫 줄의 BOM을 떼고, 값은 pandas가 추론한다."""
    return pd.read_csv(raw_dir / rel, encoding="utf-8-sig")
