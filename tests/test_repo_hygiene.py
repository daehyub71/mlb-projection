"""공개 저장소 위생 검사 (SPEC N4·E13, PLAN P4)."""

import csv
import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from mlbproj import contracts as c

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RAW = ROOT / "tests" / "fixtures" / "raw"
MAX_PLAYERS_PER_FILE = 12


def test_repo_hygiene_gitignore_blocks_raw_data() -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for path in (c.RAW_DIR, c.INTERIM_DIR):
        assert f"{path.as_posix()}/" in lines


def test_repo_hygiene_raw_data_not_tracked() -> None:
    if not (ROOT / ".git").exists() or shutil.which("git") is None:
        pytest.skip("git 저장소가 아니다")
    tracked = subprocess.run(
        ["git", "ls-files", "--", c.RAW_DIR.as_posix(), c.INTERIM_DIR.as_posix()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert tracked == []


def _player_ids(path: Path) -> set[int]:
    if path.suffix == ".json":
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "people" in doc:
            return {p["id"] for p in doc["people"]}
        return {s["player"]["id"] for s in doc["stats"][0]["splits"]}
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    key = "id" if path.stem.endswith("_frv") else "player_id"
    return {int(r[key]) for r in rows}


def _fixture_files() -> list[Path]:
    return sorted(p for p in FIXTURE_RAW.rglob("*") if p.suffix in {".json", ".csv"})


def test_repo_hygiene_fixture_files_exist() -> None:
    names = {p.relative_to(FIXTURE_RAW).as_posix() for p in _fixture_files()}
    for season in c.SEASONS:
        for role in c.ROLES:
            assert c.raw_statsapi_path(season, role).relative_to(c.RAW_DIR).as_posix() in names
        for kind in ("batter", "pitcher"):
            assert f"savant/{season}_{kind}.csv" in names
    for season in c.FIELDING_YEARS:
        assert f"savant/{season}_oaa.csv" in names
        assert f"savant/{season}_frv.csv" in names


@pytest.mark.parametrize("path", _fixture_files(), ids=lambda p: p.name)
def test_repo_hygiene_fixture_player_limit(path: Path) -> None:
    """실제 데이터 픽스처는 파일당 12명 이하 (PLAN P4)."""
    assert len(_player_ids(path)) <= MAX_PLAYERS_PER_FILE


def test_repo_hygiene_fixture_players_documented() -> None:
    source = (ROOT / "tests" / "fixtures" / "SOURCE.md").read_text(encoding="utf-8")
    all_ids = set().union(*(_player_ids(p) for p in _fixture_files()))
    undocumented = sorted(i for i in all_ids if f"| {i} |" not in source)
    assert undocumented == []
